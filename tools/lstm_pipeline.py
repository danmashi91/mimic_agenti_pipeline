# ─────────────────────────────────────────────────────────
# tools/lstm_pipeline.py
# LSTM-based ICU readmission prediction for Paper 3.
#
# Architecture:
#   - BigQuery extracts hourly-resampled vitals (last 24h)
#   - Sequence builder creates 24-timestep × 5-vital tensors
#   - Bidirectional LSTM + attention → binary readmission label
#   - Hybrid model: BiLSTM + tabular features
#   - Compared against XGBoost tabular baseline
# ─────────────────────────────────────────────────────────

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from loguru import logger
from dotenv import load_dotenv

from agents.data_agent import DataAgent

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────
VITALS = {
    220045: "heart_rate",
    220277: "spo2",
    220210: "resp_rate",
    220052: "abp_mean",
    223761: "temp_f",
}
SEQ_LEN    = 24    # 24 hourly timesteps = last 24h before discharge
N_FEATURES = 5     # 5 vitals
SEED       = 42

# ── Step 1: BigQuery — Hourly Resampled Vitals ─────────────────────────────

def fetch_hourly_vitals(data_agent: DataAgent,
                        limit: int = None) -> pd.DataFrame:
    """
    Fetch vitals resampled to hourly intervals for the last 24h
    before each ICU discharge. Returns one row per stay_id per hour.
    """
    limit_clause = f"LIMIT {limit}" if limit else ""
    query = f"""
    WITH icu_stays AS (
        SELECT stay_id, outtime
        FROM `{data_agent.source_project}.{data_agent.icu_dataset}.icustays`
    ),
    raw_vitals AS (
        SELECT
            c.stay_id,
            c.itemid,
            c.valuenum,
            -- Bucket into 1-hour slots (0=24h before discharge, 23=last hour)
            23 - CAST(
                TIMESTAMP_DIFF(i.outtime, c.charttime, MINUTE) / 60
                AS INT64
            )                                               AS hour_slot
        FROM `{data_agent.source_project}.{data_agent.icu_dataset}.chartevents` AS c
        JOIN icu_stays AS i ON c.stay_id = i.stay_id
        WHERE
            c.itemid IN (220045, 220277, 220210, 220052, 223761)
            AND c.valuenum IS NOT NULL
            AND c.charttime >= TIMESTAMP_SUB(i.outtime, INTERVAL 24 HOUR)
            AND c.charttime <= i.outtime
    )
    SELECT
        stay_id,
        hour_slot,
        AVG(CASE WHEN itemid = 220045 THEN valuenum END) AS heart_rate,
        AVG(CASE WHEN itemid = 220277 THEN valuenum END) AS spo2,
        AVG(CASE WHEN itemid = 220210 THEN valuenum END) AS resp_rate,
        AVG(CASE WHEN itemid = 220052 THEN valuenum END) AS abp_mean,
        AVG(CASE WHEN itemid = 223761 THEN valuenum END) AS temp_f
    FROM raw_vitals
    WHERE hour_slot BETWEEN 0 AND 23
    GROUP BY stay_id, hour_slot
    {limit_clause}
    """
    return data_agent._run_query(query)


# ── Step 2: Sequence Builder ───────────────────────────────────────────────

def build_sequences(
    df_hourly: pd.DataFrame,
    df_labels: pd.DataFrame,
    seq_len: int = SEQ_LEN,
    n_features: int = N_FEATURES,
) -> tuple:
    """
    Convert hourly vitals DataFrame into 3D numpy arrays for LSTM.

    Returns:
        X: (n_patients, seq_len, n_features)
        y: (n_patients,)
        stay_ids: list of stay_ids in order
        scaler_stats: normalization stats per vital
    """
    logger.info("Building LSTM sequences...")

    vital_cols = ["heart_rate", "spo2", "resp_rate", "abp_mean", "temp_f"]

    # Merge readmission labels
    df_labels = df_labels[["stay_id", "readmitted"]].drop_duplicates("stay_id")
    df_hourly = df_hourly.merge(df_labels, on="stay_id", how="inner")

    # Compute global stats for normalization
    scaler_stats = {}
    for col in vital_cols:
        vals = df_hourly[col].dropna()
        scaler_stats[col] = {"mean": vals.mean(), "std": vals.std() + 1e-8}

    stay_ids = df_hourly["stay_id"].unique()
    X_list, y_list, valid_ids = [], [], []

    for stay_id in stay_ids:
        stay_data = df_hourly[df_hourly["stay_id"] == stay_id].sort_values(
            "hour_slot"
        )

        # Build (seq_len, n_features) matrix
        seq = np.full((seq_len, n_features), np.nan)
        for _, row in stay_data.iterrows():
            slot = int(row["hour_slot"])
            if 0 <= slot < seq_len:
                for j, col in enumerate(vital_cols):
                    if not np.isnan(row[col]):
                        seq[slot, j] = row[col]

        # Skip stays with <50% coverage
        coverage = (~np.isnan(seq)).mean()
        if coverage < 0.5:
            continue

        # Normalize each vital
        for j, col in enumerate(vital_cols):
            mu  = scaler_stats[col]["mean"]
            std = scaler_stats[col]["std"]
            seq[:, j] = (seq[:, j] - mu) / std

        # Forward fill then backward fill NaNs
        seq = pd.DataFrame(seq).ffill().bfill().fillna(0).values

        label = int(stay_data["readmitted"].iloc[0])
        X_list.append(seq)
        y_list.append(label)
        valid_ids.append(stay_id)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    logger.info(
        f"Sequences built: {X.shape} | "
        f"Readmission rate: {y.mean():.2%} | "
        f"Coverage filter kept: {len(valid_ids):,} stays"
    )
    return X, y, valid_ids, scaler_stats


# ── Step 3: PyTorch Datasets ───────────────────────────────────────────────

class ICUSequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class HybridICUDataset(Dataset):
    """Dataset combining time-series sequences + tabular features."""
    def __init__(self, X_seq: np.ndarray, X_tab: np.ndarray,
                 y: np.ndarray):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.X_tab = torch.tensor(X_tab, dtype=torch.float32)
        self.y     = torch.tensor(y,     dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.X_tab[idx], self.y[idx]


# ── Step 4: Models ────────────────────────────────────────────────────────

class AttentionLayer(nn.Module):
    """Scaled dot-product attention over LSTM hidden states."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_out):
        weights = torch.softmax(self.attn(lstm_out), dim=1)
        context = (weights * lstm_out).sum(dim=1)
        return context, weights


class BiLSTMReadmission(nn.Module):
    """
    Bidirectional LSTM + attention for ICU readmission prediction.

    Architecture:
        Input (batch, 24, 5)
        → BiLSTM (hidden=64, layers=2, dropout=0.3)
        → Attention pooling
        → FC(128 → 64 → 1) → Sigmoid
    """
    def __init__(self, n_features: int = 5, hidden_dim: int = 64,
                 n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.attention = AttentionLayer(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, _  = self.attention(lstm_out)
        return self.classifier(context).squeeze(-1)


class HybridReadmissionModel(nn.Module):
    """
    Hybrid BiLSTM + Tabular model for ICU readmission prediction.

    Architecture:
        Sequence branch:
            Input (batch, 24, 5)
            → BiLSTM (hidden=64, 2 layers) → Attention → FC → 64-dim

        Tabular branch:
            Input (batch, n_tabular)
            → FC(n_tabular → 64) → ReLU → Dropout → 64-dim

        Fusion:
            Concat(64 + 64 = 128) → FC(128 → 64 → 1) → Sigmoid
    """
    def __init__(self, n_seq_features: int = 5,
                 n_tab_features: int = 34,
                 hidden_dim: int = 64,
                 n_lstm_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()

        # ── Sequence branch ───────────────────────────────
        self.lstm = nn.LSTM(
            input_size=n_seq_features,
            hidden_size=hidden_dim,
            num_layers=n_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_lstm_layers > 1 else 0,
        )
        self.attention = AttentionLayer(hidden_dim)
        self.seq_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Tabular branch ────────────────────────────────
        self.tab_fc = nn.Sequential(
            nn.Linear(n_tab_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Fusion classifier ─────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x_seq, x_tab):
        lstm_out, _ = self.lstm(x_seq)
        context, _  = self.attention(lstm_out)
        seq_emb     = self.seq_fc(context)           # (batch, 64)
        tab_emb     = self.tab_fc(x_tab)             # (batch, 64)
        fused       = torch.cat([seq_emb, tab_emb], dim=1)  # (batch, 128)
        return self.classifier(fused).squeeze(-1)


# ── Step 5: Training Functions ────────────────────────────────────────────

def _get_device():
    return "mps" if torch.backends.mps.is_available() else \
           "cuda" if torch.cuda.is_available() else "cpu"


def train_lstm(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray,   y_val: np.ndarray,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    models_dir: str = "models",
    device: str = None,
) -> dict:
    """Train the BiLSTM model and return best validation metrics."""
    if device is None:
        device = _get_device()
    logger.info(f"Training on device: {device}")

    pos_weight = torch.tensor(
        [(y_train == 0).sum() / max((y_train == 1).sum(), 1)],
        dtype=torch.float32
    ).to(device)

    train_dl = DataLoader(ICUSequenceDataset(X_train, y_train),
                          batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(ICUSequenceDataset(X_val, y_val),
                          batch_size=batch_size)

    model     = BiLSTMReadmission().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                   weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)
    criterion = nn.BCELoss()

    best_auroc, best_state, history = 0.0, None, []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_dl:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            preds   = model(X_b)
            weights = torch.where(y_b == 1, pos_weight.squeeze(),
                                  torch.ones(1).to(device))
            loss = (criterion(preds, y_b) * weights).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for X_b, y_b in val_dl:
                all_probs.extend(model(X_b.to(device)).cpu().numpy())
                all_labels.extend(y_b.numpy())

        auroc = roc_auc_score(all_labels, all_probs)
        auprc = average_precision_score(all_labels, all_probs)
        history.append({"epoch": epoch,
                         "train_loss": round(train_loss/len(train_dl), 4),
                         "val_auroc": round(auroc, 4),
                         "val_auprc": round(auprc, 4)})

        if auroc > best_auroc:
            best_auroc = auroc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        if epoch % 5 == 0:
            logger.info(f"  Epoch {epoch:>3}/{epochs} | "
                        f"Loss={train_loss/len(train_dl):.4f} | "
                        f"AUROC={auroc:.4f} | AUPRC={auprc:.4f}")

    os.makedirs(models_dir, exist_ok=True)
    torch.save(best_state, f"{models_dir}/lstm_readmission.pt")
    joblib.dump(history, f"{models_dir}/lstm_training_history.pkl")
    logger.info(f"Best LSTM saved → {models_dir}/lstm_readmission.pt")
    return max(history, key=lambda x: x["val_auroc"])


def train_hybrid(
    X_seq_train, X_tab_train, y_train,
    X_seq_val,   X_tab_val,   y_val,
    n_tab_features: int,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    models_dir: str = "models",
    device: str = None,
) -> dict:
    """Train the hybrid BiLSTM + tabular model."""
    if device is None:
        device = _get_device()
    logger.info(f"Hybrid training on device: {device}")

    pos_weight = torch.tensor(
        [(y_train == 0).sum() / max((y_train == 1).sum(), 1)],
        dtype=torch.float32
    ).to(device)

    train_dl = DataLoader(
        HybridICUDataset(X_seq_train, X_tab_train, y_train),
        batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(
        HybridICUDataset(X_seq_val, X_tab_val, y_val),
        batch_size=batch_size)

    model     = HybridReadmissionModel(
        n_tab_features=n_tab_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                   weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)
    criterion = nn.BCELoss()

    best_auroc, best_state, history = 0.0, None, []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X_seq_b, X_tab_b, y_b in train_dl:
            X_seq_b = X_seq_b.to(device)
            X_tab_b = X_tab_b.to(device)
            y_b     = y_b.to(device)
            optimizer.zero_grad()
            preds   = model(X_seq_b, X_tab_b)
            weights = torch.where(y_b == 1, pos_weight.squeeze(),
                                  torch.ones(1).to(device))
            loss = (criterion(preds, y_b) * weights).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for X_seq_b, X_tab_b, y_b in val_dl:
                probs = model(X_seq_b.to(device),
                              X_tab_b.to(device)).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(y_b.numpy())

        auroc = roc_auc_score(all_labels, all_probs)
        auprc = average_precision_score(all_labels, all_probs)
        history.append({"epoch": epoch,
                         "train_loss": round(train_loss/len(train_dl), 4),
                         "val_auroc": round(auroc, 4),
                         "val_auprc": round(auprc, 4)})

        if auroc > best_auroc:
            best_auroc = auroc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        if epoch % 5 == 0:
            logger.info(f"  Epoch {epoch:>3}/{epochs} | "
                        f"Loss={train_loss/len(train_dl):.4f} | "
                        f"AUROC={auroc:.4f} | AUPRC={auprc:.4f}")

    os.makedirs(models_dir, exist_ok=True)
    torch.save(best_state, f"{models_dir}/hybrid_readmission.pt")
    joblib.dump(history, f"{models_dir}/hybrid_training_history.pkl")
    logger.info(
        f"Best hybrid model saved → {models_dir}/hybrid_readmission.pt")
    return max(history, key=lambda x: x["val_auroc"])


# ── Step 6: Tabular Feature Builder for Hybrid ────────────────────────────

def build_tabular_features(data_agent: DataAgent,
                            stay_ids: list,
                            limit: int = None) -> tuple:
    """
    Fetch and engineer tabular features for a specific set of stay_ids.
    Returns aligned X_tab matrix and feature column names.
    """
    logger.info("Building tabular features for hybrid model...")

    df_labels  = data_agent.get_icu_readmission_cohort(limit=limit)
    df_vitals  = data_agent.get_icu_discharge_vitals(limit=limit)
    df_fluid   = data_agent.get_fluid_balance(limit=limit)
    df_trends  = data_agent.get_icu_vital_trends(limit=limit)

    # Merge all features
    df = df_labels.copy()
    for merge_df, key in [(df_vitals, "stay_id"), (df_fluid, "stay_id"),
                          (df_trends, "stay_id")]:
        if merge_df is not None:
            df = df.merge(merge_df, on=key, how="left")

    # Time features
    df["outtime"]       = pd.to_datetime(df["outtime"])
    df["discharge_hour"] = df["outtime"].dt.hour
    df["discharge_dow"]  = df["outtime"].dt.dayofweek

    # Abnormality flags
    if "heart_rate_last" in df.columns:
        df["hr_abnormal"] = (
            (df["heart_rate_last"] < 50) | (df["heart_rate_last"] > 120)
        ).astype(int)
    if "spo2_last" in df.columns:
        df["spo2_low"] = (df["spo2_last"] < 92).astype(int)
    if "resp_rate_last" in df.columns:
        df["rr_abnormal"] = (
            (df["resp_rate_last"] < 8) | (df["resp_rate_last"] > 30)
        ).astype(int)

    # Encode categoricals
    for col in ["first_careunit", "insurance", "race", "gender"]:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("UNKNOWN")
            df[col] = LabelEncoder().fit_transform(df[col])

    # Fill numerics
    num_cols = [
        "age", "icu_los_hours", "heart_rate_last", "spo2_last",
        "resp_rate_last", "abp_mean_last", "temp_f_last",
        "total_output_ml", "output_event_count",
        "hr_mean_24h", "hr_std_24h", "hr_delta_24h",
        "spo2_mean_24h", "spo2_std_24h", "spo2_delta_24h",
        "rr_mean_24h", "rr_std_24h", "rr_delta_24h",
        "abp_mean_24h", "abp_std_24h", "abp_delta_24h",
        "temp_mean_24h", "temp_std_24h", "temp_delta_24h",
        "discharge_hour", "discharge_dow",
        "hr_abnormal", "spo2_low", "rr_abnormal",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    # Drop non-feature columns
    drop_cols = ["subject_id", "hadm_id", "intime", "outtime",
                 "next_icu_intime", "readmitted", "hospital_expire_flag"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    feature_cols = [c for c in df.columns if c != "stay_id"]

    # Align to provided stay_ids order
    df = df.set_index("stay_id")
    stay_ids_set = set(df.index)
    valid_mask   = [sid in stay_ids_set for sid in stay_ids]
    valid_ids    = [sid for sid, keep in zip(stay_ids, valid_mask) if keep]

    X_tab = np.array([
        df.loc[sid, feature_cols].values.astype(float)
        if sid in df.index else np.zeros(len(feature_cols))
        for sid in valid_ids
    ], dtype=np.float32)
    X_tab = np.nan_to_num(X_tab, nan=0.0)

    # Scale
    scaler = StandardScaler()
    X_tab  = scaler.fit_transform(X_tab)

    logger.info(f"Tabular features: {X_tab.shape} | "
                f"Feature cols: {len(feature_cols)}")
    return X_tab, valid_ids, feature_cols, scaler


# ── Step 7: Pipeline Functions ─────────────────────────────────────────────

def run_lstm_pipeline(limit_vitals: int = None,
                      limit_labels: int = None,
                      epochs: int = 30) -> dict:
    """End-to-end BiLSTM pipeline for Paper 3."""
    logger.info("=== LSTM Pipeline: ICU Readmission (Paper 3) ===")

    data_agent = DataAgent()
    # NOTE: vitals are never limited — needs full hourly coverage per stay
    logger.info("Fetching hourly vitals (full dataset required)...")
    df_hourly  = fetch_hourly_vitals(data_agent, limit=None)
    logger.info("Fetching readmission labels...")
    df_labels  = data_agent.get_icu_readmission_cohort(limit=limit_labels)
    X, y, stay_ids, scaler_stats = build_sequences(df_hourly, df_labels)

    if len(X) < 500:
        logger.warning(f"Only {len(X)} valid sequences — try larger limit.")
        return {}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,} | "
                f"Positive rate: {y_train.mean():.2%}")

    train_lstm(X_train, y_train, X_test, y_test, epochs=epochs)

    device = _get_device()
    model  = BiLSTMReadmission().to(device)
    model.load_state_dict(
        torch.load("models/lstm_readmission.pt", map_location=device))
    model.eval()

    test_dl = DataLoader(ICUSequenceDataset(X_test, y_test), batch_size=256)
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X_b, y_b in test_dl:
            all_probs.extend(model(X_b.to(device)).cpu().numpy())
            all_labels.extend(y_b.numpy())

    final_auroc = roc_auc_score(all_labels, all_probs)
    final_auprc = average_precision_score(all_labels, all_probs)

    print("\n── LSTM vs XGBoost Comparison ───────────────────────")
    print(f"  {'Model':<30} {'AUROC':>8} {'AUPRC':>8}")
    print(f"  {'-'*46}")
    print(f"  {'XGBoost (tabular baseline)':<30} {'0.6095':>8} {'0.1302':>8}")
    print(f"  {'BiLSTM + Attention':<30} {final_auroc:>8.4f} {final_auprc:>8.4f}")
    print(f"\n  AUROC vs XGBoost: {(final_auroc-0.6095)/0.6095*100:+.1f}%")
    print(f"  AUPRC vs XGBoost: {(final_auprc-0.1302)/0.1302*100:+.1f}%")

    logger.info("LSTM pipeline complete.")
    return {"LSTM (BiLSTM+Attention)": {
        "AUROC": round(final_auroc, 4), "AUPRC": round(final_auprc, 4)}}


def run_hybrid_pipeline(limit: int = None, epochs: int = 30) -> dict:
    """End-to-end hybrid BiLSTM + tabular pipeline for Paper 3."""
    logger.info("=== Hybrid Pipeline: BiLSTM + Tabular (Paper 3) ===")

    data_agent = DataAgent()

    # ── Fetch sequences ───────────────────────────────────
    # NOTE: vitals are never limited — 50K rows covers only ~2K stays
    # which fails the 50% coverage filter. Labels can be limited for dev.
    logger.info("Fetching hourly vitals (full dataset required)...")
    df_hourly = fetch_hourly_vitals(data_agent, limit=None)
    logger.info("Fetching readmission labels...")
    df_labels = data_agent.get_icu_readmission_cohort(limit=limit)

    X_seq, y_seq, stay_ids, scaler_stats = build_sequences(
        df_hourly, df_labels)

    if len(X_seq) < 500:
        logger.warning(f"Only {len(X_seq)} valid sequences — try larger limit.")
        return {}

    # ── Fetch tabular features aligned to same stays ──────
    X_tab, valid_ids, feature_cols, tab_scaler = build_tabular_features(
        data_agent, stay_ids, limit=limit)

    # Align sequences to valid_ids (those present in both)
    valid_set  = set(valid_ids)
    mask       = [sid in valid_set for sid in stay_ids]
    X_seq_aln  = X_seq[mask]
    y_aln      = y_seq[mask]

    logger.info(f"Aligned: {len(valid_ids):,} stays | "
                f"Seq: {X_seq_aln.shape} | Tab: {X_tab.shape}")

    # ── Train/test split ──────────────────────────────────
    (X_seq_tr, X_seq_te,
     X_tab_tr, X_tab_te,
     y_tr,     y_te) = train_test_split(
        X_seq_aln, X_tab, y_aln,
        test_size=0.2, random_state=SEED, stratify=y_aln)

    logger.info(f"Train: {len(y_tr):,} | Test: {len(y_te):,} | "
                f"Positive rate: {y_tr.mean():.2%}")

    # ── Train ─────────────────────────────────────────────
    train_hybrid(X_seq_tr, X_tab_tr, y_tr,
                 X_seq_te, X_tab_te, y_te,
                 n_tab_features=X_tab.shape[1],
                 epochs=epochs)

    # ── Final evaluation ──────────────────────────────────
    device = _get_device()
    model  = HybridReadmissionModel(
        n_tab_features=X_tab.shape[1]).to(device)
    model.load_state_dict(
        torch.load("models/hybrid_readmission.pt", map_location=device))
    model.eval()

    test_dl = DataLoader(
        HybridICUDataset(X_seq_te, X_tab_te, y_te), batch_size=256)
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X_seq_b, X_tab_b, y_b in test_dl:
            probs = model(X_seq_b.to(device),
                          X_tab_b.to(device)).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y_b.numpy())

    final_auroc = roc_auc_score(all_labels, all_probs)
    final_auprc = average_precision_score(all_labels, all_probs)

    print("\n── Final Comparison: All Models ─────────────────────")
    print(f"  {'Model':<35} {'AUROC':>8} {'AUPRC':>8}")
    print(f"  {'-'*51}")
    print(f"  {'XGBoost (tabular only)':<35} {'0.6095':>8} {'0.1302':>8}")
    print(f"  {'BiLSTM + Attention (seq only)':<35} {'0.5848':>8} {'0.0820':>8}")
    print(f"  {'Hybrid (BiLSTM + Tabular)':<35} "
          f"{final_auroc:>8.4f} {final_auprc:>8.4f}")
    print(f"\n  Hybrid vs XGBoost — "
          f"AUROC: {(final_auroc-0.6095)/0.6095*100:+.1f}%  "
          f"AUPRC: {(final_auprc-0.1302)/0.1302*100:+.1f}%")

    joblib.dump(tab_scaler, "models/hybrid_tab_scaler.pkl")
    logger.info("Hybrid pipeline complete.")

    return {"Hybrid (BiLSTM + Tabular)": {
        "AUROC": round(final_auroc, 4),
        "AUPRC": round(final_auprc, 4)}}


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="LSTM / Hybrid ICU Readmission Pipeline"
    )
    parser.add_argument("--sample", type=int, default=None,
                        help="Row limit for dev mode (e.g. --sample 50000)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Training epochs (default: 30)")
    parser.add_argument("--mode", choices=["lstm", "hybrid", "both"],
                        default="hybrid",
                        help="Which model to run (default: hybrid)")
    args = parser.parse_args()

    if args.mode in ("lstm", "both"):
        run_lstm_pipeline(
            limit_vitals=args.sample,
            limit_labels=args.sample,
            epochs=args.epochs,
        )

    if args.mode in ("hybrid", "both"):
        run_hybrid_pipeline(
            limit=args.sample,
            epochs=args.epochs,
        )

    print("\n✅ Pipeline complete.")
    print("   Models saved to: models/")