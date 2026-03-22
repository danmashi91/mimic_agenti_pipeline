# ─────────────────────────────────────────────────────────
# agents/prediction_agent.py  (enriched version)
# ─────────────────────────────────────────────────────────

import os
import numpy as np
import pandas as pd
import yaml
import joblib

from loguru import logger
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, average_precision_score
)
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

load_dotenv()


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class PredictionAgent:

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = load_config(config_path)
        self.seed = self.config["pipeline"]["random_seed"]
        self.test_size = self.config["pipeline"]["test_size"]
        self.log_transform = self.config["cost_prediction"]["log_transform"]
        self.cv_folds = self.config["pipeline"].get("cv_folds", 5)
        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)
        logger.info("PredictionAgent initialized.")

    def engineer_cost_features(self, df_admissions, df_diagnoses=None,
                                df_labs=None, df_cci=None, df_procedures=None):
        logger.info("Engineering enriched cost prediction features...")
        df = df_admissions.copy()

        for merge_df, key in [(df_diagnoses, "hadm_id"), (df_labs, "hadm_id"),
                               (df_cci, "hadm_id"), (df_procedures, "hadm_id")]:
            if merge_df is not None:
                df = df.merge(merge_df, on=key, how="left")

        df["admittime"] = pd.to_datetime(df["admittime"])
        df["admit_hour"]  = df["admittime"].dt.hour
        df["admit_dow"]   = df["admittime"].dt.dayofweek
        df["admit_month"] = df["admittime"].dt.month
        df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(60)
        df["age_group"] = pd.cut(df["age"], bins=[0,44,64,79,120],
                                  labels=[0,1,2,3]).astype(float)

        cat_cols = ["admission_type", "admission_location", "discharge_location",
                    "insurance", "language", "marital_status", "race", "gender",
                    "drg_type"]
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna("UNKNOWN")
                df[col] = LabelEncoder().fit_transform(df[col])

        num_cols = ["drg_severity", "drg_mortality", "creatinine", "hemoglobin",
                    "wbc", "sodium", "potassium", "lactate", "cci_total_score",
                    "cci_mi", "cci_chf", "cci_pvd", "cci_copd", "cci_dm",
                    "cci_renal", "cci_cancer", "procedure_count", "unique_procedures"]
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        drop_cols = ["subject_id", "hadm_id", "admittime", "dischtime",
                     "drg_code", "drg_description", "top5_icd_codes"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        df = df.dropna(subset=["los_hours"])
        df = df[df["los_hours"] > 0]

        logger.info(f"Cost feature matrix: {df.shape[0]:,} rows x {df.shape[1]} cols")
        return df

    def train_cost_models(self, df: pd.DataFrame) -> dict:
        logger.info("Training enriched cost models...")
        X = df.drop(columns=["los_hours"]).fillna(0)
        y = np.log1p(df["los_hours"]) if self.log_transform else df["los_hours"]
        if self.log_transform:
            logger.info("Applied log1p transform to target.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.seed)
        scaler = StandardScaler()
        Xtr_s, Xte_s = scaler.fit_transform(X_train), scaler.transform(X_test)

        models = {
            "Ridge Regression": (Ridge(alpha=1.0), Xtr_s, Xte_s),
            "Random Forest":    (RandomForestRegressor(n_estimators=200,
                                  random_state=self.seed, n_jobs=-1), X_train, X_test),
            "XGBoost":          (XGBRegressor(n_estimators=300, learning_rate=0.05,
                                  max_depth=6, subsample=0.8, colsample_bytree=0.8,
                                  random_state=self.seed, verbosity=0), X_train, X_test),
            "LightGBM":         (LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                  num_leaves=63, random_state=self.seed,
                                  verbosity=-1), X_train, X_test),
        }

        results = {}
        for name, (model, Xtr, Xte) in models.items():
            logger.info(f"  Training {name}...")
            model.fit(Xtr, y_train)
            preds = model.predict(Xte)
            if self.log_transform:
                preds, y_orig = np.expm1(preds), np.expm1(y_test)
            else:
                y_orig = y_test
            rmse = np.sqrt(mean_squared_error(y_orig, preds))
            mae  = mean_absolute_error(y_orig, preds)
            r2   = r2_score(y_orig, preds)
            results[name] = {"RMSE": round(rmse,3), "MAE": round(mae,3),
                             "R2": round(r2,4)}
            logger.info(f"    {name} → RMSE={rmse:.3f}  MAE={mae:.3f}  R²={r2:.4f}")
            if name == "XGBoost":
                joblib.dump(model,  f"{self.models_dir}/cost_xgb.pkl")
                joblib.dump(scaler, f"{self.models_dir}/cost_scaler.pkl")
                joblib.dump(list(X.columns), f"{self.models_dir}/cost_features.pkl")

        logger.info("Cost training complete.")
        return results

    def engineer_readmission_features(self, df, df_vitals=None, df_fluid=None, df_trends=None):
        logger.info("Engineering enriched readmission features...")
        df = df.copy()

        if df_vitals is not None:
            df = df.merge(df_vitals, on="stay_id", how="left")
        if df_fluid is not None:
            df = df.merge(df_fluid, on="stay_id", how="left")
        if df_trends is not None:
            df = df.merge(df_trends, on="stay_id", how="left")

        df["outtime"] = pd.to_datetime(df["outtime"])
        df["discharge_hour"] = df["outtime"].dt.hour
        df["discharge_dow"]  = df["outtime"].dt.dayofweek

        if "heart_rate_last" in df.columns:
            df["hr_abnormal"] = ((df["heart_rate_last"]<50)|(df["heart_rate_last"]>120)).astype(int)
        if "spo2_last" in df.columns:
            df["spo2_low"] = (df["spo2_last"] < 92).astype(int)
        if "resp_rate_last" in df.columns:
            df["rr_abnormal"] = ((df["resp_rate_last"]<8)|(df["resp_rate_last"]>30)).astype(int)

        cat_cols = ["first_careunit", "insurance", "race", "gender"]
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna("UNKNOWN")
                df[col] = LabelEncoder().fit_transform(df[col])

        num_cols = ["age", "icu_los_hours", "heart_rate_last", "spo2_last",
                    "resp_rate_last", "abp_mean_last", "temp_f_last",
                    "total_output_ml", "output_event_count"]
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].fillna(df[col].median())

        drop_cols = ["subject_id", "hadm_id", "stay_id",
                     "intime", "outtime", "next_icu_intime"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        df = df.dropna(subset=["readmitted"])

        logger.info(f"Readmission matrix: {df.shape[0]:,} rows x {df.shape[1]} cols")
        logger.info(f"Readmission rate: {df['readmitted'].mean():.2%}")
        return df

    def train_readmission_models(self, df: pd.DataFrame) -> dict:
        logger.info("Training enriched readmission models...")
        X = df.drop(columns=["readmitted"]).fillna(0)
        y = df["readmitted"].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.seed, stratify=y)
        scaler = StandardScaler()
        Xtr_s, Xte_s = scaler.fit_transform(X_train), scaler.transform(X_test)

        pos, neg = y_train.sum(), len(y_train) - y_train.sum()
        scale_pos = neg / pos if pos > 0 else 1
        logger.info(f"  Class distribution — 0: {neg:,} | 1: {pos:,}")

        models = {
            "Logistic Regression": (LogisticRegression(max_iter=1000,
                                     random_state=self.seed,
                                     class_weight="balanced"), Xtr_s, Xte_s),
            "Random Forest":       (RandomForestClassifier(n_estimators=200,
                                     random_state=self.seed,
                                     class_weight="balanced", n_jobs=-1),
                                     X_train, X_test),
            "XGBoost":             (XGBClassifier(n_estimators=300,
                                     learning_rate=0.05, max_depth=6,
                                     subsample=0.8, colsample_bytree=0.8,
                                     scale_pos_weight=scale_pos,
                                     random_state=self.seed, verbosity=0,
                                     eval_metric="logloss"), X_train, X_test),
            "LightGBM":            (LGBMClassifier(n_estimators=300,
                                     learning_rate=0.05, num_leaves=63,
                                     random_state=self.seed,
                                     class_weight="balanced",
                                     verbosity=-1), X_train, X_test),
        }

        results = {}
        for name, (model, Xtr, Xte) in models.items():
            logger.info(f"  Training {name}...")
            model.fit(Xtr, y_train)
            probs = model.predict_proba(Xte)[:, 1]
            auroc = roc_auc_score(y_test, probs)
            auprc = average_precision_score(y_test, probs)
            results[name] = {"AUROC": round(auroc,4), "AUPRC": round(auprc,4)}
            logger.info(f"    {name} → AUROC={auroc:.4f}  AUPRC={auprc:.4f}")
            if name == "XGBoost":
                joblib.dump(model,  f"{self.models_dir}/readmission_xgb.pkl")
                joblib.dump(scaler, f"{self.models_dir}/readmission_scaler.pkl")
                joblib.dump(list(X.columns),
                            f"{self.models_dir}/readmission_features.pkl")

        logger.info("Readmission training complete.")
        return results

    def cross_validate_cost_models(
        self, df: pd.DataFrame, cv_folds: int = None
    ) -> pd.DataFrame:
        """
        Run k-fold cross-validation on all 4 cost prediction models.
        Reports mean ± SD and 95% CI for RMSE, MAE, and R² across folds.
        All metrics reported in native (back-transformed) hour units.

        Args:
            df:       feature matrix with 'los_hours' as target column
            cv_folds: number of folds (defaults to config value, usually 5)

        Returns:
            DataFrame with mean, SD, and 95% CI for each metric/model
        """
        folds = cv_folds or self.cv_folds
        logger.info(
            f"Running {folds}-fold CV on cost prediction models "
            f"(n={len(df):,})..."
        )

        target = "los_hours"
        X = df.drop(columns=[target]).fillna(0)
        y = np.log1p(df[target]) if self.log_transform else df[target]

        kf = KFold(n_splits=folds, shuffle=True, random_state=self.seed)

        models_cfg = {
            "Ridge Regression": Ridge(alpha=1.0),
            "Random Forest":    RandomForestRegressor(
                                    n_estimators=200,
                                    random_state=self.seed,
                                    n_jobs=-1),
            "XGBoost":          XGBRegressor(
                                    n_estimators=300,
                                    learning_rate=0.05,
                                    max_depth=6,
                                    subsample=0.8,
                                    colsample_bytree=0.8,
                                    random_state=self.seed,
                                    verbosity=0),
            "LightGBM":         LGBMRegressor(
                                    n_estimators=300,
                                    learning_rate=0.05,
                                    num_leaves=63,
                                    random_state=self.seed,
                                    verbosity=-1),
        }

        results = []

        for name, model in models_cfg.items():
            logger.info(f"  CV: {name}...")
            fold_rmse, fold_mae, fold_r2 = [], [], []

            for fold_idx, (train_idx, val_idx) in enumerate(
                kf.split(X), start=1
            ):
                X_tr  = X.iloc[train_idx]
                X_val = X.iloc[val_idx]
                y_tr  = y.iloc[train_idx]
                y_val = y.iloc[val_idx]

                # Ridge requires scaling; tree models do not
                if name == "Ridge Regression":
                    scaler = StandardScaler()
                    X_tr_fit  = scaler.fit_transform(X_tr)
                    X_val_fit = scaler.transform(X_val)
                else:
                    X_tr_fit  = X_tr.values
                    X_val_fit = X_val.values

                model.fit(X_tr_fit, y_tr)
                preds = model.predict(X_val_fit)

                # Back-transform from log scale
                if self.log_transform:
                    preds_bt = np.expm1(preds)
                    y_val_bt = np.expm1(y_val.values)
                else:
                    preds_bt = preds
                    y_val_bt = y_val.values

                fold_rmse.append(
                    np.sqrt(mean_squared_error(y_val_bt, preds_bt)))
                fold_mae.append(mean_absolute_error(y_val_bt, preds_bt))
                fold_r2.append(r2_score(y_val_bt, preds_bt))

                logger.info(
                    f"    Fold {fold_idx}/{folds}: "
                    f"RMSE={fold_rmse[-1]:.2f}  "
                    f"MAE={fold_mae[-1]:.2f}  "
                    f"R²={fold_r2[-1]:.4f}"
                )

            # 95% CI using t-distribution (df = folds - 1)
            def ci95(values):
                arr  = np.array(values)
                mean = arr.mean()
                se   = arr.std(ddof=1) / np.sqrt(len(arr))
                t    = stats.t.ppf(0.975, df=len(arr) - 1)
                return round(mean - t * se, 4), round(mean + t * se, 4)

            rmse_ci = ci95(fold_rmse)
            mae_ci  = ci95(fold_mae)
            r2_ci   = ci95(fold_r2)

            results.append({
                "Model":        name,
                "R2_mean":      round(np.mean(fold_r2),   4),
                "R2_sd":        round(np.std(fold_r2,   ddof=1), 4),
                "R2_ci_low":    r2_ci[0],
                "R2_ci_high":   r2_ci[1],
                "RMSE_mean":    round(np.mean(fold_rmse), 3),
                "RMSE_sd":      round(np.std(fold_rmse, ddof=1), 3),
                "RMSE_ci_low":  rmse_ci[0],
                "RMSE_ci_high": rmse_ci[1],
                "MAE_mean":     round(np.mean(fold_mae),  3),
                "MAE_sd":       round(np.std(fold_mae,  ddof=1), 3),
                "MAE_ci_low":   mae_ci[0],
                "MAE_ci_high":  mae_ci[1],
                "fold_rmse":    [round(v, 3) for v in fold_rmse],
                "fold_mae":     [round(v, 3) for v in fold_mae],
                "fold_r2":      [round(v, 4) for v in fold_r2],
            })

            logger.info(
                f"    {name} CV → "
                f"R²={np.mean(fold_r2):.4f} ± {np.std(fold_r2, ddof=1):.4f}  "
                f"RMSE={np.mean(fold_rmse):.2f} ± "
                f"{np.std(fold_rmse, ddof=1):.2f}"
            )

        cv_df = pd.DataFrame(results)
        logger.info(f"{folds}-fold CV complete.")
        return cv_df

    def print_cv_results(self, cv_df: pd.DataFrame) -> None:
        """Print CV results in manuscript-ready format."""
        folds = len(cv_df["fold_r2"].iloc[0]) if "fold_r2" in cv_df.columns else "k"
        print(f"\n── {folds}-Fold Cross-Validation Results "
              f"(mean ± SD) ──────────────────────────")
        print(f"  {'Model':<25} {'R² mean±SD':<26} "
              f"{'RMSE mean±SD (hrs)':<26} {'MAE mean±SD (hrs)'}")
        print("  " + "─" * 90)
        for _, row in cv_df.iterrows():
            print(
                f"  {row['Model']:<25} "
                f"{row['R2_mean']:.4f} ± {row['R2_sd']:.4f}   "
                f"({row['R2_ci_low']:.4f}–{row['R2_ci_high']:.4f})   "
                f"{row['RMSE_mean']:.2f} ± {row['RMSE_sd']:.2f}   "
                f"{row['MAE_mean']:.2f} ± {row['MAE_sd']:.2f}"
            )
        print()

    def print_results(self, results: dict, task: str) -> None:
        print(f"\n── {task} Results ──────────────────────────────")
        header = f"  {'Model':<25}" + "".join(
            f"  {k:>8}" for k in list(results.values())[0].keys())
        print(header)
        print("  " + "-" * (len(header) - 2))
        for model, metrics in results.items():
            print(f"  {model:<25}" + "".join(f"  {v:>8}" for v in metrics.values()))
        print()


# ── Quick test ────────────────────────────────────────────
if __name__ == "__main__":
    from agents.data_agent import DataAgent

    data_agent = DataAgent()
    pred_agent = PredictionAgent()

    logger.info("=== PAPER 1: Enriched Cost Prediction ===")
    print("\nFetching enriched data for Paper 1 (10,000 rows)...")
    df_adm   = data_agent.get_admissions_cohort(limit=None)
    df_diag  = data_agent.get_top_diagnoses(limit=None)
    df_labs  = data_agent.get_admission_labs(limit=None)
    df_cci   = data_agent.get_charlson_index(limit=None)
    df_procs = data_agent.get_procedure_counts(limit=None)

    df_cost = pred_agent.engineer_cost_features(
        df_adm, df_diag, df_labs, df_cci, df_procs)
    cost_results = pred_agent.train_cost_models(df_cost)
    pred_agent.print_results(cost_results, "Paper 1: Cost Prediction (LOS proxy)")

    logger.info("=== PAPER 3: Enriched ICU Readmission ===")
    print("\nFetching enriched data for Paper 3 (10,000 rows)...")
    df_icu    = data_agent.get_icu_readmission_cohort(limit=None)
    df_vitals = data_agent.get_icu_discharge_vitals(limit=None)
    df_fluid  = data_agent.get_fluid_balance(limit=None)
    df_trends = data_agent.get_icu_vital_trends(limit=None)

    df_read = pred_agent.engineer_readmission_features(
        df_icu, df_vitals, df_fluid, df_trends)
    read_results = pred_agent.train_readmission_models(df_read)
    pred_agent.print_results(read_results, "Paper 3: ICU Readmission Prediction")