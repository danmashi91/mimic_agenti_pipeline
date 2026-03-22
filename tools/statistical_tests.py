# ─────────────────────────────────────────────────────────
# tools/statistical_tests.py
# Statistical significance testing for Paper 1 manuscript.
#
# Computes:
# 1. Bootstrap 95% CIs for RMSE, MAE, R² (all 4 models)
# 2. Diebold-Mariano pairwise model comparison tests
# 3. Bootstrap CIs on SHAP feature importance rankings
# 4. Formatted output tables ready for manuscript
#
# Runs on saved models in models/ — no BigQuery re-query needed
# ─────────────────────────────────────────────────────────

import os
import json
import joblib
import numpy as np
import pandas as pd
from loguru import logger
from dotenv import load_dotenv
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────
N_BOOTSTRAP   = 1000   # bootstrap iterations
RANDOM_STATE  = 42
CI_LEVEL      = 0.95   # 95% confidence intervals
ALPHA         = 1 - CI_LEVEL


# ── 1. Bootstrap Confidence Intervals ─────────────────────────────────────

def bootstrap_metrics(y_true: np.ndarray,
                      y_pred: np.ndarray,
                      n_bootstrap: int = N_BOOTSTRAP,
                      log_transform: bool = True) -> dict:
    """
    Compute bootstrap 95% CIs for RMSE, MAE, and R²
    using percentile bootstrap on the test set.

    Args:
        y_true: true values (log-transformed if log_transform=True)
        y_pred: predicted values (log-transformed)
        n_bootstrap: number of bootstrap iterations
        log_transform: if True, back-transform before computing metrics

    Returns:
        dict with point estimates and 95% CIs for each metric
    """
    rng = np.random.RandomState(RANDOM_STATE)
    n   = len(y_true)

    rmse_boot, mae_boot, r2_boot = [], [], []

    for _ in range(n_bootstrap):
        idx  = rng.choice(n, size=n, replace=True)
        yt   = y_true[idx]
        yp   = y_pred[idx]

        if log_transform:
            yt = np.expm1(yt)
            yp = np.expm1(yp)

        rmse_boot.append(np.sqrt(mean_squared_error(yt, yp)))
        mae_boot.append(mean_absolute_error(yt, yp))
        r2_boot.append(r2_score(yt, yp))

    # Point estimates (on full test set)
    if log_transform:
        yt_full = np.expm1(y_true)
        yp_full = np.expm1(y_pred)
    else:
        yt_full = y_true
        yp_full = y_pred

    rmse_pt = np.sqrt(mean_squared_error(yt_full, yp_full))
    mae_pt  = mean_absolute_error(yt_full, yp_full)
    r2_pt   = r2_score(yt_full, yp_full)

    lo = (1 - CI_LEVEL) / 2
    hi = 1 - lo

    return {
        "RMSE": {
            "estimate": round(rmse_pt, 3),
            "ci_lower": round(float(np.quantile(rmse_boot, lo)), 3),
            "ci_upper": round(float(np.quantile(rmse_boot, hi)), 3),
        },
        "MAE": {
            "estimate": round(mae_pt, 3),
            "ci_lower": round(float(np.quantile(mae_boot, lo)), 3),
            "ci_upper": round(float(np.quantile(mae_boot, hi)), 3),
        },
        "R2": {
            "estimate": round(r2_pt, 4),
            "ci_lower": round(float(np.quantile(r2_boot, lo)), 4),
            "ci_upper": round(float(np.quantile(r2_boot, hi)), 4),
        },
    }


# ── 2. Diebold-Mariano Test ────────────────────────────────────────────────

def diebold_mariano_test(errors_a: np.ndarray,
                          errors_b: np.ndarray,
                          h: int = 1) -> dict:
    """
    Diebold-Mariano test for equal predictive accuracy.
    Tests H0: models A and B have equal predictive accuracy.
    Uses squared error as the loss differential.

    Args:
        errors_a: prediction errors from model A (y_true - y_pred)
        errors_b: prediction errors from model B
        h: forecast horizon (1 for one-step-ahead)

    Returns:
        dict with DM statistic, p-value, and interpretation
    """
    # Loss differential: squared errors
    d = errors_a**2 - errors_b**2

    n     = len(d)
    d_bar = d.mean()

    # Newey-West variance estimate (autocorrelation-consistent)
    gamma_0 = np.var(d, ddof=1)
    gamma_h = np.cov(d[h:], d[:-h])[0, 1] if h > 0 else 0
    var_d   = (gamma_0 + 2 * gamma_h) / n

    if var_d <= 0:
        return {"dm_stat": np.nan, "p_value": np.nan,
                "significant": False, "note": "Zero variance"}

    dm_stat = d_bar / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    return {
        "dm_stat":     round(float(dm_stat), 4),
        "p_value":     round(float(p_value), 4),
        "significant": p_value < ALPHA,
        "direction":   "A better" if dm_stat > 0 else "B better",
    }


# ── 3. SHAP Bootstrap CIs ─────────────────────────────────────────────────

def bootstrap_shap_rankings(shap_values: np.ndarray,
                             feature_names: list,
                             n_bootstrap: int = N_BOOTSTRAP,
                             top_n: int = 20) -> pd.DataFrame:
    """
    Bootstrap confidence intervals on mean |SHAP| feature rankings.
    Shows that the top feature rankings are stable across resamples.

    Args:
        shap_values: (n_samples, n_features) array of SHAP values
        feature_names: list of feature names
        n_bootstrap: number of bootstrap iterations
        top_n: number of top features to report

    Returns:
        DataFrame with mean |SHAP|, 95% CI, and rank stability
    """
    logger.info(f"Computing bootstrap CIs on SHAP values "
                f"({n_bootstrap} iterations)...")

    rng = np.random.RandomState(RANDOM_STATE)
    n   = shap_values.shape[0]

    # Bootstrap mean |SHAP| for each feature
    boot_means = np.zeros((n_bootstrap, len(feature_names)))
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_means[i] = np.abs(shap_values[idx]).mean(axis=0)

    # Point estimates
    point_means = np.abs(shap_values).mean(axis=0)

    lo = (1 - CI_LEVEL) / 2
    hi = 1 - lo

    ci_lower = np.quantile(boot_means, lo, axis=0)
    ci_upper = np.quantile(boot_means, hi, axis=0)

    # Rank stability: how often does each feature appear in top N?
    rank_counts = np.zeros(len(feature_names))
    for i in range(n_bootstrap):
        top_idx = np.argsort(boot_means[i])[::-1][:top_n]
        rank_counts[top_idx] += 1
    rank_stability = rank_counts / n_bootstrap

    df = pd.DataFrame({
        "feature":         feature_names,
        "mean_shap":       point_means,
        "ci_lower":        ci_lower,
        "ci_upper":        ci_upper,
        "rank_stability":  rank_stability,
    }).sort_values("mean_shap", ascending=False).head(top_n)

    df["rank"]            = range(1, len(df) + 1)
    df["mean_shap"]       = df["mean_shap"].round(4)
    df["ci_lower"]        = df["ci_lower"].round(4)
    df["ci_upper"]        = df["ci_upper"].round(4)
    df["rank_stability"]  = (df["rank_stability"] * 100).round(1)
    df = df[["rank", "feature", "mean_shap",
             "ci_lower", "ci_upper", "rank_stability"]]

    return df.reset_index(drop=True)


# ── 4. Main Analysis Pipeline ─────────────────────────────────────────────

def run_statistical_tests(models_dir: str = "models",
                           output_dir: str = "outputs/stats"):
    """
    Run all statistical tests for Paper 1 and save results.

    Requires in models/:
        cost_xgb.pkl, cost_scaler.pkl, cost_features.pkl

    Also requires saved predictions from the test set — if not
    cached, re-fetches from BigQuery (sample of 50K for speed).
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info("=== Statistical Tests: Paper 1 (Cost Prediction) ===")

    # ── Load models ───────────────────────────────────────
    logger.info("Loading saved models...")

    from agents.data_agent import DataAgent
    from agents.prediction_agent import PredictionAgent
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from sklearn.preprocessing import StandardScaler
    import shap

    data_agent = DataAgent()
    pred_agent = PredictionAgent()

    # ── Fetch data (use 50K sample for speed) ─────────────
    logger.info("Fetching data for statistical tests (50K sample)...")
    df_adm   = data_agent.get_admissions_cohort(limit=50000)
    df_diag  = data_agent.get_top_diagnoses(limit=50000)
    df_labs  = data_agent.get_admission_labs(limit=50000)
    df_cci   = data_agent.get_charlson_index(limit=50000)
    df_procs = data_agent.get_procedure_counts(limit=50000)

    df_cost  = pred_agent.engineer_cost_features(
        df_adm, df_diag, df_labs, df_cci, df_procs)

    X = df_cost.drop(columns=["los_hours"]).fillna(0)
    y = np.log1p(df_cost["los_hours"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Train all 4 models ────────────────────────────────
    logger.info("Training models on sample for significance testing...")
    models = {
        "Ridge Regression": (
            Ridge(alpha=1.0).fit(X_train_s, y_train),
            X_test_s
        ),
        "Random Forest": (
            RandomForestRegressor(
                n_estimators=200, random_state=42, n_jobs=-1
            ).fit(X_train, y_train),
            X_test.values
        ),
        "XGBoost": (
            XGBRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0
            ).fit(X_train, y_train),
            X_test.values
        ),
        "LightGBM": (
            LGBMRegressor(
                n_estimators=300, learning_rate=0.05, num_leaves=63,
                random_state=42, verbosity=-1
            ).fit(X_train, y_train),
            X_test.values
        ),
    }

    # ── Step 1: Bootstrap CIs for all models ─────────────
    logger.info("Step 1: Bootstrap confidence intervals...")
    bootstrap_results = {}
    predictions       = {}

    for name, (model, X_te) in models.items():
        logger.info(f"  Bootstrapping {name}...")
        preds = model.predict(X_te)
        predictions[name] = preds
        bootstrap_results[name] = bootstrap_metrics(
            y_test.values, preds, log_transform=True
        )

    # ── Step 2: Diebold-Mariano pairwise tests ────────────
    logger.info("Step 2: Diebold-Mariano pairwise tests...")
    model_names = list(models.keys())
    dm_results  = []

    y_true_bt = np.expm1(y_test.values)

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            name_a = model_names[i]
            name_b = model_names[j]
            err_a  = y_true_bt - np.expm1(predictions[name_a])
            err_b  = y_true_bt - np.expm1(predictions[name_b])
            dm     = diebold_mariano_test(err_a, err_b)
            dm_results.append({
                "Model A":      name_a,
                "Model B":      name_b,
                "DM Statistic": dm["dm_stat"],
                "p-value":      dm["p_value"],
                "Significant":  "Yes" if dm["significant"] else "No",
                "Direction":    dm["direction"],
            })

    dm_df = pd.DataFrame(dm_results)

    # ── Step 3: SHAP bootstrap CIs ───────────────────────
    logger.info("Step 3: SHAP bootstrap confidence intervals...")
    xgb_model = models["XGBoost"][0]
    X_shap    = X_test.sample(
        min(2000, len(X_test)), random_state=42
    )

    explainer   = shap.TreeExplainer(xgb_model)
    shap_vals   = explainer(X_shap).values

    shap_ci_df  = bootstrap_shap_rankings(
        shap_vals, X_test.columns.tolist()
    )

    # ── Print results ─────────────────────────────────────
    print("\n" + "="*70)
    print("TABLE 2 — Model Performance with 95% Bootstrap CIs")
    print("="*70)
    print(f"\n{'Model':<25} {'R² (95% CI)':<25} "
          f"{'RMSE hrs (95% CI)':<28} {'MAE hrs (95% CI)'}")
    print("-" * 100)
    for name, res in bootstrap_results.items():
        r2   = res["R2"]
        rmse = res["RMSE"]
        mae  = res["MAE"]
        print(
            f"{name:<25} "
            f"{r2['estimate']:.4f} "
            f"({r2['ci_lower']:.4f}–{r2['ci_upper']:.4f})   "
            f"{rmse['estimate']:.1f} "
            f"({rmse['ci_lower']:.1f}–{rmse['ci_upper']:.1f})   "
            f"{mae['estimate']:.1f} "
            f"({mae['ci_lower']:.1f}–{mae['ci_upper']:.1f})"
        )

    print("\n" + "="*70)
    print("TABLE 3 — Diebold-Mariano Pairwise Model Comparison Tests")
    print("="*70)
    print(dm_df.to_string(index=False))

    print("\n" + "="*70)
    print("TABLE 4 — SHAP Feature Importance with 95% Bootstrap CIs")
    print("="*70)
    print(shap_ci_df.to_string(index=False))
    print("\nRank stability = % of bootstrap samples where feature "
          "appears in top 20")

    # ── Save outputs ──────────────────────────────────────
    path_boot = f"{output_dir}/bootstrap_cis.json"
    path_dm   = f"{output_dir}/diebold_mariano.csv"
    path_shap = f"{output_dir}/shap_bootstrap_cis.csv"

    with open(path_boot, "w") as f:
        json.dump(bootstrap_results, f, indent=2)
    dm_df.to_csv(path_dm, index=False)
    shap_ci_df.to_csv(path_shap, index=False)

    logger.info(f"Bootstrap CIs saved → {path_boot}")
    logger.info(f"DM test results saved → {path_dm}")
    logger.info(f"SHAP CIs saved → {path_shap}")

    # ── Methods section text ──────────────────────────────
    methods_text = f"""
STATISTICAL SIGNIFICANCE METHODS (add to Section 2.7)

Model performance uncertainty was quantified using percentile bootstrap 
resampling (B={N_BOOTSTRAP} iterations) on the held-out test set. At each 
iteration, test-set observations were sampled with replacement and RMSE, 
MAE, and R² were computed on the resampled set after back-transformation 
to native hour units. The 2.5th and 97.5th percentiles of the bootstrap 
distribution were used as 95% confidence interval bounds.

Pairwise model comparisons were conducted using the Diebold-Mariano (DM) 
test for equal predictive accuracy [CITATION: Diebold & Mariano, 1995], 
which evaluates whether the squared prediction errors of two models differ 
significantly on the same test observations. The loss differential was 
defined as the difference in squared errors between each model pair, and 
the DM statistic was computed using a Newey-West variance estimator to 
account for potential autocorrelation in the loss differential series. 
A two-sided p-value < 0.05 was used as the threshold for statistical 
significance.

SHAP feature importance stability was assessed by applying the same 
bootstrap procedure (B={N_BOOTSTRAP}) to the SHAP value computation, 
reporting 95% CIs on mean absolute SHAP values and the proportion of 
bootstrap samples in which each feature appeared in the top 20 (rank 
stability). All statistical analyses were conducted in Python 3.12 using 
scipy {'.'.join([str(x) for x in __import__('scipy').version.version.split('.')[:2]])}.
"""

    with open(f"{output_dir}/methods_text.txt", "w") as f:
        f.write(methods_text)
    print(methods_text)

    logger.info("Statistical tests complete.")
    return {
        "bootstrap": bootstrap_results,
        "dm_tests":  dm_df,
        "shap_cis":  shap_ci_df,
    }


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = run_statistical_tests()
    print("\n✅ Statistical tests complete.")
    print("   Outputs saved to: outputs/stats/")