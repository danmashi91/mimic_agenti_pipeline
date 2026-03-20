# ─────────────────────────────────────────────────────────
# tools/shap_analysis.py
# SHAP feature importance analysis for Paper 1 and Paper 3.
# Loads saved XGBoost models and computes SHAP values.
# Outputs: summary plots + feature importance tables.
# ─────────────────────────────────────────────────────────

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving files
import matplotlib.pyplot as plt
import shap
from loguru import logger
from dotenv import load_dotenv

from agents.data_agent import DataAgent
from agents.prediction_agent import PredictionAgent

load_dotenv()


class SHAPAnalyst:
    """
    SHAPAnalyst computes and visualizes SHAP feature importance
    for trained XGBoost models from Papers 1 and 3.

    Outputs saved to outputs/shap/:
    - paper1_shap_summary.png      — beeswarm plot
    - paper1_shap_bar.png          — mean |SHAP| bar chart
    - paper1_top_features.csv      — top 20 features table
    - paper3_shap_summary.png
    - paper3_shap_bar.png
    - paper3_top_features.csv
    """

    def __init__(self, models_dir: str = "models",
                 output_dir: str = "outputs/shap"):
        self.models_dir = models_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("SHAPAnalyst initialized.")

    def _load_model_and_features(self, model_file: str,
                                  features_file: str):
        """Load a saved XGBoost model and its feature names."""
        model = joblib.load(f"{self.models_dir}/{model_file}")
        features = joblib.load(f"{self.models_dir}/{features_file}")
        logger.info(f"Loaded model: {model_file} | "
                    f"Features: {len(features)}")
        return model, features

    def _compute_shap(self, model, X: pd.DataFrame,
                      sample_size: int = 2000) -> shap.Explanation:
        """
        Compute SHAP values using TreeExplainer.
        Samples for speed — 2000 rows is sufficient for stable estimates.
        """
        if len(X) > sample_size:
            X_sample = X.sample(sample_size, random_state=42)
            logger.info(f"Sampled {sample_size:,} rows for SHAP computation.")
        else:
            X_sample = X

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_sample)
        return shap_values, X_sample

    def _plot_summary(self, shap_values, X_sample: pd.DataFrame,
                      title: str, output_prefix: str,
                      max_display: int = 20) -> pd.DataFrame:
        """Generate beeswarm + bar plots and return feature importance table."""

        # ── Beeswarm plot ────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X_sample,
            max_display=max_display,
            show=False,
            plot_type="dot"
        )
        plt.title(title, fontsize=13, fontweight="bold", pad=15)
        plt.tight_layout()
        beeswarm_path = f"{self.output_dir}/{output_prefix}_shap_summary.png"
        plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Beeswarm plot saved → {beeswarm_path}")

        # ── Bar plot ─────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X_sample,
            max_display=max_display,
            show=False,
            plot_type="bar"
        )
        plt.title(f"{title} — Mean |SHAP| Importance",
                  fontsize=13, fontweight="bold", pad=15)
        plt.tight_layout()
        bar_path = f"{self.output_dir}/{output_prefix}_shap_bar.png"
        plt.savefig(bar_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Bar plot saved → {bar_path}")

        # ── Feature importance table ──────────────────────
        if hasattr(shap_values, "values"):
            vals = np.abs(shap_values.values)
        else:
            vals = np.abs(shap_values)

        mean_shap = vals.mean(axis=0)
        feature_importance = pd.DataFrame({
            "feature":    X_sample.columns.tolist(),
            "mean_shap":  mean_shap,
        }).sort_values("mean_shap", ascending=False).head(max_display)
        feature_importance["rank"] = range(1, len(feature_importance) + 1)
        feature_importance = feature_importance[
            ["rank", "feature", "mean_shap"]
        ].reset_index(drop=True)

        csv_path = f"{self.output_dir}/{output_prefix}_top_features.csv"
        feature_importance.to_csv(csv_path, index=False)
        logger.info(f"Feature table saved → {csv_path}")

        return feature_importance

    # ── Paper 1: Cost Prediction SHAP ────────────────────

    def analyze_paper1(self, limit: int = None) -> pd.DataFrame:
        """
        Run SHAP analysis for Paper 1 cost prediction model.
        Requires cost_xgb.pkl and cost_features.pkl in models/.
        """
        logger.info("=== SHAP Analysis: Paper 1 (Cost Prediction) ===")

        # Load model
        model, feature_names = self._load_model_and_features(
            "cost_xgb.pkl", "cost_features.pkl"
        )

        # Rebuild feature matrix
        data_agent = DataAgent()
        pred_agent = PredictionAgent()

        logger.info("Fetching data for SHAP analysis...")
        df_adm   = data_agent.get_admissions_cohort(limit=limit)
        df_diag  = data_agent.get_top_diagnoses(limit=limit)
        df_labs  = data_agent.get_admission_labs(limit=limit)
        df_cci   = data_agent.get_charlson_index(limit=limit)
        df_procs = data_agent.get_procedure_counts(limit=limit)

        df_cost = pred_agent.engineer_cost_features(
            df_adm, df_diag, df_labs, df_cci, df_procs
        )

        X = df_cost.drop(columns=["los_hours"]).fillna(0)

        # Align columns to training features
        missing = set(feature_names) - set(X.columns)
        for col in missing:
            X[col] = 0
        X = X[feature_names]

        # Compute SHAP
        shap_values, X_sample = self._compute_shap(model, X)

        # Plot and save
        feature_table = self._plot_summary(
            shap_values, X_sample,
            title="Paper 1: Cost Prediction — SHAP Feature Importance\n"
                  "(XGBoost, MIMIC-IV, n=912K admissions)",
            output_prefix="paper1"
        )

        print("\n── Paper 1: Top 20 Features by SHAP Importance ──────")
        print(feature_table.to_string(index=False))

        return feature_table

    # ── Paper 3: Readmission Prediction SHAP ─────────────

    def analyze_paper3(self, limit: int = None) -> pd.DataFrame:
        """
        Run SHAP analysis for Paper 3 ICU readmission model.
        Requires readmission_xgb.pkl and readmission_features.pkl.
        """
        logger.info("=== SHAP Analysis: Paper 3 (ICU Readmission) ===")

        # Load model
        model, feature_names = self._load_model_and_features(
            "readmission_xgb.pkl", "readmission_features.pkl"
        )

        # Rebuild feature matrix
        data_agent = DataAgent()
        pred_agent = PredictionAgent()

        logger.info("Fetching data for SHAP analysis...")
        df_icu    = data_agent.get_icu_readmission_cohort(limit=limit)
        df_vitals = data_agent.get_icu_discharge_vitals(limit=limit)
        df_fluid  = data_agent.get_fluid_balance(limit=limit)
        df_trends = data_agent.get_icu_vital_trends(limit=limit)

        df_read = pred_agent.engineer_readmission_features(
            df_icu, df_vitals, df_fluid, df_trends
        )

        X = df_read.drop(columns=["readmitted"]).fillna(0)

        # Align columns to training features
        missing = set(feature_names) - set(X.columns)
        for col in missing:
            X[col] = 0
        X = X[feature_names]

        # Compute SHAP
        shap_values, X_sample = self._compute_shap(model, X)

        # Plot and save
        feature_table = self._plot_summary(
            shap_values, X_sample,
            title="Paper 3: ICU Readmission — SHAP Feature Importance\n"
                  "(XGBoost, MIMIC-IV, n=82K ICU stays)",
            output_prefix="paper3"
        )

        print("\n── Paper 3: Top 20 Features by SHAP Importance ──────")
        print(feature_table.to_string(index=False))

        return feature_table

    # ── Dependence plots for top features ────────────────

    def plot_dependence(self, shap_values, X_sample: pd.DataFrame,
                        feature: str, output_prefix: str) -> None:
        """
        Generate a SHAP dependence plot for a specific feature,
        showing how its value relates to its SHAP impact.
        """
        if feature not in X_sample.columns:
            logger.warning(f"Feature '{feature}' not found in dataset.")
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        shap.dependence_plot(
            feature, shap_values.values, X_sample,
            show=False, ax=ax
        )
        plt.title(f"SHAP Dependence: {feature}",
                  fontsize=12, fontweight="bold")
        plt.tight_layout()
        path = f"{self.output_dir}/{output_prefix}_dep_{feature}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Dependence plot saved → {path}")


# ── Quick test ────────────────────────────────────────────
if __name__ == "__main__":
    analyst = SHAPAnalyst()

    # Paper 1 — use sample for speed
    print("\nRunning SHAP for Paper 1 (cost prediction)...")
    table1 = analyst.analyze_paper1(limit=20000)

    # Paper 3 — use full dataset
    print("\nRunning SHAP for Paper 3 (ICU readmission)...")
    table3 = analyst.analyze_paper3(limit=None)

    print("\n✅ SHAP analysis complete.")
    print(f"   Plots and tables saved to: outputs/shap/")