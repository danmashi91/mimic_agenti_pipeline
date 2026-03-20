# ─────────────────────────────────────────────────────────
# agents/disparity_agent.py
# DisparityAgent: statistical analysis of ICU cost disparities
# across race, insurance, sex, and age groups (Paper 2).
# ─────────────────────────────────────────────────────────

import os
import json
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from dotenv import load_dotenv
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class DisparityAgent:
    """
    DisparityAgent performs statistical disparity analysis for Paper 2.

    Analyses:
    1. Descriptive statistics by demographic strata
    2. Kruskal-Wallis tests for group differences
    3. Multivariable log-linear regression (adjusted disparities)
    4. Blinder-Oaxaca decomposition (explained vs unexplained gaps)
    5. Visualization: violin plots and heatmap

    All outputs saved to outputs/disparity/
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = load_config(config_path)
        self.output_dir = "outputs/disparity"
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info("DisparityAgent initialized.")

    # ── 1. Descriptive Statistics ─────────────────────────────────────────

    def descriptive_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute mean, median, IQR cost proxy by demographic strata."""
        logger.info("Computing descriptive statistics...")
        results = []

        strata = {
            "race_group":      sorted(df["race_group"].dropna().unique()),
            "insurance_group": sorted(df["insurance_group"].dropna().unique()),
            "gender":          sorted(df["gender"].dropna().unique()),
            "age_group":       ["18-44", "45-64", "65-79", "80+"],
        }

        for col, groups in strata.items():
            for group in groups:
                subset = df[df[col] == group]["cost_proxy"].dropna()
                if len(subset) < 10:
                    continue
                results.append({
                    "strata_type": col,
                    "group":       group,
                    "n":           len(subset),
                    "mean":        round(subset.mean(), 1),
                    "median":      round(subset.median(), 1),
                    "q25":         round(subset.quantile(0.25), 1),
                    "q75":         round(subset.quantile(0.75), 1),
                    "std":         round(subset.std(), 1),
                })

        stats_df = pd.DataFrame(results)
        path = f"{self.output_dir}/descriptive_stats.csv"
        stats_df.to_csv(path, index=False)
        logger.info(f"Descriptive stats saved → {path}")

        print("\n── Descriptive Statistics by Strata ────────────────")
        print(stats_df.to_string(index=False))
        return stats_df

    # ── 2. Kruskal-Wallis Tests ───────────────────────────────────────────

    def kruskal_wallis_tests(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run Kruskal-Wallis H-tests for cost differences across groups.
        Non-parametric — appropriate for skewed cost distributions.
        """
        logger.info("Running Kruskal-Wallis tests...")
        results = []

        for col in ["race_group", "insurance_group", "gender", "age_group"]:
            groups = [
                df[df[col] == g]["cost_proxy"].dropna().values
                for g in df[col].dropna().unique()
                if len(df[df[col] == g]) >= 10
            ]
            if len(groups) < 2:
                continue
            h_stat, p_val = stats.kruskal(*groups)
            results.append({
                "variable":    col,
                "H_statistic": round(h_stat, 3),
                "p_value":     f"{p_val:.2e}",
                "significant": "Yes" if p_val < 0.05 else "No",
                "n_groups":    len(groups),
            })

        kw_df = pd.DataFrame(results)
        path = f"{self.output_dir}/kruskal_wallis_results.csv"
        kw_df.to_csv(path, index=False)
        logger.info(f"Kruskal-Wallis results saved → {path}")

        print("\n── Kruskal-Wallis Test Results ──────────────────────")
        print(kw_df.to_string(index=False))
        return kw_df

    # ── 3. Multivariable Regression ───────────────────────────────────────

    def multivariable_regression(self, df: pd.DataFrame):
        """
        Log-linear regression of cost proxy on demographic variables,
        adjusting for clinical severity (drg_severity, age).
        Returns coefficients with 95% CIs and p-values.
        """
        logger.info("Running multivariable log-linear regression...")

        df = df.copy()
        df["log_cost"] = np.log1p(df["cost_proxy"].clip(lower=0))
        df["drg_severity"] = pd.to_numeric(
            df["drg_severity"], errors="coerce"
        ).fillna(1.0)

        # Set reference categories
        df["race_group"]      = pd.Categorical(
            df["race_group"],
            categories=["White", "Black", "Hispanic", "Asian", "Other/Unknown"]
        )
        df["insurance_group"] = pd.Categorical(
            df["insurance_group"],
            categories=["Medicare", "Medicaid", "Private", "Self-pay/Other"]
        )
        df["age_group"] = pd.Categorical(
            df["age_group"],
            categories=["45-64", "18-44", "65-79", "80+"]
        )

        formula = (
            "log_cost ~ C(race_group, Treatment('White')) "
            "+ C(insurance_group, Treatment('Medicare')) "
            "+ C(gender) "
            "+ C(age_group, Treatment('45-64')) "
            "+ drg_severity"
        )

        df_model = df[[
            "log_cost", "race_group", "insurance_group",
            "gender", "age_group", "drg_severity"
        ]].dropna()

        model = smf.ols(formula, data=df_model).fit()

        coef_df = pd.DataFrame({
            "variable":    model.params.index,
            "coef":        model.params.values.round(4),
            "ci_lower":    model.conf_int()[0].values.round(4),
            "ci_upper":    model.conf_int()[1].values.round(4),
            "p_value":     [f"{p:.3e}" for p in model.pvalues.values],
            "significant": (model.pvalues.values < 0.05),
        })

        path = f"{self.output_dir}/regression_results.csv"
        coef_df.to_csv(path, index=False)
        logger.info(f"Regression results saved → {path}")
        logger.info(
            f"Model R²={model.rsquared:.4f} | "
            f"Adj. R²={model.rsquared_adj:.4f} | "
            f"N={len(df_model):,}"
        )

        print(f"\n── Regression: R²={model.rsquared:.4f} | "
              f"Adj. R²={model.rsquared_adj:.4f} ──────")
        sig = coef_df[coef_df["significant"]].copy()
        print(sig.to_string(index=False))

        return coef_df, model

    # ── 4. Blinder-Oaxaca Decomposition ──────────────────────────────────

    def oaxaca_decomposition(
        self,
        df: pd.DataFrame,
        group_col: str = "race_group",
        group_a: str = "White",
        group_b: str = "Black",
    ) -> dict:
        """
        Simplified Blinder-Oaxaca decomposition.
        Decomposes the cost gap between two groups into:
        - Explained:   due to differences in observable characteristics
        - Unexplained: structural/systemic component
        """
        logger.info(
            f"Running Oaxaca decomposition: {group_a} vs {group_b}..."
        )

        features = ["age", "drg_severity"]
        df = df.copy()
        df["log_cost"] = np.log1p(df["cost_proxy"].clip(lower=0))
        df["drg_severity"] = pd.to_numeric(
            df["drg_severity"], errors="coerce"
        ).fillna(1.0)

        df_clean = df[[group_col, "log_cost"] + features].dropna()

        grp_a = df_clean[df_clean[group_col] == group_a]
        grp_b = df_clean[df_clean[group_col] == group_b]

        if len(grp_a) < 30 or len(grp_b) < 30:
            logger.warning(
                f"Insufficient sample: {group_a}={len(grp_a)}, "
                f"{group_b}={len(grp_b)}. Skipping."
            )
            return {}

        X_a = sm.add_constant(grp_a[features].astype(float).values)
        X_b = sm.add_constant(grp_b[features].astype(float).values)
        y_a = grp_a["log_cost"].values
        y_b = grp_b["log_cost"].values

        model_a = sm.OLS(y_a, X_a).fit()
        model_b = sm.OLS(y_b, X_b).fit()

        mean_a = grp_a[features].mean().values
        mean_b = grp_b[features].mean().values

        raw_gap    = float(y_a.mean() - y_b.mean())
        explained  = float(np.array(mean_a - mean_b, dtype=float) @ model_b.params[1:].astype(float))
        unexplained = raw_gap - explained

        result = {
            "group_a":           group_a,
            "group_b":           group_b,
            "mean_log_cost_a":   round(float(y_a.mean()), 4),
            "mean_log_cost_b":   round(float(y_b.mean()), 4),
            "raw_gap":           round(raw_gap, 4),
            "explained":         round(explained, 4),
            "unexplained":       round(unexplained, 4),
            "pct_unexplained":   round(
                abs(unexplained / raw_gap) * 100, 1
            ) if raw_gap != 0 else 0,
            "n_group_a":         len(grp_a),
            "n_group_b":         len(grp_b),
        }

        fname = f"{group_a}_vs_{group_b}".replace("/", "_")
        path = f"{self.output_dir}/oaxaca_{fname}.json"
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Oaxaca decomposition saved → {path}")

        print(f"\n── Oaxaca: {group_a} vs {group_b} ──────────────────")
        for k, v in result.items():
            print(f"  {k:<25} {v}")

        return result

    # ── 5. Visualizations ────────────────────────────────────────────────

    def plot_disparity_charts(self, df: pd.DataFrame) -> None:
        """Generate violin plots and race × insurance heatmap."""
        logger.info("Generating disparity visualizations...")

        df = df.copy()
        df["log_cost"] = np.log1p(df["cost_proxy"].clip(lower=0))

        # ── Violin plots ──────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        race_order = ["White", "Black", "Hispanic", "Asian", "Other/Unknown"]
        race_data  = df[df["race_group"].isin(race_order)]
        sns.violinplot(
            data=race_data, x="race_group", y="log_cost",
            order=race_order, palette="Set2", ax=axes[0], cut=0
        )
        axes[0].set_title(
            "ICU Cost Proxy by Race/Ethnicity", fontweight="bold"
        )
        axes[0].set_xlabel("Race/Ethnicity")
        axes[0].set_ylabel("Log(Cost Proxy)")
        axes[0].tick_params(axis="x", rotation=20)

        ins_order = ["Medicare", "Medicaid", "Private", "Self-pay/Other"]
        ins_data  = df[df["insurance_group"].isin(ins_order)]
        sns.violinplot(
            data=ins_data, x="insurance_group", y="log_cost",
            order=ins_order, palette="Set1", ax=axes[1], cut=0
        )
        axes[1].set_title(
            "ICU Cost Proxy by Insurance Type", fontweight="bold"
        )
        axes[1].set_xlabel("Insurance Type")
        axes[1].set_ylabel("Log(Cost Proxy)")
        axes[1].tick_params(axis="x", rotation=20)

        plt.suptitle(
            "Paper 2: ICU Cost Disparities — MIMIC-IV",
            fontsize=14, fontweight="bold", y=1.02
        )
        plt.tight_layout()
        path = f"{self.output_dir}/violin_plots.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Violin plots saved → {path}")

        # ── Heatmap: Median cost by Race × Insurance ──────
        pivot = df.groupby(
            ["race_group", "insurance_group"]
        )["cost_proxy"].median().unstack()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            pivot, annot=True, fmt=".0f",
            cmap="YlOrRd", linewidths=0.5, ax=ax,
            cbar_kws={"label": "Median Cost Proxy (hrs × severity)"}
        )
        ax.set_title(
            "Median ICU Cost Proxy: Race/Ethnicity × Insurance Type\n"
            "(MIMIC-IV v3.1, Adult ICU Admissions)",
            fontweight="bold"
        )
        ax.set_xlabel("Insurance Type")
        ax.set_ylabel("Race/Ethnicity")
        plt.tight_layout()
        path = f"{self.output_dir}/heatmap_race_insurance.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Heatmap saved → {path}")

    # ── Full Paper 2 Pipeline ─────────────────────────────────────────────

    def run(self, df: pd.DataFrame) -> dict:
        """Run the complete Paper 2 disparity analysis pipeline."""
        logger.info("=== Running Paper 2 Disparity Analysis ===")

        df = df.copy()
        df = df[df["cost_proxy"] > 0].dropna(subset=["cost_proxy"])
        logger.info(f"Cohort after cleaning: {len(df):,} ICU stays")

        # 1. Descriptive stats
        stats_df = self.descriptive_stats(df)

        # 2. Kruskal-Wallis
        kw_df = self.kruskal_wallis_tests(df)

        # 3. Multivariable regression
        coef_df, model = self.multivariable_regression(df)

        # 4. Oaxaca decompositions
        oaxaca_results = {}
        for col, a, b in [
            ("race_group",      "White",    "Black"),
            ("race_group",      "White",    "Hispanic"),
            ("insurance_group", "Medicare", "Medicaid"),
        ]:
            result = self.oaxaca_decomposition(df, col, a, b)
            oaxaca_results[f"{a}_vs_{b}"] = result

        # 5. Visualizations
        self.plot_disparity_charts(df)

        logger.info("Paper 2 analysis complete.")
        logger.info(f"All outputs saved to ./{self.output_dir}/")

        return {
            "descriptive_stats": stats_df,
            "kruskal_wallis":    kw_df,
            "regression":        coef_df,
            "oaxaca":            oaxaca_results,
        }


# ── Quick test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from agents.data_agent import DataAgent

    data_agent     = DataAgent()
    disparity_agent = DisparityAgent()

    print("Fetching full ICU disparity cohort...")
    df = data_agent.get_icu_disparity_cohort(limit=None)
    print(f"Cohort size: {len(df):,} rows")
    print(f"Race distribution:\n{df['race_group'].value_counts()}\n")

    results = disparity_agent.run(df)

    print("\n✅ Paper 2 analysis complete.")
    print("   Outputs saved to: outputs/disparity/")