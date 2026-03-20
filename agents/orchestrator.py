# ─────────────────────────────────────────────────────────
# agents/orchestrator.py
# Orchestrator: coordinates DataAgent, PredictionAgent,
# and CostAnalystAgent into a single end-to-end pipeline.
#
# Usage:
#   python main.py                    # full pipeline
#   python main.py --paper 1          # Paper 1 only
#   python main.py --paper 3          # Paper 3 only
#   python main.py --sample 10000     # sample run (dev mode)
# ─────────────────────────────────────────────────────────

import os
import json
import argparse
import datetime
import pandas as pd
from loguru import logger
from dotenv import load_dotenv

from agents.data_agent import DataAgent
from agents.prediction_agent import PredictionAgent
from agents.cost_analyst_agent import CostAnalystAgent

load_dotenv()


class Orchestrator:
    """
    Orchestrator coordinates the full multi-agent pipeline:

    1. DataAgent    → fetches and prepares MIMIC-IV data
    2. PredictionAgent → engineers features and trains models
    3. CostAnalystAgent → interprets results via Claude API

    Outputs are saved to the outputs/ directory as JSON + text reports.
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = config_path
        self.output_dir = "outputs"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/reports", exist_ok=True)
        os.makedirs(f"{self.output_dir}/metrics", exist_ok=True)
        logger.info("Orchestrator initialized.")

    def _save_metrics(self, name: str, metrics: dict) -> None:
        """Save model metrics to JSON."""
        path = f"{self.output_dir}/metrics/{name}.json"
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved → {path}")

    def _save_report(self, name: str, content: str) -> None:
        """Save narrative report to text file."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        path = f"{self.output_dir}/reports/{name}_{timestamp}.txt"
        with open(path, "w") as f:
            f.write(content)
        logger.info(f"Report saved → {path}")

    def _print_section(self, title: str) -> None:
        print(f"\n{'='*65}")
        print(f"  {title}")
        print(f"{'='*65}\n")

    # ── Paper 1 Pipeline ─────────────────────────────────────────────────

    def run_paper1(
        self,
        data_agent: DataAgent,
        pred_agent: PredictionAgent,
        analyst: CostAnalystAgent,
        limit: int = None
    ) -> dict:
        """Full Paper 1 pipeline: cost prediction."""
        self._print_section("PAPER 1: Hospital Cost Prediction")

        # ── Step 1: Fetch data ────────────────────────────
        logger.info("Paper 1 — fetching data from MIMIC-IV...")
        df_adm   = data_agent.get_admissions_cohort(limit=limit)
        df_diag  = data_agent.get_top_diagnoses(limit=limit)
        df_labs  = data_agent.get_admission_labs(limit=limit)
        df_cci   = data_agent.get_charlson_index(limit=limit)
        df_procs = data_agent.get_procedure_counts(limit=limit)

        # ── Step 2: Feature engineering ──────────────────
        df_cost = pred_agent.engineer_cost_features(
            df_adm, df_diag, df_labs, df_cci, df_procs
        )

        # ── Step 3: Train models ──────────────────────────
        cost_results = pred_agent.train_cost_models(df_cost)
        pred_agent.print_results(cost_results, "Paper 1: Cost Prediction")
        self._save_metrics("paper1_cost_results", cost_results)

        # ── Step 4: Claude interpretation ────────────────
        cost_stats = {
            "n_rows": df_cost.shape[0],
            "n_features": df_cost.shape[1] - 1,
        }
        self._print_section("PAPER 1: AI Interpretation")
        narrative = analyst.interpret_cost_model(cost_results, cost_stats)
        print(narrative)
        self._save_report("paper1_interpretation", narrative)

        return {"results": cost_results, "stats": cost_stats}

    # ── Paper 3 Pipeline ─────────────────────────────────────────────────

    def run_paper3(
        self,
        data_agent: DataAgent,
        pred_agent: PredictionAgent,
        analyst: CostAnalystAgent,
        limit: int = None
    ) -> dict:
        """Full Paper 3 pipeline: ICU readmission prediction + cost impact."""
        self._print_section("PAPER 3: ICU Readmission Prediction")

        # ── Step 1: Fetch data ────────────────────────────
        logger.info("Paper 3 — fetching data from MIMIC-IV...")
        df_icu    = data_agent.get_icu_readmission_cohort(limit=limit)
        df_vitals = data_agent.get_icu_discharge_vitals(limit=limit)
        df_fluid  = data_agent.get_fluid_balance(limit=limit)
        df_trends = data_agent.get_icu_vital_trends(limit=limit)

        # ── Step 2: Feature engineering ──────────────────
        df_read = pred_agent.engineer_readmission_features(
            df_icu, df_vitals, df_fluid, df_trends
        )

        readmission_rate = df_read["readmitted"].mean()
        n_positive = int(df_read["readmitted"].sum())
        n_negative = int((df_read["readmitted"] == 0).sum())

        # ── Step 3: Train models ──────────────────────────
        read_results = pred_agent.train_readmission_models(df_read)
        pred_agent.print_results(read_results, "Paper 3: ICU Readmission")
        self._save_metrics("paper3_readmission_results", read_results)

        # ── Step 4: Claude interpretation ────────────────
        readmission_stats = {
            "n_rows": df_read.shape[0],
            "n_features": df_read.shape[1] - 1,
            "readmission_rate": f"{readmission_rate:.2%}",
            "n_positive": n_positive,
            "n_negative": n_negative,
        }

        self._print_section("PAPER 3: AI Interpretation")
        narrative = analyst.interpret_readmission_model(
            read_results, readmission_stats
        )
        print(narrative)
        self._save_report("paper3_interpretation", narrative)

        # ── Step 5: Cost savings scenarios ───────────────
        self._print_section("COST SAVINGS SCENARIO ANALYSIS")
        savings = analyst.model_cost_savings(
            readmission_rate=readmission_rate
        )
        print("\nScenarios:")
        for scenario, data in savings["scenarios"].items():
            print(f"  {scenario}: "
                  f"${data['annual_cost_savings_usd']:,} saved "
                  f"({data['readmissions_prevented']} readmissions prevented)")
        print("\nNarrative:")
        print(savings["narrative"])
        self._save_report("paper3_cost_savings", savings["narrative"])
        self._save_metrics("paper3_cost_scenarios", savings["scenarios"])

        return {
            "results": read_results,
            "stats": readmission_stats,
            "savings": savings["scenarios"]
        }

    # ── Full Pipeline ─────────────────────────────────────────────────────

    def run(self, papers: list = None, limit: int = None) -> None:
        """
        Run the full multi-agent pipeline.

        Args:
            papers: list of paper numbers to run, e.g. [1, 3]
                    defaults to all papers
            limit:  row limit for dev/testing (None = full dataset)
        """
        start = datetime.datetime.now()
        papers = papers or [1, 3]

        self._print_section("MIMIC-IV MULTI-AGENT PIPELINE — STARTING")
        logger.info(f"Running papers: {papers} | Limit: {limit or 'full dataset'}")

        # ── Initialize agents ─────────────────────────────
        data_agent = DataAgent(self.config_path)
        pred_agent = PredictionAgent(self.config_path)
        analyst    = CostAnalystAgent(self.config_path)

        all_results = {}

        # ── Run selected papers ───────────────────────────
        if 1 in papers:
            all_results["paper1"] = self.run_paper1(
                data_agent, pred_agent, analyst, limit=limit
            )

        if 3 in papers:
            all_results["paper3"] = self.run_paper3(
                data_agent, pred_agent, analyst, limit=limit
            )

        # ── Portfolio summary (if both papers ran) ────────
        if 1 in papers and 3 in papers:
            self._print_section("PORTFOLIO SUMMARY + NIW ARGUMENTS")
            summary = analyst.generate_portfolio_summary(
                cost_results=all_results["paper1"]["results"],
                readmission_results=all_results["paper3"]["results"],
                cost_stats=all_results["paper1"]["stats"],
                readmission_stats=all_results["paper3"]["stats"],
            )
            print(summary)
            self._save_report("portfolio_niw_summary", summary)

        # ── Save full results ─────────────────────────────
        self._save_metrics("full_pipeline_results", {
            k: v["results"] for k, v in all_results.items()
        })

        elapsed = datetime.datetime.now() - start
        self._print_section(f"PIPELINE COMPLETE — {elapsed.seconds//60}m {elapsed.seconds%60}s")
        logger.info(f"All outputs saved to ./{self.output_dir}/")
        print(f"  Reports  → ./{self.output_dir}/reports/")
        print(f"  Metrics  → ./{self.output_dir}/metrics/\n")


# ── CLI entry point ───────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="MIMIC-IV Multi-Agent Pipeline"
    )
    parser.add_argument(
        "--paper", type=int, nargs="+",
        choices=[1, 3], default=None,
        help="Which papers to run (1, 3, or both). Default: all."
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Row limit for dev mode (e.g. --sample 5000). Default: full dataset."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    orchestrator = Orchestrator()
    orchestrator.run(
        papers=args.paper,
        limit=args.sample
    )