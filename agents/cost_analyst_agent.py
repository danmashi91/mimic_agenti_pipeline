# ─────────────────────────────────────────────────────────
# agents/cost_analyst_agent.py
# CostAnalystAgent: Claude-powered agent that interprets
# ML model results, quantifies cost impact, and generates
# research-grade narrative insights.
#
# DUA COMPLIANCE: Only aggregated metrics and statistics
# are sent to the Claude API — never raw patient data.
# ─────────────────────────────────────────────────────────

import os
import json
import yaml
from loguru import logger
from dotenv import load_dotenv
import anthropic

load_dotenv()


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class CostAnalystAgent:
    """
    CostAnalystAgent uses the Claude API to interpret ML prediction
    results and generate research-grade cost impact analyses.

    It operates on AGGREGATED metrics only — no raw patient data
    is ever sent to the API, maintaining PhysioNet DUA compliance.

    Capabilities:
    - Interpret model performance metrics in clinical context
    - Generate cost-impact narratives for readmission scenarios
    - Produce Methods and Results section drafts for papers
    - Identify key findings and policy implications
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = load_config(config_path)
        self.model = self.config["llm"]["model"]
        self.max_tokens = self.config["llm"]["max_tokens"]
        api_key = os.getenv(self.config["llm"]["api_key_env"])
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. "
                "Check your .env file."
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        logger.info(f"CostAnalystAgent initialized — model: {self.model}")

    def _call_claude(self, system_prompt: str, user_prompt: str) -> str:
        """Send a prompt to Claude and return the response text."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        return response.content[0].text

    # ── Paper 1: Cost Prediction Analysis ────────────────────────────────

    def interpret_cost_model(
        self,
        model_results: dict,
        dataset_stats: dict,
    ) -> str:
        """
        Interpret cost prediction model results in clinical and
        research context. Generates a Results section narrative.

        Args:
            model_results: dict of model name → {RMSE, MAE, R2}
            dataset_stats: dict with cohort size, target stats, feature count
        """
        logger.info("CostAnalystAgent: interpreting cost prediction results...")

        system_prompt = """You are a health data analytics researcher specializing 
in healthcare cost prediction and clinical informatics. You write clearly, 
precisely, and in the style of peer-reviewed medical informatics journals 
(JAMIA, npj Digital Medicine). You interpret machine learning results 
in clinical context and highlight policy implications."""

        user_prompt = f"""I have trained machine learning models to predict 
hospital length of stay (LOS) as a cost proxy using MIMIC-IV admission data.
Here are the results:

DATASET:
- Total admissions in cohort: {dataset_stats.get('n_rows', 'N/A'):,}
- Feature count: {dataset_stats.get('n_features', 'N/A')}
- Target variable: Length of Stay (hours), log-transformed
- Train/test split: 80/20
- Key features: demographics, insurance type, DRG codes, ICD-10 diagnoses, 
  lab values at admission, Charlson Comorbidity Index, procedure counts,
  admission time features

MODEL PERFORMANCE:
{json.dumps(model_results, indent=2)}

Please provide:
1. A 2-3 paragraph Results section narrative suitable for a research paper
2. Clinical interpretation of the best-performing model's R² score
3. Why Ridge Regression underperformed compared to tree-based models
4. Two key policy implications of being able to predict LOS at admission
5. Limitations to acknowledge in the Discussion section

Write in academic style, third person, past tense."""

        return self._call_claude(system_prompt, user_prompt)

    # ── Paper 3: Readmission Cost Impact Analysis ─────────────────────────

    def interpret_readmission_model(
        self,
        model_results: dict,
        dataset_stats: dict,
    ) -> str:
        """
        Interpret ICU readmission prediction results and generate
        cost-impact analysis narrative.

        Args:
            model_results: dict of model name → {AUROC, AUPRC}
            dataset_stats: dict with cohort size, readmission rate, etc.
        """
        logger.info("CostAnalystAgent: interpreting readmission results...")

        system_prompt = """You are a critical care health economist and 
clinical informatics researcher. You specialize in ICU readmission prediction, 
cost-effectiveness analysis, and health policy. You write in the style of 
Critical Care Medicine and JAMIA journals."""

        user_prompt = f"""I have trained machine learning models to predict 
unplanned ICU readmission within 72 hours of step-down using MIMIC-IV.
Here are the results:

DATASET:
- Total ICU stays in cohort: {dataset_stats.get('n_rows', 'N/A'):,}
- Readmission rate: {dataset_stats.get('readmission_rate', 'N/A')}
- Positive cases (readmitted): {dataset_stats.get('n_positive', 'N/A'):,}
- Negative cases: {dataset_stats.get('n_negative', 'N/A'):,}
- Feature count: {dataset_stats.get('n_features', 'N/A')}
- Key features: demographics, ICU LOS, insurance, care unit type,
  last recorded vitals (HR, SpO2, RR, ABP, Temp),
  24h vital trends (mean, std, delta for each vital),
  fluid balance, discharge hour/day-of-week,
  clinical abnormality flags (HR, SpO2, RR)

MODEL PERFORMANCE:
{json.dumps(model_results, indent=2)}

COST CONTEXT:
- Average ICU readmission adds an estimated $15,000-$30,000 per episode
- U.S. ICU readmission rate is approximately 5-8% nationally
- HRRP (Hospital Readmissions Reduction Program) penalizes hospitals

Please provide:
1. A 2-3 paragraph Results section narrative for a research paper
2. Clinical interpretation of AUROC 0.61 — is this meaningful in context?
3. Why AUPRC is more important than AUROC for this imbalanced problem
4. Cost-impact analysis: estimate annual savings if the best model 
   reduced readmissions by 20% in a 500-bed hospital
5. Three specific clinical interventions this model could trigger
6. Key limitations to address in Discussion

Write in academic style, third person, past tense."""

        return self._call_claude(system_prompt, user_prompt)

    # ── Cost-Impact Scenario Modeling ────────────────────────────────────

    def model_cost_savings(
        self,
        readmission_rate: float,
        n_icu_admissions_per_year: int = 2000,
        avg_readmission_cost: int = 22000,
        reduction_scenarios: list = [0.10, 0.20, 0.30],
    ) -> str:
        """
        Generate cost-savings scenario analysis for different
        readmission reduction levels.

        Args:
            readmission_rate: observed readmission rate (e.g. 0.058)
            n_icu_admissions_per_year: annual ICU volume for a typical hospital
            avg_readmission_cost: average cost per readmission episode ($)
            reduction_scenarios: list of reduction fractions to model
        """
        logger.info("CostAnalystAgent: modeling cost savings scenarios...")

        # Compute scenarios locally — no patient data involved
        scenarios = {}
        baseline_readmissions = int(n_icu_admissions_per_year * readmission_rate)
        baseline_cost = baseline_readmissions * avg_readmission_cost

        for pct in reduction_scenarios:
            prevented = int(baseline_readmissions * pct)
            savings = prevented * avg_readmission_cost
            scenarios[f"{int(pct*100)}% reduction"] = {
                "readmissions_prevented": prevented,
                "annual_cost_savings_usd": savings,
                "as_percentage_of_baseline": round(pct * 100, 1)
            }

        system_prompt = """You are a health economist specializing in 
ICU cost-effectiveness and hospital operations research. You produce 
clear, concise cost-impact analyses suitable for publication in 
health services research journals."""

        user_prompt = f"""Based on MIMIC-IV analysis, I have the following 
ICU readmission cost-impact scenarios for a typical U.S. hospital:

BASELINE:
- Annual ICU admissions: {n_icu_admissions_per_year:,}
- Observed readmission rate: {readmission_rate:.1%}
- Baseline annual readmissions: {baseline_readmissions:,}
- Average cost per readmission: ${avg_readmission_cost:,}
- Total annual readmission cost: ${baseline_cost:,}

REDUCTION SCENARIOS:
{json.dumps(scenarios, indent=2)}

Please provide:
1. A concise cost-impact paragraph suitable for a paper's Discussion section
2. Break-even analysis: how many readmissions must be prevented to justify 
   implementing a clinical decision support tool costing $50,000/year
3. Contextualize these savings against HRRP penalty amounts
4. A one-sentence "headline finding" for the paper's abstract

Write in academic style."""

        result = self._call_claude(system_prompt, user_prompt)

        # Return both computed scenarios and narrative
        return {
            "scenarios": scenarios,
            "narrative": result,
            "baseline": {
                "annual_readmissions": baseline_readmissions,
                "annual_cost": baseline_cost
            }
        }

    # ── Full Pipeline Report ──────────────────────────────────────────────

    def generate_portfolio_summary(
        self,
        cost_results: dict,
        readmission_results: dict,
        cost_stats: dict,
        readmission_stats: dict,
    ) -> str:
        """
        Generate a high-level portfolio summary linking all three papers
        to NIW national interest arguments.
        """
        logger.info("CostAnalystAgent: generating portfolio summary...")

        system_prompt = """You are a health policy researcher and academic 
writing expert. You help researchers articulate the national importance 
of their work for immigration petitions (EB-2 NIW) and grant applications. 
You connect technical findings to federal health policy priorities."""

        user_prompt = f"""I am a data analytics researcher at Saint Louis 
University building a research portfolio on MIMIC-IV healthcare cost analytics.
Here is a summary of results from two completed analyses:

PAPER 1 — Hospital Cost Prediction:
- Dataset: {cost_stats.get('n_rows', 'N/A'):,} admissions
- Best model: Random Forest, R²={cost_results.get('Random Forest', {}).get('R2', 'N/A')}
- Predicts length of stay (cost proxy) from admission-time variables
- Features: demographics, insurance, DRG, CCI, labs, procedures

PAPER 3 — ICU Readmission Prediction:
- Dataset: {readmission_stats.get('n_rows', 'N/A'):,} ICU stays
- Best model: Random Forest AUROC={readmission_results.get('Random Forest', {}).get('AUROC', 'N/A')}, 
  LightGBM AUPRC={readmission_results.get('LightGBM', {}).get('AUPRC', 'N/A')}
- Readmission rate: {readmission_stats.get('readmission_rate', 'N/A')}
- Features: vitals, trends, fluid balance, demographics

Please provide:
1. A 150-word abstract-style summary of the research portfolio's contribution
2. Three specific connections to federal health priorities 
   (CMS, HHS, AHRQ, HRRP) that support an NIW petition
3. A one-paragraph "national importance" statement in NIW petition language
4. Two suggested next steps to strengthen the portfolio further

Be specific, cite real federal programs where relevant."""

        return self._call_claude(system_prompt, user_prompt)


# ── Quick test ────────────────────────────────────────────
if __name__ == "__main__":

    agent = CostAnalystAgent()

    # Sample results from our trained models
    cost_results = {
        "Ridge Regression": {"RMSE": 330.670, "MAE": 74.381, "R2": -2.4006},
        "Random Forest":    {"RMSE": 109.230, "MAE": 45.807, "R2": 0.6289},
        "XGBoost":          {"RMSE": 129.725, "MAE": 56.664, "R2": 0.4766},
        "LightGBM":         {"RMSE": 129.108, "MAE": 56.223, "R2": 0.4816},
    }

    readmission_results = {
        "Logistic Regression": {"AUROC": 0.5942, "AUPRC": 0.0826},
        "Random Forest":       {"AUROC": 0.6141, "AUPRC": 0.1215},
        "XGBoost":             {"AUROC": 0.6095, "AUPRC": 0.1302},
        "LightGBM":            {"AUROC": 0.6085, "AUPRC": 0.1381},
    }

    cost_stats = {
        "n_rows": 912156,
        "n_features": 38,
    }

    readmission_stats = {
        "n_rows": 82585,
        "n_features": 34,
        "readmission_rate": "5.58%",
        "n_positive": 3690,
        "n_negative": 62378,
    }

    # ── Test 1: Cost model interpretation ────────────────
    print("\n" + "="*60)
    print("PAPER 1: COST PREDICTION INTERPRETATION")
    print("="*60)
    cost_narrative = agent.interpret_cost_model(cost_results, cost_stats)
    print(cost_narrative)

    # ── Test 2: Readmission interpretation ───────────────
    print("\n" + "="*60)
    print("PAPER 3: READMISSION PREDICTION INTERPRETATION")
    print("="*60)
    readmission_narrative = agent.interpret_readmission_model(
        readmission_results, readmission_stats)
    print(readmission_narrative)

    # ── Test 3: Cost savings scenarios ───────────────────
    print("\n" + "="*60)
    print("COST SAVINGS SCENARIO ANALYSIS")
    print("="*60)
    savings = agent.model_cost_savings(readmission_rate=0.0558)
    print("\nScenarios:")
    for scenario, data in savings["scenarios"].items():
        print(f"  {scenario}: ${data['annual_cost_savings_usd']:,} saved "
              f"({data['readmissions_prevented']} readmissions prevented)")
    print("\nNarrative:")
    print(savings["narrative"])

    # ── Test 4: Portfolio summary ─────────────────────────
    print("\n" + "="*60)
    print("PORTFOLIO SUMMARY + NIW ARGUMENTS")
    print("="*60)
    summary = agent.generate_portfolio_summary(
        cost_results, readmission_results,
        cost_stats, readmission_stats
    )
    print(summary)