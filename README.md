# MIMIC-IV Multi-Agent Pipeline

A multi-agent LLM system for ICU cost prediction and readmission risk analysis.

## Agents
- **DataAgent** — queries MIMIC-IV and builds feature sets
- **PredictionAgent** — runs ML models for cost and readmission prediction
- **CostAnalystAgent** — interprets results and quantifies cost impact
- **Orchestrator** — coordinates the full pipeline

## Setup
1. Copy `.env.example` to `.env` and fill in credentials
2. Update `configs/config.yaml` with your DB details
3. Install dependencies: `pip install -r requirements.txt`

## Data Privacy
Raw MIMIC-IV records never leave the local machine.
Only aggregated/anonymized results are passed to external LLM APIs.
