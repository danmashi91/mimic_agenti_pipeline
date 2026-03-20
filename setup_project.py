"""
Run this script once to scaffold the multi-agent pipeline project.
Usage: python setup_project.py
"""

import os

BASE = "mimic_agent_pipeline"

dirs = [
    f"{BASE}",
    f"{BASE}/agents",           # Individual agent modules
    f"{BASE}/tools",            # Tools agents can call (SQL runner, feature builder, etc.)
    f"{BASE}/models",           # Trained ML models (.pkl, .pt files)
    f"{BASE}/data",             # Local cached/aggregated data (NO raw MIMIC records)
    f"{BASE}/notebooks",        # Exploratory Jupyter notebooks
    f"{BASE}/configs",          # DB connection, API keys, model params
    f"{BASE}/outputs",          # Agent outputs, reports, logs
    f"{BASE}/tests",            # Unit tests
]

files = {
    f"{BASE}/__init__.py": "",
    f"{BASE}/agents/__init__.py": "",
    f"{BASE}/agents/data_agent.py": "# DataAgent: handles MIMIC-IV SQL queries and feature extraction\n",
    f"{BASE}/agents/prediction_agent.py": "# PredictionAgent: loads ML models and runs cost/readmission predictions\n",
    f"{BASE}/agents/cost_analyst_agent.py": "# CostAnalystAgent: interprets prediction outputs and quantifies cost impact\n",
    f"{BASE}/agents/orchestrator.py": "# Orchestrator: coordinates agent communication and task sequencing\n",
    f"{BASE}/tools/__init__.py": "",
    f"{BASE}/tools/sql_runner.py": "# Executes SQL queries against local MIMIC-IV PostgreSQL instance\n",
    f"{BASE}/tools/feature_builder.py": "# Builds feature vectors from raw MIMIC-IV query results\n",
    f"{BASE}/tools/cost_calculator.py": "# Computes cost estimates and DRG-adjusted cost proxies\n",
    f"{BASE}/configs/config.yaml": (
        "# Project Configuration\n\n"
        "database:\n"
        "  host: localhost\n"
        "  port: 5432\n"
        "  name: mimiciv\n"
        "  user: YOUR_DB_USER\n"
        "  password: YOUR_DB_PASSWORD\n\n"
        "llm:\n"
        "  provider: anthropic          # anthropic | openai | local\n"
        "  model: claude-sonnet-4-6\n"
        "  api_key_env: ANTHROPIC_API_KEY  # set in .env, never hardcode\n\n"
        "pipeline:\n"
        "  readmission_window_hours: 72\n"
        "  cost_proxy: drg_adjusted     # drg_adjusted | total_charges\n"
        "  random_seed: 42\n"
    ),
    f"{BASE}/.env.example": (
        "# Copy this to .env and fill in your values\n"
        "# Never commit .env to version control\n\n"
        "ANTHROPIC_API_KEY=your_key_here\n"
        "MIMIC_DB_PASSWORD=your_db_password\n"
    ),
    f"{BASE}/.gitignore": (
        ".env\n"
        "*.pkl\n"
        "*.pt\n"
        "data/\n"
        "__pycache__/\n"
        ".venv/\n"
        "outputs/\n"
    ),
    f"{BASE}/README.md": (
        "# MIMIC-IV Multi-Agent Pipeline\n\n"
        "A multi-agent LLM system for ICU cost prediction and readmission risk analysis.\n\n"
        "## Agents\n"
        "- **DataAgent** — queries MIMIC-IV and builds feature sets\n"
        "- **PredictionAgent** — runs ML models for cost and readmission prediction\n"
        "- **CostAnalystAgent** — interprets results and quantifies cost impact\n"
        "- **Orchestrator** — coordinates the full pipeline\n\n"
        "## Setup\n"
        "1. Copy `.env.example` to `.env` and fill in credentials\n"
        "2. Update `configs/config.yaml` with your DB details\n"
        "3. Install dependencies: `pip install -r requirements.txt`\n\n"
        "## Data Privacy\n"
        "Raw MIMIC-IV records never leave the local machine.\n"
        "Only aggregated/anonymized results are passed to external LLM APIs.\n"
    ),
    f"{BASE}/main.py": (
        "# Entry point for the multi-agent pipeline\n\n"
        "from agents.orchestrator import Orchestrator\n\n"
        "if __name__ == '__main__':\n"
        "    orchestrator = Orchestrator()\n"
        "    orchestrator.run()\n"
    ),
}

# Create directories
for d in dirs:
    os.makedirs(d, exist_ok=True)
    print(f"  [DIR]  {d}/")

# Create files
for path, content in files.items():
    with open(path, "w") as f:
        f.write(content)
    print(f"  [FILE] {path}")

print(f"\nProject scaffolded at: ./{BASE}/")
print("Next step: open the folder in VS Code and run: pip install -r requirements.txt")