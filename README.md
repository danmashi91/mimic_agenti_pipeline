# MIMIC-IV Health Data Analytics Pipeline

A multi-agent machine learning pipeline for healthcare cost prediction, ICU cost disparity analysis, and ICU readmission prediction using the [MIMIC-IV](https://physionet.org/content/mimiciv/) clinical database.

Built as part of a research portfolio at **Saint Louis University** targeting publication in JAMIA, Critical Care Medicine, and Health Affairs.

---

## Research Papers

| Paper | Topic | Best Result |
|---|---|---|
| **Paper 1** | Hospital cost prediction from admission data | Random Forest R²=0.629 (n=912K) |
| **Paper 2** | ICU cost disparities by race, insurance, sex, age | Regression R²=0.377, 88.9% unexplained Medicare/Medicaid gap |
| **Paper 3** | ICU readmission prediction + cost-impact analysis | XGBoost AUROC=0.614, Hybrid BiLSTM AUROC=0.605 |

---

## Architecture

```
main.py
  └── Orchestrator
        ├── DataAgent         ← MIMIC-IV BigQuery queries
        ├── PredictionAgent   ← Feature engineering + ML models
        ├── DisparityAgent    ← Statistical disparity analysis
        ├── CostAnalystAgent  ← Claude API interpretation
        └── tools/
              ├── shap_analysis.py     ← SHAP feature importance
              └── lstm_pipeline.py     ← BiLSTM + Hybrid models
```

### Multi-Agent Design

```
MIMIC-IV (BigQuery)
        ↓
   DataAgent          Fetches & prepares clinical data
        ↓
PredictionAgent       Engineers features, trains ML models
        ↓
DisparityAgent        Statistical analysis (Paper 2)
        ↓
CostAnalystAgent      Claude API → research narratives
        ↓
   Orchestrator       Coordinates pipeline, saves outputs
```

---

## Data Privacy & DUA Compliance

This pipeline is designed to comply with the [PhysioNet Data Use Agreement](https://physionet.org/content/mimiciv/view-dua/3.1/):

- Raw patient records **never leave the local machine**
- Only **aggregated metrics and statistics** are sent to the Claude API
- All BigQuery queries run against the PhysioNet-hosted dataset
- No raw MIMIC-IV data is stored in this repository

---

## Setup

### Prerequisites

- Python 3.12+
- [PhysioNet credentialed access](https://physionet.org/register/) to MIMIC-IV
- Google Cloud account with BigQuery access to `physionet-data`
- Anthropic API key ([console.anthropic.com](https://console.anthropic.com))
- Node.js (for Word report generation)

### 1. Clone and create virtual environment

```bash
git clone https://github.com/YOUR_USERNAME/mimic_agent_pipeline.git
cd mimic_agent_pipeline
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
# .venv\Scripts\activate   # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install db-dtypes  # required by google-cloud-bigquery
```

### 3. Authenticate with Google Cloud

```bash
gcloud init
gcloud auth application-default login
```

Select your GCP project that has BigQuery access to `physionet-data.mimiciv_3_1_hosp`.

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:
```
ANTHROPIC_API_KEY=your_key_here
GCP_PROJECT_ID=your-gcp-project-id
```

Edit `configs/config.yaml` — update `project_id` with your GCP project ID.

### 5. Verify connection

```bash
python -c "
from agents.data_agent import DataAgent
agent = DataAgent()
agent.get_summary_stats()
"
```

Expected output:
```
── MIMIC-IV Table Summary ──────────────────────
  admissions              546,028 rows
  patients                364,627 rows
  icustays                 94,458 rows
  ...
```

---

## Usage

### Full pipeline (all papers)

```bash
python main.py
```

### Single paper

```bash
python main.py --paper 1      # Cost prediction
python main.py --paper 3      # ICU readmission
```

### Development mode (fast, sampled)

```bash
python main.py --sample 5000
```

### Paper 2 — Disparity analysis

```bash
python -m agents.disparity_agent
```

### SHAP feature importance

```bash
python -m tools.shap_analysis
```

### LSTM / Hybrid model (Paper 3)

```bash
# Hybrid model (recommended)
python -m tools.lstm_pipeline --mode hybrid --epochs 30

# LSTM only
python -m tools.lstm_pipeline --mode lstm --epochs 30

# Both
python -m tools.lstm_pipeline --mode both --epochs 30
```

### Generate Word report

```bash
node generate_report.js ./outputs
```

---

## Project Structure

```
mimic_agent_pipeline/
├── agents/
│   ├── data_agent.py           # MIMIC-IV BigQuery queries
│   ├── prediction_agent.py     # ML models (Papers 1 & 3)
│   ├── disparity_agent.py      # Statistical analysis (Paper 2)
│   ├── cost_analyst_agent.py   # Claude API interpretation
│   └── orchestrator.py         # Pipeline coordinator
├── tools/
│   ├── shap_analysis.py        # SHAP feature importance
│   └── lstm_pipeline.py        # BiLSTM + Hybrid models
├── configs/
│   └── config.yaml             # Database + model config
├── models/                     # Saved ML models (.pkl, .pt)
├── outputs/
│   ├── metrics/                # JSON results files
│   ├── reports/                # Claude-generated narratives
│   ├── shap/                   # SHAP plots + CSVs
│   └── disparity/              # Paper 2 analysis outputs
├── notebooks/                  # Exploratory Jupyter notebooks
├── main.py                     # Entry point
├── generate_report.js          # Word document generator
├── requirements.txt
├── .env.example
└── README.md
```

---

## Key Results

### Paper 1 — Cost Prediction (n=912,156 admissions)

| Model | RMSE (hrs) | MAE (hrs) | R² |
|---|---|---|---|
| Ridge Regression | 330.7 | 74.4 | -2.40 |
| Random Forest | **109.2** | **45.8** | **0.629** |
| XGBoost | 129.7 | 56.7 | 0.477 |
| LightGBM | 129.1 | 56.2 | 0.482 |

**Top SHAP features:** DRG type, hemoglobin, discharge location, procedure count, sodium

### Paper 2 — ICU Cost Disparities (n=183,375 ICU stays)

- All demographic disparities statistically significant (Kruskal-Wallis p<0.001)
- Adjusted regression R²=0.377
- **Medicare vs Medicaid gap: 88.9% unexplained** by clinical severity
- DRG severity strongest predictor (coef=0.569)

### Paper 3 — ICU Readmission (n=82,585 ICU stays, 5.58% readmission rate)

| Model | AUROC | AUPRC |
|---|---|---|
| Logistic Regression | 0.594 | 0.083 |
| XGBoost | 0.610 | 0.130 |
| BiLSTM + Attention | 0.585 | 0.082 |
| **Hybrid (BiLSTM + Tabular)** | **0.605** | **0.094** |

**Top SHAP features:** ICU LOS, care unit type, total fluid output, age, 24h respiratory rate mean

**Cost impact:** 20% readmission reduction → $484K annual savings; break-even at just 3 prevented readmissions/year

---

## Target Journals

| Paper | Primary Target | Secondary |
|---|---|---|
| Paper 1 | JAMIA / npj Digital Medicine | BMC Medical Informatics |
| Paper 2 | JAMA Network Open / Health Affairs | American Journal of Public Health |
| Paper 3 | Critical Care / JAMIA Open | Journal of Critical Care |

---

## Requirements

See `requirements.txt` for full list. Key dependencies:

- `google-cloud-bigquery` — MIMIC-IV data access
- `anthropic` — Claude API for result interpretation
- `langchain`, `langgraph` — agent orchestration
- `xgboost`, `lightgbm`, `scikit-learn` — ML models
- `torch` — PyTorch for LSTM/Hybrid models
- `shap` — feature importance
- `statsmodels` — disparity regression + Oaxaca decomposition

---

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{mimic_agent_pipeline,
  author    = {Abubakar},
  title     = {MIMIC-IV Health Data Analytics Pipeline},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/YOUR_USERNAME/mimic_agent_pipeline}
}
```

Also cite MIMIC-IV per PhysioNet requirements:
> Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2023). MIMIC-IV (version 3.1). PhysioNet.

---

## License

This repository contains only code. No MIMIC-IV data is included or distributed.
Code is released under the MIT License.

---

## Acknowledgements

- MIMIC-IV database: Beth Israel Deaconess Medical Center / PhysioNet
- Saint Louis University School for Professional Studies
- Anthropic Claude API for AI-powered research interpretation
