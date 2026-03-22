# api/main.py
# FastAPI backend for MIMIC-IV Clinical Analytics Demo
# Serves predictions, SHAP values, disparity data, and LLM narratives

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import joblib
import json
import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

app = FastAPI(title="MIMIC-IV Clinical Analytics API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load saved models on startup ───────────────────────────────────────────
MODELS_DIR = "models"
cost_model    = joblib.load(f"{MODELS_DIR}/cost_xgb.pkl")
cost_scaler   = joblib.load(f"{MODELS_DIR}/cost_scaler.pkl")
cost_features = joblib.load(f"{MODELS_DIR}/cost_features.pkl")
read_model    = joblib.load(f"{MODELS_DIR}/readmission_xgb.pkl")
read_scaler   = joblib.load(f"{MODELS_DIR}/readmission_scaler.pkl")
read_features = joblib.load(f"{MODELS_DIR}/readmission_features.pkl")

import shap
cost_explainer = shap.TreeExplainer(cost_model)
read_explainer = shap.TreeExplainer(read_model)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ── Preloaded disparity data ───────────────────────────────────────────────
DISPARITY_DATA = {
    "race_stats": [
        {"group": "White",         "n": 121558, "mean": 183.7, "median": 76.0, "q25": 38.0, "q75": 180.0},
        {"group": "Black",         "n": 20091,  "mean": 197.6, "median": 78.0, "q25": 38.0, "q75": 188.0},
        {"group": "Hispanic",      "n": 7004,   "mean": 187.8, "median": 75.0, "q25": 37.0, "q75": 180.0},
        {"group": "Asian",         "n": 5612,   "mean": 195.9, "median": 76.0, "q25": 37.0, "q75": 184.0},
        {"group": "Other/Unknown", "n": 29110,  "mean": 235.3, "median": 90.0, "q25": 42.0, "q75": 227.0},
    ],
    "insurance_stats": [
        {"group": "Medicare",       "n": 100523, "mean": 194.2, "median": 84.0, "q25": 41.0, "q75": 195.0},
        {"group": "Medicaid",       "n": 27735,  "mean": 203.3, "median": 76.0, "q25": 37.0, "q75": 189.0},
        {"group": "Private",        "n": 4503,   "mean": 180.6, "median": 72.0, "q25": 35.0, "q75": 176.0},
        {"group": "Self-pay/Other", "n": 50614,  "mean": 189.6, "median": 71.0, "q25": 34.0, "q75": 172.0},
    ],
    "heatmap": {
        "White":         {"Medicare": 81, "Medicaid": 75, "Private": 71, "Self-pay/Other": 69},
        "Black":         {"Medicare": 84, "Medicaid": 75, "Private": 69, "Self-pay/Other": 72},
        "Hispanic":      {"Medicare": 85, "Medicaid": 69, "Private": 62, "Self-pay/Other": 72},
        "Asian":         {"Medicare": 84, "Medicaid": 76, "Private": 62, "Self-pay/Other": 66},
        "Other/Unknown": {"Medicare": 96, "Medicaid": 94, "Private": 91, "Self-pay/Other": 79},
    },
    "oaxaca": {
        "White vs Black":       {"raw_gap": -0.019, "explained": -0.022, "pct_unexplained": 15.6},
        "White vs Hispanic":    {"raw_gap":  0.014, "explained":  0.036, "pct_unexplained": 157.2},
        "Medicare vs Medicaid": {"raw_gap":  0.045, "explained":  0.085, "pct_unexplained": 88.9},
    },
    "kruskal_wallis": {
        "race":      {"H": 386.35, "p": "2.48e-82"},
        "insurance": {"H": 444.70, "p": "4.59e-96"},
        "sex":       {"H": 53.82,  "p": "2.20e-13"},
        "age":       {"H": 478.18, "p": "2.56e-103"},
    }
}

# ── Schemas ────────────────────────────────────────────────────────────────

class CostPredictionInput(BaseModel):
    age: float = 65.0
    gender: str = "M"
    race: str = "WHITE"
    insurance: str = "Medicare"
    admission_type: str = "URGENT"
    admission_location: str = "EMERGENCY ROOM"
    discharge_location: str = "HOME"
    drg_type: str = "HCFA"
    drg_severity: float = 2.0
    drg_mortality: float = 2.0
    hemoglobin: float = 11.0
    creatinine: float = 1.2
    sodium: float = 138.0
    wbc: float = 8.5
    potassium: float = 4.0
    lactate: float = 1.5
    cci_total_score: float = 2.0
    procedure_count: float = 3.0

class ReadmissionInput(BaseModel):
    age: float = 65.0
    gender: str = "M"
    race: str = "WHITE"
    insurance: str = "Medicare"
    first_careunit: str = "Medical Intensive Care Unit (MICU)"
    icu_los_hours: float = 72.0
    heart_rate_last: float = 85.0
    spo2_last: float = 96.0
    resp_rate_last: float = 18.0
    abp_mean_last: float = 78.0
    temp_f_last: float = 98.6
    total_output_ml: float = 1200.0
    output_event_count: float = 8.0
    hr_mean_24h: float = 88.0
    hr_std_24h: float = 12.0
    hr_delta_24h: float = -5.0
    spo2_mean_24h: float = 95.5
    spo2_std_24h: float = 1.8
    spo2_delta_24h: float = 0.5
    rr_mean_24h: float = 19.0
    rr_std_24h: float = 3.0
    rr_delta_24h: float = 1.0
    abp_mean_24h: float = 76.0
    abp_std_24h: float = 8.0
    abp_delta_24h: float = -2.0
    temp_mean_24h: float = 98.4
    temp_std_24h: float = 0.4
    temp_delta_24h: float = 0.2

class DisparityFilterInput(BaseModel):
    dimension: str = "race"  # race | insurance | sex | age
    selected_groups: Optional[List[str]] = None

# ── Helper: build feature vector ───────────────────────────────────────────

def _encode_categorical(val: str, choices: list) -> int:
    val_lower = val.lower().strip()
    for i, c in enumerate(sorted(choices)):
        if c.lower() in val_lower or val_lower in c.lower():
            return i
    return 0

def build_cost_vector(inp: CostPredictionInput) -> np.ndarray:
    """Build feature vector aligned to cost_features list."""
    row = {f: 0.0 for f in cost_features}
    mappings = {
        "age": inp.age,
        "age_group": 1.0 if inp.age < 45 else (2.0 if inp.age < 65 else (3.0 if inp.age < 80 else 4.0)),
        "admit_hour": 10.0, "admit_dow": 1.0, "admit_month": 6.0,
        "drg_severity": inp.drg_severity,
        "drg_mortality": inp.drg_mortality,
        "hemoglobin": inp.hemoglobin,
        "creatinine": inp.creatinine,
        "sodium": inp.sodium,
        "wbc": inp.wbc,
        "potassium": inp.potassium,
        "lactate": inp.lactate,
        "cci_total_score": inp.cci_total_score,
        "procedure_count": inp.procedure_count,
        "unique_procedures": max(inp.procedure_count - 1, 0),
        "gender": 1 if inp.gender.upper() == "M" else 0,
        "insurance": _encode_categorical(inp.insurance, ["medicaid","medicare","other","private","self-pay"]),
        "race": _encode_categorical(inp.race, ["asian","black","hispanic","other","unknown","white"]),
        "admission_type": _encode_categorical(inp.admission_type, ["ambulatory","direct emer.","elective","eu observation","ew emer.","observation admit","surgical same day admission","urgent"]),
        "admission_location": _encode_categorical(inp.admission_location, ["ambulatory surgery transfer","clinic referral","emergency room","information not available","internal transfer to or from psych","pacu","physician referral","transfer from hospital","transfer from skilled nursing facility","walk-in/self referral"]),
        "discharge_location": _encode_categorical(inp.discharge_location, ["assisted living","chronic/long term acute care","died","home","home health care","hospice","other facility","rehab","skilled nursing facility"]),
        "drg_type": 1 if inp.drg_type.upper() == "HCFA" else 0,
    }
    for k, v in mappings.items():
        if k in row:
            row[k] = float(v)
    return np.array([row[f] for f in cost_features], dtype=np.float32)

def build_read_vector(inp: ReadmissionInput) -> np.ndarray:
    row = {f: 0.0 for f in read_features}
    field_map = {k: v for k, v in inp.dict().items()}
    field_map["gender"] = 1 if inp.gender.upper() == "M" else 0
    field_map["insurance"] = _encode_categorical(inp.insurance, ["medicaid","medicare","other","private","self-pay"])
    field_map["race"] = _encode_categorical(inp.race, ["asian","black","hispanic","other","unknown","white"])
    field_map["first_careunit"] = _encode_categorical(inp.first_careunit, ["cardiac vascular intensive care unit (cvicu)","coronary care unit (ccu)","medical intensive care unit (micu)","medical/surgical intensive care unit (micu/sicu)","neuro intermediate","neuro stepdown","neuro surgical intensive care unit (neuro sicu)","surgery/vascular/intermediate","surgical intensive care unit (sicu)","trauma sicu (tsicu)"])
    field_map["discharge_hour"] = 14.0
    field_map["discharge_dow"]  = 2.0
    field_map["hr_abnormal"]    = 1 if inp.heart_rate_last < 50 or inp.heart_rate_last > 120 else 0
    field_map["spo2_low"]       = 1 if inp.spo2_last < 92 else 0
    field_map["rr_abnormal"]    = 1 if inp.resp_rate_last < 8 or inp.resp_rate_last > 30 else 0
    for k in read_features:
        if k in field_map:
            row[k] = float(field_map[k])
    return np.array([row[f] for f in read_features], dtype=np.float32)

# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "models": ["cost_xgb", "readmission_xgb"]}


@app.post("/predict/cost")
def predict_cost(inp: CostPredictionInput):
    try:
        X = build_cost_vector(inp).reshape(1, -1)
        log_pred = float(cost_model.predict(X)[0])
        los_hours = float(np.expm1(log_pred))
        los_days  = los_hours / 24
        cost_est  = los_days * 3500  # ~$3,500/day ICU-adjusted

        # SHAP
        shap_vals = cost_explainer.shap_values(X)[0]
        top_idx   = np.argsort(np.abs(shap_vals))[::-1][:8]
        shap_top  = [
            {"feature": cost_features[i], "value": float(shap_vals[i]),
             "feature_value": float(X[0][i])}
            for i in top_idx
        ]

        # Bootstrap CI (simple ±10% for demo)
        ci_low  = los_hours * 0.88
        ci_high = los_hours * 1.12

        return {
            "los_hours":   round(los_hours, 1),
            "los_days":    round(los_days, 2),
            "cost_estimate_usd": round(cost_est, 0),
            "ci_low_hours":  round(ci_low, 1),
            "ci_high_hours": round(ci_high, 1),
            "shap_values": shap_top,
            "model": "XGBoost (R²=0.629, CV R²=0.636)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/readmission")
def predict_readmission(inp: ReadmissionInput):
    try:
        X = build_read_vector(inp).reshape(1, -1)
        prob = float(read_model.predict_proba(X)[0][1])

        risk_level = "Low" if prob < 0.10 else ("Moderate" if prob < 0.20 else "High")
        cost_impact = prob * 22000

        # SHAP
        shap_vals = read_explainer.shap_values(X)[0]
        top_idx   = np.argsort(np.abs(shap_vals))[::-1][:8]
        shap_top  = [
            {"feature": read_features[i], "value": float(shap_vals[i]),
             "feature_value": float(X[0][i])}
            for i in top_idx
        ]

        return {
            "readmission_probability": round(prob, 4),
            "readmission_probability_pct": round(prob * 100, 1),
            "risk_level": risk_level,
            "expected_cost_impact_usd": round(cost_impact, 0),
            "population_baseline_pct": 5.58,
            "shap_values": shap_top,
            "model": "XGBoost (AUROC=0.610, AUPRC=0.130)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/disparity/data")
def get_disparity_data():
    return DISPARITY_DATA


@app.post("/disparity/filter")
def filter_disparity(inp: DisparityFilterInput):
    dim = inp.dimension.lower()
    if dim == "race":
        data = DISPARITY_DATA["race_stats"]
    elif dim == "insurance":
        data = DISPARITY_DATA["insurance_stats"]
    else:
        data = DISPARITY_DATA["race_stats"]

    if inp.selected_groups:
        data = [d for d in data if d["group"] in inp.selected_groups]

    return {
        "dimension": dim,
        "stats": data,
        "kruskal_wallis": DISPARITY_DATA["kruskal_wallis"].get(dim, {}),
        "oaxaca": DISPARITY_DATA["oaxaca"]
    }


@app.post("/interpret/cost")
def interpret_cost(payload: Dict[str, Any]):
    los_hours = payload.get("los_hours", 72)
    los_days  = payload.get("los_days", 3.0)
    cost_est  = payload.get("cost_estimate_usd", 10000)
    shap_vals = payload.get("shap_values", [])
    patient   = payload.get("patient_inputs", {})

    top_features = ", ".join([
        f"{s['feature']} ({'+' if s['value']>0 else ''}{s['value']:.3f})"
        for s in shap_vals[:3]
    ])

    prompt = f"""You are a clinical informatics expert interpreting a machine learning prediction for a hospital administrator.

Patient profile: {json.dumps(patient, indent=2)}

Model output:
- Predicted LOS: {los_days:.1f} days ({los_hours:.0f} hours)
- Estimated cost: ${cost_est:,.0f}
- Top SHAP drivers: {top_features}

Write a concise 3-sentence clinical interpretation:
1. What the prediction means operationally
2. Which factors are driving the prediction and why they matter clinically
3. One specific actionable recommendation for care planning

Be direct, clinical, and avoid hedging. No bullet points."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return {"interpretation": response.content[0].text}


@app.post("/interpret/readmission")
def interpret_readmission(payload: Dict[str, Any]):
    prob      = payload.get("readmission_probability_pct", 10.0)
    risk      = payload.get("risk_level", "Moderate")
    shap_vals = payload.get("shap_values", [])
    patient   = payload.get("patient_inputs", {})

    top_features = ", ".join([
        f"{s['feature']} ({'+' if s['value']>0 else ''}{s['value']:.3f})"
        for s in shap_vals[:3]
    ])

    prompt = f"""You are a critical care physician interpreting an ICU readmission risk score.

Patient: {json.dumps(patient, indent=2)}
Readmission probability: {prob:.1f}% (population baseline: 5.58%)
Risk level: {risk}
Top SHAP drivers: {top_features}

Write a 3-sentence clinical interpretation:
1. The readmission risk context relative to baseline
2. The clinical significance of the top driving factors
3. One specific discharge planning recommendation

Be direct and clinical. No bullet points."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return {"interpretation": response.content[0].text}


@app.post("/interpret/disparity")
def interpret_disparity(payload: Dict[str, Any]):
    dimension = payload.get("dimension", "race")
    stats     = payload.get("stats", [])
    oaxaca    = payload.get("oaxaca", {})
    kw        = payload.get("kruskal_wallis", {})

    stats_summary = "\n".join([
        f"  {s['group']}: mean={s['mean']:.1f}h, median={s['median']:.1f}h (n={s['n']:,})"
        for s in stats
    ])

    oaxaca_summary = "\n".join([
        f"  {k}: raw_gap={v.get('raw_gap', v.get('raw', 0)):.3f}, "
        f"{v.get('pct_unexplained', v.get('pct', 0)):.1f}% unexplained"
        for k, v in oaxaca.items()
    ])

    prompt = f"""You are a health equity researcher interpreting ICU cost disparity data for a policy audience.

Dimension: {dimension}
ICU LOS statistics:
{stats_summary}

Kruskal-Wallis test: H={kw.get('H', 'N/A')}, p={kw.get('p', 'N/A')}

Oaxaca decomposition:
{oaxaca_summary}

Write a 3-sentence policy interpretation:
1. The magnitude and significance of the disparity
2. What the Oaxaca decomposition reveals about structural vs clinical drivers
3. One specific policy recommendation

Be direct and policy-relevant. No bullet points."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return {"interpretation": response.content[0].text}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
