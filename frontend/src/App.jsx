import { useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, ReferenceLine
} from "recharts";

const API = "http://localhost:8000";
const post = async (path, body) => {
  const r = await fetch(`${API}${path}`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
};

// ── Static data ───────────────────────────────────────────────────────────
const DISP = {
  race: [
    { group: "White",         mean: 183.7, median: 76, n: 121558 },
    { group: "Black",         mean: 197.6, median: 78, n: 20091  },
    { group: "Hispanic",      mean: 187.8, median: 75, n: 7004   },
    { group: "Asian",         mean: 195.9, median: 76, n: 5612   },
    { group: "Other/Unknown", mean: 235.3, median: 90, n: 29110  },
  ],
  insurance: [
    { group: "Medicare",       mean: 194.2, median: 84, n: 100523 },
    { group: "Medicaid",       mean: 203.3, median: 76, n: 27735  },
    { group: "Private",        mean: 180.6, median: 72, n: 4503   },
    { group: "Self-pay/Other", mean: 189.6, median: 71, n: 50614  },
  ],
  heatmap: {
    rows: ["White","Black","Hispanic","Asian","Other/Unknown"],
    cols: ["Medicare","Medicaid","Private","Self-pay"],
    data: [[81,75,71,69],[84,75,69,72],[85,69,62,72],[84,76,62,66],[96,94,91,79]],
  },
  oaxaca: [
    { label: "White vs Black",       pct: 15.6,  raw: -0.019 },
    { label: "White vs Hispanic",    pct: 157.2, raw:  0.014 },
    { label: "Medicare vs Medicaid", pct: 88.9,  raw:  0.045 },
  ],
  kw: {
    race:      { H: 386.35, p: "2.48×10⁻⁸²"  },
    insurance: { H: 444.70, p: "4.59×10⁻⁹⁶"  },
    sex:       { H: 53.82,  p: "2.20×10⁻¹³"  },
    age:       { H: 478.18, p: "2.56×10⁻¹⁰³" },
  },
};

const PAPERS = [
  {
    id: "cost",
    num: "Paper 1",
    title: "Hospital Cost Prediction",
    subtitle: "ML prediction from admission-time data",
    tag: "XGBoost · Random Forest · SHAP",
    color: "#6366f1",
    badge: "R²=0.636",
    icon: "💰",
    summary: "Trained four ML models (Ridge, Random Forest, XGBoost, LightGBM) on 912,156 MIMIC-IV hospital admissions using 38 features available at admission time. Random Forest achieved a cross-validated R²=0.636 (CI: 0.621–0.650), explaining 64% of length-of-stay variance. SHAP analysis identified DRG type and hemoglobin as the top two predictors — with admission lab values collectively outranking the Charlson Comorbidity Index, a clinically novel finding with direct implications for early-stage cost-flagging system design. Diebold-Mariano tests confirmed Random Forest significantly outperforms XGBoost (p=0.002) and LightGBM (p<0.001).",
    findings: [
      { label: "Best model",            value: "Random Forest"         },
      { label: "CV R² (95% CI)",        value: "0.636 (0.621–0.650)"  },
      { label: "MAE",                   value: "45.8 ± 0.2 hours"     },
      { label: "DM test vs XGBoost",    value: "p = 0.002"            },
      { label: "Top SHAP predictor",    value: "DRG type (0.265)"     },
      { label: "Key insight",           value: "Labs > CCI as predictors" },
    ],
    target: "JAMIA · npj Digital Medicine",
  },
  {
    id: "disparity",
    num: "Paper 2",
    title: "ICU Cost Disparities",
    subtitle: "Multi-dimensional Blinder-Oaxaca analysis",
    tag: "Oaxaca · Log-linear Regression · KW Tests",
    color: "#ef4444",
    badge: "88.9% unexplained",
    icon: "📊",
    summary: "Analysed 183,375 ICU stays to characterise cost disparities across race/ethnicity, insurance type, sex, and age simultaneously. Kruskal-Wallis tests confirmed all disparities are statistically significant (p<0.001 for all four dimensions). The Blinder-Oaxaca decomposition revealed that 88.9% of the Medicare-Medicaid ICU cost gap is unexplained by observable clinical characteristics — the headline finding, which implicates structural payer-driven mechanisms in care delivery that are independent of patient severity. The White-Black cost gap was 84.4% explained by observable clinical factors, suggesting minimal residual racial disparity after severity adjustment.",
    findings: [
      { label: "ICU cohort",             value: "183,375 stays"               },
      { label: "Regression R²",          value: "0.377"                       },
      { label: "Race KW (H / p)",        value: "H=386, p=2.5×10⁻⁸²"        },
      { label: "Insurance KW (H / p)",   value: "H=445, p=4.6×10⁻⁹⁶"        },
      { label: "Medicare vs Medicaid",   value: "88.9% unexplained"           },
      { label: "Key insight",            value: "Payer gap > race gap post-adj" },
    ],
    target: "Health Affairs · JAMA Network Open",
  },
  {
    id: "readmit",
    num: "Paper 3",
    title: "ICU Readmission Prediction",
    subtitle: "Tabular + LSTM hybrid with cost-impact analysis",
    tag: "BiLSTM · Hybrid · Cost-Impact · SHAP",
    color: "#f59e0b",
    badge: "AUPRC=0.138",
    icon: "🫀",
    summary: "Evaluated six model architectures for 72-hour unplanned ICU readmission prediction on 82,585 stays (5.58% rate, 1:17 class imbalance). LightGBM achieved the highest AUPRC (0.138, 2.5× over baseline). A Bidirectional LSTM with attention and a Hybrid BiLSTM+Tabular model were developed — the Hybrid (AUROC=0.605) nearly matched XGBoost, validating 24-hour vital trend features as incremental predictors. SHAP analysis confirmed that respiratory rate trajectory over 24h outranked the discharge-moment measurement. Break-even cost analysis showed that just 3 prevented readmissions/year justify a $50,000 CDS tool — with 20% reduction yielding $434K net annual savings.",
    findings: [
      { label: "Best AUPRC",        value: "LightGBM 0.138 (2.5×)"   },
      { label: "Best AUROC",        value: "Random Forest 0.614"      },
      { label: "Hybrid AUROC",      value: "0.605 at epoch 30"        },
      { label: "Top SHAP feature",  value: "ICU LOS hours"            },
      { label: "Break-even CDS",    value: "3 readmissions/year"      },
      { label: "20% reduction ROI", value: "$434K net savings/yr"     },
    ],
    target: "Critical Care · JAMIA Open",
  },
];

// ── Helpers ───────────────────────────────────────────────────────────────
const heatCell = (v) => {
  const n = Math.max(0, Math.min(1, (v - 60) / 40));
  const r = Math.round(34  + n * 190);
  const g = Math.round(197 - n * 150);
  const b = Math.round(94  - n * 60);
  return `rgb(${r},${g},${b})`;
};

const BAR_COLORS  = ["#6366f1","#ef4444","#f59e0b","#22c55e","#8b5cf6"];
const RISK_COLOR  = { Low: "#22c55e", Moderate: "#f59e0b", High: "#ef4444" };

// ── Shared UI components ──────────────────────────────────────────────────
const inputStyle = {
  width: "100%", boxSizing: "border-box", padding: "9px 12px",
  fontSize: 13, borderRadius: 8, border: "1px solid #e2e8f0",
  background: "#fff", color: "#0f172a", outline: "none", fontFamily: "inherit",
};

const Inp = ({ label, value, onChange, type="number", step=1, min, max }) => (
  <div style={{ marginBottom: 14 }}>
    <div style={{ fontSize: 11, fontWeight: 600, color: "#64748b",
      letterSpacing: "0.06em", textTransform: "uppercase", marginBottom: 5 }}>
      {label}
    </div>
    <input type={type} value={value} step={step} min={min} max={max}
      style={inputStyle}
      onChange={e => onChange(type === "number" ? parseFloat(e.target.value) || 0 : e.target.value)} />
  </div>
);

const Sel = ({ label, value, onChange, opts }) => (
  <div style={{ marginBottom: 14 }}>
    <div style={{ fontSize: 11, fontWeight: 600, color: "#64748b",
      letterSpacing: "0.06em", textTransform: "uppercase", marginBottom: 5 }}>
      {label}
    </div>
    <select value={value} onChange={e => onChange(e.target.value)} style={inputStyle}>
      {opts.map(o => <option key={o}>{o}</option>)}
    </select>
  </div>
);

const PrimaryBtn = ({ onClick, loading, label, color = "#6366f1" }) => (
  <button onClick={onClick} disabled={loading} style={{
    width: "100%", padding: "11px", borderRadius: 8, fontSize: 13, fontWeight: 600,
    background: loading ? "#c7d2fe" : color, color: "#fff", border: "none",
    cursor: loading ? "not-allowed" : "pointer", transition: "opacity 0.15s",
  }}>
    {loading ? "Running…" : label}
  </button>
);

const OutlineBtn = ({ onClick, loading, label, color = "#6366f1" }) => (
  <button onClick={onClick} disabled={loading} style={{
    width: "100%", padding: "11px", borderRadius: 8, fontSize: 13, fontWeight: 600,
    background: "#fff", color, border: `1.5px solid ${color}`,
    cursor: loading ? "not-allowed" : "pointer",
  }}>
    {loading ? "Generating…" : label}
  </button>
);

const Card = ({ children, style = {} }) => (
  <div style={{
    background: "#fff", borderRadius: 12, padding: "20px 22px",
    border: "1px solid #e2e8f0", boxShadow: "0 1px 3px rgba(0,0,0,0.05)", ...style,
  }}>
    {children}
  </div>
);

const SecLabel = ({ children }) => (
  <div style={{ fontSize: 11, fontWeight: 700, color: "#94a3b8",
    letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 14 }}>
    {children}
  </div>
);

const MetricCard = ({ label, value, sub, color = "#6366f1", icon }) => (
  <div style={{
    background: "#fff", border: "1px solid #e2e8f0", borderRadius: 12,
    padding: "16px 18px", display: "flex", alignItems: "center", gap: 12,
    boxShadow: "0 1px 3px rgba(0,0,0,0.05)",
  }}>
    <div style={{
      width: 42, height: 42, borderRadius: 10, flexShrink: 0,
      background: color + "18", display: "flex", alignItems: "center",
      justifyContent: "center", fontSize: 18,
    }}>{icon}</div>
    <div>
      <div style={{ fontSize: 20, fontWeight: 700, color: "#0f172a", lineHeight: 1.1 }}>
        {value}
      </div>
      <div style={{ fontSize: 11, color: "#64748b", marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 10, color: "#94a3b8", marginTop: 1 }}>{sub}</div>}
    </div>
  </div>
);

const InterpBox = ({ text, loading }) => {
  if (!text && !loading) return null;
  return (
    <div style={{ marginTop: 16, padding: "16px 18px", borderRadius: 10,
      background: "#f0f9ff", border: "1px solid #bae6fd" }}>
      <div style={{ fontSize: 11, fontWeight: 700, color: "#0369a1",
        letterSpacing: "0.07em", textTransform: "uppercase", marginBottom: 8,
        display: "flex", alignItems: "center", gap: 6 }}>
        <span style={{ color: "#6366f1" }}>✦</span> Claude AI Interpretation
      </div>
      <div style={{ fontSize: 13, lineHeight: 1.75,
        color: loading ? "#94a3b8" : "#0f172a" }}>
        {loading ? "Generating interpretation…" : text}
      </div>
    </div>
  );
};

const ShapChart = ({ data }) => {
  if (!data?.length) return null;
  const sorted = [...data].sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 8);
  const max = Math.max(...sorted.map(d => Math.abs(d.value)));
  return (
    <div>
      <SecLabel>SHAP Feature Attribution</SecLabel>
      <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
        {sorted.map((d, i) => (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{ width: 144, fontSize: 11, color: "#475569",
              textAlign: "right", flexShrink: 0, fontFamily: "monospace" }}>
              {d.feature}
            </div>
            <div style={{ flex: 1, position: "relative", height: 22,
              borderRadius: 4, background: "#f1f5f9" }}>
              <div style={{
                position: "absolute", height: "100%", borderRadius: 4, opacity: 0.85,
                left: d.value >= 0 ? "50%" : `calc(50% - ${(Math.abs(d.value) / max) * 48}%)`,
                width: `${(Math.abs(d.value) / max) * 48}%`,
                background: d.value > 0 ? "#ef4444" : "#6366f1",
              }} />
              <div style={{ position: "absolute", left: "50%", top: 0,
                bottom: 0, width: 1, background: "#cbd5e1" }} />
            </div>
            <div style={{ width: 58, fontSize: 11, fontFamily: "monospace",
              fontWeight: 600, color: d.value > 0 ? "#ef4444" : "#6366f1", flexShrink: 0 }}>
              {d.value > 0 ? "+" : ""}{d.value.toFixed(3)}
            </div>
          </div>
        ))}
      </div>
      <div style={{ display: "flex", gap: 16, marginTop: 8 }}>
        {[["#ef4444","Increases prediction"],["#6366f1","Decreases prediction"]].map(([c,l]) => (
          <div key={l} style={{ display: "flex", alignItems: "center",
            gap: 5, fontSize: 11, color: "#64748b" }}>
            <div style={{ width: 10, height: 10, borderRadius: 2, background: c }} />{l}
          </div>
        ))}
      </div>
    </div>
  );
};

// ── Paper Detail Modal ────────────────────────────────────────────────────
function PaperModal({ paper, onClose, onOpen }) {
  if (!paper) return null;
  return (
    <div onClick={onClose} style={{
      position: "fixed", inset: 0, background: "rgba(15,23,42,0.55)",
      zIndex: 1000, display: "flex", alignItems: "center",
      justifyContent: "center", padding: 24,
    }}>
      <div onClick={e => e.stopPropagation()} style={{
        background: "#fff", borderRadius: 16, padding: "32px 36px",
        maxWidth: 620, width: "100%",
        boxShadow: "0 24px 64px rgba(0,0,0,0.22)",
        maxHeight: "90vh", overflowY: "auto",
      }}>
        {/* Header */}
        <div style={{ display: "flex", alignItems: "flex-start",
          justifyContent: "space-between", marginBottom: 20 }}>
          <div>
            <div style={{ fontSize: 11, fontWeight: 700, color: paper.color,
              letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 6 }}>
              {paper.num}
            </div>
            <div style={{ fontSize: 20, fontWeight: 800, color: "#0f172a",
              marginBottom: 4 }}>{paper.title}</div>
            <div style={{ fontSize: 13, color: "#64748b" }}>{paper.subtitle}</div>
          </div>
          <button onClick={onClose} style={{ background: "none", border: "none",
            fontSize: 20, color: "#94a3b8", cursor: "pointer",
            padding: "0 4px", lineHeight: 1 }}>✕</button>
        </div>

        {/* Summary */}
        <div style={{ fontSize: 13, lineHeight: 1.8, color: "#475569",
          marginBottom: 22, paddingBottom: 20,
          borderBottom: "1px solid #f1f5f9" }}>
          {paper.summary}
        </div>

        {/* Key findings grid */}
        <div style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: "#94a3b8",
            letterSpacing: "0.08em", textTransform: "uppercase",
            marginBottom: 10 }}>Key findings</div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
            {paper.findings.map(f => (
              <div key={f.label} style={{ background: "#f8fafc", borderRadius: 8,
                padding: "10px 12px" }}>
                <div style={{ fontSize: 10, fontWeight: 600, color: "#94a3b8",
                  textTransform: "uppercase", letterSpacing: "0.06em",
                  marginBottom: 3 }}>{f.label}</div>
                <div style={{ fontSize: 13, fontWeight: 600, color: "#0f172a" }}>
                  {f.value}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Meta row */}
        <div style={{ display: "flex", gap: 10, marginBottom: 20 }}>
          <div style={{ flex: 1, background: paper.color + "10",
            borderRadius: 8, padding: "10px 14px" }}>
            <div style={{ fontSize: 10, color: paper.color, fontWeight: 700,
              textTransform: "uppercase", letterSpacing: "0.06em",
              marginBottom: 3 }}>Methods</div>
            <div style={{ fontSize: 12, color: "#475569" }}>{paper.tag}</div>
          </div>
          <div style={{ flex: 1, background: "#f8fafc", borderRadius: 8,
            padding: "10px 14px" }}>
            <div style={{ fontSize: 10, color: "#94a3b8", fontWeight: 700,
              textTransform: "uppercase", letterSpacing: "0.06em",
              marginBottom: 3 }}>Target journal</div>
            <div style={{ fontSize: 12, color: "#475569" }}>{paper.target}</div>
          </div>
        </div>

        <button onClick={() => { onOpen(paper.id); onClose(); }} style={{
          width: "100%", padding: "12px", borderRadius: 8, fontSize: 13,
          fontWeight: 700, background: paper.color, color: "#fff",
          border: "none", cursor: "pointer", letterSpacing: "0.02em",
        }}>
          Open Interactive Demo →
        </button>
      </div>
    </div>
  );
}

// ── Overview Page ─────────────────────────────────────────────────────────
function OverviewPage({ setActive }) {
  const [hoveredPaper, setHoveredPaper] = useState(null);
  const [modalPaper,   setModalPaper]   = useState(null);

  const globalMetrics = [
    { icon: "🏥", label: "Hospital admissions",    value: "912K",  sub: "MIMIC-IV v3.1",        color: "#6366f1" },
    { icon: "🛏",  label: "ICU stays analysed",    value: "183K",  sub: "Disparity analysis",   color: "#ef4444" },
    { icon: "🫀", label: "Readmission cohort",      value: "82.6K", sub: "5.58% readmit rate",   color: "#f59e0b" },
    { icon: "🤖", label: "ML models trained",        value: "14",    sub: "Across 3 papers",      color: "#22c55e" },
  ];

  const impacts = [
    {
      icon: "💡",
      title: "Clinical decision support",
      desc: "Break-even at just 3 prevented ICU readmissions per year. A 20% readmission reduction yields $434K net annual savings per 500-bed hospital — with strong economic justification for CDS deployment even in resource-constrained settings.",
    },
    {
      icon: "⚖️",
      title: "Health equity & policy",
      desc: "88.9% of the Medicare-Medicaid ICU cost gap is unexplained by clinical severity — a directly actionable finding for CMS value-based care reform. White-Black cost differences are largely explained by observable clinical factors after DRG severity adjustment.",
    },
    {
      icon: "📈",
      title: "Admission-time cost forecasting",
      desc: "Random Forest R²=0.636 using only admission-time data enables prospective bed demand planning and early-cost flagging. Admission lab values outrank the Charlson Comorbidity Index as LOS predictors — a novel, clinically actionable finding.",
    },
    {
      icon: "🧬",
      title: "Temporal vs tabular ML",
      desc: "A Hybrid BiLSTM+Tabular architecture closes the LSTM-to-XGBoost AUROC gap from −5.3% to <1%, validating 24h vital trend features. rr_mean_24h (rank 5) outranked the discharge-moment respiratory rate (rank 19), supporting trend-based clinical monitoring.",
    },
  ];

  return (
    <div>
      {/* Hero / project overview */}
      <div style={{ marginBottom: 32 }}>
        <div style={{
          display: "inline-block", background: "#eef2ff", color: "#6366f1",
          fontSize: 11, fontWeight: 700, letterSpacing: "0.08em",
          textTransform: "uppercase", padding: "5px 12px",
          borderRadius: 6, marginBottom: 14,
        }}>
          MIMIC-IV · Saint Louis University · 2025–2026
        </div>
        <h1 style={{ fontSize: 28, fontWeight: 800, color: "#0f172a",
          margin: "0 0 12px", lineHeight: 1.2 }}>
          Multi-Agent Clinical Analytics Pipeline
        </h1>
        <p style={{ fontSize: 14, color: "#475569", lineHeight: 1.85,
          maxWidth: 760, marginBottom: 20 }}>
          This project is an end-to-end machine learning pipeline built on the MIMIC-IV
          critical care database — one of the largest publicly available EHR datasets,
          covering over 912,000 hospital admissions at Beth Israel Deaconess Medical
          Center. It combines a BigQuery data extraction layer, multi-agent orchestration,
          ensemble ML, deep learning (Bidirectional LSTM), SHAP interpretability, statistical
          disparity analysis, and Claude AI narrative generation into three peer-review-targeted
          research papers covering hospital cost prediction, ICU cost equity, and ICU
          readmission prediction with economic impact modelling. The pipeline is fully
          reproducible, open-source, and designed to run against any MIMIC-IV-compatible
          dataset with minimal configuration.
        </p>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          {[
            "BigQuery · MIMIC-IV v3.1",
            "Random Forest · XGBoost · LightGBM",
            "BiLSTM + Attention · Hybrid",
            "SHAP Interpretability",
            "Blinder-Oaxaca Decomposition",
            "5-fold Cross-Validation",
            "Claude AI Narratives",
          ].map(t => (
            <div key={t} style={{
              padding: "5px 12px", borderRadius: 20,
              background: "#f8fafc", border: "1px solid #e2e8f0",
              fontSize: 12, color: "#475569",
            }}>{t}</div>
          ))}
        </div>
      </div>

      {/* Global metrics */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)",
        gap: 12, marginBottom: 32 }}>
        {globalMetrics.map(m => <MetricCard key={m.label} {...m} />)}
      </div>

      {/* Papers */}
      <div style={{ marginBottom: 12 }}>
        <h2 style={{ fontSize: 17, fontWeight: 800, color: "#0f172a",
          margin: "0 0 4px" }}>Research Papers</h2>
        <p style={{ fontSize: 13, color: "#64748b", marginBottom: 16 }}>
          Click any paper for full methodology, findings, and target journals
        </p>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)",
        gap: 14, marginBottom: 32 }}>
        {PAPERS.map(p => (
          <div key={p.id}
            onClick={() => setModalPaper(p)}
            onMouseEnter={() => setHoveredPaper(p.id)}
            onMouseLeave={() => setHoveredPaper(null)}
            style={{
              background: "#fff", borderRadius: 12, padding: "22px",
              border: `1.5px solid ${hoveredPaper === p.id ? p.color : "#e2e8f0"}`,
              boxShadow: hoveredPaper === p.id
                ? `0 8px 28px ${p.color}22`
                : "0 1px 3px rgba(0,0,0,0.05)",
              cursor: "pointer", transition: "all 0.18s",
              transform: hoveredPaper === p.id ? "translateY(-3px)" : "none",
            }}>
            <div style={{ display: "flex", alignItems: "center",
              justifyContent: "space-between", marginBottom: 14 }}>
              <div style={{ fontSize: 10, fontWeight: 700, color: p.color,
                letterSpacing: "0.08em", textTransform: "uppercase" }}>{p.num}</div>
              <div style={{ padding: "3px 9px", borderRadius: 6,
                background: p.color + "15", color: p.color,
                fontSize: 11, fontWeight: 700 }}>{p.badge}</div>
            </div>
            <div style={{ fontSize: 28, marginBottom: 10 }}>{p.icon}</div>
            <div style={{ fontSize: 15, fontWeight: 700, color: "#0f172a",
              marginBottom: 6 }}>{p.title}</div>
            <div style={{ fontSize: 12, color: "#64748b", lineHeight: 1.55,
              marginBottom: 16 }}>{p.subtitle}</div>
            <div style={{ fontSize: 11, color: "#94a3b8",
              borderTop: "1px solid #f1f5f9", paddingTop: 12,
              display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <span>{p.tag.split(" · ")[0]}</span>
              <span style={{ color: p.color, fontWeight: 600, fontSize: 10 }}>
                View details →
              </span>
            </div>
          </div>
        ))}
      </div>

      {/* Impact */}
      <div style={{ marginBottom: 12 }}>
        <h2 style={{ fontSize: 17, fontWeight: 800, color: "#0f172a",
          margin: "0 0 4px" }}>Research Impact</h2>
        <p style={{ fontSize: 13, color: "#64748b", marginBottom: 16 }}>
          Contributions across clinical decision support, health equity, and ML methodology
        </p>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(2,1fr)",
        gap: 12, marginBottom: 28 }}>
        {impacts.map(item => (
          <Card key={item.title}>
            <div style={{ display: "flex", gap: 14 }}>
              <div style={{ fontSize: 26, flexShrink: 0, marginTop: 1 }}>{item.icon}</div>
              <div>
                <div style={{ fontSize: 14, fontWeight: 700, color: "#0f172a",
                  marginBottom: 6 }}>{item.title}</div>
                <div style={{ fontSize: 13, color: "#475569", lineHeight: 1.7 }}>
                  {item.desc}
                </div>
              </div>
            </div>
          </Card>
        ))}
      </div>

      {/* All models summary table */}
      <Card style={{ marginBottom: 4 }}>
        <SecLabel>All Model Results</SecLabel>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
          <thead>
            <tr style={{ borderBottom: "1px solid #f1f5f9" }}>
              {["Paper","Model","Primary Metric","Value","n"].map(h => (
                <th key={h} style={{ padding: "8px 12px", textAlign: "left",
                  fontSize: 11, fontWeight: 700, color: "#94a3b8",
                  letterSpacing: "0.06em", textTransform: "uppercase" }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {[
              ["Paper 1","Random Forest","CV R²","0.636 (0.621–0.650)","912K admissions"],
              ["Paper 1","XGBoost","R²","0.475","912K admissions"],
              ["Paper 1","LightGBM","R²","0.478","912K admissions"],
              ["Paper 2","Log-linear Regression","R²","0.377","183K ICU stays"],
              ["Paper 2","Blinder-Oaxaca","% Unexplained","88.9% (Medicare vs Medicaid)","183K ICU stays"],
              ["Paper 3","LightGBM","AUPRC","0.138 (2.5× baseline)","82.6K ICU stays"],
              ["Paper 3","Random Forest","AUROC","0.614","82.6K ICU stays"],
              ["Paper 3","Hybrid BiLSTM","AUROC","0.605 (epoch 30)","82.6K ICU stays"],
            ].map((row, i) => (
              <tr key={i} style={{
                borderBottom: "1px solid #f8fafc",
                background: i % 2 === 0 ? "#fff" : "#fafbfc",
              }}>
                {row.map((cell, j) => (
                  <td key={j} style={{
                    padding: "9px 12px", color: "#0f172a",
                    fontWeight: j === 3 ? 600 : 400,
                  }}>{cell}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Paper modal */}
      <PaperModal
        paper={modalPaper}
        onClose={() => setModalPaper(null)}
        onOpen={id => { setActive(id); }}
      />
    </div>
  );
}

// ── Cost Page ─────────────────────────────────────────────────────────────
function CostPage() {
  const [f, sf] = useState({
    age: 65, gender: "M", insurance: "Medicare", admission_type: "URGENT",
    drg_severity: 2, cci_total_score: 2, hemoglobin: 11.0, creatinine: 1.2,
    sodium: 138, procedure_count: 3, drg_mortality: 2, wbc: 8.5,
    potassium: 4.0, lactate: 1.5, race: "WHITE",
    admission_location: "EMERGENCY ROOM", discharge_location: "HOME", drg_type: "HCFA",
  });
  const [res,    setRes]    = useState(null);
  const [interp, setInterp] = useState("");
  const [load,   setLoad]   = useState(false);
  const [iload,  setIload]  = useState(false);
  const set = (k, v) => sf(p => ({ ...p, [k]: v }));

  const run = async () => {
    setLoad(true); setRes(null); setInterp("");
    try { setRes(await post("/predict/cost", f)); }
    catch (e) { alert("Backend error: " + e.message); }
    setLoad(false);
  };
  const interpret = async () => {
    if (!res) return;
    setIload(true);
    try { setInterp((await post("/interpret/cost", { ...res, patient_inputs: f })).interpretation); }
    catch (e) { alert(e.message); }
    setIload(false);
  };

  return (
    <div>
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 24, fontWeight: 800, color: "#0f172a", margin: "0 0 4px" }}>
          Cost Prediction
        </h1>
        <p style={{ fontSize: 13, color: "#64748b" }}>
          Paper 1 · Random Forest CV R²=0.636 · 912,156 admissions ·
          Diebold-Mariano p=0.002 vs XGBoost
        </p>
      </div>

      {res && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)",
          gap: 12, marginBottom: 20 }}>
          <MetricCard icon="🗓" label="Predicted stay"
            value={`${res.los_days.toFixed(1)}d`} sub={`${res.los_hours.toFixed(0)} hours`}
            color="#6366f1" />
          <MetricCard icon="💵" label="Estimated cost"
            value={`$${(res.cost_estimate_usd / 1000).toFixed(1)}K`}
            sub="DRG-adjusted estimate" color="#ef4444" />
          <MetricCard icon="📏" label="95% CI"
            value={`${res.ci_low_hours.toFixed(0)}–${res.ci_high_hours.toFixed(0)}h`}
            sub="Bootstrap B=1,000" color="#f59e0b" />
          <MetricCard icon="✓" label="Model CV R²"
            value="0.636" sub="5-fold, seed=42" color="#22c55e" />
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "300px 1fr",
        gap: 16, alignItems: "start" }}>
        {/* Left: inputs */}
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <Card>
            <SecLabel>Patient Admission Data</SecLabel>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 12px" }}>
              <Inp label="Age" value={f.age} onChange={v => set("age", v)} min={18} max={100} />
              <Sel label="Gender" value={f.gender} onChange={v => set("gender", v)} opts={["M","F"]} />
              <Sel label="Insurance" value={f.insurance} onChange={v => set("insurance", v)}
                opts={["Medicare","Medicaid","Private","Self-pay/Other"]} />
              <Sel label="Admission type" value={f.admission_type}
                onChange={v => set("admission_type", v)}
                opts={["URGENT","EW EMER.","DIRECT EMER.","ELECTIVE","OBSERVATION ADMIT"]} />
              <Inp label="DRG severity (1–4)" value={f.drg_severity}
                onChange={v => set("drg_severity", v)} min={1} max={4} />
              <Inp label="CCI score" value={f.cci_total_score}
                onChange={v => set("cci_total_score", v)} min={0} max={15} />
            </div>
          </Card>
          <Card>
            <SecLabel>Admission Labs</SecLabel>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 12px" }}>
              <Inp label="Hemoglobin (g/dL)" value={f.hemoglobin}
                onChange={v => set("hemoglobin", v)} step={0.1} />
              <Inp label="Creatinine (mg/dL)" value={f.creatinine}
                onChange={v => set("creatinine", v)} step={0.1} />
              <Inp label="Sodium (mEq/L)" value={f.sodium}
                onChange={v => set("sodium", v)} />
              <Inp label="Procedure count" value={f.procedure_count}
                onChange={v => set("procedure_count", v)} />
            </div>
          </Card>
          <PrimaryBtn onClick={run} loading={load}
            label="Predict Length of Stay & Cost" />
        </div>

        {/* Right: results */}
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          {res ? (
            <>
              <Card><ShapChart data={res.shap_values} /></Card>
              <Card>
                <OutlineBtn onClick={interpret} loading={iload}
                  label="Generate Clinical Interpretation" />
                <InterpBox text={interp} loading={iload} />
              </Card>
            </>
          ) : (
            <Card style={{ display: "flex", flexDirection: "column",
              alignItems: "center", justifyContent: "center",
              minHeight: 340, color: "#94a3b8", gap: 10 }}>
              <div style={{ fontSize: 48 }}>💰</div>
              <div style={{ fontSize: 14, fontWeight: 500 }}>
                Enter admission data and run prediction
              </div>
              <div style={{ fontSize: 12 }}>
                SHAP attribution and AI interpretation will appear here
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Readmission Page ──────────────────────────────────────────────────────
function ReadPage() {
  const [f, sf] = useState({
    age: 65, gender: "M", race: "WHITE", insurance: "Medicare",
    first_careunit: "Medical Intensive Care Unit (MICU)",
    icu_los_hours: 72, heart_rate_last: 88, spo2_last: 95,
    resp_rate_last: 20, abp_mean_last: 76, temp_f_last: 98.6,
    total_output_ml: 1200, output_event_count: 8,
    hr_mean_24h: 88, hr_std_24h: 12, hr_delta_24h: -5,
    spo2_mean_24h: 95, spo2_std_24h: 2, spo2_delta_24h: 0.5,
    rr_mean_24h: 20, rr_std_24h: 3, rr_delta_24h: 1,
    abp_mean_24h: 76, abp_std_24h: 8, abp_delta_24h: -2,
    temp_mean_24h: 98.4, temp_std_24h: 0.4, temp_delta_24h: 0.2,
  });
  const [res,    setRes]    = useState(null);
  const [interp, setInterp] = useState("");
  const [load,   setLoad]   = useState(false);
  const [iload,  setIload]  = useState(false);
  const set = (k, v) => sf(p => ({ ...p, [k]: v }));

  const UNITS = [
    "Medical Intensive Care Unit (MICU)",
    "Cardiac Vascular Intensive Care Unit (CVICU)",
    "Medical/Surgical Intensive Care Unit (MICU/SICU)",
    "Surgical Intensive Care Unit (SICU)",
    "Trauma SICU (TSICU)", "Coronary Care Unit (CCU)",
    "Neuro Surgical ICU (Neuro SICU)",
  ];

  const run = async () => {
    setLoad(true); setRes(null); setInterp("");
    try { setRes(await post("/predict/readmission", f)); }
    catch (e) { alert("Backend error: " + e.message); }
    setLoad(false);
  };
  const interpret = async () => {
    if (!res) return;
    setIload(true);
    try { setInterp((await post("/interpret/readmission", { ...res, patient_inputs: f })).interpretation); }
    catch (e) { alert(e.message); }
    setIload(false);
  };

  return (
    <div>
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 24, fontWeight: 800, color: "#0f172a", margin: "0 0 4px" }}>
          Readmission Risk
        </h1>
        <p style={{ fontSize: 13, color: "#64748b" }}>
          Paper 3 · XGBoost AUROC=0.610 · LightGBM AUPRC=0.138 (2.5× baseline) ·
          Hybrid BiLSTM AUROC=0.605 · n=82,585 ICU stays
        </p>
      </div>

      {res && (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)",
            gap: 12, marginBottom: 14 }}>
            <div style={{
              background: "#fff", border: `2px solid ${RISK_COLOR[res.risk_level]}`,
              borderRadius: 12, padding: "16px 18px",
              display: "flex", alignItems: "center", gap: 12,
            }}>
              <div style={{ width: 42, height: 42, borderRadius: 10,
                background: RISK_COLOR[res.risk_level] + "20",
                display: "flex", alignItems: "center",
                justifyContent: "center", fontSize: 18 }}>🫀</div>
              <div>
                <div style={{ fontSize: 20, fontWeight: 700,
                  color: RISK_COLOR[res.risk_level] }}>
                  {res.readmission_probability_pct.toFixed(1)}%
                </div>
                <div style={{ fontSize: 11, color: "#64748b" }}>
                  {res.risk_level} Risk
                </div>
              </div>
            </div>
            <MetricCard icon="📊" label="vs. population baseline"
              value={`${(res.readmission_probability_pct / 5.58).toFixed(1)}×`}
              sub="Baseline = 5.58%" color="#f59e0b" />
            <MetricCard icon="💵" label="Expected cost impact"
              value={`$${(res.expected_cost_impact_usd / 1000).toFixed(1)}K`}
              sub="Avg $22K per readmission" color="#ef4444" />
            <MetricCard icon="✓" label="XGBoost AUROC"
              value="0.610" sub="AUPRC = 0.130" color="#6366f1" />
          </div>

          <Card style={{ marginBottom: 14 }}>
            <div style={{ display: "flex", justifyContent: "space-between",
              fontSize: 11, color: "#94a3b8", marginBottom: 8 }}>
              <span>0%</span><span>Baseline 5.6%</span>
              <span>20%</span><span>30%+</span>
            </div>
            <div style={{ position: "relative", height: 10, borderRadius: 5,
              background: "#f1f5f9", overflow: "hidden" }}>
              <div style={{
                position: "absolute", left: 0, top: 0, height: "100%", borderRadius: 5,
                width: `${Math.min(res.readmission_probability_pct * 3.33, 100)}%`,
                background: "linear-gradient(90deg,#22c55e 0%,#f59e0b 40%,#ef4444 80%)",
                transition: "width 0.8s ease",
              }} />
              <div style={{ position: "absolute", left: "18.6%", top: 0,
                bottom: 0, width: 2, background: "#fff" }} />
            </div>
          </Card>
        </>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "300px 1fr",
        gap: 16, alignItems: "start" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <Card>
            <SecLabel>ICU Stay Details</SecLabel>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 12px" }}>
              <Inp label="Age" value={f.age} onChange={v => set("age", v)} min={18} max={100} />
              <Sel label="Gender" value={f.gender} onChange={v => set("gender", v)} opts={["M","F"]} />
            </div>
            <Sel label="Care unit" value={f.first_careunit}
              onChange={v => set("first_careunit", v)} opts={UNITS} />
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 12px" }}>
              <Inp label="ICU LOS (hours)" value={f.icu_los_hours}
                onChange={v => set("icu_los_hours", v)} />
              <Inp label="Total output (mL)" value={f.total_output_ml}
                onChange={v => set("total_output_ml", v)} step={100} />
            </div>
          </Card>
          <Card>
            <SecLabel>Discharge Vitals</SecLabel>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 12px" }}>
              <Inp label="Heart rate" value={f.heart_rate_last}
                onChange={v => set("heart_rate_last", v)} />
              <Inp label="SpO₂ %" value={f.spo2_last}
                onChange={v => set("spo2_last", v)} step={0.5} />
              <Inp label="Resp rate" value={f.resp_rate_last}
                onChange={v => set("resp_rate_last", v)} />
              <Inp label="MAP mmHg" value={f.abp_mean_last}
                onChange={v => set("abp_mean_last", v)} />
            </div>
          </Card>
          <Card>
            <SecLabel>24h Vital Trends</SecLabel>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 12px" }}>
              <Inp label="RR mean 24h" value={f.rr_mean_24h}
                onChange={v => set("rr_mean_24h", v)} step={0.5} />
              <Inp label="SpO₂ SD 24h" value={f.spo2_std_24h}
                onChange={v => set("spo2_std_24h", v)} step={0.1} />
              <Inp label="HR mean 24h" value={f.hr_mean_24h}
                onChange={v => set("hr_mean_24h", v)} />
              <Inp label="HR delta 24h" value={f.hr_delta_24h}
                onChange={v => set("hr_delta_24h", v)} />
            </div>
          </Card>
          <PrimaryBtn onClick={run} loading={load}
            label="Score Readmission Risk" color="#ef4444" />
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          {res ? (
            <>
              <Card><ShapChart data={res.shap_values} /></Card>
              <Card>
                <OutlineBtn onClick={interpret} loading={iload}
                  label="Generate Clinical Interpretation" color="#ef4444" />
                <InterpBox text={interp} loading={iload} />
              </Card>
            </>
          ) : (
            <Card style={{ display: "flex", flexDirection: "column",
              alignItems: "center", justifyContent: "center",
              minHeight: 380, color: "#94a3b8", gap: 10 }}>
              <div style={{ fontSize: 48 }}>🫀</div>
              <div style={{ fontSize: 14, fontWeight: 500 }}>
                Enter ICU discharge data and score risk
              </div>
              <div style={{ fontSize: 12 }}>
                Risk gauge, SHAP chart, and interpretation will appear here
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Disparity Page ────────────────────────────────────────────────────────
function DisparityPage() {
  const [dim,    setDim]    = useState("race");
  const [interp, setInterp] = useState("");
  const [iload,  setIload]  = useState(false);

  const data                     = DISP[dim] || DISP.race;
  const kw                       = DISP.kw[dim] || {};
  const { rows, cols, data: hdata } = DISP.heatmap;

  const interpret = async () => {
    setIload(true);
    try {
      const payload = {
        dimension: dim,
        stats: data,
        oaxaca: DISP.oaxaca.reduce((acc, o) => ({ ...acc, [o.label]: o }), {}),
        kruskal_wallis: { H: String(kw.H ?? ""), p: String(kw.p ?? "") },
      };
      setInterp((await post("/interpret/disparity", payload)).interpretation);
    } catch (e) { alert("Interpretation error: " + e.message); }
    setIload(false);
  };

  return (
    <div>
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 24, fontWeight: 800, color: "#0f172a", margin: "0 0 4px" }}>
          ICU Cost Disparities
        </h1>
        <p style={{ fontSize: 13, color: "#64748b" }}>
          Paper 2 · Blinder-Oaxaca decomposition · n=183,375 ICU stays ·
          All Kruskal-Wallis tests p&lt;0.001
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)",
        gap: 12, marginBottom: 20 }}>
        <MetricCard icon="📊" label="Race KW statistic"    value="H=386"  sub="p=2.48×10⁻⁸²"  color="#6366f1" />
        <MetricCard icon="📊" label="Insurance KW"         value="H=445"  sub="p=4.59×10⁻⁹⁶"  color="#ef4444" />
        <MetricCard icon="⚠️" label="Medicare–Medicaid gap" value="88.9%" sub="Unexplained"     color="#f59e0b" />
        <MetricCard icon="✓"  label="Regression R²"        value="0.377"  sub="N=183,375 stays" color="#22c55e" />
      </div>

      {/* Dimension tabs */}
      <div style={{ display: "flex", gap: 8, marginBottom: 16, alignItems: "center" }}>
        {["race","insurance","sex","age"].map(d => (
          <button key={d} onClick={() => setDim(d)} style={{
            padding: "8px 18px", borderRadius: 8, fontSize: 13, fontWeight: 600,
            cursor: "pointer", textTransform: "capitalize",
            background: dim === d ? "#6366f1" : "#fff",
            color: dim === d ? "#fff" : "#475569",
            border: dim === d ? "1px solid #6366f1" : "1px solid #e2e8f0",
            transition: "all 0.15s",
          }}>{d}</button>
        ))}
        <div style={{ marginLeft: "auto", fontSize: 12, color: "#64748b",
          background: "#f8fafc", border: "1px solid #e2e8f0",
          borderRadius: 8, padding: "7px 14px", fontFamily: "monospace" }}>
          H = {kw.H?.toFixed(2)} · p = {kw.p}
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr",
        gap: 16, marginBottom: 16 }}>
        {/* Bar chart */}
        <Card>
          <SecLabel>Mean ICU LOS by {dim} (hours)</SecLabel>
          <ResponsiveContainer width="100%" height={230}>
            <BarChart data={data} margin={{ left: -10, right: 10, bottom: 24 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
              <XAxis dataKey="group" tick={{ fill: "#94a3b8", fontSize: 11 }}
                axisLine={false} tickLine={false} angle={-15} textAnchor="end" />
              <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }}
                axisLine={false} tickLine={false} domain={[150, 260]} />
              <Tooltip contentStyle={{ background: "#fff",
                border: "1px solid #e2e8f0", borderRadius: 8, fontSize: 12 }}
                formatter={(v, _, p) =>
                  [`${v.toFixed(1)}h (n=${p.payload.n?.toLocaleString()})`, "Mean LOS"]} />
              <ReferenceLine y={183.7} stroke="#e2e8f0" strokeDasharray="4 3"
                label={{ value: "White ref", position: "right",
                  fontSize: 9, fill: "#94a3b8" }} />
              <Bar dataKey="mean" radius={[6,6,0,0]} maxBarSize={56}>
                {data.map((_, i) => (
                  <Cell key={i} fill={BAR_COLORS[i % BAR_COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>

        {/* Oaxaca */}
        <Card>
          <SecLabel>Blinder-Oaxaca — % gap unexplained by clinical factors</SecLabel>
          <div style={{ display: "flex", flexDirection: "column", gap: 20, marginTop: 8 }}>
            {DISP.oaxaca.map((o, i) => (
              <div key={i}>
                <div style={{ display: "flex", justifyContent: "space-between",
                  fontSize: 13, marginBottom: 8 }}>
                  <span style={{ color: "#0f172a", fontWeight: 500 }}>{o.label}</span>
                  <span style={{ fontWeight: 700,
                    color: o.pct > 50 ? "#ef4444" : "#22c55e" }}>
                    {o.pct.toFixed(1)}% unexplained
                  </span>
                </div>
                <div style={{ position: "relative", height: 12, borderRadius: 6,
                  background: "#f1f5f9" }}>
                  <div style={{
                    height: "100%", borderRadius: 6, transition: "width 0.6s ease",
                    width: `${Math.min(Math.abs(o.pct), 100)}%`,
                    background: o.pct > 50 ? "#ef4444" : "#22c55e",
                    opacity: 0.8,
                  }} />
                </div>
                <div style={{ fontSize: 11, color: "#94a3b8", marginTop: 4,
                  fontFamily: "monospace" }}>
                  raw gap {o.raw > 0 ? "+" : ""}{o.raw.toFixed(3)} log-LOS
                </div>
              </div>
            ))}
          </div>
          <div style={{ marginTop: 16, padding: "12px 14px", borderRadius: 8,
            background: "#fef2f2", border: "1px solid #fecaca",
            fontSize: 12, color: "#991b1b" }}>
            <strong>Headline finding:</strong> The Medicare vs Medicaid gap is 88.9%
            unexplained by clinical characteristics — implicating structural
            payer-driven disparities independent of patient severity.
          </div>
        </Card>
      </div>

      {/* Heatmap */}
      <Card style={{ marginBottom: 16 }}>
        <SecLabel>Median ICU cost proxy — Race × Insurance cross-tabulation</SecLabel>
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse",
            tableLayout: "fixed" }}>
            <thead>
              <tr style={{ borderBottom: "1px solid #f1f5f9" }}>
                <th style={{ width: 160, padding: "8px 14px", fontSize: 11,
                  fontWeight: 600, color: "#94a3b8", textAlign: "left",
                  letterSpacing: "0.05em", textTransform: "uppercase" }}>
                  Race / Ethnicity
                </th>
                {cols.map(c => (
                  <th key={c} style={{ padding: "8px 12px", fontSize: 11,
                    fontWeight: 600, color: "#94a3b8", textAlign: "center",
                    letterSpacing: "0.05em", textTransform: "uppercase" }}>{c}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, ri) => (
                <tr key={row} style={{ borderBottom: "1px solid #f8fafc" }}>
                  <td style={{ padding: "10px 14px", fontSize: 13,
                    fontWeight: 500, color: "#0f172a" }}>{row}</td>
                  {hdata[ri].map((val, ci) => (
                    <td key={ci} style={{
                      padding: "10px 12px", textAlign: "center",
                      fontSize: 14, fontWeight: 700,
                      background: heatCell(val), color: "#fff",
                    }}>{val}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <div style={{ display: "flex", alignItems: "center", gap: 8,
            marginTop: 10, fontSize: 11, color: "#94a3b8" }}>
            <span>Low</span>
            <div style={{ flex: 1, height: 6, borderRadius: 3,
              background: "linear-gradient(90deg,#22c55e,#ef4444)" }} />
            <span>High (hours × severity)</span>
          </div>
        </div>
      </Card>

      {/* Interpretation */}
      <Card>
        <OutlineBtn onClick={interpret} loading={iload}
          label={`Generate Policy Interpretation — ${dim.charAt(0).toUpperCase() + dim.slice(1)} Dimension`} />
        <InterpBox text={interp} loading={iload} />
      </Card>
    </div>
  );
}

// ── Sidebar ───────────────────────────────────────────────────────────────
const NAV = [
  { id: "overview",  icon: "⊞", label: "Overview"         },
  { id: "cost",      icon: "💰", label: "Cost Prediction"  },
  { id: "readmit",   icon: "🫀", label: "Readmission Risk" },
  { id: "disparity", icon: "📊", label: "Disparities"      },
];

function Sidebar({ active, setActive }) {
  return (
    <div style={{
      width: 220, background: "#0f172a", display: "flex",
      flexDirection: "column", flexShrink: 0,
      position: "sticky", top: 0, height: "100vh", overflowY: "auto",
    }}>
      <div style={{ padding: "26px 22px 18px", borderBottom: "1px solid #1e293b" }}>
        <div style={{ fontSize: 18, fontWeight: 800, color: "#6366f1",
          letterSpacing: "-0.5px" }}>ClinicalAI</div>
        <div style={{ fontSize: 10, color: "#475569", marginTop: 3,
          letterSpacing: "0.05em", textTransform: "uppercase" }}>
          MIMIC-IV Analytics
        </div>
      </div>

      <nav style={{ flex: 1, padding: "14px 10px" }}>
        {NAV.map(n => (
          <button key={n.id} onClick={() => setActive(n.id)} style={{
            width: "100%", display: "flex", alignItems: "center", gap: 10,
            padding: "10px 12px", borderRadius: 8, marginBottom: 2,
            background: active === n.id ? "#6366f1" : "transparent",
            color: active === n.id ? "#fff" : "#64748b",
            border: "none", cursor: "pointer", fontSize: 13,
            textAlign: "left", fontWeight: active === n.id ? 600 : 400,
            transition: "all 0.15s",
          }}>
            <span style={{ fontSize: 15, width: 20, textAlign: "center" }}>
              {n.icon}
            </span>
            {n.label}
          </button>
        ))}

        <div style={{ margin: "20px 0 8px 12px", fontSize: 10, color: "#334155",
          fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase" }}>
          Model Results
        </div>
        {[
          ["RF",     "R²=0.636",   "#22c55e"],
          ["XGB",    "AUPRC=0.130","#6366f1"],
          ["LGB",    "AUPRC=0.138","#f59e0b"],
          ["BiLSTM", "AUROC=0.585","#8b5cf6"],
          ["Hybrid", "AUROC=0.605","#06b6d4"],
        ].map(([name, val, color]) => (
          <div key={name} style={{ display: "flex", justifyContent: "space-between",
            alignItems: "center", padding: "5px 12px", fontSize: 11 }}>
            <span style={{ color: "#64748b" }}>{name}</span>
            <span style={{ color, fontFamily: "monospace", fontWeight: 600,
              fontSize: 10 }}>{val}</span>
          </div>
        ))}
      </nav>

      <div style={{ padding: "14px 18px", borderTop: "1px solid #1e293b" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ width: 32, height: 32, borderRadius: "50%",
            background: "#6366f1", display: "flex", alignItems: "center",
            justifyContent: "center", fontSize: 12, fontWeight: 700,
            color: "#fff", flexShrink: 0 }}>A</div>
          <div>
            <div style={{ fontSize: 12, fontWeight: 600, color: "#e2e8f0" }}>
              Abubakar
            </div>
            <div style={{ fontSize: 10, color: "#475569" }}>SLU · Health Analytics</div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── App Shell ─────────────────────────────────────────────────────────────
const PAGES = {
  overview:  OverviewPage,
  cost:      CostPage,
  readmit:   ReadPage,
  disparity: DisparityPage,
};

export default function App() {
  const [active, setActive] = useState("overview");
  const Page = PAGES[active] || OverviewPage;

  return (
    <div style={{
      display: "flex", width: "100%", minHeight: "100vh",
      background: "#f8fafc",
      fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    }}>
      <Sidebar active={active} setActive={setActive} />
      <main style={{
        flex: 1, minWidth: 0, padding: "32px 36px",
        overflowY: "auto", height: "100vh",
      }}>
        <Page setActive={setActive} />
      </main>
    </div>
  );
}
