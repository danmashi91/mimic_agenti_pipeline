import { useState, useEffect, useRef } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, ReferenceLine
} from "recharts";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const post = async (path, body) => {
  const r = await fetch(`${API_URL}${path}`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
};

// ── Responsive hook ───────────────────────────────────────────────────────
const useIsMobile = () => {
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);
  useEffect(() => {
    const handler = () => setIsMobile(window.innerWidth < 768);
    window.addEventListener("resize", handler);
    return () => window.removeEventListener("resize", handler);
  }, []);
  return isMobile;
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
    id: "cost", num: "Paper 1", title: "Hospital Cost Prediction",
    subtitle: "ML prediction from admission-time data",
    tag: "XGBoost · Random Forest · SHAP", color: "#6366f1", badge: "R²=0.636", icon: "💰",
    summary: "Trained four ML models on 912,156 MIMIC-IV admissions using 38 admission-time features. Random Forest achieved CV R²=0.636 (CI: 0.621–0.650). SHAP analysis identified DRG type and hemoglobin as top predictors — admission labs outranked the Charlson Comorbidity Index.",
    findings: [
      { label: "Best model", value: "Random Forest" },
      { label: "CV R² (95% CI)", value: "0.636 (0.621–0.650)" },
      { label: "MAE", value: "45.8 ± 0.2 hours" },
      { label: "DM test vs XGBoost", value: "p = 0.002" },
      { label: "Top SHAP predictor", value: "DRG type (0.265)" },
      { label: "Key insight", value: "Labs > CCI as predictors" },
    ],
    target: "JAMIA · npj Digital Medicine",
  },
  {
    id: "disparity", num: "Paper 2", title: "ICU Cost Disparities",
    subtitle: "Multi-dimensional Blinder-Oaxaca analysis",
    tag: "Oaxaca · Regression · KW Tests", color: "#ef4444", badge: "88.9% unexplained", icon: "📊",
    summary: "Analysed 183,375 ICU stays across race/ethnicity, insurance, sex, and age. KW tests confirmed all disparities significant (p<0.001). Blinder-Oaxaca decomposition revealed 88.9% of the Medicare-Medicaid gap is unexplained by clinical characteristics.",
    findings: [
      { label: "ICU cohort", value: "183,375 stays" },
      { label: "Regression R²", value: "0.377" },
      { label: "Race KW (H / p)", value: "H=386, p=2.5×10⁻⁸²" },
      { label: "Insurance KW (H / p)", value: "H=445, p=4.6×10⁻⁹⁶" },
      { label: "Medicare vs Medicaid", value: "88.9% unexplained" },
      { label: "Key insight", value: "Payer gap > race gap post-adj" },
    ],
    target: "Health Affairs · JAMA Network Open",
  },
  {
    id: "readmit", num: "Paper 3", title: "ICU Readmission Prediction",
    subtitle: "Tabular + LSTM hybrid with cost-impact analysis",
    tag: "BiLSTM · Hybrid · Cost-Impact", color: "#f59e0b", badge: "AUPRC=0.138", icon: "🫀",
    summary: "Evaluated 6 model architectures for 72-hour ICU readmission prediction on 82,585 stays (5.58% rate). LightGBM AUPRC=0.138 (2.5× baseline). Hybrid BiLSTM+Tabular AUROC=0.605. Break-even: 3 prevented readmissions/year justifies $50K CDS tool.",
    findings: [
      { label: "Best AUPRC", value: "LightGBM 0.138 (2.5×)" },
      { label: "Best AUROC", value: "Random Forest 0.614" },
      { label: "Hybrid AUROC", value: "0.605 at epoch 30" },
      { label: "Top SHAP feature", value: "ICU LOS hours" },
      { label: "Break-even CDS", value: "3 readmissions/year" },
      { label: "20% reduction ROI", value: "$434K net savings/yr" },
    ],
    target: "Critical Care · JAMIA Open",
  },
];

const heatCell = (v) => {
  const n = Math.max(0, Math.min(1, (v - 60) / 40));
  const r = Math.round(34  + n * 190);
  const g = Math.round(197 - n * 150);
  const b = Math.round(94  - n * 60);
  return `rgb(${r},${g},${b})`;
};

const BAR_COLORS  = ["#6366f1","#ef4444","#f59e0b","#22c55e","#8b5cf6"];
const RISK_COLOR  = { Low: "#22c55e", Moderate: "#f59e0b", High: "#ef4444" };

// ── Shared UI ──────────────────────────────────────────────────────────────
const inputStyle = {
  width: "100%", boxSizing: "border-box", padding: "10px 12px",
  fontSize: 14, borderRadius: 8, border: "1px solid #e2e8f0",
  background: "#fff", color: "#0f172a", outline: "none", fontFamily: "inherit",
};

const Inp = ({ label, value, onChange, type="number", step=1, min, max }) => (
  <div style={{ marginBottom: 14 }}>
    <div style={{ fontSize: 11, fontWeight: 600, color: "#64748b",
      letterSpacing: "0.06em", textTransform: "uppercase", marginBottom: 5 }}>{label}</div>
    <input type={type} value={value} step={step} min={min} max={max}
      style={inputStyle}
      onChange={e => onChange(type === "number" ? parseFloat(e.target.value) || 0 : e.target.value)} />
  </div>
);

const Sel = ({ label, value, onChange, opts }) => (
  <div style={{ marginBottom: 14 }}>
    <div style={{ fontSize: 11, fontWeight: 600, color: "#64748b",
      letterSpacing: "0.06em", textTransform: "uppercase", marginBottom: 5 }}>{label}</div>
    <select value={value} onChange={e => onChange(e.target.value)} style={inputStyle}>
      {opts.map(o => <option key={o}>{o}</option>)}
    </select>
  </div>
);

const PrimaryBtn = ({ onClick, loading, label, color = "#6366f1" }) => (
  <button onClick={onClick} disabled={loading} style={{
    width: "100%", padding: "13px", borderRadius: 8, fontSize: 14, fontWeight: 600,
    background: loading ? "#c7d2fe" : color, color: "#fff", border: "none",
    cursor: loading ? "not-allowed" : "pointer",
  }}>{loading ? "Running…" : label}</button>
);

const OutlineBtn = ({ onClick, loading, label, color = "#6366f1" }) => (
  <button onClick={onClick} disabled={loading} style={{
    width: "100%", padding: "13px", borderRadius: 8, fontSize: 14, fontWeight: 600,
    background: "#fff", color, border: `1.5px solid ${color}`,
    cursor: loading ? "not-allowed" : "pointer",
  }}>{loading ? "Generating…" : label}</button>
);

const Card = ({ children, style = {} }) => (
  <div style={{
    background: "#fff", borderRadius: 12, padding: "16px 18px",
    border: "1px solid #e2e8f0", boxShadow: "0 1px 3px rgba(0,0,0,0.05)", ...style,
  }}>{children}</div>
);

const SecLabel = ({ children }) => (
  <div style={{ fontSize: 11, fontWeight: 700, color: "#94a3b8",
    letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 12 }}>
    {children}
  </div>
);

const MetricCard = ({ label, value, sub, color = "#6366f1", icon }) => (
  <div style={{
    background: "#fff", border: "1px solid #e2e8f0", borderRadius: 12,
    padding: "14px 16px", display: "flex", alignItems: "center", gap: 12,
    boxShadow: "0 1px 3px rgba(0,0,0,0.05)", minWidth: 140,
  }}>
    <div style={{ width: 40, height: 40, borderRadius: 10, flexShrink: 0,
      background: color + "18", display: "flex", alignItems: "center",
      justifyContent: "center", fontSize: 18 }}>{icon}</div>
    <div style={{ minWidth: 0 }}>
      <div style={{ fontSize: 18, fontWeight: 700, color: "#0f172a",
        lineHeight: 1.1, whiteSpace: "nowrap" }}>{value}</div>
      <div style={{ fontSize: 11, color: "#64748b", marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 10, color: "#94a3b8", marginTop: 1 }}>{sub}</div>}
    </div>
  </div>
);

const MetricsRow = ({ metrics }) => (
  <div style={{ display: "flex", gap: 10, overflowX: "auto",
    paddingBottom: 4, marginBottom: 16, WebkitOverflowScrolling: "touch" }}>
    {metrics.map(m => <MetricCard key={m.label} {...m} />)}
  </div>
);

const InterpBox = ({ text, loading }) => {
  if (!text && !loading) return null;
  return (
    <div style={{ marginTop: 16, padding: "14px 16px", borderRadius: 10,
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
  const sorted = [...data].sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 7);
  const max = Math.max(...sorted.map(d => Math.abs(d.value)));
  return (
    <div>
      <SecLabel>SHAP Feature Attribution</SecLabel>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {sorted.map((d, i) => (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div style={{ width: 120, fontSize: 10, color: "#475569",
              textAlign: "right", flexShrink: 0, fontFamily: "monospace",
              overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {d.feature}
            </div>
            <div style={{ flex: 1, position: "relative", height: 20,
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
            <div style={{ width: 48, fontSize: 10, fontFamily: "monospace",
              fontWeight: 600, color: d.value > 0 ? "#ef4444" : "#6366f1",
              flexShrink: 0, textAlign: "right" }}>
              {d.value > 0 ? "+" : ""}{d.value.toFixed(3)}
            </div>
          </div>
        ))}
      </div>
      <div style={{ display: "flex", gap: 12, marginTop: 8 }}>
        {[["#ef4444","Increases"],["#6366f1","Decreases"]].map(([c,l]) => (
          <div key={l} style={{ display: "flex", alignItems: "center",
            gap: 4, fontSize: 11, color: "#64748b" }}>
            <div style={{ width: 8, height: 8, borderRadius: 2, background: c }} />{l}
          </div>
        ))}
      </div>
    </div>
  );
};

// ── Collapsible section for mobile ────────────────────────────────────────
const Collapsible = ({ title, children, defaultOpen = true }) => {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <Card style={{ padding: 0, overflow: "hidden" }}>
      <button onClick={() => setOpen(!open)} style={{
        width: "100%", padding: "14px 18px", background: "none", border: "none",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        cursor: "pointer", fontSize: 13, fontWeight: 600, color: "#0f172a",
      }}>
        {title}
        <span style={{ color: "#94a3b8", fontSize: 16,
          transform: open ? "rotate(180deg)" : "none",
          transition: "transform 0.2s" }}>▾</span>
      </button>
      {open && <div style={{ padding: "0 18px 16px" }}>{children}</div>}
    </Card>
  );
};

// ── Paper Modal ────────────────────────────────────────────────────────────
function PaperModal({ paper, onClose, onOpen }) {
  if (!paper) return null;
  return (
    <div onClick={onClose} style={{
      position: "fixed", inset: 0, background: "rgba(15,23,42,0.55)",
      zIndex: 1000, display: "flex", alignItems: "flex-end",
      justifyContent: "center", padding: "0",
    }}>
      <div onClick={e => e.stopPropagation()} style={{
        background: "#fff", borderRadius: "16px 16px 0 0", padding: "24px 20px",
        width: "100%", maxWidth: 600,
        boxShadow: "0 -8px 40px rgba(0,0,0,0.15)",
        maxHeight: "85vh", overflowY: "auto",
      }}>
        <div style={{ width: 40, height: 4, background: "#e2e8f0",
          borderRadius: 2, margin: "0 auto 20px" }} />
        <div style={{ display: "flex", alignItems: "flex-start",
          justifyContent: "space-between", marginBottom: 16 }}>
          <div>
            <div style={{ fontSize: 11, fontWeight: 700, color: paper.color,
              letterSpacing: "0.08em", textTransform: "uppercase",
              marginBottom: 4 }}>{paper.num}</div>
            <div style={{ fontSize: 18, fontWeight: 800, color: "#0f172a",
              marginBottom: 2 }}>{paper.title}</div>
            <div style={{ fontSize: 13, color: "#64748b" }}>{paper.subtitle}</div>
          </div>
          <button onClick={onClose} style={{ background: "none", border: "none",
            fontSize: 20, color: "#94a3b8", cursor: "pointer" }}>✕</button>
        </div>

        <p style={{ fontSize: 13, lineHeight: 1.75, color: "#475569",
          marginBottom: 18, paddingBottom: 16,
          borderBottom: "1px solid #f1f5f9" }}>{paper.summary}</p>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr",
          gap: 8, marginBottom: 16 }}>
          {paper.findings.map(f => (
            <div key={f.label} style={{ background: "#f8fafc",
              borderRadius: 8, padding: "10px 12px" }}>
              <div style={{ fontSize: 10, fontWeight: 600, color: "#94a3b8",
                textTransform: "uppercase", letterSpacing: "0.06em",
                marginBottom: 2 }}>{f.label}</div>
              <div style={{ fontSize: 12, fontWeight: 600, color: "#0f172a" }}>
                {f.value}
              </div>
            </div>
          ))}
        </div>

        <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
          <div style={{ flex: 1, background: paper.color + "10",
            borderRadius: 8, padding: "10px 12px" }}>
            <div style={{ fontSize: 10, color: paper.color, fontWeight: 700,
              textTransform: "uppercase", letterSpacing: "0.06em",
              marginBottom: 2 }}>Methods</div>
            <div style={{ fontSize: 11, color: "#475569" }}>{paper.tag}</div>
          </div>
          <div style={{ flex: 1, background: "#f8fafc", borderRadius: 8,
            padding: "10px 12px" }}>
            <div style={{ fontSize: 10, color: "#94a3b8", fontWeight: 700,
              textTransform: "uppercase", letterSpacing: "0.06em",
              marginBottom: 2 }}>Target journal</div>
            <div style={{ fontSize: 11, color: "#475569" }}>{paper.target}</div>
          </div>
        </div>

        <button onClick={() => { onOpen(paper.id); onClose(); }} style={{
          width: "100%", padding: "13px", borderRadius: 8, fontSize: 14,
          fontWeight: 700, background: paper.color, color: "#fff",
          border: "none", cursor: "pointer",
        }}>Open Interactive Demo →</button>
      </div>
    </div>
  );
}

// ── Overview ───────────────────────────────────────────────────────────────
function OverviewPage({ setActive }) {
  const isMobile = useIsMobile();
  const [hoveredPaper, setHoveredPaper] = useState(null);
  const [modalPaper, setModalPaper] = useState(null);

  const globalMetrics = [
    { icon: "🏥", label: "Admissions", value: "912K", sub: "MIMIC-IV v3.1", color: "#6366f1" },
    { icon: "🛏", label: "ICU stays", value: "183K", sub: "Disparity analysis", color: "#ef4444" },
    { icon: "🫀", label: "Readmission", value: "82.6K", sub: "5.58% rate", color: "#f59e0b" },
    { icon: "🤖", label: "ML models", value: "14", sub: "Across 3 papers", color: "#22c55e" },
  ];

  const impacts = [
    { icon: "💡", title: "Clinical decision support", desc: "Break-even at 3 prevented ICU readmissions/year. 20% reduction = $434K net annual savings per 500-bed hospital." },
    { icon: "⚖️", title: "Health equity policy", desc: "88.9% of the Medicare-Medicaid ICU cost gap is unexplained by clinical severity — a directly actionable policy finding for CMS reform." },
    { icon: "📈", title: "Admission-time forecasting", desc: "Random Forest R²=0.636 using only admission-time data enables prospective bed demand planning and early cost flagging." },
    { icon: "🧬", title: "Temporal vs tabular ML", desc: "Hybrid BiLSTM+Tabular closes the LSTM-to-XGBoost gap to <1% AUROC. 24h vital trends outrank discharge-moment snapshots." },
  ];

  return (
    <div>
      <div style={{ marginBottom: 24 }}>
        <div style={{ display: "inline-block", background: "#eef2ff", color: "#6366f1",
          fontSize: 11, fontWeight: 700, letterSpacing: "0.08em",
          textTransform: "uppercase", padding: "5px 12px",
          borderRadius: 6, marginBottom: 12 }}>
          MIMIC-IV · Saint Louis University · 2025–2026
        </div>
        <h1 style={{ fontSize: isMobile ? 22 : 26, fontWeight: 800,
          color: "#0f172a", margin: "0 0 10px", lineHeight: 1.2 }}>
          Multi-Agent Clinical Analytics Pipeline
        </h1>
        <p style={{ fontSize: 13, color: "#475569", lineHeight: 1.8,
          marginBottom: 16 }}>
          An end-to-end ML pipeline built on 912K+ MIMIC-IV hospital admissions —
          combining BigQuery extraction, ensemble ML, BiLSTM deep learning,
          SHAP interpretability, statistical disparity analysis, and Claude AI
          narrative generation across three research papers.
        </p>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          {["BigQuery · MIMIC-IV","Random Forest · XGBoost","BiLSTM + Hybrid",
            "SHAP","Blinder-Oaxaca","Claude AI"].map(t => (
            <div key={t} style={{ padding: "4px 10px", borderRadius: 20,
              background: "#f8fafc", border: "1px solid #e2e8f0",
              fontSize: 11, color: "#475569" }}>{t}</div>
          ))}
        </div>
      </div>

      <MetricsRow metrics={globalMetrics} />

      <div style={{ marginBottom: 10 }}>
        <h2 style={{ fontSize: 15, fontWeight: 700, color: "#0f172a",
          margin: "0 0 4px" }}>Research Papers</h2>
        <p style={{ fontSize: 12, color: "#64748b", marginBottom: 14 }}>
          Tap any paper for full methodology and findings
        </p>
      </div>

      <div style={{ display: "grid",
        gridTemplateColumns: isMobile ? "1fr" : "repeat(3,1fr)",
        gap: 12, marginBottom: 24 }}>
        {PAPERS.map(p => (
          <div key={p.id}
            onClick={() => setModalPaper(p)}
            style={{
              background: "#fff", borderRadius: 12, padding: "18px",
              border: `1.5px solid ${hoveredPaper === p.id ? p.color : "#e2e8f0"}`,
              boxShadow: "0 1px 3px rgba(0,0,0,0.05)",
              cursor: "pointer", transition: "all 0.18s",
              display: "flex", flexDirection: isMobile ? "row" : "column",
              alignItems: isMobile ? "center" : "flex-start",
              gap: isMobile ? 14 : 0,
            }}
            onMouseEnter={() => setHoveredPaper(p.id)}
            onMouseLeave={() => setHoveredPaper(null)}>
            {isMobile ? (
              <>
                <div style={{ fontSize: 28, flexShrink: 0 }}>{p.icon}</div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 10, fontWeight: 700, color: p.color,
                    letterSpacing: "0.08em", textTransform: "uppercase",
                    marginBottom: 2 }}>{p.num}</div>
                  <div style={{ fontSize: 14, fontWeight: 700, color: "#0f172a",
                    marginBottom: 2 }}>{p.title}</div>
                  <div style={{ fontSize: 12, color: "#64748b" }}>{p.subtitle}</div>
                </div>
                <div style={{ padding: "4px 8px", borderRadius: 6,
                  background: p.color + "15", color: p.color,
                  fontSize: 11, fontWeight: 700, flexShrink: 0 }}>{p.badge}</div>
              </>
            ) : (
              <>
                <div style={{ display: "flex", alignItems: "center",
                  justifyContent: "space-between", width: "100%", marginBottom: 12 }}>
                  <div style={{ fontSize: 10, fontWeight: 700, color: p.color,
                    letterSpacing: "0.08em", textTransform: "uppercase" }}>{p.num}</div>
                  <div style={{ padding: "3px 9px", borderRadius: 6,
                    background: p.color + "15", color: p.color,
                    fontSize: 11, fontWeight: 700 }}>{p.badge}</div>
                </div>
                <div style={{ fontSize: 26, marginBottom: 10 }}>{p.icon}</div>
                <div style={{ fontSize: 14, fontWeight: 700, color: "#0f172a",
                  marginBottom: 4 }}>{p.title}</div>
                <div style={{ fontSize: 12, color: "#64748b", lineHeight: 1.5,
                  marginBottom: 14 }}>{p.subtitle}</div>
                <div style={{ fontSize: 11, color: "#94a3b8",
                  borderTop: "1px solid #f1f5f9", paddingTop: 10,
                  width: "100%", display: "flex", justifyContent: "space-between" }}>
                  <span>{p.tag.split(" · ")[0]}</span>
                  <span style={{ color: p.color, fontWeight: 600 }}>View →</span>
                </div>
              </>
            )}
          </div>
        ))}
      </div>

      <div style={{ marginBottom: 10 }}>
        <h2 style={{ fontSize: 15, fontWeight: 700, color: "#0f172a",
          margin: "0 0 4px" }}>Research Impact</h2>
      </div>
      <div style={{ display: "grid",
        gridTemplateColumns: isMobile ? "1fr" : "repeat(2,1fr)",
        gap: 10, marginBottom: 24 }}>
        {impacts.map(item => (
          <Card key={item.title}>
            <div style={{ display: "flex", gap: 12 }}>
              <div style={{ fontSize: 22, flexShrink: 0 }}>{item.icon}</div>
              <div>
                <div style={{ fontSize: 13, fontWeight: 700, color: "#0f172a",
                  marginBottom: 4 }}>{item.title}</div>
                <div style={{ fontSize: 12, color: "#475569", lineHeight: 1.65 }}>
                  {item.desc}
                </div>
              </div>
            </div>
          </Card>
        ))}
      </div>

      <Card>
        <SecLabel>All Model Results</SecLabel>
        <div style={{ overflowX: "auto", WebkitOverflowScrolling: "touch" }}>
          <table style={{ width: "100%", borderCollapse: "collapse",
            fontSize: 12, minWidth: 480 }}>
            <thead>
              <tr style={{ borderBottom: "1px solid #f1f5f9" }}>
                {["Paper","Model","Metric","Value","n"].map(h => (
                  <th key={h} style={{ padding: "8px 10px", textAlign: "left",
                    fontSize: 10, fontWeight: 700, color: "#94a3b8",
                    letterSpacing: "0.06em", textTransform: "uppercase",
                    whiteSpace: "nowrap" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[
                ["P1","Random Forest","CV R²","0.636 (0.621–0.650)","912K"],
                ["P1","XGBoost","R²","0.475","912K"],
                ["P2","Regression","R²","0.377","183K"],
                ["P2","Oaxaca","% Unexpl.","88.9% (Medicare vs Medicaid)","183K"],
                ["P3","LightGBM","AUPRC","0.138 (2.5× baseline)","82.6K"],
                ["P3","Random Forest","AUROC","0.614","82.6K"],
                ["P3","Hybrid BiLSTM","AUROC","0.605","82.6K"],
              ].map((row, i) => (
                <tr key={i} style={{ borderBottom: "1px solid #f8fafc",
                  background: i % 2 === 0 ? "#fff" : "#fafbfc" }}>
                  {row.map((cell, j) => (
                    <td key={j} style={{ padding: "8px 10px", color: "#0f172a",
                      fontWeight: j === 3 ? 600 : 400,
                      whiteSpace: j === 3 ? "normal" : "nowrap" }}>{cell}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <PaperModal paper={modalPaper} onClose={() => setModalPaper(null)}
        onOpen={id => { setActive(id); }} />
    </div>
  );
}

// ── Cost Page ──────────────────────────────────────────────────────────────
function CostPage() {
  const isMobile = useIsMobile();
  const [f, sf] = useState({
    age: 65, gender: "M", insurance: "Medicare", admission_type: "URGENT",
    drg_severity: 2, cci_total_score: 2, hemoglobin: 11.0, creatinine: 1.2,
    sodium: 138, procedure_count: 3, drg_mortality: 2, wbc: 8.5,
    potassium: 4.0, lactate: 1.5, race: "WHITE",
    admission_location: "EMERGENCY ROOM", discharge_location: "HOME", drg_type: "HCFA",
  });
  const [res, setRes] = useState(null);
  const [interp, setInterp] = useState("");
  const [load, setLoad] = useState(false);
  const [iload, setIload] = useState(false);
  const set = (k, v) => sf(p => ({ ...p, [k]: v }));
  const resultsRef = useRef(null);

  const run = async () => {
    setLoad(true); setRes(null); setInterp("");
    try {
      const r = await post("/predict/cost", f);
      setRes(r);
      if (isMobile && resultsRef.current) {
        setTimeout(() => resultsRef.current.scrollIntoView({ behavior: "smooth" }), 100);
      }
    } catch(e) { alert("Backend error: " + e.message); }
    setLoad(false);
  };
  const interpret = async () => {
    if (!res) return;
    setIload(true);
    try { setInterp((await post("/interpret/cost", {...res, patient_inputs: f})).interpretation); }
    catch(e) { alert(e.message); }
    setIload(false);
  };

  const metrics = res ? [
    { icon: "🗓", label: "Predicted stay", value: `${res.los_days.toFixed(1)}d`, sub: `${res.los_hours.toFixed(0)}h`, color: "#6366f1" },
    { icon: "💵", label: "Est. cost", value: `$${(res.cost_estimate_usd/1000).toFixed(1)}K`, sub: "DRG-adjusted", color: "#ef4444" },
    { icon: "📏", label: "95% CI", value: `${res.ci_low_hours.toFixed(0)}–${res.ci_high_hours.toFixed(0)}h`, sub: "Bootstrap", color: "#f59e0b" },
    { icon: "✓", label: "CV R²", value: "0.636", sub: "5-fold", color: "#22c55e" },
  ] : [];

  return (
    <div>
      <div style={{ marginBottom: 20 }}>
        <h1 style={{ fontSize: isMobile ? 20 : 24, fontWeight: 800,
          color: "#0f172a", margin: "0 0 4px" }}>Cost Prediction</h1>
        <p style={{ fontSize: 12, color: "#64748b" }}>
          Paper 1 · Random Forest CV R²=0.636 · 912,156 admissions
        </p>
      </div>

      {res && <MetricsRow metrics={metrics} />}

      <div style={{ display: "grid",
        gridTemplateColumns: isMobile ? "1fr" : "300px 1fr",
        gap: 14, alignItems: "start" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <Collapsible title="Patient Admission Data">
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 12px" }}>
              <Inp label="Age" value={f.age} onChange={v=>set("age",v)} min={18} max={100} />
              <Sel label="Gender" value={f.gender} onChange={v=>set("gender",v)} opts={["M","F"]} />
              <Sel label="Insurance" value={f.insurance} onChange={v=>set("insurance",v)}
                opts={["Medicare","Medicaid","Private","Self-pay/Other"]} />
              <Sel label="Admission type" value={f.admission_type}
                onChange={v=>set("admission_type",v)}
                opts={["URGENT","EW EMER.","DIRECT EMER.","ELECTIVE","OBSERVATION ADMIT"]} />
              <Inp label="DRG severity (1–4)" value={f.drg_severity}
                onChange={v=>set("drg_severity",v)} min={1} max={4} />
              <Inp label="CCI score" value={f.cci_total_score}
                onChange={v=>set("cci_total_score",v)} min={0} max={15} />
            </div>
          </Collapsible>
          <Collapsible title="Admission Labs" defaultOpen={!isMobile}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 12px" }}>
              <Inp label="Hemoglobin (g/dL)" value={f.hemoglobin}
                onChange={v=>set("hemoglobin",v)} step={0.1} />
              <Inp label="Creatinine (mg/dL)" value={f.creatinine}
                onChange={v=>set("creatinine",v)} step={0.1} />
              <Inp label="Sodium (mEq/L)" value={f.sodium} onChange={v=>set("sodium",v)} />
              <Inp label="Procedures" value={f.procedure_count}
                onChange={v=>set("procedure_count",v)} />
            </div>
          </Collapsible>
          <PrimaryBtn onClick={run} loading={load} label="Predict Length of Stay & Cost" />
        </div>

        <div ref={resultsRef} style={{ display: "flex", flexDirection: "column", gap: 12 }}>
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
              minHeight: 200, color: "#94a3b8", gap: 8 }}>
              <div style={{ fontSize: 40 }}>💰</div>
              <div style={{ fontSize: 13, fontWeight: 500 }}>Enter admission data and predict</div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Readmission Page ───────────────────────────────────────────────────────
function ReadPage() {
  const isMobile = useIsMobile();
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
  const [res, setRes] = useState(null);
  const [interp, setInterp] = useState("");
  const [load, setLoad] = useState(false);
  const [iload, setIload] = useState(false);
  const set = (k, v) => sf(p => ({ ...p, [k]: v }));
  const resultsRef = useRef(null);

  const UNITS = ["Medical Intensive Care Unit (MICU)",
    "Cardiac Vascular Intensive Care Unit (CVICU)",
    "Medical/Surgical Intensive Care Unit (MICU/SICU)",
    "Surgical Intensive Care Unit (SICU)", "Trauma SICU (TSICU)",
    "Coronary Care Unit (CCU)", "Neuro Surgical ICU (Neuro SICU)"];

  const run = async () => {
    setLoad(true); setRes(null); setInterp("");
    try {
      const r = await post("/predict/readmission", f);
      setRes(r);
      if (isMobile && resultsRef.current) {
        setTimeout(() => resultsRef.current.scrollIntoView({ behavior: "smooth" }), 100);
      }
    } catch(e) { alert("Backend error: " + e.message); }
    setLoad(false);
  };
  const interpret = async () => {
    if (!res) return;
    setIload(true);
    try { setInterp((await post("/interpret/readmission", {...res, patient_inputs: f})).interpretation); }
    catch(e) { alert(e.message); }
    setIload(false);
  };

  const metrics = res ? [
    { icon: "🫀", label: `${res.risk_level} Risk`, value: `${res.readmission_probability_pct.toFixed(1)}%`, color: RISK_COLOR[res.risk_level] },
    { icon: "📊", label: "vs baseline", value: `${(res.readmission_probability_pct/5.58).toFixed(1)}×`, sub: "5.58% baseline", color: "#f59e0b" },
    { icon: "💵", label: "Cost impact", value: `$${(res.expected_cost_impact_usd/1000).toFixed(1)}K`, color: "#ef4444" },
    { icon: "✓", label: "AUROC", value: "0.610", sub: "XGBoost", color: "#6366f1" },
  ] : [];

  return (
    <div>
      <div style={{ marginBottom: 20 }}>
        <h1 style={{ fontSize: isMobile ? 20 : 24, fontWeight: 800,
          color: "#0f172a", margin: "0 0 4px" }}>Readmission Risk</h1>
        <p style={{ fontSize: 12, color: "#64748b" }}>
          Paper 3 · LightGBM AUPRC=0.138 · Hybrid BiLSTM AUROC=0.605
        </p>
      </div>

      {res && (
        <>
          <MetricsRow metrics={metrics} />
          <Card style={{ marginBottom: 12 }}>
            <div style={{ display: "flex", justifyContent: "space-between",
              fontSize: 10, color: "#94a3b8", marginBottom: 6 }}>
              <span>0%</span><span>Baseline 5.6%</span><span>20%</span><span>30%+</span>
            </div>
            <div style={{ position: "relative", height: 10, borderRadius: 5,
              background: "#f1f5f9", overflow: "hidden" }}>
              <div style={{
                position: "absolute", left: 0, top: 0, height: "100%", borderRadius: 5,
                width: `${Math.min(res.readmission_probability_pct*3.33,100)}%`,
                background: "linear-gradient(90deg,#22c55e 0%,#f59e0b 40%,#ef4444 80%)",
              }} />
              <div style={{ position: "absolute", left: "18.6%", top: 0,
                bottom: 0, width: 2, background: "#fff" }} />
            </div>
          </Card>
        </>
      )}

      <div style={{ display: "grid",
        gridTemplateColumns: isMobile ? "1fr" : "300px 1fr",
        gap: 14, alignItems: "start" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <Collapsible title="ICU Stay Details">
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 12px" }}>
              <Inp label="Age" value={f.age} onChange={v=>set("age",v)} min={18} max={100} />
              <Sel label="Gender" value={f.gender} onChange={v=>set("gender",v)} opts={["M","F"]} />
            </div>
            <Sel label="Care unit" value={f.first_careunit}
              onChange={v=>set("first_careunit",v)} opts={UNITS} />
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 12px" }}>
              <Inp label="ICU LOS (hrs)" value={f.icu_los_hours} onChange={v=>set("icu_los_hours",v)} />
              <Inp label="Output (mL)" value={f.total_output_ml} onChange={v=>set("total_output_ml",v)} step={100} />
            </div>
          </Collapsible>
          <Collapsible title="Discharge Vitals" defaultOpen={!isMobile}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 12px" }}>
              <Inp label="Heart rate" value={f.heart_rate_last} onChange={v=>set("heart_rate_last",v)} />
              <Inp label="SpO₂ %" value={f.spo2_last} onChange={v=>set("spo2_last",v)} step={0.5} />
              <Inp label="Resp rate" value={f.resp_rate_last} onChange={v=>set("resp_rate_last",v)} />
              <Inp label="MAP mmHg" value={f.abp_mean_last} onChange={v=>set("abp_mean_last",v)} />
            </div>
          </Collapsible>
          <Collapsible title="24h Vital Trends" defaultOpen={!isMobile}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 12px" }}>
              <Inp label="RR mean 24h" value={f.rr_mean_24h} onChange={v=>set("rr_mean_24h",v)} step={0.5} />
              <Inp label="SpO₂ SD 24h" value={f.spo2_std_24h} onChange={v=>set("spo2_std_24h",v)} step={0.1} />
              <Inp label="HR mean 24h" value={f.hr_mean_24h} onChange={v=>set("hr_mean_24h",v)} />
              <Inp label="HR delta 24h" value={f.hr_delta_24h} onChange={v=>set("hr_delta_24h",v)} />
            </div>
          </Collapsible>
          <PrimaryBtn onClick={run} loading={load} label="Score Readmission Risk" color="#ef4444" />
        </div>

        <div ref={resultsRef} style={{ display: "flex", flexDirection: "column", gap: 12 }}>
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
              minHeight: 200, color: "#94a3b8", gap: 8 }}>
              <div style={{ fontSize: 40 }}>🫀</div>
              <div style={{ fontSize: 13, fontWeight: 500 }}>Enter discharge data and score risk</div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Disparity Page ─────────────────────────────────────────────────────────
function DisparityPage() {
  const isMobile = useIsMobile();
  const [dim, setDim] = useState("race");
  const [interp, setInterp] = useState("");
  const [iload, setIload] = useState(false);

  const data = DISP[dim] || DISP.race;
  const kw = DISP.kw[dim] || {};
  const { rows, cols, data: hdata } = DISP.heatmap;

  const interpret = async () => {
    setIload(true);
    try {
      const payload = {
        dimension: dim, stats: data,
        oaxaca: DISP.oaxaca.reduce((acc, o) => ({ ...acc, [o.label]: o }), {}),
        kruskal_wallis: { H: String(kw.H ?? ""), p: String(kw.p ?? "") },
      };
      setInterp((await post("/interpret/disparity", payload)).interpretation);
    } catch(e) { alert("Interpretation error: " + e.message); }
    setIload(false);
  };

  const metrics = [
    { icon: "📊", label: "Race KW", value: "H=386", sub: "p=2.48×10⁻⁸²", color: "#6366f1" },
    { icon: "📊", label: "Insurance KW", value: "H=445", sub: "p=4.59×10⁻⁹⁶", color: "#ef4444" },
    { icon: "⚠️", label: "Unexplained gap", value: "88.9%", sub: "Medicare vs Medicaid", color: "#f59e0b" },
    { icon: "✓", label: "Regression R²", value: "0.377", sub: "N=183,375", color: "#22c55e" },
  ];

  return (
    <div>
      <div style={{ marginBottom: 20 }}>
        <h1 style={{ fontSize: isMobile ? 20 : 24, fontWeight: 800,
          color: "#0f172a", margin: "0 0 4px" }}>ICU Cost Disparities</h1>
        <p style={{ fontSize: 12, color: "#64748b" }}>
          Paper 2 · Blinder-Oaxaca · n=183,375 ICU stays · All KW p&lt;0.001
        </p>
      </div>

      <MetricsRow metrics={metrics} />

      <div style={{ display: "flex", gap: 8, marginBottom: 14,
        overflowX: "auto", WebkitOverflowScrolling: "touch", paddingBottom: 4 }}>
        {["race","insurance","sex","age"].map(d => (
          <button key={d} onClick={() => setDim(d)} style={{
            padding: "8px 16px", borderRadius: 8, fontSize: 13, fontWeight: 600,
            cursor: "pointer", textTransform: "capitalize", flexShrink: 0,
            background: dim===d ? "#6366f1" : "#fff",
            color: dim===d ? "#fff" : "#475569",
            border: dim===d ? "1px solid #6366f1" : "1px solid #e2e8f0",
          }}>{d}</button>
        ))}
      </div>

      <div style={{ display: "grid",
        gridTemplateColumns: isMobile ? "1fr" : "1fr 1fr",
        gap: 12, marginBottom: 12 }}>
        <Card>
          <SecLabel>Mean ICU LOS by {dim} (hours)</SecLabel>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={data} margin={{ left: -10, right: 8, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
              <XAxis dataKey="group" tick={{ fill: "#94a3b8", fontSize: 10 }}
                axisLine={false} tickLine={false} angle={-15} textAnchor="end" />
              <YAxis tick={{ fill: "#94a3b8", fontSize: 10 }}
                axisLine={false} tickLine={false} domain={[150, 260]} />
              <Tooltip contentStyle={{ background: "#fff",
                border: "1px solid #e2e8f0", borderRadius: 8, fontSize: 11 }}
                formatter={(v,_,p) => [`${v.toFixed(1)}h (n=${p.payload.n?.toLocaleString()})`, "Mean LOS"]} />
              <ReferenceLine y={183.7} stroke="#e2e8f0" strokeDasharray="4 3" />
              <Bar dataKey="mean" radius={[6,6,0,0]} maxBarSize={52}>
                {data.map((_, i) => <Cell key={i} fill={BAR_COLORS[i % BAR_COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card>
          <SecLabel>Blinder-Oaxaca — % gap unexplained</SecLabel>
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {DISP.oaxaca.map((o, i) => (
              <div key={i}>
                <div style={{ display: "flex", justifyContent: "space-between",
                  fontSize: 12, marginBottom: 6 }}>
                  <span style={{ color: "#0f172a", fontWeight: 500,
                    fontSize: 11 }}>{o.label}</span>
                  <span style={{ fontWeight: 700, fontSize: 12,
                    color: o.pct > 50 ? "#ef4444" : "#22c55e" }}>
                    {o.pct.toFixed(1)}%
                  </span>
                </div>
                <div style={{ position: "relative", height: 10,
                  borderRadius: 6, background: "#f1f5f9" }}>
                  <div style={{ height: "100%", borderRadius: 6,
                    width: `${Math.min(Math.abs(o.pct), 100)}%`,
                    background: o.pct > 50 ? "#ef4444" : "#22c55e",
                    opacity: 0.8 }} />
                </div>
                <div style={{ fontSize: 10, color: "#94a3b8", marginTop: 3,
                  fontFamily: "monospace" }}>
                  raw gap {o.raw > 0 ? "+" : ""}{o.raw.toFixed(3)} log-LOS
                </div>
              </div>
            ))}
          </div>
          <div style={{ marginTop: 12, padding: "10px 12px", borderRadius: 8,
            background: "#fef2f2", border: "1px solid #fecaca",
            fontSize: 11, color: "#991b1b" }}>
            <strong>Headline:</strong> Medicare vs Medicaid gap is 88.9% unexplained
            by clinical characteristics.
          </div>
        </Card>
      </div>

      <Card style={{ marginBottom: 12 }}>
        <SecLabel>Median ICU cost proxy — Race × Insurance heatmap</SecLabel>
        <div style={{ overflowX: "auto", WebkitOverflowScrolling: "touch" }}>
          <table style={{ borderCollapse: "collapse", minWidth: 360 }}>
            <thead>
              <tr style={{ borderBottom: "1px solid #f1f5f9" }}>
                <th style={{ width: 120, padding: "6px 10px", fontSize: 10,
                  fontWeight: 600, color: "#94a3b8", textAlign: "left",
                  textTransform: "uppercase", letterSpacing: "0.05em" }}>Race</th>
                {cols.map(c => (
                  <th key={c} style={{ padding: "6px 8px", fontSize: 10,
                    fontWeight: 600, color: "#94a3b8", textAlign: "center",
                    textTransform: "uppercase", letterSpacing: "0.05em",
                    whiteSpace: "nowrap" }}>{c}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, ri) => (
                <tr key={row} style={{ borderBottom: "1px solid #f8fafc" }}>
                  <td style={{ padding: "8px 10px", fontSize: 12,
                    fontWeight: 500, color: "#0f172a", whiteSpace: "nowrap" }}>
                    {row}
                  </td>
                  {hdata[ri].map((val, ci) => (
                    <td key={ci} style={{ padding: "8px", textAlign: "center",
                      fontSize: 13, fontWeight: 700,
                      background: heatCell(val), color: "#fff" }}>
                      {val}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <div style={{ display: "flex", alignItems: "center", gap: 8,
            marginTop: 10, fontSize: 10, color: "#94a3b8" }}>
            <span>Low</span>
            <div style={{ flex: 1, height: 5, borderRadius: 3,
              background: "linear-gradient(90deg,#22c55e,#ef4444)" }} />
            <span>High (hrs × severity)</span>
          </div>
        </div>
      </Card>

      <Card>
        <OutlineBtn onClick={interpret} loading={iload}
          label={`Generate Policy Interpretation — ${dim.charAt(0).toUpperCase()+dim.slice(1)}`} />
        <InterpBox text={interp} loading={iload} />
      </Card>
    </div>
  );
}

// ── Nav items ──────────────────────────────────────────────────────────────
const NAV = [
  { id: "overview",  icon: "⊞", label: "Overview"         },
  { id: "cost",      icon: "💰", label: "Cost Prediction"  },
  { id: "readmit",   icon: "🫀", label: "Readmission Risk" },
  { id: "disparity", icon: "📊", label: "Disparities"      },
];

// ── App Shell ──────────────────────────────────────────────────────────────
const PAGES = { overview: OverviewPage, cost: CostPage, readmit: ReadPage, disparity: DisparityPage };

export default function App() {
  const [active, setActive] = useState("overview");
  const [menuOpen, setMenuOpen] = useState(false);
  const isMobile = useIsMobile();
  const Page = PAGES[active] || OverviewPage;

  const navigate = (id) => {
    setActive(id);
    setMenuOpen(false);
  };

  return (
    <div style={{ display: "flex", width: "100%", minHeight: "100vh",
      background: "#f8fafc",
      fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" }}>

      {/* Desktop sidebar */}
      {!isMobile && (
        <div style={{ width: 220, background: "#0f172a", display: "flex",
          flexDirection: "column", flexShrink: 0,
          position: "sticky", top: 0, height: "100vh", overflowY: "auto" }}>
          <div style={{ padding: "24px 22px 16px", borderBottom: "1px solid #1e293b" }}>
            <div style={{ fontSize: 18, fontWeight: 800, color: "#6366f1",
              letterSpacing: "-0.5px" }}>ClinicalAI</div>
            <div style={{ fontSize: 10, color: "#475569", marginTop: 3,
              letterSpacing: "0.05em", textTransform: "uppercase" }}>MIMIC-IV Analytics</div>
          </div>
          <nav style={{ flex: 1, padding: "14px 10px" }}>
            {NAV.map(n => (
              <button key={n.id} onClick={() => navigate(n.id)} style={{
                width: "100%", display: "flex", alignItems: "center", gap: 10,
                padding: "10px 12px", borderRadius: 8, marginBottom: 2,
                background: active===n.id ? "#6366f1" : "transparent",
                color: active===n.id ? "#fff" : "#64748b",
                border: "none", cursor: "pointer", fontSize: 13,
                textAlign: "left", fontWeight: active===n.id ? 600 : 400,
              }}>
                <span style={{ fontSize: 15, width: 20, textAlign: "center" }}>{n.icon}</span>
                {n.label}
              </button>
            ))}
            <div style={{ margin: "20px 0 8px 12px", fontSize: 10, color: "#334155",
              fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase" }}>
              Model Results
            </div>
            {[["RF","R²=0.636","#22c55e"],["XGB","AUPRC=0.130","#6366f1"],
              ["LGB","AUPRC=0.138","#f59e0b"],["Hybrid","AUROC=0.605","#06b6d4"]].map(([name,val,color]) => (
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
                <div style={{ fontSize: 12, fontWeight: 600, color: "#e2e8f0" }}>Abubakar</div>
                <div style={{ fontSize: 10, color: "#475569" }}>SLU · Health Analytics</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Mobile header */}
      {isMobile && (
        <div style={{ position: "fixed", top: 0, left: 0, right: 0, zIndex: 100,
          background: "#0f172a", padding: "12px 16px",
          display: "flex", alignItems: "center", justifyContent: "space-between",
          boxShadow: "0 2px 8px rgba(0,0,0,0.2)" }}>
          <div>
            <div style={{ fontSize: 16, fontWeight: 800, color: "#6366f1" }}>ClinicalAI</div>
            <div style={{ fontSize: 9, color: "#475569", textTransform: "uppercase",
              letterSpacing: "0.05em" }}>MIMIC-IV Analytics</div>
          </div>
          <button onClick={() => setMenuOpen(!menuOpen)} style={{
            background: "none", border: "none", cursor: "pointer",
            padding: 8, display: "flex", flexDirection: "column",
            gap: 5, alignItems: "center", justifyContent: "center",
          }}>
            {menuOpen ? (
              <span style={{ color: "#fff", fontSize: 20, lineHeight: 1 }}>✕</span>
            ) : (
              <>
                <div style={{ width: 22, height: 2, background: "#fff", borderRadius: 1 }} />
                <div style={{ width: 22, height: 2, background: "#fff", borderRadius: 1 }} />
                <div style={{ width: 22, height: 2, background: "#fff", borderRadius: 1 }} />
              </>
            )}
          </button>
        </div>
      )}

      {/* Mobile slide-down menu */}
      {isMobile && menuOpen && (
        <div style={{ position: "fixed", top: 56, left: 0, right: 0, zIndex: 99,
          background: "#0f172a", padding: "8px 12px 16px",
          boxShadow: "0 4px 12px rgba(0,0,0,0.3)" }}>
          {NAV.map(n => (
            <button key={n.id} onClick={() => navigate(n.id)} style={{
              width: "100%", display: "flex", alignItems: "center", gap: 12,
              padding: "12px 14px", borderRadius: 8, marginBottom: 4,
              background: active===n.id ? "#6366f1" : "transparent",
              color: active===n.id ? "#fff" : "#94a3b8",
              border: "none", cursor: "pointer", fontSize: 15,
              textAlign: "left", fontWeight: active===n.id ? 600 : 400,
            }}>
              <span style={{ fontSize: 18 }}>{n.icon}</span>
              {n.label}
            </button>
          ))}
        </div>
      )}

      {/* Main content */}
      <main style={{
        flex: 1, minWidth: 0,
        padding: isMobile ? "80px 16px 24px" : "32px 36px",
        overflowY: "auto",
      }}>
        <Page setActive={navigate} />
      </main>
    </div>
  );
}
