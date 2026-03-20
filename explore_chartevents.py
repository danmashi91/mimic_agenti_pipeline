# explore_chartevents.py
# Run this once to find the correct itemids for key clinical features
# in MIMIC-IV chartevents before building enriched feature queries.

from agents.data_agent import DataAgent

agent = DataAgent()

# ── 1. Find itemids for key vitals ────────────────────────────────────────
print("\n── Searching for key vital itemids ─────────────────────────────")
query = """
SELECT
    itemid,
    label,
    unitname,
    COUNT(*) as n_records
FROM `physionet-data.mimiciv_3_1_icu.d_items`
WHERE LOWER(label) IN (
    'heart rate',
    'respiratory rate',
    'o2 saturation pulseoxymetry',
    'arterial blood pressure mean',
    'temperature fahrenheit',
    'gcs total',
    'mean airway pressure'
)
GROUP BY itemid, label, unitname
ORDER BY label
"""
df = agent._run_query(query)
print(df.to_string(index=False))

# ── 2. Check fluid balance availability ──────────────────────────────────
print("\n── Checking outputevents availability ──────────────────────────")
query2 = """
SELECT COUNT(*) as n_output_events
FROM `physionet-data.mimiciv_3_1_icu.outputevents`
"""
df2 = agent._run_query(query2)
print(df2.to_string(index=False))

# ── 3. Check procedures_icd availability ─────────────────────────────────
print("\n── Checking procedures_icd availability ────────────────────────")
query3 = """
SELECT COUNT(*) as n_procedures
FROM `physionet-data.mimiciv_3_1_hosp.procedures_icd`
"""
df3 = agent._run_query(query3)
print(df3.to_string(index=False))