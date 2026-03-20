# ─────────────────────────────────────────────────────────
# agents/data_agent.py
# DataAgent: handles all MIMIC-IV BigQuery queries and
# returns clean DataFrames for downstream agents.
# ─────────────────────────────────────────────────────────

import os
import yaml
import pandas as pd
from google.cloud import bigquery
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class DataAgent:
    """
    DataAgent is responsible for all interactions with MIMIC-IV on BigQuery.
    It exposes clean query methods for each paper's data needs.
    Raw patient data never leaves this agent — only aggregated DataFrames
    are passed to other agents (DUA compliance).
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = load_config(config_path)
        self.project_id = self.config["bigquery"]["project_id"]
        self.source_project = self.config["bigquery"]["source_project"]
        self.dataset = self.config["bigquery"]["dataset"]
        self.icu_dataset = self.config["bigquery"]["icu_dataset"]
        self.client = bigquery.Client(project=self.project_id)
        logger.info(f"DataAgent initialized — connected to {self.source_project}")

    def _run_query(self, query: str) -> pd.DataFrame:
        """Execute a BigQuery SQL query and return a DataFrame."""
        logger.info("Running BigQuery query...")
        df = self.client.query(query).to_dataframe()
        logger.info(f"Query returned {len(df):,} rows.")
        return df

    # ── Paper 1: Cost Prediction ──────────────────────────────────────────

    def get_admissions_cohort(self, limit: int = None) -> pd.DataFrame:
        """
        Fetch admission-level data for Paper 1 cost prediction.
        Includes demographics, insurance, admission type, and DRG codes.
        """
        limit_clause = f"LIMIT {limit}" if limit else ""
        query = f"""
        SELECT
            a.subject_id,
            a.hadm_id,
            a.admittime,
            a.dischtime,
            a.admission_type,
            a.admission_location,
            a.discharge_location,
            a.insurance,
            a.language,
            a.marital_status,
            a.race,
            p.gender,
            p.anchor_age                        AS age,
            TIMESTAMP_DIFF(
                a.dischtime, a.admittime, HOUR
            )                                   AS los_hours,
            d.drg_type,
            d.drg_code,
            d.description                       AS drg_description,
            d.drg_severity,
            d.drg_mortality
        FROM
            `{self.source_project}.{self.dataset}.admissions`     AS a
        JOIN
            `{self.source_project}.{self.dataset}.patients`       AS p
            ON a.subject_id = p.subject_id
        LEFT JOIN
            `{self.source_project}.{self.dataset}.drgcodes`       AS d
            ON a.hadm_id = d.hadm_id
        WHERE
            a.dischtime IS NOT NULL
            AND p.anchor_age >= 18
        {limit_clause}
        """
        return self._run_query(query)

    def get_top_diagnoses(self, limit: int = None) -> pd.DataFrame:
        """
        Fetch top ICD-10 diagnoses per admission for feature engineering.
        """
        limit_clause = f"LIMIT {limit}" if limit else ""
        query = f"""
        SELECT
            hadm_id,
            STRING_AGG(icd_code ORDER BY seq_num LIMIT 5) AS top5_icd_codes
        FROM
            `{self.source_project}.{self.dataset}.diagnoses_icd`
        WHERE
            icd_version = 10
        GROUP BY
            hadm_id
        {limit_clause}
        """
        return self._run_query(query)

    def get_admission_labs(self, limit: int = None) -> pd.DataFrame:
        """
        Fetch first lab values after admission for key clinical indicators.
        Used as features in Paper 1 cost prediction model.
        """
        limit_clause = f"LIMIT {limit}" if limit else ""
        query = f"""
        WITH ranked_labs AS (
            SELECT
                l.hadm_id,
                d.label,
                l.valuenum,
                ROW_NUMBER() OVER (
                    PARTITION BY l.hadm_id, d.label
                    ORDER BY l.charttime ASC
                ) AS rn
            FROM
                `{self.source_project}.{self.dataset}.labevents`  AS l
            JOIN
                `{self.source_project}.{self.dataset}.d_labitems` AS d
                ON l.itemid = d.itemid
            WHERE
                d.label IN (
                    'Creatinine', 'Hemoglobin', 'White Blood Cells',
                    'Sodium', 'Potassium', 'Lactate'
                )
                AND l.valuenum IS NOT NULL
        )
        SELECT
            hadm_id,
            MAX(CASE WHEN label = 'Creatinine'         THEN valuenum END) AS creatinine,
            MAX(CASE WHEN label = 'Hemoglobin'         THEN valuenum END) AS hemoglobin,
            MAX(CASE WHEN label = 'White Blood Cells'  THEN valuenum END) AS wbc,
            MAX(CASE WHEN label = 'Sodium'             THEN valuenum END) AS sodium,
            MAX(CASE WHEN label = 'Potassium'          THEN valuenum END) AS potassium,
            MAX(CASE WHEN label = 'Lactate'            THEN valuenum END) AS lactate
        FROM ranked_labs
        WHERE rn = 1
        GROUP BY hadm_id
        {limit_clause}
        """
        return self._run_query(query)

    # ── Paper 2: ICU Cost Disparities ────────────────────────────────────

    def get_icu_cohort(self, limit: int = None) -> pd.DataFrame:
        """
        Fetch ICU stay data linked with demographics and insurance.
        """
        limit_clause = f"LIMIT {limit}" if limit else ""
        query = f"""
        SELECT
            i.subject_id,
            i.hadm_id,
            i.stay_id,
            i.first_careunit,
            i.last_careunit,
            TIMESTAMP_DIFF(i.outtime, i.intime, HOUR)  AS icu_los_hours,
            a.insurance,
            a.race,
            a.marital_status,
            p.gender,
            p.anchor_age                                AS age,
            CASE
                WHEN p.anchor_age BETWEEN 18 AND 44 THEN '18-44'
                WHEN p.anchor_age BETWEEN 45 AND 64 THEN '45-64'
                WHEN p.anchor_age BETWEEN 65 AND 79 THEN '65-79'
                ELSE '80+'
            END                                         AS age_group,
            d.drg_code,
            d.drg_severity,
            d.drg_mortality
        FROM
            `{self.source_project}.{self.icu_dataset}.icustays`   AS i
        JOIN
            `{self.source_project}.{self.dataset}.admissions`     AS a
            ON i.hadm_id = a.hadm_id
        JOIN
            `{self.source_project}.{self.dataset}.patients`       AS p
            ON i.subject_id = p.subject_id
        LEFT JOIN
            `{self.source_project}.{self.dataset}.drgcodes`       AS d
            ON i.hadm_id = d.hadm_id
        WHERE
            p.anchor_age >= 18
            AND TIMESTAMP_DIFF(i.outtime, i.intime, HOUR) >= 4
        {limit_clause}
        """
        return self._run_query(query)

    def get_icu_disparity_cohort(self, limit: int = None) -> pd.DataFrame:
        """
        Fetch ICU stay data enriched with cost proxy, demographics,
        insurance, and clinical severity for Paper 2 disparity analysis.
        Cost proxy: DRG-adjusted LOS (icu_los_hours * drg_severity weight).
        """
        limit_clause = f"LIMIT {limit}" if limit else ""
        query = f"""
        SELECT
            i.subject_id,
            i.hadm_id,
            i.stay_id,
            i.first_careunit,
            TIMESTAMP_DIFF(i.outtime, i.intime, HOUR)      AS icu_los_hours,
            a.insurance,
            a.race,
            a.marital_status,
            p.gender,
            p.anchor_age                                    AS age,
            CASE
                WHEN p.anchor_age BETWEEN 18 AND 44 THEN '18-44'
                WHEN p.anchor_age BETWEEN 45 AND 64 THEN '45-64'
                WHEN p.anchor_age BETWEEN 65 AND 79 THEN '65-79'
                ELSE '80+'
            END                                             AS age_group,
            CASE
                WHEN UPPER(a.race) LIKE '%WHITE%'    THEN 'White'
                WHEN UPPER(a.race) LIKE '%BLACK%'    THEN 'Black'
                WHEN UPPER(a.race) LIKE '%HISPANIC%' THEN 'Hispanic'
                WHEN UPPER(a.race) LIKE '%ASIAN%'    THEN 'Asian'
                ELSE 'Other/Unknown'
            END                                             AS race_group,
            CASE
                WHEN UPPER(a.insurance) LIKE '%MEDICARE%'  THEN 'Medicare'
                WHEN UPPER(a.insurance) LIKE '%MEDICAID%'  THEN 'Medicaid'
                WHEN UPPER(a.insurance) LIKE '%OTHER%'     THEN 'Private'
                ELSE 'Self-pay/Other'
            END                                             AS insurance_group,
            d.drg_severity,
            d.drg_mortality,
            d.drg_code,
            TIMESTAMP_DIFF(i.outtime, i.intime, HOUR) *
                COALESCE(SAFE_CAST(d.drg_severity AS FLOAT64), 1.0)
                                                            AS cost_proxy
        FROM
            `{self.source_project}.{self.icu_dataset}.icustays`       AS i
        JOIN
            `{self.source_project}.{self.dataset}.admissions`         AS a
            ON i.hadm_id = a.hadm_id
        JOIN
            `{self.source_project}.{self.dataset}.patients`           AS p
            ON i.subject_id = p.subject_id
        LEFT JOIN
            `{self.source_project}.{self.dataset}.drgcodes`           AS d
            ON i.hadm_id = d.hadm_id
        WHERE
            p.anchor_age >= 18
            AND TIMESTAMP_DIFF(i.outtime, i.intime, HOUR) >= 4
            AND TIMESTAMP_DIFF(i.outtime, i.intime, HOUR) <= 2000
        {limit_clause}
        """
        return self._run_query(query)

    # ── Paper 3: ICU Readmission ──────────────────────────────────────────

    def get_icu_readmission_cohort(self, window_hours: int = 72,
                                   limit: int = None) -> pd.DataFrame:
        """
        Build ICU readmission labels.
        A readmission is defined as a return to ICU within window_hours
        of a prior ICU discharge. Excludes patients who died in the ICU.
        """
        limit_clause = f"LIMIT {limit}" if limit else ""
        query = f"""
        WITH icu_ordered AS (
            SELECT
                i.subject_id,
                i.hadm_id,
                i.stay_id,
                i.intime,
                i.outtime,
                i.first_careunit,
                TIMESTAMP_DIFF(i.outtime, i.intime, HOUR)   AS icu_los_hours,
                a.hospital_expire_flag,
                a.insurance,
                a.race,
                p.gender,
                p.anchor_age                                 AS age,
                LEAD(i.intime) OVER (
                    PARTITION BY i.subject_id
                    ORDER BY i.intime
                )                                            AS next_icu_intime
            FROM
                `{self.source_project}.{self.icu_dataset}.icustays`  AS i
            JOIN
                `{self.source_project}.{self.dataset}.admissions`    AS a
                ON i.hadm_id = a.hadm_id
            JOIN
                `{self.source_project}.{self.dataset}.patients`      AS p
                ON i.subject_id = p.subject_id
            WHERE
                p.anchor_age >= 18
                AND a.hospital_expire_flag = 0
        )
        SELECT
            subject_id,
            hadm_id,
            stay_id,
            intime,
            outtime,
            first_careunit,
            icu_los_hours,
            insurance,
            race,
            gender,
            age,
            next_icu_intime,
            CASE
                WHEN next_icu_intime IS NOT NULL
                AND TIMESTAMP_DIFF(
                    next_icu_intime, outtime, HOUR
                ) <= {window_hours}
                THEN 1
                ELSE 0
            END AS readmitted
        FROM icu_ordered
        WHERE icu_los_hours >= 4
        {limit_clause}
        """
        return self._run_query(query)

    def get_procedure_counts(self, limit: int = None) -> pd.DataFrame:
        """
        Count procedures per admission from procedures_icd.
        Used as a complexity feature for Paper 1 cost prediction.
        """
        limit_clause = f"LIMIT {limit}" if limit else ""
        query = f"""
        SELECT
            hadm_id,
            COUNT(*)                                    AS procedure_count,
            COUNT(DISTINCT icd_code)                    AS unique_procedures
        FROM
            `{self.source_project}.{self.dataset}.procedures_icd`
        GROUP BY
            hadm_id
        {limit_clause}
        """
        return self._run_query(query)

    def get_charlson_index(self, limit: int = None) -> pd.DataFrame:
        """
        Compute a simplified Charlson Comorbidity Index (CCI) proxy
        using ICD-10 diagnosis codes per admission.
        Covers 12 major comorbidity groups.
        Used as a key feature for Paper 1 cost prediction.
        """
        limit_clause = f"LIMIT {limit}" if limit else ""
        query = f"""
        WITH diag AS (
            SELECT DISTINCT hadm_id, icd_code
            FROM `{self.source_project}.{self.dataset}.diagnoses_icd`
            WHERE icd_version = 10
        )
        SELECT
            hadm_id,
            MAX(CASE WHEN REGEXP_CONTAINS(icd_code, r'^I2[12]') THEN 1 ELSE 0 END) AS cci_mi,
            MAX(CASE WHEN REGEXP_CONTAINS(icd_code, r'^I50') THEN 1 ELSE 0 END) AS cci_chf,
            MAX(CASE WHEN REGEXP_CONTAINS(icd_code, r'^I7[01]') THEN 1 ELSE 0 END) AS cci_pvd,
            MAX(CASE WHEN REGEXP_CONTAINS(icd_code, r'^I6[0-9]') THEN 1 ELSE 0 END) AS cci_cvd,
            MAX(CASE WHEN REGEXP_CONTAINS(icd_code, r'^J4[0-7]') THEN 1 ELSE 0 END) AS cci_copd,
            MAX(CASE WHEN REGEXP_CONTAINS(icd_code, r'^E1[0-4]') THEN 1 ELSE 0 END) AS cci_dm,
            MAX(CASE WHEN REGEXP_CONTAINS(icd_code, r'^E1[0-4][.][2-5]') THEN 1 ELSE 0 END) AS cci_dm_comp,
            MAX(CASE WHEN REGEXP_CONTAINS(icd_code, r'^N1[0-9]') THEN 1 ELSE 0 END) AS cci_renal,
            MAX(CASE WHEN REGEXP_CONTAINS(icd_code, r'^K7[0-6]') THEN 1 ELSE 0 END) AS cci_liver_mild,
            MAX(CASE WHEN REGEXP_CONTAINS(icd_code, r'^K72') THEN 1 ELSE 0 END) AS cci_liver_severe,
            MAX(CASE WHEN REGEXP_CONTAINS(icd_code, r'^C[0-9][0-9]') THEN 1 ELSE 0 END) AS cci_cancer,
            MAX(CASE WHEN REGEXP_CONTAINS(icd_code, r'^B2[0-4]') THEN 1 ELSE 0 END) AS cci_hiv,
            SUM(CASE
                WHEN REGEXP_CONTAINS(icd_code, r'^I2[12]')          THEN 1
                WHEN REGEXP_CONTAINS(icd_code, r'^I50')              THEN 1
                WHEN REGEXP_CONTAINS(icd_code, r'^I7[01]')           THEN 1
                WHEN REGEXP_CONTAINS(icd_code, r'^I6[0-9]')          THEN 1
                WHEN REGEXP_CONTAINS(icd_code, r'^J4[0-7]')          THEN 1
                WHEN REGEXP_CONTAINS(icd_code, r'^E1[0-4]')          THEN 1
                WHEN REGEXP_CONTAINS(icd_code, r'^E1[0-4][.][2-5]')  THEN 2
                WHEN REGEXP_CONTAINS(icd_code, r'^N1[0-9]')          THEN 2
                WHEN REGEXP_CONTAINS(icd_code, r'^K7[0-6]')          THEN 1
                WHEN REGEXP_CONTAINS(icd_code, r'^K72')               THEN 3
                WHEN REGEXP_CONTAINS(icd_code, r'^C[0-9][0-9]')      THEN 2
                WHEN REGEXP_CONTAINS(icd_code, r'^B2[0-4]')          THEN 6
                ELSE 0
            END)                                                     AS cci_total_score
        FROM diag
        GROUP BY hadm_id
        {limit_clause}
        """
        return self._run_query(query)

    def get_icu_discharge_vitals(self, limit: int = None) -> pd.DataFrame:
        """
        Fetch last recorded vitals in the final 6 hours before ICU discharge.
        Used as features for Paper 3 ICU readmission prediction.
        """
        limit_clause = f"LIMIT {limit}" if limit else ""
        query = f"""
        WITH icu_stays AS (
            SELECT stay_id, outtime
            FROM `{self.source_project}.{self.icu_dataset}.icustays`
        ),
        last_vitals AS (
            SELECT
                c.stay_id,
                c.itemid,
                c.valuenum,
                ROW_NUMBER() OVER (
                    PARTITION BY c.stay_id, c.itemid
                    ORDER BY c.charttime DESC
                ) AS rn
            FROM `{self.source_project}.{self.icu_dataset}.chartevents` AS c
            JOIN icu_stays AS i ON c.stay_id = i.stay_id
            WHERE
                c.itemid IN (220045, 220277, 220210, 220052, 223761)
                AND c.valuenum IS NOT NULL
                AND c.charttime >= TIMESTAMP_SUB(i.outtime, INTERVAL 6 HOUR)
                AND c.charttime <= i.outtime
        )
        SELECT
            stay_id,
            MAX(CASE WHEN itemid = 220045 THEN valuenum END) AS heart_rate_last,
            MAX(CASE WHEN itemid = 220277 THEN valuenum END) AS spo2_last,
            MAX(CASE WHEN itemid = 220210 THEN valuenum END) AS resp_rate_last,
            MAX(CASE WHEN itemid = 220052 THEN valuenum END) AS abp_mean_last,
            MAX(CASE WHEN itemid = 223761 THEN valuenum END) AS temp_f_last
        FROM last_vitals
        WHERE rn = 1
        GROUP BY stay_id
        {limit_clause}
        """
        return self._run_query(query)

    def get_icu_vital_trends(self, limit: int = None) -> pd.DataFrame:
        """
        Compute 24h vital trend features before ICU discharge.
        For each key vital: mean, std, delta (last - first).
        Used as enriched features for Paper 3 ICU readmission prediction.
        """
        limit_clause = f"LIMIT {limit}" if limit else ""
        query = f"""
        WITH icu_stays AS (
            SELECT stay_id, outtime
            FROM `{self.source_project}.{self.icu_dataset}.icustays`
        ),
        vitals_24h AS (
            SELECT
                c.stay_id,
                c.itemid,
                c.valuenum,
                c.charttime,
                ROW_NUMBER() OVER (
                    PARTITION BY c.stay_id, c.itemid
                    ORDER BY c.charttime ASC
                ) AS rn_asc,
                ROW_NUMBER() OVER (
                    PARTITION BY c.stay_id, c.itemid
                    ORDER BY c.charttime DESC
                ) AS rn_desc
            FROM `{self.source_project}.{self.icu_dataset}.chartevents` AS c
            JOIN icu_stays AS i ON c.stay_id = i.stay_id
            WHERE
                c.itemid IN (220045, 220277, 220210, 220052, 223761)
                AND c.valuenum IS NOT NULL
                AND c.charttime >= TIMESTAMP_SUB(i.outtime, INTERVAL 24 HOUR)
                AND c.charttime <= i.outtime
        )
        SELECT
            stay_id,
            AVG(CASE WHEN itemid = 220045 THEN valuenum END)                 AS hr_mean_24h,
            STDDEV(CASE WHEN itemid = 220045 THEN valuenum END)              AS hr_std_24h,
            MAX(CASE WHEN itemid = 220045 AND rn_desc = 1 THEN valuenum END) -
            MAX(CASE WHEN itemid = 220045 AND rn_asc  = 1 THEN valuenum END) AS hr_delta_24h,
            AVG(CASE WHEN itemid = 220277 THEN valuenum END)                 AS spo2_mean_24h,
            STDDEV(CASE WHEN itemid = 220277 THEN valuenum END)              AS spo2_std_24h,
            MAX(CASE WHEN itemid = 220277 AND rn_desc = 1 THEN valuenum END) -
            MAX(CASE WHEN itemid = 220277 AND rn_asc  = 1 THEN valuenum END) AS spo2_delta_24h,
            AVG(CASE WHEN itemid = 220210 THEN valuenum END)                 AS rr_mean_24h,
            STDDEV(CASE WHEN itemid = 220210 THEN valuenum END)              AS rr_std_24h,
            MAX(CASE WHEN itemid = 220210 AND rn_desc = 1 THEN valuenum END) -
            MAX(CASE WHEN itemid = 220210 AND rn_asc  = 1 THEN valuenum END) AS rr_delta_24h,
            AVG(CASE WHEN itemid = 220052 THEN valuenum END)                 AS abp_mean_24h,
            STDDEV(CASE WHEN itemid = 220052 THEN valuenum END)              AS abp_std_24h,
            MAX(CASE WHEN itemid = 220052 AND rn_desc = 1 THEN valuenum END) -
            MAX(CASE WHEN itemid = 220052 AND rn_asc  = 1 THEN valuenum END) AS abp_delta_24h,
            AVG(CASE WHEN itemid = 223761 THEN valuenum END)                 AS temp_mean_24h,
            STDDEV(CASE WHEN itemid = 223761 THEN valuenum END)              AS temp_std_24h,
            MAX(CASE WHEN itemid = 223761 AND rn_desc = 1 THEN valuenum END) -
            MAX(CASE WHEN itemid = 223761 AND rn_asc  = 1 THEN valuenum END) AS temp_delta_24h
        FROM vitals_24h
        GROUP BY stay_id
        {limit_clause}
        """
        return self._run_query(query)

    def get_fluid_balance(self, limit: int = None) -> pd.DataFrame:
        """
        Compute total fluid output per ICU stay from outputevents.
        Used as a feature for Paper 3 readmission prediction.
        """
        limit_clause = f"LIMIT {limit}" if limit else ""
        query = f"""
        SELECT
            stay_id,
            SUM(value)      AS total_output_ml,
            COUNT(*)        AS output_event_count
        FROM
            `{self.source_project}.{self.icu_dataset}.outputevents`
        WHERE
            value IS NOT NULL
            AND value > 0
        GROUP BY
            stay_id
        {limit_clause}
        """
        return self._run_query(query)

    # ── Utility ───────────────────────────────────────────────────────────

    def get_summary_stats(self) -> None:
        """Print a quick summary of key table sizes."""
        tables = {
            "admissions":    f"`{self.source_project}.{self.dataset}.admissions`",
            "patients":      f"`{self.source_project}.{self.dataset}.patients`",
            "icustays":      f"`{self.source_project}.{self.icu_dataset}.icustays`",
            "diagnoses_icd": f"`{self.source_project}.{self.dataset}.diagnoses_icd`",
            "drgcodes":      f"`{self.source_project}.{self.dataset}.drgcodes`",
            "labevents":     f"`{self.source_project}.{self.dataset}.labevents`",
        }
        print("\n── MIMIC-IV Table Summary ──────────────────────")
        for name, table in tables.items():
            result = self.client.query(
                f"SELECT COUNT(*) as n FROM {table}"
            ).result()
            for row in result:
                print(f"  {name:<20} {row.n:>12,} rows")
        print("────────────────────────────────────────────────\n")


# ── Quick test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = DataAgent()

    agent.get_summary_stats()

    print("Fetching admissions cohort sample (100 rows)...")
    df = agent.get_admissions_cohort(limit=100)
    print(df.head())
    print(f"\nColumns: {list(df.columns)}\n")

    print("Fetching ICU readmission cohort sample (100 rows)...")
    df_icu = agent.get_icu_readmission_cohort(limit=100)
    print(df_icu["readmitted"].value_counts())