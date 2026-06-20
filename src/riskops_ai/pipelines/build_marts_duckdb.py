from __future__ import annotations

import json
from pathlib import Path
import duckdb
import pandas as pd

from riskops_ai.config import BRONZE_DIR, SILVER_DIR, GOLD_DIR, ensure_dirs


def _con() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(database=":memory:")


def _csv(path: Path) -> str:
    return str(path).replace("'", "''")


def build_silver(con: duckdb.DuckDBPyConnection) -> None:
    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    table_sql = {
        "customers": f"""
            SELECT
                customer_id::VARCHAR AS customer_id,
                state::VARCHAR AS state,
                city::VARCHAR AS city,
                age::INTEGER AS age,
                employment_type::VARCHAR AS employment_type,
                monthly_income::DOUBLE AS monthly_income,
                credit_segment::VARCHAR AS credit_segment,
                created_at::DATE AS created_at
            FROM read_csv_auto('{_csv(BRONZE_DIR / 'customers.csv')}')
        """,
        "bureau_features": f"""
            SELECT
                customer_id::VARCHAR AS customer_id,
                bureau_score::INTEGER AS bureau_score,
                active_loans::INTEGER AS active_loans,
                credit_utilization::DOUBLE AS credit_utilization,
                enquiries_3m::INTEGER AS enquiries_3m,
                delinq_12m::INTEGER AS delinq_12m
            FROM read_csv_auto('{_csv(BRONZE_DIR / 'bureau_features.csv')}')
        """,
        "loan_applications": f"""
            SELECT
                app_id::VARCHAR AS app_id,
                customer_id::VARCHAR AS customer_id,
                application_date::DATE AS application_date,
                product::VARCHAR AS product,
                requested_amount::DOUBLE AS requested_amount,
                tenure_months::INTEGER AS tenure_months,
                channel::VARCHAR AS channel,
                approval_status::VARCHAR AS approval_status,
                rejection_reason::VARCHAR AS rejection_reason,
                approved_amount::DOUBLE AS approved_amount,
                interest_rate::DOUBLE AS interest_rate,
                risk_score::DOUBLE AS risk_score
            FROM read_csv_auto('{_csv(BRONZE_DIR / 'loan_applications.csv')}')
        """,
        "disbursements": f"""
            SELECT
                loan_id::VARCHAR AS loan_id,
                app_id::VARCHAR AS app_id,
                customer_id::VARCHAR AS customer_id,
                disbursed_date::DATE AS disbursed_date,
                principal::DOUBLE AS principal,
                tenure_months::INTEGER AS tenure_months,
                interest_rate::DOUBLE AS interest_rate,
                risk_score::DOUBLE AS risk_score
            FROM read_csv_auto('{_csv(BRONZE_DIR / 'disbursements.csv')}')
        """,
        "repayments": f"""
            SELECT
                loan_id::VARCHAR AS loan_id,
                due_date::DATE AS due_date,
                paid_date::DATE AS paid_date,
                mob::INTEGER AS mob,
                due_amount::DOUBLE AS due_amount,
                paid_amount::DOUBLE AS paid_amount,
                dpd::INTEGER AS dpd
            FROM read_csv_auto('{_csv(BRONZE_DIR / 'repayments.csv')}')
        """,
        "collection_calls": f"""
            SELECT
                call_id::VARCHAR AS call_id,
                loan_id::VARCHAR AS loan_id,
                call_date::DATE AS call_date,
                outcome::VARCHAR AS outcome,
                agent_notes::VARCHAR AS agent_notes
            FROM read_csv_auto('{_csv(BRONZE_DIR / 'collection_calls.csv')}')
        """,
    }
    for table, sql in table_sql.items():
        con.execute(f"CREATE OR REPLACE TABLE {table} AS {sql}")
        con.execute(f"COPY {table} TO '{_csv(SILVER_DIR / (table + '.parquet'))}' (FORMAT PARQUET)")


def build_gold(con: duckdb.DuckDBPyConnection) -> None:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    con.execute("""
        CREATE OR REPLACE TABLE gold_loan_features AS
        SELECT
            d.loan_id,
            d.app_id,
            d.customer_id,
            d.disbursed_date,
            date_trunc('month', d.disbursed_date)::DATE AS disbursed_month,
            c.state,
            c.city,
            c.employment_type,
            c.credit_segment,
            a.product,
            a.channel,
            d.principal,
            d.tenure_months,
            d.interest_rate,
            d.risk_score,
            b.bureau_score,
            b.active_loans,
            b.credit_utilization,
            b.enquiries_3m,
            b.delinq_12m,
            CASE
                WHEN d.risk_score < 220 THEN 'Low'
                WHEN d.risk_score < 300 THEN 'Medium'
                WHEN d.risk_score < 380 THEN 'High'
                ELSE 'Very High'
            END AS risk_band
        FROM disbursements d
        JOIN loan_applications a ON d.app_id = a.app_id
        JOIN customers c ON d.customer_id = c.customer_id
        LEFT JOIN bureau_features b ON d.customer_id = b.customer_id
    """)

    con.execute("""
        CREATE OR REPLACE TABLE gold_repayment_enriched AS
        SELECT
            r.*,
            f.disbursed_month,
            f.state,
            f.city,
            f.employment_type,
            f.credit_segment,
            f.product,
            f.channel,
            f.principal,
            f.risk_band,
            CASE
                WHEN r.dpd = 0 THEN '0 Current'
                WHEN r.dpd BETWEEN 1 AND 30 THEN '1-30 Early'
                WHEN r.dpd BETWEEN 31 AND 60 THEN '31-60 Moderate'
                WHEN r.dpd BETWEEN 61 AND 90 THEN '61-90 Severe'
                ELSE '90+ Critical'
            END AS dpd_bucket,
            CASE WHEN r.dpd >= 30 THEN 1 ELSE 0 END AS is_30_plus_dpd
        FROM repayments r
        LEFT JOIN gold_loan_features f ON r.loan_id = f.loan_id
    """)

    con.execute("""
        CREATE OR REPLACE TABLE gold_approval_funnel AS
        SELECT
            date_trunc('month', application_date)::DATE AS application_month,
            product,
            channel,
            c.employment_type,
            c.credit_segment,
            COUNT(*) AS applications,
            SUM(CASE WHEN approval_status = 'Approved' THEN 1 ELSE 0 END) AS approved_applications,
            ROUND(100.0 * SUM(CASE WHEN approval_status = 'Approved' THEN 1 ELSE 0 END) / COUNT(*), 2) AS approval_rate_pct,
            ROUND(AVG(risk_score), 2) AS avg_risk_score,
            SUM(approved_amount) AS approved_amount
        FROM loan_applications a
        LEFT JOIN customers c ON a.customer_id = c.customer_id
        GROUP BY 1,2,3,4,5
    """)

    con.execute("""
        CREATE OR REPLACE TABLE gold_portfolio_summary AS
        SELECT
            disbursed_month,
            product,
            channel,
            risk_band,
            COUNT(DISTINCT loan_id) AS loans,
            ROUND(SUM(principal), 2) AS principal_disbursed,
            ROUND(AVG(risk_score), 2) AS avg_risk_score,
            ROUND(AVG(bureau_score), 2) AS avg_bureau_score
        FROM gold_loan_features
        GROUP BY 1,2,3,4
    """)

    con.execute("""
        CREATE OR REPLACE TABLE gold_dpd_bucket_analysis AS
        SELECT
            date_trunc('month', due_date)::DATE AS due_month,
            product,
            channel,
            risk_band,
            dpd_bucket,
            COUNT(*) AS scheduled_instalments,
            ROUND(SUM(due_amount), 2) AS amount_due,
            ROUND(SUM(COALESCE(paid_amount, 0)), 2) AS amount_paid,
            ROUND(100.0 * SUM(is_30_plus_dpd) / COUNT(*), 2) AS thirty_plus_dpd_rate_pct
        FROM gold_repayment_enriched
        GROUP BY 1,2,3,4,5
    """)

    con.execute("""
        CREATE OR REPLACE TABLE gold_vintage_analysis AS
        SELECT
            disbursed_month,
            mob,
            risk_band,
            COUNT(DISTINCT loan_id) AS loans_observed,
            ROUND(100.0 * SUM(is_30_plus_dpd) / COUNT(*), 2) AS thirty_plus_dpd_rate_pct
        FROM gold_repayment_enriched
        GROUP BY 1,2,3
    """)

    for table in [
        "gold_loan_features",
        "gold_repayment_enriched",
        "gold_approval_funnel",
        "gold_portfolio_summary",
        "gold_dpd_bucket_analysis",
        "gold_vintage_analysis",
    ]:
        con.execute(f"COPY {table} TO '{_csv(GOLD_DIR / (table + '.parquet'))}' (FORMAT PARQUET)")


def build_data_quality_report(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    checks: list[dict] = []

    def add_check(table: str, rule: str, status: str, severity: str, metric_value: float, details: str):
        checks.append({
            "table_name": table,
            "rule_name": rule,
            "status": status,
            "severity": severity,
            "metric_value": metric_value,
            "details": details,
        })

    critical_tables = {
        "customers": ["customer_id", "state", "age", "monthly_income"],
        "loan_applications": ["app_id", "customer_id", "application_date", "approval_status", "risk_score"],
        "disbursements": ["loan_id", "app_id", "customer_id", "principal"],
        "repayments": ["loan_id", "due_date", "due_amount", "paid_amount", "dpd"],
        "bureau_features": ["customer_id", "bureau_score", "delinq_12m"],
    }
    for table, cols in critical_tables.items():
        row_count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for col in cols:
            nulls = con.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} IS NULL").fetchone()[0]
            pct = round(100 * nulls / max(row_count, 1), 3)
            add_check(table, f"null_rate_{col}", "PASS" if pct <= 5 else "FAIL", "HIGH" if pct > 5 else "LOW", pct, f"{nulls} nulls out of {row_count} rows")

    pk_checks = {
        "customers": "customer_id",
        "loan_applications": "app_id",
        "disbursements": "loan_id",
        "collection_calls": "call_id",
    }
    for table, key in pk_checks.items():
        dupes = con.execute(f"SELECT COUNT(*) - COUNT(DISTINCT {key}) FROM {table}").fetchone()[0]
        add_check(table, f"duplicate_primary_key_{key}", "PASS" if dupes == 0 else "FAIL", "CRITICAL" if dupes else "LOW", float(dupes), f"Duplicate {key} rows: {dupes}")

    repayment_dupes = con.execute("""
        SELECT COUNT(*) FROM (
            SELECT loan_id, due_date, mob, COUNT(*) AS cnt
            FROM repayments
            GROUP BY 1,2,3
            HAVING COUNT(*) > 1
        )
    """).fetchone()[0]
    add_check("repayments", "duplicate_repayment_schedule_key", "PASS" if repayment_dupes == 0 else "FAIL", "HIGH" if repayment_dupes else "LOW", float(repayment_dupes), f"Duplicate repayment schedule keys: {repayment_dupes}")

    fk_missing_apps = con.execute("""
        SELECT COUNT(*)
        FROM loan_applications a
        LEFT JOIN customers c ON a.customer_id = c.customer_id
        WHERE c.customer_id IS NULL
    """).fetchone()[0]
    add_check("loan_applications", "fk_customer_id_exists", "PASS" if fk_missing_apps == 0 else "FAIL", "CRITICAL" if fk_missing_apps else "LOW", float(fk_missing_apps), "Applications with missing customer_id")

    fk_missing_repayments = con.execute("""
        SELECT COUNT(*)
        FROM repayments r
        LEFT JOIN disbursements d ON r.loan_id = d.loan_id
        WHERE d.loan_id IS NULL
    """).fetchone()[0]
    add_check("repayments", "fk_loan_id_exists", "PASS" if fk_missing_repayments == 0 else "FAIL", "CRITICAL" if fk_missing_repayments else "LOW", float(fk_missing_repayments), "Repayments with missing loan_id")

    outlier_principal = con.execute("SELECT COUNT(*) FROM disbursements WHERE principal <= 0 OR principal > 500000").fetchone()[0]
    add_check("disbursements", "principal_outlier_threshold", "PASS" if outlier_principal == 0 else "FAIL", "HIGH" if outlier_principal else "LOW", float(outlier_principal), "Principal should be between 0 and 500000")

    # Approval anomaly rule: latest month approval rate drop vs previous month.
    anomaly = con.execute("""
        WITH monthly AS (
            SELECT date_trunc('month', application_date)::DATE AS month,
                   COUNT(*) AS apps,
                   100.0 * SUM(CASE WHEN approval_status = 'Approved' THEN 1 ELSE 0 END) / COUNT(*) AS approval_rate
            FROM loan_applications
            GROUP BY 1
        ), ranked AS (
            SELECT *, ROW_NUMBER() OVER (ORDER BY month DESC) AS rn
            FROM monthly
        )
        SELECT
            MAX(CASE WHEN rn = 1 THEN approval_rate END) AS latest_rate,
            MAX(CASE WHEN rn = 2 THEN approval_rate END) AS previous_rate
        FROM ranked
        WHERE rn IN (1,2)
    """).fetchone()
    latest, previous = anomaly
    drop = round(float(previous - latest), 2) if latest is not None and previous is not None else 0.0
    add_check(
        "loan_applications",
        "approval_rate_monthly_drop_gt_5pp",
        "FAIL" if drop > 5 else "PASS",
        "HIGH" if drop > 5 else "LOW",
        drop,
        f"Latest approval rate {latest:.2f}% vs previous {previous:.2f}%; drop={drop:.2f}pp" if latest is not None and previous is not None else "Insufficient history",
    )

    dq = pd.DataFrame(checks)
    dq.to_parquet(GOLD_DIR / "gold_data_quality_report.parquet", index=False)
    dq.to_csv(GOLD_DIR / "gold_data_quality_report.csv", index=False)
    return dq


def build_all() -> dict[str, int]:
    ensure_dirs()
    con = _con()
    build_silver(con)
    # Re-load silver into named tables to mimic separate stages.
    for table in ["customers", "bureau_features", "loan_applications", "disbursements", "repayments", "collection_calls"]:
        con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_parquet('{_csv(SILVER_DIR / (table + '.parquet'))}')")
    build_gold(con)
    dq = build_data_quality_report(con)
    counts = {}
    for p in GOLD_DIR.glob("*.parquet"):
        counts[p.stem] = con.execute(f"SELECT COUNT(*) FROM read_parquet('{_csv(p)}')").fetchone()[0]
    counts["dq_checks"] = len(dq)
    return counts


if __name__ == "__main__":
    counts = build_all()
    print("Built silver and gold marts:")
    for table, count in sorted(counts.items()):
        print(f"- {table}: {count:,} rows")
