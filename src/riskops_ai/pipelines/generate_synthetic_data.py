from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from riskops_ai.config import BRONZE_DIR, ensure_dirs

RNG = np.random.default_rng(42)

STATES = ["Karnataka", "Maharashtra", "Tamil Nadu", "Telangana", "Delhi", "Gujarat"]
CITIES = {
    "Karnataka": ["Bangalore", "Mysore", "Hubli"],
    "Maharashtra": ["Mumbai", "Pune", "Nagpur"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai"],
    "Telangana": ["Hyderabad", "Warangal"],
    "Delhi": ["Delhi"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara"],
}
EMPLOYMENT = ["Salaried", "Self-employed", "Gig worker", "Small business"]
PRODUCTS = ["Personal Loan", "Two Wheeler Loan", "Consumer Durable Loan", "Merchant Loan"]
CHANNELS = ["App", "Branch", "Partner", "Referral"]


def _month_starts(end: str = "2026-06-01", months: int = 12) -> pd.DatetimeIndex:
    return pd.date_range(end=pd.Timestamp(end), periods=months, freq="MS")


def generate_customers(n_customers: int = 2500) -> pd.DataFrame:
    customer_ids = [f"CUST{i:06d}" for i in range(1, n_customers + 1)]
    states = RNG.choice(STATES, size=n_customers, p=[0.28, 0.22, 0.16, 0.14, 0.10, 0.10])
    cities = [RNG.choice(CITIES[s]) for s in states]
    employment = RNG.choice(EMPLOYMENT, size=n_customers, p=[0.48, 0.25, 0.12, 0.15])
    age = RNG.integers(21, 62, size=n_customers)
    income_base = np.where(employment == "Salaried", 52000, np.where(employment == "Self-employed", 65000, 33000))
    monthly_income = np.maximum(12000, RNG.normal(income_base, 18000)).round(0).astype(int)
    credit_segment = pd.cut(
        monthly_income,
        bins=[0, 25000, 50000, 90000, 10_000_000],
        labels=["Mass", "Emerging", "Prime", "Affluent"],
    ).astype(str)
    created_at = pd.Timestamp("2025-01-01") + pd.to_timedelta(RNG.integers(0, 500, size=n_customers), unit="D")
    return pd.DataFrame({
        "customer_id": customer_ids,
        "state": states,
        "city": cities,
        "age": age,
        "employment_type": employment,
        "monthly_income": monthly_income,
        "credit_segment": credit_segment,
        "created_at": created_at.date.astype(str),
    })


def generate_bureau(customers: pd.DataFrame) -> pd.DataFrame:
    n = len(customers)
    score_shift = customers["credit_segment"].map({"Mass": -70, "Emerging": -25, "Prime": 20, "Affluent": 45}).fillna(0).to_numpy()
    bureau_score = np.clip(RNG.normal(685 + score_shift, 65), 300, 900).round(0).astype(int)
    active_loans = np.clip(RNG.poisson(2.0, n) + (bureau_score < 620).astype(int), 0, 9)
    credit_utilization = np.clip(RNG.beta(2, 4, n) + (bureau_score < 620) * 0.25, 0, 1).round(2)
    enquiries_3m = np.clip(RNG.poisson(1.0, n) + (bureau_score < 620).astype(int), 0, 8)
    delinq_12m = np.clip(RNG.poisson(0.25, n) + (bureau_score < 580).astype(int), 0, 5)
    return pd.DataFrame({
        "customer_id": customers["customer_id"],
        "bureau_score": bureau_score,
        "active_loans": active_loans,
        "credit_utilization": credit_utilization,
        "enquiries_3m": enquiries_3m,
        "delinq_12m": delinq_12m,
    })


def generate_applications(customers: pd.DataFrame, bureau: pd.DataFrame, n_apps: int = 7200) -> pd.DataFrame:
    months = _month_starts()
    month_probs = np.array([0.065, 0.07, 0.075, 0.08, 0.08, 0.085, 0.085, 0.09, 0.09, 0.095, 0.1, 0.085])
    month_probs = month_probs / month_probs.sum()
    app_month = RNG.choice(months, size=n_apps, p=month_probs)
    app_day = app_month + pd.to_timedelta(RNG.integers(0, 27, size=n_apps), unit="D")
    cust_sample = customers.sample(n_apps, replace=True, random_state=7).reset_index(drop=True)
    bureau_map = bureau.set_index("customer_id")
    b = bureau_map.loc[cust_sample["customer_id"]].reset_index(drop=True)
    product = RNG.choice(PRODUCTS, size=n_apps, p=[0.46, 0.22, 0.20, 0.12])
    channel = RNG.choice(CHANNELS, size=n_apps, p=[0.50, 0.15, 0.25, 0.10])
    requested_amount = np.where(product == "Personal Loan", RNG.normal(120000, 45000, n_apps),
                         np.where(product == "Merchant Loan", RNG.normal(180000, 75000, n_apps), RNG.normal(65000, 25000, n_apps)))
    requested_amount = np.clip(requested_amount, 10000, 500000).round(-3).astype(int)
    tenure_months = RNG.choice([6, 9, 12, 18, 24, 36], size=n_apps, p=[0.10, 0.10, 0.35, 0.20, 0.18, 0.07])

    # Higher risk score means riskier applicant.
    risk_score = (
        900 - b["bureau_score"].to_numpy()
        + 45 * b["delinq_12m"].to_numpy()
        + 18 * b["enquiries_3m"].to_numpy()
        + 75 * b["credit_utilization"].to_numpy()
        + np.where(cust_sample["employment_type"].eq("Gig worker"), 35, 0)
        + np.where(cust_sample["employment_type"].eq("Self-employed"), 18, 0)
    )

    # Controlled business incident: recent-month approval policy tightened for partner/self-employed/high-risk.
    incident_month = pd.Timestamp("2026-06-01")
    incident_penalty = (
        (app_month == incident_month)
        & (channel == "Partner")
        & cust_sample["employment_type"].isin(["Self-employed", "Small business"]).to_numpy()
    ) * 70
    risk_score = np.clip(risk_score + incident_penalty + RNG.normal(0, 25, n_apps), 50, 700).round(1)

    approve_probability = 1 / (1 + np.exp((risk_score - 285) / 45))
    approved = RNG.random(n_apps) < approve_probability
    rejection_reason = np.where(
        approved,
        "",
        np.select(
            [b["bureau_score"].to_numpy() < 580, b["delinq_12m"].to_numpy() > 1, b["enquiries_3m"].to_numpy() > 4, risk_score > 330],
            ["Low bureau score", "Recent delinquency", "High recent enquiries", "Policy score cutoff"],
            default="Affordability policy",
        )
    )
    approved_amount = np.where(approved, requested_amount * RNG.uniform(0.75, 1.0, n_apps), 0).round(-3).astype(int)
    interest_rate = np.where(approved, np.clip(12 + risk_score / 40 + RNG.normal(0, 1.1, n_apps), 11, 28), 0).round(2)

    return pd.DataFrame({
        "app_id": [f"APP{i:07d}" for i in range(1, n_apps + 1)],
        "customer_id": cust_sample["customer_id"],
        "application_date": pd.Series(app_day).dt.date.astype(str),
        "product": product,
        "requested_amount": requested_amount,
        "tenure_months": tenure_months,
        "channel": channel,
        "approval_status": np.where(approved, "Approved", "Rejected"),
        "rejection_reason": rejection_reason,
        "approved_amount": approved_amount,
        "interest_rate": interest_rate,
        "risk_score": risk_score,
    })


def generate_disbursements(applications: pd.DataFrame) -> pd.DataFrame:
    approved = applications[applications["approval_status"] == "Approved"].copy().reset_index(drop=True)
    disbursed_date = pd.to_datetime(approved["application_date"]) + pd.to_timedelta(RNG.integers(1, 7, len(approved)), unit="D")
    return pd.DataFrame({
        "loan_id": [f"LOAN{i:07d}" for i in range(1, len(approved) + 1)],
        "app_id": approved["app_id"],
        "customer_id": approved["customer_id"],
        "disbursed_date": disbursed_date.dt.date.astype(str),
        "principal": approved["approved_amount"],
        "tenure_months": approved["tenure_months"],
        "interest_rate": approved["interest_rate"],
        "risk_score": approved["risk_score"],
    })


def generate_repayments(disbursements: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in disbursements.itertuples(index=False):
        n_due = min(int(row.tenure_months), 8)
        monthly_due = float(row.principal) / float(row.tenure_months) * (1 + float(row.interest_rate) / 1200)
        start = pd.Timestamp(row.disbursed_date) + pd.offsets.MonthBegin(1)
        risk = float(row.risk_score)
        for mob in range(1, n_due + 1):
            due_date = start + pd.DateOffset(months=mob - 1)
            p_late = min(0.62, max(0.03, (risk - 180) / 520))
            is_late = RNG.random() < p_late
            if is_late:
                dpd = int(RNG.choice([5, 12, 25, 40, 65, 95], p=[0.25, 0.25, 0.18, 0.16, 0.10, 0.06]))
                paid_date = due_date + pd.to_timedelta(dpd, unit="D")
                paid_amount = monthly_due * RNG.uniform(0.70, 1.0)
            else:
                dpd = 0
                paid_date = due_date - pd.to_timedelta(int(RNG.integers(0, 3)), unit="D")
                paid_amount = monthly_due
            rows.append({
                "loan_id": row.loan_id,
                "due_date": due_date.date().isoformat(),
                "paid_date": paid_date.date().isoformat(),
                "mob": mob,
                "due_amount": round(monthly_due, 2),
                "paid_amount": round(float(paid_amount), 2),
                "dpd": dpd,
            })
    repayments = pd.DataFrame(rows)
    # Inject tiny controlled DQ issues: a null paid amount and duplicate row.
    if len(repayments) > 20:
        repayments.loc[5, "paid_amount"] = np.nan
        repayments = pd.concat([repayments, repayments.iloc[[10]]], ignore_index=True)
    return repayments


def generate_collection_calls(repayments: pd.DataFrame) -> pd.DataFrame:
    overdue = repayments[repayments["dpd"] > 0].copy().reset_index(drop=True)
    sample = overdue.sample(min(1800, len(overdue)), random_state=11) if len(overdue) else overdue
    outcomes = ["Connected", "No answer", "Promise to pay", "Wrong number", "Dispute raised"]
    return pd.DataFrame({
        "call_id": [f"CALL{i:07d}" for i in range(1, len(sample) + 1)],
        "loan_id": sample["loan_id"].to_numpy(),
        "call_date": (pd.to_datetime(sample["due_date"]) + pd.to_timedelta(RNG.integers(1, 25, len(sample)), unit="D")).dt.date.astype(str),
        "outcome": RNG.choice(outcomes, size=len(sample), p=[0.35, 0.28, 0.22, 0.08, 0.07]),
        "agent_notes": RNG.choice(["Customer requested extension", "Will pay this week", "Could not reach", "Income delayed", "Disputed amount"], size=len(sample)),
    })


def generate_all() -> dict[str, int]:
    ensure_dirs()
    customers = generate_customers()
    bureau = generate_bureau(customers)
    applications = generate_applications(customers, bureau)
    disbursements = generate_disbursements(applications)
    repayments = generate_repayments(disbursements)
    calls = generate_collection_calls(repayments)

    outputs = {
        "customers": customers,
        "bureau_features": bureau,
        "loan_applications": applications,
        "disbursements": disbursements,
        "repayments": repayments,
        "collection_calls": calls,
    }
    for name, df in outputs.items():
        df.to_csv(BRONZE_DIR / f"{name}.csv", index=False)
    return {name: len(df) for name, df in outputs.items()}


if __name__ == "__main__":
    counts = generate_all()
    print("Generated bronze datasets:")
    for table, count in counts.items():
        print(f"- {table}: {count:,} rows")
