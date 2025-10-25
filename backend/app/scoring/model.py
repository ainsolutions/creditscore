from typing import Dict, Any
from app.schemas import ScoreRequest


def engineer_features(request: ScoreRequest) -> Dict[str, float]:
    a = request.applicant
    l = request.loan
    features: Dict[str, float] = {
        "age_years": float(a.age_years),
        "income_monthly": float(a.monthly_income),
        "dependents": float(a.dependents or 0),
        "existing_debt_monthly": float(a.existing_monthly_debt_payments),
        "tenor_months": float(l.tenor_months),
        "amount": float(l.amount),
        "down_payment": float(l.down_payment or 0),
        "ecib_negative": 1.0 if a.e_cib_negative else 0.0,
    }
    return features


def simple_pd_model(features: Dict[str, float], product_type: str, approval_threshold: float) -> Dict[str, Any]:
    # Baseline PD influenced by income, debt and age. This is a placeholder model.
    income = max(0.0, features["income_monthly"]) + 1e-6
    debt = max(0.0, features["existing_debt_monthly"]) + 1e-6
    dti = debt / income
    age = features["age_years"]
    ecib_penalty = 0.25 if features["ecib_negative"] > 0 else 0.0

    base = 0.15 + 0.6 * dti - 0.001 * max(0.0, age - 25) + ecib_penalty
    # Product risk adjustment
    product_adj = {
        "personal": 0.05,
        "cash": 0.08,
        "car": -0.03,
        "housing": -0.05,
    }[product_type]
    pd = min(max(base + product_adj, 0.01), 0.95)

    reasons = []
    if dti > 0.4:
        reasons.append(f"High DTI {dti:.2f}")
    if features["ecib_negative"] > 0:
        reasons.append("Negative e-CIB penalty applied")
    if pd < (1 - approval_threshold) * 0.5:
        reasons.append("Low model risk")

    return {"probability": float(pd), "reasons": reasons}


def score_application(request: ScoreRequest, settings) -> Dict[str, Any]:
    features = engineer_features(request)
    model_out = simple_pd_model(features, request.product_type, settings.approval_threshold)
    return {"probability": model_out["probability"], "reasons": model_out["reasons"], "features": features}


