from typing import Dict, Any, List
from app.schemas import ScoreRequest


# Simplified encoding of SBP Prudential Regulations for Consumer Financing.
# NOTE: Thresholds should be validated by compliance and can be adjusted via config.


def evaluate_sbp_rules(request: ScoreRequest) -> Dict[str, Any]:
    flags: Dict[str, bool] = {}
    reasons: List[str] = []
    ok = True

    a = request.applicant
    l = request.loan
    product = request.product_type

    # Age limits (typical ranges)
    min_age = 21 if product in ("personal", "cash", "car") else 25
    max_age = 65 if product in ("personal", "cash", "car") else 70
    flags["min_age"] = a.age_years >= min_age
    flags["max_age"] = a.age_years <= max_age

    if not flags["min_age"]:
        ok = False
        reasons.append(f"Applicant age below minimum for {product} (min {min_age})")
    if not flags["max_age"]:
        ok = False
        reasons.append(f"Applicant age above maximum for {product} (max {max_age})")

    # e-CIB negative history should decline
    flags["no_negative_ecib"] = not bool(a.e_cib_negative)
    if not flags["no_negative_ecib"]:
        ok = False
        reasons.append("Negative e-CIB history")

    # Debt Burden Ratio (DBR) / DTI limits
    monthly_income = max(0.0, a.monthly_income)
    existing_debt = max(0.0, a.existing_monthly_debt_payments)
    # Estimate EMI using a flat proxy if property_value or vehicle_value missing
    proposed_emi = estimate_monthly_installment(amount=l.amount, tenor_months=l.tenor_months, annual_rate=0.28)
    dbr = (existing_debt + proposed_emi) / monthly_income if monthly_income > 0 else 1.0

    # SBP often requires DBR <= 0.4-0.5 for unsecured; higher tolerances for secured
    if product in ("personal", "cash"):
        max_dbr = 0.4
    elif product == "car":
        max_dbr = 0.5
    else:  # housing
        max_dbr = 0.6
    flags["dbr_within_limit"] = dbr <= max_dbr
    if not flags["dbr_within_limit"]:
        ok = False
        reasons.append(f"Debt Burden Ratio {dbr:.2f} exceeds limit {max_dbr:.2f}")

    # LTV and Down Payment rules for secured products
    if product == "car" and l.vehicle_value:
        ltv = l.amount / l.vehicle_value
        min_down = 0.15  # 15% minimum down payment
        flags["min_down_payment"] = (l.down_payment or 0) >= l.vehicle_value * min_down
        flags["ltv_within_limit"] = ltv <= 0.85
        if not flags["min_down_payment"]:
            ok = False
            reasons.append("Down payment below 15% for auto financing")
        if not flags["ltv_within_limit"]:
            ok = False
            reasons.append("Auto LTV exceeds 85% limit")

    if product == "housing" and l.property_value:
        ltv = l.amount / l.property_value
        # Housing LTV caps vary; using 85% as a typical cap for demonstration
        flags["ltv_within_limit"] = ltv <= 0.85
        if not flags["ltv_within_limit"]:
            ok = False
            reasons.append("Housing LTV exceeds 85% limit")

    # Tenor limits (illustrative)
    tenor = l.tenor_months
    tenor_limits = {
        "personal": 60,
        "cash": 36,
        "car": 84,
        "housing": 360,
    }
    flags["tenor_within_limit"] = tenor <= tenor_limits[product]
    if not flags["tenor_within_limit"]:
        ok = False
        reasons.append(f"Tenor exceeds {tenor_limits[product]} months limit for {product}")

    return {"pass": ok, "flags": flags, "reasons": reasons, "metrics": {"dbr": dbr, "emi": proposed_emi}}


def estimate_monthly_installment(amount: float, tenor_months: int, annual_rate: float) -> float:
    # Simple amortization formula M = P * r * (1+r)^n / ((1+r)^n - 1)
    if tenor_months <= 0:
        return amount
    monthly_rate = annual_rate / 12.0
    if monthly_rate <= 0:
        return amount / tenor_months
    pow_term = (1 + monthly_rate) ** tenor_months
    return amount * monthly_rate * pow_term / (pow_term - 1)


