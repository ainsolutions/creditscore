"""
Credit scoring model that uses a trained LightGBM classifier (85.05% accuracy).

This module:
1. Loads a pre-trained LightGBM model and encoders on startup
2. Engineers features matching the advanced training pipeline
3. Returns probability of default predictions

Model: LightGBM (500 estimators, max_depth=8, learning_rate=0.05)
Accuracy: 85.05%
ROC AUC: 0.9392
Trained with SMOTE class balancing
"""

from typing import Dict, Any, Optional
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from app.schemas import ScoreRequest

# Global model artifacts (loaded once at startup)
_model = None
_encoders = None
_model_loaded = False


def load_model():
    """Load the trained LightGBM model and encoders from disk."""
    global _model, _encoders, _model_loaded
    
    if _model_loaded:
        return
    
    try:
        models_dir = Path(__file__).parent.parent / "models"
        model_path = models_dir / "credit_model.joblib"
        encoders_path = models_dir / "credit_model_encoders.joblib"
        
        if not model_path.exists():
            print(f"WARNING: Trained model not found at {model_path}")
            print("Using fallback placeholder model. Run train_model_advanced.py to create the model.")
            _model_loaded = True
            return
        
        _model = joblib.load(model_path)
        _encoders = joblib.load(encoders_path)
        _model_loaded = True
        print(f"✓ Loaded trained LightGBM model (85.05% accuracy) from {model_path}")
        
    except Exception as e:
        print(f"ERROR loading model: {e}")
        print("Falling back to placeholder model.")
        _model_loaded = True


def engineer_features_for_model(request: ScoreRequest) -> pd.DataFrame:
    """
    Engineer features matching the advanced training pipeline (85.05% accuracy model).
    
    Returns a DataFrame with a single row containing all 28 required features.
    """
    a = request.applicant
    l = request.loan
    
    # Basic features
    age = float(a.age_years)
    income = float(a.monthly_income)
    loan_amount = float(l.amount)
    tenor = float(l.tenor_months)
    existing_debt = float(a.existing_monthly_debt_payments) * 12  # Convert to annual
    dependents = int(a.dependents if a.dependents is not None else 2)  # Default 2 if not provided
    tenure_months = int(a.months_at_job if a.months_at_job is not None else 24)  # Default 24 if not provided
    
    # Calculate DBR
    monthly_installment = loan_amount / tenor if tenor > 0 else loan_amount
    dbr = (existing_debt/12 + monthly_installment) / (income + 1e-6)
    dbr = min(dbr, 1.5)
    
    # LTV for secured loans
    if request.product_type == "housing":
        down_payment = float(l.down_payment or 0)
        property_value = loan_amount + down_payment
        ltv = loan_amount / (property_value + 1e-6) if property_value > 0 else 0.0
    elif request.product_type == "car":
        down_payment = float(l.down_payment or 0)
        car_value = loan_amount + down_payment
        ltv = loan_amount / (car_value + 1e-6) if car_value > 0 else 0.0
    else:
        ltv = 0.0
    ltv = min(ltv, 1.0)
    
    # Credit score proxy
    if a.e_cib_negative:
        credit_score = 20.0
    else:
        credit_score = min(100.0, 50 + (income / 2000) + (age - 21) / 2)
    
    # Map product type to training categories
    product_type_map = {
        "personal": "Personal Loan",
        "cash": "Personal Loan",
        "car": "Auto Finance",
        "housing": "Home Finance"
    }
    product_type = product_type_map.get(request.product_type, "Personal Loan")
    
    # Map purpose
    purpose = "Other"  # Default
    
    # e-CIB status
    ecib_status = "Negative" if a.e_cib_negative else "Good"
    
    # Employment type
    employment_type = "Salaried"  # Default
    
    # Risk indicators
    dbr_risk = 1 if dbr > 0.6 else 0
    ltv_risk = 1 if ltv > 0.85 else 0
    credit_risk = 1 if credit_score < 30 else 0
    
    # Interaction features
    credit_dbr_interaction = credit_score * dbr
    risk_concentration = dbr_risk + ltv_risk + credit_risk
    payment_to_income = monthly_installment / (income + 1e-6)
    total_debt_to_income = (existing_debt + loan_amount) / (income * 12 + 1e-6)
    loan_per_tenor_year = loan_amount / ((tenor / 12) + 1e-6)
    income_stability = tenure_months * income
    
    # Advanced features
    high_risk_score = (
        (3 if dbr > 0.7 else 0) +
        (3 if credit_score < 25 else 0) +
        (3 if ecib_status == 'Negative' else 0) +
        (2 if ltv > 0.9 else 0) +
        (2 if tenure_months < 12 else 0)
    )
    
    debt_capacity = income * (0.6 - dbr)
    age_income_ratio = age / ((income / 10000) + 1e-6)
    loan_to_credit_score = loan_amount / (credit_score + 1)
    income_per_dependent = income / (dependents + 1)
    
    # Create DataFrame matching training feature order exactly
    # Feature order must match train_model_advanced.py:
    feature_cols = [
        'age', 'income', 'loan_amount', 'tenor', 'dbr', 'ltv',
        'credit_score', 'existing_debt', 'tenure_months', 'dependents',
        'dbr_risk', 'ltv_risk', 'credit_risk',
        'credit_dbr_interaction', 'risk_concentration', 'payment_to_income',
        'total_debt_to_income', 'loan_per_tenor_year', 'income_stability',
        'high_risk_score', 'debt_capacity', 'age_income_ratio',
        'loan_to_credit_score', 'income_per_dependent',
        'product_type_encoded', 'purpose_encoded', 'ecib_status_encoded',
        'employment_type_encoded'
    ]
    
    data = {
        'age': age,
        'income': income,
        'loan_amount': loan_amount,
        'tenor': tenor,
        'dbr': dbr,
        'ltv': ltv,
        'credit_score': credit_score,
        'existing_debt': existing_debt,
        'tenure_months': tenure_months,
        'dependents': dependents,
        'dbr_risk': dbr_risk,
        'ltv_risk': ltv_risk,
        'credit_risk': credit_risk,
        'credit_dbr_interaction': credit_dbr_interaction,
        'risk_concentration': risk_concentration,
        'payment_to_income': payment_to_income,
        'total_debt_to_income': total_debt_to_income,
        'loan_per_tenor_year': loan_per_tenor_year,
        'income_stability': income_stability,
        'high_risk_score': high_risk_score,
        'debt_capacity': debt_capacity,
        'age_income_ratio': age_income_ratio,
        'loan_to_credit_score': loan_to_credit_score,
        'income_per_dependent': income_per_dependent,
        'product_type': product_type,
        'purpose': purpose,
        'ecib_status': ecib_status,
        'employment_type': employment_type,
    }
    
    df = pd.DataFrame([data])
    
    # Encode categorical features using loaded encoders
    if _encoders is not None:
        try:
            df['product_type_encoded'] = _encoders['product_type'].transform(df['product_type'])
            df['purpose_encoded'] = _encoders['purpose'].transform(df['purpose'])
            df['ecib_status_encoded'] = _encoders['ecib_status'].transform(df['ecib_status'])
            df['employment_type_encoded'] = _encoders['employment_type'].transform(df['employment_type'])
        except Exception as e:
            # Fallback encoding
            print(f"Warning: Encoding error {e}, using fallback")
            df['product_type_encoded'] = 0
            df['purpose_encoded'] = 0
            df['ecib_status_encoded'] = 0 if ecib_status == 'Negative' else 1
            df['employment_type_encoded'] = 0
    else:
        # Default encodings
        df['product_type_encoded'] = 0
        df['purpose_encoded'] = 0
        df['ecib_status_encoded'] = 0 if ecib_status == 'Negative' else 1
        df['employment_type_encoded'] = 0
    
    return df[feature_cols]


def prepare_features_for_inference(df: pd.DataFrame) -> np.ndarray:
    """
    Prepare features for model inference.
    Features are already engineered and encoded in engineer_features_for_model().
    This function just converts to numpy array in the correct order.
    """
    # All 28 features are already in the DataFrame in the correct order
    return df.values


def predict_with_trained_model(request: ScoreRequest) -> Dict[str, Any]:
    """
    Use the trained Random Forest model to predict probability of default.
    """
    # Engineer features
    df = engineer_features_for_model(request)
    
    # Prepare features for model
    X = prepare_features_for_inference(df)
    
    # Predict probability
    probability = float(_model.predict_proba(X)[0, 1])
    
    # Generate explanation reasons based on features
    reasons = []
    
    dbr = df["dbr"].values[0]
    credit_score = df["credit_score"].values[0]
    ltv = df["ltv"].values[0]
    debt_capacity = df["debt_capacity"].values[0]
    risk_concentration = df["risk_concentration"].values[0]
    
    if dbr > 0.60:
        reasons.append(f"Very high debt burden ratio: {dbr:.2%}")
    elif dbr > 0.50:
        reasons.append(f"High debt burden ratio: {dbr:.2%}")
    elif dbr > 0.40:
        reasons.append(f"Moderate debt burden ratio: {dbr:.2%}")
    
    if credit_score < 30:
        reasons.append(f"Very poor credit score: {credit_score:.0f}")
    elif credit_score < 50:
        reasons.append(f"Poor credit score: {credit_score:.0f}")
    elif credit_score < 70:
        reasons.append(f"Fair credit score: {credit_score:.0f}")
    
    if ltv > 0.85:
        reasons.append(f"High loan-to-value ratio: {ltv:.2%}")
    elif ltv > 0.75:
        reasons.append(f"Moderate loan-to-value ratio: {ltv:.2%}")
    
    if debt_capacity < 0:
        reasons.append("Negative remaining debt capacity")
    elif debt_capacity < 10000:
        reasons.append("Low remaining debt capacity")
    
    if risk_concentration >= 2:
        reasons.append(f"Multiple risk factors present (score: {risk_concentration:.0f})")
    
    if probability < 0.3:
        reasons.append("Low predicted default risk")
    elif probability < 0.5:
        reasons.append("Moderate predicted default risk")
    else:
        reasons.append("High predicted default risk")
    
    # Extract key features for audit
    features = {
        "age": float(df["age"].values[0]),
        "income": float(df["income"].values[0]),
        "loan_amount": float(df["loan_amount"].values[0]),
        "dbr": float(dbr),
        "ltv": float(ltv),
        "credit_score": float(credit_score),
        "tenor": float(df["tenor"].values[0]),
        "debt_capacity": float(debt_capacity),
        "risk_concentration": float(risk_concentration),
    }
    
    return {
        "probability": probability,
        "reasons": reasons,
        "features": features,
    }


def fallback_placeholder_model(request: ScoreRequest, approval_threshold: float) -> Dict[str, Any]:
    """
    Fallback placeholder model (used if trained model is not available).
    """
    a = request.applicant
    l = request.loan
    
    income = max(0.0, float(a.monthly_income)) + 1e-6
    debt = max(0.0, float(a.existing_monthly_debt_payments)) + 1e-6
    dti = debt / income
    age = float(a.age_years)
    ecib_penalty = 0.25 if a.e_cib_negative else 0.0

    base = 0.15 + 0.6 * dti - 0.001 * max(0.0, age - 25) + ecib_penalty
    product_adj = {
        "personal": 0.05,
        "cash": 0.08,
        "car": -0.03,
        "housing": -0.05,
    }[request.product_type]
    pd = min(max(base + product_adj, 0.01), 0.95)

    reasons = []
    if dti > 0.4:
        reasons.append(f"High DTI {dti:.2f}")
    if a.e_cib_negative:
        reasons.append("Negative e-CIB penalty applied")
    if pd < (1 - approval_threshold) * 0.5:
        reasons.append("Low model risk")

    features = {
        "age_years": age,
        "income_monthly": income,
        "existing_debt_monthly": debt,
        "dti": dti,
    }

    return {"probability": float(pd), "reasons": reasons, "features": features}


def score_application(request: ScoreRequest, settings) -> Dict[str, Any]:
    """
    Main scoring function called by the API.
    
    Uses trained model if available, otherwise falls back to placeholder.
    """
    # Ensure model is loaded
    load_model()
    
    # Use trained model if available
    if _model is not None and _encoders is not None:
        return predict_with_trained_model(request)
    else:
        # Fallback to placeholder
        return fallback_placeholder_model(request, settings.approval_threshold)


