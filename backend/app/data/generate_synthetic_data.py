"""
Generate synthetic credit application data for training a credit scoring model.

This script creates realistic loan applications across product types (personal, housing, cash, car)
with features that align to SBP prudential rules and common credit risk factors.

Features generated:
- Applicant age
- Monthly income
- Loan amount requested
- Debt burden ratio (DBR/DTI)
- Loan-to-value ratio (LTV) for secured loans
- Credit history score (proxy for e-CIB)
- Tenor (loan term in months)
- Product type
- Employment type
- Target: default (0=no default, 1=default)

The model will learn patterns like:
- High DBR increases default risk
- Negative credit history increases default risk
- Higher income reduces default risk
- Higher LTV for secured loans increases risk
- Longer tenor may increase risk
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
import random


def generate_synthetic_credit_data(n_samples: int = 10000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic credit application data.
    
    Args:
        n_samples: Number of samples to generate
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with features and target variable
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Product types and their typical characteristics
    product_types = ["personal", "housing", "cash", "car"]
    product_weights = [0.35, 0.25, 0.20, 0.20]  # Distribution
    
    data = []
    
    for _ in range(n_samples):
        product_type = random.choices(product_types, weights=product_weights, k=1)[0]
        
        # Age: 21-65, skewed towards 30-45
        age = int(np.random.beta(2, 2) * 44 + 21)
        
        # Monthly income: PKR 20k-500k, log-normal distribution
        income = int(np.random.lognormal(11.5, 0.7))  # Mean ~100k, varied
        income = max(20000, min(income, 500000))
        
        # Loan amount based on product type
        if product_type == "personal":
            loan_amount = int(np.random.uniform(50000, min(income * 48, 2000000)))
        elif product_type == "housing":
            loan_amount = int(np.random.uniform(500000, min(income * 240, 15000000)))
        elif product_type == "cash":
            loan_amount = int(np.random.uniform(20000, min(income * 24, 500000)))
        else:  # car
            loan_amount = int(np.random.uniform(500000, min(income * 60, 5000000)))
        
        # Existing monthly debt obligations
        existing_debt = int(np.random.uniform(0, income * 0.5))
        
        # Calculate tenor based on product type
        if product_type == "personal":
            tenor = random.randint(12, 60)
        elif product_type == "housing":
            tenor = random.randint(60, 360)
        elif product_type == "cash":
            tenor = random.randint(6, 36)
        else:  # car
            tenor = random.randint(12, 84)
        
        # Monthly installment (simple calculation)
        monthly_rate = 0.15 / 12  # Approximate 15% annual rate
        if tenor > 0:
            monthly_installment = loan_amount * (monthly_rate * (1 + monthly_rate)**tenor) / ((1 + monthly_rate)**tenor - 1)
        else:
            monthly_installment = loan_amount
        
        # Debt Burden Ratio (DBR/DTI)
        dbr = (existing_debt + monthly_installment) / income
        
        # LTV for secured loans (housing and car)
        if product_type == "housing":
            property_value = loan_amount / np.random.uniform(0.50, 0.90)
            ltv = loan_amount / property_value
        elif product_type == "car":
            car_value = loan_amount / np.random.uniform(0.60, 0.90)
            ltv = loan_amount / car_value
        else:
            ltv = 0.0  # Not applicable for unsecured
        
        # Credit history score (0-100, higher is better)
        # Simulate e-CIB score proxy
        credit_score = int(np.random.beta(5, 2) * 100)  # Skewed towards good scores
        
        # Employment type
        employment_types = ["salaried", "self_employed", "business"]
        employment_type = random.choices(employment_types, weights=[0.6, 0.25, 0.15], k=1)[0]
        
        # --- Target variable: default (0 or 1) ---
        # Calculate default probability based on risk factors
        default_prob = 0.05  # Base rate
        
        # Age risk: very young or old increases risk slightly
        if age < 25 or age > 60:
            default_prob += 0.03
        
        # Income risk: lower income increases risk
        if income < 40000:
            default_prob += 0.08
        elif income < 60000:
            default_prob += 0.04
        
        # DBR risk: higher DBR significantly increases risk
        if dbr > 0.60:
            default_prob += 0.25
        elif dbr > 0.50:
            default_prob += 0.15
        elif dbr > 0.40:
            default_prob += 0.08
        
        # LTV risk for secured loans
        if ltv > 0.85:
            default_prob += 0.10
        elif ltv > 0.75:
            default_prob += 0.05
        
        # Credit score risk: low score dramatically increases risk
        if credit_score < 30:
            default_prob += 0.40
        elif credit_score < 50:
            default_prob += 0.20
        elif credit_score < 70:
            default_prob += 0.10
        
        # Tenor risk: very long tenors slightly increase risk
        if tenor > 240:
            default_prob += 0.05
        elif tenor > 120:
            default_prob += 0.02
        
        # Employment type risk
        if employment_type == "self_employed":
            default_prob += 0.03
        elif employment_type == "business":
            default_prob += 0.05
        
        # Cap probability
        default_prob = min(default_prob, 0.85)
        
        # Generate binary outcome
        default = 1 if np.random.random() < default_prob else 0
        
        data.append({
            "age": age,
            "monthly_income": income,
            "loan_amount": loan_amount,
            "existing_debt": existing_debt,
            "monthly_installment": monthly_installment,
            "dbr": round(dbr, 4),
            "ltv": round(ltv, 4),
            "credit_score": credit_score,
            "tenor": tenor,
            "product_type": product_type,
            "employment_type": employment_type,
            "default": default,
        })
    
    df = pd.DataFrame(data)
    return df


def main():
    """Generate and save synthetic data."""
    print("Generating synthetic credit data...")
    
    # Generate training data
    df_train = generate_synthetic_credit_data(n_samples=10000, random_seed=42)
    
    # Generate test data
    df_test = generate_synthetic_credit_data(n_samples=2000, random_seed=123)
    
    # Save to CSV
    train_path = "synthetic_credit_train.csv"
    test_path = "synthetic_credit_test.csv"
    
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    
    print(f"✓ Training data saved: {train_path} ({len(df_train)} samples)")
    print(f"✓ Test data saved: {test_path} ({len(df_test)} samples)")
    print(f"\nDefault rate (train): {df_train['default'].mean():.2%}")
    print(f"Default rate (test): {df_test['default'].mean():.2%}")
    print("\nFeature summary:")
    print(df_train.describe())
    print("\nProduct type distribution:")
    print(df_train["product_type"].value_counts())


if __name__ == "__main__":
    main()
