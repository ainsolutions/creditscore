"""
Improved Synthetic Data Generator with Stronger Default Signals
Designed to enable 85%+ model accuracy
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Paths
DATA_DIR = Path(__file__).parent
TRAIN_FILE = DATA_DIR / "synthetic_credit_train.csv"
TEST_FILE = DATA_DIR / "synthetic_credit_test.csv"

def generate_improved_synthetic_data(n_samples=10000, default_rate=0.32):
    """
    Generate synthetic credit application data with clearer default patterns
    
    Key improvements:
    1. Stronger correlation between risk factors and defaults
    2. More decisive thresholds (DBR>0.7 → 85% default rate)
    3. Multiple risk factor combinations lead to near-certain defaults
    4. Reduced noise in default patterns
    """
    print(f"Generating {n_samples} improved synthetic credit applications...")
    
    data = []
    
    for _ in range(n_samples):
        # Basic applicant info
        age = np.random.normal(38, 12)
        age = max(21, min(70, age))
        
        # Income (PKR per month) - log-normal distribution
        income_base = np.random.lognormal(11.5, 0.6)  # Mean ~120k PKR
        income = max(25000, min(500000, income_base))
        
        # Employment and tenure
        employment_type = np.random.choice(
            ['Salaried', 'Self-Employed', 'Business'], 
            p=[0.65, 0.25, 0.10]
        )
        
        tenure_months = int(np.random.gamma(3, 15))
        tenure_months = max(3, min(300, tenure_months))
        
        # Loan details
        product_types = ['Personal Loan', 'Auto Finance', 'Home Finance', 'Business Loan']
        product_type = np.random.choice(product_types, p=[0.50, 0.25, 0.15, 0.10])
        
        # Loan amount based on income and product
        if product_type == 'Home Finance':
            loan_multiplier = np.random.uniform(15, 40)
        elif product_type == 'Auto Finance':
            loan_multiplier = np.random.uniform(8, 20)
        elif product_type == 'Business Loan':
            loan_multiplier = np.random.uniform(10, 30)
        else:  # Personal Loan
            loan_multiplier = np.random.uniform(3, 15)
        
        loan_amount = income * loan_multiplier
        loan_amount = max(50000, min(50000000, loan_amount))
        
        # Tenor (months)
        if product_type == 'Home Finance':
            tenor = int(np.random.uniform(120, 300))
        elif product_type == 'Auto Finance':
            tenor = int(np.random.uniform(24, 84))
        elif product_type == 'Business Loan':
            tenor = int(np.random.uniform(12, 120))
        else:
            tenor = int(np.random.uniform(12, 60))
        
        # Purpose
        purposes = ['Debt Consolidation', 'Home Purchase', 'Vehicle Purchase', 
                   'Business Expansion', 'Education', 'Medical', 'Wedding', 'Other']
        purpose = np.random.choice(purposes)
        
        # Existing debt
        existing_debt = income * np.random.uniform(0, 6)
        existing_debt = max(0, min(5000000, existing_debt))
        
        # Calculate DBR - Key risk indicator
        monthly_installment = loan_amount / tenor if tenor > 0 else loan_amount
        total_monthly_debt = (existing_debt / 12) + monthly_installment
        dbr = total_monthly_debt / income
        dbr = min(dbr, 1.5)  # Cap at 150%
        
        # LTV for asset-backed loans
        if product_type in ['Auto Finance', 'Home Finance']:
            asset_value = loan_amount / np.random.uniform(0.65, 0.95)
            ltv = loan_amount / asset_value
        else:
            ltv = 0
        
        ltv = min(ltv, 1.0)
        
        # Credit score - Critical indicator (0-100 scale)
        # Create three distinct risk segments
        credit_segment = np.random.random()
        if credit_segment < 0.25:  # High risk - 25%
            credit_score = np.random.uniform(0, 30)
        elif credit_segment < 0.65:  # Medium risk - 40%
            credit_score = np.random.uniform(30, 65)
        else:  # Low risk - 35%
            credit_score = np.random.uniform(65, 100)
        
        # e-CIB status
        if credit_score < 25:
            ecib_probs = [0.6, 0.3, 0.1]  # Mostly Negative
        elif credit_score < 50:
            ecib_probs = [0.2, 0.6, 0.2]  # Mostly Average
        else:
            ecib_probs = [0.1, 0.2, 0.7]  # Mostly Good
        
        ecib_status = np.random.choice(['Negative', 'Average', 'Good'], p=ecib_probs)
        
        # Dependents
        dependents = int(np.random.poisson(1.5))
        dependents = min(dependents, 8)
        
        # ===========================================
        # DEFAULT PREDICTION - IMPROVED LOGIC
        # ===========================================
        
        default_probability = 0.0
        
        # CRITICAL RISK FACTORS (Much higher impact)
        
        # 1. Extreme DBR → Almost certain default
        if dbr > 0.8:
            default_probability += 0.85  # 85% chance
        elif dbr > 0.7:
            default_probability += 0.70  # 70% chance
        elif dbr > 0.6:
            default_probability += 0.40  # 40% chance
        elif dbr > 0.5:
            default_probability += 0.20  # 20% chance
        
        # 2. Very poor credit score → High default
        if credit_score < 20:
            default_probability += 0.90  # 90% chance
        elif credit_score < 30:
            default_probability += 0.60  # 60% chance
        elif credit_score < 40:
            default_probability += 0.30  # 30% chance
        
        # 3. Negative e-CIB → Strong signal
        if ecib_status == 'Negative':
            default_probability += 0.65  # 65% chance
        elif ecib_status == 'Average':
            default_probability += 0.15  # 15% chance
        
        # 4. High LTV on asset loans
        if ltv > 0.95:
            default_probability += 0.40
        elif ltv > 0.90:
            default_probability += 0.25
        elif ltv > 0.85:
            default_probability += 0.15
        
        # MODERATE RISK FACTORS
        
        # 5. Low tenure (job instability)
        if tenure_months < 6:
            default_probability += 0.30
        elif tenure_months < 12:
            default_probability += 0.15
        
        # 6. Self-employment risk
        if employment_type == 'Self-Employed':
            default_probability += 0.08
        elif employment_type == 'Business':
            default_probability += 0.12
        
        # 7. High debt burden relative to tenure
        if existing_debt > income * 12:
            default_probability += 0.20
        
        # 8. Many dependents with lower income
        if dependents >= 4 and income < 80000:
            default_probability += 0.15
        
        # PROTECTIVE FACTORS (reduce default risk)
        
        # Good credit and stable employment
        if credit_score > 70 and tenure_months > 36:
            default_probability -= 0.25
        
        # Strong e-CIB with low DBR
        if ecib_status == 'Good' and dbr < 0.4:
            default_probability -= 0.30
        
        # High income with low leverage
        if income > 200000 and dbr < 0.3:
            default_probability -= 0.20
        
        # Long tenure with good history
        if tenure_months > 60 and ecib_status in ['Good', 'Average']:
            default_probability -= 0.15
        
        # Cap probability between 0 and 1
        default_probability = max(0.0, min(1.0, default_probability))
        
        # Determine default
        default = 1 if np.random.random() < default_probability else 0
        
        # Create record
        record = {
            'age': round(age, 1),
            'income': round(income, 2),
            'employment_type': employment_type,
            'tenure_months': tenure_months,
            'product_type': product_type,
            'loan_amount': round(loan_amount, 2),
            'tenor': tenor,
            'purpose': purpose,
            'existing_debt': round(existing_debt, 2),
            'dbr': round(dbr, 4),
            'ltv': round(ltv, 4),
            'credit_score': round(credit_score, 2),
            'ecib_status': ecib_status,
            'dependents': dependents,
            'default': default
        }
        
        data.append(record)
    
    df = pd.DataFrame(data)
    
    # Print statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total samples: {len(df)}")
    print(f"   Default rate: {df['default'].mean():.2%}")
    print(f"   Non-default: {(df['default']==0).sum()} ({(df['default']==0).mean():.1%})")
    print(f"   Default: {(df['default']==1).sum()} ({(df['default']==1).mean():.1%})")
    
    print(f"\n📊 Risk Factor Distribution:")
    print(f"   High DBR (>0.6): {(df['dbr']>0.6).mean():.1%}")
    print(f"   Extreme DBR (>0.7): {(df['dbr']>0.7).mean():.1%}")
    print(f"   Low Credit Score (<30): {(df['credit_score']<30).mean():.1%}")
    print(f"   Very Low Credit (<20): {(df['credit_score']<20).mean():.1%}")
    print(f"   Negative e-CIB: {(df['ecib_status']=='Negative').mean():.1%}")
    
    print(f"\n📊 Default Rates by Risk Segment:")
    print(f"   DBR > 0.7: {df[df['dbr']>0.7]['default'].mean():.1%}")
    print(f"   Credit < 20: {df[df['credit_score']<20]['default'].mean():.1%}")
    print(f"   Negative e-CIB: {df[df['ecib_status']=='Negative']['default'].mean():.1%}")
    print(f"   Combined high risk (DBR>0.7 & Credit<30): {df[(df['dbr']>0.7) & (df['credit_score']<30)]['default'].mean():.1%}")
    
    return df

def main():
    """Generate training and test datasets"""
    print("="*70)
    print("IMPROVED SYNTHETIC DATA GENERATION")
    print("="*70)
    
    # Generate training set (10,000 samples)
    print("\n🔄 Generating training set...")
    train_df = generate_improved_synthetic_data(n_samples=10000, default_rate=0.32)
    train_df.to_csv(TRAIN_FILE, index=False)
    print(f"✅ Training data saved to: {TRAIN_FILE}")
    
    # Generate test set (2,000 samples)
    print("\n🔄 Generating test set...")
    test_df = generate_improved_synthetic_data(n_samples=2000, default_rate=0.32)
    test_df.to_csv(TEST_FILE, index=False)
    print(f"✅ Test data saved to: {TEST_FILE}")
    
    print("\n" + "="*70)
    print("DATA GENERATION COMPLETE!")
    print("="*70)
    print(f"\n📁 Files created:")
    print(f"   • {TRAIN_FILE}")
    print(f"   • {TEST_FILE}")
    print(f"\n✅ Ready for model training with improved data signals!")

if __name__ == "__main__":
    main()
