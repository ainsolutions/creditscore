"""
Advanced Model Training Script with XGBoost, LightGBM, and Ensemble
Targets 85%+ accuracy through improved data signals and algorithms
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path(__file__).parent
MODEL_DIR = DATA_DIR.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

def engineer_features(df):
    """
    Enhanced feature engineering with stronger predictive signals
    """
    # Original features
    df['dbr_risk'] = (df['dbr'] > 0.6).astype(int)
    df['ltv_risk'] = (df['ltv'] > 0.85).astype(int)
    df['credit_risk'] = (df['credit_score'] < 30).astype(int)
    
    # Interaction features (strongly predictive)
    df['credit_dbr_interaction'] = df['credit_score'] * df['dbr']
    df['risk_concentration'] = df['dbr_risk'] + df['ltv_risk'] + df['credit_risk']
    df['payment_to_income'] = df['loan_amount'] / (df['tenor'] * df['income'] + 1)
    df['total_debt_to_income'] = (df['loan_amount'] + df['existing_debt']) / df['income']
    df['loan_per_tenor_year'] = df['loan_amount'] / (df['tenor'] / 12)
    df['income_stability'] = df['tenure_months'] * df['income']
    
    # New advanced features
    df['high_risk_score'] = (
        (df['dbr'] > 0.7) * 3 +
        (df['credit_score'] < 25) * 3 +
        (df['ecib_status'] == 'Negative') * 3 +
        (df['ltv'] > 0.9) * 2 +
        (df['tenure_months'] < 12) * 2
    )
    
    df['debt_capacity'] = df['income'] * (0.6 - df['dbr'])  # Remaining debt capacity
    df['age_income_ratio'] = df['age'] / (df['income'] / 10000)
    df['loan_to_credit_score'] = df['loan_amount'] / (df['credit_score'] + 1)
    df['income_per_dependent'] = df['income'] / (df['dependents'] + 1)
    
    # Categorical encoding
    product_encoder = LabelEncoder()
    purpose_encoder = LabelEncoder()
    ecib_encoder = LabelEncoder()
    employment_encoder = LabelEncoder()
    
    df['product_type_encoded'] = product_encoder.fit_transform(df['product_type'])
    df['purpose_encoded'] = purpose_encoder.fit_transform(df['purpose'])
    df['ecib_status_encoded'] = ecib_encoder.fit_transform(df['ecib_status'])
    df['employment_type_encoded'] = employment_encoder.fit_transform(df['employment_type'])
    
    # Save encoders
    encoders = {
        'product_type': product_encoder,
        'purpose': purpose_encoder,
        'ecib_status': ecib_encoder,
        'employment_type': employment_encoder
    }
    
    return df, encoders

def load_and_prep_data():
    """Load and prepare training and test data"""
    print("="*70)
    print("ADVANCED MODEL TRAINING - TARGETING 85%+ ACCURACY")
    print("="*70)
    
    # Load data
    train_path = DATA_DIR / "synthetic_credit_train.csv"
    test_path = DATA_DIR / "synthetic_credit_test.csv"
    
    print(f"\n📂 Loading training data from: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"📂 Loading test data from: {test_path}")
    test_df = pd.read_csv(test_path)
    
    print(f"\n📊 Training set: {train_df.shape[0]} samples")
    print(f"📊 Test set: {test_df.shape[0]} samples")
    print(f"📊 Default rate (train): {train_df['default'].mean():.2%}")
    print(f"📊 Default rate (test): {test_df['default'].mean():.2%}")
    
    # Engineer features
    print("\n🔧 Engineering features...")
    train_df, encoders = engineer_features(train_df)
    test_df, _ = engineer_features(test_df)
    
    # Feature columns
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
    
    X_train = train_df[feature_cols]
    y_train = train_df['default']
    X_test = test_df[feature_cols]
    y_test = test_df['default']
    
    print(f"✅ Feature engineering complete. Total features: {len(feature_cols)}")
    
    return X_train, y_train, X_test, y_test, feature_cols, encoders

def train_xgboost(X_train, y_train):
    """Train XGBoost model with hyperparameter tuning"""
    print("\n" + "="*70)
    print("TRAINING XGBOOST MODEL")
    print("="*70)
    
    xgb_model = XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=2,  # Handle class imbalance
        random_state=42,
        eval_metric='logloss'
    )
    
    print("🔄 Training XGBoost (500 estimators, max_depth=8, lr=0.05)...")
    xgb_model.fit(X_train, y_train)
    print("✅ XGBoost training complete!")
    
    return xgb_model

def train_lightgbm(X_train, y_train):
    """Train LightGBM model"""
    print("\n" + "="*70)
    print("TRAINING LIGHTGBM MODEL")
    print("="*70)
    
    lgb_model = LGBMClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=2,
        random_state=42,
        verbose=-1
    )
    
    print("🔄 Training LightGBM (500 estimators, max_depth=8, lr=0.05)...")
    lgb_model.fit(X_train, y_train)
    print("✅ LightGBM training complete!")
    
    return lgb_model

def train_random_forest(X_train, y_train):
    """Train Random Forest as baseline"""
    print("\n" + "="*70)
    print("TRAINING RANDOM FOREST (BASELINE)")
    print("="*70)
    
    rf_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    print("🔄 Training Random Forest (500 estimators, max_depth=20)...")
    rf_model.fit(X_train, y_train)
    print("✅ Random Forest training complete!")
    
    return rf_model

def create_ensemble(rf_model, xgb_model, lgb_model):
    """Create voting ensemble of all three models"""
    print("\n" + "="*70)
    print("CREATING ENSEMBLE MODEL")
    print("="*70)
    
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('xgb', xgb_model),
            ('lgb', lgb_model)
        ],
        voting='soft',
        weights=[1, 2, 2]  # Give more weight to gradient boosting models
    )
    
    print("✅ Ensemble model created (RF + XGBoost + LightGBM)")
    return ensemble

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Comprehensive model evaluation"""
    print(f"\n{'='*70}")
    print(f"EVALUATING {model_name.upper()}")
    print(f"{'='*70}")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    print(f"   ROC AUC:   {roc_auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📋 CONFUSION MATRIX:")
    print(f"   True Negatives:  {cm[0][0]}")
    print(f"   False Positives: {cm[0][1]}")
    print(f"   False Negatives: {cm[1][0]}")
    print(f"   True Positives:  {cm[1][1]}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist()
    }

def save_model(model, encoders, feature_cols, metrics, model_name="credit_model"):
    """Save model and metadata"""
    print(f"\n💾 Saving {model_name}...")
    
    # Save model
    model_path = MODEL_DIR / f"{model_name}.joblib"
    joblib.dump(model, model_path)
    print(f"   ✅ Model saved to: {model_path}")
    
    # Save encoders
    encoders_path = MODEL_DIR / f"{model_name}_encoders.joblib"
    joblib.dump(encoders, encoders_path)
    print(f"   ✅ Encoders saved to: {encoders_path}")
    
    # Save metadata
    metadata = {
        'model_type': model.__class__.__name__,
        'features': feature_cols,
        'num_features': len(feature_cols),
        'metrics': metrics,
        'training_date': pd.Timestamp.now().isoformat()
    }
    
    metadata_path = MODEL_DIR / f"{model_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✅ Metadata saved to: {metadata_path}")

def main():
    """Main training pipeline"""
    # Load and prepare data
    X_train, y_train, X_test, y_test, feature_cols, encoders = load_and_prep_data()
    
    # Apply SMOTE for class balancing
    print("\n" + "="*70)
    print("APPLYING SMOTE FOR CLASS BALANCING")
    print("="*70)
    print(f"📊 Before SMOTE - Class 0: {(y_train==0).sum()}, Class 1: {(y_train==1).sum()}")
    
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    print(f"📊 After SMOTE  - Class 0: {(y_train_smote==0).sum()}, Class 1: {(y_train_smote==1).sum()}")
    print("✅ SMOTE applied successfully!")
    
    # Train individual models
    rf_model = train_random_forest(X_train_smote, y_train_smote)
    xgb_model = train_xgboost(X_train_smote, y_train_smote)
    lgb_model = train_lightgbm(X_train_smote, y_train_smote)
    
    # Evaluate individual models
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    lgb_metrics = evaluate_model(lgb_model, X_test, y_test, "LightGBM")
    
    # Create and fit ensemble on SMOTE data
    print("\n" + "="*70)
    print("FITTING ENSEMBLE MODEL")
    print("="*70)
    
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('xgb', xgb_model),
            ('lgb', lgb_model)
        ],
        voting='soft',
        weights=[1, 2, 2]
    )
    
    print("🔄 Fitting ensemble model...")
    # Fit ensemble (will use already fitted base estimators)
    ensemble.fit(X_train_smote, y_train_smote)
    print("✅ Ensemble model ready!")
    
    # Evaluate ensemble
    ensemble_metrics = evaluate_model(ensemble, X_test, y_test, "Ensemble (RF+XGB+LGB)")
    
    # Model comparison
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    print(f"\n{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'ROC AUC':<12}")
    print("-"*80)
    print(f"{'Random Forest':<20} {rf_metrics['accuracy']:.4f}       {rf_metrics['precision']:.4f}       {rf_metrics['recall']:.4f}       {rf_metrics['f1']:.4f}       {rf_metrics['roc_auc']:.4f}")
    print(f"{'XGBoost':<20} {xgb_metrics['accuracy']:.4f}       {xgb_metrics['precision']:.4f}       {xgb_metrics['recall']:.4f}       {xgb_metrics['f1']:.4f}       {xgb_metrics['roc_auc']:.4f}")
    print(f"{'LightGBM':<20} {lgb_metrics['accuracy']:.4f}       {lgb_metrics['precision']:.4f}       {lgb_metrics['recall']:.4f}       {lgb_metrics['f1']:.4f}       {lgb_metrics['roc_auc']:.4f}")
    print(f"{'Ensemble':<20} {ensemble_metrics['accuracy']:.4f}       {ensemble_metrics['precision']:.4f}       {ensemble_metrics['recall']:.4f}       {ensemble_metrics['f1']:.4f}       {ensemble_metrics['roc_auc']:.4f}")
    
    # Determine best model
    best_accuracy = max(rf_metrics['accuracy'], xgb_metrics['accuracy'], 
                       lgb_metrics['accuracy'], ensemble_metrics['accuracy'])
    
    if ensemble_metrics['accuracy'] == best_accuracy:
        best_model = ensemble
        best_metrics = ensemble_metrics
        best_name = "ensemble"
        print(f"\n🏆 BEST MODEL: Ensemble ({best_accuracy*100:.2f}% accuracy)")
    elif xgb_metrics['accuracy'] == best_accuracy:
        best_model = xgb_model
        best_metrics = xgb_metrics
        best_name = "xgboost"
        print(f"\n🏆 BEST MODEL: XGBoost ({best_accuracy*100:.2f}% accuracy)")
    elif lgb_metrics['accuracy'] == best_accuracy:
        best_model = lgb_model
        best_metrics = lgb_metrics
        best_name = "lightgbm"
        print(f"\n🏆 BEST MODEL: LightGBM ({best_accuracy*100:.2f}% accuracy)")
    else:
        best_model = rf_model
        best_metrics = rf_metrics
        best_name = "random_forest"
        print(f"\n🏆 BEST MODEL: Random Forest ({best_accuracy*100:.2f}% accuracy)")
    
    # Save best model as primary
    save_model(best_model, encoders, feature_cols, best_metrics, "credit_model")
    
    # Also save ensemble separately
    if best_name != "ensemble":
        save_model(ensemble, encoders, feature_cols, ensemble_metrics, "credit_model_ensemble")
    
    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\n✅ Best Model: {best_name.upper()}")
    print(f"✅ Accuracy: {best_accuracy*100:.2f}%")
    print(f"✅ {'🎯 TARGET ACHIEVED!' if best_accuracy >= 0.85 else '⚠️  Target not reached (need 85%+)'}")
    print(f"\n💾 Models saved to: {MODEL_DIR}")
    
    return best_model, best_metrics

if __name__ == "__main__":
    main()
