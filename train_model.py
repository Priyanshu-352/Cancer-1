"""
Cancer Ensemble Model Trainer
=============================
Trains a multi-model ensemble (XGBoost + LightGBM + CatBoost)
on the Cancer_Data.csv dataset and saves the trained model.

Dataset: Cancer_Data.csv
- Columns: id, diagnosis, 30 feature columns

Usage:
    python train_model.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Import advanced gradient boosting frameworks
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'Cancer_Data.csv')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'cancer_ensemble_model.pkl')
SCALER_SAVE_PATH = os.path.join(BASE_DIR, 'cancer_scaler.pkl')

# Feature columns to use (8 features matching index.html frontend)
# index.html FEATURE_NAMES:
#   'worst_texture', 'se_radius', 'worst_symmetry', 'mean_concave_points',
#   'worst_concavity', 'se_area', 'worst_radius', 'worst_concave_points'
FEATURE_COLUMNS = [
    'texture_worst',     # worst_texture
    'radius_se',         # se_radius
    'symmetry_worst',    # worst_symmetry
    'concave points_mean',  # mean_concave_points
    'concavity_worst',   # worst_concavity
    'area_se',           # se_area
    'radius_worst',      # worst_radius
    'concave points_worst'  # worst_concave_points
]

# Diagnosis mapping: M = Malignant (1), B = Benign (0)
DIAGNOSIS_MAP = {'M': 1, 'B': 0}
CLASS_NAMES = ['Benign', 'Malignant']

# ---------------------------------------------------------------------------
# Load and prepare data
# ---------------------------------------------------------------------------
print("[INFO] Loading Cancer_Data.csv...")
df = pd.read_csv(CSV_PATH)
print(f"[INFO] Loaded {len(df)} rows, {len(df.columns)} columns")

# Separate features and target
X = df[FEATURE_COLUMNS]
y = df['diagnosis'].map(DIAGNOSIS_MAP)

# Handle any mapping errors (check for unexpected values)
if y.isnull().any():
    print(f"[WARN] Found {y.isnull().sum()} rows with unexpected diagnosis values")

# Split into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"[INFO] Training set: {X_train.shape[0]} samples")
print(f"[INFO] Test set: {X_test.shape[0]} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# Initialize ensemble models
# ---------------------------------------------------------------------------
print("\n[INFO] Initializing ensemble models...")

xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
    eval_metric='logloss',
    n_jobs=-1
)

lgb_model = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
    verbose=-1,
    n_jobs=-1
)

cat_model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.05,
    depth=4,
    random_state=42,
    verbose=0,
    thread_count=-1
)

# ---------------------------------------------------------------------------
# Train individual models
# ---------------------------------------------------------------------------
print("\n[INFO] Training individual component models...")

xgb_model.fit(X_train_scaled, y_train)
lgb_model.fit(X_train_scaled, y_train)
cat_model.fit(X_train_scaled, y_train)

print("[INFO] All individual models trained.")

# ---------------------------------------------------------------------------
# Ensemble prediction
# ---------------------------------------------------------------------------
print("\n[INFO] Generating ensemble predictions...")

xgb_probs = xgb_model.predict_proba(X_test_scaled)[:, 1]
lgb_probs = lgb_model.predict_proba(X_test_scaled)[:, 1]
cat_probs = cat_model.predict_proba(X_test_scaled)[:, 1]

# Soft voting: average of all three model probabilities
ensemble_probs = (xgb_probs + lgb_probs + cat_probs) / 3.0
ensemble_preds = np.where(ensemble_probs >= 0.5, 1, 0)

# ---------------------------------------------------------------------------
# Evaluate ensemble
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  MULTI-MODEL ENSEMBLE PERFORMANCE")
print("=" * 60)
print(f"Accuracy Score: {accuracy_score(y_test, ensemble_preds):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, ensemble_probs):.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, ensemble_preds, target_names=CLASS_NAMES))
print("=" * 60)

# ---------------------------------------------------------------------------
# Save model and scaler
# ---------------------------------------------------------------------------
print("\n[INFO] Saving trained model and scaler...")

joblib.dump(scaler, SCALER_SAVE_PATH)
print(f"[INFO] Scaler saved to: {SCALER_SAVE_PATH}")

# Save the ensemble models as a pipeline
ensemble_pipeline = {
    'xgboost': xgb_model,
    'lightgbm': lgb_model,
    'catboost': cat_model,
    'scaler': scaler,
    'feature_columns': FEATURE_COLUMNS
}

joblib.dump(ensemble_pipeline, MODEL_SAVE_PATH)
print(f"[INFO] Ensemble model saved to: {MODEL_SAVE_PATH}")

print("\n[INFO] Training complete!")
print(f"[INFO] Model path: {MODEL_SAVE_PATH}")
