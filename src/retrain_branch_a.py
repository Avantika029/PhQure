import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import joblib

print("Loading data...")
df = pd.read_csv('data/branch_a_features_v2.csv')
X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

print("\nTraining XGBoost...")
xgb = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)
xgb.fit(X_train, y_train)

xgb_train_acc = accuracy_score(y_train, xgb.predict(X_train))
xgb_test_acc = accuracy_score(y_test, xgb.predict(X_test))
xgb_auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])

print(f"  Train Accuracy : {xgb_train_acc:.4f}")
print(f"  Test Accuracy  : {xgb_test_acc:.4f}")
print(f"  AUC            : {xgb_auc:.4f}")

print("\nTraining Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

rf_train_acc = accuracy_score(y_train, rf.predict(X_train))
rf_test_acc = accuracy_score(y_test, rf.predict(X_test))
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

print(f"  Train Accuracy : {rf_train_acc:.4f}")
print(f"  Test Accuracy  : {rf_test_acc:.4f}")
print(f"  AUC            : {rf_auc:.4f}")

print("\nSaving models...")
joblib.dump(xgb, 'models/branch_a_xgb.pkl')
joblib.dump(rf, 'models/branch_a_rf.pkl')
print("Done. Models saved.")