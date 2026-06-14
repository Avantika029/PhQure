import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

BASE   = os.path.join(os.path.expanduser("~"), "Desktop",
                      "Dissertation", "PhQure")
DATA   = os.path.join(BASE, "data", "branch_a_features.csv")
MODELS = os.path.join(BASE, "models")

df = pd.read_csv(DATA)
X  = df.drop("label", axis=1)
y  = df["label"]

xgb = joblib.load(os.path.join(MODELS, "xgboost_branchA.pkl"))
rf  = joblib.load(os.path.join(MODELS, "randomforest_branchA.pkl"))

print("=" * 60)
print("  Branch A — Overfitting Validation")
print("=" * 60)

# ── Test 1: 5-Fold Cross Validation ───────────────
# If model is overfitted, CV scores will be much lower
# than training scores. If genuine, they stay high.
print("\n  [Test 1] 5-Fold Cross Validation")
print("  (Tests if model generalises across different data splits)")
print("  Running... takes 3-5 minutes\n")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in [("XGBoost", xgb), ("Random Forest", rf)]:
    scores = cross_val_score(model, X, y,
                             cv=cv, scoring="f1", n_jobs=-1)
    print(f"  {name}")
    print(f"    Fold scores : {[round(s*100,2) for s in scores]}")
    print(f"    Mean F1     : {scores.mean()*100:.2f}%")
    print(f"    Std Dev     : {scores.std()*100:.3f}%")
    print(f"    Verdict     : ", end="")
    if scores.mean() > 0.97 and scores.std() < 0.01:
        print("GENUINE - consistent high performance across all folds")
    elif scores.mean() > 0.94:
        print("LIKELY GENUINE - good generalisation")
    else:
        print("WARNING - possible overfitting, investigate further")
    print()

# ── Test 2: Train vs Test score comparison ────────
print("\n  [Test 2] Training Score vs Test Score Gap")
print("  (Large gap = overfitting, Small gap = genuine)")
print()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

for name, model in [("XGBoost", xgb), ("Random Forest", rf)]:
    train_score = f1_score(y_train, model.predict(X_train))
    test_score  = f1_score(y_test,  model.predict(X_test))
    gap = train_score - test_score
    print(f"  {name}")
    print(f"    Train F1 : {train_score*100:.2f}%")
    print(f"    Test F1  : {test_score*100:.2f}%")
    print(f"    Gap      : {gap*100:.3f}%")
    print(f"    Verdict  : ", end="")
    if gap < 0.005:
        print("NO OVERFITTING - train/test gap is negligible")
    elif gap < 0.02:
        print("MINIMAL OVERFITTING - acceptable for this domain")
    else:
        print("OVERFITTING DETECTED - gap too large")
    print()

print("=" * 60)
print("  Interpretation Guide:")
print("  CV Std Dev < 1%  = very stable, not overfitted")
print("  Train/Test gap < 0.5% = genuinely high accuracy")
print("  If both tests pass = results are publishable")
print("=" * 60)