import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix)
from xgboost import XGBClassifier
import shap
import warnings
warnings.filterwarnings("ignore")

BASE     = os.path.join(os.path.expanduser("~"), "Desktop",
                        "Dissertation", "PhQure")
DATA     = os.path.join(BASE, "data", "branch_a_features.csv")
MODELS   = os.path.join(BASE, "models")
os.makedirs(MODELS, exist_ok=True)

print("=" * 60)
print("  PhQure — Branch A Training (XGBoost + Random Forest)")
print("=" * 60)

# ── Load data ──────────────────────────────────────
df = pd.read_csv(DATA)
X  = df.drop("label", axis=1)
y  = df["label"]
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n  Training samples : {len(X_train):,}")
print(f"  Testing samples  : {len(X_test):,}")
print(f"  Features         : {len(feature_names)}")

# ── Evaluation helper ──────────────────────────────
def evaluate(name, model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba)
    print(f"\n  {'─'*50}")
    print(f"  {name}")
    print(f"  {'─'*50}")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  F1 Score  : {f1*100:.2f}%")
    print(f"  AUC       : {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Legit','Phishing'])}")
    return y_pred, y_proba, acc, f1, auc

# ── Train XGBoost ──────────────────────────────────
print("\n  [1/2] Training XGBoost ...")
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train, y_train)
xgb_pred, xgb_proba, xgb_acc, xgb_f1, xgb_auc = evaluate(
    "XGBoost", xgb, X_test, y_test)

# ── Train Random Forest ────────────────────────────
print("\n  [2/2] Training Random Forest ...")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_pred, rf_proba, rf_acc, rf_f1, rf_auc = evaluate(
    "Random Forest", rf, X_test, y_test)

# ── Save models ────────────────────────────────────
joblib.dump(xgb, os.path.join(MODELS, "xgboost_branchA.pkl"))
joblib.dump(rf,  os.path.join(MODELS, "randomforest_branchA.pkl"))
print("\n  Models saved to PhQure/models/")

# ── Confusion matrices ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Branch A — Confusion Matrices", fontsize=14, fontweight="bold")

for ax, name, pred in zip(axes,
        ["XGBoost", "Random Forest"],
        [xgb_pred,   rf_pred]):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Legit","Phishing"],
                yticklabels=["Legit","Phishing"])
    ax.set_title(name, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig(os.path.join(BASE, "data", "branchA_confusion_matrices.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Confusion matrices saved.")

# ── SHAP — XGBoost ────────────────────────────────
print("\n  Generating SHAP explanations for XGBoost...")
print("  (Takes 2-3 minutes)")

explainer   = shap.TreeExplainer(xgb)
shap_sample = X_test.sample(n=1000, random_state=42)
shap_values = explainer.shap_values(shap_sample)

# SHAP Summary Plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, shap_sample,
                  feature_names=feature_names,
                  show=False)
plt.title("Branch A — SHAP Feature Importance (XGBoost)", 
          fontweight="bold", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(BASE, "data", "branchA_shap_summary.png"),
            dpi=150, bbox_inches="tight")
plt.close()

# SHAP Bar Plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, shap_sample,
                  feature_names=feature_names,
                  plot_type="bar", show=False)
plt.title("Branch A — SHAP Mean Feature Importance (XGBoost)",
          fontweight="bold", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(BASE, "data", "branchA_shap_bar.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  SHAP plots saved.")

# ── Final summary ──────────────────────────────────
print(f"\n{'='*60}")
print(f"  BRANCH A — FINAL RESULTS SUMMARY")
print(f"{'='*60}")
print(f"  {'Model':<20} {'Accuracy':>10} {'F1':>10} {'AUC':>10}")
print(f"  {'─'*50}")
print(f"  {'XGBoost':<20} {xgb_acc*100:>9.2f}% {xgb_f1*100:>9.2f}% {xgb_auc:>10.4f}")
print(f"  {'Random Forest':<20} {rf_acc*100:>9.2f}% {rf_f1*100:>9.2f}% {rf_auc:>10.4f}")
print(f"{'='*60}")
print(f"\n  Outputs saved to PhQure/data/:")
print(f"  - branchA_confusion_matrices.png")
print(f"  - branchA_shap_summary.png")
print(f"  - branchA_shap_bar.png")
print(f"\n  Models saved to PhQure/models/:")
print(f"  - xgboost_branchA.pkl")
print(f"  - randomforest_branchA.pkl")
print(f"{'='*60}")