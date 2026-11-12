"""
Step 3 - Supervised fraud models


Install first:  python -m pip install scikit-learn matplotlib joblib xgboost lightgbm
(xgboost / lightgbm are optional: if they fail to install on your Python version,
 the script skips them and still runs the other three models.)
Outputs: model_comparison.csv, best_fraud_model.joblib, plots in /plots
"""
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, confusion_matrix, classification_report,
                             precision_recall_curve, roc_curve, ConfusionMatrixDisplay)
 
warnings.filterwarnings("ignore")
PROJECT_DIR = r"C:\Users\HomePC\Desktop\Financial_Fraud_Detection_Model"
os.makedirs(os.path.join(PROJECT_DIR, "plots"), exist_ok=True)
RNG = 42
 
#  data

df = pd.read_csv(os.path.join(PROJECT_DIR, "features_clean.csv"))
TARGET = "Fraudulent"
 
cat_features = ["Merchant_Category", "Payment_Method", "Device_Type", "Location"]
num_features = [
    "Transaction_Amount", "Is_International", "Previous_Transactions", "Average_Spend",
    "Account_Age_Days", "hour", "day_of_week", "is_weekend", "is_night",
    "amount_to_avg_ratio", "amount_minus_avg", "log_amount", "is_new_account",
    "txn_per_account_day", "suspicious_keyword_flag", "customer_txn_number",
    "hours_since_last_txn", "intl_and_new_account", "intl_and_night", "risk_flag_count",
]
# Dropped on purpose: IDs, raw date, raw Suspicious_Keyword (duplicate of the flag),
# month (data covers few months, would not generalise), is_zero_amount (only 2 rows)
 
X = df[cat_features + num_features]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RNG)
print(f"Train: {X_train.shape}, Test: {X_test.shape}, fraud rate train {y_train.mean():.3%}")
 
# Scaling + encoding are fitted on TRAIN data only (inside the pipeline) -> no leakage
preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
])
 
# models
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()   # ~9.4, for imbalance
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20,
                                            class_weight="balanced", random_state=RNG),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                            class_weight="balanced_subsample",
                                            n_jobs=-1, random_state=RNG),
}
try:
    from xgboost import XGBClassifier
    models["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                      subsample=0.8, colsample_bytree=0.8,
                                      scale_pos_weight=pos_weight, eval_metric="aucpr",
                                      random_state=RNG)
except Exception as e:
    print("XGBoost skipped:", e)
try:
    from lightgbm import LGBMClassifier
    models["LightGBM"] = LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                        subsample=0.8, colsample_bytree=0.8,
                                        scale_pos_weight=pos_weight, random_state=RNG,
                                        verbose=-1)
except Exception as e:
    print("LightGBM skipped:", e)
 
# train + evaluate

rows, fitted = [], {}
for name, clf in models.items():
    pipe = Pipeline([("prep", preprocess), ("clf", clf)])
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    rows.append({
        "Model": name,
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC_AUC": roc_auc_score(y_test, proba),
        "PR_AUC": average_precision_score(y_test, proba),
    })
    fitted[name] = (pipe, proba)
 
results = pd.DataFrame(rows).sort_values("PR_AUC", ascending=False).round(4)
print("\n=== Model comparison (test set, threshold 0.5) ===")
print(results.to_string(index=False))
results.to_csv(os.path.join(PROJECT_DIR, "model_comparison.csv"), index=False)
 
# best model: threshold tuning
best_name = results.iloc[0]["Model"]
best_pipe, best_proba = fitted[best_name]
print(f"\nBest model by PR-AUC: {best_name}")
 
# choose the threshold on TRAIN data via cross-validation (not on the test set)

oof = cross_val_predict(best_pipe, X_train, y_train, method="predict_proba",
                        cv=StratifiedKFold(5, shuffle=True, random_state=RNG))[:, 1]
prec, rec, thr = precision_recall_curve(y_train, oof)
f1s = 2 * prec * rec / (prec + rec + 1e-9)
best_thr = float(thr[np.argmax(f1s[:-1])])
print(f"Tuned decision threshold (from cross-validation): {best_thr:.3f}")
 
final_pred = (best_proba >= best_thr).astype(int)
print("\nClassification report on the test set at tuned threshold:")
print(classification_report(y_test, final_pred, target_names=["Legit", "Fraud"], digits=3))
print("Confusion matrix [[TN FP],[FN TP]]:\n", confusion_matrix(y_test, final_pred))
 
# save model
joblib.dump({"pipeline": best_pipe, "threshold": best_thr,
             "features": cat_features + num_features, "name": best_name},
            os.path.join(PROJECT_DIR, "best_fraud_model.joblib"))
print("Saved best_fraud_model.joblib")
 
# plots
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for name, (_, p) in fitted.items():
    fpr, tpr, _ = roc_curve(y_test, p)
    ax[0].plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_test, p):.3f})")
    pr, rc, _ = precision_recall_curve(y_test, p)
    ax[1].plot(rc, pr, label=f"{name} ({average_precision_score(y_test, p):.3f})")
ax[0].plot([0, 1], [0, 1], "k--"); ax[0].set_title("ROC curves"); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR")
ax[1].set_title("Precision-Recall curves"); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
ax[0].legend(); ax[1].legend()
plt.tight_layout(); plt.savefig(os.path.join(PROJECT_DIR, "plots", "roc_pr_curves.png"), dpi=120); plt.close()
 
ConfusionMatrixDisplay.from_predictions(y_test, final_pred, display_labels=["Legit", "Fraud"], cmap="Blues")
plt.title(f"{best_name} - confusion matrix"); plt.tight_layout()
plt.savefig(os.path.join(PROJECT_DIR, "plots", "confusion_matrix.png"), dpi=120); plt.close()
 
# feature importance (tree models) or coefficients (logistic regression)
clf = best_pipe.named_steps["clf"]
names = best_pipe.named_steps["prep"].get_feature_names_out()
if hasattr(clf, "feature_importances_"):
    imp = pd.Series(clf.feature_importances_, index=names)
elif hasattr(clf, "coef_"):
    imp = pd.Series(np.abs(clf.coef_[0]), index=names)
else:
    imp = None
if imp is not None:
    top = imp.sort_values(ascending=False).head(15)
    print("\nTop 15 features:\n", top.round(4))
    top[::-1].plot(kind="barh", figsize=(8, 6), title=f"{best_name} - top features")
    plt.tight_layout(); plt.savefig(os.path.join(PROJECT_DIR, "plots", "feature_importance.png"), dpi=120); plt.close()
 
print("\nDone. Paste everything above back to me.")
 