"""
Unsupervised / anomaly-detection models

Important libraries scikit-learn, matplotlib 

Models
  * Isolation Forest   - trained on ALL training rows, no labels used
  * One-Class SVM      - trained on LEGIT training rows only (novelty detection)
  * Autoencoder        - trained on LEGIT training rows only; reconstruction error = anomaly score
                         (built with scikit-learn's MLPRegressor so no TensorFlow is needed;
                          TensorFlow does not support Python 3.14)
Labels are used ONLY to evaluate, and (for OCSVM / autoencoder) to select the normal rows.
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

warnings.filterwarnings("ignore")
PROJECT_DIR = r"C:\Users\HomePC\Desktop\Financial_Fraud_Detection_Model"
RNG = 42

df = pd.read_csv(os.path.join(PROJECT_DIR, "features_clean.csv"))
y = df["Fraudulent"]

# Behavioural features only. Hour is encoded as a circle so 23:00 and 00:00 are neighbours.
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
features = ["Transaction_Amount", "Average_Spend", "amount_to_avg_ratio", "Previous_Transactions",
            "Account_Age_Days", "Is_International", "suspicious_keyword_flag", "is_night",
            "hour_sin", "hour_cos"]
X = df[features]

# same split as Step 3 (same seed + stratify) so results are comparable
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RNG)

scaler = StandardScaler().fit(X_train)
Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)
Xtr_normal = Xtr[(y_train == 0).values]
print(f"Train {Xtr.shape}, normal-only train {Xtr_normal.shape}, test {Xte.shape}")

scores = {}

# 1. Isolation Forest
iso = IsolationForest(n_estimators=300, contamination=float(y_train.mean()), random_state=RNG)
iso.fit(Xtr)
scores["Isolation Forest"] = -iso.score_samples(Xte)          # higher = more anomalous

# 2. One-Class SVM
ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.10)
ocsvm.fit(Xtr_normal)
scores["One-Class SVM"] = -ocsvm.decision_function(Xte)

# 3. Autoencoder
# Input -> 12 -> 4 (bottleneck) -> 12 -> Input. Learns to rebuild NORMAL transactions;
# transactions it rebuilds badly are suspicious.
ae = MLPRegressor(hidden_layer_sizes=(12, 4, 12), activation="relu", solver="adam",
                  max_iter=500, early_stopping=True, validation_fraction=0.1,
                  random_state=RNG)
ae.fit(Xtr_normal, Xtr_normal)
recon = ae.predict(Xte)
scores["Autoencoder"] = np.mean((Xte - recon) ** 2, axis=1)   # reconstruction error

# evaluation
def top_k_report(score, y_true, frac=0.10):
    k = int(len(score) * frac)
    idx = np.argsort(score)[::-1][:k]
    caught = y_true.values[idx].sum()
    return caught / k, caught / y_true.sum()                  # precision@k, recall@k

rows = []
for name, s in scores.items():
    p, r = top_k_report(s, y_test, 0.10)
    rows.append({"Model": name,
                 "ROC_AUC": roc_auc_score(y_test, s),
                 "PR_AUC": average_precision_score(y_test, s),
                 "Precision@top10%": p, "Recall@top10%": r,
                 "Lift@top10%": p / y_test.mean()})
res = pd.DataFrame(rows).sort_values("PR_AUC", ascending=False).round(4)
print("\n=== Unsupervised models on the test set ===")
print(res.to_string(index=False))
print(f"\n(Random guessing would give PR_AUC ~ {y_test.mean():.3f}, ROC_AUC 0.5, lift 1.0)")

sup = os.path.join(PROJECT_DIR, "model_comparison.csv")
if os.path.exists(sup):
    print("\nSupervised results from Step 3 for reference:")
    print(pd.read_csv(sup).to_string(index=False))

res.to_csv(os.path.join(PROJECT_DIR, "unsupervised_comparison.csv"), index=False)

# save scores for the dashboard
out = df.loc[X_test.index, ["Transaction_ID", "Fraudulent"]].copy()
for name, s in scores.items():
    # rescale to 0-100 so scores are readable on a dashboard
    out["score_" + name.lower().replace(" ", "_").replace("-", "_")] = \
        (100 * (s - s.min()) / (s.max() - s.min())).round(2)
out.to_csv(os.path.join(PROJECT_DIR, "unsupervised_scores_test.csv"), index=False)

# plots
os.makedirs(os.path.join(PROJECT_DIR, "plots"), exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for name, s in scores.items():
    pr, rc, _ = precision_recall_curve(y_test, s)
    ax[0].plot(rc, pr, label=f"{name} ({average_precision_score(y_test, s):.3f})")
ax[0].axhline(y_test.mean(), color="k", ls="--", label="random")
ax[0].set_title("Precision-Recall (unsupervised)"); ax[0].set_xlabel("Recall"); ax[0].set_ylabel("Precision")
ax[0].legend()

s = scores["Autoencoder"]
ax[1].hist(s[y_test.values == 0], bins=40, alpha=0.6, label="Legit", density=True)
ax[1].hist(s[y_test.values == 1], bins=40, alpha=0.6, label="Fraud", density=True)
ax[1].set_title("Autoencoder reconstruction error"); ax[1].set_xlabel("Error"); ax[1].legend()
ax[1].set_xlim(0, np.percentile(s, 99))
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_DIR, "plots", "unsupervised_results.png"), dpi=120)
plt.close()
print("\nSaved: unsupervised_comparison.csv, unsupervised_scores_test.csv, plots/unsupervised_results.png")

print("Done. Paste everything above back to me.")
