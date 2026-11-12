"""
Step 6a - Real-time fraud monitoring simulator (no Kafka / Spark needed)

Replays the held-out test transactions one by one, in time order, as if they were arriving live.
For each transaction it:
  1. scores it with the saved model            (fraud probability)
  2. assigns a risk tier (LOW / MEDIUM / HIGH)
  3. raises an alert for HIGH risk             (console + alerts_log.csv + optional email)
  4. every BATCH transactions, "analyst feedback" (true labels) arrives and the adaptive model
     is retrained on training data + feedback, while a static model is left untouched, so we can
     compare the two.

Run:   python stream_simulator.py            (fast, no emails)
       python stream_simulator.py --email    (sends up to 3 emails if env vars are set)
Needs: alerts.py in the same folder, best_fraud_model.joblib from Step 3
"""
import os
import sys
import time
import argparse
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, average_precision_score

from alerts import send_email_alert, log_alert

warnings.filterwarnings("ignore")
PROJECT_DIR = r"C:\Users\HomePC\Desktop\Financial_Fraud_Detection_Model"
RNG = 42
BATCH = 100

ap = argparse.ArgumentParser()
ap.add_argument("--email", action="store_true", help="send email alerts (max 3)")
ap.add_argument("--delay", type=float, default=0.0, help="seconds between transactions")
args = ap.parse_args()

# load model + data
bundle = joblib.load(os.path.join(PROJECT_DIR, "best_fraud_model.joblib"))
static_model = bundle["pipeline"]
THRESH = bundle["threshold"]
FEATURES = bundle["features"]
print(f"Loaded model: {bundle['name']} | alert threshold {THRESH:.3f}")

df = pd.read_csv(os.path.join(PROJECT_DIR, "features_clean.csv"), parse_dates=["Transaction_Date"])
X, y = df[FEATURES], df["Fraudulent"]
# identical split to Step 3 -> the stream only contains transactions the model never trained on
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RNG)
stream = df.loc[X_test.index].sort_values("Transaction_Date")

adaptive_model = clone(static_model).fit(X_train, y_train)     # starts identical to static
pool_X, pool_y = X_train.copy(), y_train.copy()

HIGH = THRESH                 # at or above the tuned threshold  -> alert
MEDIUM = THRESH * 0.6         # worth a look


def tier(p):
    return "HIGH" if p >= HIGH else "MEDIUM" if p >= MEDIUM else "LOW"


alert_log = os.path.join(PROJECT_DIR, "alerts_log.csv")
if os.path.exists(alert_log):
    os.remove(alert_log)

# stream
records, batch_rows, emails_sent = [], [], 0
print(f"\nStreaming {len(stream)} transactions...\n")
for i, (idx, row) in enumerate(stream.iterrows(), start=1):
    x = X_test.loc[[idx]]
    p_static = float(static_model.predict_proba(x)[0, 1])
    p_adapt = float(adaptive_model.predict_proba(x)[0, 1])
    t = tier(p_adapt)

    rec = {
        "Transaction_ID": row["Transaction_ID"], "Transaction_Date": row["Transaction_Date"],
        "Customer_ID": row["Customer_ID"], "Transaction_Amount": row["Transaction_Amount"],
        "Merchant_Category": row["Merchant_Category"], "Payment_Method": row["Payment_Method"],
        "Device_Type": row["Device_Type"], "Location": row["Location"], "hour": row["hour"],
        "Is_International": row["Is_International"], "is_night": row["is_night"],
        "suspicious_keyword_flag": row["suspicious_keyword_flag"],
        "fraud_prob_static": round(p_static, 4), "fraud_prob": round(p_adapt, 4),
        "risk_tier": t, "Fraudulent": int(row["Fraudulent"]),
    }
    records.append(rec)

    if t == "HIGH":
        alert = {"Transaction_ID": rec["Transaction_ID"], "Customer_ID": rec["Customer_ID"],
                 "Amount": rec["Transaction_Amount"], "Location": rec["Location"],
                 "Device": rec["Device_Type"], "fraud_prob": rec["fraud_prob"]}
        log_alert(alert_log, alert)
        if i <= 400:   # keep the console readable; every alert is still in alerts_log.csv
            print(f"ALERT! {rec['Transaction_ID']} | {rec['Customer_ID']} | "
                  f"amount {rec['Transaction_Amount']:.2f} | {rec['Location']} | risk {p_adapt:.0%}")
        if args.email and emails_sent < 3 and p_adapt >= 0.9:
            if send_email_alert({**alert, "fraud_prob": p_adapt}):
                emails_sent += 1

    if args.delay:
        time.sleep(args.delay)

    # adaptive learning (feedback arrives)
    if i % BATCH == 0:
        b = pd.DataFrame(records[-BATCH:])
        yb = b["Fraudulent"]
        for label, col in [("static", "fraud_prob_static"), ("adaptive", "fraud_prob")]:
            pred = (b[col] >= THRESH).astype(int)
            batch_rows.append({"batch": i // BATCH, "model": label,
                               "recall": recall_score(yb, pred, zero_division=0),
                               "precision": precision_score(yb, pred, zero_division=0),
                               "PR_AUC": average_precision_score(yb, b[col]) if yb.sum() else np.nan})
        # feedback: add this batch with its true labels, retrain the adaptive model
        bx = X_test.loc[[r for r in stream.index[i - BATCH:i]]]
        pool_X = pd.concat([pool_X, bx])
        pool_y = pd.concat([pool_y, y_test.loc[bx.index]])
        adaptive_model = clone(static_model).fit(pool_X, pool_y)
        print(f"-- batch {i // BATCH} done: adaptive model retrained on {len(pool_X)} labelled rows")

# summary
out = pd.DataFrame(records)
out.to_csv(os.path.join(PROJECT_DIR, "stream_scores.csv"), index=False)

print("\n=== Stream summary ===")
print(out["risk_tier"].value_counts().to_string())
hi = out[out["risk_tier"] == "HIGH"]
print(f"\nHIGH alerts: {len(hi)} | of those truly fraud: {int(hi['Fraudulent'].sum())} "
      f"({hi['Fraudulent'].mean():.1%} precision) | fraud caught: "
      f"{int(hi['Fraudulent'].sum())}/{int(out['Fraudulent'].sum())}")
print("\nFraud rate by risk tier:")
print(out.groupby("risk_tier")["Fraudulent"].agg(["mean", "sum", "count"]).round(3).to_string())

bt = pd.DataFrame(batch_rows).round(3)
print("\nPer-batch comparison (static vs adaptive):")
print(bt.pivot(index="batch", columns="model", values=["recall", "precision"]).to_string())
print("\nOverall PR_AUC  static: %.4f | adaptive: %.4f" % (
    average_precision_score(out["Fraudulent"], out["fraud_prob_static"]),
    average_precision_score(out["Fraudulent"], out["fraud_prob"])))
print("\nSaved: stream_scores.csv (for the dashboard), alerts_log.csv")
print("Done. Paste everything from 'Stream summary' down back to me.")
