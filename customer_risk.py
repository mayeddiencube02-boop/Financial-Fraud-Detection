"""
Step 8a - Customer risk profiling (prototype for "AI-driven risk scoring")

IMPORTANT - what this is and is not:
  This dataset has no loan, repayment or default data, so true CREDITWORTHINESS cannot be modelled.
  What we CAN do is a customer RISK PROFILE: how risky is this customer's observed transaction
  behaviour (night activity, international use, suspicious keywords, model fraud scores)?
  The output feeds fraud review queues and could later be one input to a credit model.

Run:  python customer_risk.py
Needs: features_clean.csv and best_fraud_model.joblib (Steps 2-3)
Outputs: customer_risk.csv  (one row per customer) and a printed summary

Method
  1. Out-of-fold fraud probabilities for EVERY transaction (5-fold CV), so no transaction is scored
     by a model that was trained on it.
  2. Aggregate per customer: max / mean fraud probability, share of risky behaviours, volume.
  3. Combine into a 0-100 risk score and three bands (LOW / MEDIUM / HIGH), plus plain-English reasons.
"""
import os
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

warnings.filterwarnings("ignore")
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
P = lambda n: os.path.join(PROJECT_DIR, n)

bundle = joblib.load(P("best_fraud_model.joblib"))
FEATURES, THRESH = bundle["features"], bundle["threshold"]
df = pd.read_csv(P("features_clean.csv"), parse_dates=["Transaction_Date"])

# 1. out-of-fold probabilities
oof = cross_val_predict(clone(bundle["pipeline"]), df[FEATURES], df["Fraudulent"], method="predict_proba",
                        cv=StratifiedKFold(5, shuffle=True, random_state=42))[:, 1]
df["fraud_prob_oof"] = oof

# 2. per-customer aggregation
g = df.groupby("Customer_ID")
cust = pd.DataFrame({
    "transactions": g.size(),
    "avg_fraud_prob": g["fraud_prob_oof"].mean(),
    "max_fraud_prob": g["fraud_prob_oof"].max(),
    "night_share": g["is_night"].mean(),
    "international_share": g["Is_International"].mean(),
    "keyword_share": g["suspicious_keyword_flag"].mean(),
    "avg_amount": g["Transaction_Amount"].mean(),
    "avg_spend_profile": g["Average_Spend"].mean(),
    "account_age_days": g["Account_Age_Days"].max(),
    "last_seen": g["Transaction_Date"].max(),
    "confirmed_frauds": g["Fraudulent"].sum(),      # label, kept for reporting only, NOT used in the score
})

# 3. score: worst transaction matters most, average behaviour adds context
cust["risk_score"] = (100 * (0.6 * cust["max_fraud_prob"] + 0.4 * cust["avg_fraud_prob"])).round(1)
hi_cut, med_cut = 100 * THRESH, 100 * THRESH * 0.6
cust["risk_band"] = np.select([cust["risk_score"] >= hi_cut, cust["risk_score"] >= med_cut],
                              ["HIGH", "MEDIUM"], default="LOW")


def reasons(r):
    out = []
    if r["night_share"] > 0:
        out.append("transacts between 00:00-05:59")
    if r["international_share"] > 0:
        out.append("international transactions")
    if r["keyword_share"] > 0:
        out.append("suspicious keyword")
    return "; ".join(out) if out else "no risk flags"


cust["reasons"] = cust.apply(reasons, axis=1)
cust = cust.sort_values("risk_score", ascending=False).reset_index()
cust.to_csv(P("customer_risk.csv"), index=False)

# summary 
print(f"Customers scored: {len(cust)}  (band cut-offs: MEDIUM >= {med_cut:.0f}, HIGH >= {hi_cut:.0f})\n")
summary = cust.groupby("risk_band").agg(customers=("Customer_ID", "count"),
                                        with_confirmed_fraud=("confirmed_frauds", lambda s: int((s > 0).sum())),
                                        avg_score=("risk_score", "mean")).reindex(["LOW", "MEDIUM", "HIGH"])
summary["share_with_fraud"] = (summary["with_confirmed_fraud"] / summary["customers"]).round(3)
print(summary.round(2).to_string())
print("\nTop 10 highest-risk customers:")
print(cust[["Customer_ID", "risk_score", "risk_band", "transactions", "confirmed_frauds", "reasons"]]
      .head(10).to_string(index=False))
print("\nSaved customer_risk.csv")
print("Reminder: this is a fraud-exposure profile, not a credit score (no repayment data in this dataset).")
