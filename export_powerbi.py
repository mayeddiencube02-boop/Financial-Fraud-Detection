"""
Step 7b - Export tidy tables for Power BI
Run:  python export_powerbi.py
Creates the folder  powerbi_export/  with CSVs you can load via Power BI Desktop:
  Home > Get data > Text/CSV  (or Get data > Folder to load them all at once)
"""
import os
import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
P = lambda *a: os.path.join(PROJECT_DIR, *a)
OUT = P("powerbi_export")
os.makedirs(OUT, exist_ok=True)
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# ---- fact table: every scored transaction ------------------------------------
s = pd.read_csv(P("stream_scores.csv"), parse_dates=["Transaction_Date"])
if os.path.exists(P("unsupervised_scores_test.csv")):
    s = s.merge(pd.read_csv(P("unsupervised_scores_test.csv")).drop(columns=["Fraudulent"]),
                on="Transaction_ID", how="left")
s["Date"] = s["Transaction_Date"].dt.date
s["Day_Of_Week_No"] = s["Transaction_Date"].dt.dayofweek
s["Day_Of_Week"] = s["Day_Of_Week_No"].map(dict(enumerate(DAYS)))
s["Risk_Score"] = (s["fraud_prob"] * 100).round(1)
s["Risk_Tier_Order"] = s["risk_tier"].map({"LOW": 1, "MEDIUM": 2, "HIGH": 3})
s["Is_High_Risk"] = (s["risk_tier"] == "HIGH").astype(int)
s["Fraud_Label"] = s["Fraudulent"].map({0: "Legit", 1: "Fraud"})
s["Caught_By_High"] = ((s["risk_tier"] == "HIGH") & (s["Fraudulent"] == 1)).astype(int)
s.to_csv(os.path.join(OUT, "fact_transactions.csv"), index=False)

# ---- ready-made summary tables (handy for quick visuals) ---------------------
(s.groupby("risk_tier").agg(transactions=("Transaction_ID", "count"), frauds=("Fraudulent", "sum"),
                            fraud_rate=("Fraudulent", "mean")).reset_index()
   .to_csv(os.path.join(OUT, "summary_by_tier.csv"), index=False))
(s.groupby(["Day_Of_Week_No", "Day_Of_Week", "hour"]).agg(transactions=("Transaction_ID", "count"),
                                                         avg_risk=("Risk_Score", "mean"),
                                                         high_alerts=("Is_High_Risk", "sum"),
                                                         frauds=("Fraudulent", "sum")).reset_index()
   .to_csv(os.path.join(OUT, "summary_hour_by_day.csv"), index=False))

# ---- model results + graph tables -------------------------------------------
for src, dst in [("model_comparison.csv", "models_supervised.csv"),
                 ("unsupervised_comparison.csv", "models_unsupervised.csv"),
                 ("graph_nodes.csv", "graph_nodes.csv"), ("graph_edges.csv", "graph_edges.csv")]:
    if os.path.exists(P(src)):
        pd.read_csv(P(src)).to_csv(os.path.join(OUT, dst), index=False)

print("Wrote to", OUT)
for f in sorted(os.listdir(OUT)):
    print("  ", f)
print("\nSuggested Power BI visuals:")
print("  Cards: count of Transaction_ID, sum of Is_High_Risk, sum of Caught_By_High / sum of Fraudulent")
print("  Clustered column: risk_tier (sort by Risk_Tier_Order) vs count of Transaction_ID")
print("  Matrix (heatmap): rows Day_Of_Week (sort by Day_Of_Week_No), columns hour, values avg_risk, conditional formatting")
print("  Bar: Merchant_Category / Location vs sum of Is_High_Risk")
print("  Slicers: risk_tier, Location, Date, Is_International")
