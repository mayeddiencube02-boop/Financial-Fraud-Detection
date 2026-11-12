""" 
Diagnostics- Where does the fraud live?

"""
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

PROJECT_DIR = r"C:\Users\HomePC\Desktop\Financial_Fraud_Detection_Model"
df = pd.read_csv(os.path.join(PROJECT_DIR, "features_clean.csv"))

print("=== Fraud rate by hour of day ===")
print(df.groupby("hour")["Fraudulent"].agg(["mean", "sum", "count"]).round(3).to_string())

print("\n=== Fraud rate by night x international x suspicious keyword ===")
combo = df.groupby(["is_night", "Is_International", "suspicious_keyword_flag"])["Fraudulent"] \
         .agg(["mean", "sum", "count"]).round(3)
print(combo.to_string())

print("\n=== Share of all fraud that has NONE of the 3 risk flags ===")
none_flag = (df["is_night"] == 0) & (df["Is_International"] == 0) & (df["suspicious_keyword_flag"] == 0)
print(f"Fraud cases with no flag: {int(df.loc[none_flag, 'Fraudulent'].sum())} of {int(df['Fraudulent'].sum())}")
print(f"Fraud rate inside that group: {df.loc[none_flag, 'Fraudulent'].mean():.3%}")

print("\n=== Rules learned by a tiny decision tree (depth 3) ===")
cols = ["hour", "Is_International", "suspicious_keyword_flag", "Transaction_Amount",
        "amount_to_avg_ratio", "Account_Age_Days", "Previous_Transactions"]
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=30, random_state=42)
tree.fit(df[cols], df["Fraudulent"])
print(export_text(tree, feature_names=cols, show_weights=True))
