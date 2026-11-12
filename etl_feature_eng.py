"""
Step 2 - Extract- Transform- Load (ETL) + cleaning + feature engineering

Output: fraud_detection.db (SQLite), features_clean.csv

"""
import os
import sqlite3
import numpy as np
import pandas as pd
 
PROJECT_DIR = r"C:\Users\HomePC\Desktop\Financial_Fraud_Detection_Model"
SRC = os.path.join(PROJECT_DIR, "financial_fraud_detection_dataset.csv")
DB = os.path.join(PROJECT_DIR, "fraud_detection.db")
OUT_CSV = os.path.join(PROJECT_DIR, "features_clean.csv")
 
# EXTRACT
df = pd.read_csv(SRC)
print("Extracted:", df.shape)
 
# TRANSFORM
# 1. Clean types and text
df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"], format="%d-%m-%Y %H:%M")
for col in ["Merchant_Category", "Payment_Method", "Device_Type", "Location", "Suspicious_Keyword"]:
    df[col] = df[col].str.strip()
 
# 2. Missing values (none in this dataset, but kept so the pipeline is reusable)
num_cols = df.select_dtypes(include="number").columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
cat_cols = df.select_dtypes(include=["object", "string"]).columns
for c in cat_cols:
    if df[c].isnull().any():
        df[c] = df[c].fillna(df[c].mode()[0])
 
df = df.drop_duplicates().sort_values(["Transaction_Date"]).reset_index(drop=True)
 
# 3. Feature engineering -- time patterns
df["hour"] = df["Transaction_Date"].dt.hour
df["day_of_week"] = df["Transaction_Date"].dt.dayofweek          # 0 = Monday
df["month"] = df["Transaction_Date"].dt.month
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
df["is_night"] = df["hour"].between(0, 5).astype(int)   # fraud peaks 00:00-05:59
 
# 4. Feature engineering -- amount behaviour vs the customer's normal spend
df["amount_to_avg_ratio"] = df["Transaction_Amount"] / df["Average_Spend"].replace(0, np.nan)
df["amount_to_avg_ratio"] = df["amount_to_avg_ratio"].fillna(0)
df["amount_minus_avg"] = df["Transaction_Amount"] - df["Average_Spend"]
df["log_amount"] = np.log1p(df["Transaction_Amount"])
df["is_zero_amount"] = (df["Transaction_Amount"] == 0).astype(int)
 
# 5. Feature engineering -- account / customer behaviour
df["is_new_account"] = (df["Account_Age_Days"] < 90).astype(int)
df["txn_per_account_day"] = df["Previous_Transactions"] / df["Account_Age_Days"].clip(lower=1)
df["suspicious_keyword_flag"] = (df["Suspicious_Keyword"].str.lower() == "yes").astype(int)
 
# Running count of this customer's transactions in the data (uses only the past)
df["customer_txn_number"] = df.groupby("Customer_ID").cumcount() + 1
# Hours since the same customer's previous transaction (-1 = first time seen)
gap = df.groupby("Customer_ID")["Transaction_Date"].diff().dt.total_seconds() / 3600
df["hours_since_last_txn"] = gap.fillna(-1)
 
# Risky combinations
df["risk_flag_count"] = df["is_night"] + df["Is_International"] + df["suspicious_keyword_flag"]
df["intl_and_new_account"] = df["Is_International"] * df["is_new_account"]
df["intl_and_night"] = df["Is_International"] * df["is_night"]
 
# NOTE: scaling / one-hot encoding happens in Step 3 inside a pipeline fitted on the
# TRAINING data only. Doing it here would leak test-set information into training.
 
# LOAD

conn = sqlite3.connect(DB)
cur = conn.cursor()
cur.execute("DROP TABLE IF EXISTS transactions")
cur.execute("""
CREATE TABLE transactions (
    Transaction_ID          TEXT PRIMARY KEY,
    Customer_ID             TEXT,
    Transaction_Date        TEXT,
    Transaction_Amount      REAL,
    Merchant_Category       TEXT,
    Payment_Method          TEXT,
    Device_Type             TEXT,
    Location                TEXT,
    Is_International        INTEGER,
    Previous_Transactions   INTEGER,
    Average_Spend           REAL,
    Account_Age_Days        INTEGER,
    Suspicious_Keyword      TEXT,
    Fraudulent              INTEGER,
    hour                    INTEGER,
    day_of_week             INTEGER,
    month                   INTEGER,
    is_weekend              INTEGER,
    is_night                INTEGER,
    amount_to_avg_ratio     REAL,
    amount_minus_avg        REAL,
    log_amount              REAL,
    is_zero_amount          INTEGER,
    is_new_account          INTEGER,
    txn_per_account_day     REAL,
    suspicious_keyword_flag INTEGER,
    customer_txn_number     INTEGER,
    hours_since_last_txn    REAL,
    risk_flag_count         INTEGER,
    intl_and_new_account    INTEGER,
    intl_and_night          INTEGER
)
""")

to_db = df.copy()
to_db["Transaction_Date"] = to_db["Transaction_Date"].dt.strftime("%Y-%m-%d %H:%M:%S")
to_db.to_sql("transactions", conn, if_exists="append", index=False)
conn.commit()
 
# quick check straight from the database
n = cur.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
conn.close()
df.to_csv(OUT_CSV, index=False)
print(f"Loaded {n} rows into {DB}")
print("ETL process completed.\n")
 
# QUICK INSIGHTS

print("Fraud rate by category (paste this back to me):")
for col in ["Merchant_Category", "Payment_Method", "Device_Type", "Location",
            "Is_International", "Suspicious_Keyword", "is_night", "is_new_account"]:
    rate = df.groupby(col)["Fraudulent"].agg(["mean", "count"]).round(3)
    print(f"\n{col}\n{rate}")
 
print("\nCorrelation of numeric features with Fraudulent:")
corr = df.select_dtypes(include="number").corr()["Fraudulent"].drop("Fraudulent")
print(corr.sort_values(key=abs, ascending=False).round(3))
print("\nZero-amount transactions:", int(df["is_zero_amount"].sum()))
 