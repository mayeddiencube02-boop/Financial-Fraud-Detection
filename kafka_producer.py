"""
Step 6b (Kafka) - PRODUCER: publishes test transactions to a Kafka topic as JSON.
Run in terminal 1:  python kafka_producer.py
Needs: python -m pip install kafka-python   and a Kafka broker on localhost:9092
"""
import os
import json
import time
import argparse
import joblib
import pandas as pd
from kafka import KafkaProducer
from sklearn.model_selection import train_test_split

PROJECT_DIR = r"C:\Users\HomePC\Desktop\Financial_Fraud_Detection_Model"
ap = argparse.ArgumentParser()
ap.add_argument("--server", default="localhost:9092")
ap.add_argument("--topic", default="transaction_stream")
ap.add_argument("--delay", type=float, default=0.05, help="seconds between messages")
ap.add_argument("--limit", type=int, default=0, help="send only the first N (0 = all)")
args = ap.parse_args()

bundle = joblib.load(os.path.join(PROJECT_DIR, "best_fraud_model.joblib"))
FEATURES = bundle["features"]
df = pd.read_csv(os.path.join(PROJECT_DIR, "features_clean.csv"), parse_dates=["Transaction_Date"])
X, y = df[FEATURES], df["Fraudulent"]
_, X_test, _, _ = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
stream = df.loc[X_test.index].sort_values("Transaction_Date")
if args.limit:
    stream = stream.head(args.limit)

producer = KafkaProducer(
    bootstrap_servers=[args.server],
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

print(f"Sending {len(stream)} transactions to topic '{args.topic}'...")
for n, (_, row) in enumerate(stream.iterrows(), start=1):
    message = {
        "Transaction_ID": row["Transaction_ID"],
        "Customer_ID": row["Customer_ID"],
        "Transaction_Date": str(row["Transaction_Date"]),
        "Fraudulent": int(row["Fraudulent"]),                   # only used to check results
        "features": json.loads(row[FEATURES].to_json()),        # plain Python numbers/strings
    }
    producer.send(args.topic, message)
    if n % 100 == 0:
        print(f"  sent {n}")
    time.sleep(args.delay)

producer.flush()
print("All messages sent.")
