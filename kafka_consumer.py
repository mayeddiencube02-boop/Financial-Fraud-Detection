"""
Step 6b (Kafka) - CONSUMER: reads transactions from Kafka, scores them, raises alerts.
Run in terminal 2 (start it BEFORE or AFTER the producer, both work):  python kafka_consumer.py
Stops by itself after 30 s of silence.
Needs alerts.py in the same folder.
"""
import os
import json
import argparse
import joblib
import pandas as pd
from kafka import KafkaConsumer
from alerts import log_alert, send_email_alert

PROJECT_DIR = r"C:\Users\HomePC\Desktop\Financial_Fraud_Detection_Model"
ap = argparse.ArgumentParser()
ap.add_argument("--server", default="localhost:9092")
ap.add_argument("--topic", default="transaction_stream")
ap.add_argument("--idle", type=int, default=30, help="stop after this many idle seconds")
ap.add_argument("--email", action="store_true")
args = ap.parse_args()

bundle = joblib.load(os.path.join(PROJECT_DIR, "best_fraud_model.joblib"))
model, THRESH, FEATURES = bundle["pipeline"], bundle["threshold"], bundle["features"]

consumer = KafkaConsumer(
    args.topic,
    bootstrap_servers=[args.server],
    group_id="fraud-detector",
    auto_offset_reset="earliest",
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    consumer_timeout_ms=args.idle * 1000,
)

alert_log = os.path.join(PROJECT_DIR, "kafka_alerts_log.csv")
scored, emails = [], 0
print(f"Listening on '{args.topic}' (alert threshold {THRESH:.3f})...")
for message in consumer:
    tx = message.value
    x = pd.DataFrame([tx["features"]])[FEATURES]            # model expects a named-column row
    prob = float(model.predict_proba(x)[0, 1])
    tier = "HIGH" if prob >= THRESH else "MEDIUM" if prob >= THRESH * 0.6 else "LOW"
    scored.append({"Transaction_ID": tx["Transaction_ID"], "Customer_ID": tx["Customer_ID"],
                   "fraud_prob": round(prob, 4), "risk_tier": tier, "Fraudulent": tx["Fraudulent"]})
    if tier == "HIGH":
        details = {"Transaction_ID": tx["Transaction_ID"], "Customer_ID": tx["Customer_ID"],
                   "Amount": tx["features"]["Transaction_Amount"],
                   "Location": tx["features"]["Location"], "fraud_prob": round(prob, 4)}
        log_alert(alert_log, details)
        print(f"ALERT! Fraudulent transaction suspected: {details}")
        if args.email and emails < 3 and prob >= 0.9 and send_email_alert(details):
            emails += 1

print("\nNo new messages - stopping.")
out = pd.DataFrame(scored)
if len(out):
    out.to_csv(os.path.join(PROJECT_DIR, "kafka_scores.csv"), index=False)
    print(f"Processed {len(out)} messages")
    print(out.groupby("risk_tier")["Fraudulent"].agg(["mean", "sum", "count"]).round(3).to_string())
