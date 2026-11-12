"""
alerts.py - alerting helpers (console log + CSV log + optional email)

Email setup (optional). Use a Gmail APP PASSWORD, never your real password:
  Google Account > Security > 2-Step Verification > App passwords
Then set environment variables in PowerShell (current window only):
  $env:ALERT_SENDER="you@gmail.com"
  $env:ALERT_APP_PASSWORD="xxxx xxxx xxxx xxxx"
  $env:ALERT_RECIPIENT="analyst@example.com"
"""
import os
import smtplib
import csv
from datetime import datetime
from email.message import EmailMessage


def send_email_alert(details: dict) -> bool:
    sender = os.environ.get("ALERT_SENDER")
    password = os.environ.get("ALERT_APP_PASSWORD")
    recipient = os.environ.get("ALERT_RECIPIENT")
    if not (sender and password and recipient):
        print("  [email skipped: ALERT_SENDER / ALERT_APP_PASSWORD / ALERT_RECIPIENT not set]")
        return False

    msg = EmailMessage()
    msg["Subject"] = f"FRAUD ALERT: transaction {details['Transaction_ID']} (risk {details['fraud_prob']:.0%})"
    msg["From"] = sender
    msg["To"] = recipient
    msg.set_content(
        "A high-risk transaction was detected.\n\n"
        + "\n".join(f"{k}: {v}" for k, v in details.items())
        + "\n\nPlease review it in the dashboard."
    )
    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)
        return True
    except Exception as e:                       # never let an email failure stop the stream
        print("  [email failed]", e)
        return False


def log_alert(path: str, details: dict) -> None:
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["logged_at"] + list(details.keys()))
        if new_file:
            w.writeheader()
        w.writerow({"logged_at": datetime.now().isoformat(timespec="seconds"), **details})
