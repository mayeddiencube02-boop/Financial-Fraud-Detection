"""
Step 8c - Mobile alerts through a Telegram bot (free, works on any phone)

One-time setup (5 minutes):
  1. In Telegram, open a chat with @BotFather, send /newbot, follow the prompts.
     BotFather gives you a TOKEN. Keep it secret: anyone with it controls your bot.
  2. Open a chat with your new bot and send it any message (e.g. "hi").
  3. In a browser open  https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
     and find  "chat":{"id": 123456789 ...}  -> that number is your CHAT_ID.
  4. In PowerShell (valid for that window only):
        $env:TELEGRAM_BOT_TOKEN="123456:ABC..."
        $env:TELEGRAM_CHAT_ID="123456789"

Test:   python telegram_alerts.py --test
Use in code:
        from telegram_alerts import send_telegram_alert
        send_telegram_alert({"Transaction_ID": "T1", "Amount": 99.5, "Location": "Pune", "fraud_prob": 0.93})

With no environment variables set it only prints what it WOULD send (dry run), so it is safe to run.
"""
import os
import sys
import json
import urllib.request
import urllib.error


def format_alert(details: dict) -> str:
    prob = details.get("fraud_prob")
    risk = f"{float(prob):.0%}" if prob is not None else "n/a"
    lines = [f"FRAUD ALERT  (risk {risk})"]
    lines += [f"{k}: {v}" for k, v in details.items() if k != "fraud_prob"]
    return "\n".join(lines)


def send_telegram_alert(details: dict) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    text = format_alert(details)
    if not (token and chat_id):
        print("[telegram dry run: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set] would send:\n" + text)
        return False
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=json.dumps({"chat_id": chat_id, "text": text}).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError) as e:      # never let a failed push stop the stream
        print("[telegram failed]", e)
        return False


if __name__ == "__main__":
    if "--test" in sys.argv:
        ok = send_telegram_alert({"Transaction_ID": "TEST-001", "Amount": 123.45,
                                  "Location": "Mumbai", "fraud_prob": 0.93})
        print("Sent." if ok else "Not sent (see message above).")
    else:
        print(__doc__)
