"""
Step 8b - Tamper-evident alert ledger (a simple hash chain, the core idea behind a blockchain)

Each alert becomes a block that stores the SHA-256 hash of the block before it. Edit, delete or
reorder any past alert and every hash after it stops matching, so tampering is detectable.

This is a local, single-writer ledger. It proves INTEGRITY of the log, not distributed consensus.
A real blockchain would add multiple nodes agreeing on the chain.

Commands:
  python hashchain_log.py build     add new alerts from alerts_log.csv / kafka_alerts_log.csv
  python hashchain_log.py verify    check the whole chain
  python hashchain_log.py demo      copy the ledger, tamper with one alert, show it gets caught
"""
import os
import sys
import json
import csv
import shutil
import tempfile
import hashlib
from datetime import datetime

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(PROJECT_DIR, "alert_ledger.jsonl")
SOURCES = ["alerts_log.csv", "kafka_alerts_log.csv"]
GENESIS = "0" * 64


def block_hash(index, timestamp, data, prev_hash):
    payload = json.dumps({"index": index, "timestamp": timestamp, "data": data, "prev_hash": prev_hash},
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_ledger(path=LEDGER):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_ledger(blocks, path=LEDGER):
    with open(path, "w", encoding="utf-8") as f:
        for b in blocks:
            f.write(json.dumps(b, sort_keys=True) + "\n")


def verify(path=LEDGER, quiet=False):
    blocks = read_ledger(path)
    prev = GENESIS
    for i, b in enumerate(blocks):
        if b["index"] != i or b["prev_hash"] != prev:
            if not quiet:
                print(f"BROKEN at block {i}: chain link does not match (a block was removed or reordered).")
            return False
        if block_hash(b["index"], b["timestamp"], b["data"], b["prev_hash"]) != b["hash"]:
            if not quiet:
                print(f"TAMPERED at block {i}: contents no longer match the stored hash "
                      f"(transaction {b['data'].get('Transaction_ID')}).")
            return False
        prev = b["hash"]
    if not quiet:
        print(f"Ledger OK: {len(blocks)} blocks, chain intact. Head hash: {prev[:16]}...")
    return True


def build():
    blocks = read_ledger()
    if blocks and not verify(quiet=True):
        print("Existing ledger is not valid. Run 'verify' and investigate before adding more.")
        return
    seen = {b["data"].get("Transaction_ID") for b in blocks}
    prev = blocks[-1]["hash"] if blocks else GENESIS
    added = 0
    for name in SOURCES:
        path = os.path.join(PROJECT_DIR, name)
        if not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                tid = row.get("Transaction_ID")
                if tid in seen:
                    continue
                ts = row.pop("logged_at", None) or datetime.now().isoformat(timespec="seconds")
                data = {k: v for k, v in row.items()}
                data["source"] = name
                idx = len(blocks)
                h = block_hash(idx, ts, data, prev)
                blocks.append({"index": idx, "timestamp": ts, "data": data, "prev_hash": prev, "hash": h})
                seen.add(tid)
                prev = h
                added += 1
    write_ledger(blocks)
    print(f"Added {added} new alerts. Ledger now has {len(blocks)} blocks -> {LEDGER}")
    verify()


def demo():
    blocks = read_ledger()
    if len(blocks) < 3:
        print("Run 'build' first (needs at least 3 alerts).")
        return
    tmp = os.path.join(tempfile.mkdtemp(), "ledger_copy.jsonl")
    shutil.copy(LEDGER, tmp)
    copy = read_ledger(tmp)
    k = len(copy) // 2
    old = copy[k]["data"].get("Amount")
    copy[k]["data"]["Amount"] = "1.00"            # an attacker quietly lowers the amount
    write_ledger(copy, tmp)
    print(f"Changed alert in block {k}: Amount {old} -> 1.00 (in a COPY, your real ledger is untouched)")
    print("Verifying the tampered copy:")
    verify(tmp)
    print("Verifying your real ledger:")
    verify()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"build": build, "verify": verify, "demo": demo}.get(cmd, lambda: print(__doc__))()
