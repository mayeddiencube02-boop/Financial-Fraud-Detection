"""
Step 9 - Build the small data folder used by the deployed dashboard (Streamlit Community Cloud)

Run:  python make_demo_data.py
Creates:
  demo_data/                    only what dashboard.py needs, no raw dataset
  requirements.txt              dashboard-only packages pinned to the versions installed on THIS machine
  requirements-pipeline.txt     your previous requirements.txt (everything needed to run the full pipeline)

What goes into demo_data/ is derived from your dataset (1,000 scored test transactions with their
IDs, amounts and labels, plus the trained model). Only commit it if you are allowed to publish
that data, e.g. it is a public or synthetic dataset.
"""
import os
import json
import shutil
from importlib import metadata

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(ROOT, "demo_data")
os.makedirs(os.path.join(DEMO, "plots"), exist_ok=True)

# ---- 1. copy the files the dashboard reads -------------------------------------
FILES = ["stream_scores.csv", "unsupervised_scores_test.csv", "model_comparison.csv",
         "unsupervised_comparison.csv", "graph_nodes.csv", "graph_edges.csv",
         "best_fraud_model.joblib"]
missing = []
for name in FILES:
    src = os.path.join(ROOT, name)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(DEMO, name))
        print("copied", name)
    else:
        missing.append(name)
png = os.path.join(ROOT, "plots", "fraud_graph.png")
if os.path.exists(png):
    shutil.copy2(png, os.path.join(DEMO, "plots", "fraud_graph.png"))
    print("copied plots/fraud_graph.png")
else:
    missing.append("plots/fraud_graph.png")

# ---- 2. tiny defaults file for the 'Score a transaction' tab (no raw rows shipped) ----
import joblib
fc = os.path.join(ROOT, "features_clean.csv")
mp = os.path.join(ROOT, "best_fraud_model.joblib")
if os.path.exists(fc) and os.path.exists(mp):
    feats = joblib.load(mp)["features"]
    base = pd.read_csv(fc)
    row = {c: (float(base[c].median()) if pd.api.types.is_numeric_dtype(base[c]) else str(base[c].mode()[0]))
           for c in feats}
    # build everything first, then write, so a failure can never leave an empty/broken file behind
    payload = {"row": row, "locations": sorted(base["Location"].unique().tolist()),
               "merchants": sorted(base["Merchant_Category"].unique().tolist())}
    with open(os.path.join(DEMO, "scorer_defaults.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    print("wrote scorer_defaults.json")
else:
    missing.append("scorer_defaults.json (needs features_clean.csv and best_fraud_model.joblib)")

# ---- 3. dashboard-only requirements, pinned to the versions you trained with -----------
pkgs = ["streamlit", "plotly", "pandas", "numpy", "scikit-learn", "joblib"]
lines = []
for p in pkgs:
    try:
        lines.append(f"{p}=={metadata.version(p)}")
    except metadata.PackageNotFoundError:
        lines.append(p)
req = os.path.join(ROOT, "requirements.txt")
old = os.path.join(ROOT, "requirements-pipeline.txt")
if os.path.exists(req) and not os.path.exists(old):
    shutil.copy2(req, old)
    print("saved your previous requirements.txt as requirements-pipeline.txt")
with open(req, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\nrequirements.txt (dashboard only):")
print("\n".join("  " + l for l in lines))

print("\nDemo folder size: %.1f MB" % (sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(DEMO) for f in fs) / 1e6))
if missing:
    print("\nNOT FOUND (the matching dashboard tab will show a notice):")
    for m in missing:
        print("  -", m)
print("\nNext: test with  python -m streamlit run dashboard.py  then commit demo_data/ and deploy.")
