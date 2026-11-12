"""
Step 5 - Graph-based fraud analysis

Install first:  python -m pip install networkx
Outputs: plots/fraud_graph.png, graph_nodes.csv, graph_edges.csv (usable in Power BI)

"""
import os
import itertools
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from networkx.algorithms.community import greedy_modularity_communities

PROJECT_DIR = r"C:\Users\HomePC\Desktop\Financial_Fraud_Detection_Model"
os.makedirs(os.path.join(PROJECT_DIR, "plots"), exist_ok=True)
df = pd.read_csv(os.path.join(PROJECT_DIR, "features_clean.csv"), parse_dates=["Transaction_Date"])
df = df.sort_values("Transaction_Date").reset_index(drop=True)
base = df["Fraudulent"].mean()


# PART 1 - Does a customer's fraud history predict future fraud?

print("=== PART 1: repeat-customer behaviour ===")
cust = df.groupby("Customer_ID")["Fraudulent"].agg(n="count", frauds="sum")
print(f"Customers: {len(cust)} | with 2+ transactions: {(cust.n >= 2).sum()} "
      f"| with at least 1 fraud: {(cust.frauds >= 1).sum()}")

# If fraud hit customers at random, how many customers would have 2+ frauds?

p = base
n = cust["n"].values
expected_multi = np.sum(1 - (1 - p) ** n - n * p * (1 - p) ** (n - 1))
observed_multi = int((cust["frauds"] >= 2).sum())
print(f"Customers with 2+ frauds: observed {observed_multi}, expected by pure chance {expected_multi:.1f}")

# Fraud history known BEFORE each transaction (no peeking at the future)

df["prior_fraud_count"] = df.groupby("Customer_ID")["Fraudulent"].cumsum() - df["Fraudulent"]
df["prior_txn_count"] = df.groupby("Customer_ID").cumcount()
has_hist = df[df["prior_txn_count"] > 0]
print("\nFraud rate of a transaction vs. the customer's earlier fraud count:")
print(has_hist.groupby(has_hist["prior_fraud_count"].clip(upper=1))["Fraudulent"]
      .agg(["mean", "count"]).rename(index={0: "no earlier fraud", 1: "had earlier fraud"}).round(3))


# PART 2 - Entity graph (bipartite: customers <-> attribute nodes)

print("\n=== PART 2: entity graph ===")
df["flag_night"] = df["is_night"]
attr_cols = {"Merchant_Category": "merchant", "Payment_Method": "payment",
             "Device_Type": "device", "Location": "location"}
G = nx.Graph()
for _, r in df.iterrows():
    c = "cust:" + r["Customer_ID"]
    G.add_node(c, kind="customer")
    nodes = [(f"{k}:{r[col]}", k) for col, k in attr_cols.items()]
    nodes += [("flag:night(0-5h)", "flag")] * int(r["is_night"])
    nodes += [("flag:international", "flag")] * int(r["Is_International"])
    nodes += [("flag:keyword", "flag")] * int(r["suspicious_keyword_flag"])
    for name, kind in nodes:
        G.add_node(name, kind=kind)
        if G.has_edge(c, name):
            G[c][name]["txns"] += 1
            G[c][name]["frauds"] += int(r["Fraudulent"])
        else:
            G.add_edge(c, name, txns=1, frauds=int(r["Fraudulent"]))

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# PageRank personalised on fraud: customers with fraud get weight, spreads to linked entities

personal = {n: 0.0 for n in G}
for c, fr in df.groupby("Customer_ID")["Fraudulent"].sum().items():
    personal["cust:" + c] = float(fr)
total = sum(personal.values())
personal = {k: v / total for k, v in personal.items()}
pr = nx.pagerank(G, alpha=0.85, personalization=personal)
attr_nodes = [n for n, d in G.nodes(data=True) if d["kind"] != "customer"]
pr_tab = pd.Series({n: pr[n] for n in attr_nodes}).sort_values(ascending=False)
print("\nAttribute nodes ranked by fraud-personalised PageRank (top 10):")
print(pr_tab.head(10).round(4))


# PART 3 - Fraud co-occurrence network (what shows up together in fraud)

print("\n=== PART 3: fraud co-occurrence network ===")
fraud = df[df["Fraudulent"] == 1]
def items(r):
    out = [f"{k}:{r[c]}" for c, k in attr_cols.items()]
    if r["is_night"]: out.append("flag:night(0-5h)")
    if r["Is_International"]: out.append("flag:international")
    if r["suspicious_keyword_flag"]: out.append("flag:keyword")
    return out
co = {}
for _, r in fraud.iterrows():
    for a, b in itertools.combinations(sorted(items(r)), 2):
        co[(a, b)] = co.get((a, b), 0) + 1
H = nx.Graph()
for (a, b), w in co.items():
    H.add_edge(a, b, weight=w)

# node fraud rate for sizing

rates = {}
for col, k in attr_cols.items():
    for v, g in df.groupby(col)["Fraudulent"]:
        rates[f"{k}:{v}"] = g.mean()
rates["flag:night(0-5h)"] = df.loc[df.is_night == 1, "Fraudulent"].mean()
rates["flag:international"] = df.loc[df.Is_International == 1, "Fraudulent"].mean()
rates["flag:keyword"] = df.loc[df.suspicious_keyword_flag == 1, "Fraudulent"].mean()

# keep only strong edges for a readable picture

strong = [(a, b) for a, b, d in H.edges(data=True) if d["weight"] >= np.percentile(list(co.values()), 75)]
H2 = H.edge_subgraph(strong).copy()
comms = list(greedy_modularity_communities(H2, weight="weight"))
print(f"Communities found in the strong fraud network: {len(comms)}")
for i, c in enumerate(comms, 1):
    print(f"  Community {i}: {sorted(c)}")

colors = {"merchant": "#4C72B0", "payment": "#55A868", "device": "#C44E52",
          "location": "#8172B2", "flag": "#DD8452"}
pos = nx.spring_layout(H2, seed=7, k=0.9, weight="weight")
plt.figure(figsize=(12, 9))
sizes = [300 + 6000 * rates.get(n, base) for n in H2.nodes()]
cols = [colors[n.split(":")[0]] for n in H2.nodes()]
widths = [0.3 + 5 * H2[a][b]["weight"] / max(co.values()) for a, b in H2.edges()]
nx.draw_networkx_edges(H2, pos, width=widths, alpha=0.35)
nx.draw_networkx_nodes(H2, pos, node_size=sizes, node_color=cols, alpha=0.9)
nx.draw_networkx_labels(H2, pos, font_size=8)
plt.title("Fraud co-occurrence network (node size = fraud rate, edge width = shared fraud cases)")
plt.axis("off"); plt.tight_layout()
plt.savefig(os.path.join(PROJECT_DIR, "plots", "fraud_graph.png"), dpi=130); plt.close()

# export for Power BI / dashboard

node_rows = [{"node": n, "type": n.split(":")[0], "fraud_rate": round(rates.get(n, np.nan), 4),
              "pagerank": round(pr.get(n, 0), 6)} for n in H.nodes()]
pd.DataFrame(node_rows).to_csv(os.path.join(PROJECT_DIR, "graph_nodes.csv"), index=False)
pd.DataFrame([{"source": a, "target": b, "shared_fraud_cases": w} for (a, b), w in co.items()]) \
    .sort_values("shared_fraud_cases", ascending=False) \
    .to_csv(os.path.join(PROJECT_DIR, "graph_edges.csv"), index=False)
print("\nTop 10 attribute pairs most often seen together in fraud:")
print(pd.Series(co).sort_values(ascending=False).head(10))
print("\nSaved plots/fraud_graph.png, graph_nodes.csv, graph_edges.csv")
print("Done. Paste everything above back to me.")
