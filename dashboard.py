"""
Step 7 - Fraud monitoring dashboard (Streamlit)

Install:  python -m pip install streamlit plotly
Run:      streamlit run dashboard.py
(run it from the project folder; it opens in your browser)

Reads files created in earlier steps:
  stream_scores.csv            (Step 6a)  - scored live-stream transactions
  unsupervised_scores_test.csv (Step 4)   - anomaly scores
  model_comparison.csv, unsupervised_comparison.csv, graph_nodes.csv, graph_edges.csv
  best_fraud_model.joblib      (Step 3)   - for the 'score a transaction' tool
"""
import json
import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def P(*parts):
    """Path of a data file: the project folder first, then demo_data/ (what gets deployed)."""
    direct = os.path.join(PROJECT_DIR, *parts)
    return direct if os.path.exists(direct) else os.path.join(PROJECT_DIR, "demo_data", *parts)


st.set_page_config(page_title="Fraud Monitoring", page_icon="🛡️", layout="wide")

TIER_ORDER = ["LOW", "MEDIUM", "HIGH"]
TIER_COLORS = {"LOW": "#2a9d8f", "MEDIUM": "#e9a23b", "HIGH": "#d1495b"}
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ----------------------------------------------------------------- data loading
@st.cache_data
def load_stream():
    if not os.path.exists(P("stream_scores.csv")):
        return None
    d = pd.read_csv(P("stream_scores.csv"), parse_dates=["Transaction_Date"])
    d["day_of_week"] = d["Transaction_Date"].dt.dayofweek
    d["day_name"] = d["day_of_week"].map(dict(enumerate(DAYS)))
    d["risk_tier"] = pd.Categorical(d["risk_tier"], TIER_ORDER, ordered=True)
    d["risk_score"] = (d["fraud_prob"] * 100).round(1)
    return d


@st.cache_data
def load_optional(name):
    return pd.read_csv(P(name)) if os.path.exists(P(name)) else None


stream = load_stream()
if stream is None:
    st.error("stream_scores.csv not found. Run `python stream_simulator.py` first (Step 6a).")
    st.stop()

anom = load_optional("unsupervised_scores_test.csv")
if anom is not None:
    stream = stream.merge(anom.drop(columns=["Fraudulent"]), on="Transaction_ID", how="left")

# ---------------------------------------------------------------------- sidebar
st.sidebar.title("Filters")
tiers = st.sidebar.multiselect("Risk tier", TIER_ORDER, default=TIER_ORDER)
locs = st.sidebar.multiselect("Location", sorted(stream["Location"].unique()),
                              default=sorted(stream["Location"].unique()))
merchants = st.sidebar.multiselect("Merchant category", sorted(stream["Merchant_Category"].unique()),
                                   default=sorted(stream["Merchant_Category"].unique()))
intl = st.sidebar.radio("International", ["All", "Yes", "No"], horizontal=True)
dmin, dmax = stream["Transaction_Date"].min().date(), stream["Transaction_Date"].max().date()
date_range = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
show_truth = st.sidebar.checkbox("Show confirmed fraud labels", value=True,
                                 help="In real life labels arrive later, after investigation.")

f = stream[stream["risk_tier"].isin(tiers) & stream["Location"].isin(locs)
           & stream["Merchant_Category"].isin(merchants)]
if intl != "All":
    f = f[f["Is_International"] == (1 if intl == "Yes" else 0)]
if isinstance(date_range, tuple) and len(date_range) == 2:
    f = f[(f["Transaction_Date"].dt.date >= date_range[0]) & (f["Transaction_Date"].dt.date <= date_range[1])]

# ------------------------------------------------------------------------ header
st.title("🛡️ Financial Fraud Monitoring")
st.caption("Scores come from the Step 3 model replaying held-out transactions (Step 6a stream). "
           "HIGH = at or above the tuned alert threshold; MEDIUM = worth a look; LOW = auto-clear.")

if f.empty:
    st.warning("No transactions match the current filters.")
    st.stop()

high = f[f["risk_tier"] == "HIGH"]
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Transactions", f"{len(f):,}")
k2.metric("HIGH-risk alerts", f"{len(high):,}")
k2.caption(f"{len(high) / len(f):.1%} of volume")
k3.metric("Amount under alert", f"{high['Transaction_Amount'].sum():,.0f}")
if show_truth:
    total_fraud = int(f["Fraudulent"].sum())
    caught = int(high["Fraudulent"].sum())
    med_caught = int(f[f["risk_tier"] == "MEDIUM"]["Fraudulent"].sum())
    k4.metric("Fraud caught (HIGH)", f"{caught}/{total_fraud}")
    k4.caption(f"{caught / total_fraud:.0%} recall" if total_fraud else "no fraud in view")
    k5.metric("Alert precision", f"{high['Fraudulent'].mean():.1%}" if len(high) else "n/a")
    k5.caption(f"{med_caught} more fraud sits in MEDIUM")
else:
    k4.metric("MEDIUM", f"{(f['risk_tier'] == 'MEDIUM').sum():,}")
    k5.metric("Avg risk score", f"{f['risk_score'].mean():.1f}")

tab_over, tab_heat, tab_anom, tab_alerts, tab_models, tab_score = st.tabs(
    ["Overview", "Heatmaps", "Anomaly scores", "Alerts", "Models & network", "Score a transaction"])

# ----------------------------------------------------------------------- Overview
with tab_over:
    c1, c2 = st.columns(2)
    tier_counts = f["risk_tier"].value_counts().reindex(TIER_ORDER).fillna(0).reset_index()
    tier_counts.columns = ["risk_tier", "transactions"]
    fig = px.bar(tier_counts, x="risk_tier", y="transactions", color="risk_tier",
                 color_discrete_map=TIER_COLORS, title="Transactions by risk tier", text="transactions")
    fig.update_layout(showlegend=False, xaxis_title=None)
    c1.plotly_chart(fig, use_container_width=True)

    if show_truth:
        rate = f.groupby("risk_tier", observed=True)["Fraudulent"].agg(["mean", "sum", "count"]).reindex(TIER_ORDER).reset_index()
        fig = px.bar(rate, x="risk_tier", y="mean", color="risk_tier", color_discrete_map=TIER_COLORS,
                     title="Confirmed fraud rate within each tier",
                     text=rate["mean"].map(lambda v: f"{v:.0%}"))
        fig.update_layout(showlegend=False, xaxis_title=None, yaxis_tickformat=".0%", yaxis_title="fraud rate")
        c2.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.histogram(f, x="risk_score", nbins=40, title="Distribution of risk scores")
        c2.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    for col, title, holder in [("Merchant_Category", "HIGH-risk alerts by merchant category", c3),
                               ("Location", "HIGH-risk alerts by location", c4)]:
        g = high.groupby(col).size().sort_values().reset_index(name="alerts")
        fig = px.bar(g, x="alerts", y=col, orientation="h", title=title,
                     color_discrete_sequence=[TIER_COLORS["HIGH"]])
        fig.update_layout(yaxis_title=None)
        holder.plotly_chart(fig, use_container_width=True)

    daily = f.assign(date=f["Transaction_Date"].dt.date).groupby(["date", "risk_tier"], observed=True).size().reset_index(name="n")
    fig = px.bar(daily, x="date", y="n", color="risk_tier", color_discrete_map=TIER_COLORS,
                 category_orders={"risk_tier": TIER_ORDER}, title="Daily volume by risk tier")
    fig.update_layout(xaxis_title=None, yaxis_title="transactions")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------- Heatmaps
with tab_heat:
    metric = st.radio("Colour by", ["Average risk score", "HIGH-risk alert count"]
                      + (["Confirmed fraud rate"] if show_truth else []), horizontal=True)
    if metric == "Average risk score":
        pv = f.pivot_table(index="day_of_week", columns="hour", values="risk_score", aggfunc="mean")
        scale, label = "Reds", "avg risk"
    elif metric == "HIGH-risk alert count":
        pv = high.pivot_table(index="day_of_week", columns="hour", values="Transaction_ID", aggfunc="count")
        scale, label = "Reds", "alerts"
    else:
        pv = f.pivot_table(index="day_of_week", columns="hour", values="Fraudulent", aggfunc="mean")
        scale, label = "Reds", "fraud rate"
    pv = pv.reindex(index=range(7), columns=range(24))
    pv.index = DAYS
    fig = px.imshow(pv, aspect="auto", color_continuous_scale=scale, labels=dict(x="hour of day", y="", color=label),
                    title=f"{metric}: day of week × hour")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Fraud is concentrated between 00:00 and 05:59. Empty cells mean no transactions in that slot after filtering.")

    c1, c2 = st.columns(2)
    pv2 = f.pivot_table(index="Location", columns="Merchant_Category", values="risk_score", aggfunc="mean")
    c1.plotly_chart(px.imshow(pv2, aspect="auto", color_continuous_scale="Reds",
                              title="Average risk: location × merchant"), use_container_width=True)
    pv3 = f.pivot_table(index="Payment_Method", columns="Device_Type", values="risk_score", aggfunc="mean")
    c2.plotly_chart(px.imshow(pv3, aspect="auto", color_continuous_scale="Reds",
                              title="Average risk: payment method × device"), use_container_width=True)
    st.caption("Location, merchant, payment and device barely change risk in this dataset (the pattern is mostly "
               "time, international and suspicious keyword), so these maps look fairly flat. That is a finding, not a bug.")

# ------------------------------------------------------------------ Anomaly scores
with tab_anom:
    score_cols = [c for c in f.columns if c.startswith("score_")]
    if not score_cols:
        st.info("Run Step 4 (`python step4_unsupervised.py`) to create unsupervised_scores_test.csv.")
    else:
        pick = st.selectbox("Anomaly model", score_cols, format_func=lambda c: c.replace("score_", "").replace("_", " ").title())
        c1, c2 = st.columns(2)
        if show_truth:
            fig = px.histogram(f.assign(label=f["Fraudulent"].map({0: "Legit", 1: "Fraud"})), x=pick, color="label",
                               nbins=40, barmode="overlay", opacity=0.65, histnorm="probability density",
                               color_discrete_map={"Legit": "#2a9d8f", "Fraud": "#d1495b"},
                               title="Anomaly score: fraud vs legit")
        else:
            fig = px.histogram(f, x=pick, nbins=40, title="Anomaly score distribution")
        c1.plotly_chart(fig, use_container_width=True)
        fig = px.scatter(f, x=pick, y="risk_score", color="risk_tier", color_discrete_map=TIER_COLORS,
                         category_orders={"risk_tier": TIER_ORDER}, opacity=0.6,
                         hover_data=["Transaction_ID", "Location"],
                         title="Anomaly score vs supervised risk score")
        fig.update_layout(yaxis_title="supervised risk score")
        c2.plotly_chart(fig, use_container_width=True)
        st.markdown("**Most anomalous transactions**")
        top = f.sort_values(pick, ascending=False).head(15)
        st.dataframe(top[["Transaction_ID", "Transaction_Date", "Location", "Transaction_Amount",
                          pick, "risk_score", "risk_tier"] + (["Fraudulent"] if show_truth else [])],
                     use_container_width=True, hide_index=True)
        st.caption("Anomaly scores are rescaled to 0-100. They flag unusual behaviour without using labels, "
                   "so they can catch new fraud patterns the supervised model has never seen.")

# ----------------------------------------------------------------------- Alerts
with tab_alerts:
    st.markdown(f"**{len(high)} HIGH-risk transactions** in the current view, highest risk first.")
    cols = ["Transaction_ID", "Transaction_Date", "Customer_ID", "Transaction_Amount", "Location",
            "Merchant_Category", "Device_Type", "risk_score", "risk_tier"]
    if show_truth:
        cols.append("Fraudulent")
    table = high.sort_values("risk_score", ascending=False)[cols]
    st.dataframe(table, use_container_width=True, hide_index=True,
                 column_config={"risk_score": st.column_config.ProgressColumn("risk score", min_value=0, max_value=100, format="%.0f")})
    st.download_button("Download alerts as CSV", table.to_csv(index=False).encode("utf-8"),
                       file_name="high_risk_alerts.csv", mime="text/csv")
    why = high[["is_night", "Is_International", "suspicious_keyword_flag"]].mean().rename(
        {"is_night": "Night (00-05h)", "Is_International": "International", "suspicious_keyword_flag": "Suspicious keyword"})
    fig = px.bar(why.reset_index(), x="index", y=0, title="What triggers the alerts (share of HIGH alerts with each flag)",
                 color_discrete_sequence=["#d1495b"])
    fig.update_layout(xaxis_title=None, yaxis_title="share", yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- Models & network
with tab_models:
    c1, c2 = st.columns(2)
    sup = load_optional("model_comparison.csv")
    uns = load_optional("unsupervised_comparison.csv")
    if sup is not None:
        c1.markdown("**Supervised models (test set)**")
        c1.dataframe(sup, hide_index=True, use_container_width=True)
    if uns is not None:
        c2.markdown("**Unsupervised models (test set)**")
        c2.dataframe(uns, hide_index=True, use_container_width=True)
    nodes, edges = load_optional("graph_nodes.csv"), load_optional("graph_edges.csv")
    if edges is not None:
        st.markdown("**Attributes most often seen together in fraud (graph analysis, Step 5)**")
        st.dataframe(edges.head(15), hide_index=True, use_container_width=True)
    graph_png = P("plots", "fraud_graph.png")
    if os.path.exists(graph_png):
        st.image(graph_png, caption="Fraud co-occurrence network")
    st.caption("The graph shows co-occurrence, not fraud rings: the dataset has no shared device, IP or account links.")

# -------------------------------------------------------------- Score a transaction
with tab_score:
    st.markdown("Try the model on a hypothetical transaction. Fields not shown use typical (median) values.")
    mp = P("best_fraud_model.joblib")

    @st.cache_data
    def scorer_defaults(features):
        """Typical values for fields the form does not ask for: from the small JSON shipped
        with the deployed app, or computed from features_clean.csv when run locally."""
        js, fc = P("scorer_defaults.json"), P("features_clean.csv")
        if os.path.exists(js):
            try:
                with open(js, encoding="utf-8") as fh:
                    return json.load(fh)
            except (ValueError, OSError):
                pass                      # unreadable file: fall back to features_clean.csv below
        if os.path.exists(fc):
            base = pd.read_csv(fc)
            row = {col: (float(base[col].median()) if pd.api.types.is_numeric_dtype(base[col])
                         else str(base[col].mode()[0])) for col in features}
            return {"row": row, "locations": sorted(base["Location"].unique().tolist()),
                    "merchants": sorted(base["Merchant_Category"].unique().tolist())}
        return None

    if not os.path.exists(mp):
        st.info("Needs best_fraud_model.joblib (run supervised_models.py, or make_demo_data.py for deployment).")
    else:
        bundle = joblib.load(mp)
        defaults = scorer_defaults(tuple(bundle["features"]))
        if defaults is None:
            st.info("Needs scorer_defaults.json or features_clean.csv.")
            st.stop()
        a, b, c = st.columns(3)
        hour = a.slider("Hour of day", 0, 23, 14)
        amount = a.number_input("Amount", 0.0, 5000.0, 80.0)
        is_intl = b.checkbox("International")
        kw = b.checkbox("Suspicious keyword")
        loc = c.selectbox("Location", defaults["locations"])
        merch = c.selectbox("Merchant", defaults["merchants"])
        row = dict(defaults["row"])
        night = int(0 <= hour <= 5)
        avg = float(row["Average_Spend"])
        row.update({"hour": hour, "Transaction_Amount": amount, "Is_International": int(is_intl),
                    "suspicious_keyword_flag": int(kw), "is_night": night, "Location": loc,
                    "Merchant_Category": merch, "log_amount": np.log1p(amount),
                    "amount_to_avg_ratio": amount / avg if avg else 0, "amount_minus_avg": amount - avg,
                    "risk_flag_count": night + int(is_intl) + int(kw), "intl_and_night": int(is_intl) * night})
        prob = float(bundle["pipeline"].predict_proba(pd.DataFrame([row])[bundle["features"]])[0, 1])
        t = bundle["threshold"]
        tier = "HIGH" if prob >= t else "MEDIUM" if prob >= t * 0.6 else "LOW"
        g = go.Figure(go.Indicator(mode="gauge+number", value=prob * 100, number={"suffix": "%"},
                                   title={"text": f"Risk tier: {tier}"},
                                   gauge={"axis": {"range": [0, 100]}, "bar": {"color": TIER_COLORS[tier]},
                                          "steps": [{"range": [0, t * 60], "color": "#e8f5f3"},
                                                    {"range": [t * 60, t * 100], "color": "#fcf0d9"},
                                                    {"range": [t * 100, 100], "color": "#f8dde1"}]}))
        g.update_layout(height=300)
        st.plotly_chart(g, use_container_width=True)
