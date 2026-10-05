"""Market Forecast Benchmark dashboard. Reads CSVs in reports/. No model runs here.
Run: streamlit run app/dashboard.py
"""
import re
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mfb.stats import paired_tests  # noqa: E402

st.set_page_config(page_title="Market Forecast Benchmark", layout="wide")

MODEL_ORDER = ["zero", "train_mean", "arima_101", "lstm", "rolling_mean_20", "last_return"]
DEFAULT_MODELS = ["train_mean", "arima_101", "lstm", "rolling_mean_20"]


@st.cache_data
def load_folds():
    files = list(ROOT.glob("reports/**/results_folds_lstm*.csv"))
    frames = []
    for f in files:
        m = re.search(r"stage(\d+)", str(f))
        df = pd.read_csv(f, parse_dates=["test_start", "test_end"])
        df["stage"] = int(m.group(1)) if m else 1
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["stage", "ticker", "model", "fold"])


@st.cache_data
def summarize(folds):
    g = folds.groupby(["stage", "ticker", "sector", "model"], as_index=False)[
        ["rmse", "mae", "dir_acc", "up_rate"]].mean()
    z = g[g["model"] == "zero"][["ticker", "rmse"]].rename(columns={"rmse": "rmse_zero"})
    g = g.merge(z, on="ticker")
    g["rmse_vs_zero"] = g["rmse"] / g["rmse_zero"]
    return g.drop(columns="rmse_zero")


@st.cache_data
def significance(folds, base):
    parts = []
    for stage, g in folds.groupby("stage"):   # Holm family = one stage, as in README
        t = paired_tests(g, base)
        t.insert(0, "stage", stage)
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


folds = load_folds()
if folds.empty:
    st.error("No reports/**/results_folds_lstm*.csv found. Run scripts/run_baselines.py first.")
    st.stop()
summ = summarize(folds)

st.title("Market Forecast Benchmark")
st.markdown("**Does an LSTM predict daily stock returns better than doing nothing?** "
            "Walk-forward test on 12 assets against 5 baselines. Short answer: no.")

st.sidebar.header("Filters")
stages = sorted(folds["stage"].unique())
stage_names = {1: "1: AAPL", 2: "2: Sector stocks", 3: "3: Oil and gas"}
pick_stages = st.sidebar.multiselect("Stage", stages, default=stages,
                                     format_func=lambda s: stage_names.get(s, str(s)))
avail = [m for m in MODEL_ORDER if m in set(folds["model"]) and m != "zero"]
pick_models = st.sidebar.multiselect("Models", avail, default=[m for m in DEFAULT_MODELS if m in avail])
view = summ[summ["stage"].isin(pick_stages)]
vf = folds[folds["stage"].isin(pick_stages)]

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Asset deep dive", "Significance", "Method and caveats"])

with tab1:
    sig_all = significance(folds, "train_mean")
    cand = sig_all[~sig_all["model"].isin(["last_return", "rolling_mean_20"])]
    wins = int(((cand["p_wilcoxon_holm"] < 0.05) & (cand["pct_diff"] < 0)).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Assets", view["ticker"].nunique())
    c2.metric("Walk-forward folds per asset", int(folds.groupby("ticker")["fold"].nunique().median()))
    lstm = view[view["model"] == "lstm"]
    c3.metric("Assets where LSTM RMSE < zero forecast", f"{int((lstm['rmse_vs_zero'] < 1).sum())} / {len(lstm)}")
    c4.metric("Significant wins vs train-mean (Holm)", f"{wins} / {len(cand)}")
    st.caption(f"LSTM edges the zero forecast on {int((lstm['rmse_vs_zero'] < 1).sum())} of {len(lstm)} assets, but by under 1 percent and never significant after correction.")
    d = view[view["model"].isin(pick_models)].copy()
    d["ticker"] = pd.Categorical(d["ticker"], categories=list(dict.fromkeys(view["ticker"]))[::-1], ordered=True)
    fig = px.scatter(d.sort_values("ticker"), x="rmse_vs_zero", y="ticker", color="model", symbol="model",
                     hover_data={"rmse": ":.5f", "dir_acc": ":.3f", "up_rate": ":.3f"},
                     labels={"rmse_vs_zero": "RMSE / RMSE of zero forecast (below 1 = better)"})
    fig.add_vline(x=1.0, line_color="black")
    fig.update_traces(marker_size=11)
    fig.update_layout(height=120 + 38 * d["ticker"].nunique(), legend_title_text="")
    st.plotly_chart(fig, width="stretch")
    st.caption("Models cluster on the line. last_return (about 1.4x worse) is off scale; "
               "add it in the sidebar to see why.")

with tab2:
    asset = st.selectbox("Asset", list(dict.fromkeys(folds["ticker"])))
    a = summ[summ["ticker"] == asset]
    left, right = st.columns(2)
    bar = px.bar(a[a["model"] != "zero"].assign(delta=lambda x: (x["rmse_vs_zero"] - 1) * 100),
                 x="model", y="delta", labels={"delta": "RMSE vs zero forecast (%)"})
    bar.update_layout(height=340, title="Average error vs doing nothing")
    left.plotly_chart(bar, width="stretch")
    af = folds[(folds["ticker"] == asset)]
    z = af[af["model"] == "zero"].set_index("fold")["rmse"]
    t = af[af["model"].isin(pick_models)].copy()
    t["ratio"] = t["rmse"].values / z.reindex(t["fold"]).values
    line = px.line(t, x="test_start", y="ratio", color="model",
                   labels={"ratio": "Fold RMSE / zero forecast", "test_start": "Fold start"})
    line.add_hline(y=1.0, line_color="black")
    line.update_layout(height=340, title="Fold-by-fold: no stable edge over time", legend_title_text="")
    right.plotly_chart(line, width="stretch")
    st.markdown("**Directional accuracy vs always-up (`up_rate`)**")
    da = a[a["model"] != "zero"][["model", "rmse", "mae", "dir_acc", "up_rate"]].copy()
    da["dir_acc_minus_up_rate"] = da["dir_acc"] - da["up_rate"]
    st.dataframe(da.round(5), hide_index=True, width="stretch")

with tab3:
    base = st.radio("Baseline", ["train_mean", "zero"], horizontal=True,
                    help="train_mean isolates skill beyond drift.")
    sig = significance(folds, base)
    sig = sig[sig["stage"].isin(pick_stages) & sig["model"].isin(pick_models + ["last_return"])]
    show = sig[["stage", "ticker", "model", "pct_diff", "folds_beat_base", "folds",
                "p_wilcoxon", "p_wilcoxon_holm", "p_ttest_holm"]]
    st.dataframe(show.style.format({"pct_diff": "{:+.2f}", "p_wilcoxon": "{:.4f}",
                                    "p_wilcoxon_holm": "{:.4f}", "p_ttest_holm": "{:.4f}"})
                 .map(lambda v: "background-color: #7f1d1d; color: #ffffff" if isinstance(v, float) and v < 0.05 else "",
                      subset=["p_wilcoxon_holm", "p_ttest_holm"]),
                 hide_index=True, width="stretch", height=480)
    st.caption("pct_diff: mean fold RMSE difference vs baseline, negative = better. "
               "Holm correction applied within each stage across all asset-model tests. "
               "Red cells: adjusted p < 0.05 (check pct_diff sign: red with positive diff means significantly WORSE).")

with tab4:
    st.markdown("""
**Target:** next-day log return, split/dividend adjusted. **Validation:** expanding-window walk-forward,
3 years initial train, 63-day test blocks, about 30 folds per asset. One-step-ahead for every model.

**Models:** zero, train mean, last return, 20-day mean, ARIMA(1,0,1), LSTM (32 units, window 60, early stopping,
seed 42, train-only scaling).

**Reading the numbers:** ratio near 1.0 = same as doing nothing. Positive drift gives train-mean, ARIMA and LSTM
a shared ~0.2 to 0.3 percent edge over zero; none beats train-mean after Holm correction.

**Caveats**
- CL=F negative settle on 2020-04-20 dropped; CL=F and NG=F are continuous front-month futures with roll jumps.
- Folds share training history, so p-values are optimistic. Null conclusion is conservative.
- One LSTM configuration, not tuned after seeing results. Daily horizon. Not investment advice.
""")
    st.markdown("Code and write-up: github.com/ashwinbasil/market-forecast-benchmark")
