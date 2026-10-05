"""Dot plot: RMSE ratio vs zero-return forecast, per asset and model.
Reads all reports/**/results_summary_lstm*.csv. Writes reports/figures/rmse_vs_zero.png
Usage: python scripts/make_figure.py
"""
import re
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
MODELS = {  # name: (marker, colour)
    "train_mean": ("s", "#7f7f7f"),
    "arima_101": ("^", "#1f77b4"),
    "lstm": ("o", "#d62728"),
    "rolling_mean_20": ("x", "#2ca02c"),
}


def main():
    def stage_key(f):
        m = re.search(r"stage(\d+)", str(f))
        return int(m.group(1)) if m else 99

    files = sorted(ROOT.glob("reports/**/results_summary_lstm*.csv"), key=stage_key)
    if not files:
        raise SystemExit("no results_summary_lstm*.csv found under reports/")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(["ticker", "model"], keep="last")
    order = list(dict.fromkeys(df["ticker"]))
    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(order) + 1.6))
    for name, (mk, col) in MODELS.items():
        d = df[df["model"] == name].set_index("ticker").reindex(order)
        ax.scatter(d["rmse_vs_zero"], range(len(order)), marker=mk, s=42,
                   color=col, label=name, zorder=3)
    ax.axvline(1.0, color="black", lw=1, zorder=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlim(0.985, 1.035)
    ax.set_xlabel("Walk-forward RMSE / RMSE of 'tomorrow's return = 0'   (below 1 = better than doing nothing)")
    ax.set_title("No model meaningfully beats a flat forecast on daily returns")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.text(0.01, 0.005, "last_return (about 1.4x worse on every asset) omitted: off scale.",
             fontsize=7, color="#555")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out = ROOT / "reports" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "rmse_vs_zero.png", dpi=160)
    print("wrote", out / "rmse_vs_zero.png")


if __name__ == "__main__":
    main()
