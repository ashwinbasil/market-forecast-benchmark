"""Paired fold-level comparison of every model vs a chosen baseline.
Usage: python scripts/compare_models.py reports/results_folds_lstm_stage2.csv [baseline]
Default baseline: train_mean (isolates skill beyond drift).
Negative diff = model better than baseline.
Holm correction applied across ALL (ticker, model) tests in the file,
because 7 tickers x 4 models = many chances for a fluke.
"""
import sys
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def main(path, base="train_mean"):
    df = pd.read_csv(path)
    rows = []
    for t, g in df.groupby("ticker"):
        piv = g.pivot(index="fold", columns="model", values="rmse")
        for m in piv.columns:
            if m == base:
                continue
            d = (piv[m] - piv[base]).dropna()
            if d.abs().sum() == 0:
                p_t = p_w = 1.0
            else:
                p_t = stats.ttest_1samp(d, 0.0).pvalue
                p_w = stats.wilcoxon(d).pvalue
            rows.append({
                "ticker": t, "model": m, "folds": len(d),
                "pct_diff": 100 * d.mean() / piv[base].mean(),
                "folds_beat_base": int((d < 0).sum()),
                "p_ttest": p_t, "p_wilcoxon": p_w,
            })
    out = pd.DataFrame(rows)
    out["p_wilcoxon_holm"] = multipletests(out["p_wilcoxon"], method="holm")[1]
    out["p_ttest_holm"] = multipletests(out["p_ttest"], method="holm")[1]
    pd.set_option("display.width", 220)
    print(f"baseline = {base}, tests = {len(out)}")
    print(out.round(5).to_string(index=False))
    out.to_csv(path.replace("folds", f"paired_{base}"), index=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "train_mean")
