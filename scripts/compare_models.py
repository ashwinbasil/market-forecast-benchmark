"""Paired fold-level comparison of every model vs the zero-return baseline.
Usage: python scripts/compare_models.py reports/results_folds_lstm.csv
Reports: mean RMSE diff, folds won, paired t-test and Wilcoxon p-values.
Negative diff = model better than zero. p < 0.05 AND consistent wins
needed before claiming any edge. With ~30 folds, power is low: be modest.
"""
import sys
import pandas as pd
from scipy import stats


BASE = sys.argv[2] if len(sys.argv) > 2 else "zero"


def main(path):
    df = pd.read_csv(path)
    rows = []
    for t, g in df.groupby("ticker"):
        piv = g.pivot(index="fold", columns="model", values="rmse")
        for m in piv.columns:
            if m == BASE:
                continue
            d = (piv[m] - piv[BASE]).dropna()
            if d.abs().sum() == 0:
                p_t = p_w = float("nan")
            else:
                p_t = stats.ttest_1samp(d, 0.0).pvalue
                p_w = stats.wilcoxon(d).pvalue
            rows.append({
                "ticker": t, "model": m, "folds": len(d),
                "mean_rmse_diff": d.mean(),
                "pct_diff": 100 * d.mean() / piv[BASE].mean(),
                "folds_beat_base": int((d < 0).sum()),
                "p_ttest": p_t, "p_wilcoxon": p_w,
            })
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(out.round(6).to_string(index=False))
    out.to_csv(path.replace("folds", "paired"), index=False)


if __name__ == "__main__":
    main(sys.argv[1])
