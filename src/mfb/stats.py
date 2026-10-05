"""Paired fold-level tests with Holm correction. Light deps: numpy, pandas, scipy."""
import numpy as np
import pandas as pd
from scipy import stats


def holm(p):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, idx in enumerate(np.argsort(p)):
        run = max(run, (m - rank) * p[idx])
        adj[idx] = min(1.0, run)
    return adj


def paired_tests(folds, base="train_mean"):
    """folds: tidy per-fold results (ticker, model, fold, rmse).
    Compares each model with `base` on fold-level RMSE. Negative diff = better.
    Holm applied across all (ticker, model) tests in `folds`."""
    rows = []
    for t, g in folds.groupby("ticker"):
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
    out["p_wilcoxon_holm"] = holm(out["p_wilcoxon"])
    out["p_ttest_holm"] = holm(out["p_ttest"])
    return out
