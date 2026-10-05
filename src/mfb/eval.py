"""Metrics and walk-forward runner. Output is one tidy DataFrame."""
import numpy as np
import pandas as pd
from .splits import walk_forward_splits


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def directional_accuracy(y, p):
    """Share of days sign(pred)==sign(actual). NaN if model never takes a side."""
    mask = p != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.sign(y[mask]) == np.sign(p[mask])))


def run_walk_forward(returns, models, meta=None, initial_train=756, test_size=63):
    """returns: Series of daily log returns. models: dict name -> fn(train, test).
    Returns tidy DataFrame, one row per (model, fold)."""
    meta = meta or {}
    rows = []
    n = len(returns)
    for fold, (tr_end, te_start, te_end) in enumerate(
        walk_forward_splits(n, initial_train, test_size)
    ):
        train = returns.iloc[:tr_end]
        test = returns.iloc[te_start:te_end]
        y = test.to_numpy()
        for name, fn in models.items():
            p = np.asarray(fn(train, test), dtype=float)
            assert len(p) == len(y), f"{name}: pred len {len(p)} != {len(y)}"
            rows.append({
                **meta, "model": name, "fold": fold,
                "test_start": test.index[0], "test_end": test.index[-1],
                "rmse": rmse(y, p), "mae": mae(y, p),
                "dir_acc": directional_accuracy(y, p),
                "up_rate": float(np.mean(y > 0)),
            })
    return pd.DataFrame(rows)


def summarize(results):
    """Mean per (ticker, model) plus ratio of RMSE vs zero baseline.
    ratio < 1 means model beats zero-return forecast."""
    keys = [c for c in ["ticker", "sector"] if c in results.columns]
    g = results.groupby(keys + ["model"], as_index=False)[
        ["rmse", "mae", "dir_acc", "up_rate"]].mean()
    base = g[g["model"] == "zero"][keys + ["rmse"]].rename(columns={"rmse": "rmse_zero"})
    g = g.merge(base, on=keys)
    g["rmse_vs_zero"] = g["rmse"] / g["rmse_zero"]
    return g.drop(columns="rmse_zero")
