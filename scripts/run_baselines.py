"""Run baselines on every asset in a stage.
Usage: python scripts/run_baselines.py --stage 1
"""
import argparse
from pathlib import Path
import pandas as pd
from mfb.data import load_universe, load_prices, log_returns, ROOT
from mfb.models import BASELINES
from mfb.eval import run_walk_forward, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=None, help="stage prefix e.g. 1 or 2")
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()

    start, end, assets = load_universe(stage_prefix=a.stage)
    out = []
    for asset in assets:
        t = asset["ticker"]
        try:
            px = load_prices(t, start, end, refresh=a.refresh)
        except Exception as e:
            print(f"skip {t}: {e}")
            continue
        r = log_returns(px["Close"])
        print(f"{t}: {len(r)} returns")
        out.append(run_walk_forward(r, BASELINES, meta={"ticker": t, "sector": asset["sector"]}))
    res = pd.concat(out, ignore_index=True)
    rep = ROOT / "reports"
    rep.mkdir(exist_ok=True)
    res.to_csv(rep / "results_folds.csv", index=False)
    summ = summarize(res)
    summ.to_csv(rep / "results_summary.csv", index=False)
    print(summ.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
