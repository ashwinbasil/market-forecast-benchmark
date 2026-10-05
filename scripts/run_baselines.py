"""Run baselines (and optionally LSTM) on every asset in a stage.
Usage: python scripts/run_baselines.py --stage 1 [--lstm]
"""
import argparse
import pandas as pd
from mfb.data import load_universe, load_prices, log_returns, ROOT
from mfb.models import BASELINES
from mfb.eval import run_walk_forward, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=None, help="stage prefix e.g. 1 or 2")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--lstm", action="store_true", help="add LSTM (slow)")
    a = ap.parse_args()

    models = dict(BASELINES)
    if a.lstm:
        from mfb.lstm import lstm_factory
        models["lstm"] = lstm_factory()

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
        print(f"{t}: {len(r)} returns", flush=True)
        out.append(run_walk_forward(r, models, meta={"ticker": t, "sector": asset["sector"]}))
    res = pd.concat(out, ignore_index=True)
    rep = ROOT / "reports"
    rep.mkdir(exist_ok=True)
    tag = ("_lstm" if a.lstm else "") + (f"_stage{a.stage}" if a.stage else "")
    res.to_csv(rep / f"results_folds{tag}.csv", index=False)
    summ = summarize(res)
    summ.to_csv(rep / f"results_summary{tag}.csv", index=False)
    print(summ.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
