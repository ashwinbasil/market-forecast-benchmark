"""Paired fold-level comparison of every model vs a chosen baseline.
Usage: python scripts/compare_models.py reports/results_folds_lstm_stage2.csv [baseline]
Default baseline: train_mean (isolates skill beyond drift).
Negative diff = model better than baseline. Holm correction across all tests in file.
"""
import sys
import pandas as pd
from mfb.stats import paired_tests


def main(path, base="train_mean"):
    out = paired_tests(pd.read_csv(path), base)
    pd.set_option("display.width", 220)
    print(f"baseline = {base}, tests = {len(out)}")
    print(out.round(5).to_string(index=False))
    out.to_csv(path.replace("folds", f"paired_{base}"), index=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "train_mean")
