import numpy as np
import pandas as pd
from mfb.splits import walk_forward_splits
from mfb.eval import run_walk_forward, summarize
from mfb.models import BASELINES


def test_no_leakage_train_before_test():
    for tr_end, te_start, te_end in walk_forward_splits(2000, 756, 63):
        assert tr_end <= te_start < te_end <= 2000


def test_folds_cover_without_overlap():
    folds = list(walk_forward_splits(1000, 500, 100))
    starts = [f[1] for f in folds]
    assert starts == [500, 600, 700, 800, 900]


def test_runner_all_baselines_shape():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 1200),
                  index=pd.bdate_range("2015-01-01", periods=1200))
    res = run_walk_forward(r, BASELINES, meta={"ticker": "X", "sector": "s"})
    assert set(res["model"]) == set(BASELINES)
    s = summarize(res)
    assert abs(s.loc[s.model == "zero", "rmse_vs_zero"].iloc[0] - 1) < 1e-9


def test_persistence_uses_only_past():
    tr = pd.Series([0.1, 0.2]); te = pd.Series([0.3, 0.4, 0.5])
    p = BASELINES["last_return"](tr, te)
    assert list(p) == [0.2, 0.3, 0.4]
