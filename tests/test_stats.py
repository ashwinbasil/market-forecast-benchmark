import numpy as np
import pandas as pd
from mfb.stats import holm, paired_tests


def test_holm_known_values():
    adj = holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_holm_caps_at_one():
    assert holm([0.6, 0.7]).max() == 1.0


def test_paired_tests_shape():
    rng = np.random.default_rng(0)
    rows = [dict(ticker=t, fold=f, model=m, rmse=0.01 + rng.normal(0, 1e-4))
            for t in "AB" for f in range(30) for m in ["zero", "train_mean", "lstm"]]
    out = paired_tests(pd.DataFrame(rows), "train_mean")
    assert len(out) == 4 and (out["p_wilcoxon_holm"] >= out["p_wilcoxon"]).all()
