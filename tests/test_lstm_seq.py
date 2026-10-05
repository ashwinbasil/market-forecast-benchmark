import numpy as np
from mfb.lstm import make_sequences


def test_sequence_never_sees_target_or_future():
    v = np.arange(100, dtype=float)
    X, y = make_sequences(v, window=10, start=50, end=60)
    assert X.shape == (10, 10, 1)
    for k in range(10):
        i = 50 + k
        assert y[k] == v[i]
        assert X[k, :, 0].max() == v[i - 1]
