"""Expanding-window walk-forward splits. Train always strictly before test."""


def walk_forward_splits(n, initial_train=756, test_size=63, step=None):
    """Yield (train_end, test_start, test_end) index bounds.
    train = [0, train_end), test = [train_end, test_end).
    756 ~ 3y, 63 ~ 1 quarter of trading days.
    """
    step = step or test_size
    train_end = initial_train
    while train_end + test_size <= n:
        yield train_end, train_end, train_end + test_size
        train_end += step
