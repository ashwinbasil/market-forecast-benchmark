"""Baseline models. Each: fn(train: Series, test: Series) -> np.ndarray of
one-step-ahead predictions for test[i] using info up to test[i-1] only."""
import warnings
import numpy as np
import pandas as pd


def zero(train, test):
    """Random-walk price: expected return 0."""
    return np.zeros(len(test))


def train_mean(train, test):
    return np.full(len(test), train.mean())


def last_return(train, test):
    """Persistence: tomorrow = today."""
    full = pd.concat([train.iloc[-1:], test.iloc[:-1]]).to_numpy()
    return full


def rolling_mean_20(train, test):
    full = pd.concat([train.iloc[-20:], test])
    return full.rolling(20).mean().shift(1).iloc[20:].to_numpy()


def arima_101(train, test):
    from statsmodels.tsa.arima.model import ARIMA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ARIMA(train.to_numpy(), order=(1, 0, 1)).fit()
        ext = res.append(test.to_numpy(), refit=False)
        pred = ext.predict(start=len(train), end=len(train) + len(test) - 1)
    return np.asarray(pred)


BASELINES = {
    "zero": zero,
    "train_mean": train_mean,
    "last_return": last_return,
    "rolling_mean_20": rolling_mean_20,
    "arima_101": arima_101,
}
