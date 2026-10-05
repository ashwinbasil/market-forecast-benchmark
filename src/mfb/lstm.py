"""LSTM on next-day log return, fixed from the dissertation version:
- target is return, not price
- scaling stats from TRAIN only (no leakage)
- chronological validation split + early stopping (not 1 epoch)
- seeded
- same walk-forward folds and one-step-ahead contract as baselines
"""
import os
import numpy as np
import pandas as pd

WINDOW = 60


def make_sequences(values, window, start, end):
    """Rows i in [start, end): X[i] = values[i-window:i], y[i] = values[i].
    X never contains values[i] or later. Needs start >= window."""
    idx = np.arange(start, end)
    X = np.stack([values[i - window:i] for i in idx])[..., None]
    y = values[idx]
    return X.astype("float32"), y.astype("float32")


def lstm_factory(window=WINDOW, units=32, max_epochs=30, patience=3,
                 batch_size=64, seed=42, val_frac=0.1):
    def lstm_model(train, test):
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)

        mu, sd = train.mean(), train.std()          # train stats only
        full = ((pd.concat([train, test]) - mu) / sd).to_numpy()
        n_tr = len(train)

        X, y = make_sequences(full, window, window, n_tr)
        n_val = max(int(len(X) * val_frac), 1)
        Xtr, ytr, Xva, yva = X[:-n_val], y[:-n_val], X[-n_val:], y[-n_val:]

        model = tf.keras.Sequential([
            tf.keras.layers.Input((window, 1)),
            tf.keras.layers.LSTM(units),
            tf.keras.layers.Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=max_epochs,
                  batch_size=batch_size, verbose=0, shuffle=False,
                  callbacks=[tf.keras.callbacks.EarlyStopping(
                      patience=patience, restore_best_weights=True)])

        Xte, _ = make_sequences(full, window, n_tr, n_tr + len(test))
        pred = model.predict(Xte, verbose=0).ravel() * sd + mu
        tf.keras.backend.clear_session()
        return pred
    return lstm_model
