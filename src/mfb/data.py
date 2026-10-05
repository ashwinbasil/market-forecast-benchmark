"""Price loading with local parquet cache. No ticker is hardcoded."""
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_universe(path=None, stage_prefix=None):
    path = Path(path) if path else ROOT / "configs" / "universe.yaml"
    cfg = yaml.safe_load(path.read_text())
    assets = []
    for stage, items in cfg["stages"].items():
        if stage_prefix is None or stage.startswith(str(stage_prefix)):
            for it in items:
                assets.append({**it, "stage": stage})
    return cfg["start"], cfg["end"], assets


def _cache_path(ticker, start, end, cache_dir):
    safe = ticker.replace("=", "_").replace("^", "_").replace("/", "_")
    return Path(cache_dir) / f"{safe}_{start}_{end}.parquet"


def load_prices(ticker, start, end, cache_dir=None, refresh=False):
    """Daily OHLCV, split/dividend adjusted. Cached to data/raw."""
    cache_dir = Path(cache_dir) if cache_dir else ROOT / "data" / "raw"
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = _cache_path(ticker, start, end, cache_dir)
    if f.exists() and not refresh:
        return pd.read_parquet(f)
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Close"])
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    df.to_parquet(f)
    return df


def log_returns(close):
    """Next-day target is returns.shift(-1); here we only compute r_t."""
    return np.log(close).diff().dropna()
