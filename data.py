# data.py -- adapter for market-data-hub
# Preserves existing import interface exactly.
# All other files import from here unchanged.

import os
import importlib

os.environ.setdefault(
    "MARKET_DB_PATH",
    os.path.join(os.path.expanduser("~"), "market_data", "hub.db"),
)

import pandas as pd

from market_data_hub import Hub as _Hub


PROJECT_NAME = "pattern-finder"
_hub = _Hub(project_name=PROJECT_NAME)


def _normalize_ohlcv(df):
    normalized = df.copy()
    normalized.index = pd.to_datetime(normalized.index)
    normalized.sort_index(inplace=True)

    if {"open", "high", "low", "close"}.issubset(normalized.columns):
        close = normalized["close"].astype(float)
        adj_close = normalized.get("adj_close")
        if adj_close is not None:
            adj_close = adj_close.astype(float)
            scale = (adj_close / close.replace(0, pd.NA)).fillna(1.0)
            normalized["open"] = normalized["open"].astype(float) * scale
            normalized["high"] = normalized["high"].astype(float) * scale
            normalized["low"] = normalized["low"].astype(float) * scale
            normalized["close"] = adj_close

        normalized = normalized.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )

    return normalized


def download_data(ticker=None, cache_file=None):
    from config import TICKER

    t = ticker or TICKER
    _hub.register([t])
    df = _normalize_ohlcv(_hub.ohlcv(t))
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    n = len(df)
    last = df.index[-1].date()
    print(f"[data] Loaded {n} bars for {t} (last={last}) from market-data-hub")
    return df


def download_vix():
    df = _normalize_ohlcv(_hub.ohlcv("^VIX"))
    if df.empty:
        return pd.Series(dtype=float)
    return df["Close"].rename("VIX")


def build_features(df):
    import config as _config

    legacy_cache_file = "data" + "_cache" + "." + "pkl"
    legacy_vix_cache_file = "vix" + "_cache" + "." + "pkl"

    if not hasattr(_config, "CACHE_FILE"):
        _config.CACHE_FILE = legacy_cache_file
    if not hasattr(_config, "VIX_CACHE_FILE"):
        _config.VIX_CACHE_FILE = legacy_vix_cache_file

    legacy_data = importlib.import_module("data_legacy")
    legacy_data.download_vix = download_vix

    return legacy_data.build_features(df)


__all__ = ["download_data", "download_vix", "build_features"]