# data.py -- adapter for market-data-hub
# Preserves existing import interface exactly.
# All other files import from here unchanged.

import math
import os

os.environ.setdefault(
    "MARKET_DB_PATH",
    os.path.join(os.path.expanduser("~"), "market_data", "hub.db"),
)

import numpy as np
import pandas as pd

from config import ENABLE_DOW_EMBEDDING, ENABLE_MOMENTUM_FEATURE, ENABLE_VIX, ROLLING_ATR_WINDOW
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


def compute_atr(df: pd.DataFrame, window: int = ROLLING_ATR_WINDOW) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    hl = h - l
    denom = hl + 1e-8

    body = (c - o) / denom
    upper_wick = (h - np.maximum(o, c)) / denom
    lower_wick = (np.minimum(o, c) - l) / denom
    gap = (o - c.shift(1)) / (c.shift(1) + 1e-8)
    atr = compute_atr(df, ROLLING_ATR_WINDOW)
    rel_range = hl / (atr + 1e-8)

    feats = pd.DataFrame(
        {
            "body": body,
            "upper_wick": upper_wick,
            "lower_wick": lower_wick,
            "gap": gap,
            "rel_range": rel_range,
        },
        index=df.index,
    )
    feats.dropna(inplace=True)

    if ENABLE_VIX:
        try:
            vix = download_vix()
            vix = vix.reindex(feats.index).ffill(limit=5).fillna(0.0)
            vix_ma = vix.rolling(252, min_periods=1).mean()
            vix_std = vix.rolling(252, min_periods=1).std().fillna(1.0)
            vix_norm = (vix - vix_ma) / (vix_std + 1e-8)
            feats["vix_norm"] = vix_norm.values
        except Exception as exc:
            print(f"[data] WARNING: VIX feature skipped ({exc}), using 0.0")
            feats["vix_norm"] = 0.0

    if ENABLE_DOW_EMBEDDING:
        feats["dow_sin"] = [math.sin(2 * math.pi * d.dayofweek / 5) for d in feats.index]
        feats["dow_cos"] = [math.cos(2 * math.pi * d.dayofweek / 5) for d in feats.index]

    if ENABLE_MOMENTUM_FEATURE:
        log_ret = np.log(c / c.shift(1))
        mom20 = log_ret.rolling(20, min_periods=1).mean()
        mom60 = log_ret.rolling(60, min_periods=1).mean()
        momentum = (mom20 / (mom60.abs() + 1e-8)).reindex(feats.index)
        feats["momentum"] = momentum.values

    return feats


__all__ = ["download_data", "download_vix", "build_features"]