# data.py — download, cache, normalize

import math
import os
import pickle
import re
import numpy as np
import pandas as pd
import yfinance as yf

from config import TICKER, CACHE_FILE, ROLLING_ATR_WINDOW, ENABLE_VIX, VIX_CACHE_FILE, ENABLE_DOW_EMBEDDING, ENABLE_MOMENTUM_FEATURE
from session_log_utils import append_session_log


def resolve_cache_file(ticker: str, cache_file: str = CACHE_FILE) -> str:
    if cache_file != CACHE_FILE:
        return cache_file

    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", ticker.strip().upper())
    legacy_default = cache_file
    if normalized == TICKER and os.path.exists(legacy_default):
        return legacy_default

    stem, ext = os.path.splitext(cache_file)
    ext = ext or ".pkl"
    return f"{stem}_{normalized}{ext}"


def download_data(ticker: str = TICKER, cache_file: str = CACHE_FILE) -> pd.DataFrame:
    resolved_cache = resolve_cache_file(ticker, cache_file)

    if os.path.exists(resolved_cache):
        with open(resolved_cache, "rb") as f:
            df = pickle.load(f)
        if len(df) >= 5000:
            print(f"[data] Loaded {len(df)} rows from cache ({resolved_cache})")
            return df
        append_session_log("DATA", f"Cache {resolved_cache} too short ({len(df)} bars); forcing full refresh", "data.py:28")

    print(f"[data] Downloading {ticker} from Yahoo Finance...")
    raw = yf.download(ticker, period="max", auto_adjust=True, progress=False)
    raw = raw[["Open", "High", "Low", "Close"]].copy()
    raw.columns = ["Open", "High", "Low", "Close"]
    raw.dropna(inplace=True)
    raw.index = pd.to_datetime(raw.index)
    raw.sort_index(inplace=True)

    if len(raw) < 5000:
        append_session_log("DATA", f"Insufficient history for {ticker}: {len(raw)} bars", "data.py:39")
        raise ValueError(f"Insufficient data: {len(raw)} bars. Expected 5000+.")

    with open(resolved_cache, "wb") as f:
        pickle.dump(raw, f)
    print(
        f"[data] Downloaded {len(raw)} bars for {ticker} "
        f"(from {raw.index[0].date()} to {raw.index[-1].date()})"
    )
    append_session_log("DATA", f"Downloaded {len(raw)} bars for {ticker}", "data.py:46")
    return raw


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
    """
    Returns a DataFrame with one row per bar, containing the 5 normalised features.
    Rows with NaN (warmup or missing prev_close) are dropped.
    """
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

    # VIX normalised z-score feature
    if ENABLE_VIX:
        try:
            vix = download_vix()
            vix = vix.reindex(feats.index).ffill(limit=5).fillna(0.0)
            vix_ma = vix.rolling(252, min_periods=1).mean()
            vix_std = vix.rolling(252, min_periods=1).std().fillna(1.0)
            vix_norm = (vix - vix_ma) / (vix_std + 1e-8)
            feats["vix_norm"] = vix_norm.values
        except Exception as e:
            print(f"[data] WARNING: VIX feature skipped ({e}), using 0.0")
            feats["vix_norm"] = 0.0

    # Day-of-week sinusoidal embedding
    if ENABLE_DOW_EMBEDDING:
        feats["dow_sin"] = [math.sin(2 * math.pi * d.dayofweek / 5) for d in feats.index]
        feats["dow_cos"] = [math.cos(2 * math.pi * d.dayofweek / 5) for d in feats.index]

    # Momentum: 20-day vs 60-day log-return ratio
    if ENABLE_MOMENTUM_FEATURE:
        _log_ret = np.log(c / c.shift(1))
        _mom20 = _log_ret.rolling(20, min_periods=1).mean()
        _mom60 = _log_ret.rolling(60, min_periods=1).mean()
        momentum = (_mom20 / (_mom60.abs() + 1e-8)).reindex(feats.index)
        feats["momentum"] = momentum.values

    return feats


def download_vix() -> pd.Series:
    """Download or load cached full-history ^VIX close prices."""
    if os.path.exists(VIX_CACHE_FILE):
        with open(VIX_CACHE_FILE, "rb") as f:
            vix = pickle.load(f)
        print(f"[data] Loaded VIX from cache: {len(vix)} bars")
        return vix

    print("[data] Downloading ^VIX (full history) from Yahoo Finance...")
    try:
        raw = yf.download("^VIX", period="max", auto_adjust=True, progress=False)
        if raw.empty:
            raise ValueError("Empty VIX data")
        vix = raw["Close"]
        if isinstance(vix, pd.DataFrame):
            vix = vix.iloc[:, 0]
        vix = vix.squeeze()
        vix.name = "vix"
        vix.index = pd.to_datetime(vix.index)
        vix.sort_index(inplace=True)
        with open(VIX_CACHE_FILE, "wb") as f:
            pickle.dump(vix, f)
        n_nan = int(vix.isna().sum())
        print(f"[data] VIX loaded: {len(vix)} bars, {n_nan} NaN filled")
        return vix
    except Exception as e:
        print(f"[data] WARNING: VIX download failed ({e}), using 0.0 fill")
        return pd.Series(dtype=float, name="vix")


def fallback_features(df: pd.DataFrame) -> np.ndarray:
    """
    Discrete hand-crafted feature vector (5-dim) used when autoencoder fails.
    Returns array shape (N_bars, 5) aligned with df index.
    """
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    hl = h - l + 1e-8
    body = (c - o) / hl
    upper_wick = (h - np.maximum(o, c)) / hl
    lower_wick = (np.minimum(o, c) - l) / hl
    body_dir = np.sign(body)
    body_size = np.abs(body)
    return np.stack([body_dir, body_size, upper_wick, lower_wick,
                     (hl / (hl.mean() + 1e-8))], axis=1)
