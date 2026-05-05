from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import ceil

import numpy as np

import pandas as pd

import backtest as backtest_module
from backtest import run_backtest
from autoencoder import compute_embeddings, train_autoencoder
from config import FORWARD_WINDOW, RETRAIN_EVERY_DAYS, TICKER, WARMUP_BARS
from data import build_features, download_data
from patterns import cosine_similarity_matrix
from reporting import compute_stats, create_equity_figure, create_pattern_similarity_figure


@dataclass
class BacktestRequest:
    ticker: str = TICKER
    start_date: str | pd.Timestamp | None = None
    end_date: str | pd.Timestamp | None = None
    lookback_days: int | None = None
    max_bars: int | None = None
    warmup_bars: int | None = None
    target_trades_per_month: float | None = None


def _discrete_window_embedding(window: np.ndarray) -> np.ndarray:
    flat = window.flatten().astype(np.float32)
    out = np.zeros(flat.shape[0], dtype=np.float32)
    out[:len(flat)] = flat
    return out


def _non_overlapping_top_matches(similarities: np.ndarray, seq_len: int,
                                 max_idx_exclusive: int, top_k: int) -> list[int]:
    order = np.argsort(-similarities)
    selected: list[int] = []
    last_end = -1
    for idx in order:
        idx = int(idx)
        if idx >= max_idx_exclusive:
            continue
        if idx > last_end:
            selected.append(idx)
            last_end = idx + seq_len - 1
        if len(selected) >= top_k:
            break
    selected.sort()
    return selected


def _normalize_timestamp(value: str | pd.Timestamp | None) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    return pd.Timestamp(value).normalize()


def _normalize_ticker(value: str) -> str:
    ticker = value.strip().upper()
    if not ticker:
        raise ValueError("Ticker cannot be empty")
    return ticker


def _filter_data_window(
    df: pd.DataFrame,
    feat_df: pd.DataFrame,
    start_date: str | pd.Timestamp | None,
    end_date: str | pd.Timestamp | None,
    lookback_days: int | None,
    max_bars: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp | None, pd.Timestamp | None]:
    shared_idx = df.index.intersection(feat_df.index)
    if len(shared_idx) == 0:
        raise ValueError("No aligned OHLC/features rows available")

    start_ts = _normalize_timestamp(start_date)
    end_ts = _normalize_timestamp(end_date)
    if lookback_days is not None:
        if lookback_days <= 0:
            raise ValueError("lookback_days must be a positive integer")
        if start_ts is None:
            raise ValueError("start_date is required when lookback_days is set")
        end_ts = start_ts + pd.Timedelta(days=lookback_days - 1)

    if start_ts is not None or end_ts is not None:
        lower = start_ts if start_ts is not None else shared_idx[0]
        upper = end_ts if end_ts is not None else shared_idx[-1]
        mask = (shared_idx >= lower) & (shared_idx <= upper)
        shared_idx = shared_idx[mask]
        if len(shared_idx) == 0:
            raise ValueError("No aligned bars left after date filtering")

    if max_bars is not None:
        if max_bars <= 0:
            raise ValueError("max_bars must be a positive integer")
        shared_idx = shared_idx[-max_bars:]

    return df.loc[shared_idx].copy(), feat_df.loc[shared_idx].copy(), start_ts, end_ts


def get_history_summary(ticker: str) -> dict:
    normalized_ticker = _normalize_ticker(ticker)
    df = download_data(ticker=normalized_ticker)
    feat_df = build_features(df)
    shared_idx = df.index.intersection(feat_df.index)
    if len(shared_idx) == 0:
        raise ValueError("No aligned OHLC/features rows available")

    start_date = shared_idx[0]
    end_date = shared_idx[-1]
    max_lookback_days = int((end_date - start_date).days) + 1
    return {
        "ticker": normalized_ticker,
        "start_date": start_date,
        "end_date": end_date,
        "bars": len(shared_idx),
        "max_lookback_days": max_lookback_days,
    }


def estimate_backtest_runtime(request: BacktestRequest) -> dict:
    ticker = _normalize_ticker(request.ticker)
    df = download_data(ticker=ticker)
    feat_df = build_features(df)
    df, feat_df, start_ts, end_ts = _filter_data_window(
        df,
        feat_df,
        request.start_date,
        request.end_date,
        request.lookback_days,
        request.max_bars,
    )

    bars = len(df)
    warmup_bars = request.warmup_bars or WARMUP_BARS
    effective_warmup = min(warmup_bars, max(20, bars // 5))
    iterations = max(0, bars - FORWARD_WINDOW - 1 - effective_warmup)
    approx_retrains = max(1, ceil(iterations / max(RETRAIN_EVERY_DAYS, 1))) if iterations else 0

    low_seconds = iterations * 0.45
    mid_seconds = iterations * 0.65
    high_seconds = iterations * 0.90

    return {
        "ticker": ticker,
        "bars": bars,
        "effective_warmup": effective_warmup,
        "iterations": iterations,
        "approx_retrains": approx_retrains,
        "eta_seconds_low": low_seconds,
        "eta_seconds_mid": mid_seconds,
        "eta_seconds_high": high_seconds,
        "actual_start": df.index[0] if len(df) else start_ts,
        "actual_end": df.index[-1] if len(df) else end_ts,
    }


def run_backtest_request(
    request: BacktestRequest,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    ticker = _normalize_ticker(request.ticker)
    df = download_data(ticker=ticker)
    feat_df = build_features(df)
    print(f"[runner] Feature columns ({len(feat_df.columns)}): {list(feat_df.columns)}")
    df, feat_df, start_ts, end_ts = _filter_data_window(
        df,
        feat_df,
        request.start_date,
        request.end_date,
        request.lookback_days,
        request.max_bars,
    )

    original_warmup = backtest_module.WARMUP_BARS
    original_target_trades_per_month = backtest_module.TARGET_TRADES_PER_MONTH
    if request.warmup_bars is not None:
        if request.warmup_bars <= 0:
            raise ValueError("warmup_bars must be a positive integer")
        backtest_module.WARMUP_BARS = request.warmup_bars
    if request.target_trades_per_month is not None:
        if request.target_trades_per_month <= 0:
            raise ValueError("target_trades_per_month must be positive")
        backtest_module.TARGET_TRADES_PER_MONTH = float(request.target_trades_per_month)

    try:
        trades_df = run_backtest(df, feat_df, progress_callback=progress_callback)
    finally:
        backtest_module.WARMUP_BARS = original_warmup
        backtest_module.TARGET_TRADES_PER_MONTH = original_target_trades_per_month

    stats = compute_stats(trades_df) if not trades_df.empty else {}
    figure = create_equity_figure(trades_df) if not trades_df.empty else None
    actual_start = df.index[0] if len(df) else start_ts
    actual_end = df.index[-1] if len(df) else end_ts

    return {
        "ticker": ticker,
        "df": df,
        "features": feat_df,
        "trades_df": trades_df,
        "stats": stats,
        "figure": figure,
        "actual_start": actual_start,
        "actual_end": actual_end,
        "bars": len(df),
        "requested_start": start_ts,
        "requested_end": end_ts,
        "lookback_days": request.lookback_days,
        "warmup_bars": request.warmup_bars or original_warmup,
        "target_trades_per_month": request.target_trades_per_month or original_target_trades_per_month,
    }


def analyze_pattern_similarity(df: pd.DataFrame, feat_df: pd.DataFrame,
                               anchor_date: str | pd.Timestamp,
                               seq_len: int, top_k: int = 5) -> dict:
    if seq_len <= 1:
        raise ValueError("seq_len must be greater than 1")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    shared_idx = df.index.intersection(feat_df.index)
    if len(shared_idx) == 0:
        raise ValueError("No aligned rows available for similarity analysis")

    anchor_ts = _normalize_timestamp(anchor_date)
    if anchor_ts not in shared_idx:
        raise ValueError("Selected date is not present in the aligned dataset")

    anchor_idx = int(shared_idx.get_loc(anchor_ts))
    if anchor_idx < seq_len:
        raise ValueError("Selected date is too early for the chosen pattern length")

    feat_arr = feat_df.loc[shared_idx].values.astype(np.float32)
    close_arr = df.loc[shared_idx, "Close"].values.astype(np.float64)
    history_arr = feat_arr[:anchor_idx]
    if len(history_arr) <= seq_len:
        raise ValueError("Not enough historical bars before the selected date")

    use_fallback = False
    try:
        model = train_autoencoder(history_arr, seq_len, device="cpu")
        corpus_embs = compute_embeddings(model, history_arr, seq_len, device="cpu")
        query_window = feat_arr[anchor_idx - seq_len: anchor_idx][np.newaxis].astype(np.float32)
        query_emb = model.encode(__import__("torch").from_numpy(query_window)).cpu().numpy()[0]
    except Exception:
        use_fallback = True
        windows = np.stack([history_arr[i: i + seq_len] for i in range(len(history_arr) - seq_len + 1)], axis=0)
        corpus_embs = np.array([_discrete_window_embedding(window) for window in windows], dtype=np.float32)
        query_emb = _discrete_window_embedding(feat_arr[anchor_idx - seq_len: anchor_idx])

    max_search_idx = min(anchor_idx - seq_len, len(corpus_embs))
    if max_search_idx <= 0:
        raise ValueError("Not enough prior windows to compare against")

    similarities = cosine_similarity_matrix(query_emb, corpus_embs[:max_search_idx])
    match_indices = _non_overlapping_top_matches(similarities, seq_len, max_search_idx, top_k)

    rows = []
    for idx in match_indices:
        match_close = close_arr[idx: idx + seq_len]
        rows.append(
            {
                "match_start_date": shared_idx[idx],
                "match_end_date": shared_idx[idx + seq_len - 1],
                "similarity": float(similarities[idx]),
                "close_path": match_close.tolist(),
            }
        )

    matches_df = pd.DataFrame(rows)
    query_close = close_arr[anchor_idx - seq_len: anchor_idx]
    figure = create_pattern_similarity_figure(anchor_ts, query_close, matches_df)

    if not matches_df.empty:
        matches_df = matches_df.drop(columns=["close_path"])

    return {
        "anchor_date": anchor_ts,
        "seq_len": seq_len,
        "query_start_date": shared_idx[anchor_idx - seq_len],
        "query_end_date": shared_idx[anchor_idx - 1],
        "query_close": query_close,
        "matches_df": matches_df,
        "figure": figure,
        "used_fallback_embedding": use_fallback,
    }