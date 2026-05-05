# backtest.py — walk-forward loop

import time
import warnings
from collections.abc import Callable
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import (
    PATTERN_LENGTHS, WARMUP_BARS, RETRAIN_EVERY_DAYS, FORWARD_WINDOW,
    MIN_OCCURRENCES, COSINE_SIMILARITY_THRESHOLD,
    META_RANKER_CONFIDENCE_THRESHOLD, META_RANKER_MIN_TRADES,
    CAPITAL_PER_TRADE, ENABLE_SLIPPAGE, SLIPPAGE_PCT,
    EMBEDDING_DIM, SEED, FALLBACK_LOG,
    MIN_SIMILARITY_THRESHOLD_FLOOR, TARGET_TRADES_PER_MONTH,
    MATCH_VALIDATION_SPLIT, MATCH_VALIDATION_MIN_BARS,
    SL_MULT, TP_MULT, HMM_LOOKBACK,
)
from autoencoder import train_autoencoder, compute_embeddings, _build_windows
from patterns import (
    find_occurrences, collect_forward_paths, cosine_similarity_matrix,
    score_pattern, fit_hmm,
    FAISSIndex, _FAISS_AVAILABLE,
)
from meta_ranker import MetaRanker, build_feature_vector
from session_log_utils import append_session_log

warnings.filterwarnings("ignore")
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cpu"
MAX_DTW_PATHS = 40   # cap occurrences for DTW


def _emit_progress(progress_callback: Callable[[dict], None] | None, **payload):
    if progress_callback is None:
        return
    progress_callback(payload)


def _scaled_int(config_value: int, short_history_floor: int,
                history_bars: int, full_history_bars: int = 250) -> int:
    if history_bars <= 100:
        return short_history_floor
    if history_bars >= full_history_bars:
        return config_value
    frac = (history_bars - 100) / (full_history_bars - 100)
    scaled = short_history_floor + (config_value - short_history_floor) * frac
    return max(short_history_floor, int(round(scaled)))


def _scaled_float(config_value: float, short_history_floor: float,
                  history_bars: int, full_history_bars: int = 250) -> float:
    if history_bars <= 100:
        return short_history_floor
    if history_bars >= full_history_bars:
        return config_value
    frac = (history_bars - 100) / (full_history_bars - 100)
    return short_history_floor + (config_value - short_history_floor) * frac


def _adaptive_similarity_threshold(
    base_threshold: float,
    trades_taken: int,
    iterations_completed: int,
) -> float:
    if TARGET_TRADES_PER_MONTH <= 0:
        return base_threshold

    elapsed_months = max(iterations_completed / 21.0, 1.0)
    trade_rate = trades_taken / elapsed_months
    if trade_rate >= TARGET_TRADES_PER_MONTH:
        return base_threshold

    shortfall = 1.0 - (trade_rate / TARGET_TRADES_PER_MONTH)
    relaxed = base_threshold - (base_threshold - MIN_SIMILARITY_THRESHOLD_FLOOR) * shortfall
    return max(MIN_SIMILARITY_THRESHOLD_FLOOR, relaxed)


def calibrate_thresholds(df: pd.DataFrame, feat_df: pd.DataFrame) -> float:
    shared_idx = df.index.intersection(feat_df.index)
    if len(shared_idx) == 0:
        return COSINE_SIMILARITY_THRESHOLD

    feat_arr = feat_df.loc[shared_idx].values.astype(np.float32)
    n_bars = len(feat_arr)
    lower_bound = max(WARMUP_BARS, int(n_bars * 0.25))
    upper_bound = min(n_bars - FORWARD_WINDOW - 1, int(n_bars * 0.75))
    if upper_bound - lower_bound < 50:
        return COSINE_SIMILARITY_THRESHOLD

    thresholds = [0.65, 0.70, 0.75, 0.78, 0.82, 0.85]
    sample_count = min(50, upper_bound - lower_bound)
    rng = np.random.default_rng(SEED)
    sampled_days = np.sort(rng.choice(np.arange(lower_bound, upper_bound), size=sample_count, replace=False))
    totals = {threshold: [] for threshold in thresholds}

    for seq_len in PATTERN_LENGTHS:
        corpus = _discrete_embed_all(feat_arr, seq_len)
        if len(corpus) == 0:
            continue
        for bar_idx in sampled_days:
            if bar_idx < seq_len:
                continue
            query_emb = _discrete_window_emb(feat_arr[bar_idx - seq_len: bar_idx])
            max_search_idx = min(bar_idx - seq_len, len(corpus))
            if max_search_idx <= 0:
                continue
            for threshold in thresholds:
                count = len(find_occurrences(query_emb, corpus, seq_len, max_search_idx, threshold=threshold))
                totals[threshold].append(count)

    averages = {
        threshold: (float(np.mean(counts)) if counts else 0.0)
        for threshold, counts in totals.items()
    }
    target_occurrences = MIN_OCCURRENCES * 1.5
    calibrated = thresholds[0]
    for threshold in thresholds:
        if averages[threshold] >= target_occurrences:
            calibrated = threshold
            break

    if COSINE_SIMILARITY_THRESHOLD > calibrated:
        warning = (
            f"COSINE_SIMILARITY_THRESHOLD={COSINE_SIMILARITY_THRESHOLD:.2f} may be too high. "
            f"Calibration suggests {calibrated:.2f}. Auto-adjusting to {calibrated:.2f}."
        )
        print(f"[calibration] WARNING: {warning}")
        append_session_log("CALIBRATION", warning, "backtest.py:94")
    else:
        append_session_log(
            "CALIBRATION",
            f"Configured threshold {COSINE_SIMILARITY_THRESHOLD:.2f} retained; suggestion {calibrated:.2f}",
            "backtest.py:96",
        )
    return min(COSINE_SIMILARITY_THRESHOLD, calibrated)


# ─── Fallback helpers ──────────────────────────────────────────────────────

def _discrete_window_emb(window: np.ndarray) -> np.ndarray:
    flat = window.flatten().astype(np.float32)
    out = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    out[:min(len(flat), EMBEDDING_DIM)] = flat[:EMBEDDING_DIM]
    return out


def _discrete_embed_all(feat_array: np.ndarray, seq_len: int) -> np.ndarray:
    n = len(feat_array)
    if n < seq_len:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    return np.array(
        [_discrete_window_emb(feat_array[i: i + seq_len])
         for i in range(n - seq_len + 1)],
        dtype=np.float32,
    )


def _log_fallback(msg: str):
    with open(FALLBACK_LOG, "a") as f:
        f.write(msg + "\n")


# ─── Batch-precompute query embeddings ─────────────────────────────────────

def _batch_embed(model, feat_array: np.ndarray, seq_len: int,
                 is_fallback: bool, bar_indices: list) -> dict:
    """
    For each bar_idx in bar_indices, embed the window feat_array[bar_idx-seq_len:bar_idx].
    Returns dict: bar_idx -> emb (EMBEDDING_DIM,)
    Processes as a single batched forward pass.
    """
    result = {}
    valid = [(i, feat_array[i - seq_len: i]) for i in bar_indices
             if i >= seq_len and i - seq_len + seq_len <= len(feat_array)]

    if not valid:
        return result

    if is_fallback or model is None:
        for bar_idx, window in valid:
            result[bar_idx] = _discrete_window_emb(window)
        return result

    try:
        batch = np.stack([w for _, w in valid], axis=0).astype(np.float32)
        with torch.no_grad():
            t = torch.from_numpy(batch).to(DEVICE)
            embs = model.encode(t).cpu().numpy()
        if np.isnan(embs).any():
            raise ValueError("NaN in batch embeddings")
        for k, (bar_idx, _) in enumerate(valid):
            result[bar_idx] = embs[k]
    except Exception:
        for bar_idx, window in valid:
            result[bar_idx] = _discrete_window_emb(window)

    return result


def _build_hmm_feature_cache(close_arr: np.ndarray) -> np.ndarray:
    if len(close_arr) < 2:
        return np.empty((0, 3), dtype=np.float64)

    log_ret = np.log(close_arr[1:] / close_arr[:-1])
    prefix = np.concatenate(([0.0], np.cumsum(log_ret)))
    prefix_sq = np.concatenate(([0.0], np.cumsum(log_ret * log_ret)))
    idx = np.arange(1, len(close_arr))
    short_mask = idx >= 20
    long_mask = idx >= 60
    short_start = np.maximum(idx - 20, 0)
    long_start = np.maximum(idx - 60, 0)

    short_sum = prefix[idx] - prefix[short_start]
    short_sq_sum = prefix_sq[idx] - prefix_sq[short_start]
    short_mean = np.zeros(len(idx), dtype=np.float64)
    short_mean[short_mask] = short_sum[short_mask] / 20.0

    short_var = np.zeros(len(idx), dtype=np.float64)
    short_var[short_mask] = (short_sq_sum[short_mask] / 20.0) - (short_mean[short_mask] ** 2)
    short_var = np.clip(short_var, 0.0, None)
    roll_vol = np.sqrt(short_var)
    roll_trend = short_mean

    long_sum = prefix[idx] - prefix[long_start]
    long_mean = np.zeros(len(idx), dtype=np.float64)
    long_mean[long_mask] = long_sum[long_mask] / 60.0
    roll_mom = np.zeros(len(idx), dtype=np.float64)
    roll_mom[long_mask] = roll_trend[long_mask] / (np.abs(long_mean[long_mask]) + 1e-8)

    return np.stack([roll_vol, roll_trend, roll_mom], axis=1)


def _get_current_regime_cached(hmm_model, hmm_features: np.ndarray,
                               bar_idx: int, lookback: int = HMM_LOOKBACK) -> int:
    if hmm_model is None or len(hmm_features) == 0 or bar_idx <= 1:
        return 0
    try:
        end_idx = min(bar_idx - 1, len(hmm_features))
        feat_slice = hmm_features[:end_idx]
        if len(feat_slice) == 0:
            return 0
        feat_scaled = hmm_model._scaler.transform(feat_slice[-lookback:])
        states = hmm_model.predict(feat_scaled)
        return int(states[-1])
    except Exception:
        return 0


# ─── Embedding Cache ────────────────────────────────────────────────────────

class EmbeddingCache:
    """
    On retrain: builds corpus embeddings + precomputes query embeddings for
    the next RETRAIN_EVERY_DAYS bars in a single batched forward pass.
    """
    def __init__(self):
        self.models: dict = {}        # seq_len -> model
        self.corpus: dict = {}        # seq_len -> (n_windows, EMB_DIM) — up to retrain bar
        self.query_cache: dict = {}   # seq_len -> {bar_idx: emb}
        self.is_fallback: dict = {}
        self.faiss_indexes: dict = {} # seq_len -> FAISSIndex (when faiss-cpu available)
        self.last_retrain_date = None

    def retrain(self, feat_array: np.ndarray, retrain_date,
                next_n_bars: int, total_feat_array: np.ndarray,
                close_arr: np.ndarray | None = None):
        """
        feat_array: rows 0..bar_idx-1 (corpus)
        next_n_bars: how many future bars to precompute queries for
        total_feat_array: full feature array (rows 0..n_bars-1) — read-only for query windows
        close_arr: full close price array — used to compute outcome_returns for contrastive loss
        """
        self.last_retrain_date = retrain_date
        retrain_size = len(feat_array)
        future_bar_indices = list(range(retrain_size, retrain_size + next_n_bars + 1))

        # Compute per-seq_len outcome_returns for contrastive loss (no-lookahead safe)
        outcome_returns_map = {}
        if close_arr is not None:
            n_close = len(close_arr)
            for seq_len in PATTERN_LENGTHS:
                n_windows = max(0, retrain_size - seq_len + 1)
                ors = np.full(n_windows, np.nan, dtype=np.float64)
                for i in range(n_windows):
                    outcome_bar = i + seq_len
                    if outcome_bar + FORWARD_WINDOW < n_close:
                        c0 = close_arr[outcome_bar]
                        c1 = close_arr[outcome_bar + FORWARD_WINDOW]
                        if c0 > 0:
                            ors[i] = (c1 - c0) / c0
                outcome_returns_map[seq_len] = ors

        for seq_len in PATTERN_LENGTHS:
            try:
                model = train_autoencoder(
                    feat_array, seq_len, device=DEVICE,
                    outcome_returns=outcome_returns_map.get(seq_len),
                )
                embs = compute_embeddings(model, feat_array, seq_len, device=DEVICE)
                self.models[seq_len] = model
                self.corpus[seq_len] = embs
                self.is_fallback[seq_len] = False
            except Exception as e:
                _log_fallback(f"[{retrain_date}] AE train failed seq_len={seq_len}: {e}")
                self.models[seq_len] = None
                self.corpus[seq_len] = _discrete_embed_all(feat_array, seq_len)
                self.is_fallback[seq_len] = True

            # Build FAISS index for fast corpus search
            if _FAISS_AVAILABLE:
                _faiss_embs = self.corpus[seq_len]
                if len(_faiss_embs) > 0:
                    faiss_idx = FAISSIndex()
                    faiss_idx.build(_faiss_embs, np.arange(len(_faiss_embs), dtype=np.int64))
                    self.faiss_indexes[seq_len] = faiss_idx
                    print(f"[FAISS] seq_len={seq_len}: index built with {len(_faiss_embs)} vectors")

            # Precompute queries for the next cycle (batched)
            qcache = _batch_embed(
                self.models[seq_len], total_feat_array, seq_len,
                self.is_fallback[seq_len], future_bar_indices
            )
            # Also store queries for bars in corpus (the last window)
            # so bar_idx = retrain_size is handled
            corpus = self.corpus[seq_len]
            # corpus[i] = window starting at i; query for bar_idx = i + seq_len
            # so corpus[-1] = query for bar_idx = retrain_size
            if len(corpus) > 0:
                qcache[retrain_size] = corpus[-1]
            self.query_cache[seq_len] = qcache

    def get_query(self, seq_len: int, bar_idx: int,
                  total_feat_array: np.ndarray) -> np.ndarray:
        """Returns query embedding or None."""
        qcache = self.query_cache.get(seq_len, {})
        if bar_idx in qcache:
            return qcache[bar_idx]
        # Fallback: compute on-the-fly
        if bar_idx < seq_len or bar_idx > len(total_feat_array):
            return None
        window = total_feat_array[bar_idx - seq_len: bar_idx]
        model = self.models.get(seq_len)
        is_fb = self.is_fallback.get(seq_len, True)
        emb = _discrete_window_emb(window) if (is_fb or model is None) else None
        if emb is None:
            try:
                t = torch.from_numpy(window[np.newaxis].astype(np.float32)).to(DEVICE)
                model.eval()
                with torch.no_grad():
                    emb = model.encode(t).cpu().numpy()[0]
                if np.isnan(emb).any():
                    emb = _discrete_window_emb(window)
            except Exception:
                emb = _discrete_window_emb(window)
        return emb


# ─── Main walk-forward loop ─────────────────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    feat_df: pd.DataFrame,
    progress_callback: Callable[[dict], None] | None = None,
) -> pd.DataFrame:
    shared_idx = df.index.intersection(feat_df.index)
    df = df.loc[shared_idx].copy()
    feat_df = feat_df.loc[shared_idx].copy()

    close_arr = df["Close"].values.astype(np.float64)
    open_arr = df["Open"].values.astype(np.float64)
    feat_arr = feat_df.values.astype(np.float32)
    low_arr = df["Low"].values.astype(np.float64)
    high_arr = df["High"].values.astype(np.float64)
    atr_arr = (feat_arr[:, 4] * close_arr).astype(np.float64)  # rel_range * close ≈ abs ATR
    hmm_features = _build_hmm_feature_cache(close_arr)
    dates = df.index
    n_bars = len(dates)

    trades = []
    skipped_no_pattern = 0
    skipped_low_conf = 0
    skipped_low_edge = 0
    gate_counters = {
        "no_pattern_found": 0,
        "below_min_occ": 0,
        "split_too_small": 0,
        "directional_fail": 0,
        "dtw_fail": 0,
        "regime_fail": 0,
        "edge_fail": 0,
        "meta_blocked": 0,
        "traded": 0,
    }
    calibrated_similarity_threshold = calibrate_thresholds(df, feat_df)

    cache = EmbeddingCache()
    meta = MetaRanker()
    hmm_model = None
    last_retrain_calendar = None
    regime_win_rates: dict = {}
    past_trade_X = []
    past_trade_y = []
    rolling_pnl = []

    effective_warmup = min(WARMUP_BARS, max(20, n_bars // 8))
    total_iterations = max(0, n_bars - FORWARD_WINDOW - 1 - effective_warmup)

    _emit_progress(
        progress_callback,
        stage="starting",
        completed=0,
        total=total_iterations,
        progress=0.0,
        bars=n_bars,
        effective_warmup=effective_warmup,
        message="Avvio backtest...",
    )

    pbar = tqdm(range(effective_warmup, n_bars - FORWARD_WINDOW - 1),
                desc="Walk-forward", unit="day", dynamic_ncols=True)

    iter_times = []
    eta_printed = False

    for bar_idx in pbar:
        t0 = time.time()
        today = dates[bar_idx]
        history_bars = bar_idx
        bar_gate_reason = None
        iterations_completed = max((bar_idx - effective_warmup) + 1, 1)
        effective_min_occurrences = _scaled_int(MIN_OCCURRENCES, 2, history_bars)
        base_similarity_threshold = _scaled_float(
            calibrated_similarity_threshold, 0.68, history_bars
        )
        effective_similarity_threshold = _adaptive_similarity_threshold(
            base_similarity_threshold,
            trades_taken=len(trades),
            iterations_completed=iterations_completed,
        )

        # ── Retrain check ────────────────────────────────────────────────
        need_retrain = last_retrain_calendar is None
        if not need_retrain:
            need_retrain = (today - last_retrain_calendar).days >= RETRAIN_EVERY_DAYS

        if need_retrain:
            _emit_progress(
                progress_callback,
                stage="retraining",
                completed=bar_idx - effective_warmup,
                total=total_iterations,
                progress=((bar_idx - effective_warmup) / total_iterations) if total_iterations else 1.0,
                bars=n_bars,
                effective_warmup=effective_warmup,
                date=today,
                message=f"Riaddestramento modelli su storico fino al {today.date()}...",
            )
            # Precompute queries for next RETRAIN_EVERY_DAYS bars (calendar -> ~45 trading days)
            next_n = min(RETRAIN_EVERY_DAYS + 20, n_bars - bar_idx)
            cache.retrain(
                feat_arr[:bar_idx], today,
                next_n_bars=next_n,
                total_feat_array=feat_arr,
                close_arr=close_arr,
            )
            last_retrain_calendar = today
            hmm_model = fit_hmm(close_arr[:bar_idx])
            if len(past_trade_X) >= META_RANKER_MIN_TRADES:
                meta.fit(
                    np.array(past_trade_X, dtype=np.float32),
                    np.array(past_trade_y, dtype=np.int32),
                )
            total_trades_ever = len(trades)
            if not meta.is_trained:
                meta_phase = "bypass"
                meta_threshold = 0.0
            elif total_trades_ever < 100:
                meta_phase = "warmup"
                meta_threshold = 0.50 + (META_RANKER_CONFIDENCE_THRESHOLD - 0.50) * (total_trades_ever / 100.0)
            else:
                meta_phase = "full"
                meta_threshold = META_RANKER_CONFIDENCE_THRESHOLD
            append_session_log(
                "META_GATE",
                f"phase={meta_phase}, threshold={meta_threshold:.4f}, trades={total_trades_ever}",
                "backtest.py:292",
            )

        # ── Regime & rolling win-rate ────────────────────────────────────
        regime = _get_current_regime_cached(hmm_model, hmm_features, bar_idx)
        rwr = float(np.mean([p > 0 for p in rolling_pnl[-20:]])) if rolling_pnl else 0.5
        current_atr = float(feat_arr[bar_idx, 4])
        dow = today.dayofweek

        # ── Pattern search ───────────────────────────────────────────────
        best_candidate = None
        best_meta_score = -1.0

        for seq_len in PATTERN_LENGTHS:
            corpus = cache.corpus.get(seq_len)
            if corpus is None or len(corpus) < effective_min_occurrences + 1:
                if bar_gate_reason is None:
                    bar_gate_reason = "below_min_occ"
                continue

            query_emb = cache.get_query(seq_len, bar_idx, feat_arr)
            if query_emb is None:
                continue

            # FIX 1: Match/Validation corpus split to remove selection bias
            split_idx = int(bar_idx * MATCH_VALIDATION_SPLIT)
            use_val_split = (bar_idx - split_idx) >= MATCH_VALIDATION_MIN_BARS
            if use_val_split:
                match_corpus = corpus[:split_idx]
                val_close = close_arr[split_idx:]
                val_offset = split_idx
                max_search_idx = min(split_idx - seq_len, len(match_corpus))
            else:
                match_corpus = corpus
                val_close = None
                val_offset = 0
                max_search_idx = min(bar_idx - seq_len, len(corpus))
                if bar_gate_reason is None:
                    bar_gate_reason = "split_too_small"

            if max_search_idx < effective_min_occurrences:
                if bar_gate_reason is None:
                    bar_gate_reason = "below_min_occ"
                continue

            _faiss_idx = cache.faiss_indexes.get(seq_len)
            if _faiss_idx is not None and _faiss_idx._index is not None:
                occurrences = _faiss_idx.search(
                    query_emb, k=min(200, max_search_idx),
                    threshold=effective_similarity_threshold,
                    max_idx=max_search_idx, seq_len=seq_len,
                )
            else:
                occurrences = find_occurrences(
                    query_emb, match_corpus, seq_len, max_search_idx,
                    threshold=effective_similarity_threshold,
                )
            if len(occurrences) == 0 and bar_gate_reason is None:
                bar_gate_reason = "no_pattern_found"
            if len(occurrences) < effective_min_occurrences:
                if bar_gate_reason is None or bar_gate_reason == "no_pattern_found":
                    bar_gate_reason = "below_min_occ"
                continue

            if len(occurrences) > MAX_DTW_PATHS:
                sims = cosine_similarity_matrix(query_emb, match_corpus[occurrences])
                top_k = np.argsort(-sims)[:MAX_DTW_PATHS]
                occurrences = [occurrences[i] for i in top_k]

            paths, stopped_flags, barrier_info = collect_forward_paths(
                occurrences, close_arr, seq_len, FORWARD_WINDOW,
                low_prices=low_arr, high_prices=high_arr, atr_arr=atr_arr,
                val_close=val_close, val_offset=val_offset,
            )
            if len(paths) < effective_min_occurrences:
                if bar_gate_reason is None:
                    bar_gate_reason = "split_too_small"
                continue

            scored, reject_reason = score_pattern(
                paths,
                regime,
                regime_win_rates,
                history_bars=history_bars,
                stopped_flags=stopped_flags,
                barrier_info=barrier_info,
                return_reason=True,
            )
            if scored is None:
                if bar_gate_reason is None and reject_reason is not None:
                    bar_gate_reason = reject_reason
                continue

            fv = build_feature_vector(
                seq_len,
                scored["directional_consistency"],
                scored["dtw_compactness"],
                scored["regime_weight"],
                scored["edge_quality"],
                len(occurrences),
                current_atr,
                regime,
                dow,
                rwr,
                stop_rate=scored.get("stop_rate", 0.0),
                mean_mae=scored.get("mean_mae", 0.0),
            )
            conf = meta.predict_proba(fv) if meta.is_trained else min(scored["final_score"], 1.0)

            if conf > best_meta_score:
                best_meta_score = conf
                best_candidate = {
                    "seq_len": seq_len,
                    "direction": scored["direction"],
                    "directional_consistency": scored["directional_consistency"],
                    "dtw_compactness": scored["dtw_compactness"],
                    "regime_weight": scored["regime_weight"],
                    "edge_quality": scored["edge_quality"],
                    "mean_outcome": scored["mean_outcome"],
                    "final_score": scored["final_score"],
                    "n_occurrences": len(occurrences),
                    "meta_conf": conf,
                    "fv": fv,
                    "regime": regime,
                    "stop_rate": scored.get("stop_rate", 0.0),
                    "tp_rate": scored.get("tp_rate", 0.0),
                    "mean_mae": scored.get("mean_mae", 0.0),
                }

        # ── Gate & simulate ──────────────────────────────────────────────
        if best_candidate is None:
            skipped_no_pattern += 1
            gate_counters[bar_gate_reason or "no_pattern_found"] += 1
        elif best_candidate["edge_quality"] < 0.15:
            skipped_low_edge += 1
            gate_counters["edge_fail"] += 1
        else:
            total_trades_ever = len(trades)
            use_meta_gate = False
            effective_threshold = 0.0
            if not meta.is_trained:
                effective_confidence = best_candidate["final_score"]
            elif total_trades_ever < 100:
                use_meta_gate = True
                effective_threshold = 0.50 + (
                    META_RANKER_CONFIDENCE_THRESHOLD - 0.50
                ) * (total_trades_ever / 100.0)
                effective_confidence = best_candidate["meta_conf"]
            else:
                use_meta_gate = True
                effective_threshold = META_RANKER_CONFIDENCE_THRESHOLD
                effective_confidence = best_candidate["meta_conf"]

            if use_meta_gate and effective_confidence < effective_threshold:
                skipped_low_conf += 1
                gate_counters["meta_blocked"] += 1
                continue

            entry_idx = bar_idx + 1
            exit_idx = bar_idx + 1 + FORWARD_WINDOW
            if exit_idx < n_bars:
                entry_price = open_arr[entry_idx]
                exit_price = close_arr[exit_idx]
                direction = best_candidate["direction"]
                stopped_out = False
                barrier_type_live = "time"
                barrier_day_live = FORWARD_WINDOW
                # Apply triple barrier on live trade (SL + TP + time)
                atr_live = float(atr_arr[bar_idx])
                sl_level = entry_price - direction * SL_MULT * atr_live
                tp_level = entry_price + direction * TP_MULT * atr_live
                for _fwd_j in range(entry_idx, min(exit_idx + 1, n_bars)):
                    lo = low_arr[_fwd_j]
                    hi = high_arr[_fwd_j]
                    if direction == 1:
                        if lo <= sl_level:
                            exit_price = float(sl_level)
                            stopped_out = True; barrier_type_live = "stop"
                            barrier_day_live = _fwd_j - entry_idx + 1; break
                        elif hi >= tp_level:
                            exit_price = float(tp_level)
                            barrier_type_live = "tp"
                            barrier_day_live = _fwd_j - entry_idx + 1; break
                    else:
                        if hi >= sl_level:
                            exit_price = float(sl_level)
                            stopped_out = True; barrier_type_live = "stop"
                            barrier_day_live = _fwd_j - entry_idx + 1; break
                        elif lo <= tp_level:
                            exit_price = float(tp_level)
                            barrier_type_live = "tp"
                            barrier_day_live = _fwd_j - entry_idx + 1; break
                raw_ret = (exit_price - entry_price) / (entry_price + 1e-8)
                if ENABLE_SLIPPAGE:
                    raw_ret -= 2 * SLIPPAGE_PCT
                pnl = CAPITAL_PER_TRADE * direction * raw_ret

                rolling_pnl.append(pnl)
                key = (best_candidate["regime"], direction)
                regime_win_rates[key] = (
                    0.95 * regime_win_rates.get(key, 0.5)
                    + 0.05 * (1.0 if pnl > 0 else 0.0)
                )
                past_trade_X.append(best_candidate["fv"].tolist())
                past_trade_y.append(1 if pnl > 0 else 0)

                trades.append({
                    "date": today,
                    "pattern_length": best_candidate["seq_len"],
                    "direction": "long" if direction == 1 else "short",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "n_occurrences": best_candidate["n_occurrences"],
                    "directional_consistency": best_candidate["directional_consistency"],
                    "dtw_compactness": best_candidate["dtw_compactness"],
                    "edge_quality": best_candidate["edge_quality"],
                    "mean_outcome": best_candidate["mean_outcome"],
                    "regime": best_candidate["regime"],
                    "meta_ranker_confidence": best_candidate["meta_conf"],
                    "embedding_id": f"N{best_candidate['seq_len']}_R{best_candidate['regime']}",
                    "stopped_out": stopped_out,
                    "barrier_type": barrier_type_live,
                    "barrier_day": barrier_day_live,
                    "stop_rate": best_candidate.get("stop_rate", 0.0),
                    "tp_rate": best_candidate.get("tp_rate", 0.0),
                    "mean_mae": best_candidate.get("mean_mae", 0.0),
                })
                gate_counters["traded"] += 1

        if bar_idx % 100 == 0 and bar_idx > effective_warmup:
            total_decisions = sum(gate_counters.values())
            if total_decisions > 0:
                print(f"\n[bar {bar_idx}] Gate breakdown:")
                for gate, count in gate_counters.items():
                    pct = count / total_decisions * 100.0
                    print(f"  {gate:25s}: {count:4d} ({pct:.0f}%)")

        elapsed = time.time() - t0
        iter_times.append(elapsed)
        if len(iter_times) == 10 and not eta_printed:
            avg = np.mean(iter_times)
            remaining = (n_bars - FORWARD_WINDOW - 1 - effective_warmup) - 10
            print(f"\n[ETA] ~{avg * remaining / 60:.1f} min remaining ({avg:.3f}s/bar)")
            eta_printed = True

        completed = (bar_idx - effective_warmup) + 1
        if completed == total_iterations or completed % 25 == 0:
            avg_iter = float(np.mean(iter_times[-50:])) if iter_times else 0.0
            remaining_iters = max(total_iterations - completed, 0)
            _emit_progress(
                progress_callback,
                stage="walk_forward",
                completed=completed,
                total=total_iterations,
                progress=(completed / total_iterations) if total_iterations else 1.0,
                bars=n_bars,
                effective_warmup=effective_warmup,
                date=today,
                avg_seconds_per_iteration=avg_iter,
                eta_seconds=remaining_iters * avg_iter,
                message=f"Analisi barra {completed}/{total_iterations} ({today.date()})",
            )

    pbar.close()
    stopped_out_count = sum(1 for t in trades if t.get("stopped_out", False))
    print(f"\n[backtest] Trades: {len(trades)} (stopped_out: {stopped_out_count}) | "
          f"skipped_no_pattern: {skipped_no_pattern} | "
          f"skipped_low_edge: {skipped_low_edge} | "
          f"skipped_low_conf: {skipped_low_conf}")

    trades_df = pd.DataFrame(trades)
    trades_df.attrs["skipped_no_pattern"] = skipped_no_pattern
    trades_df.attrs["skipped_low_edge"] = skipped_low_edge
    trades_df.attrs["skipped_low_conf"] = skipped_low_conf
    trades_df.attrs["total_bars"] = n_bars
    trades_df.attrs["stopped_out_count"] = stopped_out_count
    _emit_progress(
        progress_callback,
        stage="completed",
        completed=total_iterations,
        total=total_iterations,
        progress=1.0,
        bars=n_bars,
        effective_warmup=effective_warmup,
        trades=len(trades),
        skipped_no_pattern=skipped_no_pattern,
        skipped_low_edge=skipped_low_edge,
        skipped_low_conf=skipped_low_conf,
        message="Backtest completato.",
    )
    return trades_df
