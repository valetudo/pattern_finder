# patterns.py — embedding, similarity search, DTW clustering, HMM regime

import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler

from config import (
    PATTERN_LENGTHS, COSINE_SIMILARITY_THRESHOLD, MIN_OCCURRENCES,
    FORWARD_WINDOW, MIN_DIRECTIONAL_CONSISTENCY, MIN_DTW_COMPACTNESS,
    MIN_REGIME_WEIGHT, MIN_EDGE_QUALITY, MIN_ABS_MEAN_OUTCOME,
    EMBARGO_BARS, USE_EMBARGO, HMM_STATES, HMM_LOOKBACK, SEED, EMBEDDING_DIM,
    SL_MULT, TP_MULT, BARRIER_USE_INTRADAY, FALLBACK_LOG,
)

warnings.filterwarnings("ignore")
np.random.seed(SEED)

SHORT_HISTORY_BARS = 100
FULL_HISTORY_BARS = 250


def _history_fraction(history_bars: int) -> float:
    if history_bars <= SHORT_HISTORY_BARS:
        return 0.0
    if history_bars >= FULL_HISTORY_BARS:
        return 1.0
    return (history_bars - SHORT_HISTORY_BARS) / (FULL_HISTORY_BARS - SHORT_HISTORY_BARS)


def _scaled_threshold(config_value: float, short_history_floor: float,
                      history_bars: int) -> float:
    frac = _history_fraction(history_bars)
    return short_history_floor + (config_value - short_history_floor) * frac


# ── cosine similarity (vectorised) ─────────────────────────────────────────

def cosine_similarity_matrix(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """
    query : (D,) or (1, D)
    corpus: (N, D)
    returns: (N,) similarities
    """
    q = query / (np.linalg.norm(query) + 1e-8)
    norms = np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-8
    c_norm = corpus / norms
    return c_norm @ q


# ── non-overlapping occurrence finder ──────────────────────────────────────

def find_occurrences(query_emb: np.ndarray, all_embs: np.ndarray,
                     seq_len: int, max_idx_exclusive: int,
                     threshold: float = COSINE_SIMILARITY_THRESHOLD):
    """
    Returns sorted list of start-indices of past non-overlapping windows
    with cosine similarity >= threshold.
    PERF FIX 3: greedy selection uses sorted numpy ops instead of Python O(N) scan.
    """
    corpus = all_embs[:max_idx_exclusive]
    if len(corpus) == 0:
        return []
    sims = cosine_similarity_matrix(query_emb, corpus)
    candidates = np.where(sims >= threshold)[0]
    if len(candidates) == 0:
        return []
    # Sort by position (ascending) for temporal diversity in non-overlap greedy
    order = np.sort(candidates)
    selected = []
    last_end = -1
    for idx in order:
        if int(idx) > last_end:
            selected.append(int(idx))
            last_end = int(idx) + seq_len - 1
    selected.sort()
    return selected


# ── FAISS HNSW index for fast approximate cosine search ────────────────────

_FAISS_AVAILABLE = False
try:
    import faiss as _faiss_lib  # noqa: F401
    _FAISS_AVAILABLE = True
except ImportError:
    pass


class FAISSIndex:
    """HNSW-based approximate nearest-neighbour index using FAISS.
    Falls back silently to find_occurrences() if faiss-cpu is not installed.
    """

    def __init__(self, dim: int | None = None):
        self._index = None
        self._dim: int | None = dim

    def build(self, embeddings: np.ndarray, ids: np.ndarray) -> None:
        if not _FAISS_AVAILABLE or len(embeddings) == 0:
            return
        import faiss
        d = int(embeddings.shape[1])
        self._dim = d
        vecs = embeddings.copy().astype(np.float32)
        faiss.normalize_L2(vecs)
        # BUG_FIX: use METRIC_INNER_PRODUCT so D[0] contains cosine similarities
        # (not L2 distances). With L2, `sims >= threshold` was inverted — similar
        # patterns (small L2) failed the filter while dissimilar ones passed.
        hnsw = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        hnsw.hnsw.efConstruction = 200
        hnsw.hnsw.efSearch = 64
        self._index = faiss.IndexIDMap(hnsw)
        self._index.add_with_ids(vecs, ids.astype(np.int64))

    def search(self, query_emb: np.ndarray, k: int, threshold: float,
               max_idx: int | None = None, seq_len: int = 1) -> list[int]:
        """Return non-overlapping bar indices sorted by position."""
        if not _FAISS_AVAILABLE or self._index is None:
            return []
        import faiss
        q = query_emb[np.newaxis].copy().astype(np.float32)
        faiss.normalize_L2(q)
        actual_k = min(k, self._index.ntotal)
        if actual_k == 0:
            return []
        D, I = self._index.search(q, actual_k)
        sims = D[0]
        idxs = I[0]
        if max_idx is None:
            max_idx = np.iinfo(np.int64).max
        mask = (sims >= threshold) & (idxs >= 0) & (idxs < max_idx)
        sims = sims[mask]
        idxs = idxs[mask]
        if len(idxs) == 0:
            return []
        # Sort by ascending position for temporal diversity: prevents consecutive
        # high-similarity bars from blocking all others in the greedy non-overlap step
        order = np.argsort(idxs)
        selected: list[int] = []
        last_end = -1
        for i in order:
            idx = int(idxs[i])
            if idx > last_end:
                selected.append(idx)
                last_end = idx + seq_len - 1
        selected.sort()
        return selected

    def update(self, new_embeddings: np.ndarray, new_ids: np.ndarray) -> None:
        """Incrementally add new vectors to the existing index."""
        if not _FAISS_AVAILABLE or self._index is None or len(new_embeddings) == 0:
            return
        import faiss
        vecs = new_embeddings.copy().astype(np.float32)
        faiss.normalize_L2(vecs)
        self._index.add_with_ids(vecs, new_ids.astype(np.int64))


# ── forward return paths ────────────────────────────────────────────────────

def apply_triple_barrier_vectorized(
    starts: np.ndarray,
    seq_len: int,
    direction: int,
    entry_prices: np.ndarray,
    atr_values: np.ndarray,
    low_arr: np.ndarray,
    high_arr: np.ndarray,
    close_arr: np.ndarray,
    forward_window: int,
    sl_mult: float,
    tp_mult: float,
) -> tuple:
    """
    PERF FIX 2 — Vectorized triple barrier for N occurrences simultaneously.
    O(N×T) via numpy index matrices instead of O(N×T) Python loops.
    Returns: exit_prices (N,), barrier_types (N, str), barrier_days (N,)
    """
    N = len(starts)
    fwd = forward_window
    n_arr = len(low_arr)

    # Build index matrix (N, fwd): entry bar + j for each occurrence
    idx_matrix = (starts[:, None] + seq_len + np.arange(fwd)[None, :])
    idx_matrix = np.clip(idx_matrix, 0, n_arr - 1)

    low_mat  = low_arr[idx_matrix]      # (N, fwd)
    high_mat = high_arr[idx_matrix]     # (N, fwd)
    close_mat = close_arr[idx_matrix]   # (N, fwd)

    sl_levels = (entry_prices - direction * sl_mult * atr_values)[:, None]
    tp_levels = (entry_prices + direction * tp_mult * atr_values)[:, None]

    if direction == 1:
        stop_hits = low_mat  <= sl_levels   # (N, fwd)
        tp_hits   = high_mat >= tp_levels
    else:
        stop_hits = high_mat >= sl_levels
        tp_hits   = low_mat  <= tp_levels

    has_stop = stop_hits.any(axis=1)
    has_tp   = tp_hits.any(axis=1)
    stop_days = np.where(has_stop, stop_hits.argmax(axis=1), fwd)
    tp_days   = np.where(has_tp,   tp_hits.argmax(axis=1),   fwd)

    is_stop = has_stop & (stop_days <= tp_days)
    is_tp   = has_tp   & (tp_days   <  stop_days)

    exit_prices = np.where(
        is_stop, sl_levels.squeeze(),
        np.where(is_tp, tp_levels.squeeze(), close_mat[:, -1])
    )
    barrier_days = np.minimum(stop_days, tp_days)

    # String barrier types
    barrier_types = np.where(is_stop, "stop", np.where(is_tp, "tp", "time"))
    return exit_prices, barrier_types, barrier_days + 1


def collect_forward_paths(occurrences: list, close_prices: np.ndarray,
                          seq_len: int, fwd: int = FORWARD_WINDOW,
                          low_prices: np.ndarray | None = None,
                          high_prices: np.ndarray | None = None,
                          atr_arr: np.ndarray | None = None,
                          bar_idx: int = 0):
    """
    Returns (paths, stopped_flags, barrier_info).
    - paths: (n_valid, fwd) float32 — forward return paths (flat after barrier exit)
    - stopped_flags: (n_valid,) bool — True if stopped out (SL hit)
    - barrier_info: list of dicts {barrier_type, barrier_day, exit_price, mae}

    BUG1_FIX: Removed broken match/val split. Measures outcomes directly from
    close_prices using an embargo to prevent near-future contamination.
    """
    n_full = len(close_prices)
    use_barriers = (low_prices is not None and high_prices is not None
                    and atr_arr is not None and BARRIER_USE_INTRADAY)

    # Embargo: pattern_end must be at least EMBARGO_BARS before the current query start
    query_start = bar_idx - seq_len

    # ── First pass: filter occurrences + determine direction ──────────────
    raw_final_rets = []
    valid_occurrences = []
    for start in occurrences:
        pattern_end = start + seq_len - 1
        outcome_end = start + seq_len + fwd  # exclusive

        # Embargo: exclude occurrences too close to the current query
        if USE_EMBARGO and pattern_end >= query_start - EMBARGO_BARS:
            continue
        # No-lookahead: outcome must be fully in the past
        if outcome_end > bar_idx:
            continue
        if outcome_end > n_full:
            continue

        entry_close = close_prices[start + seq_len - 1]
        raw_ret = (float(close_prices[start + seq_len + fwd - 1]) - entry_close) / (entry_close + 1e-8)
        raw_final_rets.append(raw_ret)
        valid_occurrences.append(start)

    if not valid_occurrences:
        return (np.empty((0, fwd), dtype=np.float32),
                np.empty(0, dtype=bool), [])

    mean_raw = float(np.mean(raw_final_rets))
    direction = 1 if mean_raw >= 0 else -1

    # ── Second pass: apply triple barrier (PERF FIX 2 — vectorized) ─────
    starts_arr = np.array(valid_occurrences, dtype=np.int64)
    full_entries = starts_arr + seq_len
    entry_closes = close_prices[np.clip(full_entries - 1, 0, len(close_prices) - 1)]

    atr_values = np.zeros(len(starts_arr), dtype=np.float64)
    if use_barriers:
        atr_idx = np.clip(full_entries - 1, 0, len(atr_arr) - 1)
        atr_values = atr_arr[atr_idx].astype(np.float64)

    if use_barriers and atr_values.max() > 0:
        v_exit_prices, v_barrier_types, v_barrier_days = apply_triple_barrier_vectorized(
            starts_arr, seq_len, direction,
            entry_closes, atr_values,
            low_prices, high_prices, close_prices,
            fwd, SL_MULT, TP_MULT,
        )
    else:
        v_exit_prices = None
        v_barrier_types = np.full(len(starts_arr), "time")
        v_barrier_days = np.full(len(starts_arr), fwd)

    paths = []
    stopped_flags = []
    barrier_info = []

    for k, start in enumerate(valid_occurrences):
        entry_close = entry_closes[k]
        entry_idx = int(full_entries[k])
        fwd_closes = close_prices[entry_idx: entry_idx + fwd]

        if len(fwd_closes) < fwd:
            continue

        barrier_type = str(v_barrier_types[k])
        barrier_day  = int(v_barrier_days[k])
        exit_price   = float(v_exit_prices[k]) if v_exit_prices is not None \
                       else float(fwd_closes[-1])
        barrier_hit  = barrier_type != "time"

        if use_barriers and atr_values[k] > 0:
            j_end = min(barrier_day, fwd)
            lo_fwd = low_prices[entry_idx: entry_idx + j_end].astype(np.float64)
            hi_fwd = high_prices[entry_idx: entry_idx + j_end].astype(np.float64)
            if direction == 1:
                mae = float(((lo_fwd - entry_close) / (entry_close + 1e-8)).min()) \
                      if len(lo_fwd) else 0.0
            else:
                mae = float(((hi_fwd - entry_close) / (entry_close + 1e-8)).max()) \
                      if len(hi_fwd) else 0.0
        else:
            mae = 0.0

        final_ret = (exit_price - entry_close) / (entry_close + 1e-8)
        if barrier_hit and barrier_day < fwd:
            path = ((fwd_closes - entry_close) / (entry_close + 1e-8)).astype(np.float32)
            path[barrier_day:] = final_ret
        else:
            path = ((fwd_closes - entry_close) / (entry_close + 1e-8)).astype(np.float32)

        paths.append(path)
        stopped_flags.append(barrier_type == "stop")
        barrier_info.append({
            "barrier_type": barrier_type,
            "barrier_day": barrier_day,
            "exit_price": exit_price,
            "mae": mae,
        })

    if not paths:
        return (np.empty((0, fwd), dtype=np.float32),
                np.empty(0, dtype=bool), [])
    return (np.array(paths, dtype=np.float32),
            np.array(stopped_flags, dtype=bool),
            barrier_info)


# ── DTW compactness via directional agreement ───────────────────────────────

def dtw_compactness(paths: list) -> float:
    """
    BUG3_FIX: Measures outcome path coherence via directional agreement.

    Replaces DBSCAN clustering (which set geometrically impossible thresholds
    with only 2-4 points per cluster on financial data). Instead: measures
    what fraction of (path, day) pairs agree with the mean path's direction.

    A score of 0.60 means 60% of directions agree — realistic for good patterns.
    Calibrated threshold: MIN_DTW_COMPACTNESS = 0.58.
    """
    if len(paths) < 3:
        return 0.0
    arr = np.array(paths, dtype=np.float64)  # (N, T)
    mean_path = arr.mean(axis=0)             # (T,)
    mean_direction = np.sign(mean_path)      # +1 or -1 per day
    path_directions = np.sign(arr)           # (N, T)
    agreement = (path_directions == mean_direction[None, :])
    return float(agreement.mean())


# ── HMM regime detection ────────────────────────────────────────────────────

def fit_hmm(close_prices: np.ndarray, lookback: int = HMM_LOOKBACK):
    """
    Fits a 3-state Gaussian HMM on (rolling_vol, trend, momentum) features.
    Returns the fitted model or None on failure.
    """
    try:
        from hmmlearn import hmm
        n = len(close_prices)
        if n < lookback + 60:
            return None

        log_ret = np.log(close_prices[1:] / close_prices[:-1])
        # rolling vol (20-day)
        roll_vol = np.array([
            log_ret[max(0, i - 20): i].std() if i >= 20 else np.nan
            for i in range(1, n)
        ])
        # 20-day trend
        roll_trend = np.array([
            log_ret[max(0, i - 20): i].mean() if i >= 20 else np.nan
            for i in range(1, n)
        ])
        # momentum: 20-day return / 60-day return ratio
        roll_mom = np.array([
            (log_ret[max(0, i - 20): i].mean() /
             (abs(log_ret[max(0, i - 60): i].mean()) + 1e-8))
            if i >= 60 else np.nan
            for i in range(1, n)
        ])
        feat = np.stack([roll_vol, roll_trend, roll_mom], axis=1)
        mask = ~np.isnan(feat).any(axis=1)
        feat = feat[mask]

        if len(feat) < lookback:
            return None

        feat = feat[-lookback:]
        scaler = StandardScaler()
        feat_scaled = scaler.fit_transform(feat)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = hmm.GaussianHMM(
                n_components=HMM_STATES, covariance_type="diag",
                n_iter=100, random_state=SEED
            )
            model.fit(feat_scaled)
        model._scaler = scaler
        model._n_features = 3
        return model
    except Exception:
        return None


def get_current_regime(hmm_model, close_prices: np.ndarray,
                       lookback: int = HMM_LOOKBACK) -> int:
    """Returns regime label (0, 1, or 2) or 0 on failure."""
    if hmm_model is None:
        return 0
    try:
        n = len(close_prices)
        log_ret = np.log(close_prices[1:] / close_prices[:-1])
        roll_vol = np.array([
            log_ret[max(0, i - 20): i].std() if i >= 20 else 0.0
            for i in range(1, n)
        ])
        roll_trend = np.array([
            log_ret[max(0, i - 20): i].mean() if i >= 20 else 0.0
            for i in range(1, n)
        ])
        roll_mom = np.array([
            (log_ret[max(0, i - 20): i].mean() /
             (abs(log_ret[max(0, i - 60): i].mean()) + 1e-8))
            if i >= 60 else 0.0
            for i in range(1, n)
        ])
        feat = np.stack([roll_vol, roll_trend, roll_mom], axis=1)
        feat_scaled = hmm_model._scaler.transform(feat[-HMM_LOOKBACK:])
        states = hmm_model.predict(feat_scaled)
        return int(states[-1])
    except Exception:
        return 0


# ── Pattern scorer ──────────────────────────────────────────────────────────

def score_pattern(paths: np.ndarray, regime: int,
                  regime_win_rates: dict, history_bars: int | None = None,
                  stopped_flags: np.ndarray | None = None,
                  barrier_info: list | None = None,
                  return_reason: bool = False) -> dict | tuple[dict | None, str | None] | None:
    """
    Returns dict with scoring components, barrier metrics, and final_score.
    Returns None if thresholds not met.
    """
    def _result(payload: dict | None, reason: str | None = None):
        if return_reason:
            return payload, reason
        return payload

    if len(paths) == 0:
        return _result(None, "split_too_small")

    history_bars = history_bars or FULL_HISTORY_BARS
    min_abs_mean_outcome = _scaled_threshold(MIN_ABS_MEAN_OUTCOME, 0.0, history_bars)
    min_directional_consistency = _scaled_threshold(MIN_DIRECTIONAL_CONSISTENCY, 0.50, history_bars)
    min_dtw_compactness = _scaled_threshold(MIN_DTW_COMPACTNESS, 0.40, history_bars)
    min_regime_weight = _scaled_threshold(MIN_REGIME_WEIGHT, 0.40, history_bars)
    min_edge_quality = _scaled_threshold(MIN_EDGE_QUALITY, 0.10, history_bars)

    final_day = paths[:, -1]
    mean_outcome = float(final_day.mean())
    abs_mean_outcome = abs(mean_outcome)
    if abs_mean_outcome < min_abs_mean_outcome:
        return _result(None, "edge_fail")

    direction = 1 if mean_outcome >= 0 else -1
    dir_consistency = float((final_day * direction > 0).mean())

    if dir_consistency < min_directional_consistency:
        return _result(None, "directional_fail")

    compactness = dtw_compactness(paths)
    if compactness < min_dtw_compactness:
        return _result(None, "dtw_fail")

    # Avoid cold-start deadlock: when no regime history exists yet,
    # use a neutral prior that does not fail the regime gate by default.
    key = (regime, direction)
    if key in regime_win_rates:
        regime_weight = regime_win_rates[key]
    else:
        regime_weight = max(0.5, min_regime_weight)

    if regime_weight < min_regime_weight:
        return _result(None, "regime_fail")

    # BUG2_FIX: t-statistic edge quality — scale-invariant, robust on fat-tailed returns
    # t_stat / 3.0 maps to [0,1]: 0.33 ≈ borderline (t=1), 0.67 ≈ significant (t=2)
    n_paths = len(final_day)
    if n_paths < 4:
        edge_quality = 0.0
    else:
        outcomes_arr = final_day.astype(np.float64)
        std_out = float(outcomes_arr.std(ddof=1)) + 1e-8
        t_stat = abs_mean_outcome / (std_out / np.sqrt(n_paths))
        edge_quality = float(min(1.0, t_stat / 3.0))
    if edge_quality < min_edge_quality:
        return _result(None, "edge_fail")

    # Barrier metrics from FIX 2
    stop_rate = 0.0
    tp_rate = 0.0
    mean_mae = 0.0
    if barrier_info:
        stop_rate = sum(1 for b in barrier_info if b["barrier_type"] == "stop") / len(barrier_info)
        tp_rate = sum(1 for b in barrier_info if b["barrier_type"] == "tp") / len(barrier_info)
        mean_mae = float(np.mean([b["mae"] for b in barrier_info]))
    elif stopped_flags is not None and len(stopped_flags) > 0:
        # fallback to old stopped_flags
        stop_rate = float(stopped_flags.sum() / len(stopped_flags))

    final_score = dir_consistency * compactness * regime_weight * edge_quality
    if stop_rate > 0.4:
        final_score *= (1.0 - stop_rate)

    return _result({
        "directional_consistency": dir_consistency,
        "dtw_compactness": compactness,
        "regime_weight": regime_weight,
        "edge_quality": edge_quality,
        "final_score": final_score,
        "direction": direction,
        "mean_outcome": mean_outcome,
        "stop_rate": stop_rate,
        "tp_rate": tp_rate,
        "mean_mae": mean_mae,
    })
