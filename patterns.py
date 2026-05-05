# patterns.py — embedding, similarity search, DTW clustering, HMM regime

import warnings
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from config import (
    PATTERN_LENGTHS, COSINE_SIMILARITY_THRESHOLD, MIN_OCCURRENCES,
    FORWARD_WINDOW, MIN_DIRECTIONAL_CONSISTENCY, MIN_DTW_COMPACTNESS,
    MIN_REGIME_WEIGHT, MIN_EDGE_QUALITY, MIN_ABS_MEAN_OUTCOME,
    DBSCAN_PERCENTILE, HMM_STATES, HMM_LOOKBACK, SEED, EMBEDDING_DIM,
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
    # Sort candidates by descending similarity, then apply non-overlap filter
    order = candidates[np.argsort(-sims[candidates])]
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

    def __init__(self):
        self._index = None
        self._dim: int | None = None

    def build(self, embeddings: np.ndarray, ids: np.ndarray) -> None:
        if not _FAISS_AVAILABLE or len(embeddings) == 0:
            return
        import faiss
        d = int(embeddings.shape[1])
        self._dim = d
        vecs = embeddings.copy().astype(np.float32)
        faiss.normalize_L2(vecs)
        hnsw = faiss.IndexHNSWFlat(d, 32)      # M=32
        hnsw.hnsw.efConstruction = 200
        self._index = faiss.IndexIDMap(hnsw)
        self._index.add_with_ids(vecs, ids.astype(np.int64))

    def search(self, query_emb: np.ndarray, k: int, threshold: float,
               max_idx: int, seq_len: int) -> list[int]:
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
        mask = (sims >= threshold) & (idxs >= 0) & (idxs < max_idx)
        sims = sims[mask]
        idxs = idxs[mask]
        if len(idxs) == 0:
            return []
        order = np.argsort(-sims)
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
                          val_close: np.ndarray | None = None,
                          val_offset: int = 0):
    """
    Returns (paths, stopped_flags, barrier_info).
    - paths: (n_valid, fwd) float32 — forward return paths (flat after barrier exit)
    - stopped_flags: (n_valid,) bool — True if stopped out (SL hit)
    - barrier_info: list of dicts {barrier_type, barrier_day, exit_price, mae}

    FIX 1: val split — if val_close is provided, outcomes are measured in
    the validation window (close_prices[val_offset:]) to remove selection bias.
    FIX 2: triple barrier — SL=SL_MULT*ATR, TP=TP_MULT*ATR, time=fwd.
    """
    n_full = len(close_prices)
    use_val = val_close is not None and len(val_close) > 0
    use_barriers = (low_prices is not None and high_prices is not None
                    and atr_arr is not None and BARRIER_USE_INTRADAY)

    # ── First pass: raw paths to determine global direction ──────────────
    raw_final_rets = []
    valid_occurrences = []
    for start in occurrences:
        if use_val:
            val_entry = start + seq_len - val_offset   # index in val_close
            val_exit = val_entry + fwd
            if val_entry < 0 or val_exit >= len(val_close):
                continue
            entry_close = close_prices[start + seq_len - 1]
            raw_ret = (float(val_close[val_exit - 1]) - entry_close) / (entry_close + 1e-8)
        else:
            entry_idx = start + seq_len
            exit_idx = entry_idx + fwd
            if exit_idx >= n_full:
                continue
            entry_close = close_prices[entry_idx - 1]
            raw_ret = (float(close_prices[exit_idx - 1]) - entry_close) / (entry_close + 1e-8)
        raw_final_rets.append(raw_ret)
        valid_occurrences.append(start)

    if not valid_occurrences:
        return (np.empty((0, fwd), dtype=np.float32),
                np.empty(0, dtype=bool), [])

    mean_raw = float(np.mean(raw_final_rets))
    direction = 1 if mean_raw >= 0 else -1

    # ── Second pass: apply triple barrier (PERF FIX 2 — vectorized) ─────
    paths = []
    stopped_flags = []
    barrier_info = []

    starts_arr = np.array(valid_occurrences, dtype=np.int64)

    # Collect entry prices and ATR values per occurrence
    if use_val:
        full_entries = np.array([val_offset + (s + seq_len - val_offset)
                                 for s in valid_occurrences], dtype=np.int64)
        entry_closes = close_prices[starts_arr + seq_len - 1]
    else:
        full_entries = starts_arr + seq_len
        entry_closes = close_prices[np.clip(full_entries - 1, 0, len(close_prices) - 1)]

    atr_values = np.zeros(len(starts_arr), dtype=np.float64)
    if use_barriers:
        atr_idx = np.clip(full_entries - 1, 0, len(atr_arr) - 1)
        atr_values = atr_arr[atr_idx].astype(np.float64)

    # Vectorized triple barrier for all occurrences at once
    if use_barriers and atr_values.max() > 0:
        _full_close = close_prices if not use_val else np.concatenate([close_prices, val_close])
        v_exit_prices, v_barrier_types, v_barrier_days = apply_triple_barrier_vectorized(
            starts_arr, seq_len, direction,
            entry_closes, atr_values,
            low_prices, high_prices, _full_close,
            fwd, SL_MULT, TP_MULT,
        )
    else:
        # No barriers: time exit only
        v_exit_prices = None
        v_barrier_types = np.full(len(starts_arr), "time")
        v_barrier_days = np.full(len(starts_arr), fwd)

    # Build path arrays (vectorized where possible, per-occurrence for MAE)
    for k, start in enumerate(valid_occurrences):
        if use_val:
            val_entry = start + seq_len - val_offset
            entry_close = entry_closes[k]
            fwd_closes = val_close[val_entry: val_entry + fwd]
            full_entry = int(full_entries[k])
        else:
            entry_close = entry_closes[k]
            entry_idx = int(full_entries[k])
            fwd_closes = close_prices[entry_idx: entry_idx + fwd]
            full_entry = entry_idx

        if len(fwd_closes) < fwd:
            continue

        barrier_type = str(v_barrier_types[k])
        barrier_day  = int(v_barrier_days[k])
        exit_price   = float(v_exit_prices[k]) if v_exit_prices is not None \
                       else float(fwd_closes[-1])
        barrier_hit  = barrier_type != "time"

        # MAE: vectorized over forward window
        if use_barriers and atr_values[k] > 0:
            j_end = min(barrier_day, fwd)
            lo_fwd = low_prices[full_entry: full_entry + j_end].astype(np.float64)
            hi_fwd = high_prices[full_entry: full_entry + j_end].astype(np.float64)
            if direction == 1:
                mae = float(((lo_fwd - entry_close) / (entry_close + 1e-8)).min()) \
                      if len(lo_fwd) else 0.0
            else:
                mae = float(((hi_fwd - entry_close) / (entry_close + 1e-8)).max()) \
                      if len(hi_fwd) else 0.0
        else:
            mae = 0.0

        # Build path
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


# ── DTW compactness via DBSCAN ──────────────────────────────────────────────

def dtw_compactness_numpy(paths: np.ndarray) -> float:
    """
    PERF FIX 1 — O(N×T) vectorized centroid-distance compactness.
    Replaces the O(N²×T²) DTW pairwise matrix.

    Normalises each path to [0,1], computes centroid, then measures what
    fraction of paths lie within one std-dev of the mean centroid distance.
    Comparable in signal quality to DTW compactness for short paths (T=5).
    """
    if len(paths) < 2:
        return 0.0
    arr = np.array(paths, dtype=np.float64)          # (N, T)
    rng = arr.max(axis=1) - arr.min(axis=1)
    rng = np.where(rng < 1e-8, 1.0, rng)
    arr_norm = (arr - arr.min(axis=1, keepdims=True)) / rng[:, None]
    centroid = arr_norm.mean(axis=0)
    dists = np.sqrt(((arr_norm - centroid) ** 2).sum(axis=1))
    mean_d = dists.mean()
    std_d  = dists.std() + 1e-8
    return float((dists <= mean_d + std_d).sum()) / len(paths)


def dtw_compactness_original(paths: np.ndarray) -> float:
    """Original O(N²) DTW+DBSCAN implementation — reserved for borderline cases."""
    if len(paths) < 3:
        return 0.0
    try:
        from dtaidistance import dtw_ndim as dtw_mod
        paths_list = [p.astype(np.float64) for p in paths]
        dist_matrix = dtw_mod.distance_matrix_fast(paths_list, ndim=1, compact=False)
    except Exception:
        try:
            from dtaidistance import dtw as dtw_1d
            n = len(paths)
            dist_matrix = np.zeros((n, n), dtype=np.float64)
            for i in range(n):
                for j in range(i + 1, n):
                    d = dtw_1d.distance_fast(paths[i].astype(np.float64),
                                             paths[j].astype(np.float64))
                    dist_matrix[i, j] = dist_matrix[j, i] = d
        except Exception:
            from scipy.spatial.distance import pdist, squareform
            dist_matrix = squareform(pdist(paths, metric="euclidean"))
    upper = dist_matrix[np.triu_indices(len(paths), k=1)]
    if len(upper) == 0 or upper.max() == 0:
        return 1.0
    eps = float(np.percentile(upper, DBSCAN_PERCENTILE))
    if eps <= 0:
        eps = float(upper.mean()) * 0.5
    db = DBSCAN(eps=eps, min_samples=2, metric="precomputed")
    labels = db.fit_predict(dist_matrix)
    if labels.max() < 0:
        return 0.0
    counts = np.bincount(labels[labels >= 0])
    return float(counts.max()) / len(paths)


def dtw_compactness(paths: np.ndarray) -> float:
    """
    PERF FIX 1 — Three-layer strategy:
      1. Fast numpy centroid score (O(N×T))
      2. Call original DTW only in borderline range [threshold-0.05, threshold+0.10]
      3. Otherwise use numpy result directly
    """
    numpy_score = dtw_compactness_numpy(paths)
    lo = MIN_DTW_COMPACTNESS - 0.05
    hi = MIN_DTW_COMPACTNESS + 0.10
    if lo <= numpy_score <= hi:
        try:
            return dtw_compactness_original(paths)
        except Exception:
            return numpy_score
    return numpy_score


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

    mean_abs_outcome = float(np.mean(np.abs(final_day)))
    edge_quality = abs_mean_outcome / (mean_abs_outcome + 1e-8)
    edge_quality = min(max(edge_quality, 0.0), 1.0)
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
