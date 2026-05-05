# meta_ranker.py — LightGBM meta-model training and prediction

import warnings
import numpy as np

from config import META_RANKER_MIN_TRADES, META_RANKER_CONFIDENCE_THRESHOLD, SEED, FORWARD_WINDOW

warnings.filterwarnings("ignore")


# FIX 3: decorrelated feature vector (remove final_score, use raw components)
# FIX 5: regime encoded as two binary features (regime_0, regime_1) for 3-state HMM
FEATURE_NAMES = [
    "pattern_length",          # [0]  2–6
    "directional_consistency", # [1]  0–1
    "dtw_compactness",         # [2]  0–1
    "regime_weight",           # [3]  0–1
    "edge_quality",            # [4]  0–1
    "n_occurrences_norm",      # [5]  n_occ / 50
    "current_atr",             # [6]  rel_range value
    "regime_0",                # [7]  1 if regime==0
    "regime_1",                # [8]  1 if regime==1  (regime==2 → both 0)
    "day_of_week_norm",        # [9]  dow / 4
    "rolling_win_rate",        # [10] 0–1
    "stop_rate",               # [11] 0–1  (from triple barrier)
    "mean_mae_norm",           # [12] clipped MAE / 0.10  (0–1 after abs)
]


def build_feature_vector(pattern_length: int,
                         directional_consistency: float,
                         dtw_compactness: float,
                         regime_weight: float,
                         edge_quality: float,
                         n_occurrences: int,
                         current_atr: float,
                         regime: int,
                         day_of_week: int,
                         rolling_win_rate: float,
                         stop_rate: float = 0.0,
                         mean_mae: float = 0.0) -> np.ndarray:
    regime_0 = 1.0 if regime == 0 else 0.0
    regime_1 = 1.0 if regime == 1 else 0.0
    # normalise MAE: clip to [-0.10, 0], then divide by 0.10 → [0, 1]
    mae_norm = float(np.clip(-mean_mae, 0.0, 0.10) / 0.10)
    return np.array([
        float(pattern_length),
        float(directional_consistency),
        float(dtw_compactness),
        float(regime_weight),
        float(edge_quality),
        float(n_occurrences) / 50.0,
        float(current_atr),
        regime_0,
        regime_1,
        float(day_of_week) / 4.0,
        float(rolling_win_rate),
        float(stop_rate),
        mae_norm,
    ], dtype=np.float32)


class MetaRanker:
    def __init__(self):
        self.model = None
        self._trained = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> bool:
        """
        X: (n_samples, n_features), y: binary (1=profitable, 0=not)
        FIX 7: exponential decay weights + purged walk-forward split.
        """
        if len(X) < META_RANKER_MIN_TRADES:
            self._trained = False
            return False
        try:
            import lightgbm as lgb
            n_samples = len(X)

            # Exponential decay sample weights
            if n_samples >= 30:
                half_life = 60.0
                weights = np.exp(-np.log(2) / half_life * np.arange(n_samples - 1, -1, -1))
                weights = (weights / weights.sum() * n_samples).astype(np.float32)
            else:
                weights = np.ones(n_samples, dtype=np.float32)

            # Graduated model complexity with L1+L2 regularization
            if n_samples < 50:
                num_leaves = 6
                n_estimators = 200
                min_child = 5
            elif n_samples < 100:
                num_leaves = 12
                n_estimators = 200
                min_child = 8
            else:
                num_leaves = 20
                n_estimators = 200
                min_child = 10

            # Purged walk-forward split:
            # Val = last 20% of trades chronologically.
            # Purge training samples whose forward window overlaps with val period.
            val_size = max(1, int(n_samples * 0.20))
            purge_size = FORWARD_WINDOW  # purge this many before val to avoid leakage
            train_end = n_samples - val_size - purge_size
            if train_end < 15:
                # Not enough after purging — train on all, no val
                X_train, y_train, w_train = X, y, weights
                X_val, y_val = None, None
            else:
                X_train = X[:train_end]
                y_train = y[:train_end]
                w_train = weights[:train_end]
                X_val = X[-val_size:]
                y_val = y[-val_size:]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf = lgb.LGBMClassifier(
                    n_estimators=n_estimators,
                    learning_rate=0.05,
                    max_depth=4,
                    num_leaves=num_leaves,
                    min_child_samples=min_child,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    random_state=SEED,
                    verbose=-1,
                )
                callbacks = [
                    lgb.early_stopping(20, verbose=False),
                    lgb.log_evaluation(period=-1),
                ]
                if X_val is not None and len(np.unique(y_train)) >= 2:
                    try:
                        clf.fit(X_train, y_train, sample_weight=w_train,
                                eval_set=[(X_val, y_val)], callbacks=callbacks)
                    except Exception:
                        clf.fit(X, y, sample_weight=weights)
                else:
                    clf.fit(X, y, sample_weight=weights)
            self.model = clf
            self._trained = True
            print(f"[MetaRanker] Trained: n={n_samples}, feat_dim={X.shape[1]}, "
                  f"leaves={num_leaves}")
            return True
        except Exception:
            self._trained = False
            return False

    def predict_proba(self, x: np.ndarray) -> float:
        if not self._trained or self.model is None:
            return 0.5
        try:
            prob = self.model.predict_proba(x.reshape(1, -1))[0, 1]
            return float(prob)
        except Exception:
            return 0.5

    @property
    def is_trained(self) -> bool:
        return self._trained

