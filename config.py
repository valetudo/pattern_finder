# config.py — all constants and toggles

TICKER = "SPY"
TICKERS = ["SPY"]
MULTI_TICKER_MODE = False
DB_FILE = "market_data.db"
CACHE_REFRESH_DAYS = 1
CACHE_FILE = "data_cache.pkl"
FALLBACK_LOG = "fallback_log.txt"
SESSION_LOG_FILE = "session_log.txt"

# Feature engineering
ROLLING_ATR_WINDOW = 20

# Sequence lengths to try
PATTERN_LENGTHS = [2, 3, 4, 5, 6]

# Embedding
EMBEDDING_DIM = 32
LSTM_HIDDEN = 64
LSTM_LAYERS = 2
AUTOENCODER_EPOCHS = 40
AUTOENCODER_LR = 1e-3
AUTOENCODER_BATCH = 64

# Walk-forward
WARMUP_BARS = 252
RETRAIN_EVERY_DAYS = 60       # calendar days between retrains
MIN_OCCURRENCES = 10
FORWARD_WINDOW = 5            # days to hold trade
COSINE_SIMILARITY_THRESHOLD = 0.75
MIN_SIMILARITY_THRESHOLD_FLOOR = 0.62
TARGET_TRADES_PER_MONTH = 2.0

# Scoring thresholds (each component must exceed this)
MIN_DIRECTIONAL_CONSISTENCY = 0.52
MIN_DTW_COMPACTNESS = 0.50
MIN_REGIME_WEIGHT = 0.50
MIN_EDGE_QUALITY = 0.15
MIN_ABS_MEAN_OUTCOME = 0.0010

# Meta-ranker
META_RANKER_MIN_TRADES = 20
META_RANKER_CONFIDENCE_THRESHOLD = 0.55

# Trade simulation
CAPITAL_PER_TRADE = 1000.0    # EUR, fixed notional
ENABLE_SLIPPAGE = False
SLIPPAGE_PCT = 0.001          # 0.1% each side if enabled

# HMM
HMM_STATES = 3
HMM_LOOKBACK = 252
HMM_FEATURES = ["vol", "trend", "momentum"]

# DTW
DBSCAN_PERCENTILE = 20        # percentile of pairwise DTW distances → eps

# Reproducibility
SEED = 42

# Output files
EQUITY_CHART = "equity_line.png"
TRADES_CSV = "trades_log.csv"
SUMMARY_TXT = "summary.txt"

# Feature toggles
ENABLE_VIX = True
VIX_CACHE_FILE = "vix_cache.pkl"
ENABLE_DOW_EMBEDDING = True
ENABLE_MOMENTUM_FEATURE = True

# Match/Validation corpus split (removes selection bias)
MATCH_VALIDATION_SPLIT = 0.7
MATCH_VALIDATION_MIN_BARS = 60

# Triple barrier
SL_MULT = 2.0
TP_MULT = 3.0
BARRIER_USE_INTRADAY = True

# Supervised Contrastive Loss
CONTRASTIVE_LOSS = True
CONTRASTIVE_TEMPERATURE = 0.07
CONTRASTIVE_MIN_WINDOWS = 30
OUTCOME_SIMILARITY_THRESHOLD = 0.5
