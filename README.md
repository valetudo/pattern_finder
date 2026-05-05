# Pattern Finder — Walk-Forward OHLC Pattern Mining Backtest

A sophisticated walk-forward backtesting system that discovers predictive OHLC candlestick patterns using neural embeddings, clustering, and a LightGBM meta-ranker.

## Architecture

| Module | Responsibility |
|---|---|
| `config.py` | All constants and feature toggles |
| `data.py` | Yahoo Finance download, disk caching, feature engineering |
| `autoencoder.py` | LSTM Autoencoder (+ optional supervised contrastive loss) |
| `patterns.py` | Cosine similarity search, DTW clustering, HMM regime detection |
| `meta_ranker.py` | LightGBM meta-model with purged walk-forward CV |
| `backtest.py` | Walk-forward loop, embedding cache, trade simulation |
| `runner.py` | `BacktestRequest` interface callable from CLI or web app |
| `reporting.py` | Equity chart (3-panel), trades CSV, summary stats |
| `webapp.py` | Streamlit web UI |
| `main.py` | CLI entry point |
| `bootstrap.py` | Auto-install missing dependencies |
| `session_log_utils.py` | Structured session event logging |

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

### Environment overrides

| Variable | Default | Description |
|---|---|---|
| `PF_TICKER` | `SPY` | Ticker symbol |
| `PF_MAX_BARS` | *(all)* | Limit history length |
| `PF_WARMUP_BARS` | `252` | Override warmup period |
| `PF_START_DATE` | *(all)* | `YYYY-MM-DD` start date |
| `PF_END_DATE` | *(all)* | `YYYY-MM-DD` end date |
| `PF_SKIP_ARTIFACTS` | `0` | Skip writing chart/CSV |

### Web app

```bash
streamlit run webapp.py
```

## Key features

- **No-lookahead bias** — strictly uses data up to day T−1 for each decision
- **Neural embeddings** — LSTM Autoencoder with optional supervised contrastive loss encodes OHLC sequences into 32-dim space
- **Walk-forward retraining** — encoder, HMM, and meta-ranker all retrain every 60 calendar days
- **Triple-barrier labelling** — SL/TP/time exits used for pattern quality scoring
- **Regime filtering** — 3-state Gaussian HMM weights pattern scores by historical win-rate per regime
- **LightGBM meta-ranker** — predicts P(profitable) with purged walk-forward CV and exponential decay sample weights
- **Adaptive similarity threshold** — relaxes cosine threshold if trade frequency falls below `TARGET_TRADES_PER_MONTH`

## Outputs

| File | Description |
|---|---|
| `equity_line.png` | 3-panel chart: cumulative PnL, rolling win rate, pattern lengths |
| `trades_log.csv` | One row per trade with full metadata |
| `summary.txt` | Aggregate statistics |
| `fallback_log.txt` | Records every graceful fallback event |
| `session_log.txt` | Structured event log for this run |

## Dependencies

```
pandas numpy yfinance matplotlib scipy torch lightgbm hmmlearn dtaidistance scikit-learn tqdm streamlit
```
