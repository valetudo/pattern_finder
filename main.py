# main.py — entry point

import os
from bootstrap import ensure_runtime_dependencies


ensure_runtime_dependencies()

# ── Now safe to import project modules ────────────────────────────────────

import time
import pandas as pd
from config import TICKER
from reporting import print_and_save_summary, save_trades_csv, save_equity_chart
from runner import BacktestRequest, run_backtest_request


def _get_env_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        value = int(raw)
        return value if value > 0 else None
    except ValueError:
        print(f"[main] Ignoring invalid {name}={raw!r}; expected positive integer")
        return None


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _get_env_date(name: str) -> pd.Timestamp | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return pd.Timestamp(raw).normalize()
    except ValueError:
        print(f"[main] Ignoring invalid {name}={raw!r}; expected YYYY-MM-DD")
        return None


def main():
    max_bars = _get_env_int("PF_MAX_BARS")
    warmup_override = _get_env_int("PF_WARMUP_BARS")
    skip_artifacts = _env_flag("PF_SKIP_ARTIFACTS")
    start_date = _get_env_date("PF_START_DATE")
    end_date = _get_env_date("PF_END_DATE")
    ticker = os.getenv("PF_TICKER", TICKER).strip().upper() or TICKER

    print("=" * 60)
    print(f"  Walk-Forward OHLC Pattern Mining Backtest")
    print(f"  Ticker: {ticker}")
    print("=" * 60)

    t_start = time.time()
    if warmup_override is not None:
        print(f"[main] PF_WARMUP_BARS active: using warmup={warmup_override}")

    result = run_backtest_request(
        BacktestRequest(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            max_bars=max_bars,
            warmup_bars=warmup_override,
        )
    )
    print(
        f"[main] Features ready: {result['bars']} bars from {result['actual_start'].date()} "
        f"to {result['actual_end'].date()}"
    )

    trades_df = result["trades_df"]
    if not trades_df.empty:
        print_and_save_summary(result["stats"])
        if skip_artifacts:
            print("[main] PF_SKIP_ARTIFACTS active: skipping CSV/chart writes")
        else:
            save_trades_csv(trades_df)
            save_equity_chart(trades_df)
    else:
        print("[main] No trades generated — check warmup/threshold settings")

    elapsed = (time.time() - t_start) / 60
    print(f"\n[main] Total runtime: {elapsed:.1f} min")


if __name__ == "__main__":
    main()
