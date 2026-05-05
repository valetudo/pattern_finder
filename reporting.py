# reporting.py — charts, CSV, summary stats

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from config import CAPITAL_PER_TRADE, EQUITY_CHART, TRADES_CSV, SUMMARY_TXT, FORWARD_WINDOW


# ─── Statistics ─────────────────────────────────────────────────────────────

def compute_stats(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {}

    pnl = trades_df["pnl"].values
    cum = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cum)
    drawdown = cum - running_max

    total_trades = len(pnl)
    win_rate = float((pnl > 0).mean())
    avg_pnl = float(pnl.mean())
    median_pnl = float(np.median(pnl))
    max_dd = float(drawdown.min())
    total_pnl = float(cum[-1])

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    payoff_ratio = float(avg_win / abs(avg_loss)) if avg_loss < 0 else 0.0
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0
    expectancy = avg_pnl
    avg_return_per_trade_pct = float((avg_pnl / CAPITAL_PER_TRADE) * 100.0)
    best_trade = float(pnl.max())
    worst_trade = float(pnl.min())
    pnl_std = float(pnl.std())

    consecutive_wins = 0
    consecutive_losses = 0
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    for trade_pnl in pnl:
        if trade_pnl > 0:
            consecutive_wins += 1
            consecutive_losses = 0
        elif trade_pnl < 0:
            consecutive_losses += 1
            consecutive_wins = 0
        else:
            consecutive_wins = 0
            consecutive_losses = 0
        max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
        max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)

    # Sharpe (annualised, assuming ~252 / FORWARD_WINDOW independent trade-periods/year)
    if pnl_std > 0:
        periods_per_year = 252 / FORWARD_WINDOW
        sharpe = float(avg_pnl / pnl_std * np.sqrt(periods_per_year))
    else:
        sharpe = 0.0

    # Calmar
    calmar = float(-avg_pnl * total_trades / max_dd) if max_dd < 0 else 0.0
    recovery_factor = float(total_pnl / abs(max_dd)) if max_dd < 0 else 0.0

    pattern_lengths = trades_df["pattern_length"].values
    most_used = int(pd.Series(pattern_lengths).mode().iloc[0])

    if "date" in trades_df.columns and len(trades_df) > 1:
        dates = pd.to_datetime(trades_df["date"])
        span_days = max((dates.max() - dates.min()).days, 1)
        trades_per_year = float(total_trades * 365.25 / span_days)
        monthly_pnl = trades_df.assign(date=dates).groupby(pd.Grouper(key="date", freq="M"))["pnl"].sum()
        profitable_months_rate = float((monthly_pnl > 0).mean()) if len(monthly_pnl) else 0.0
    else:
        trades_per_year = float(total_trades)
        profitable_months_rate = 0.0

    skipped_no = trades_df.attrs.get("skipped_no_pattern", 0)
    skipped_lc = trades_df.attrs.get("skipped_low_conf", 0)

    # pct_days_with_trade
    total_bars = trades_df.attrs.get("total_bars", 0)
    pct_days_with_trade = float(total_trades / total_bars * 100) if total_bars > 0 else 0.0

    # avg meta-ranker confidence
    if "meta_ranker_confidence" in trades_df.columns:
        avg_meta_conf = float(trades_df["meta_ranker_confidence"].mean())
    else:
        avg_meta_conf = 0.0

    # regime-split win rates
    regime_0_win_rate = 0.0
    regime_1_win_rate = 0.0
    if "regime" in trades_df.columns:
        for r, col_name in [(0, "regime_0_win_rate"), (1, "regime_1_win_rate")]:
            mask = trades_df["regime"] == r
            if mask.sum() > 0:
                wr = float((pnl[mask.values] > 0).mean())
            else:
                wr = 0.0
            if r == 0:
                regime_0_win_rate = wr
            else:
                regime_1_win_rate = wr

    # best pattern length by win rate (min 5 trades)
    best_pattern_length = int(most_used)
    if "pattern_length" in trades_df.columns:
        best_wr = -1.0
        for pl in trades_df["pattern_length"].unique():
            mask = trades_df["pattern_length"] == pl
            if mask.sum() >= 5:
                wr = float((pnl[mask.values] > 0).mean())
                if wr > best_wr:
                    best_wr = wr
                    best_pattern_length = int(pl)

    return {
        "total_trades": total_trades,
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "median_pnl": median_pnl,
        "avg_return_per_trade_pct": avg_return_per_trade_pct,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "payoff_ratio": payoff_ratio,
        "profit_factor": profit_factor,
        "expectancy_per_trade": expectancy,
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe,
        "calmar_ratio": calmar,
        "recovery_factor": recovery_factor,
        "pnl_std": pnl_std,
        "trades_per_year": trades_per_year,
        "profitable_months_rate": profitable_months_rate,
        "max_consecutive_wins": max_consecutive_wins,
        "max_consecutive_losses": max_consecutive_losses,
        "avg_holding_days": FORWARD_WINDOW,
        "most_used_pattern_length": most_used,
        "best_pattern_length": best_pattern_length,
        "pct_days_with_trade": pct_days_with_trade,
        "avg_meta_ranker_confidence": avg_meta_conf,
        "regime_0_win_rate": regime_0_win_rate,
        "regime_1_win_rate": regime_1_win_rate,
        "trades_skipped_no_pattern": skipped_no,
        "trades_skipped_low_confidence": skipped_lc,
    }


# ─── Summary text ────────────────────────────────────────────────────────────

def print_and_save_summary(stats: dict, path: str = SUMMARY_TXT):
    lines = [
        "=" * 50,
        "  BACKTEST SUMMARY",
        "=" * 50,
    ]
    for k, v in stats.items():
        if isinstance(v, float):
            lines.append(f"  {k:<38} {v:.4f}")
        else:
            lines.append(f"  {k:<38} {v}")
    lines.append("=" * 50)
    text = "\n".join(lines)
    print(text)
    with open(path, "w") as f:
        f.write(text + "\n")


# ─── CSV ─────────────────────────────────────────────────────────────────────

def save_trades_csv(trades_df: pd.DataFrame, path: str = TRADES_CSV):
    trades_df.to_csv(path, index=False, float_format="%.6f")
    print(f"[reporting] Saved {len(trades_df)} trades -> {path}")


# ─── Equity chart ─────────────────────────────────────────────────────────────

def build_equity_curve(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()

    pnl = trades_df["pnl"].astype(float)
    cumulative_pnl = pnl.cumsum()
    running_max = cumulative_pnl.cummax()
    drawdown = cumulative_pnl - running_max
    rolling_win_rate = (pnl > 0).astype(float).rolling(20, min_periods=1).mean()

    dates = pd.to_datetime(trades_df["date"]) if "date" in trades_df.columns else pd.RangeIndex(len(trades_df))
    return pd.DataFrame(
        {
            "date": dates,
            "pnl": pnl.values,
            "cumulative_pnl": cumulative_pnl.values,
            "drawdown": drawdown.values,
            "rolling_win_rate": rolling_win_rate.values,
            "pattern_length": trades_df["pattern_length"].values,
            "direction": trades_df["direction"].values,
        }
    )


def create_equity_figure(trades_df: pd.DataFrame):
    if trades_df.empty:
        return None

    curve_df = build_equity_curve(trades_df)
    cum = curve_df["cumulative_pnl"].values
    drawdown = curve_df["drawdown"].values
    running_max = np.maximum.accumulate(cum)
    n = len(curve_df)

    dates = curve_df["date"].values
    lengths = curve_df["pattern_length"].values
    directions = curve_df["direction"].values
    roll_wr = curve_df["rolling_win_rate"].values

    fig = plt.figure(figsize=(14, 10), facecolor="#0f0f0f")
    gs = GridSpec(3, 1, figure=fig, hspace=0.45,
                  top=0.93, bottom=0.07, left=0.07, right=0.97,
                  height_ratios=[3, 1.2, 1.2])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor("#161616")
        ax.tick_params(colors="#cccccc", labelsize=8)
        ax.spines[:].set_color("#333333")
        ax.yaxis.label.set_color("#cccccc")
        ax.xaxis.label.set_color("#cccccc")
        ax.title.set_color("#eeeeee")

    ax1.plot(dates, cum, color="#00d4aa", linewidth=1.0, label="Cumulative PnL")
    ax1.fill_between(dates, cum, running_max,
                     where=(drawdown < 0), alpha=0.35,
                     color="#e04444", label="Drawdown")
    ax1.axhline(0, color="#aaaaaa", linewidth=1.0, linestyle="--", label="Breakeven")
    # Annotate final PnL
    final_pnl = float(cum[-1])
    ax1.annotate(
        f"Final: {final_pnl:+.0f} €",
        xy=(dates[-1], final_pnl),
        xytext=(-70, 12),
        textcoords="offset points",
        color="#00d4aa",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#00d4aa", lw=1.0),
    )
    ax1.set_title("Cumulative PnL (EUR, fixed 1000 EUR notional)", fontsize=10)
    ax1.set_ylabel("PnL (EUR)")
    ax1.legend(fontsize=8, facecolor="#222222", labelcolor="#cccccc")

    ax2.plot(dates, roll_wr, color="#ffa040", linewidth=1.0)
    ax2.axhline(0.5, color="#555555", linewidth=0.5, linestyle="--")
    ax2.set_ylim(0, 1)
    ax2.set_title("Rolling Win Rate (20-trade window)", fontsize=10)
    ax2.set_ylabel("Win Rate")

    colors = ["#3399ff" if d == "long" else "#ff4466" for d in directions]
    ax3.bar(np.arange(n), lengths, color=colors, width=1.0, alpha=0.8)
    ax3.set_yticks(sorted(set(lengths)))
    ax3.set_title("Pattern Length N per Trade (blue=long, red=short)", fontsize=10)
    ax3.set_ylabel("N bars")
    ax3.set_xlabel("Trade #")

    return fig


def _normalized_close_path(close_values: np.ndarray) -> np.ndarray:
    base = float(close_values[0]) if len(close_values) else 1.0
    return (close_values / (base + 1e-8)) * 100.0


def create_pattern_similarity_figure(query_date, query_close: np.ndarray,
                                     matches_df: pd.DataFrame):
    fig = plt.figure(figsize=(13, 8), facecolor="#0f0f0f")
    gs = GridSpec(2, 1, figure=fig, hspace=0.35,
                  top=0.92, bottom=0.08, left=0.08, right=0.97,
                  height_ratios=[2.5, 1.2])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    for ax in [ax1, ax2]:
        ax.set_facecolor("#161616")
        ax.tick_params(colors="#cccccc", labelsize=8)
        ax.spines[:].set_color("#333333")
        ax.yaxis.label.set_color("#cccccc")
        ax.xaxis.label.set_color("#cccccc")
        ax.title.set_color("#eeeeee")

    x = np.arange(len(query_close))
    query_path = _normalized_close_path(query_close)
    ax1.plot(x, query_path, color="#ffd166", linewidth=2.2, label=f"Query {pd.Timestamp(query_date).date()}")

    if not matches_df.empty:
        for _, row in matches_df.iterrows():
            match_path = _normalized_close_path(np.asarray(row["close_path"], dtype=np.float64))
            ax1.plot(x, match_path, linewidth=1.0, alpha=0.45, color="#4cc9f0")

        stacked = np.stack(matches_df["close_path"].apply(lambda arr: _normalized_close_path(np.asarray(arr, dtype=np.float64))))
        ax1.plot(x, stacked.mean(axis=0), color="#06d6a0", linewidth=2.0, linestyle="--", label="Media match")

    ax1.set_title("Pattern Query vs Match Storici", fontsize=11)
    ax1.set_ylabel("Prezzo normalizzato = 100")
    ax1.set_xlabel("Barra nel pattern")
    ax1.legend(fontsize=8, facecolor="#222222", labelcolor="#cccccc")

    if not matches_df.empty:
        labels = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in matches_df["match_end_date"]]
        sims = matches_df["similarity"].values
        bars = ax2.bar(np.arange(len(labels)), sims, color="#4cc9f0", alpha=0.85)
        ax2.set_xticks(np.arange(len(labels)))
        ax2.set_xticklabels(labels, rotation=35, ha="right", color="#cccccc")
        ax2.set_ylim(0, min(1.05, max(0.1, sims.max() + 0.05)))
        ax2.set_title("Score di Similarità", fontsize=10)
        ax2.set_ylabel("Cosine similarity")
        for bar, sim in zip(bars, sims):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{sim:.3f}", ha="center", va="bottom", color="#cccccc", fontsize=8)
    else:
        ax2.text(0.5, 0.5, "Nessun match trovato", ha="center", va="center",
                 color="#cccccc", fontsize=11, transform=ax2.transAxes)
        ax2.set_xticks([])
        ax2.set_yticks([])

    return fig

def save_equity_chart(trades_df: pd.DataFrame, path: str = EQUITY_CHART):
    if trades_df.empty:
        print("[reporting] No trades — skipping chart")
        return

    fig = create_equity_figure(trades_df)
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[reporting] Chart saved -> {path}")
