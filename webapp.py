from __future__ import annotations

from datetime import date, timedelta

import math

from bootstrap import ensure_runtime_dependencies


ensure_runtime_dependencies()

import streamlit as st

from config import TICKER, WARMUP_BARS
from reporting import build_equity_curve
from runner import (
    BacktestRequest,
    analyze_pattern_similarity,
    estimate_backtest_runtime,
    get_history_summary,
    run_backtest_request,
)


if __name__ == "__main__" and not st.runtime.exists():
    print("Questo file va avviato con Streamlit, non con 'python webapp.py'.")
    print("Usa: python -m streamlit run webapp.py --server.headless true --server.port 8501")
    raise SystemExit(1)


st.set_page_config(
    page_title="Pattern Finder Backtest",
    page_icon="PF",
    layout="wide",
)


def _default_start_date() -> date:
    return date.today() - timedelta(days=365 * 5)


@st.cache_data(show_spinner=False)
def _get_history_summary_cached(ticker: str) -> dict:
    return get_history_summary(ticker)


@st.cache_data(show_spinner=False)
def _estimate_backtest_runtime_cached(
    ticker: str,
    start_date: str | None,
    lookback_days: int | None,
    warmup_bars: int,
) -> dict:
    return estimate_backtest_runtime(
        BacktestRequest(
            ticker=ticker,
            start_date=start_date,
            lookback_days=lookback_days,
            warmup_bars=warmup_bars,
        )
    )


def _format_eta(seconds: float) -> str:
    if seconds <= 0:
        return "meno di 1 minuto"
    total_minutes = int(round(seconds / 60))
    if total_minutes < 1:
        return "meno di 1 minuto"
    hours, minutes = divmod(total_minutes, 60)
    if hours == 0:
        return f"circa {minutes} min"
    if minutes == 0:
        return f"circa {hours} h"
    return f"circa {hours} h {minutes} min"


def _format_metric_value(metric_name: str, value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "inf"
        pct_metrics = {
            "win_rate",
            "avg_return_per_trade_pct",
            "profitable_months_rate",
        }
        if metric_name in pct_metrics:
            return f"{value:.2%}" if metric_name != "avg_return_per_trade_pct" else f"{value:.2f}%"
        if metric_name in {"trades_per_year", "sharpe_ratio", "calmar_ratio", "recovery_factor", "profit_factor", "payoff_ratio", "pnl_std"}:
            return f"{value:.2f}"
        return f"{value:.2f}"
    return str(value)


def _build_metric_rows(stats: dict, items: list[tuple[str, str]]) -> list[dict]:
    return [
        {"metrica": label, "valore": _format_metric_value(key, stats.get(key))}
        for key, label in items
    ]


SIGNAL_FREQUENCY_PRESETS = {
    "Conservativo": 0.5,
    "Bilanciato": 1.0,
    "Aggressivo": 2.0,
}


st.title("Pattern Finder Backtest")
st.caption("Scegli ticker, giorno iniziale e lookback in giorni. L'output mostra equity PnL e drawdown.")

if "backtest_result" not in st.session_state:
    st.session_state["backtest_result"] = None

with st.sidebar:
    st.header("Parametri")
    ticker = st.text_input("Ticker", value=TICKER, max_chars=15).strip().upper()
    history_summary = None
    if ticker:
        try:
            history_summary = _get_history_summary_cached(ticker)
        except Exception as exc:
            st.warning(f"Storico non disponibile per {ticker}: {exc}")
        else:
            st.caption(
                f"Storico disponibile: {history_summary['start_date'].date()} -> {history_summary['end_date'].date()}"
            )
            st.caption(
                f"Barre allineate: {history_summary['bars']} | Max lookback: {history_summary['max_lookback_days']} giorni"
            )

    use_full_history = st.checkbox("Usa tutto lo storico disponibile", value=False)
    default_start = _default_start_date()
    if history_summary is not None:
        default_start = max(history_summary["start_date"].date(), default_start)

    start_date = st.date_input("Data iniziale", value=default_start, disabled=use_full_history)
    lookback_default = 365 * 5
    lookback_max = 10000
    if history_summary is not None:
        lookback_default = min(lookback_default, history_summary["max_lookback_days"])
        lookback_max = max(90, history_summary["max_lookback_days"])
    lookback_days = st.number_input(
        "Lookback (giorni di calendario)",
        min_value=90,
        max_value=lookback_max,
        value=lookback_default,
        step=30,
        disabled=use_full_history,
    )
    warmup_bars = st.number_input("Warmup bars", min_value=60, max_value=2000, value=WARMUP_BARS, step=10)
    signal_frequency_label = st.select_slider(
        "Frequenza segnali target",
        options=list(SIGNAL_FREQUENCY_PRESETS.keys()),
        value="Bilanciato",
    )
    signal_frequency_target = SIGNAL_FREQUENCY_PRESETS[signal_frequency_label]
    st.caption(f"Target grezzo: circa {signal_frequency_target:.1f} trade al mese")

    runtime_estimate = None
    if ticker:
        try:
            runtime_estimate = _estimate_backtest_runtime_cached(
                ticker,
                None if use_full_history else start_date.isoformat(),
                None if use_full_history else int(lookback_days),
                int(warmup_bars),
            )
        except Exception as exc:
            st.caption(f"Stima runtime non disponibile: {exc}")
        else:
            st.caption(
                "Stima runtime: "
                f"{_format_eta(runtime_estimate['eta_seconds_mid'])} "
                f"(range {_format_eta(runtime_estimate['eta_seconds_low'])} - {_format_eta(runtime_estimate['eta_seconds_high'])})"
            )
            st.caption(
                f"Loop previsto: {runtime_estimate['iterations']} barre dopo warmup {runtime_estimate['effective_warmup']}"
            )

    run_clicked = st.button("Lancia backtest", type="primary", width="stretch")

if run_clicked:
    progress_box = st.empty()
    progress_bar = st.progress(0.0, text="Preparazione backtest...")
    estimate_box = st.empty()
    if runtime_estimate is not None:
        estimate_box.info(
            "Tempo atteso: "
            f"{_format_eta(runtime_estimate['eta_seconds_mid'])} "
            f"su {runtime_estimate['iterations']} iterazioni"
        )

    def _streamlit_progress(payload: dict):
        progress = float(payload.get("progress", 0.0))
        message = payload.get("message", "Backtest in esecuzione...")
        eta_seconds = payload.get("eta_seconds")
        completed = payload.get("completed")
        total = payload.get("total")
        progress_bar.progress(min(max(progress, 0.0), 1.0), text=message)

        details = []
        if completed is not None and total is not None:
            details.append(f"Progresso: {completed}/{total}")
        if eta_seconds is not None and math.isfinite(float(eta_seconds)):
            details.append(f"ETA residua: {_format_eta(float(eta_seconds))}")
        if payload.get("stage") == "retraining":
            details.append("Fase: riaddestramento modelli")
        elif payload.get("stage") == "walk_forward":
            details.append("Fase: scansione walk-forward")
        elif payload.get("stage") == "completed":
            details.append("Fase: completato")

        progress_box.caption(" | ".join(details) if details else message)

    with st.spinner("Esecuzione backtest in corso..."):
        try:
            result = run_backtest_request(
                BacktestRequest(
                    ticker=ticker,
                    start_date=None if use_full_history else start_date.isoformat(),
                    lookback_days=None if use_full_history else int(lookback_days),
                    warmup_bars=int(warmup_bars),
                    target_trades_per_month=signal_frequency_target,
                ),
                progress_callback=_streamlit_progress,
            )
        except Exception as exc:
            st.error(f"Backtest non eseguibile: {exc}")
        else:
            st.session_state["backtest_result"] = result
            progress_bar.progress(1.0, text="Backtest completato.")
            progress_box.caption("Elaborazione terminata, risultati pronti.")

result = st.session_state.get("backtest_result")

if result is not None:
    trades_df = result["trades_df"]
    stats = result["stats"]
    curve_df = build_equity_curve(trades_df)

    st.subheader(f"{result['ticker']} | {result['actual_start'].date()} -> {result['actual_end'].date()}")

    top_cols = st.columns(4)
    top_cols[0].metric("Barre allineate", f"{result['bars']}")
    top_cols[1].metric("Trade", f"{len(trades_df)}")
    top_cols[2].metric("PnL totale", _format_metric_value("total_pnl", stats.get("total_pnl")) if stats else "n/a")
    top_cols[3].metric("Sharpe", _format_metric_value("sharpe_ratio", stats.get("sharpe_ratio")) if stats else "n/a")

    if trades_df.empty:
        st.warning("Nessun trade generato per questa finestra. Prova ad aumentare il lookback o ridurre il warmup.")
    else:
        report_cols = st.columns(4)
        report_cols[0].metric("Win rate", _format_metric_value("win_rate", stats.get("win_rate")))
        report_cols[1].metric("Profitto medio/trade", _format_metric_value("avg_pnl", stats.get("avg_pnl")))
        report_cols[2].metric("Profit factor", _format_metric_value("profit_factor", stats.get("profit_factor")))
        report_cols[3].metric("Max drawdown", _format_metric_value("max_drawdown", stats.get("max_drawdown")))

        st.subheader("Report di impiegabilita'")
        rep_col1, rep_col2, rep_col3 = st.columns(3)

        profitability_items = [
            ("total_pnl", "PnL totale"),
            ("avg_pnl", "Profitto medio per trade"),
            ("median_pnl", "Mediana PnL per trade"),
            ("avg_return_per_trade_pct", "Rendimento medio per trade sul nozionale"),
            ("avg_win", "Vincita media"),
            ("avg_loss", "Perdita media"),
            ("best_trade", "Miglior trade"),
            ("worst_trade", "Peggior trade"),
        ]
        risk_items = [
            ("sharpe_ratio", "Sharpe ratio"),
            ("calmar_ratio", "Calmar ratio"),
            ("recovery_factor", "Recovery factor"),
            ("max_drawdown", "Max drawdown"),
            ("pnl_std", "Volatilita' del PnL per trade"),
            ("max_consecutive_losses", "Massima sequenza di loss"),
            ("max_consecutive_wins", "Massima sequenza di win"),
        ]
        deployment_items = [
            ("win_rate", "Win rate"),
            ("profit_factor", "Profit factor"),
            ("payoff_ratio", "Payoff ratio"),
            ("expectancy_per_trade", "Expectancy per trade"),
            ("trades_per_year", "Trade stimati per anno"),
            ("profitable_months_rate", "Mesi profittevoli"),
            ("most_used_pattern_length", "Pattern length piu' usata"),
            ("trades_skipped_no_pattern", "Barre scartate per assenza pattern"),
            ("trades_skipped_low_confidence", "Barre scartate per bassa confidenza"),
        ]

        with rep_col1:
            st.caption("Redditivita'")
            st.dataframe(_build_metric_rows(stats, profitability_items), width="stretch", hide_index=True)
        with rep_col2:
            st.caption("Rischio e stabilita'")
            st.dataframe(_build_metric_rows(stats, risk_items), width="stretch", hide_index=True)
        with rep_col3:
            st.caption("Utilizzabilita' operativa")
            st.dataframe(_build_metric_rows(stats, deployment_items), width="stretch", hide_index=True)

        chart_col, table_col = st.columns([2, 1])
        with chart_col:
            st.pyplot(result["figure"], clear_figure=False, width="stretch")

        with table_col:
            st.subheader("Metriche")
            st.dataframe(
                {
                    "metrica": list(stats.keys()),
                    "valore": [round(v, 4) if isinstance(v, float) else v for v in stats.values()],
                },
                width="stretch",
                hide_index=True,
            )

        st.subheader("Equity e drawdown")
        st.dataframe(
            curve_df[["date", "pnl", "cumulative_pnl", "drawdown"]],
            width="stretch",
            hide_index=True,
        )

        st.subheader("Ultimi trade")
        st.dataframe(trades_df.tail(25), width="stretch", hide_index=True)

    st.subheader("Verifica visiva della similarità")
    similarity_col1, similarity_col2, similarity_col3 = st.columns(3)
    available_dates = [ts.date() for ts in result["df"].index]
    default_date = trades_df["date"].iloc[-1].date() if not trades_df.empty else available_dates[-1]
    selected_date = similarity_col1.selectbox("Data da ispezionare", options=available_dates, index=available_dates.index(default_date))
    seq_len = similarity_col2.selectbox("Lunghezza pattern", options=[2, 3, 4, 5, 6], index=1)
    top_k = similarity_col3.slider("Numero match storici", min_value=3, max_value=10, value=5)

    try:
        similarity_result = analyze_pattern_similarity(
            result["df"],
            result["features"],
            selected_date.isoformat(),
            seq_len=seq_len,
            top_k=top_k,
        )
    except Exception as exc:
        st.info(f"Analisi similarità non disponibile: {exc}")
    else:
        st.caption(
            f"Pattern query: {similarity_result['query_start_date'].date()} -> {similarity_result['query_end_date'].date()}"
            + (" | embedding fallback" if similarity_result["used_fallback_embedding"] else " | embedding autoencoder")
        )
        st.pyplot(similarity_result["figure"], clear_figure=False, width="stretch")
        if not similarity_result["matches_df"].empty:
            st.dataframe(similarity_result["matches_df"], width="stretch", hide_index=True)
else:
    st.info("Imposta i parametri sulla sinistra e avvia il backtest.")