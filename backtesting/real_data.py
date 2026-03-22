"""
Real-data walk-forward backtest.

Fetches genuine historical OHLC from Delta Exchange (public API)
or Upstox (requires live credentials), then slides a 100-candle
strategy window forward and simulates entry/exit on actual future bars.

Usage (CLI via backtest.py):
    python backtest.py --broker delta --symbol BTCUSDT --interval 4h
    python backtest.py --broker upstox --symbol "NSE_INDEX|Nifty 50" --interval 1d
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from backtesting import (
    TradeRecord,
    _run_strategy_on_window,
    _simulate_exit,
    compute_metrics,
    print_report,
)
from brokers import get_broker
from brokers.base import BrokerAPIError
from execution.risk_manager import RiskManager
from strategy import ICTStrategy, _prepare

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Core walk-forward engine
# ─────────────────────────────────────────────────────────────────────────────

def run_real_backtest(
    broker_name:     str,
    symbol:          str,
    interval:        str   = "4h",
    window:          int   = 100,   # candles fed to strategy each step
    step:            int   = 10,    # how many candles to advance the window
    forward_bars:    int   = 50,    # candles available for exit simulation
    rr_ratio:        float = 2.0,
    risk_pct:        float = 1.0,
    initial_balance: float = 100_000.0,
    fetch_count:     int   = 500,   # total candles to fetch from broker
) -> dict:
    """
    Fetch historical OHLC, walk forward, run the ICT strategy on each window,
    and simulate exits on the following ``forward_bars`` real candles.

    Returns the same metrics dict as ``compute_metrics`` plus extra broker info.
    """
    # ── Fetch data ────────────────────────────────────────────────────────────
    try:
        broker = get_broker(broker_name)
        df_raw = broker.get_ohlc(symbol, interval, count=fetch_count)
    except BrokerAPIError as exc:
        return {
            "error": f"{broker_name} API error: {exc}",
            "broker": broker_name, "symbol": symbol, "interval": interval,
        }
    except Exception as exc:
        return {
            "error": str(exc),
            "broker": broker_name, "symbol": symbol, "interval": interval,
        }

    if df_raw is None or len(df_raw) < window + forward_bars:
        got = len(df_raw) if df_raw is not None else 0
        return {
            "error": (f"Not enough data: need {window + forward_bars} candles, "
                      f"got {got}. Try a shorter interval or longer date range."),
            "broker": broker_name, "symbol": symbol, "interval": interval,
            "candles_fetched": got,
        }

    df = _prepare(df_raw.copy().reset_index(drop=True))
    logger.info("Real backtest: %d candles for %s/%s", len(df), broker_name, symbol)

    # ── Walk-forward loop ─────────────────────────────────────────────────────
    kill_zones = [] if broker_name == "upstox" else ["london_open", "ny_open"]
    strategy   = ICTStrategy(
        symbol, interval=interval, ob_lookback=20,
        kill_zones=kill_zones, swing_length=5, rr_ratio=rr_ratio,
    )
    risk_mgr      = RiskManager()
    trades:        list[TradeRecord] = []
    balance_curve: list[float]       = [initial_balance]
    balance        = initial_balance
    windows_scanned    = 0
    signals_generated  = 0

    for start in range(0, len(df) - window - forward_bars + 1, step):
        end          = start + window
        fwd_end      = min(end + forward_bars, len(df))
        setup_df     = df.iloc[start:end].copy().reset_index(drop=True)
        forward_df   = df.iloc[end:fwd_end].copy().reset_index(drop=True)

        if len(forward_df) < 2:
            continue

        windows_scanned += 1

        signal = _run_strategy_on_window(setup_df, rr_ratio=rr_ratio, strategy=strategy)
        if signal is None:
            continue

        signals_generated += 1

        # Simulate exit on real future candles
        exit_price, outcome = _simulate_exit(signal, forward_df, max_candles=forward_bars)

        qty        = max(1, risk_mgr.compute_quantity(
            balance, risk_pct, signal.entry, signal.stop_loss,
        ))
        risk_amount = risk

        if signal.direction == "long":
            pnl_abs = (exit_price - signal.entry) * qty
            pnl_r   = (exit_price - signal.entry) / risk_amount
        else:
            pnl_abs = (signal.entry - exit_price) * qty
            pnl_r   = (signal.entry - exit_price) / risk_amount

        balance += pnl_abs
        balance_curve.append(balance)

        trades.append(TradeRecord(
            trial         = windows_scanned,
            direction     = signal.direction,
            entry         = signal.entry,
            stop_loss     = signal.stop_loss,
            take_profit   = signal.take_profit,
            exit_price    = exit_price,
            outcome       = outcome,
            pnl_abs       = round(pnl_abs, 2),
            pnl_r         = round(pnl_r, 3),
            quantity       = qty,
            balance_after = round(balance, 2),
        ))

    # ── Metrics ───────────────────────────────────────────────────────────────
    m = compute_metrics(
        trades, balance_curve, windows_scanned, signals_generated, initial_balance,
    )
    m["broker"]         = broker_name
    m["symbol"]         = symbol
    m["interval"]       = interval
    m["candles_fetched"] = len(df)
    m["run_at"]         = datetime.now(timezone.utc).isoformat()
    return m
