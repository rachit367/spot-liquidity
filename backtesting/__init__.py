"""
Walk-forward backtesting engine for the ICT strategy.

Methodology
───────────
• Generates N independent test cases, each consisting of:
    – 100-candle ICT setup window  (strategy decision point)
    – 50-candle forward section    (exit simulation)
• The forward section drifts toward TP (win=True) or SL (win=False),
  controlled by the ``win_rate`` parameter.
• Position sizing mirrors the live RiskManager (risk_pct % of equity).
• Worst-case exit is assumed when SL and TP are both touched in the same candle.

Key metrics
───────────
  win_rate        – % of trades that hit take-profit
  profit_factor   – gross profit ÷ gross loss
  max_drawdown    – largest peak-to-trough equity decline (%)
  sharpe_per_trade– mean PnL / std PnL (per-trade, not annualised)
  expectancy_r    – average PnL expressed in R multiples
  signal_rate     – % of scanned windows that produced a valid ICT signal
  max_consec_losses – longest consecutive losing streak

Usage:
    python backtest.py
    python backtest.py --trials 1000 --win-rate 0.55 --rr 2.0 --risk 1.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd

from brokers.mock import _build_ict_candles
from execution.risk_manager import RiskManager
from strategy import ICTStrategy, _prepare, _price_in_zone


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    trial:         int
    direction:     str
    entry:         float
    stop_loss:     float
    take_profit:   float
    exit_price:    float
    outcome:       Literal["tp", "sl", "timeout"]
    pnl_abs:       float   # absolute P&L in currency units
    pnl_r:         float   # P&L in R multiples  (1R = 1× initial risk)
    quantity:      int
    balance_after: float


# ─────────────────────────────────────────────────────────────────────────────
# Candle generators
# ─────────────────────────────────────────────────────────────────────────────

def _make_forward_candles(
    last_close: float,
    volatility: float,
    count: int,
    seed: int,
    going_up: bool,
) -> list[tuple]:
    """
    Generate ``count`` OHLC candles with a directional drift.

    ``going_up=True``  → price trends toward TP (long trade wins).
    ``going_up=False`` → price trends toward SL (long trade loses).
    Noise is added on top of the drift so exit timing varies naturally.
    """
    rng = np.random.RandomState(seed)
    direction = 1.0 if going_up else -1.0
    now = datetime.now()
    price = last_close
    candles: list[tuple] = []

    for i in range(count):
        drift = price * volatility * direction * 0.40
        noise = price * volatility * rng.randn()  * 0.60
        o = price
        c = price + drift + noise
        h = max(o, c) + abs(price * volatility * rng.randn() * 0.15)
        l = min(o, c) - abs(price * volatility * rng.randn() * 0.15)
        ts = (now + timedelta(minutes=30 * (100 + i))).isoformat()
        candles.append((
            ts,
            round(o, 2),
            round(max(h, o, c), 2),
            round(min(l, o, c), 2),
            round(c, 2),
            10_000,
        ))
        price = c

    return candles


def _make_test_series(
    base_price: float,
    volatility: float,
    seed_setup: int,
    seed_fwd:   int,
    forward_len: int,
    win: bool,
) -> pd.DataFrame:
    """
    Return a DataFrame of 100 ICT-setup candles + ``forward_len`` exit candles.
    The forward section drifts upward when ``win=True`` (bullish TP hit).
    """
    np.random.seed(seed_setup)
    setup   = _build_ict_candles(base_price, "bullish", volatility, 100)
    last_c  = setup[-1][4]
    forward = _make_forward_candles(last_c, volatility, forward_len,
                                    seed=seed_fwd, going_up=win)
    rows = setup + forward
    return pd.DataFrame(rows,
                        columns=["timestamp", "open", "high", "low", "close", "volume"])


# ─────────────────────────────────────────────────────────────────────────────
# Exit simulation
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_exit(
    signal,
    forward_df: pd.DataFrame,
    max_candles: int = 50,
) -> tuple[float, str]:
    """
    Walk ``forward_df`` bar-by-bar and return ``(exit_price, outcome)`` where
    outcome ∈ {"tp", "sl", "timeout"}.

    SL takes priority when both levels are touched on the same candle (worst case).
    """
    for _, row in forward_df.head(max_candles).iterrows():
        if signal.direction == "long":
            if row["low"] <= signal.stop_loss:
                return signal.stop_loss, "sl"
            if row["high"] >= signal.take_profit:
                return signal.take_profit, "tp"
        else:
            if row["high"] >= signal.stop_loss:
                return signal.stop_loss, "sl"
            if row["low"] <= signal.take_profit:
                return signal.take_profit, "tp"

    # Time-out: exit at last available close
    last_idx = min(max_candles - 1, len(forward_df) - 1)
    return float(forward_df.iloc[last_idx]["close"]), "timeout"


# ─────────────────────────────────────────────────────────────────────────────
# Strategy runner (stateless, operates on a DataFrame window)
# ─────────────────────────────────────────────────────────────────────────────

_STRAT = ICTStrategy("SIM", ob_lookback=20, kill_zones=[], swing_length=5, rr_ratio=2.0)


def _run_strategy_on_window(df: pd.DataFrame, rr_ratio: float = 2.0):
    """
    Run all ICT checks on a single 100-candle window.
    Returns a TradeSignal or None.
    """
    df = _prepare(df.copy())
    if len(df) < 20:
        return None

    structure = _STRAT._detect_market_structure(df)
    if structure == "neutral":
        return None

    sweep = _STRAT._detect_liquidity_sweep(df, structure)
    if not sweep:
        return None

    ob = _STRAT._find_order_block(df, structure)
    if not ob:
        return None

    ltp = float(df["close"].iloc[-1])
    if not _price_in_zone(ltp, ob["low"], ob["high"]):
        return None

    fvg = _STRAT._find_fvg(df, structure)
    if not fvg:
        return None

    signal = _STRAT._build_signal(ltp, ob, structure, sweep, "backtest")
    if signal is None:
        return None

    risk   = abs(signal.entry - signal.stop_loss)
    reward = abs(signal.take_profit - signal.entry)
    if risk <= 0 or (reward / risk) < rr_ratio:
        return None

    return signal


# ─────────────────────────────────────────────────────────────────────────────
# Main backtest engine
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(
    n_trials:        int   = 500,
    base_price:      float = 22_500.0,
    win_rate:        float = 0.58,
    rr_ratio:        float = 2.0,
    risk_pct:        float = 1.0,
    volatility:      float = 0.002,
    initial_balance: float = 100_000.0,
    seed:            int   = 42,
) -> dict:
    """
    Run the ICT strategy over ``n_trials`` independent synthetic test cases
    and return a metrics dictionary (see ``compute_metrics``).

    Parameters
    ----------
    n_trials        : Number of independent 150-candle test series.
    base_price      : Reference price (jittered ±0.5 % per trial).
    win_rate        : Fraction of forward sections that drift toward TP.
                      0.58 → 58 % of trades expected to win.
    rr_ratio        : Minimum reward-to-risk ratio enforced by the strategy.
    risk_pct        : Equity percentage risked per trade.
    volatility      : Candle noise as fraction of price (same as MockBroker).
    initial_balance : Starting equity.
    seed            : Master seed for full reproducibility.
    """
    rng      = np.random.RandomState(seed)
    risk_mgr = RiskManager()

    trades:        list[TradeRecord] = []
    balance_curve: list[float]       = [initial_balance]
    balance        = initial_balance
    windows_scanned    = 0
    signals_generated  = 0

    for trial in range(n_trials):
        s_setup = int(rng.randint(0, 10_000_000))
        s_fwd   = int(rng.randint(0, 10_000_000))
        win     = rng.random() < win_rate
        bp      = base_price * (1.0 + rng.randn() * 0.005)   # ±0.5 % price jitter

        df_full    = _make_test_series(bp, volatility, s_setup, s_fwd, 50, win)
        setup_df   = df_full.iloc[:100].copy().reset_index(drop=True)
        forward_df = df_full.iloc[100:].copy().reset_index(drop=True)

        windows_scanned += 1
        signal = _run_strategy_on_window(setup_df, rr_ratio)
        if signal is None:
            continue

        signals_generated += 1
        exit_price, outcome = _simulate_exit(signal, forward_df)

        qty = max(1, risk_mgr.compute_quantity(
            balance, risk_pct, signal.entry, signal.stop_loss,
        ))
        risk_amount = abs(signal.entry - signal.stop_loss)

        if signal.direction == "long":
            pnl_abs = (exit_price - signal.entry) * qty
            pnl_r   = (exit_price - signal.entry) / risk_amount
        else:
            pnl_abs = (signal.entry - exit_price) * qty
            pnl_r   = (signal.entry - exit_price) / risk_amount

        balance += pnl_abs
        balance_curve.append(balance)

        trades.append(TradeRecord(
            trial         = trial,
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

    return compute_metrics(
        trades, balance_curve, windows_scanned, signals_generated, initial_balance,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metrics calculator
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    trades:            list[TradeRecord],
    balance_curve:     list[float],
    windows_scanned:   int,
    signals_generated: int,
    initial_balance:   float,
) -> dict:
    """Compute full performance metrics from a completed backtest run."""
    if not trades:
        return {
            "error":             "no trades generated",
            "windows_scanned":   windows_scanned,
            "signals_generated": signals_generated,
        }

    df       = pd.DataFrame([t.__dict__ for t in trades])
    wins     = df[df["outcome"] == "tp"]
    losses   = df[df["outcome"] == "sl"]
    timeouts = df[df["outcome"] == "timeout"]

    n_trades      = len(df)
    win_rate      = len(wins) / n_trades
    gross_profit  = float(wins["pnl_abs"].sum())   if len(wins)   > 0 else 0.0
    gross_loss    = float(abs(losses["pnl_abs"].sum())) if len(losses) > 0 else 1e-9
    profit_factor = gross_profit / gross_loss

    # ── Drawdown ──────────────────────────────────────────────────────────────
    curve     = np.array(balance_curve)
    peak      = np.maximum.accumulate(curve)
    dd_abs    = peak - curve
    dd_pct    = dd_abs / np.where(peak > 0, peak, 1.0) * 100.0
    max_dd_pct = float(dd_pct.max())
    max_dd_abs = float(dd_abs.max())

    # ── Consecutive streak ────────────────────────────────────────────────────
    max_cw = max_cl = cur = 0
    for o in df["outcome"]:
        if o == "tp":
            cur = max(cur, 0) + 1
            max_cw = max(max_cw, cur)
        elif o == "sl":
            cur = min(cur, 0) - 1
            max_cl = max(max_cl, -cur)
        else:
            cur = 0

    # ── Sharpe (per-trade, not annualised) ────────────────────────────────────
    pnl_arr = df["pnl_abs"].values
    sharpe  = (float(pnl_arr.mean() / pnl_arr.std())
               if pnl_arr.std() > 0 else 0.0)

    # ── Capital ───────────────────────────────────────────────────────────────
    final_balance   = balance_curve[-1]
    net_profit      = final_balance - initial_balance
    net_profit_pct  = net_profit / initial_balance * 100.0
    recovery_factor = net_profit / max_dd_abs if max_dd_abs > 0 else math.inf
    signal_rate_pct = (signals_generated / windows_scanned * 100.0
                       if windows_scanned else 0.0)

    # ── Break-even win rate (theoretical) ─────────────────────────────────────
    avg_win_r  = float(wins["pnl_r"].mean())   if len(wins)   > 0 else 0.0
    avg_loss_r = float(losses["pnl_r"].mean()) if len(losses) > 0 else 0.0
    # BE = |avg_loss| / (avg_win + |avg_loss|)
    be_win_rate = (abs(avg_loss_r) / (avg_win_r + abs(avg_loss_r))
                   if (avg_win_r + abs(avg_loss_r)) > 0 else 0.0)

    return {
        # Signal selectivity
        "windows_scanned":    windows_scanned,
        "signals_generated":  signals_generated,
        "signal_rate_pct":    round(signal_rate_pct, 1),
        # Trade counts
        "total_trades":       n_trades,
        "wins":               len(wins),
        "losses":             len(losses),
        "timeouts":           len(timeouts),
        # Core performance
        "win_rate_pct":       round(win_rate * 100, 1),
        "breakeven_wr_pct":   round(be_win_rate * 100, 1),
        "profit_factor":      round(profit_factor, 2),
        "expectancy_r":       round(float(df["pnl_r"].mean()), 3),
        "avg_win_r":          round(avg_win_r, 2),
        "avg_loss_r":         round(avg_loss_r, 2),
        # Risk / drawdown
        "max_drawdown_pct":   round(max_dd_pct, 2),
        "max_drawdown_abs":   round(max_dd_abs, 2),
        "max_consec_wins":    max_cw,
        "max_consec_losses":  max_cl,
        # Capital
        "initial_balance":    round(initial_balance, 2),
        "final_balance":      round(final_balance, 2),
        "net_profit":         round(net_profit, 2),
        "net_profit_pct":     round(net_profit_pct, 2),
        "recovery_factor":    (round(recovery_factor, 2)
                               if math.isfinite(recovery_factor) else "inf"),
        "sharpe_per_trade":   round(sharpe, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report printer
# ─────────────────────────────────────────────────────────────────────────────

def print_report(m: dict) -> None:
    """Pretty-print a metrics dict to stdout."""
    if "error" in m:
        print(f"\n[BACKTEST ERROR] {m['error']}")
        print(f"  Windows scanned : {m.get('windows_scanned', 0)}")
        print(f"  Signals found   : {m.get('signals_generated', 0)}")
        return

    sep = "=" * 62
    print(f"\n{sep}")
    print("   ICT STRATEGY  —  BACKTEST REPORT")
    print(sep)

    print("\n  SIGNAL SELECTIVITY")
    print(f"    Windows scanned    : {m['windows_scanned']}")
    print(f"    Signals generated  : {m['signals_generated']}  "
          f"({m['signal_rate_pct']} % of windows)")

    print("\n  TRADE SUMMARY")
    print(f"    Total trades       : {m['total_trades']}")
    print(f"    Wins  (TP hit)     : {m['wins']}")
    print(f"    Losses (SL hit)    : {m['losses']}")
    print(f"    Timeouts           : {m['timeouts']}")
    print(f"    Win rate           : {m['win_rate_pct']} %")
    print(f"    Break-even WR      : {m['breakeven_wr_pct']} %  "
          f"(min needed to profit)")

    print("\n  PERFORMANCE")
    print(f"    Profit factor      : {m['profit_factor']}")
    print(f"    Expectancy         : {m['expectancy_r']} R  per trade")
    print(f"    Avg win            : +{m['avg_win_r']} R")
    print(f"    Avg loss           :  {m['avg_loss_r']} R")
    print(f"    Sharpe (per-trade) : {m['sharpe_per_trade']}")

    print("\n  RISK / DRAWDOWN")
    print(f"    Max drawdown       : {m['max_drawdown_pct']} %  "
          f"(${m['max_drawdown_abs']:,.2f})")
    print(f"    Max consec. wins   : {m['max_consec_wins']}")
    print(f"    Max consec. losses : {m['max_consec_losses']}")
    print(f"    Recovery factor    : {m['recovery_factor']}")

    print("\n  CAPITAL")
    print(f"    Initial balance    : ${m['initial_balance']:>12,.2f}")
    print(f"    Final balance      : ${m['final_balance']:>12,.2f}")
    print(f"    Net profit         : ${m['net_profit']:>12,.2f}  "
          f"({m['net_profit_pct']} %)")
    print(f"\n{sep}\n")


__all__ = ["run_backtest", "compute_metrics", "print_report", "TradeRecord"]
