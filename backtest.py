"""
ICT Strategy — Backtest CLI

Supports both synthetic (fast, parameterised) and real-data (live API) modes.

Usage:
    # Synthetic — fast, no API keys needed
    python backtest.py
    python backtest.py --trials 1000 --win-rate 0.55
    python backtest.py --scenarios

    # Real data — Delta Exchange (public historical candles, no auth)
    python backtest.py --broker delta --symbol BTCUSDT --interval 4h
    python backtest.py --broker delta --symbol BTCUSDT --interval 1h --step 5

    # Real data — Upstox (requires UPSTOX_LIVE_ACCESS_TOKEN in .env)
    python backtest.py --broker upstox --symbol "NSE_INDEX|Nifty 50" --interval 1d
    python backtest.py --broker upstox --symbol "NSE_FO|NIFTY25MARFUT" --interval 1h
"""

from __future__ import annotations

import argparse
import logging
import sys

# Suppress verbose trading logs during backtest
logging.disable(logging.CRITICAL)

from backtesting import run_backtest, print_report           # noqa: E402
from backtesting.real_data import run_real_backtest          # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Sensitivity table (synthetic only)
# ─────────────────────────────────────────────────────────────────────────────

def _sensitivity_table(base_trials: int = 300, rr: float = 2.0, risk: float = 1.0) -> None:
    win_rates = [0.40, 0.45, 0.50, 0.55, 0.58, 0.60, 0.65, 0.70]
    sep = "-" * 86
    header = (
        f"{'Win Rate':>9} | {'Trades':>6} | {'Sig%':>5} | "
        f"{'PF':>5} | {'Expect':>7} | {'MaxDD%':>6} | "
        f"{'Net%':>8} | {'Consec-L':>8}"
    )
    print(f"\n{'=' * 86}")
    print(f"  ICT STRATEGY — SENSITIVITY TABLE   (win-rate sweep, synthetic)")
    print(f"  Trials={base_trials}  RR={rr}  Risk={risk}%  Balance=$100,000")
    print(f"{'=' * 86}")
    print(header)
    print(sep)

    for wr in win_rates:
        m = run_backtest(n_trials=base_trials, win_rate=wr, rr_ratio=rr, risk_pct=risk, seed=42)
        if "error" in m:
            print(f"{wr:>8.0%}  | (no trades)")
            continue
        print(
            f"{wr:>8.0%}   | {m['total_trades']:>6} | {m['signal_rate_pct']:>4.0f}% | "
            f"{m['profit_factor']:>5.2f} | {m['expectancy_r']:>+7.3f}R | "
            f"{m['max_drawdown_pct']:>6.2f}% | "
            f"{m['net_profit_pct']:>+7.2f}% | {m['max_consec_losses']:>8}"
        )

    print(sep)
    print(f"  Break-even WR for RR={rr}: {1/(1+rr)*100:.1f}%\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="ICT Strategy Backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode selection
    p.add_argument("--broker",   default="mock",
                   choices=["mock", "delta", "upstox"],
                   help="'mock' = synthetic; 'delta'/'upstox' = real historical data")
    p.add_argument("--symbol",   default="",
                   help="Instrument symbol for real-data mode "
                        "(e.g. BTCUSDT for Delta, 'NSE_INDEX|Nifty 50' for Upstox)")
    p.add_argument("--interval", default="4h",
                   choices=["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"],
                   help="Candle interval for real-data mode (default: 4h)")

    # Synthetic-only
    p.add_argument("--trials",    type=int,   default=500,
                   help="[synthetic] Number of test cases (default: 500)")
    p.add_argument("--win-rate",  type=float, default=0.58,
                   help="[synthetic] Fraction reaching TP (default: 0.58)")
    p.add_argument("--scenarios", action="store_true",
                   help="[synthetic] Print sensitivity table across win rates")

    # Shared
    p.add_argument("--rr",      type=float, default=2.0,  help="Min R:R ratio (default: 2.0)")
    p.add_argument("--risk",    type=float, default=1.0,  help="Risk %% per trade (default: 1.0)")
    p.add_argument("--balance", type=float, default=100_000.0, help="Initial balance")
    p.add_argument("--step",    type=int,   default=10,
                   help="[real] Window advance step in candles (default: 10)")
    p.add_argument("--fetch",   type=int,   default=500,
                   help="[real] Max candles to fetch (default: 500)")
    p.add_argument("--seed",    type=int,   default=42,   help="[synthetic] Random seed")

    args = p.parse_args()

    # ── Synthetic mode ─────────────────────────────────────────────────────
    if args.broker == "mock":
        if args.scenarios:
            _sensitivity_table(base_trials=300, rr=args.rr, risk=args.risk)
            return

        print(
            f"\nICT Backtest (synthetic) — {args.trials} trials | "
            f"win_rate={args.win_rate:.0%} | RR={args.rr} | risk={args.risk}%"
        )
        metrics = run_backtest(
            n_trials        = args.trials,
            win_rate        = args.win_rate,
            rr_ratio        = args.rr,
            risk_pct        = args.risk,
            initial_balance = args.balance,
            seed            = args.seed,
        )
        print_report(metrics)
        return

    # ── Real-data mode ─────────────────────────────────────────────────────
    symbol = args.symbol
    if not symbol:
        symbol = "BTCUSD" if args.broker == "delta" else "NSE_INDEX|Nifty 50"

    print(
        f"\nICT Backtest (real data) — broker={args.broker} | "
        f"symbol={symbol} | interval={args.interval} | "
        f"RR={args.rr} | risk={args.risk}%"
    )
    print("Fetching historical data…")

    m = run_real_backtest(
        broker_name     = args.broker,
        symbol          = symbol,
        interval        = args.interval,
        step            = args.step,
        rr_ratio        = args.rr,
        risk_pct        = args.risk,
        initial_balance = args.balance,
        fetch_count     = args.fetch,
    )

    if "error" in m:
        print(f"\n[ERROR] {m['error']}")
        print(f"  Broker  : {m.get('broker', '—')}")
        print(f"  Symbol  : {m.get('symbol', '—')}")
        print(f"  Interval: {m.get('interval', '—')}")
        sys.exit(1)

    # Extra real-data info before standard report
    print(f"  Candles fetched    : {m.get('candles_fetched', '—')}")
    print(f"  Run at (UTC)       : {m.get('run_at', '—')}")
    print_report(m)


if __name__ == "__main__":
    main()
