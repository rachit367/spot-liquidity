"""
ICT Trading Bot — Live & Paper Trading Engine

Setup:
    python -m venv venv
    # Windows:  venv\\Scripts\\activate
    # Linux/Mac: source venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env      # fill in your API keys
    python main.py
"""

from __future__ import annotations

import argparse
import logging
import sys

from config.settings import settings
from brokers import get_broker
from execution import TradeSignal
from execution.risk_manager import RiskManager
from execution.paper_executor import PaperExecutor
from execution.live_executor import LiveExecutor
from strategy import ICTStrategy

logger = logging.getLogger("main")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ICT Trading Bot")
    p.add_argument(
        "--mode",
        choices=["paper", "live"],
        default=settings.trading_mode,
        help="Trading mode (default: from .env)",
    )
    p.add_argument(
        "--broker",
        choices=["upstox", "delta"],
        default=settings.default_broker,
        help="Broker to use (default: from .env)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mode = args.mode
    broker_name = args.broker

    logger.info("=" * 60)
    logger.info("ICT Trading Bot starting")
    logger.info("  Mode   : %s", mode)
    logger.info("  Broker : %s", broker_name)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Initialise broker
    # ------------------------------------------------------------------
    broker = get_broker(broker_name)
    logger.info("Broker initialised: %s", broker.__class__.__name__)

    # ------------------------------------------------------------------
    # 2. Initialise risk manager and executor
    # ------------------------------------------------------------------
    risk_mgr = RiskManager(
        max_daily_loss_pct=settings.max_daily_loss_pct,
        daily_reset_hour=settings.daily_reset_hour,
    )

    if mode == "paper":
        executor = PaperExecutor(
            initial_balance=settings.initial_paper_balance,
            risk_pct=settings.risk_per_trade_pct,
            risk_manager=risk_mgr,
        )
        logger.info(
            "Paper executor ready (balance: %.2f)", settings.initial_paper_balance
        )
    else:
        executor = LiveExecutor(
            broker=broker,
            risk_pct=settings.risk_per_trade_pct,
            risk_manager=risk_mgr,
        )
        logger.info("Live executor ready")

    # ------------------------------------------------------------------
    # 3. Initialise ICT strategy
    # ------------------------------------------------------------------
    symbol = "NSE_FO|NIFTY25MARFUT" if broker_name == "upstox" else "BTCUSD"
    strategy = ICTStrategy(
        symbol=symbol,
        interval="15m",
        swing_length=5,
        ob_lookback=20,
        # Indian market (NSE/BSE): no kill zone filter needed
        # Delta (crypto): only trade during London + NY sessions
        kill_zones=["london_open", "ny_open"] if broker_name == "delta" else [],
        rr_ratio=2.0,
    )
    logger.info("ICT strategy ready — symbol=%s interval=%s RR=%.1f",
                symbol, strategy.interval, strategy.rr_ratio)

    # ------------------------------------------------------------------
    # 4. Generate signal from strategy
    # ------------------------------------------------------------------
    logger.info("Scanning for ICT setup...")
    signal = strategy.generate_signal(broker)

    if signal is None:
        logger.info("No valid ICT setup at this time — no trade placed.")
        logger.info(
            "Conditions required: Kill Zone + BOS + Liquidity Sweep + "
            "Order Block + FVG + R:R >= %.1f", strategy.rr_ratio
        )
        # In a live loop you would wait and re-scan on the next candle close
        sys.exit(0)

    logger.info("Signal: %s %s @ %.2f  SL=%.2f  TP=%.2f",
                signal.direction.upper(), signal.symbol,
                signal.entry, signal.stop_loss, signal.take_profit)
    logger.info("Reason: %s", signal.reason)

    # ------------------------------------------------------------------
    # 5. Execute
    # ------------------------------------------------------------------
    try:
        result = executor.execute(signal)
        logger.info("Result: %s", result)
    except Exception as exc:
        logger.error("Trade execution failed: %s", exc)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 6. Report state
    # ------------------------------------------------------------------
    if mode == "paper":
        logger.info("Paper balance : %.2f", executor.get_balance())
        logger.info("Open positions: %d", len(executor.get_positions()))
        history = executor.get_trade_history()
        logger.info("Closed trades : %d", len(history))

        # Simulate TP hit for demo purposes
        logger.info("--- Simulating TP hit ---")
        tp_prices = {signal.symbol: signal.take_profit}
        closed = executor.check_exits(tp_prices)
        if closed:
            for t in closed:
                logger.info(
                    "  %s PnL=%.2f (%s)", t["symbol"], t["pnl"], t["outcome"]
                )
        logger.info("Final paper balance: %.2f", executor.get_balance())
    else:
        try:
            balance = broker.get_balance()
            logger.info("Live balance: %.2f", balance)
        except Exception as exc:
            logger.warning("Could not fetch live balance: %s", exc)


if __name__ == "__main__":
    main()
