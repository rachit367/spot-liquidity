"""Live trading executor — places real orders via broker API."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from brokers.base import BaseBroker, BrokerAPIError, InsufficientBalanceError
from execution import TradeSignal
from execution.risk_manager import RiskManager

logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trades")


class LiveExecutor:
    """
    Executes trades against a real broker.

    Places a market entry order and a separate stop-loss order.
    """

    def __init__(
        self,
        broker: BaseBroker,
        risk_pct: float = 1.0,
        risk_manager: RiskManager | None = None,
    ) -> None:
        self._broker = broker
        self._risk_pct = risk_pct
        self._risk_mgr = risk_manager or RiskManager()

    def execute(self, signal: TradeSignal) -> dict:
        """Place a live order based on the trade signal."""
        now = datetime.now(timezone.utc).isoformat()

        # 1. Risk checks
        balance = self._broker.get_balance()
        self._risk_mgr.check_trade_allowed(balance)

        qty = self._risk_mgr.compute_quantity(
            balance, self._risk_pct, signal.entry, signal.stop_loss,
        )
        if qty <= 0:
            logger.warning("Computed quantity is 0 — trade skipped")
            return {"status": "skipped", "reason": "quantity_zero"}

        # 2. Check sufficient balance (rough estimate)
        estimated_cost = signal.entry * qty
        if estimated_cost > balance:
            raise InsufficientBalanceError(
                f"Need ~{estimated_cost:.2f} but only {balance:.2f} available"
            )

        # 3. Map direction to side
        side = "buy" if signal.direction == "long" else "sell"

        # 4. Place entry order
        try:
            entry_order_id = self._broker.place_order(
                symbol=signal.symbol,
                side=side,
                order_type="market",
                quantity=qty,
            )
        except BrokerAPIError as exc:
            logger.error("Entry order failed: %s", exc)
            trade_logger.info({
                "event": "order_failed",
                "mode": "live",
                "symbol": signal.symbol,
                "error": str(exc),
                "timestamp": now,
            })
            raise

        result = {
            "status": "placed",
            "entry_order_id": entry_order_id,
            "symbol": signal.symbol,
            "direction": signal.direction,
            "quantity": qty,
            "entry_price": signal.entry,
            "mode": "live",
        }

        # 5. Place stop-loss order
        sl_side = "sell" if signal.direction == "long" else "buy"
        try:
            sl_order_id = self._broker.place_order(
                symbol=signal.symbol,
                side=sl_side,
                order_type="sl",
                quantity=qty,
                stop_price=signal.stop_loss,
            )
            result["sl_order_id"] = sl_order_id
        except BrokerAPIError as exc:
            logger.warning("SL order failed (entry still placed): %s", exc)
            result["sl_error"] = str(exc)

        trade_logger.info({
            "event": "order_placed",
            "mode": "live",
            **result,
            "timestamp": now,
            "reason": signal.reason,
        })
        logger.info(
            "Live trade: %s %s %d @ market  SL=%.2f TP=%.2f [entry=%s]",
            signal.direction.upper(), signal.symbol, qty,
            signal.stop_loss, signal.take_profit, entry_order_id,
        )
        return result
