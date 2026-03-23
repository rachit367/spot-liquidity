"""Paper trading executor — simulates trades in memory."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from execution import TradeSignal
from execution.risk_manager import RiskManager
from execution.trailing_stop import TrailingStopManager
from execution.correlation_filter import CorrelationFilter

logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trades")


class PaperExecutor:
    """
    Simulates trade execution without touching real APIs.

    Maintains virtual balance, open positions, and a full trade log.
    """

    def __init__(
        self,
        initial_balance: float = 100_000.0,
        risk_pct: float = 1.0,
        risk_manager: RiskManager | None = None,
        use_trailing_stop: bool = True,
        use_correlation_filter: bool = True,
    ) -> None:
        self._balance = initial_balance
        self._risk_pct = risk_pct
        self._risk_mgr = risk_manager or RiskManager()
        self._positions: list[dict] = []
        self._trade_log: list[dict] = []
        self._order_counter = 0
        self._trailing = TrailingStopManager() if use_trailing_stop else None
        self._corr_filter = CorrelationFilter() if use_correlation_filter else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, signal: TradeSignal) -> dict:
        """Simulate order placement and immediate fill at signal.entry."""
        # Correlation filter check
        if self._corr_filter:
            allowed, reason = self._corr_filter.can_open(
                signal.symbol, signal.direction, self._positions,
            )
            if not allowed:
                logger.warning("Paper trade BLOCKED: %s", reason)
                return {"status": "blocked", "reason": reason}

        self._risk_mgr.check_trade_allowed(self._balance)

        qty = self._risk_mgr.compute_quantity(
            self._balance, self._risk_pct, signal.entry, signal.stop_loss,
        )
        if qty <= 0:
            logger.warning("Computed quantity is 0 — trade skipped")
            return {"status": "skipped", "reason": "quantity_zero"}

        self._order_counter += 1
        order_id = f"PAPER-{self._order_counter:06d}"
        now = datetime.now(timezone.utc).isoformat()

        position = {
            "order_id": order_id,
            "symbol": signal.symbol,
            "direction": signal.direction,
            "entry": signal.entry,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "quantity": qty,
            "opened_at": now,
            "reason": signal.reason,
        }
        self._positions.append(position)

        # Register with trailing stop manager
        if self._trailing:
            self._trailing.register(
                order_id=order_id,
                entry=signal.entry,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                direction=signal.direction,
            )

        result = {
            "status": "filled",
            "order_id": order_id,
            "symbol": signal.symbol,
            "direction": signal.direction,
            "quantity": qty,
            "entry_price": signal.entry,
            "mode": "paper",
        }

        trade_logger.info({
            "event": "order_filled",
            "mode": "paper",
            "broker": "paper",
            **result,
            "timestamp": now,
            "reason": signal.reason,
        })
        logger.info(
            "Paper trade: %s %s %d @ %.2f  SL=%.2f TP=%.2f [%s]",
            signal.direction.upper(), signal.symbol, qty, signal.entry,
            signal.stop_loss, signal.take_profit, order_id,
        )
        return result

    def check_exits(self, current_prices: dict[str, float]) -> list[dict]:
        """
        Check open positions against current prices and close those
        that hit SL or TP.

        Args:
            current_prices: mapping of symbol → current LTP.

        Returns:
            List of closed position dicts.
        """
        closed = []
        still_open = []

        for pos in self._positions:
            price = current_prices.get(pos["symbol"])
            if price is None:
                still_open.append(pos)
                continue

            # Update trailing stop if enabled
            active_sl = pos["stop_loss"]
            if self._trailing:
                active_sl = self._trailing.update(pos["order_id"], price)

            hit_tp = hit_sl = False
            if pos["direction"] == "long":
                hit_tp = price >= pos["take_profit"]
                hit_sl = price <= active_sl
            else:
                hit_tp = price <= pos["take_profit"]
                hit_sl = price >= active_sl

            if hit_tp or hit_sl:
                exit_price = pos["take_profit"] if hit_tp else pos["stop_loss"]
                pnl = self._calc_pnl(pos, exit_price)
                self._balance += pnl
                if pnl < 0:
                    self._risk_mgr.record_loss(abs(pnl))

                outcome = "tp_hit" if hit_tp else "sl_hit"
                record = {
                    **pos,
                    "exit_price": exit_price,
                    "pnl": round(pnl, 2),
                    "outcome": outcome,
                    "closed_at": datetime.now(timezone.utc).isoformat(),
                }
                self._trade_log.append(record)
                closed.append(record)

                # Remove from trailing stop manager
                if self._trailing:
                    self._trailing.remove(pos["order_id"])

                trade_logger.info({
                    "event": "position_closed",
                    "mode": "paper",
                    **record,
                })
                logger.info(
                    "Paper exit: %s %s @ %.2f → %.2f  PnL=%.2f (%s)",
                    pos["direction"].upper(), pos["symbol"],
                    pos["entry"], exit_price, pnl, outcome,
                )
            else:
                still_open.append(pos)

        self._positions = still_open
        return closed

    def get_balance(self) -> float:
        return self._balance

    def get_positions(self) -> list[dict]:
        return list(self._positions)

    def get_trade_history(self) -> list[dict]:
        return list(self._trade_log)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_pnl(position: dict, exit_price: float) -> float:
        qty = position["quantity"]
        entry = position["entry"]
        if position["direction"] == "long":
            return (exit_price - entry) * qty
        else:
            return (entry - exit_price) * qty
