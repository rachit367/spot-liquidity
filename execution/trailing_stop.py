"""
Trailing stop-loss manager — dynamically adjusts SL as price moves
in the trade's favour.

Rules
-----
1. After 1R of profit → move SL to breakeven (entry price)
2. Beyond 1R → trail SL at 0.5R behind the best price seen
3. SL only moves up (long) or down (short), never backwards

Usage
-----
    from execution.trailing_stop import TrailingStopManager
    tsm = TrailingStopManager()
    tsm.register(order_id, entry=50000, stop_loss=49500, take_profit=51000, direction="long")
    new_sl = tsm.update(order_id, current_price=50600)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

BREAKEVEN_TRIGGER = 1.0   # R-multiples of profit to trigger breakeven
TRAIL_FACTOR      = 0.5   # trail at this × R behind best price


@dataclass
class _TrackedPosition:
    """Internal tracking state for one position."""
    order_id:    str
    entry:       float
    original_sl: float
    take_profit: float
    direction:   str        # "long" or "short"
    risk:        float      # absolute risk in price units (1R)
    best_price:  float      # best price seen since entry
    current_sl:  float      # latest (possibly trailed) SL
    at_breakeven: bool = False


class TrailingStopManager:
    """Manages trailing stops for all open positions."""

    def __init__(self) -> None:
        self._positions: dict[str, _TrackedPosition] = {}

    def register(
        self,
        order_id: str,
        entry: float,
        stop_loss: float,
        take_profit: float,
        direction: str,
    ) -> None:
        """Register a new position for trailing stop management."""
        risk = abs(entry - stop_loss)
        self._positions[order_id] = _TrackedPosition(
            order_id=order_id,
            entry=entry,
            original_sl=stop_loss,
            take_profit=take_profit,
            direction=direction,
            risk=risk,
            best_price=entry,
            current_sl=stop_loss,
        )
        logger.debug(
            "Trailing SL registered: %s %s entry=%.2f sl=%.2f risk=%.2f",
            order_id, direction, entry, stop_loss, risk,
        )

    def update(self, order_id: str, current_price: float) -> float:
        """
        Update the trailing stop for a position based on current price.

        Returns the current (possibly updated) stop-loss price.
        """
        pos = self._positions.get(order_id)
        if pos is None:
            return current_price  # unknown position

        if pos.risk <= 0:
            return pos.current_sl

        # Update best price
        if pos.direction == "long":
            pos.best_price = max(pos.best_price, current_price)
        else:
            pos.best_price = min(pos.best_price, current_price)

        # Calculate R-multiple of current profit
        if pos.direction == "long":
            r_profit = (pos.best_price - pos.entry) / pos.risk
        else:
            r_profit = (pos.entry - pos.best_price) / pos.risk

        # Rule 1: breakeven after 1R
        if r_profit >= BREAKEVEN_TRIGGER and not pos.at_breakeven:
            pos.current_sl = pos.entry
            pos.at_breakeven = True
            logger.info(
                "Trailing SL: %s moved to BREAKEVEN @ %.2f (%.1fR profit)",
                order_id, pos.entry, r_profit,
            )

        # Rule 2: trail at 0.5R behind best price (beyond breakeven)
        if pos.at_breakeven and r_profit > BREAKEVEN_TRIGGER:
            trail_distance = pos.risk * TRAIL_FACTOR

            if pos.direction == "long":
                new_sl = pos.best_price - trail_distance
                # SL only moves up
                if new_sl > pos.current_sl:
                    pos.current_sl = new_sl
                    logger.debug(
                        "Trailing SL: %s trailed to %.2f (best=%.2f, %.1fR)",
                        order_id, new_sl, pos.best_price, r_profit,
                    )
            else:
                new_sl = pos.best_price + trail_distance
                # SL only moves down for shorts
                if new_sl < pos.current_sl:
                    pos.current_sl = new_sl
                    logger.debug(
                        "Trailing SL: %s trailed to %.2f (best=%.2f, %.1fR)",
                        order_id, new_sl, pos.best_price, r_profit,
                    )

        return pos.current_sl

    def get_stop(self, order_id: str) -> float | None:
        """Return the current stop-loss for a position."""
        pos = self._positions.get(order_id)
        return pos.current_sl if pos else None

    def remove(self, order_id: str) -> None:
        """Remove a position (after it's closed)."""
        self._positions.pop(order_id, None)

    def status(self) -> list[dict]:
        """Return status of all tracked positions."""
        return [
            {
                "order_id":     p.order_id,
                "direction":    p.direction,
                "entry":        p.entry,
                "original_sl":  p.original_sl,
                "current_sl":   round(p.current_sl, 2),
                "best_price":   round(p.best_price, 2),
                "at_breakeven": p.at_breakeven,
                "r_profit":     round(
                    (p.best_price - p.entry) / p.risk if p.direction == "long"
                    else (p.entry - p.best_price) / p.risk, 2
                ) if p.risk > 0 else 0,
            }
            for p in self._positions.values()
        ]
