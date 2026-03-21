"""Risk management — position sizing and daily loss limits."""

from __future__ import annotations

import logging
from datetime import datetime

from brokers.base import RiskLimitExceeded

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Enforces per-trade risk sizing and a cumulative daily loss cap.

    Args:
        max_daily_loss_pct: Maximum daily loss as a percentage of balance.
        daily_reset_hour: Hour (0-23) at which the daily loss counter resets.
    """

    def __init__(
        self,
        max_daily_loss_pct: float = 3.0,
        daily_reset_hour: int = 9,
    ) -> None:
        self.max_daily_loss_pct = max_daily_loss_pct
        self.daily_reset_hour = daily_reset_hour
        self._daily_loss: float = 0.0
        self._last_reset_date: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_quantity(
        self,
        balance: float,
        risk_pct: float,
        entry: float,
        stop_loss: float,
    ) -> int:
        """
        Position size based on fixed-percentage risk.

        ``quantity = (balance * risk_pct / 100) / |entry - stop_loss|``
        """
        sl_distance = abs(entry - stop_loss)
        if sl_distance == 0:
            logger.warning("SL distance is zero — skipping trade")
            return 0
        risk_amount = balance * (risk_pct / 100.0)
        qty = int(risk_amount / sl_distance)
        return max(qty, 0)

    def can_trade(self, balance: float) -> bool:
        """Return False if the daily loss limit has been breached."""
        self._maybe_reset()
        limit = balance * (self.max_daily_loss_pct / 100.0)
        if self._daily_loss >= limit:
            logger.warning(
                "Daily loss limit reached: %.2f / %.2f",
                self._daily_loss, limit,
            )
            return False
        return True

    def check_trade_allowed(self, balance: float) -> None:
        """Raise ``RiskLimitExceeded`` if daily limit is breached."""
        if not self.can_trade(balance):
            raise RiskLimitExceeded(
                f"Daily loss {self._daily_loss:.2f} exceeds "
                f"{self.max_daily_loss_pct}% of {balance:.2f}"
            )

    def record_loss(self, amount: float) -> None:
        """Record a loss for daily tracking (pass positive value)."""
        self._maybe_reset()
        self._daily_loss += abs(amount)
        logger.info("Daily loss updated: %.2f", self._daily_loss)

    @property
    def daily_loss(self) -> float:
        return self._daily_loss

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _maybe_reset(self) -> None:
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        if today != self._last_reset_date and now.hour >= self.daily_reset_hour:
            self._daily_loss = 0.0
            self._last_reset_date = today
            logger.debug("Daily loss counter reset for %s", today)
