"""
Correlation filter — prevents doubling exposure on correlated symbols.

Groups correlated symbols together. Only allows one directional
position per group (e.g., can't go long BTC AND long ETH simultaneously
since they're ~90% correlated).

Hedging (opposite direction) is allowed within a group.

Usage
-----
    from execution.correlation_filter import CorrelationFilter
    cf = CorrelationFilter()
    allowed, reason = cf.can_open("ETHUSD", "long", open_positions)
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Correlation groups
# ─────────────────────────────────────────────────────────────────────────────

CORRELATION_GROUPS: dict[str, list[str]] = {
    "crypto_major": ["BTCUSD", "ETHUSD", "BTCUSDT", "ETHUSDT"],
    "crypto_alt_l1": ["SOLUSD", "AVAXUSD", "DOTUSD", "ADAUSD", "SOLUSDT"],
    "crypto_alt_l2": ["MATICUSD", "LINKUSD", "UNIUSD", "AAVEUSD"],
    "crypto_meme": ["DOGEUSD", "SHIBUSD", "PEPEUSD"],
    # Indian equity groups (for Upstox)
    "nifty_bank": ["NIFTY BANK", "BANKNIFTY"],
    "nifty_it": ["NIFTY IT", "TCS", "INFY", "WIPRO"],
}


class CorrelationFilter:
    """
    Filters trades based on correlation group exposure.

    Rules:
    - Max 1 directional position per correlation group
    - Opposite direction (hedge) within a group is allowed
    - Symbols not in any group are always allowed
    """

    def __init__(
        self,
        groups: dict[str, list[str]] | None = None,
        max_per_group: int = 1,
    ) -> None:
        self.groups = groups or CORRELATION_GROUPS
        self.max_per_group = max_per_group
        # Build reverse lookup: symbol → group name
        self._symbol_to_group: dict[str, str] = {}
        for group_name, symbols in self.groups.items():
            for sym in symbols:
                self._symbol_to_group[sym.upper()] = group_name

    def get_group(self, symbol: str) -> Optional[str]:
        """Return the correlation group for a symbol, or None."""
        return self._symbol_to_group.get(symbol.upper())

    def can_open(
        self,
        symbol: str,
        direction: str,
        open_positions: list[dict],
    ) -> tuple[bool, str]:
        """
        Check if a new position can be opened.

        Parameters
        ----------
        symbol         : The symbol to trade (e.g. "ETHUSD")
        direction      : "long" or "short"
        open_positions : List of dicts with at least {symbol, direction}

        Returns
        -------
        (allowed: bool, reason: str)
        """
        group = self.get_group(symbol)
        if group is None:
            return True, "Symbol not in any correlation group — allowed"

        # Find existing positions in the same group and same direction
        same_group_same_dir = []
        for pos in open_positions:
            pos_sym = pos.get("symbol", "").upper()
            pos_dir = pos.get("direction", "")
            pos_group = self.get_group(pos_sym)

            if pos_group == group and pos_dir == direction:
                same_group_same_dir.append(pos_sym)

        if len(same_group_same_dir) >= self.max_per_group:
            reason = (
                f"Correlation filter: BLOCKED — already {direction} on "
                f"{', '.join(same_group_same_dir)} in group '{group}' "
                f"(max {self.max_per_group} per group per direction)"
            )
            logger.info(reason)
            return False, reason

        return True, f"Allowed ({group}: {len(same_group_same_dir)}/{self.max_per_group})"

    def get_exposure(self, open_positions: list[dict]) -> dict:
        """
        Get current exposure breakdown by correlation group.

        Returns dict of {group_name: {long: [symbols], short: [symbols]}}.
        """
        exposure: dict[str, dict[str, list[str]]] = {}

        for pos in open_positions:
            sym = pos.get("symbol", "").upper()
            direction = pos.get("direction", "")
            group = self.get_group(sym)

            if group is None:
                group = "ungrouped"

            if group not in exposure:
                exposure[group] = {"long": [], "short": []}

            if direction in ("long", "short"):
                exposure[group][direction].append(sym)

        return exposure
