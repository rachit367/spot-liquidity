"""
Dynamic threshold adjustment — auto-tunes ML confidence threshold
based on recent trade performance.

Raises threshold during losing streaks (be more selective),
lowers it after wins (take more trades). Bounded [0.50, 0.85].

Usage
-----
    from ml.dynamic_threshold import DynamicThreshold
    dt = DynamicThreshold(base=0.60)
    dt.record_outcome(win=True)
    current = dt.get_threshold()  # → 0.58
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

MIN_THRESHOLD = 0.50
MAX_THRESHOLD = 0.85
STREAK_STEP   = 0.02   # threshold adjustment per streak trade
WINDOW_SIZE   = 20      # rolling window of recent trades


@dataclass
class DynamicThreshold:
    """Auto-adjusting confidence threshold."""
    base:       float = 0.60
    enabled:    bool  = True
    _history:   deque = field(default_factory=lambda: deque(maxlen=WINDOW_SIZE))
    _streak:    int   = 0       # positive = win streak, negative = loss streak

    def record_outcome(self, win: bool) -> float:
        """Record a trade outcome and return the updated threshold."""
        self._history.append(1 if win else 0)

        # Update streak
        if win:
            self._streak = max(self._streak, 0) + 1
        else:
            self._streak = min(self._streak, 0) - 1

        new_threshold = self.get_threshold()
        logger.debug(
            "DynamicThreshold: win=%s streak=%d threshold=%.3f",
            win, self._streak, new_threshold,
        )
        return new_threshold

    def get_threshold(self) -> float:
        """Return the current adjusted threshold."""
        if not self.enabled:
            return self.base

        # Adjust based on streak
        adjustment = -self._streak * STREAK_STEP  # losses raise, wins lower

        # Also factor in recent win rate
        if len(self._history) >= 5:
            recent_wr = sum(self._history) / len(self._history)
            # If winning > 60%, we can be less selective
            if recent_wr > 0.60:
                adjustment -= 0.02
            # If winning < 40%, be more selective
            elif recent_wr < 0.40:
                adjustment += 0.02

        return max(MIN_THRESHOLD, min(MAX_THRESHOLD, self.base + adjustment))

    def reset(self) -> None:
        """Reset to base threshold."""
        self._history.clear()
        self._streak = 0

    def status(self) -> dict:
        """Return current state."""
        recent_trades = list(self._history)
        recent_wr = sum(recent_trades) / len(recent_trades) if recent_trades else 0
        return {
            "enabled":          self.enabled,
            "base_threshold":   self.base,
            "current_threshold": round(self.get_threshold(), 3),
            "streak":           self._streak,
            "recent_trades":    len(recent_trades),
            "recent_win_rate":  round(recent_wr, 3),
        }


# Module-level singleton
_instance: DynamicThreshold | None = None


def get_dynamic_threshold() -> DynamicThreshold:
    """Return the global DynamicThreshold instance."""
    global _instance
    if _instance is None:
        _instance = DynamicThreshold()
    return _instance
