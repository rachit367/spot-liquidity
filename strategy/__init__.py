"""
ICT (Inner Circle Trading) strategy implementation.

Entry logic — ALL of the following must align:
  1. Kill Zone       : current UTC time is within London Open or NY Open
  2. Market Structure: BOS (Break of Structure) confirms trend direction
  3. Liquidity Sweep : price wicked through a prior swing high/low then reversed
  4. Order Block     : price has returned to the last significant OB
  5. Fair Value Gap  : an FVG exists near the OB (imbalance to be filled)
  6. R:R check       : reward-to-risk ratio meets minimum threshold

Kill Zones (UTC):
  London Open : 08:00 – 10:00  (IST 13:30 – 15:30)
  NY Open     : 13:30 – 15:30  (IST 19:00 – 21:00)
  Asian Range : 00:00 – 04:00  (IST 05:30 – 09:30)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone

import pandas as pd

from brokers.base import BaseBroker
from execution import TradeSignal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kill Zone windows  (start_hour, start_min, end_hour, end_min)  — all UTC
# ---------------------------------------------------------------------------
KILL_ZONES: dict[str, tuple[int, int, int, int]] = {
    "london_open": (8,  0,  10, 0),
    "ny_open":     (13, 30, 15, 30),
    "asian_range": (0,  0,  4,  0),
}


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseStrategy(ABC):
    """All strategies must implement ``generate_signal``."""

    @abstractmethod
    def generate_signal(self, broker: BaseBroker) -> TradeSignal | None:
        """Return a TradeSignal if a setup is present, else None."""


# ---------------------------------------------------------------------------
# ICT Strategy
# ---------------------------------------------------------------------------

class ICTStrategy(BaseStrategy):
    """
    Full ICT strategy combining BOS/CHoCH, liquidity sweeps,
    order blocks, and fair value gaps.

    Parameters
    ----------
    symbol       : Broker instrument key (e.g. ``"NSE_FO|NIFTY25MARFUT"``)
    interval     : OHLC candle interval forwarded to broker (e.g. ``"15m"``)
    swing_length : Candles on each side used to identify swing highs/lows
    ob_lookback  : How many recent candles to search for order blocks / FVGs
    kill_zones   : Subset of KILL_ZONES keys to respect (None = all three)
    rr_ratio     : Minimum reward-to-risk ratio; trade skipped if not met
    """

    def __init__(
        self,
        symbol: str,
        interval: str = "15m",
        swing_length: int = 5,
        ob_lookback: int = 10,
        kill_zones: list[str] | None = None,
        rr_ratio: float = 2.0,
    ) -> None:
        self.symbol = symbol
        self.interval = interval
        self.swing_length = swing_length
        self.ob_lookback = ob_lookback
        self.kill_zones = kill_zones if kill_zones is not None else ["london_open", "ny_open"]
        self.rr_ratio = rr_ratio

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def generate_signal(self, broker: BaseBroker) -> TradeSignal | None:
        """Run the full ICT checklist and return a signal or None."""

        # Step 1 — Kill zone filter (skipped when kill_zones=[])
        if self.kill_zones:
            active_zone = self._active_kill_zone()
            if not active_zone:
                logger.debug("ICT: not in a kill zone — skipping")
                return None
        else:
            active_zone = "none"  # no filter — trade any time (e.g. Indian market)

        # Step 2 — Fetch OHLC
        df = broker.get_ohlc(self.symbol, self.interval, count=100)
        if df is None or len(df) < 20:
            logger.warning("ICT: insufficient OHLC data for %s", self.symbol)
            return None
        df = _prepare(df)

        # Step 3 — Market structure
        structure = self._detect_market_structure(df)
        if structure == "neutral":
            logger.debug("ICT: market structure neutral — skipping")
            return None
        logger.debug("ICT: market structure = %s", structure)

        # Step 4 — Liquidity sweep
        sweep = self._detect_liquidity_sweep(df, structure)
        if not sweep:
            logger.debug("ICT: no liquidity sweep — skipping")
            return None
        logger.debug("ICT: sweep detected %s", sweep)

        # Step 5 — Order block
        ob = self._find_order_block(df, structure)
        if not ob:
            logger.debug("ICT: no order block found — skipping")
            return None
        logger.debug("ICT: OB [%.2f – %.2f]", ob["low"], ob["high"])

        # Step 6 — Price must be inside the OB zone
        ltp = float(df["close"].iloc[-1])
        if not _price_in_zone(ltp, ob["low"], ob["high"]):
            logger.debug(
                "ICT: price %.2f outside OB [%.2f – %.2f] — skipping",
                ltp, ob["low"], ob["high"],
            )
            return None

        # Step 7 — Fair value gap
        fvg = self._find_fvg(df, structure)
        if not fvg:
            logger.debug("ICT: no FVG near OB — skipping")
            return None
        logger.debug("ICT: FVG %s", fvg)

        # Step 8 — Build signal
        signal = self._build_signal(ltp, ob, structure, sweep, active_zone)
        if signal is None:
            return None

        # Step 9 — R:R gate
        risk   = abs(signal.entry - signal.stop_loss)
        reward = abs(signal.take_profit - signal.entry)
        rr = reward / risk if risk > 0 else 0
        if rr < self.rr_ratio:
            logger.debug("ICT: R:R %.2f below minimum %.2f — skipping", rr, self.rr_ratio)
            return None

        logger.info(
            "ICT SIGNAL  %s  %s  entry=%.2f  SL=%.2f  TP=%.2f  R:R=%.1fx  zone=%s",
            signal.direction.upper(), self.symbol,
            signal.entry, signal.stop_loss, signal.take_profit, rr, active_zone,
        )
        return signal

    # ------------------------------------------------------------------
    # Step 1 — Kill zone
    # ------------------------------------------------------------------

    def _active_kill_zone(self) -> str | None:
        """Return the name of the currently active kill zone, or None."""
        now = datetime.now(timezone.utc)
        for name in self.kill_zones:
            sh, sm, eh, em = KILL_ZONES[name]
            start = now.replace(hour=sh, minute=sm, second=0, microsecond=0)
            end   = now.replace(hour=eh, minute=em, second=0, microsecond=0)
            if start <= now <= end:
                return name
        return None

    # ------------------------------------------------------------------
    # Step 3 — Market structure: BOS / CHoCH
    # ------------------------------------------------------------------

    def _detect_market_structure(self, df: pd.DataFrame) -> str:
        """
        Classify trend by comparing the FIRST vs LAST confirmed swing points.

        Comparing only the last two swings breaks when price is pulling back
        into an order block — that pullback creates a "lower high" even in a
        healthy bullish trend.  Looking at oldest-vs-newest captured swing
        gives the true higher-timeframe bias.

        - Bullish : latest swing high > earliest swing high
                    AND latest swing low > earliest swing low
        - Bearish : latest swing high < earliest swing high
                    AND latest swing low < earliest swing low
        - Neutral : mixed or insufficient data
        """
        n = self.swing_length
        highs = _swing_highs(df, n)
        lows  = _swing_lows(df, n)

        if len(highs) < 2 or len(lows) < 2:
            return "neutral"

        # Trend = direction from the oldest detected swing to the most recent one
        hh = highs["high"].iloc[-1] > highs["high"].iloc[0]   # overall higher highs
        hl = lows["low"].iloc[-1]   > lows["low"].iloc[0]     # overall higher lows
        lh = highs["high"].iloc[-1] < highs["high"].iloc[0]   # overall lower highs
        ll = lows["low"].iloc[-1]   < lows["low"].iloc[0]     # overall lower lows

        if hh and hl:
            return "bullish"
        if lh and ll:
            return "bearish"
        return "neutral"

    # ------------------------------------------------------------------
    # Step 4 — Liquidity sweep
    # ------------------------------------------------------------------

    def _detect_liquidity_sweep(
        self, df: pd.DataFrame, structure: str
    ) -> dict | None:
        """
        Detect a liquidity sweep within the recent ``ob_lookback`` candles.

        Bullish setup: a candle's LOW wicks below the rolling minimum of the
                       preceding 20 candles, but its CLOSE finishes above that
                       minimum → equal-lows / stop cluster taken, then reclaimed.
        Bearish setup: mirror logic using highs.

        Using a rolling window minimum (rather than formal swing lows) avoids
        the circular problem where the sweep wick itself prevents its target
        level from being classified as a swing low.
        """
        lookback = self.ob_lookback
        if len(df) < lookback + 5:
            return None

        # Reference window: candles before the sweep search zone
        ref_end   = len(df) - lookback
        ref_start = max(0, ref_end - 20)
        reference = df.iloc[ref_start:ref_end]

        search = df.iloc[-lookback:]   # candles to scan for the sweep candle

        if structure == "bullish":
            if reference.empty:
                return None
            prior_low = float(reference["low"].min())
            for _, row in search.iterrows():
                if row["low"] < prior_low < row["close"]:
                    return {"type": "sweep_low", "level": round(prior_low, 2)}

        elif structure == "bearish":
            if reference.empty:
                return None
            prior_high = float(reference["high"].max())
            for _, row in search.iterrows():
                if row["close"] < prior_high < row["high"]:
                    return {"type": "sweep_high", "level": round(prior_high, 2)}

        return None

    # ------------------------------------------------------------------
    # Step 5 — Order block
    # ------------------------------------------------------------------

    def _find_order_block(self, df: pd.DataFrame, structure: str) -> dict | None:
        """
        Find the most recent valid order block.

        Bullish OB : the last BEARISH candle immediately before 2+ consecutive
                     bullish candles (the origin of the impulsive move up).
        Bearish OB : the last BULLISH candle immediately before 2+ consecutive
                     bearish candles (the origin of the impulsive move down).

        The OB zone spans [candle.low, candle.high].
        """
        # Search in a recent but not-too-recent window
        end   = len(df) - 2          # exclude last 2 candles
        start = max(0, end - self.ob_lookback - 10)
        window = df.iloc[start:end].reset_index(drop=True)

        for i in range(len(window) - 3, 0, -1):
            c0 = window.iloc[i]
            c1 = window.iloc[i + 1]
            c2 = window.iloc[i + 2]

            if structure == "bullish":
                # c0 bearish, c1+c2 bullish impulse
                if (c0["close"] < c0["open"]
                        and c1["close"] > c1["open"]
                        and c2["close"] > c2["open"]):
                    return {
                        "type": "bullish_ob",
                        "high": float(c0["high"]),
                        "low":  float(c0["low"]),
                    }

            elif structure == "bearish":
                # c0 bullish, c1+c2 bearish impulse
                if (c0["close"] > c0["open"]
                        and c1["close"] < c1["open"]
                        and c2["close"] < c2["open"]):
                    return {
                        "type": "bearish_ob",
                        "high": float(c0["high"]),
                        "low":  float(c0["low"]),
                    }

        return None

    # ------------------------------------------------------------------
    # Step 7 — Fair value gap
    # ------------------------------------------------------------------

    def _find_fvg(self, df: pd.DataFrame, structure: str) -> dict | None:
        """
        Detect a Fair Value Gap (3-candle imbalance) in the recent window.

        Bullish FVG : candle[i].high  <  candle[i+2].low
                      Gap exists between the top of candle i and bottom of candle i+2.
        Bearish FVG : candle[i].low   >  candle[i+2].high
                      Gap exists between the bottom of candle i and top of candle i+2.
        """
        end   = len(df) - 1
        start = max(0, end - self.ob_lookback - 5)
        window = df.iloc[start:end].reset_index(drop=True)

        for i in range(len(window) - 3, 0, -1):
            c0 = window.iloc[i]
            c2 = window.iloc[i + 2]

            if structure == "bullish" and c0["high"] < c2["low"]:
                return {
                    "type":   "bullish_fvg",
                    "top":    float(c2["low"]),
                    "bottom": float(c0["high"]),
                }

            if structure == "bearish" and c0["low"] > c2["high"]:
                return {
                    "type":   "bearish_fvg",
                    "top":    float(c0["low"]),
                    "bottom": float(c2["high"]),
                }

        return None

    # ------------------------------------------------------------------
    # Step 8 — Signal construction
    # ------------------------------------------------------------------

    def _build_signal(
        self,
        ltp: float,
        ob: dict,
        structure: str,
        sweep: dict,
        kill_zone: str,
    ) -> TradeSignal | None:
        """
        Construct a TradeSignal from the confirmed ICT confluences.

        Entry      : current LTP (market order at OB)
        Stop Loss  : just beyond the OB edge with a 0.1 % buffer
        Take Profit: entry ± (risk × rr_ratio)
        """
        buffer = ltp * 0.001  # 0.1 % buffer

        if structure == "bullish":
            entry       = ltp
            stop_loss   = round(ob["low"] - buffer, 2)
            risk        = entry - stop_loss
            if risk <= 0:
                return None
            take_profit = round(entry + risk * self.rr_ratio, 2)
            direction   = "long"
            reason = (
                f"ICT Bullish | KZ={kill_zone} | BOS up | "
                f"sweep_low@{sweep['level']:.2f} | "
                f"bullish OB [{ob['low']:.2f}–{ob['high']:.2f}]"
            )

        elif structure == "bearish":
            entry       = ltp
            stop_loss   = round(ob["high"] + buffer, 2)
            risk        = stop_loss - entry
            if risk <= 0:
                return None
            take_profit = round(entry - risk * self.rr_ratio, 2)
            direction   = "short"
            reason = (
                f"ICT Bearish | KZ={kill_zone} | BOS down | "
                f"sweep_high@{sweep['level']:.2f} | "
                f"bearish OB [{ob['low']:.2f}–{ob['high']:.2f}]"
            )

        else:
            return None

        return TradeSignal(
            symbol=self.symbol,
            entry=round(entry, 2),
            stop_loss=stop_loss,
            take_profit=take_profit,
            direction=direction,
            reason=reason,
        )


# ---------------------------------------------------------------------------
# Module-level helpers (pure functions, reusable in backtesting)
# ---------------------------------------------------------------------------

def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce OHLC columns to float and drop NaN rows."""
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def _swing_highs(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Return rows where 'high' is the rolling maximum over a window of
    ``2n + 1`` candles centred on that row (i.e. local swing high).
    """
    mask = df["high"].rolling(2 * n + 1, center=True, min_periods=n + 1).max() == df["high"]
    return df[mask]


def _swing_lows(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Return rows where 'low' is the rolling minimum over a window of
    ``2n + 1`` candles centred on that row (i.e. local swing low).
    """
    mask = df["low"].rolling(2 * n + 1, center=True, min_periods=n + 1).min() == df["low"]
    return df[mask]


def _price_in_zone(price: float, low: float, high: float) -> bool:
    """Return True if price is within [low, high]."""
    return low <= price <= high


__all__ = ["BaseStrategy", "ICTStrategy", "KILL_ZONES"]
