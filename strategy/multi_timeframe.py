"""
Multi-timeframe analysis — checks higher-timeframe trend to confirm
lower-timeframe entries.

Maps entry timeframes to appropriate confirmation timeframes:
  5m  → 1h
  15m → 4h
  30m → 4h
  1h  → 1d
  4h  → 1d

Usage
-----
    from strategy.multi_timeframe import get_htf_bias, HTF_MAP
    bias = get_htf_bias(broker, "BTCUSD", entry_interval="15m")
    # → "bullish", "bearish", or "neutral"
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from brokers.base import BaseBroker

logger = logging.getLogger(__name__)


# Mapping: entry timeframe → higher timeframe for confirmation
HTF_MAP: dict[str, str] = {
    "1m":  "15m",
    "3m":  "30m",
    "5m":  "1h",
    "15m": "4h",
    "30m": "4h",
    "1h":  "1d",
    "4h":  "1d",
    "1d":  "1w",
}


def _detect_trend(df: pd.DataFrame, swing_length: int = 5) -> str:
    """
    Detect market structure trend from OHLCV data.
    Returns "bullish", "bearish", or "neutral".

    Uses the same logic as ICTStrategy._detect_market_structure
    but as a standalone function.
    """
    n = swing_length

    # Swing highs
    mask_h = df["high"].rolling(2 * n + 1, center=True, min_periods=n + 1).max() == df["high"]
    highs = df[mask_h]

    # Swing lows
    mask_l = df["low"].rolling(2 * n + 1, center=True, min_periods=n + 1).min() == df["low"]
    lows = df[mask_l]

    if len(highs) < 2 or len(lows) < 2:
        return "neutral"

    hh = highs["high"].iloc[-1] > highs["high"].iloc[0]
    hl = lows["low"].iloc[-1]   > lows["low"].iloc[0]
    lh = highs["high"].iloc[-1] < highs["high"].iloc[0]
    ll = lows["low"].iloc[-1]   < lows["low"].iloc[0]

    if hh and hl:
        return "bullish"
    if lh and ll:
        return "bearish"
    return "neutral"


def get_htf_bias(
    broker: BaseBroker,
    symbol: str,
    entry_interval: str,
    htf_interval: Optional[str] = None,
    candle_count: int = 100,
) -> str:
    """
    Fetch higher-timeframe data and determine the trend bias.

    Parameters
    ----------
    broker          : Broker instance for data fetching
    symbol          : Trading symbol (e.g. "BTCUSD")
    entry_interval  : The entry timeframe (e.g. "15m")
    htf_interval    : Override HTF interval (uses HTF_MAP if None)
    candle_count    : Number of HTF candles to fetch

    Returns
    -------
    "bullish", "bearish", or "neutral"
    """
    if htf_interval is None:
        htf_interval = HTF_MAP.get(entry_interval)
        if htf_interval is None:
            logger.debug("No HTF mapping for %s — skipping", entry_interval)
            return "neutral"

    try:
        df = broker.get_ohlc(symbol, htf_interval, count=candle_count)
        if df is None or len(df) < 20:
            logger.debug("HTF: insufficient data for %s %s", symbol, htf_interval)
            return "neutral"

        # Ensure numeric
        for col in ("open", "high", "low", "close"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

        bias = _detect_trend(df, swing_length=5)
        logger.info(
            "HTF bias: %s %s → %s (from %d candles)",
            symbol, htf_interval, bias, len(df),
        )
        return bias
    except Exception as exc:
        logger.warning("HTF analysis failed: %s", exc)
        return "neutral"


def check_alignment(entry_bias: str, htf_bias: str) -> bool:
    """
    Check if entry-timeframe bias aligns with higher-timeframe bias.

    Rules:
    - HTF neutral → no alignment filter (allow trade)
    - HTF bullish + entry bullish → aligned
    - HTF bearish + entry bearish → aligned
    - Otherwise → not aligned (skip trade)
    """
    if htf_bias == "neutral":
        return True   # No filter if HTF is unclear
    return entry_bias == htf_bias
