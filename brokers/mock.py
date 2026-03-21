"""
Mock broker for local strategy and paper-trading tests.

Generates realistic synthetic OHLC data so the full ICT pipeline
(strategy → execution → paper executor) can be tested without
any live API credentials.

Usage:
    python main.py --broker mock --mode paper
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from brokers.base import BaseBroker

logger = logging.getLogger(__name__)


class MockBroker(BaseBroker):
    """
    Synthetic broker that produces a deterministic OHLC series embedding
    all five ICT conditions:

      1. Bullish market structure  (HH + HL across the full series)
      2. Order block               (last bearish candle + 2 bullish)
      3. Fair value gap            (3-candle imbalance)
      4. Liquidity sweep           (wick past prior level, close back above)
      5. Price at OB zone          (settlement inside the OB range)

    Parameters
    ----------
    base_price  : Starting / reference price
    trend       : ``"bullish"`` or ``"bearish"``
    volatility  : Wick noise as fraction of price (default 0.002 = 0.2 %)
    balance     : Initial virtual account balance
    """

    def __init__(
        self,
        base_price: float = 22500.0,
        trend: str = "bullish",
        volatility: float = 0.002,
        balance: float = 100_000.0,
    ) -> None:
        self._base_price = base_price
        self._trend      = trend.lower()
        self._volatility = volatility
        self._balance    = balance
        self._orders: dict[str, dict] = {}
        self._order_counter = 0
        logger.info(
            "MockBroker ready — base=%.2f trend=%s vol=%.3f",
            base_price, trend, volatility,
        )

    # ------------------------------------------------------------------
    # BaseBroker interface
    # ------------------------------------------------------------------

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str = "market",
        quantity: int = 1,
        price: float | None = None,
        stop_price: float | None = None,
    ) -> str:
        self._order_counter += 1
        order_id = f"MOCK-{self._order_counter:06d}"
        fill_price = price if price and order_type == "limit" else self._base_price
        self._orders[order_id] = {
            "order_id":   order_id,
            "symbol":     symbol,
            "side":       side,
            "order_type": order_type,
            "quantity":   quantity,
            "price":      fill_price,
            "status":     "complete",
        }
        logger.info("Mock order placed: %s %s %d @ %.2f", order_id, side, quantity, fill_price)
        return order_id

    def get_price(self, symbol: str) -> float:
        noise = self._base_price * self._volatility * random.uniform(-1, 1)
        return round(self._base_price + noise, 2)

    def get_ohlc(self, symbol: str, interval: str = "30m", count: int = 100) -> pd.DataFrame:
        """Return ``count`` synthetic candles with embedded ICT structure."""
        random.seed(42)
        np.random.seed(42)
        candles = _build_ict_candles(self._base_price, self._trend, self._volatility, count)
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        logger.debug("MockBroker: generated %d candles for %s", len(df), symbol)
        return df

    def get_balance(self) -> float:
        return self._balance

    def get_order_status(self, order_id: str) -> dict:
        return self._orders.get(order_id, {"status": "not_found"})

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self._orders:
            self._orders[order_id]["status"] = "cancelled"
            logger.info("Mock order cancelled: %s", order_id)
            return True
        return False


# ---------------------------------------------------------------------------
# Deterministic ICT candle generator (module-level, easy to unit-test)
# ---------------------------------------------------------------------------

def _build_ict_candles(
    base_price: float,
    trend: str,
    volatility: float,
    count: int,
) -> list[tuple]:
    """
    Build ``count`` candles with a guaranteed ICT setup.

    KEY INSIGHT: the OB candle (which has a big bearish body) must sit
    INSIDE the search window (last 20 candles), NOT inside the reference
    window that determines prior_low for the sweep check.

    Bullish price roadmap (base_price = 22500):
    ─────────────────────────────────────────────────────────────────
     idx  0-9    flat baseline                      ~22500
     idx 10-24   impulse 1 up     22500 → 22995     (SH1)
     idx 25-34   retrace 1 down   22995 → 22702     (SL1, HL vs base)
     idx 35-54   impulse 2 up     22702 → 23490     (SH2, HH vs SH1)
     idx 55-79   retrace 2 down   23490 → 23255     (sl2)
                 Reference zone (60-80) min ≈ 23255 = prior_low ✓
     idx 80      SWEEP candle     wick → 23200 (<sl2), close 23265 (>sl2) ✓
     idx 81      OB bearish       23265 → 23189     (OB candle, in search zone)
     idx 82-83   OB impulse up    23189 → 23325     (FVG gap between 82 and 84)
     idx 84      FVG candle 3     low > candle[82].high  → bullish FVG ✓
     idx 85-99   settle in OB     price ≈ 23265 ∈ [23189, 23325] ✓
    ─────────────────────────────────────────────────────────────────
    Structure (first vs last swing):
      Swing highs: SH1≈22995, SH2≈23490  → HH ✓
      Swing lows:  ~22500, sl2≈23255      → HL ✓
    """
    np.random.seed(42)
    now = datetime.now().replace(second=0, microsecond=0)
    s   = 1 if trend == "bullish" else -1
    bp  = base_price
    v   = volatility

    # ── Key price levels ──────────────────────────────────────────────────
    if trend == "bullish":
        sh1   = bp * 1.022    # 22995  — first swing high
        sl1   = bp * 1.009    # 22702  — first swing low (HL vs base)
        sh2   = bp * 1.044    # 23490  — second swing high (HH)
        sl2   = sh2 * 0.990   # 23255  — second swing low (HL vs sl1)
        ob_h  = sl2 * 1.003   # 23325  — OB zone top
        ob_l  = sl2 * 0.992   # 23189  — OB zone bottom (OB candle low)
        fvg_push = bp * 0.003  # size of FVG impulse
    else:
        sl1   = bp * 0.978
        sh1   = bp * 0.991
        sl2   = sh1 * 1.010
        ob_l  = sl2 * 0.997
        ob_h  = sl2 * 1.008
        sh2   = sl2 * 1.001
        fvg_push = -bp * 0.003

    # settlement = slightly above sl2 so it's inside OB AND above prior_low
    settle = sl2 * 1.0004   # e.g., 23264 (> 23255 = sl2, inside [23189, 23325])
    sweep_wick = sl2 * 0.993 # wick goes below sl2 by 0.7 %

    # ── Segment helpers ───────────────────────────────────────────────────
    def lerp(a: float, b: float, t: float) -> float:
        return a + (b - a) * t

    def tiny() -> float:
        """Tiny noise: < 0.05 % of base_price, never changes direction."""
        return bp * v * np.random.randn() * 0.05

    def make(idx: int, o: float, c: float, wt: float = 0.0, wb: float = 0.0) -> tuple:
        ts  = now - timedelta(minutes=30 * (count - idx))
        h   = max(o, c) + abs(wt)
        l   = min(o, c) - abs(wb)
        vol = int(abs(np.random.normal(12000, 2000)))
        return (ts.isoformat(), round(o, 2), round(h, 2), round(l, 2), round(c, 2), vol)

    w = bp * v * 0.4   # standard wick

    # Precompute close targets for segments 0-79 via linear interpolation
    milestones = [
        (0,  9,  bp,   bp),       # flat baseline
        (10, 24, bp,   sh1),      # impulse 1
        (25, 34, sh1,  sl1),      # retrace 1
        (35, 54, sl1,  sh2),      # impulse 2
        (55, 79, sh2,  sl2),      # retrace 2 — ends exactly at sl2
    ]
    targets: list[float] = [sl2] * count   # default
    for seg_start, seg_end, p_start, p_end in milestones:
        n = seg_end - seg_start + 1
        for k in range(n):
            t = k / (n - 1) if n > 1 else 0.0
            targets[seg_start + k] = lerp(p_start, p_end, t)

    candles: list[tuple] = []
    prev_close = bp

    for i in range(count):

        # ── Trend + retrace 2  (idx 0-79) ─────────────────────────────────
        # Reference zone = df[60:80], which covers the tail of retrace 2.
        # min_low in that zone ≈ sl2 ± tiny  →  prior_low ≈ sl2 ✓
        if i <= 79:
            o = prev_close
            c = targets[i] + tiny()
            candles.append(make(i, o, c, w * 0.4, w * 0.4))

        # ── SWEEP candle (idx 80) ──────────────────────────────────────────
        # wick BELOW sl2 (prior_low = min of idx 60-80 ≈ sl2),
        # CLOSE ABOVE sl2  →  sweep condition satisfied
        elif i == 80:
            o = prev_close                   # ≈ sl2
            c = settle                       # settle = sl2 * 1.0004 > sl2 ✓
            if s == 1:                       # bullish: long lower wick
                wb = abs(o - sweep_wick) + w * 0.2   # wick to sweep_wick < sl2
                wt = w * 0.1
            else:                            # bearish: long upper wick
                wt = abs(sweep_wick - o) + w * 0.2
                wb = w * 0.1
            candles.append(make(i, o, c, wt, wb))

        # ── OB bearish candle (idx 81) ─────────────────────────────────────
        elif i == 81:
            o = prev_close                   # ≈ settle ≈ sl2
            c = ob_l                         # closes at OB bottom
            candles.append(make(i, o, c, w * 0.1, w * 0.1))

        # ── OB impulse candle 1 (idx 82) — also FVG candle 1 ─────────────
        # Tight top wick so candle[82].high stays low for the FVG gap
        elif i == 82:
            o = prev_close
            c = lerp(ob_l, ob_h, 0.45) + tiny() * 0.1   # ≈ ob_l + 60 %
            candles.append(make(i, o, c, w * 0.05, w * 0.3))

        # ── OB impulse candle 2 / FVG candle 2 (idx 83) ──────────────────
        # Big bullish body, tiny wicks
        elif i == 83:
            o = prev_close
            c = ob_h
            candles.append(make(i, o, c, w * 0.05, w * 0.05))

        # ── FVG candle 3 (idx 84) ─────────────────────────────────────────
        # low > candle[82].high  →  bullish FVG  ✓
        elif i == 84:
            c82_high = candles[82][2]           # high of candle 82
            # open ABOVE c82_high so low = min(o, c) - wb > c82_high
            o = max(prev_close, c82_high + 2.0)
            c = settle + tiny() * 0.1
            candles.append(make(i, o, c, w * 0.2, 0.0))  # wb=0 → low=min(o,c) > c82_high ✓

        # ── Settle inside OB zone (idx 85-99) ─────────────────────────────
        else:
            o = prev_close
            c = settle + tiny() * 0.3
            c = max(ob_l * 1.0001, min(ob_h * 0.9999, c))
            candles.append(make(i, o, c, w * 0.15, w * 0.15))

        prev_close = candles[-1][4]

    return candles
