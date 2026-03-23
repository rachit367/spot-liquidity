"""
Feature extraction for the ICT ML model.

Produces a 15-element float vector from a 100-candle OHLC DataFrame and a
TradeSignal.  All features are normalised so the model is scale-invariant.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from execution import TradeSignal


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

FEATURE_NAMES: list[str] = [
    "atr_pct",           # ATR / close  — volatility
    "trend_slope",       # linear regression slope of close (normalised)
    "price_chg_pct",     # % move in last 20 candles
    "n_swing_highs",     # swing high count in lookback (normalised 0-1)
    "n_swing_lows",      # swing low count in lookback
    "sweep_depth_pct",   # how far the sweep wick extended beyond prior level
    "ob_width_atr",      # OB height as multiple of ATR
    "price_in_ob",       # 1 = price is inside OB, 0 = outside
    "fvg_size_atr",      # FVG size as multiple of ATR
    "rr_ratio",          # actual R:R from signal (normalised)
    "volume_trend",      # up-volume fraction in last 20 candles
    "hour_sin",          # time-of-day cyclical encoding (sin)
    "hour_cos",          # time-of-day cyclical encoding (cos)
    "body_ratio",        # avg candle body / range — measures conviction
    "close_vs_ob_mid",   # (close - OB_mid) / ATR — position within OB
]

N_FEATURES = len(FEATURE_NAMES)


def extract(df: pd.DataFrame, signal: TradeSignal) -> np.ndarray:
    """
    Return a (15,) float32 feature vector.

    Parameters
    ----------
    df     : OHLC DataFrame (at least 20 rows).  Columns: open, high, low, close.
             Optional: volume.
    signal : The TradeSignal that was generated from this df window.
    """
    df = df.copy()
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

    if len(df) < 5:
        return np.zeros(N_FEATURES, dtype=np.float32)

    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    opens = df["open"].values

    # --- ATR (14) ---
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:]  - close[:-1]),
        ),
    )
    atr = float(tr[-14:].mean()) if len(tr) >= 14 else float(tr.mean())
    last_close = float(close[-1])
    atr_pct = atr / last_close if last_close > 0 else 0.0

    # --- Trend slope (linear regression of last 20 closes) ---
    n_reg = min(20, len(close))
    x = np.arange(n_reg, dtype=float)
    y = close[-n_reg:]
    slope = float(np.polyfit(x, y, 1)[0])
    trend_slope = (slope / last_close) * n_reg if last_close > 0 else 0.0

    # --- Price change over last 20 candles ---
    price_chg_pct = (
        float((close[-1] - close[-min(20, len(close))]) / close[-min(20, len(close))])
        if close[-min(20, len(close))] > 0
        else 0.0
    )

    # --- Swing count (last 20 candles, n=3) ---
    n_sw = 3
    w20 = df.iloc[-20:].reset_index(drop=True) if len(df) >= 20 else df.reset_index(drop=True)
    sh_mask = w20["high"].rolling(2 * n_sw + 1, center=True, min_periods=n_sw + 1).max() == w20["high"]
    sl_mask = w20["low"].rolling(2 * n_sw + 1, center=True, min_periods=n_sw + 1).min()  == w20["low"]
    n_swing_highs = float(sh_mask.sum()) / len(w20)
    n_swing_lows  = float(sl_mask.sum()) / len(w20)

    # --- Sweep depth ---
    risk    = abs(signal.entry - signal.stop_loss)
    sweep_depth_pct = (risk / last_close) if last_close > 0 else 0.0

    # --- OB width in ATR ---
    ob_width = abs(signal.stop_loss - signal.entry)  # entry is at OB edge
    ob_width_atr = (ob_width / atr) if atr > 0 else 0.0

    # --- Price inside OB ---
    ob_low  = min(signal.entry, signal.stop_loss)
    ob_high = max(signal.entry, signal.stop_loss)
    price_in_ob = 1.0 if ob_low <= last_close <= ob_high else 0.0

    # --- FVG size (estimated as 0.5 × ATR as proxy when not passed explicitly) ---
    fvg_size_atr = 0.5  # placeholder; real value set by caller if available

    # --- R:R ratio (capped at 5 to normalise) ---
    reward = abs(signal.take_profit - signal.entry)
    rr = (reward / risk) if risk > 0 else 0.0
    rr_ratio = min(rr / 5.0, 1.0)

    # --- Volume trend ---
    if "volume" in df.columns:
        vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0).values
        up_bars = np.sum((close[-20:] > opens[-20:]) & (vol[-20:] > 0))
        volume_trend = float(up_bars) / min(20, len(close))
    else:
        volume_trend = 0.5  # neutral if no volume data

    # --- Hour cyclical (use last bar's index as proxy if no timestamp) ---
    hour = 0.0
    if "timestamp" in df.columns or df.index.dtype == "datetime64[ns, UTC]":
        try:
            ts = df.index[-1] if hasattr(df.index[-1], "hour") else pd.to_datetime(df.index[-1])
            hour = float(ts.hour) + float(getattr(ts, "minute", 0)) / 60.0
        except Exception:
            hour = 0.0
    hour_sin = math.sin(2 * math.pi * hour / 24.0)
    hour_cos = math.cos(2 * math.pi * hour / 24.0)

    # --- Body ratio ---
    body = np.abs(close[-20:] - opens[-20:])
    rang = high[-20:] - low[-20:]
    body_ratio = float(np.mean(body / np.where(rang > 0, rang, 1e-9)))

    # --- Close vs OB mid ---
    ob_mid = (ob_low + ob_high) / 2.0
    close_vs_ob_mid = ((last_close - ob_mid) / atr) if atr > 0 else 0.0

    features = np.array([
        atr_pct,
        trend_slope,
        price_chg_pct,
        n_swing_highs,
        n_swing_lows,
        sweep_depth_pct,
        ob_width_atr,
        price_in_ob,
        fvg_size_atr,
        rr_ratio,
        volume_trend,
        hour_sin,
        hour_cos,
        body_ratio,
        close_vs_ob_mid,
    ], dtype=np.float32)

    # Replace any NaN/Inf with 0 to keep the model happy
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features
