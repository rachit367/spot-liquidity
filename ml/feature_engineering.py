"""
Feature engineering for the ICT ML learning pipeline.

All functions take a pandas DataFrame with OHLCV columns and return
numerical features that describe market conditions at trade entry time.

Column requirements
-------------------
Required : open, high, low, close
Optional : volume  (features using volume fall back to 0.5/neutral when absent)

Feature groups
--------------
  Group A — Technical indicators   : rsi, ema_dist_20, ema_dist_50,
                                      atr_pct, bb_width, macd_hist
  Group B — Market structure       : trend_slope, hh_count, ll_count,
                                      trend_strength
  Group C — ICT-specific           : sweep_depth, ob_width_atr,
                                      price_in_ob, fvg_size_atr, kill_zone
  Group D — Volume / momentum      : vol_spike, vol_trend, body_ratio
  Group E — Time encoding          : hour_sin, hour_cos, day_sin, day_cos
  Group F — Signal quality         : rr_ratio, risk_pct
  Group G — Candle context         : close_vs_ob_mid, prev_close_chg
  Group H — Order flow / volume    : vwap_dist, vol_delta, rel_volume,
                                      obv_slope, vol_price_corr,
                                      vol_concentration

Total: 32 features
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from execution import TradeSignal


# ─────────────────────────────────────────────────────────────────────────────
# Feature name registry (used by dataset_builder + inference for column names)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES: list[str] = [
    # A — Technical indicators
    "rsi_14",
    "ema_dist_20",
    "ema_dist_50",
    "atr_pct",
    "bb_width",
    "macd_hist",
    # B — Market structure
    "trend_slope",
    "hh_count",
    "ll_count",
    "trend_strength",
    # C — ICT-specific
    "sweep_depth",
    "ob_width_atr",
    "price_in_ob",
    "fvg_size_atr",
    "kill_zone",
    # D — Volume / momentum
    "vol_spike",
    "vol_trend",
    "body_ratio",
    # E — Time encoding
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    # F — Signal quality
    "rr_ratio",
    "risk_pct",
    # G — Candle context
    "close_vs_ob_mid",
    "prev_close_chg",
    # H — Order flow / volume profile
    "vwap_dist",
    "vol_delta",
    "rel_volume",
    "obv_slope",
    "vol_price_corr",
    "vol_concentration",
]

N_FEATURES = len(FEATURE_NAMES)


# ─────────────────────────────────────────────────────────────────────────────
# Group A — Technical indicators
# ─────────────────────────────────────────────────────────────────────────────

def _rsi(close: np.ndarray, period: int = 14) -> float:
    """
    Wilder's RSI, returned as 0–1 (divide by 100).

    If fewer than period+1 bars, returns 0.5 (neutral).
    """
    if len(close) < period + 1:
        return 0.5
    deltas = np.diff(close[-(period + 1):])
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss == 0:
        return 1.0
    rs = avg_gain / avg_loss
    return rs / (1 + rs)          # equivalent to RSI/100 without the ×100


def _ema(series: np.ndarray, span: int) -> np.ndarray:
    """Simple exponential moving average via pandas (mirrors TA libraries)."""
    return pd.Series(series).ewm(span=span, adjust=False).mean().values


def _ema_dist(close: np.ndarray, span: int) -> float:
    """(close[-1] - EMA[-1]) / close[-1]  — normalised distance."""
    if len(close) < span:
        return 0.0
    ema_val = float(_ema(close, span)[-1])
    c = float(close[-1])
    return (c - ema_val) / c if c != 0 else 0.0


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
         period: int = 14) -> float:
    """Average True Range as a float."""
    if len(close) < 2:
        return 0.0
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:]  - close[:-1]),
        ),
    )
    return float(tr[-period:].mean()) if len(tr) >= period else float(tr.mean())


def _bollinger_width(close: np.ndarray, period: int = 20) -> float:
    """
    (upper - lower) / middle  — Bollinger Band %width.

    Returns 0 when insufficient data.
    """
    if len(close) < period:
        return 0.0
    window = close[-period:]
    mid = window.mean()
    std = window.std(ddof=1)
    if mid == 0:
        return 0.0
    return (4 * std) / mid   # (upper-lower)/mid = 4σ/mid for 2-std bands


def _macd_hist(close: np.ndarray) -> float:
    """
    MACD histogram: (EMA12 - EMA26) - EMA9(EMA12-EMA26).

    Returns value normalised by close[-1].
    """
    if len(close) < 26:
        return 0.0
    fast = _ema(close, 12)
    slow = _ema(close, 26)
    macd_line = fast - slow
    signal    = _ema(macd_line, 9)
    hist = float(macd_line[-1] - signal[-1])
    c = float(close[-1])
    return hist / c if c != 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Group B — Market structure
# ─────────────────────────────────────────────────────────────────────────────

def _trend_slope(close: np.ndarray, n: int = 20) -> float:
    """
    Normalised linear-regression slope over last n closes.

    slope_points_per_bar / close[-1]  — positive = up, negative = down.
    """
    n = min(n, len(close))
    if n < 3:
        return 0.0
    x = np.arange(n, dtype=float)
    y = close[-n:].astype(float)
    slope = float(np.polyfit(x, y, 1)[0])
    c = float(close[-1])
    return (slope / c) * n if c != 0 else 0.0


def _swing_counts(high: np.ndarray, low: np.ndarray,
                  n: int = 3, lookback: int = 30) -> tuple[float, float]:
    """
    Count swing highs / swing lows in the last ``lookback`` bars.

    Returns (hh_frac, ll_frac) normalised by lookback.
    """
    if len(high) < lookback:
        lookback = len(high)
    h = pd.Series(high[-lookback:])
    l = pd.Series(low[-lookback:])
    win = 2 * n + 1
    sh = (h.rolling(win, center=True, min_periods=n + 1).max() == h).sum()
    sl = (l.rolling(win, center=True, min_periods=n + 1).min() == l).sum()
    return float(sh) / lookback, float(sl) / lookback


def _trend_strength(close: np.ndarray, period: int = 14) -> float:
    """
    ADX-like trend strength proxy: std of returns / mean(abs returns).

    Returns 0–1 (higher = stronger directional move).
    """
    if len(close) < period + 1:
        return 0.5
    rets = np.diff(close[-(period + 1):]) / close[-(period + 1):-1]
    abs_rets = np.abs(rets)
    if abs_rets.mean() == 0:
        return 0.0
    # Ratio of mean signed return to mean absolute return (direction consistency)
    return float(min(abs(rets.mean()) / abs_rets.mean(), 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Group C — ICT-specific
# ─────────────────────────────────────────────────────────────────────────────

def _ict_features(
    signal: TradeSignal,
    atr: float,
    last_close: float,
    timestamp: Optional[datetime] = None,
) -> dict[str, float]:
    """
    Extract ICT features directly from the TradeSignal geometry.

    sweep_depth  : risk as % of close (how far SL is from entry)
    ob_width_atr : OB height in ATR multiples (capped at 3)
    price_in_ob  : 1 if close is within OB zone, else 0
    fvg_size_atr : estimated FVG size (0.5 × ATR normalised; enhanced by caller)
    kill_zone    : 1 if timestamp is London/NY open, else 0
    """
    risk = abs(signal.entry - signal.stop_loss)
    sweep_depth  = (risk / last_close) if last_close > 0 else 0.0
    ob_width_atr = min((risk / atr), 3.0) if atr > 0 else 0.0

    ob_low  = min(signal.entry, signal.stop_loss)
    ob_high = max(signal.entry, signal.stop_loss)
    price_in_ob = 1.0 if ob_low <= last_close <= ob_high else 0.0

    fvg_size_atr = 0.5   # default; can be overridden by scan logic

    kz = 0.0
    if timestamp is not None:
        h, m = timestamp.hour, timestamp.minute
        if (8 <= h < 10) or (h == 13 and m >= 30) or (h == 14) or (h == 15 and m <= 30):
            kz = 1.0

    return {
        "sweep_depth":  sweep_depth,
        "ob_width_atr": ob_width_atr,
        "price_in_ob":  price_in_ob,
        "fvg_size_atr": fvg_size_atr,
        "kill_zone":    kz,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Group D — Volume / momentum
# ─────────────────────────────────────────────────────────────────────────────

def _get_volume(df: pd.DataFrame) -> np.ndarray | None:
    """Safely extract volume array. Returns None if absent or all-zero."""
    if "volume" not in df.columns:
        return None
    vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0).values
    if vol.sum() == 0:
        return None
    return vol


def _volume_features(
    df: pd.DataFrame,
    opens: np.ndarray,
    close: np.ndarray,
    lookback: int = 20,
) -> tuple[float, float]:
    """
    vol_spike : last-bar volume / rolling 20-bar average (capped at 3)
    vol_trend : fraction of up-bars (close > open) in last 20 bars
    """
    vol = _get_volume(df)
    if vol is None:
        return 1.0, 0.5

    lb  = min(lookback, len(vol))

    avg_vol   = float(vol[-lb:].mean())
    vol_spike = float(vol[-1] / avg_vol) if avg_vol > 0 else 1.0
    vol_spike = min(vol_spike, 3.0)

    up_bars   = float(np.sum(close[-lb:] > opens[-lb:]))
    vol_trend = up_bars / lb

    return vol_spike, vol_trend


def _body_ratio(opens: np.ndarray, close: np.ndarray,
                high: np.ndarray, low: np.ndarray,
                lookback: int = 20) -> float:
    """Average (body / total range) in last lookback bars."""
    lb   = min(lookback, len(close))
    body = np.abs(close[-lb:] - opens[-lb:])
    rang = high[-lb:] - low[-lb:]
    return float(np.mean(body / np.where(rang > 0, rang, 1e-9)))


# ─────────────────────────────────────────────────────────────────────────────
# Group H — Order flow / volume profile
# ─────────────────────────────────────────────────────────────────────────────

def _vwap_distance(
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    vol: np.ndarray | None, lookback: int = 20,
) -> float:
    """
    Distance of close from VWAP, normalised by close.

    Positive = above VWAP (bullish), negative = below (bearish).
    Returns 0 when volume data is absent.
    """
    if vol is None:
        return 0.0
    lb = min(lookback, len(close))
    typical = (high[-lb:] + low[-lb:] + close[-lb:]) / 3.0
    v = vol[-lb:]
    total_vol = v.sum()
    if total_vol == 0:
        return 0.0
    vwap = float(np.sum(typical * v) / total_vol)
    c = float(close[-1])
    return (c - vwap) / c if c != 0 else 0.0


def _volume_delta(
    opens: np.ndarray, close: np.ndarray, vol: np.ndarray | None,
    lookback: int = 20,
) -> float:
    """
    Net buying vs selling pressure.

    Sum of (volume on up candles - volume on down candles) / total volume.
    Returns 0 when no volume data. Range: -1 to +1.
    """
    if vol is None:
        return 0.0
    lb = min(lookback, len(close))
    c = close[-lb:]
    o = opens[-lb:]
    v = vol[-lb:]
    buy_vol  = v[c >= o].sum()
    sell_vol = v[c < o].sum()
    total = buy_vol + sell_vol
    if total == 0:
        return 0.0
    return float((buy_vol - sell_vol) / total)


def _relative_volume(vol: np.ndarray | None, lookback: int = 20) -> float:
    """
    Relative volume: current bar vs 20-bar average.

    A spike > 2 often indicates institutional activity (liquidity sweep).
    Capped at 5. Returns 1.0 (neutral) when no volume data.
    """
    if vol is None:
        return 1.0
    lb = min(lookback, len(vol))
    avg = float(vol[-lb:].mean())
    if avg == 0:
        return 1.0
    return min(float(vol[-1] / avg), 5.0)


def _obv_slope(close: np.ndarray, vol: np.ndarray | None,
              lookback: int = 20) -> float:
    """
    Normalised slope of On-Balance Volume over last N bars.

    OBV confirms a trend: rising OBV + rising price = strong trend.
    Returns 0 when no volume data.
    """
    if vol is None or len(close) < 3:
        return 0.0
    lb = min(lookback, len(close))
    c = close[-lb:]
    v = vol[-lb:]

    # Build OBV
    direction = np.sign(np.diff(c))
    obv = np.zeros(len(c))
    obv[0] = v[0]
    for i in range(1, len(c)):
        obv[i] = obv[i - 1] + direction[i - 1] * v[i]

    # Normalised slope
    if len(obv) < 3:
        return 0.0
    x = np.arange(len(obv), dtype=float)
    slope = float(np.polyfit(x, obv, 1)[0])
    obv_range = obv.max() - obv.min()
    if obv_range == 0:
        return 0.0
    return float(np.clip(slope / (obv_range / len(obv) + 1e-9), -1, 1))


def _vol_price_corr(
    close: np.ndarray, vol: np.ndarray | None, lookback: int = 20,
) -> float:
    """
    Correlation between volume and absolute price changes.

    High correlation = smart money is moving price on volume.
    Returns 0 when no volume data. Range: -1 to +1.
    """
    if vol is None or len(close) < 5:
        return 0.0
    lb = min(lookback, len(close) - 1)
    price_chg = np.abs(np.diff(close[-(lb + 1):]))
    v = vol[-lb:]
    if len(price_chg) != len(v):
        v = v[:len(price_chg)]
    if v.std() == 0 or price_chg.std() == 0:
        return 0.0
    return float(np.corrcoef(v, price_chg)[0, 1])


def _vol_concentration(
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    vol: np.ndarray | None, lookback: int = 20, n_bins: int = 10,
) -> float:
    """
    Volume concentration at current price level (0–1).

    Builds a simple volume-at-price profile and returns the fraction
    of total volume that occurred at the price bin containing close[-1].
    High concentration = strong support/resistance level.
    Returns 0 when no volume data.
    """
    if vol is None or len(close) < 5:
        return 0.0
    lb = min(lookback, len(close))
    h = high[-lb:]
    l = low[-lb:]
    c = close[-lb:]
    v = vol[-lb:]

    price_min = l.min()
    price_max = h.max()
    if price_max == price_min:
        return 0.0

    # Assign each bar's volume to a price bin based on typical price
    typical = (h + l + c) / 3.0
    bins = np.linspace(price_min, price_max, n_bins + 1)
    bin_idx = np.digitize(typical, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    vol_profile = np.zeros(n_bins)
    for i in range(len(v)):
        vol_profile[bin_idx[i]] += v[i]

    # Which bin is close[-1] in?
    current_bin = int(np.clip(np.digitize(float(c[-1]), bins) - 1, 0, n_bins - 1))
    total_vol = vol_profile.sum()
    if total_vol == 0:
        return 0.0
    return float(vol_profile[current_bin] / total_vol)


# ─────────────────────────────────────────────────────────────────────────────
# Group E — Time encoding
# ─────────────────────────────────────────────────────────────────────────────

def _time_features(ts: Optional[datetime]) -> tuple[float, float, float, float]:
    """
    Cyclical encoding of hour-of-day and day-of-week.

    Returns (hour_sin, hour_cos, day_sin, day_cos).
    """
    if ts is None:
        ts = datetime.now(timezone.utc)

    frac_hour = ts.hour + ts.minute / 60.0
    frac_day  = ts.weekday()   # 0=Mon, 6=Sun

    return (
        math.sin(2 * math.pi * frac_hour / 24.0),
        math.cos(2 * math.pi * frac_hour / 24.0),
        math.sin(2 * math.pi * frac_day  / 7.0),
        math.cos(2 * math.pi * frac_day  / 7.0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────────────────────────────────────

def extract(
    df: pd.DataFrame,
    signal: TradeSignal,
    timestamp: Optional[datetime] = None,
    fvg_size: Optional[float] = None,
) -> np.ndarray:
    """
    Build the full (32,) feature vector for a trade entry.

    Parameters
    ----------
    df        : OHLCV DataFrame (≥ 26 rows recommended).
    signal    : The TradeSignal generated at entry time.
    timestamp : UTC datetime of the entry candle.  Uses now() if None.
    fvg_size  : Optional explicit FVG height (abs price units).

    Returns
    -------
    np.ndarray of shape (32,) dtype float32.
    All NaN / Inf values are replaced with 0.
    """
    # ── Prepare data ────────────────────────────────────────────────────────
    df = df.copy()
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

    if len(df) < 5:
        return np.zeros(N_FEATURES, dtype=np.float32)

    close = df["close"].values.astype(float)
    high  = df["high"].values.astype(float)
    low   = df["low"].values.astype(float)
    opens = df["open"].values.astype(float)

    last_close = float(close[-1])

    # ── Group A ─────────────────────────────────────────────────────────────
    rsi_14     = _rsi(close)
    ema_d20    = _ema_dist(close, 20)
    ema_d50    = _ema_dist(close, 50)
    atr        = _atr(high, low, close)
    atr_pct    = (atr / last_close) if last_close > 0 else 0.0
    bb_width   = _bollinger_width(close)
    macd_h     = _macd_hist(close)

    # ── Group B ─────────────────────────────────────────────────────────────
    t_slope     = _trend_slope(close)
    hh, ll      = _swing_counts(high, low)
    t_strength  = _trend_strength(close)

    # ── Group C ─────────────────────────────────────────────────────────────
    ict = _ict_features(signal, atr, last_close, timestamp)
    if fvg_size is not None and atr > 0:
        ict["fvg_size_atr"] = min(fvg_size / atr, 3.0)

    # ── Group D ─────────────────────────────────────────────────────────────
    vol_spike, vol_trend = _volume_features(df, opens, close)
    body_r = _body_ratio(opens, close, high, low)

    # ── Group E ─────────────────────────────────────────────────────────────
    h_sin, h_cos, d_sin, d_cos = _time_features(timestamp)

    # ── Group F ─────────────────────────────────────────────────────────────
    risk   = abs(signal.entry - signal.stop_loss)
    reward = abs(signal.take_profit - signal.entry)
    rr     = (reward / risk) if risk > 0 else 0.0
    rr_ratio_norm = min(rr / 5.0, 1.0)
    risk_pct = (risk / last_close) if last_close > 0 else 0.0

    # ── Group G ─────────────────────────────────────────────────────────────
    ob_low  = min(signal.entry, signal.stop_loss)
    ob_high = max(signal.entry, signal.stop_loss)
    ob_mid  = (ob_low + ob_high) / 2.0
    close_vs_ob = ((last_close - ob_mid) / atr) if atr > 0 else 0.0

    prev_chg = ((close[-1] - close[-2]) / close[-2]) if len(close) >= 2 and close[-2] != 0 else 0.0

    # ── Group H — Order flow / volume profile ──────────────────────────────
    vol_arr = _get_volume(df)
    vwap_d   = _vwap_distance(high, low, close, vol_arr)
    v_delta  = _volume_delta(opens, close, vol_arr)
    rel_vol  = _relative_volume(vol_arr)
    obv_sl   = _obv_slope(close, vol_arr)
    vp_corr  = _vol_price_corr(close, vol_arr)
    vol_conc = _vol_concentration(high, low, close, vol_arr)

    # ── Assemble ─────────────────────────────────────────────────────────────
    vec = np.array([
        rsi_14,
        ema_d20,
        ema_d50,
        atr_pct,
        bb_width,
        macd_h,
        t_slope,
        hh,
        ll,
        t_strength,
        ict["sweep_depth"],
        ict["ob_width_atr"],
        ict["price_in_ob"],
        ict["fvg_size_atr"],
        ict["kill_zone"],
        vol_spike,
        vol_trend,
        body_r,
        h_sin,
        h_cos,
        d_sin,
        d_cos,
        rr_ratio_norm,
        risk_pct,
        close_vs_ob,
        prev_chg,
        # H — Order flow
        vwap_d,
        v_delta,
        rel_vol,
        obv_sl,
        vp_corr,
        vol_conc,
    ], dtype=np.float32)

    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


def features_to_dict(vec: np.ndarray) -> dict[str, float]:
    """Return a {name: value} mapping for a feature vector."""
    return {k: float(v) for k, v in zip(FEATURE_NAMES, vec)}
