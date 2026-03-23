"""
Mistake analyzer — learns from losing trades.

Reads completed trades from trades_ml.csv and identifies patterns that
correlate with losses. Provides actionable insights that help the model
and trader understand *why* trades fail.

Usage
-----
    from ml.mistake_analyzer import get_mistake_report
    report = get_mistake_report()
    print(report["insights"])
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from ml.dataset_builder import TRADES_CSV, load_trades, _parse_features_row
from ml.feature_engineering import FEATURE_NAMES, N_FEATURES

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Feature-based analysis
# ─────────────────────────────────────────────────────────────────────────────

def _feature_loss_correlation(df: pd.DataFrame) -> list[dict]:
    """
    Identify which features are most different between wins and losses.

    For each feature, compare the mean value in winning vs losing trades.
    Rank by the magnitude of the difference (normalised by std).
    """
    # Parse feature vectors
    X_list, results = [], []
    for _, row in df.iterrows():
        vec = _parse_features_row(str(row.get("features_json", "")))
        if vec is not None:
            X_list.append(vec)
            results.append(int(row["result"]))

    if len(X_list) < 10:
        return []

    X = np.vstack(X_list)
    y = np.array(results)

    wins  = X[y == 1]
    losses = X[y == 0]

    if len(wins) < 3 or len(losses) < 3:
        return []

    correlations = []
    for i, name in enumerate(FEATURE_NAMES):
        win_mean  = float(wins[:, i].mean())
        loss_mean = float(losses[:, i].mean())
        diff      = loss_mean - win_mean
        pooled_std = float(X[:, i].std())

        if pooled_std > 1e-9:
            effect_size = abs(diff) / pooled_std
        else:
            effect_size = 0.0

        correlations.append({
            "feature":     name,
            "win_mean":    round(win_mean, 4),
            "loss_mean":   round(loss_mean, 4),
            "difference":  round(diff, 4),
            "effect_size": round(effect_size, 4),
            "direction":   "higher_in_losses" if diff > 0 else "lower_in_losses",
        })

    # Sort by effect size (largest difference first)
    correlations.sort(key=lambda x: -x["effect_size"])
    return correlations


def _feature_range_loss_rates(df: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """
    For the top loss-correlated features, compute loss rates
    when the feature is high vs low (split at median).
    """
    X_list, results = [], []
    for _, row in df.iterrows():
        vec = _parse_features_row(str(row.get("features_json", "")))
        if vec is not None:
            X_list.append(vec)
            results.append(int(row["result"]))

    if len(X_list) < 10:
        return []

    X = np.vstack(X_list)
    y = np.array(results)

    corr = _feature_loss_correlation(df)
    insights = []

    for feat_info in corr[:top_n]:
        idx = FEATURE_NAMES.index(feat_info["feature"])
        col = X[:, idx]
        median = float(np.median(col))

        high_mask = col >= median
        low_mask  = col < median

        high_loss_rate = float(1 - y[high_mask].mean()) if high_mask.sum() > 0 else 0.0
        low_loss_rate  = float(1 - y[low_mask].mean())  if low_mask.sum() > 0 else 0.0

        # Generate human-readable insight
        if feat_info["direction"] == "higher_in_losses":
            insight = (
                f"When {feat_info['feature']} is above median ({median:.3f}), "
                f"loss rate is {high_loss_rate*100:.0f}% vs {low_loss_rate*100:.0f}% "
                f"when below — consider filtering signals with high {feat_info['feature']}"
            )
        else:
            insight = (
                f"When {feat_info['feature']} is below median ({median:.3f}), "
                f"loss rate is {low_loss_rate*100:.0f}% vs {high_loss_rate*100:.0f}% "
                f"when above — low {feat_info['feature']} signals underperform"
            )

        insights.append({
            "feature":        feat_info["feature"],
            "effect_size":    feat_info["effect_size"],
            "median":         round(median, 4),
            "high_loss_rate": round(high_loss_rate, 4),
            "low_loss_rate":  round(low_loss_rate, 4),
            "insight":        insight,
        })

    return insights


# ─────────────────────────────────────────────────────────────────────────────
# Time-based analysis
# ─────────────────────────────────────────────────────────────────────────────

def _time_loss_patterns(df: pd.DataFrame) -> dict:
    """
    Analyze loss rates by hour-of-day and day-of-week.
    """
    if "timestamp" not in df.columns:
        return {}

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    if len(df) < 10:
        return {}

    df["hour"] = df["timestamp"].dt.hour
    df["dow"]  = df["timestamp"].dt.day_name()

    # Loss rate by hour
    hourly = {}
    for hour, group in df.groupby("hour"):
        n = len(group)
        losses = int((group["result"] == 0).sum())
        hourly[int(hour)] = {
            "trades": n,
            "losses": losses,
            "loss_rate": round(losses / n, 3) if n > 0 else 0,
        }

    # Loss rate by day of week
    daily = {}
    for dow, group in df.groupby("dow"):
        n = len(group)
        losses = int((group["result"] == 0).sum())
        daily[dow] = {
            "trades": n,
            "losses": losses,
            "loss_rate": round(losses / n, 3) if n > 0 else 0,
        }

    # Find worst hour and day
    worst_hour = max(hourly.items(), key=lambda x: x[1]["loss_rate"]) if hourly else None
    worst_day  = max(daily.items(), key=lambda x: x[1]["loss_rate"]) if daily else None

    return {
        "by_hour":    hourly,
        "by_day":     daily,
        "worst_hour": {
            "hour":      worst_hour[0],
            "loss_rate": worst_hour[1]["loss_rate"],
            "trades":    worst_hour[1]["trades"],
        } if worst_hour else None,
        "worst_day": {
            "day":       worst_day[0],
            "loss_rate": worst_day[1]["loss_rate"],
            "trades":    worst_day[1]["trades"],
        } if worst_day else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Streak and consecutive loss analysis
# ─────────────────────────────────────────────────────────────────────────────

def _streak_analysis(df: pd.DataFrame) -> dict:
    """Analyze losing and winning streaks."""
    if len(df) < 5:
        return {}

    results   = df["result"].astype(int).tolist()
    max_win   = 0
    max_loss  = 0
    cur       = 0

    for r in results:
        if r == 1:
            cur = max(cur, 0) + 1
            max_win = max(max_win, cur)
        else:
            cur = min(cur, 0) - 1
            max_loss = max(max_loss, -cur)

    # Average losing streak length
    streaks = []
    cur_streak = 0
    for r in results:
        if r == 0:
            cur_streak += 1
        else:
            if cur_streak > 0:
                streaks.append(cur_streak)
            cur_streak = 0
    if cur_streak > 0:
        streaks.append(cur_streak)

    avg_loss_streak = float(np.mean(streaks)) if streaks else 0.0

    return {
        "max_win_streak":       max_win,
        "max_loss_streak":      max_loss,
        "avg_loss_streak":      round(avg_loss_streak, 1),
        "total_loss_streaks":   len(streaks),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Direction analysis
# ─────────────────────────────────────────────────────────────────────────────

def _direction_analysis(df: pd.DataFrame) -> dict:
    """Compare win rates for long vs short trades."""
    if "direction" not in df.columns or len(df) < 5:
        return {}

    result = {}
    for direction in ["long", "short"]:
        subset = df[df["direction"] == direction]
        if len(subset) > 0:
            wins   = int((subset["result"] == 1).sum())
            losses = int((subset["result"] == 0).sum())
            result[direction] = {
                "trades":   len(subset),
                "wins":     wins,
                "losses":   losses,
                "win_rate": round(wins / len(subset), 3),
            }

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main report generator
# ─────────────────────────────────────────────────────────────────────────────

def get_mistake_report(min_trades: int = 10) -> dict:
    """
    Generate a full mistake analysis report.

    Returns a dict with:
        summary           — basic win/loss stats
        feature_insights  — top features correlated with losses
        time_patterns     — loss rates by hour and day
        streaks           — losing streak statistics
        direction         — long vs short performance
        top_insights      — human-readable actionable insights
    """
    df = load_trades(min_rows=min_trades)
    if df is None:
        return {
            "error": f"Not enough trades (need at least {min_trades})",
            "total_trades": 0,
        }

    wins   = int((df["result"] == 1).sum())
    losses = int((df["result"] == 0).sum())
    total  = len(df)

    summary = {
        "total_trades": total,
        "wins":         wins,
        "losses":       losses,
        "win_rate":     round(wins / total, 3) if total > 0 else 0,
    }

    # Feature analysis
    feature_insights = _feature_range_loss_rates(df)
    feature_corr     = _feature_loss_correlation(df)

    # Time patterns
    time_patterns = _time_loss_patterns(df)

    # Streaks
    streaks = _streak_analysis(df)

    # Direction
    direction = _direction_analysis(df)

    # Compile top insights (human-readable)
    top_insights = []
    for fi in feature_insights[:3]:
        top_insights.append(fi["insight"])

    if time_patterns.get("worst_hour"):
        wh = time_patterns["worst_hour"]
        if wh["trades"] >= 5:
            top_insights.append(
                f"Hour {wh['hour']}:00 UTC has the highest loss rate "
                f"({wh['loss_rate']*100:.0f}% over {wh['trades']} trades)"
            )

    if time_patterns.get("worst_day"):
        wd = time_patterns["worst_day"]
        if wd["trades"] >= 5:
            top_insights.append(
                f"{wd['day']} has the highest loss rate "
                f"({wd['loss_rate']*100:.0f}% over {wd['trades']} trades)"
            )

    if direction:
        for d_name, d_info in direction.items():
            if d_info["win_rate"] < 0.4 and d_info["trades"] >= 5:
                top_insights.append(
                    f"{d_name.capitalize()} trades have a low win rate "
                    f"({d_info['win_rate']*100:.0f}%) — consider reducing "
                    f"{d_name} exposure"
                )

    return {
        "summary":           summary,
        "feature_insights":  feature_insights,
        "feature_correlations": feature_corr[:10],   # top 10
        "time_patterns":     time_patterns,
        "streaks":           streaks,
        "direction":         direction,
        "top_insights":      top_insights,
    }
