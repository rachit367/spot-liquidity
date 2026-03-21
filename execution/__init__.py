"""Execution engine — trade signals, dispatch, and executors."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TradeSignal(BaseModel):
    """Validated trade signal passed from strategy to executor."""
    symbol: str
    entry: float = Field(gt=0)
    stop_loss: float = Field(gt=0)
    take_profit: float = Field(gt=0)
    direction: Literal["long", "short"]
    reason: str = ""


def execute_trade(signal: TradeSignal, executor) -> dict:
    """
    Convenience dispatch — delegates to the pre-built executor.

    In practice, instantiate the executor once in ``main.py`` and call
    ``executor.execute(signal)`` directly so that paper state persists.
    """
    return executor.execute(signal)


__all__ = ["TradeSignal", "execute_trade"]
