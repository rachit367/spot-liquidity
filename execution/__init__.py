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


__all__ = ["TradeSignal"]
