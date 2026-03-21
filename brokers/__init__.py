"""Broker factory — returns the correct broker instance by name."""

from __future__ import annotations

from brokers.base import BaseBroker


def get_broker(name: str) -> BaseBroker:
    """
    Instantiate a broker by name.

    Args:
        name: ``"upstox"`` or ``"delta"``

    Returns:
        A fully initialised broker instance.
    """
    name = name.lower().strip()

    if name == "upstox":
        from brokers.upstox import UpstoxBroker
        return UpstoxBroker()
    elif name == "delta":
        from brokers.delta import DeltaBroker
        return DeltaBroker()
    elif name == "mock":
        from brokers.mock import MockBroker
        return MockBroker()
    else:
        raise ValueError(
            f"Unknown broker: {name!r}. Supported: 'upstox', 'delta', 'mock'"
        )


__all__ = ["get_broker", "BaseBroker"]
