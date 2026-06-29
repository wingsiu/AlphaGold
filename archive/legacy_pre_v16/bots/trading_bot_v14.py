#!/usr/bin/env python3
"""Deprecated shim — production base bot is trading_bot_base.py (used by v15 hybrid)."""
from __future__ import annotations

from trading_bot_base import AlphaGoldBaseBot, BotState

AlphaGoldV14Bot = AlphaGoldBaseBot  # backward-compat alias

__all__ = ["AlphaGoldBaseBot", "AlphaGoldV14Bot", "BotState"]
