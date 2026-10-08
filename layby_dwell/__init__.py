"""Layby-Dwell: when will an idle LLM session send its next request? A survival curve per session
boundary, for KV cache placement in serving engines."""
from layby_dwell.model import Dwell
from layby_dwell.session import Tracker

__all__ = ["Dwell", "Tracker"]
