"""LangSmith tracing helpers."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any


DEFAULT_LANGSMITH_PROJECT = "research-mini-lite"


def configure_langsmith() -> None:
    """Enable LangSmith tracing when a LangSmith API key is configured."""

    api_key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
    if not api_key:
        return

    os.environ.setdefault("LANGSMITH_API_KEY", api_key)
    os.environ.setdefault("LANGCHAIN_API_KEY", api_key)
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", DEFAULT_LANGSMITH_PROJECT)
    os.environ.setdefault("LANGCHAIN_PROJECT", os.environ["LANGSMITH_PROJECT"])


def langsmith_enabled() -> bool:
    tracing_enabled = os.environ.get("LANGSMITH_TRACING") or os.environ.get("LANGCHAIN_TRACING_V2")
    api_key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
    return bool(api_key and str(tracing_enabled).strip().lower() in {"1", "true", "yes", "on"})


try:
    from langsmith import traceable as _traceable
except Exception:  # pragma: no cover - optional dependency fallback
    _traceable = None


def traceable(*args: Any, **kwargs: Any) -> Callable:
    if _traceable is None:
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(func: Callable) -> Callable:
            return func

        return decorator
    return _traceable(*args, **kwargs)
