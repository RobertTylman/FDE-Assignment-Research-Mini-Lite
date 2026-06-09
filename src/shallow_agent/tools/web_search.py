"""Tavily-backed web search tool."""

from __future__ import annotations

import os
from typing import Any

import httpx
from langchain_core.tools import tool

DEFAULT_MAX_RESULTS = 5
DEFAULT_SEARCH_DEPTH = "basic"
DEFAULT_TIMEOUT_SECONDS = 30.0


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from exc


def _env_float(name: str, default: float | None = None) -> float | None:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {value!r}.") from exc


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}.")


def _format_result(item: dict[str, Any], index: int) -> str:
    title = item.get("title") or "Untitled"
    url = item.get("url") or ""
    content = item.get("content") or ""
    max_chars = _env_int("TAVILY_SNIPPET_CHARS")
    if max_chars is not None and max_chars > 0 and len(content) > max_chars:
        content = f"{content[:max_chars].rstrip()}..."
    return f"[{index}] {title}\nURL: {url}\nSnippet: {content}".strip()


@tool
async def web_search(query: str) -> str:
    """Search the web for current information and return cited snippets."""

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set.")

    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": _env_int("TAVILY_MAX_RESULTS", DEFAULT_MAX_RESULTS),
        "search_depth": os.environ.get("TAVILY_SEARCH_DEPTH", DEFAULT_SEARCH_DEPTH),
        "include_answer": _env_bool("TAVILY_INCLUDE_ANSWER", False),
        "include_raw_content": _env_bool("TAVILY_INCLUDE_RAW_CONTENT", False),
    }
    timeout = _env_float("TAVILY_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post("https://api.tavily.com/search", json=payload)
            response.raise_for_status()
            data = response.json()
    except httpx.TimeoutException:
        return "Search timed out before results were available."

    results = data.get("results") or []
    if not results:
        return "No results found."

    return "\n\n".join(_format_result(item, index) for index, item in enumerate(results, start=1))
