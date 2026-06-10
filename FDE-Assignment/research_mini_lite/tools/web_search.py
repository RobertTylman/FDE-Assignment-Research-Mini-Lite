"""Tavily-backed web search tool."""

from __future__ import annotations

import os
from typing import Any

import httpx
from langchain_core.tools import tool

from research_mini_lite.observability import traceable

DEFAULT_CHUNKS_PER_SOURCE = 2
DEFAULT_INCLUDE_ANSWER = "advanced"
DEFAULT_MAX_RESULTS = 8
DEFAULT_SEARCH_DEPTH = "advanced"
DEFAULT_TIMEOUT_SECONDS = 5.0


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


def _env_bool_or_string(name: str, default: bool | str) -> bool | str:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return value.strip()


def _format_result(item: dict[str, Any], index: int) -> str:
    title = item.get("title") or "Untitled"
    url = item.get("url") or ""
    content = item.get("content") or ""
    max_chars = _env_int("TAVILY_SNIPPET_CHARS")
    if max_chars is not None and max_chars > 0 and len(content) > max_chars:
        content = f"{content[:max_chars].rstrip()}..."
    return f"[{index}] {title}\nURL: {url}\nSnippet: {content}".strip()


@traceable(name="tavily_search", run_type="tool", tags=["tavily", "search"])
async def search_web(
    query: str,
    *,
    max_results: int | None = None,
    search_depth: str | None = None,
    include_answer: bool | str | None = None,
    include_raw_content: bool | None = None,
    chunks_per_source: int | None = None,
    timeout_seconds: float | None = None,
) -> str:
    """Search Tavily and return a compact evidence bundle."""

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set.")

    payload = {
        "query": query,
        "max_results": max_results if max_results is not None else _env_int("TAVILY_MAX_RESULTS", DEFAULT_MAX_RESULTS),
        "search_depth": search_depth or os.environ.get("TAVILY_SEARCH_DEPTH", DEFAULT_SEARCH_DEPTH),
        "include_answer": (
            include_answer
            if include_answer is not None
            else _env_bool_or_string("TAVILY_INCLUDE_ANSWER", DEFAULT_INCLUDE_ANSWER)
        ),
        "include_raw_content": (
            include_raw_content
            if include_raw_content is not None
            else _env_bool("TAVILY_INCLUDE_RAW_CONTENT", False)
        ),
        "chunks_per_source": (
            chunks_per_source
            if chunks_per_source is not None
            else _env_int("TAVILY_CHUNKS_PER_SOURCE", DEFAULT_CHUNKS_PER_SOURCE)
        ),
    }
    timeout = timeout_seconds if timeout_seconds is not None else _env_float("TAVILY_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
    except httpx.TimeoutException:
        return "Search timed out before results were available."

    results = data.get("results") or []
    answer = data.get("answer")
    if not results and not answer:
        return "No results found."

    parts = []
    if answer:
        parts.append(f"Search answer:\n{answer}")
    if results:
        parts.append("\n\n".join(_format_result(item, index) for index, item in enumerate(results, start=1)))
    return "\n\n".join(parts)


@tool
async def web_search(query: str) -> str:
    """Search the web for current information and return cited snippets."""

    return await search_web(query)
