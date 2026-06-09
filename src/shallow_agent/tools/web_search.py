"""Tavily-backed web search tool."""

from __future__ import annotations

import os
from typing import Any

import httpx
from langchain_core.tools import tool


def _format_result(item: dict[str, Any], index: int) -> str:
    title = item.get("title") or "Untitled"
    url = item.get("url") or ""
    content = item.get("content") or ""
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
        "max_results": 5,
        "search_depth": "basic",
        "include_answer": False,
        "include_raw_content": False,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post("https://api.tavily.com/search", json=payload)
        response.raise_for_status()
        data = response.json()

    results = data.get("results") or []
    if not results:
        return "No results found."

    return "\n\n".join(_format_result(item, index) for index, item in enumerate(results, start=1))
