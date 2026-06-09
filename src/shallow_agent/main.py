"""CLI entry point for the standalone shallow agent."""

from __future__ import annotations

import argparse
import asyncio
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from shallow_agent.agent import ShallowResearcherAgent
from shallow_agent.state import ShallowResearchAgentState
from shallow_agent.tools import web_search


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


async def answer(query: str) -> str:
    load_dotenv()
    model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
    max_tool_iterations = _env_int("MAX_TOOL_ITERATIONS", 5)

    llm_kwargs = {
        "model": model,
        "temperature": _env_float("OPENAI_TEMPERATURE", 0.1),
    }
    openai_timeout = _env_float("OPENAI_TIMEOUT_SECONDS")
    max_retries = _env_int("OPENAI_MAX_RETRIES")
    max_tokens = _env_int("OPENAI_MAX_TOKENS")

    if openai_timeout is not None:
        llm_kwargs["timeout"] = openai_timeout
    if max_retries is not None:
        llm_kwargs["max_retries"] = max_retries
    if max_tokens is not None:
        llm_kwargs["max_tokens"] = max_tokens

    llm = ChatOpenAI(**llm_kwargs)
    agent = ShallowResearcherAgent(
        llm=llm,
        tools=[web_search],
        max_tool_iterations=max_tool_iterations,
    )
    state = ShallowResearchAgentState(messages=[HumanMessage(content=query)])
    result = await agent.run(state)
    return str(result.messages[-1].content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the standalone shallow research agent.")
    parser.add_argument("query", help="Research question to answer.")
    args = parser.parse_args()
    print(asyncio.run(answer(args.query)))


if __name__ == "__main__":
    main()
