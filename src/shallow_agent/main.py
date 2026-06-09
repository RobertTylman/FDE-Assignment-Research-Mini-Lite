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


async def answer(query: str) -> str:
    load_dotenv()
    model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
    max_tool_iterations = int(os.environ.get("MAX_TOOL_ITERATIONS", "5"))

    llm = ChatOpenAI(model=model, temperature=0.1)
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
