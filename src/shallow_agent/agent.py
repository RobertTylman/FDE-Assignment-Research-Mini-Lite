"""Bounded shallow research agent."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Callable

from jinja2 import Environment
from jinja2 import FileSystemLoader
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

from shallow_agent.state import ShallowResearchAgentState

AGENT_DIR = Path(__file__).parent


class ShallowResearcherAgent:
    """Fast tool-augmented research agent with a hard tool-call budget."""

    def __init__(
        self,
        llm: BaseChatModel,
        tools: Sequence[BaseTool],
        *,
        system_prompt: str | Callable[..., str] | None = None,
        max_llm_turns: int = 10,
        max_tool_iterations: int = 5,
    ) -> None:
        self.llm = llm
        self.tools = list(tools)
        self.llm_with_tools = self.llm.bind_tools(self.tools, parallel_tool_calls=True)
        self.max_llm_turns = max_llm_turns
        self.max_tool_iterations = max_tool_iterations
        self.system_prompt = system_prompt or self._load_system_prompt()
        self.tools_info = self._build_tools_info()
        self._graph = self._build_graph()

    def _load_system_prompt(self) -> Callable[..., str]:
        env = Environment(
            loader=FileSystemLoader(AGENT_DIR / "prompts"),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        return env.get_template("researcher.j2").render

    def _render_system_prompt(
        self,
        *,
        tools_info: list[dict[str, Any]],
        user_info: dict[str, Any] | None,
    ) -> str:
        prompt = self.system_prompt
        if callable(prompt):
            return prompt(
                tools=tools_info,
                user_info=user_info,
                current_datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
        return prompt

    def _build_tools_info(self) -> list[dict[str, Any]]:
        return [
            {
                "name": getattr(tool, "name", str(tool)),
                "description": getattr(tool, "description", ""),
            }
            for tool in self.tools
        ]

    def _build_graph(self) -> CompiledStateGraph:
        async def agent_node(state: ShallowResearchAgentState) -> dict[str, Any]:
            tools_info = state.tools_info or self.tools_info
            rendered_system_prompt = self._render_system_prompt(
                tools_info=tools_info,
                user_info=state.user_info,
            )
            system_message = SystemMessage(content=rendered_system_prompt)

            if state.tool_iterations >= self.max_tool_iterations:
                synthesis_anchor = HumanMessage(
                    content=(
                        "You have exhausted your research budget. Synthesize the final answer now. "
                        "Use inline citations like [1] and include a **References:** section. "
                        "Do not make any more tool calls."
                    )
                )
                response = await self.llm.ainvoke([system_message] + state.messages + [synthesis_anchor])
                return {"messages": [response], "tool_iterations": state.tool_iterations}

            response = await self.llm_with_tools.ainvoke([system_message] + state.messages)

            next_iterations = state.tool_iterations
            if getattr(response, "tool_calls", None):
                next_iterations += len(response.tool_calls)

            return {"messages": [response], "tool_iterations": next_iterations}

        builder = StateGraph(ShallowResearchAgentState)
        builder.add_node("agent", agent_node)
        builder.add_node("tools", ToolNode(self.tools))
        builder.set_entry_point("agent")
        builder.add_conditional_edges(
            "agent",
            tools_condition,
            {"tools": "tools", "__end__": "__end__"},
        )
        builder.add_edge("tools", "agent")
        return builder.compile()

    async def run(self, state: ShallowResearchAgentState) -> ShallowResearchAgentState:
        recursion_limit = (self.max_llm_turns * 2) + 10
        result = await self._graph.ainvoke(state, config={"recursion_limit": recursion_limit})
        return ShallowResearchAgentState.model_validate(result)

    @property
    def graph(self) -> CompiledStateGraph:
        return self._graph
