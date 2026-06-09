"""Bounded Research Mini Lite agent."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Callable

from jinja2 import Environment
from jinja2 import FileSystemLoader
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

from research_mini_lite.state import ResearchMiniLiteState

AGENT_DIR = Path(__file__).parent


class ResearchMiniLiteAgent:
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

    def _schema_response_format(self, schema: dict[str, Any], schema_name: str) -> dict[str, Any]:
        safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", schema_name).strip("_") or "research_output"
        return {
            "type": "json_schema",
            "json_schema": {
                "name": safe_name[:64],
                "schema": schema,
                "strict": False,
            },
        }

    async def _format_with_output_schema(self, state: ResearchMiniLiteState) -> ResearchMiniLiteState:
        if not state.output_schema:
            return state

        final_content = str(state.messages[-1].content)
        original_query = next(
            (
                str(message.content)
                for message in state.messages
                if isinstance(message, HumanMessage)
            ),
            "",
        )
        prompt = (
            "Convert the research report into JSON that conforms to the provided JSON Schema. "
            "Use only information supported by the report. Preserve citations or source URLs in fields "
            "where the schema provides a place for them. Return JSON only.\n\n"
            f"Original query:\n{original_query}\n\n"
            f"JSON Schema:\n{json.dumps(state.output_schema, indent=2)}\n\n"
            f"Research report:\n{final_content}"
        )
        structured_llm = self.llm.bind(
            response_format=self._schema_response_format(
                state.output_schema,
                state.output_schema_name,
            )
        )
        response = await structured_llm.ainvoke([HumanMessage(content=prompt)])
        content = response.content
        if not isinstance(content, str):
            content = json.dumps(content)
        json.loads(content)
        state.messages[-1] = AIMessage(content=content)
        return state

    def _build_graph(self) -> CompiledStateGraph:
        async def agent_node(state: ResearchMiniLiteState) -> dict[str, Any]:
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

        builder = StateGraph(ResearchMiniLiteState)
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

    async def run(self, state: ResearchMiniLiteState) -> ResearchMiniLiteState:
        recursion_limit = (self.max_llm_turns * 2) + 10
        result = await self._graph.ainvoke(state, config={"recursion_limit": recursion_limit})
        validated = ResearchMiniLiteState.model_validate(result)
        validated.output_schema = validated.output_schema or state.output_schema
        validated.output_schema_name = state.output_schema_name
        return await self._format_with_output_schema(validated)

    @property
    def graph(self) -> CompiledStateGraph:
        return self._graph
