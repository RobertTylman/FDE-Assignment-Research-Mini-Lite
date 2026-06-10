"""Bounded Research Mini Lite agent."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Callable
from typing import TypedDict

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

from research_mini_lite.observability import traceable
from research_mini_lite.state import ResearchMiniLiteState
from research_mini_lite.tools.web_search import search_web

AGENT_DIR = Path(__file__).parent


class EvidenceSource(TypedDict):
    index: str
    title: str
    url: str
    snippet: str


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
        fast_mode: bool = True,
        target_latency_seconds: float = 9.5,
        fast_search_max_results: int = 12,
    ) -> None:
        self.llm = llm
        self.tools = list(tools)
        self.llm_with_tools = self.llm.bind_tools(self.tools, parallel_tool_calls=True)
        self.max_llm_turns = max_llm_turns
        self.max_tool_iterations = max_tool_iterations
        self.fast_mode = fast_mode
        self.target_latency_seconds = target_latency_seconds
        self.fast_search_max_results = fast_search_max_results
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

    def _extract_user_query(self, state: ResearchMiniLiteState) -> str:
        return next(
            (
                str(message.content)
                for message in state.messages
                if isinstance(message, HumanMessage)
            ),
            "",
        )

    def _build_fast_search_query(self, query: str) -> str:
        lower_query = query.lower()
        if "history" in lower_query:
            return f"{query} founding timeline major milestones leadership products authoritative sources"
        broad_terms = ("compare", "analyze", "overview", "market", "timeline", "latest")
        if any(term in lower_query for term in broad_terms):
            return (
                f"{query} overview key milestones current status "
                "authoritative sources analysis implications"
            )
        return f"{query} authoritative sources concise analysis key facts"

    def _split_evidence(self, evidence: str) -> tuple[str, str]:
        stripped = evidence.strip()
        if not stripped.startswith("Search answer:"):
            return "", stripped

        _, _, rest = stripped.partition("Search answer:")
        answer, _, sources = rest.strip().partition("\n\n[1]")
        if sources:
            sources = f"[1]{sources}"
        else:
            sources = rest.strip()
        return answer.strip(), sources.strip()

    def _parse_sources(self, sources: str) -> list[EvidenceSource]:
        parsed = []
        blocks = re.split(r"\n\n(?=\[\d+\] )", sources.strip())
        for block in blocks:
            match = re.match(r"\[(\d+)\]\s*(.*)", block.strip())
            if not match:
                continue
            lines = block.strip().splitlines()
            url = ""
            snippet_lines = []
            for line in lines[1:]:
                if line.startswith("URL:"):
                    url = line.removeprefix("URL:").strip()
                elif line.startswith("Snippet:"):
                    snippet_lines.append(line.removeprefix("Snippet:").strip())
                elif snippet_lines:
                    snippet_lines.append(line.strip())
            parsed.append(
                {
                    "index": match.group(1),
                    "title": match.group(2).strip(),
                    "url": url,
                    "snippet": " ".join(part for part in snippet_lines if part).strip(),
                }
            )
        return parsed

    def _compact_sources_for_synthesis(self, sources: str, *, snippet_chars: int = 180) -> str:
        compact_blocks = []
        for source in self._parse_sources(sources):
            title = f"[{source['index']}] {source['title']}"
            url = f"URL: {source['url']}" if source["url"] else ""
            snippet = source["snippet"]
            if len(snippet) > snippet_chars:
                snippet = f"{snippet[:snippet_chars].rstrip()}..."
            compact = "\n".join(part for part in (title, url, f"Snippet: {snippet}" if snippet else "") if part)
            if compact:
                compact_blocks.append(compact)
        return "\n\n".join(compact_blocks)

    def _build_synthesis_evidence(self, evidence: str) -> str:
        answer, sources = self._split_evidence(evidence)
        compact_sources = self._compact_sources_for_synthesis(sources)
        parts = []
        if answer:
            parts.append(f"Search answer:\n{answer}")
        if compact_sources:
            parts.append(f"Source notes:\n{compact_sources}")
        return "\n\n".join(parts) or evidence[:6000]

    def _references_from_evidence(self, evidence: str) -> str:
        _, sources = self._split_evidence(evidence)
        references = []
        for source in self._parse_sources(sources):
            if source["url"]:
                references.append(f"[{source['index']}] {source['title']}\nURL: {source['url']}")
        return "\n\n".join(references).strip()

    def _fallback_report(self, query: str, evidence: str, reason: str) -> str:
        answer, _ = self._split_evidence(evidence[:14000])
        references = self._references_from_evidence(evidence)
        return (
            "## Executive Summary\n\n"
            f"{answer or 'Research Mini Lite returned the fastest available evidence inside the latency budget.'}\n\n"
            "## Coverage Note\n\n"
            "This report used the fastest available synthesized search answer because the full synthesis step did not complete inside the latency budget.\n\n"
            f"## Query\n\n{query}\n\n"
            f"## References\n\n{references}"
        )

    async def _search_once(self, query: str) -> str:
        return await search_web(
            self._build_fast_search_query(query),
            max_results=max(1, self.fast_search_max_results),
            search_depth="advanced",
            include_answer="advanced",
            include_raw_content=False,
            chunks_per_source=1,
            timeout_seconds=6.75,
        )

    async def _synthesize_fast_report(
        self,
        *,
        query: str,
        evidence: str,
        state: ResearchMiniLiteState,
        remaining_seconds: float,
    ) -> str:
        if remaining_seconds < 5.0:
            raise TimeoutError("Not enough remaining latency budget for LLM synthesis.")

        schema_instruction = ""
        response_format = None
        if state.output_schema:
            schema_instruction = (
                "Return JSON only. The JSON must conform to this schema:\n"
                f"{json.dumps(state.output_schema, indent=2)}\n"
            )
            response_format = self._schema_response_format(state.output_schema, state.output_schema_name)

        prompt = (
            "You are Research Mini Lite. Write a concise synthesized research report within a strict latency budget.\n"
            "Use only the provided evidence. Prioritize concrete facts, tradeoffs, next steps, and implications.\n"
            "If returning markdown, use 4 short sections: Executive Summary, Key Findings, Analysis or Tradeoffs, "
            "and What To Track or Current State. Keep the report between 250 and 450 words before references. "
            "Use compact paragraphs or bullets; do not copy source snippets verbatim. "
            "Cite source numbers inline and list URLs in References. Renumber references sequentially.\n"
            f"{schema_instruction}\n"
            f"Research query:\n{query}\n\n"
            f"Evidence:\n{self._build_synthesis_evidence(evidence)}"
        )
        llm_kwargs: dict[str, Any] = {"max_tokens": 700}
        if response_format:
            llm_kwargs["response_format"] = response_format
        llm = self.llm.bind(**llm_kwargs)
        llm = llm.with_config(
            {
                "run_name": "research_mini_lite_synthesis_llm",
                "tags": ["research-mini-lite", "synthesis"],
                "metadata": {
                    "has_output_schema": bool(state.output_schema),
                    "target_latency_seconds": self.target_latency_seconds,
                },
            }
        )
        response = await asyncio.wait_for(
            llm.ainvoke([HumanMessage(content=prompt)]),
            timeout=max(0.5, remaining_seconds),
        )
        content = response.content
        if not isinstance(content, str):
            content = json.dumps(content)
        if state.output_schema:
            json.loads(content)
        else:
            references = self._references_from_evidence(evidence)
            if references and "## References" not in content and "## References and Evidence" not in content:
                content = f"{content.rstrip()}\n\n## References\n\n{references}"
            elif references and "## Additional Sources" not in content:
                content = f"{content.rstrip()}\n\n## Additional Sources\n\n{references}"
        return content

    @traceable(name="research_mini_lite_fast_path", run_type="chain", tags=["research-mini-lite", "fast-path"])
    async def _run_fast(self, state: ResearchMiniLiteState) -> ResearchMiniLiteState:
        query = self._extract_user_query(state)
        started_at = time.perf_counter()
        evidence = ""
        metadata: dict[str, Any] = {
            "fast_mode": True,
            "target_latency_seconds": self.target_latency_seconds,
            "fast_search_max_results": self.fast_search_max_results,
            "fallback_used": False,
        }
        try:
            search_budget = min(6.75, max(1.0, self.target_latency_seconds - 1.5))
            search_started_at = time.perf_counter()
            evidence = await asyncio.wait_for(self._search_once(query), timeout=search_budget)
            metadata["search_seconds"] = time.perf_counter() - search_started_at
            remaining = self.target_latency_seconds - (time.perf_counter() - started_at) - 0.25
            metadata["synthesis_budget_seconds"] = max(0.0, remaining)
            synthesis_started_at = time.perf_counter()
            output = await self._synthesize_fast_report(
                query=query,
                evidence=evidence,
                state=state,
                remaining_seconds=remaining,
            )
            metadata["synthesis_seconds"] = time.perf_counter() - synthesis_started_at
        except (asyncio.TimeoutError, TimeoutError):
            metadata["fallback_used"] = True
            metadata["fallback_reason"] = "latency_budget_reached"
            output = self._fallback_report(
                query=query,
                evidence=evidence or "Search did not complete inside the latency budget.",
                reason="the latency budget was reached",
            )
        except Exception as exc:
            metadata["fallback_used"] = True
            metadata["fallback_reason"] = type(exc).__name__
            output = self._fallback_report(
                query=query,
                evidence=evidence or f"Search failed: {exc}",
                reason="an upstream call failed before a full synthesis could complete",
            )

        state.messages.append(AIMessage(content=output))
        state.tool_iterations = 1 if evidence else 0
        metadata["total_seconds"] = time.perf_counter() - started_at
        metadata["evidence_chars"] = len(evidence)
        metadata["output_chars"] = len(output)
        state.metadata.update(metadata)
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

    @traceable(name="research_mini_lite_agent", run_type="chain", tags=["research-mini-lite"])
    async def run(self, state: ResearchMiniLiteState) -> ResearchMiniLiteState:
        if self.fast_mode:
            return await self._run_fast(state)

        recursion_limit = (self.max_llm_turns * 2) + 10
        result = await self._graph.ainvoke(state, config={"recursion_limit": recursion_limit})
        validated = ResearchMiniLiteState.model_validate(result)
        validated.output_schema = validated.output_schema or state.output_schema
        validated.output_schema_name = state.output_schema_name
        return await self._format_with_output_schema(validated)

    @property
    def graph(self) -> CompiledStateGraph:
        return self._graph
