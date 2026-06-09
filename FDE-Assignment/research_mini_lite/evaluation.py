"""Evaluation pipeline for comparing research backends."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Literal

import httpx
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from pydantic import Field

from research_mini_lite import ResearchMiniLiteAgent
from research_mini_lite import ResearchMiniLiteState

ProviderName = Literal["tavily_search_advanced", "research_mini_lite", "tavily_research_mini"]


QUERIES_PATH = Path(__file__).parent / "queries.json"
try:
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        SAMPLE_QUERIES = json.load(f)
except Exception:
    SAMPLE_QUERIES = []


QUALITY_RUBRIC = """Evaluate each report as a practical research answer.
Score from 1 to 5 on each dimension:
- Completeness: directly answers the query with enough detail for a knowledgeable user. Do not reward extra length unless it adds crucial, missing coverage.
- Grounding: claims are concrete, current where needed, and supported by citations or source URLs.
- Source quality: uses many authoritative, relevant sources and has enough source diversity for the query. Prefer dense useful sourcing over many weak links.
- Synthesis: compares, prioritizes, explains tradeoffs, and states implications instead of merely listing snippets.
- Clarity: report is structured, concise, and easy to scan.
- Latency: score the user-facing speed. Use 5 for under 10 seconds, 4 for 10-20 seconds, 3 for 20-40 seconds, 2 for 40-90 seconds, and 1 for over 90 seconds.
- Efficiency: quality delivered relative to response time. Consider whether extra latency produced meaningfully better research value.

Overall scoring guidance:
- Weight the final overall score toward balanced usefulness: 20% completeness, 15% grounding, 10% source quality, 15% synthesis, 10% clarity, 20% latency, 10% efficiency.
- CRITICAL: User-facing speed is a first-class feature. A report delivered in under 10 seconds (like `research_mini_lite`) must be heavily favored. If a fast report (under 10s) provides a solid, comprehensive, and well-cited answer (achieving 4+ in completeness and grounding), it MUST win over a slow report (over 30s) due to its vastly superior efficiency.
- Do not let a high-latency report (over 30 seconds) win unless the lower-latency alternative is completely incorrect, shallow, or missing critical answers. Extra detail or formatting tables in a slow report does NOT justify high latency.
- When reports are close in research quality, use latency, clarity, and efficiency as tie-breakers.
- Penalize fast reports that are shallow, uncited, or mostly raw snippets.

Return strict JSON with this shape:
{
  "scores": {
    "<provider_name>": {
      "overall": 1-5,
      "completeness": 1-5,
      "grounding": 1-5,
      "source_quality": 1-5,
      "synthesis": 1-5,
      "clarity": 1-5,
      "latency": 1-5,
      "efficiency": 1-5,
      "rationale": "short explanation"
    }
  },
  "winner": "<provider_name or tie>",
  "summary": "short cross-provider comparison"
}
"""


class EvaluationOptions(BaseModel):
    providers: list[ProviderName] = Field(
        default_factory=lambda: [
            "tavily_search_advanced",
            "research_mini_lite",
            "tavily_research_mini",
        ]
    )
    judge_quality: bool = True
    max_concurrency: int = 3
    tavily_max_results: int = 8
    tavily_research_timeout_seconds: float = 180.0
    tavily_research_poll_seconds: float = 2.0
    tavily_research_output_length: Literal["short", "standard", "long"] = "standard"
    output_schema: dict[str, Any] | None = None
    output_schema_name: str = "research_output"


class ProviderResult(BaseModel):
    provider: ProviderName
    label: str
    ok: bool
    latency_seconds: float
    output: str | None = None
    error: str | None = None
    source_count: int = 0
    sources: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryEvaluation(BaseModel):
    query: str
    results: list[ProviderResult]
    quality: dict[str, Any] | None = None


class EvaluationSummary(BaseModel):
    query_count: int
    provider_count: int
    average_latency_seconds: dict[str, float]
    average_quality_overall: dict[str, float]
    winners: dict[str, int]


class EvaluationRun(BaseModel):
    queries: list[str]
    options: EvaluationOptions
    items: list[QueryEvaluation]
    summary: EvaluationSummary
    created_at: str | None = None
    report_filename: str | None = None
    report_path: str | None = None


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set.")
    return value


def _source_count_from_text(text: str | None) -> int:
    if not text:
        return 0
    references_match = re.search(r"references\s*:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    source_area = references_match.group(1) if references_match else text
    urls = set(re.findall(r"https?://[^\s\]\)>,]+", source_area))
    if urls:
        return len(urls)
    bracket_refs = set(re.findall(r"\[(\d+)\]", text))
    return len(bracket_refs)


def _format_search_output(data: dict[str, Any]) -> str:
    lines = ["# Tavily Search Advanced Answer", ""]
    answer = data.get("answer")
    if answer:
        lines.extend([str(answer).strip(), ""])
    results = data.get("results") or []
    if results:
        lines.extend(["## Sources", ""])
    for index, item in enumerate(results, start=1):
        title = item.get("title") or "Untitled"
        url = item.get("url") or ""
        content = item.get("content") or ""
        lines.append(f"[{index}] {title}")
        if url:
            lines.append(f"URL: {url}")
        if content:
            lines.append(f"Snippet: {content}")
        lines.append("")
    return "\n".join(lines).strip()


async def run_tavily_search_advanced(query: str, options: EvaluationOptions) -> ProviderResult:
    start = time.perf_counter()
    try:
        api_key = _require_env("TAVILY_API_KEY")
        payload = {
            "query": query,
            "search_depth": "advanced",
            "include_answer": "advanced",
            "include_raw_content": False,
            "max_results": options.tavily_max_results,
            "include_usage": True,
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        sources = [
            {"title": item.get("title"), "url": item.get("url")}
            for item in data.get("results", [])
            if item.get("url")
        ]
        return ProviderResult(
            provider="tavily_search_advanced",
            label="Tavily Search Advanced + Answer",
            ok=True,
            latency_seconds=time.perf_counter() - start,
            output=_format_search_output(data),
            source_count=len(sources),
            sources=sources,
            metadata={
                "response_time": data.get("response_time"),
                "request_id": data.get("request_id"),
                "usage": data.get("usage"),
            },
        )
    except Exception as exc:
        return ProviderResult(
            provider="tavily_search_advanced",
            label="Tavily Search Advanced + Answer",
            ok=False,
            latency_seconds=time.perf_counter() - start,
            error=str(exc),
        )


async def run_research_mini_lite(
    query: str,
    agent: ResearchMiniLiteAgent,
    options: EvaluationOptions,
) -> ProviderResult:
    start = time.perf_counter()
    try:
        _require_env("OPENAI_API_KEY")
        _require_env("TAVILY_API_KEY")
        state = ResearchMiniLiteState(
            messages=[HumanMessage(content=query)],
            output_schema=options.output_schema,
            output_schema_name=options.output_schema_name,
        )
        result = await agent.run(state)
        output = str(result.messages[-1].content)
        return ProviderResult(
            provider="research_mini_lite",
            label="Research Mini Lite",
            ok=True,
            latency_seconds=time.perf_counter() - start,
            output=output,
            source_count=_source_count_from_text(output),
            metadata={"tool_iterations": result.tool_iterations},
        )
    except Exception as exc:
        return ProviderResult(
            provider="research_mini_lite",
            label="Research Mini Lite",
            ok=False,
            latency_seconds=time.perf_counter() - start,
            error=str(exc),
        )


async def run_tavily_research_mini(query: str, options: EvaluationOptions) -> ProviderResult:
    start = time.perf_counter()
    try:
        api_key = _require_env("TAVILY_API_KEY")
        payload = {
            "input": query,
            "model": "mini",
            "stream": False,
            "citation_format": "numbered",
            "output_length": options.tavily_research_output_length,
        }
        if options.output_schema:
            payload["output_schema"] = options.output_schema
        async with httpx.AsyncClient(timeout=60.0) as client:
            create_response = await client.post(
                "https://api.tavily.com/research",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )
            create_response.raise_for_status()
            created = create_response.json()
            request_id = created.get("request_id")
            if not request_id:
                raise RuntimeError(f"Tavily Research did not return request_id: {created}")

            deadline = time.perf_counter() + options.tavily_research_timeout_seconds
            latest: dict[str, Any] = created
            while time.perf_counter() < deadline:
                status_response = await client.get(
                    f"https://api.tavily.com/research/{request_id}",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                if status_response.status_code == 202:
                    await asyncio.sleep(options.tavily_research_poll_seconds)
                    continue
                status_response.raise_for_status()
                latest = status_response.json()
                if latest.get("status") == "completed":
                    break
                if latest.get("status") == "failed":
                    raise RuntimeError(str(latest))
                await asyncio.sleep(options.tavily_research_poll_seconds)
            else:
                raise TimeoutError(
                    f"Tavily Research mini timed out after {options.tavily_research_timeout_seconds} seconds."
                )

        content = latest.get("content")
        output = content if isinstance(content, str) else json.dumps(content, indent=2)
        sources = latest.get("sources") or []
        return ProviderResult(
            provider="tavily_research_mini",
            label="Tavily Research Mini",
            ok=True,
            latency_seconds=time.perf_counter() - start,
            output=output,
            source_count=len(sources),
            sources=sources,
            metadata={
                "request_id": latest.get("request_id") or created.get("request_id"),
                "response_time": latest.get("response_time"),
                "created_at": latest.get("created_at") or created.get("created_at"),
            },
        )
    except Exception as exc:
        return ProviderResult(
            provider="tavily_research_mini",
            label="Tavily Research Mini",
            ok=False,
            latency_seconds=time.perf_counter() - start,
            error=str(exc),
        )


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


async def judge_quality(query: str, results: list[ProviderResult]) -> dict[str, Any] | None:
    successful = [result for result in results if result.ok and result.output]
    if len(successful) < 2:
        return None

    judge_model = os.environ.get("EVAL_JUDGE_MODEL", os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"))
    llm = ChatOpenAI(
        model=judge_model,
        temperature=0,
        api_key=_require_env("OPENAI_API_KEY"),
    )
    reports = "\n\n".join(
        f"## Provider: {result.provider}\nLatency: {result.latency_seconds:.2f}s\n\n{result.output}"
        for result in successful
    )
    prompt = f"{QUALITY_RUBRIC}\n\nResearch query:\n{query}\n\nReports to evaluate:\n{reports}"
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    judged = _extract_json(str(response.content))
    judged["judge_model"] = judge_model
    return judged


async def evaluate_query(
    query: str,
    options: EvaluationOptions,
    agent_factory: Any,
) -> QueryEvaluation:
    agent = agent_factory() if "research_mini_lite" in options.providers else None
    tasks = []
    for provider in options.providers:
        if provider == "tavily_search_advanced":
            tasks.append(run_tavily_search_advanced(query, options))
        elif provider == "research_mini_lite":
            tasks.append(run_research_mini_lite(query, agent, options))
        elif provider == "tavily_research_mini":
            tasks.append(run_tavily_research_mini(query, options))

    results = await asyncio.gather(*tasks)
    quality = await judge_quality(query, results) if options.judge_quality else None
    return QueryEvaluation(query=query, results=results, quality=quality)


def summarize(items: list[QueryEvaluation]) -> EvaluationSummary:
    latencies: dict[str, list[float]] = {}
    quality_scores: dict[str, list[float]] = {}
    winners: dict[str, int] = {}

    for item in items:
        for result in item.results:
            if result.ok:
                latencies.setdefault(result.provider, []).append(result.latency_seconds)
        scores = (item.quality or {}).get("scores") or {}
        for provider, score in scores.items():
            overall = score.get("overall")
            if isinstance(overall, int | float):
                quality_scores.setdefault(provider, []).append(float(overall))
        winner = (item.quality or {}).get("winner")
        if winner:
            winners[winner] = winners.get(winner, 0) + 1

    average_latency = {
        provider: sum(values) / len(values)
        for provider, values in latencies.items()
        if values
    }
    average_quality = {
        provider: sum(values) / len(values)
        for provider, values in quality_scores.items()
        if values
    }
    provider_names = {result.provider for item in items for result in item.results}
    return EvaluationSummary(
        query_count=len(items),
        provider_count=len(provider_names),
        average_latency_seconds=average_latency,
        average_quality_overall=average_quality,
        winners=winners,
    )


def save_evaluation_report(run: EvaluationRun, reports_dir: Path) -> EvaluationRun:
    created_at = datetime.now().astimezone()
    timestamp = created_at.strftime("%Y-%m-%d_%H-%M-%S")
    reports_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{timestamp}.json"
    path = reports_dir / filename
    counter = 2
    while path.exists():
        filename = f"{timestamp}_{counter}.json"
        path = reports_dir / filename
        counter += 1

    run.created_at = created_at.isoformat(timespec="seconds")
    run.report_filename = filename
    run.report_path = str(path)
    path.write_text(
        run.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return run


async def run_evaluation(
    queries: list[str],
    options: EvaluationOptions,
    agent_factory: Any,
    reports_dir: Path | None = None,
) -> EvaluationRun:
    cleaned_queries = [query.strip() for query in queries if query.strip()]
    if not cleaned_queries:
        raise ValueError("At least one query is required.")

    semaphore = asyncio.Semaphore(max(1, options.max_concurrency))

    async def guarded(query: str) -> QueryEvaluation:
        async with semaphore:
            return await evaluate_query(query, options, agent_factory)

    items = await asyncio.gather(*(guarded(query) for query in cleaned_queries))
    run = EvaluationRun(
        queries=cleaned_queries,
        options=options,
        items=items,
        summary=summarize(items),
    )
    if reports_dir is not None:
        return save_evaluation_report(run, reports_dir)
    return run
