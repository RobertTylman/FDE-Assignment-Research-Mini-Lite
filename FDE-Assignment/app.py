import json
import os
import random
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from pydantic import Field

from research_mini_lite import ResearchMiniLiteAgent
from research_mini_lite import ResearchMiniLiteState
from research_mini_lite.evaluation import EvaluationOptions
from research_mini_lite.evaluation import SAMPLE_QUERIES
from research_mini_lite.evaluation import run_evaluation
from research_mini_lite.observability import configure_langsmith
from research_mini_lite.observability import langsmith_enabled
from research_mini_lite.observability import traceable
from research_mini_lite.tools import web_search

APP_DIR = Path(__file__).parent
REPO_ROOT = APP_DIR.parent
EVAL_REPORTS_DIR = REPO_ROOT / "eval-reports"


def load_environment() -> None:
    """Load non-empty env values from repo root, then FDE-Assignment."""

    for env_path in (REPO_ROOT / ".env", APP_DIR / ".env"):
        if not env_path.exists():
            continue
        for key, value in dotenv_values(env_path).items():
            if value:
                os.environ[key] = value


load_environment()
configure_langsmith()


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


def build_llm() -> ChatOpenAI:
    llm_kwargs = {
        "model": os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
        "temperature": _env_float("OPENAI_TEMPERATURE", 0.1),
        "api_key": os.environ.get("OPENAI_API_KEY"),
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

    return ChatOpenAI(**llm_kwargs)


def build_agent() -> ResearchMiniLiteAgent:
    return ResearchMiniLiteAgent(
        llm=build_llm(),
        tools=[web_search],
        max_tool_iterations=_env_int("MAX_TOOL_ITERATIONS", 5),
    )


app = FastAPI(title="Web Agent API")


class QueryRequest(BaseModel):
    query: str
    output_schema: dict[str, Any] | None = None
    output_schema_name: str = "research_output"


class EvaluationRequest(BaseModel):
    queries: list[str]
    options: EvaluationOptions = Field(default_factory=EvaluationOptions)


@app.get("/", response_class=HTMLResponse)
async def evaluation_ui():
    return (APP_DIR / "static" / "index.html").read_text()


@app.get("/chat", response_class=HTMLResponse)
async def chat_ui():
    return (APP_DIR / "static" / "chat.html").read_text()


@app.get("/background.webp")
async def background_image():
    return FileResponse(APP_DIR / "static" / "background.webp", media_type="image/webp")


@app.get("/eval/sample-queries")
async def sample_queries(response: Response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return {"queries": SAMPLE_QUERIES}


@app.get("/langsmith/status")
async def langsmith_status():
    return {
        "enabled": langsmith_enabled(),
        "project": os.environ.get("LANGSMITH_PROJECT") or os.environ.get("LANGCHAIN_PROJECT"),
        "endpoint": os.environ.get("LANGSMITH_ENDPOINT") or "https://api.smith.langchain.com",
    }


@app.post("/eval/run")
@traceable(name="evaluation_api", run_type="chain", tags=["api", "evaluation"])
async def run_eval(request: EvaluationRequest):
    """
    Compare Tavily Search Advanced, Research Mini Lite, and Tavily Research Mini.
    """
    try:
        return await run_evaluation(
            request.queries,
            request.options,
            build_agent,
            reports_dir=EVAL_REPORTS_DIR,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run")
@traceable(name="research_mini_lite_api", run_type="chain", tags=["api", "research-mini-lite"])
async def run_agent(request: QueryRequest):
    """
    Execute the bounded Research Mini Lite agent on a query string.
    """
    try:
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")

        if not tavily_api_key:
            raise HTTPException(status_code=500, detail="TAVILY_API_KEY not set")
        if not openai_api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")

        state = ResearchMiniLiteState(
            messages=[HumanMessage(content=request.query)],
            output_schema=request.output_schema,
            output_schema_name=request.output_schema_name,
        )
        result = await build_agent().run(state)
        output = result.messages[-1].content
        response = {"query": request.query, "output": output}
        if request.output_schema:
            response["output_json"] = json.loads(str(output))

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
