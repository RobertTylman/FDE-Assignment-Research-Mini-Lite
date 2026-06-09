import os
from pathlib import Path

from dotenv import dotenv_values
from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from research_mini_lite import ResearchMiniLiteAgent
from research_mini_lite import ResearchMiniLiteState
from research_mini_lite.tools import web_search

APP_DIR = Path(__file__).parent
REPO_ROOT = APP_DIR.parent


def load_environment() -> None:
    """Load non-empty env values from repo root, then FDE-Assignment."""

    for env_path in (REPO_ROOT / ".env", APP_DIR / ".env"):
        if not env_path.exists():
            continue
        for key, value in dotenv_values(env_path).items():
            if value:
                os.environ[key] = value


load_environment()


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


@app.post("/run")
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

        state = ResearchMiniLiteState(messages=[HumanMessage(content=request.query)])
        result = await build_agent().run(state)
        output = result.messages[-1].content

        return {"query": request.query, "output": output}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
