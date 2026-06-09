# Shallow Agent Test

Standalone testbed for the AI-Q shallow researcher architecture.

It implements a bounded LangGraph loop:

```text
agent -> tools -> agent -> ... -> final answer
```

The agent uses one chat model, a small tool list, and a tool-call budget. When the budget is exhausted, it forces final synthesis instead of allowing more tool calls.

## Setup

```bash
cd /Users/robbietylman/GitHub/shallow-agent-test
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Edit `.env` and set:

```text
OPENAI_API_KEY=...
TAVILY_API_KEY=...
```

Optional latency and quality knobs are available in `.env`:

```text
OPENAI_MODEL=gpt-4.1-mini
OPENAI_TEMPERATURE=0.1
# OPENAI_TIMEOUT_SECONDS=30
# OPENAI_MAX_RETRIES=2
# OPENAI_MAX_TOKENS=700
MAX_TOOL_ITERATIONS=5
TAVILY_TIMEOUT_SECONDS=30
TAVILY_MAX_RESULTS=5
TAVILY_SEARCH_DEPTH=basic
TAVILY_INCLUDE_ANSWER=false
TAVILY_INCLUDE_RAW_CONTENT=false
# TAVILY_SNIPPET_CHARS=700
```

For lower p95 latency, the biggest lever is `MAX_TOOL_ITERATIONS`. Each extra
search iteration adds another model round trip plus a web request before final
synthesis. `TAVILY_MAX_RESULTS`, `TAVILY_SNIPPET_CHARS`, `OPENAI_TIMEOUT_SECONDS`,
and `OPENAI_MAX_TOKENS` are the next most useful controls.

## Run

```bash
shallow-agent "What is CUDA?"
```

Or:

```bash
python -m shallow_agent.main "What is CUDA?"
```

## Files

- `src/shallow_agent/state.py` defines the LangGraph state.
- `src/shallow_agent/agent.py` implements the bounded agent/tool loop.
- `src/shallow_agent/tools/web_search.py` defines a Tavily-backed LangChain tool.
- `src/shallow_agent/prompts/researcher.j2` is the system prompt.
- `src/shallow_agent/main.py` wires the model, tools, and CLI together.
