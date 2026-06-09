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
