# FDE Assignment: Research Mini Lite

This project implements a FastAPI research agent backed by a bounded LangGraph loop:

```text
agent -> tools -> agent -> ... -> final answer
```

The implementation improves the starter Tavily agent by enforcing a hard tool-call budget, using a custom citation-focused system prompt, and exposing latency/quality controls through environment variables. When the research budget is exhausted, the agent is forced to synthesize a final answer instead of continuing to call tools.

## Project Structure

```text
FDE-Assignment/
  app.py
  requirements.txt
  research_mini_lite/
    agent.py
    state.py
    tools/web_search.py
    prompts/researcher.j2
```

There is no separate top-level `src` package. The API and agent live together in `FDE-Assignment`.

## Setup

```bash
cd FDE-Assignment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in `FDE-Assignment` or the repository root with:

```text
TAVILY_API_KEY=your_tavily_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

Optional controls:

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

## Running the Application

```bash
python app.py
```

The API will start on `http://localhost:8000`

## Usage

Send a POST request to `/run` with a query:

```bash
curl -X POST "http://localhost:8000/run" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the latest developments in AI?"}'
```
