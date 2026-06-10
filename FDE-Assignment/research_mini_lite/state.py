"""State model for the Research Mini Lite graph."""

from typing import Annotated
from typing import Any

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from pydantic import Field


class ResearchMiniLiteState(BaseModel):
    """Conversation state passed between agent and tool nodes."""

    messages: Annotated[list[AnyMessage], add_messages]
    user_info: dict[str, Any] | None = None
    tools_info: list[dict[str, Any]] | None = None
    tool_iterations: int = 0
    output_schema: dict[str, Any] | None = None
    output_schema_name: str = "research_output"
    metadata: dict[str, Any] = Field(default_factory=dict)
