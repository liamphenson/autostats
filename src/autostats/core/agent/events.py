from typing import Any, Literal

from pydantic import BaseModel


class TextDelta(BaseModel):
    kind: Literal["text_delta"] = "text_delta"
    text: str


class ToolCallStarted(BaseModel):
    kind: Literal["tool_call_started"] = "tool_call_started"
    tool_name: str
    arguments: dict[str, Any]


class ToolResultReady(BaseModel):
    kind: Literal["tool_result_ready"] = "tool_result_ready"
    tool_name: str
    result: dict[str, Any]


class PlotReady(BaseModel):
    kind: Literal["plot_ready"] = "plot_ready"
    plot_id: str
    path: str


class TurnComplete(BaseModel):
    kind: Literal["turn_complete"] = "turn_complete"
    final_text: str


class ErrorEvent(BaseModel):
    kind: Literal["error"] = "error"
    message: str


AgentEvent = TextDelta | ToolCallStarted | ToolResultReady | PlotReady | TurnComplete | ErrorEvent
