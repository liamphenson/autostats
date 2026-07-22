from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from autostats.core.schemas.stat_result import StatResult


class Message(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None
    tool_name: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AnalysisResult(BaseModel):
    result: StatResult
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
