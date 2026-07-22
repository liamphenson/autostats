from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ColumnInfo(BaseModel):
    name: str
    dtype: str


class DatasetMeta(BaseModel):
    """The only representation of a dataset that is allowed into LLM context."""

    dataset_id: str
    n_rows: int
    n_cols: int
    columns: list[ColumnInfo]
    preview: list[dict[str, Any]]
    source: Literal["upload", "fred", "world_bank", "census", "web_scrape"]
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    trust_level: Literal["high", "medium", "low"] = "high"
    validation_warnings: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DatasetHandle(DatasetMeta):
    """Alias kept distinct from DatasetMeta for readability at call sites that
    just registered/loaded a dataset versus ones reading catalog metadata."""
