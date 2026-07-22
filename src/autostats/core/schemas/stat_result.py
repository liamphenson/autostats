from typing import Any

from pydantic import BaseModel, Field


class AssumptionCheck(BaseModel):
    name: str
    passed: bool
    statistic: float | None = None
    p_value: float | None = None
    detail: str


class TableArtifact(BaseModel):
    table_id: str
    title: str
    columns: list[str]
    rows: list[list[Any]]


class PlotArtifact(BaseModel):
    plot_id: str
    title: str
    path: str


class StatResult(BaseModel):
    """The single contract every statistical tool returns.

    `interpretation` is templated (deterministic), never LLM-generated, so
    narrated conclusions are guaranteed consistent with the computed numbers.
    """

    tool_name: str
    test_name: str
    statistic: float | None = None
    p_value: float | None = None
    degrees_of_freedom: float | tuple[float, float] | None = None
    effect_size: dict[str, float] | None = None
    confidence_interval: tuple[float, float] | None = None
    confidence_level: float = 0.95
    sample_sizes: dict[str, int] = Field(default_factory=dict)
    assumptions: list[AssumptionCheck] = Field(default_factory=list)
    assumptions_met: bool = True
    recommended_alternative: str | None = None
    interpretation: str
    warnings: list[str] = Field(default_factory=list)
    tables: list[TableArtifact] = Field(default_factory=list)
    plots: list[PlotArtifact] = Field(default_factory=list)
    raw_summary: dict[str, Any] = Field(default_factory=dict)
