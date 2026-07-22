from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from autostats.core.data.manager import DataManager


@dataclass
class ToolContext:
    """Everything a tool's `run()` needs, besides its own params.

    Deliberately does NOT give tools access to the LLM or conversation
    history -- tools are pure functions over data, never prompt-aware.
    """

    session_id: str
    data_manager: "DataManager"
    plots_dir: str


class ToolInput(BaseModel):
    """Base class for every tool's input schema. `dataset_id` is included
    (rather than hidden) so the model must reference a concrete, already-
    loaded dataset instead of inventing data."""

    dataset_id: str


class BaseTool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    category: ClassVar[str]
    input_model: ClassVar[type[ToolInput]]

    @abstractmethod
    def run(self, ctx: ToolContext, params: ToolInput) -> BaseModel:
        """Execute the tool and return a pydantic result model
        (StatResult for stat tools, DatasetHandle for data tools)."""

    @classmethod
    def openai_schema(cls) -> dict:
        """Responses API tool schema (flat -- distinct from Chat Completions'
        nested {"function": {...}} shape)."""
        schema = cls.input_model.model_json_schema()
        schema.pop("title", None)
        return {
            "type": "function",
            "name": cls.name,
            "description": cls.description,
            "parameters": schema,
        }
