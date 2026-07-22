from pydantic import BaseModel

from autostats.core.tools.base import BaseTool, ToolContext


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool_cls: type[BaseTool]) -> type[BaseTool]:
        instance = tool_cls()
        if instance.name in self._tools:
            raise ValueError(f"Tool '{instance.name}' is already registered")
        self._tools[instance.name] = instance
        return tool_cls

    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def all_tools(self, categories: set[str] | None = None) -> list[BaseTool]:
        tools = list(self._tools.values())
        if categories is not None:
            tools = [t for t in tools if t.category in categories]
        return tools

    def to_openai_tools(self, categories: set[str] | None = None) -> list[dict]:
        return [t.openai_schema() for t in self.all_tools(categories)]

    def dispatch(self, ctx: ToolContext, name: str, arguments: dict) -> BaseModel:
        tool = self.get(name)
        params = tool.input_model.model_validate(arguments)
        return tool.run(ctx, params)


REGISTRY = ToolRegistry()
