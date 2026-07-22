"""The agent core every interface (CLI, API, notebook, web) calls through.

Deliberately synchronous/generator-based rather than the SDK's server-side
conversation chaining: `SessionStore` owns canonical history so the exact
same call works for a stateless API request and a long-lived chat session.
"""

import json
from collections.abc import Iterator

from openai import OpenAI

from autostats.core.agent.events import (
    AgentEvent,
    ErrorEvent,
    PlotReady,
    TextDelta,
    ToolCallStarted,
    ToolResultReady,
    TurnComplete,
)
from autostats.core.agent.prompts import build_system_prompt
from autostats.core.config import Settings, get_settings
from autostats.core.data.manager import DataManager
from autostats.core.schemas.stat_result import StatResult
from autostats.core.session.models import AnalysisResult, Message
from autostats.core.session.store import SessionStore
from autostats.core.tools import REGISTRY, load_all_tools
from autostats.core.tools.base import ToolContext

load_all_tools()

MAX_TOOL_ITERATIONS = 8


class AutoStatsAgent:
    def __init__(self, session_store: SessionStore, settings: Settings | None = None):
        self.session_store = session_store
        self.settings = settings or get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self._data_managers: dict[str, DataManager] = {}

    def data_manager(self, session_id: str) -> DataManager:
        if session_id not in self._data_managers:
            storage_dir = self.settings.sessions_dir / session_id / "datasets"
            self._data_managers[session_id] = DataManager(session_id, storage_dir)
        return self._data_managers[session_id]

    def run_turn(self, session_id: str, user_message: str) -> Iterator[AgentEvent]:
        data_manager = self.data_manager(session_id)
        plots_dir = str(self.settings.sessions_dir / session_id / "plots")
        ctx = ToolContext(session_id=session_id, data_manager=data_manager, plots_dir=plots_dir)

        state = self.session_store.get(session_id)
        input_items: list[dict] = [
            {"role": "system", "content": build_system_prompt(data_manager.catalog_text())}
        ]
        for m in state.messages:
            role = "assistant" if m.role == "tool" else m.role
            input_items.append({"role": role, "content": m.content})
        input_items.append({"role": "user", "content": user_message})

        self.session_store.append_message(session_id, Message(role="user", content=user_message))

        tools = REGISTRY.to_openai_tools()
        final_text = ""

        try:
            for _ in range(MAX_TOOL_ITERATIONS):
                response = self.client.responses.create(
                    model=self.settings.openai_model,
                    input=input_items,
                    tools=tools,
                )

                function_calls = [item for item in response.output if item.type == "function_call"]
                message_items = [item for item in response.output if item.type == "message"]

                for item in message_items:
                    text = "".join(
                        c.text for c in item.content if getattr(c, "type", None) == "output_text"
                    )
                    if text:
                        final_text += text
                        yield TextDelta(text=text)

                if not function_calls:
                    break

                input_items.extend(response.output)

                for call in function_calls:
                    arguments = json.loads(call.arguments)
                    yield ToolCallStarted(tool_name=call.name, arguments=arguments)
                    try:
                        result = REGISTRY.dispatch(ctx, call.name, arguments)
                    except Exception as exc:  # noqa: BLE001 -- surfaced to the model, not raised
                        output = {"error": str(exc)}
                    else:
                        output = result.model_dump(mode="json")
                        if isinstance(result, StatResult):
                            self.session_store.add_result(session_id, AnalysisResult(result=result))
                            for plot in result.plots:
                                yield PlotReady(plot_id=plot.plot_id, path=plot.path)
                    yield ToolResultReady(tool_name=call.name, result=output)
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": json.dumps(output),
                        }
                    )
            else:
                final_text += "\n(Reached the maximum number of tool-call iterations for this turn.)"

        except Exception as exc:  # noqa: BLE001
            yield ErrorEvent(message=str(exc))
            return

        self.session_store.append_message(session_id, Message(role="assistant", content=final_text))
        yield TurnComplete(final_text=final_text)
