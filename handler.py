"""Path-B adapter: exposes the AutoStats statistics agent through the A2A
`AgentHandler` contract.

Q1 (entry point):   AutoStatsAgent.run_turn(session_id, message) -> Iterator[AgentEvent]
                     -- the "one conversational turn" call. A single
                     AutoStatsAgent + SessionStore is built once, in __init__,
                     so its DataManager (dataset catalog) and message history
                     survive across calls for the life of this process --
                     exactly like autostats' own CLI `chat` session.

Q2 (input mapping): `user_input` becomes the turn's user message text.
                     Attached files are written to real paths (kept open for
                     the duration of the call, since FileInput.as_tempfile()
                     cleans up on exit) and listed in the message so the model
                     can call its own `upload_dataset(file_path=...)` tool --
                     mirrors the CLI's `autostats upload` command.

Q3 (the answer):    run_turn yields AgentEvents; TurnComplete.final_text is the
                     human-readable reply. Tool calls and any generated plots
                     (base64-encoded -- a local file path means nothing to a
                     remote caller) ride along as structured output.

Q4 (blocking):      run_turn makes blocking OpenAI network calls, so the whole
                     drain loop runs under asyncio.to_thread.

Q5 (credentials):   declares `context`; an injected `openai_api_key` swaps the
                     shared agent's OpenAI client for the call's duration (see
                     `_with_api_key` for the concurrency caveat).

Q6 (dependencies):  the `agent_skeleton` package must be installed/importable
                     alongside this file -- it is deliberately NOT a declared
                     dependency of autostats' own pyproject.toml (the real
                     deployment platform injects its own `agent_skeleton` at
                     serve time). For local dev/testing, install it explicitly;
                     see A2A_INTEGRATION.md for the exact command.

Known limitation -- session continuity: the A2A HandlerExecutor does not pass
a per-conversation id into handle_structured (only `credentials`/`user_id`),
so there is no way to key one autostats session per A2A task/context. This
adapter keys sessions by `user_id` instead (falling back to a fixed id when
absent), which gives per-user continuity -- multiple unrelated conversations
from the same user will share one autostats session/dataset catalog. Revisit
if handler_executor.py ever grows a context_id/task_id passthrough.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import mimetypes
from pathlib import Path
from typing import Any

from agent_skeleton import AgentHandler, FileInput

from autostats.core.agent.events import (
    ErrorEvent,
    PlotReady,
    ToolCallStarted,
    ToolResultReady,
    TurnComplete,
)
from autostats.core.agent.loop import AutoStatsAgent
from autostats.core.config import get_settings
from autostats.core.session.store import SessionStore


class AutoStatsHandler(AgentHandler):
    """Wraps the existing AutoStatsAgent tool loop as an A2A agent."""

    def __init__(self, config: dict):
        super().__init__(config)
        settings = get_settings()
        store = SessionStore(settings.data_dir / "sessions.db")
        self._agent = AutoStatsAgent(store, settings)

    async def handle_structured(
        self,
        user_input: str,
        files: list[FileInput] = [],
        context: dict | None = None,
    ) -> dict:
        session_id = (context or {}).get("user_id") or "local-session"
        api_key = (((context or {}).get("credentials") or {}).get("openai_api_key") or {}).get(
            "api_key"
        )

        with contextlib.ExitStack() as stack:
            message = _build_message(user_input, files, stack)
            with _with_api_key(self._agent, api_key):
                answer, tool_log, plots, error = await asyncio.to_thread(
                    _drain_turn, self._agent, session_id, message
                )

        result: dict[str, Any] = {"answer": answer, "session_id": session_id, "tools_used": tool_log}
        if plots:
            result["plots"] = plots
        if error:
            result["error"] = error
        return result

# --- Necessary Helper Routines ------------------------------------------------------------

def _build_message(user_input: str, files: list[FileInput], stack: contextlib.ExitStack) -> str:
    parts: list[str] = []
    if files:
        lines = []
        for f in files:
            path = stack.enter_context(f.as_tempfile())
            lines.append(f"- {f.name or path.name}: file_path={path}")
        parts.append(
            "The user attached the following file(s). Call upload_dataset with "
            "file_path set to one of these paths to load it:\n" + "\n".join(lines)
        )
    text = (user_input or "").strip()
    if text:
        parts.append(text)
    elif not files:
        parts.append("Please describe what statistical analysis I can help with.")
    return "\n\n".join(parts)

def _drain_turn(
    agent: AutoStatsAgent, session_id: str, message: str
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], str | None]:
    tool_log: list[dict[str, Any]] = []
    plots: list[dict[str, Any]] = []
    answer = ""
    error: str | None = None
    current_call: dict[str, Any] | None = None

    for event in agent.run_turn(session_id, message):
        if isinstance(event, ToolCallStarted):
            current_call = {"name": event.tool_name, "arguments": event.arguments}
        elif isinstance(event, ToolResultReady):
            if current_call is not None:
                current_call["result"] = event.result
                tool_log.append(current_call)
                current_call = None
        elif isinstance(event, PlotReady):
            plots.append(_encode_plot(event.plot_id, event.path))
        elif isinstance(event, TurnComplete):
            answer = event.final_text
        elif isinstance(event, ErrorEvent):
            error = event.message

    if not answer:
        answer = f"AutoStats hit an error: {error}" if error else "(no response produced)"
    return answer, tool_log, plots, error

def _encode_plot(plot_id: str, path: str) -> dict[str, Any]:
    p = Path(path)
    try:
        data = p.read_bytes()
    except OSError as exc:
        return {"plot_id": plot_id, "path": path, "error": str(exc)}
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    return {"plot_id": plot_id, "mime_type": mime, "base64": base64.b64encode(data).decode("ascii")}

@contextlib.contextmanager
def _with_api_key(agent: AutoStatsAgent, api_key: str | None):
    """Swap the shared agent's OpenAI client for the call's duration.

    NOTE: `agent.client` is one shared attribute, so concurrent calls for
    different users with different injected keys can race. Fine for a
    single-tenant/local deployment; revisit (e.g. thread a client through
    run_turn) before relying on this under real multi-tenant concurrency.
    """
    if not api_key:
        yield
        return
    from openai import OpenAI

    previous = agent.client
    agent.client = OpenAI(api_key=api_key)
    try:
        yield
    finally:
        agent.client = previous
