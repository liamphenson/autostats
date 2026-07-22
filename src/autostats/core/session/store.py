"""Session state persistence.

Three logically distinct pieces of state live behind one interface so the
same code path serves a stateless API request and a long-lived chat session:
conversation history, the dataset registry (owned by DataManager, not here),
and the accumulated analysis results (`ResultStore`, read later by the report
builder -- independent of whether a result is still in the active chat
context window).
"""

import json
import sqlite3
from pathlib import Path

from pydantic import BaseModel

from autostats.core.session.models import AnalysisResult, Message


class SessionState(BaseModel):
    session_id: str
    messages: list[Message] = []
    results: list[AnalysisResult] = []


class SessionStore:
    """SQLite-backed session store. Same schema works for a single-user
    CLI/notebook (local file) or a web/API deployment (shared file or,
    ported later, Postgres) -- callers only see this interface."""

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS messages (session_id TEXT, idx INTEGER, data TEXT)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS results (session_id TEXT, idx INTEGER, data TEXT)"
        )
        self._conn.commit()

    def get(self, session_id: str) -> SessionState:
        messages = [
            Message.model_validate_json(row[0])
            for row in self._conn.execute(
                "SELECT data FROM messages WHERE session_id = ? ORDER BY idx", (session_id,)
            )
        ]
        results = [
            AnalysisResult.model_validate_json(row[0])
            for row in self._conn.execute(
                "SELECT data FROM results WHERE session_id = ? ORDER BY idx", (session_id,)
            )
        ]
        return SessionState(session_id=session_id, messages=messages, results=results)

    def append_message(self, session_id: str, message: Message) -> None:
        (count,) = self._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
        ).fetchone()
        self._conn.execute(
            "INSERT INTO messages (session_id, idx, data) VALUES (?, ?, ?)",
            (session_id, count, message.model_dump_json()),
        )
        self._conn.commit()

    def add_result(self, session_id: str, result: AnalysisResult) -> None:
        (count,) = self._conn.execute(
            "SELECT COUNT(*) FROM results WHERE session_id = ?", (session_id,)
        ).fetchone()
        self._conn.execute(
            "INSERT INTO results (session_id, idx, data) VALUES (?, ?, ?)",
            (session_id, count, result.model_dump_json()),
        )
        self._conn.commit()
