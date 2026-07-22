from pathlib import Path

import pytest

from autostats.core.reporting.builder import ReportBuilder
from autostats.core.schemas.stat_result import StatResult
from autostats.core.session.models import AnalysisResult, Message
from autostats.core.session.store import SessionStore


def _make_result() -> StatResult:
    return StatResult(
        tool_name="two_sample_t_test",
        test_name="two_sample_t_test",
        statistic=2.5,
        p_value=0.01,
        interpretation="Groups A and B differ significantly (t=2.5, p=0.01).",
    )


def test_report_builder_end_to_end(tmp_path: Path):
    store = SessionStore(tmp_path / "sessions.db")
    session_id = "sess1"
    store.append_message(session_id, Message(role="user", content="compare groups A and B"))
    store.add_result(session_id, AnalysisResult(result=_make_result()))

    builder = ReportBuilder(store)
    output_path = builder.build(session_id, tmp_path / "report.html", fmt="html")

    assert output_path.exists()
    content = output_path.read_text()
    assert "two_sample_t_test" in content
    assert "differ significantly" in content


def test_report_builder_notebook_export(tmp_path: Path):
    pytest.importorskip("nbformat")
    store = SessionStore(tmp_path / "sessions.db")
    session_id = "sess2"
    store.add_result(session_id, AnalysisResult(result=_make_result()))

    builder = ReportBuilder(store)
    output_path = builder.build(session_id, tmp_path / "report.ipynb", fmt="notebook")

    assert output_path.exists()
    assert "two_sample_t_test" in output_path.read_text()
