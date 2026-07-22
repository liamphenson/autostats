"""End-to-end check of the non-LLM parts of a session: upload -> tool calls ->
ResultStore accumulation -> report export. Mirrors the CLI flow without
requiring a real OpenAI API call."""

import pandas as pd

from autostats.core.data.manager import DataManager
from autostats.core.reporting.builder import ReportBuilder
from autostats.core.schemas.stat_result import StatResult
from autostats.core.session.models import AnalysisResult, Message
from autostats.core.session.store import SessionStore
from autostats.core.tools import load_all_tools
from autostats.core.tools.base import ToolContext
from autostats.core.tools.registry import REGISTRY

load_all_tools()


def test_full_session_upload_analyze_report(tmp_path):
    session_id = "e2e"
    store = SessionStore(tmp_path / "sessions.db")
    data_manager = DataManager(session_id, tmp_path / "datasets")
    ctx = ToolContext(session_id=session_id, data_manager=data_manager, plots_dir=str(tmp_path / "plots"))

    csv_path = tmp_path / "groups.csv"
    pd.DataFrame(
        {
            "value": [10.1, 9.8, 10.5, 9.9, 10.2, 15.1, 14.8, 15.4, 14.9, 15.2],
            "group": ["A"] * 5 + ["B"] * 5,
        }
    ).to_csv(csv_path, index=False)

    store.append_message(session_id, Message(role="user", content="analyze groups.csv"))

    handle = REGISTRY.dispatch(ctx, "upload_dataset", {"file_path": str(csv_path)})
    assert data_manager.get_meta(handle.dataset_id).n_rows == 10

    desc_result = REGISTRY.dispatch(ctx, "describe_dataset", {"dataset_id": handle.dataset_id})
    store.add_result(session_id, AnalysisResult(result=desc_result))

    ttest_result = REGISTRY.dispatch(
        ctx,
        "two_sample_t_test",
        {"dataset_id": handle.dataset_id, "group_col": "group", "value_col": "value"},
    )
    assert isinstance(ttest_result, StatResult)
    assert ttest_result.p_value < 0.001
    store.add_result(session_id, AnalysisResult(result=ttest_result))

    state = store.get(session_id)
    assert len(state.results) == 2
    assert len(state.messages) == 1

    builder = ReportBuilder(store)
    report_path = builder.build(session_id, tmp_path / "report.html", fmt="html")
    content = report_path.read_text()
    assert "two_sample_t_test" in content
    assert "descriptive_statistics" in content
