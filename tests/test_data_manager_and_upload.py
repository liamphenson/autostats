import pandas as pd

from autostats.core.tools.registry import REGISTRY


def test_upload_dataset_roundtrip(tool_ctx, tmp_path):
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}).to_csv(csv_path, index=False)

    handle = REGISTRY.dispatch(tool_ctx, "upload_dataset", {"file_path": str(csv_path)})

    assert handle.n_rows == 3
    assert handle.n_cols == 2
    assert handle.source == "upload"

    reloaded = tool_ctx.data_manager.load(handle.dataset_id)
    assert list(reloaded.columns) == ["a", "b"]
    assert len(reloaded) == 3


def test_dataset_catalog_text_hides_raw_rows(tool_ctx, two_group_df):
    tool_ctx.data_manager.register(two_group_df, source="upload")
    catalog = tool_ctx.data_manager.catalog_text()
    assert "dataset_id=" in catalog
    # catalog must never contain the full row count worth of raw values
    assert catalog.count("\n") < 5
