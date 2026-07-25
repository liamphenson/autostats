import numpy as np
import pandas as pd
import pytest

from autostats.core.tools.registry import REGISTRY


def _register(tool_ctx, df, **kwargs):
    return tool_ctx.data_manager.register(df, source="upload", **kwargs)


def test_one_hot_encode_creates_one_column_per_category(tool_ctx):
    df = pd.DataFrame({"color": ["red", "green", "blue", "red"], "value": [1, 2, 3, 4]})
    handle = _register(tool_ctx, df)

    result = REGISTRY.dispatch(tool_ctx, "one_hot_encode", {"dataset_id": handle.dataset_id, "columns": ["color"]})

    assert result.source == "derived"
    assert result.n_cols == 4  # value + 3 color columns
    encoded = tool_ctx.data_manager.load(result.dataset_id)
    assert "color" not in encoded.columns
    assert {"color_red", "color_green", "color_blue"} <= set(encoded.columns)
    assert encoded["color_red"].tolist() == [1, 0, 0, 1]


def test_dummy_encode_drops_one_reference_category(tool_ctx):
    df = pd.DataFrame({"color": ["red", "green", "blue", "red"], "value": [1, 2, 3, 4]})
    handle = _register(tool_ctx, df)

    result = REGISTRY.dispatch(tool_ctx, "dummy_encode", {"dataset_id": handle.dataset_id, "columns": ["color"]})

    encoded = tool_ctx.data_manager.load(result.dataset_id)
    new_cols = [c for c in encoded.columns if c.startswith("color_")]
    assert len(new_cols) == 2  # k - 1 for 3 categories


def test_ordinal_encode_with_explicit_order(tool_ctx):
    df = pd.DataFrame({"size": ["small", "large", "medium", "small"]})
    handle = _register(tool_ctx, df)

    result = REGISTRY.dispatch(
        tool_ctx,
        "ordinal_encode",
        {"dataset_id": handle.dataset_id, "column": "size", "order": ["small", "medium", "large"]},
    )

    encoded = tool_ctx.data_manager.load(result.dataset_id)
    assert encoded["size"].tolist() == [0, 2, 1, 0]
    assert result.validation_warnings == []


def test_ordinal_encode_without_order_warns(tool_ctx):
    df = pd.DataFrame({"size": ["small", "large", "medium"]})
    handle = _register(tool_ctx, df)

    result = REGISTRY.dispatch(tool_ctx, "ordinal_encode", {"dataset_id": handle.dataset_id, "column": "size"})

    assert len(result.validation_warnings) == 1
    assert "alphabetically" in result.validation_warnings[0]


def test_ordinal_encode_rejects_incomplete_order(tool_ctx):
    df = pd.DataFrame({"size": ["small", "large", "medium"]})
    handle = _register(tool_ctx, df)

    with pytest.raises(ValueError, match="missing categories"):
        REGISTRY.dispatch(
            tool_ctx,
            "ordinal_encode",
            {"dataset_id": handle.dataset_id, "column": "size", "order": ["small", "large"]},
        )


def test_label_encode_assigns_arbitrary_codes_and_warns(tool_ctx):
    df = pd.DataFrame({"city": ["nyc", "la", "nyc", "sf"]})
    handle = _register(tool_ctx, df)

    result = REGISTRY.dispatch(tool_ctx, "label_encode", {"dataset_id": handle.dataset_id, "column": "city"})

    encoded = tool_ctx.data_manager.load(result.dataset_id)
    # sorted alphabetically: la=0, nyc=1, sf=2
    assert encoded["city"].tolist() == [1, 0, 1, 2]
    assert any("arbitrary integer codes" in w for w in result.validation_warnings)


def test_target_encode_replaces_category_with_target_mean(tool_ctx):
    df = pd.DataFrame(
        {
            "group": ["a", "a", "b", "b"],
            "score": [10.0, 20.0, 100.0, 200.0],
        }
    )
    handle = _register(tool_ctx, df)

    result = REGISTRY.dispatch(
        tool_ctx,
        "target_encode",
        {"dataset_id": handle.dataset_id, "column": "group", "target_column": "score"},
    )

    encoded = tool_ctx.data_manager.load(result.dataset_id)
    assert encoded["group"].tolist() == [15.0, 15.0, 150.0, 150.0]
    assert any("leakage" in w for w in result.validation_warnings)


def test_target_encode_rejects_non_numeric_target(tool_ctx):
    df = pd.DataFrame({"group": ["a", "b"], "label": ["x", "y"]})
    handle = _register(tool_ctx, df)

    with pytest.raises(ValueError, match="must be numeric"):
        REGISTRY.dispatch(
            tool_ctx,
            "target_encode",
            {"dataset_id": handle.dataset_id, "column": "group", "target_column": "label"},
        )


def test_encoding_tool_raises_on_missing_column(tool_ctx):
    df = pd.DataFrame({"a": [1, 2, 3]})
    handle = _register(tool_ctx, df)

    with pytest.raises(ValueError, match="not found"):
        REGISTRY.dispatch(tool_ctx, "one_hot_encode", {"dataset_id": handle.dataset_id, "columns": ["missing"]})


def test_derived_dataset_inherits_parent_trust_level(tool_ctx):
    df = pd.DataFrame({"color": ["red", "blue"], "value": [1, 2]})
    handle = _register(tool_ctx, df, trust_level="low", validation_warnings=["scraped from the web"])

    result = REGISTRY.dispatch(tool_ctx, "one_hot_encode", {"dataset_id": handle.dataset_id, "columns": ["color"]})

    assert result.trust_level == "low"


def test_encoded_dataset_feeds_directly_into_linear_regression(tool_ctx, rng):
    baseline = np.array([10.0] * 10 + [20.0] * 10)
    df = pd.DataFrame(
        {
            "group": ["control"] * 10 + ["treatment"] * 10,
            "score": baseline + rng.normal(scale=0.1, size=20),
        }
    )
    handle = _register(tool_ctx, df)

    encoded = REGISTRY.dispatch(tool_ctx, "dummy_encode", {"dataset_id": handle.dataset_id, "columns": ["group"]})

    result = REGISTRY.dispatch(
        tool_ctx,
        "linear_regression",
        {"dataset_id": encoded.dataset_id, "target": "score", "predictors": ["group_treatment"]},
    )
    assert result.effect_size["r_squared"] > 0.99
