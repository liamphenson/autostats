import numpy as np
import pandas as pd
import pytest

from autostats.core.tools.registry import REGISTRY


def _register(tool_ctx, df, **kwargs):
    return tool_ctx.data_manager.register(df, source="upload", **kwargs)


def test_box_cox_transform_prefers_log_when_ci_contains_zero(tool_ctx, rng):
    # Log-normal data: the true generating lambda is 0 (log), and with a large enough
    # sample the CI should comfortably contain 0.
    normal_data = rng.normal(loc=5, scale=1, size=500)
    df = pd.DataFrame({"y": np.exp(normal_data)})
    handle = _register(tool_ctx, df)

    result = REGISTRY.dispatch(tool_ctx, "box_cox_transform", {"dataset_id": handle.dataset_id, "column": "y"})

    assert result.source == "derived"
    assert result.source_metadata["lambda"] == 0
    assert "lambda=0" in result.validation_warnings[0]
    transformed = tool_ctx.data_manager.load(result.dataset_id)
    np.testing.assert_allclose(transformed["y"].to_numpy(), np.log(df["y"].to_numpy()))


def test_box_cox_transform_uses_raw_mle_when_no_interpretable_lambda_fits(tool_ctx, monkeypatch):
    # Rather than hunting for random data whose CI happens to miss every candidate,
    # control scipy's return value directly: a CI of (0.6, 0.85) excludes all of
    # [-2, -1, -0.5, 0, 0.5, 1, 2], so the raw MLE must be used.
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0]})
    handle = _register(tool_ctx, df)

    fake_lambda_mle, fake_ci = 0.73, (0.6, 0.85)
    monkeypatch.setattr(
        "autostats.core.tools.preprocessing.transforms.scipy_stats.boxcox",
        lambda data, alpha: (data, fake_lambda_mle, fake_ci),
    )

    result = REGISTRY.dispatch(tool_ctx, "box_cox_transform", {"dataset_id": handle.dataset_id, "column": "y"})

    assert result.source_metadata["lambda"] == fake_lambda_mle
    assert result.source_metadata["lambda"] == result.source_metadata["lambda_mle"]
    assert "MLE estimate" in result.validation_warnings[0]


def test_box_cox_transform_rejects_nonpositive_values(tool_ctx):
    df = pd.DataFrame({"y": [1.0, 2.0, 0.0, 4.0]})
    handle = _register(tool_ctx, df)

    with pytest.raises(ValueError, match="strictly positive"):
        REGISTRY.dispatch(tool_ctx, "box_cox_transform", {"dataset_id": handle.dataset_id, "column": "y"})


def test_box_cox_transform_rejects_non_numeric_column(tool_ctx):
    df = pd.DataFrame({"y": ["a", "b", "c"]})
    handle = _register(tool_ctx, df)

    with pytest.raises(ValueError, match="must be numeric"):
        REGISTRY.dispatch(tool_ctx, "box_cox_transform", {"dataset_id": handle.dataset_id, "column": "y"})


def test_box_cox_transform_rejects_missing_column(tool_ctx):
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0]})
    handle = _register(tool_ctx, df)

    with pytest.raises(ValueError, match="not found"):
        REGISTRY.dispatch(tool_ctx, "box_cox_transform", {"dataset_id": handle.dataset_id, "column": "missing"})


def test_box_cox_transform_inherits_parent_trust_level(tool_ctx, rng):
    normal_data = rng.normal(loc=5, scale=1, size=200)
    df = pd.DataFrame({"y": np.exp(normal_data)})
    handle = _register(tool_ctx, df, trust_level="low", validation_warnings=["scraped from the web"])

    result = REGISTRY.dispatch(tool_ctx, "box_cox_transform", {"dataset_id": handle.dataset_id, "column": "y"})

    assert result.trust_level == "low"


def test_box_cox_transform_feeds_directly_into_linear_regression(tool_ctx, rng):
    x = rng.uniform(1, 5, size=300)
    normal_resid = rng.normal(scale=0.1, size=300)
    y = np.exp(1.0 + 0.5 * x + normal_resid)  # log(y) is linear in x
    df = pd.DataFrame({"x": x, "y": y})
    handle = _register(tool_ctx, df)

    transformed = REGISTRY.dispatch(tool_ctx, "box_cox_transform", {"dataset_id": handle.dataset_id, "column": "y"})

    result = REGISTRY.dispatch(
        tool_ctx,
        "linear_regression",
        {"dataset_id": transformed.dataset_id, "target": "y", "predictors": ["x"]},
    )
    assert result.effect_size["r_squared"] > 0.9
