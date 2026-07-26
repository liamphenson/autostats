import numpy as np
import pandas as pd
import pytest

from autostats.core.tools.registry import REGISTRY


def _make_known_structure_data(rng, n=200):
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)  # noise, unrelated to y
    x4 = rng.normal(size=n)  # noise, unrelated to y
    y = 5 + 2 * x1 + 3 * x2 + rng.normal(scale=1.0, size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "y": y})


@pytest.mark.parametrize("criterion", ["aic", "bic", "p_value"])
def test_forward_selection_finds_true_predictors_and_stops(tool_ctx, rng, criterion):
    df = _make_known_structure_data(rng)
    handle = tool_ctx.data_manager.register(df, source="upload")

    result = REGISTRY.dispatch(
        tool_ctx,
        "forward_selection",
        {
            "dataset_id": handle.dataset_id,
            "target": "y",
            "candidate_predictors": ["x1", "x2", "x3", "x4"],
            "criterion": criterion,
        },
    )

    selected = {s["added"] for s in result.raw_summary["steps"]}
    assert selected == {"x1", "x2"}
    assert result.effect_size["r_squared"] > 0.85


def test_forward_selection_r_squared_uses_adjusted_r_squared(tool_ctx, rng):
    # Documented, expected behavior: adjusted R-squared is a more liberal stopping
    # criterion than AIC/BIC/p-value and can retain a marginally-improving predictor
    # that isn't really part of the true model -- this isn't a bug, it's exactly why
    # the tool's description recommends aic/bic/p_value over r_squared.
    df = _make_known_structure_data(rng)
    handle = tool_ctx.data_manager.register(df, source="upload")

    result = REGISTRY.dispatch(
        tool_ctx,
        "forward_selection",
        {
            "dataset_id": handle.dataset_id,
            "target": "y",
            "candidate_predictors": ["x1", "x2", "x3", "x4"],
            "criterion": "r_squared",
        },
    )

    selected = [s["added"] for s in result.raw_summary["steps"]]
    assert {"x1", "x2"} <= set(selected)
    # each successive step's recorded score must be a strict improvement
    scores = [s["r_squared"] for s in result.raw_summary["steps"]]
    assert scores == sorted(scores)


def test_forward_selection_respects_max_predictors(tool_ctx, rng):
    df = _make_known_structure_data(rng)
    handle = tool_ctx.data_manager.register(df, source="upload")

    result = REGISTRY.dispatch(
        tool_ctx,
        "forward_selection",
        {
            "dataset_id": handle.dataset_id,
            "target": "y",
            "candidate_predictors": ["x1", "x2", "x3", "x4"],
            "criterion": "aic",
            "max_predictors": 1,
        },
    )

    assert len(result.raw_summary["steps"]) == 1
    assert result.raw_summary["steps"][0]["added"] == "x2"  # the single strongest predictor


def test_forward_selection_raises_when_nothing_improves(tool_ctx, rng):
    n = 300
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "y": rng.normal(size=n),  # pure noise, unrelated to any candidate
        }
    )
    handle = tool_ctx.data_manager.register(df, source="upload")

    with pytest.raises(ValueError, match="did not add any"):
        REGISTRY.dispatch(
            tool_ctx,
            "forward_selection",
            {
                "dataset_id": handle.dataset_id,
                "target": "y",
                "candidate_predictors": ["x1", "x2"],
                "criterion": "bic",
            },
        )


def test_forward_selection_rejects_empty_candidates(tool_ctx):
    df = pd.DataFrame({"x1": [1, 2, 3], "y": [1, 2, 3]})
    handle = tool_ctx.data_manager.register(df, source="upload")

    with pytest.raises(ValueError, match="must not be empty"):
        REGISTRY.dispatch(
            tool_ctx,
            "forward_selection",
            {"dataset_id": handle.dataset_id, "target": "y", "candidate_predictors": []},
        )


def test_forward_selection_rejects_invalid_max_predictors(tool_ctx):
    df = pd.DataFrame({"x1": [1, 2, 3], "y": [1, 2, 3]})
    handle = tool_ctx.data_manager.register(df, source="upload")

    with pytest.raises(ValueError, match="max_predictors"):
        REGISTRY.dispatch(
            tool_ctx,
            "forward_selection",
            {
                "dataset_id": handle.dataset_id,
                "target": "y",
                "candidate_predictors": ["x1"],
                "max_predictors": 0,
            },
        )
