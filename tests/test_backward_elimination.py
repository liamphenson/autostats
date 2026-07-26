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
def test_backward_elimination_removes_noise_and_keeps_true_predictors(tool_ctx, rng, criterion):
    df = _make_known_structure_data(rng)
    handle = tool_ctx.data_manager.register(df, source="upload")

    result = REGISTRY.dispatch(
        tool_ctx,
        "backward_elimination",
        {
            "dataset_id": handle.dataset_id,
            "target": "y",
            "predictors": ["x1", "x2", "x3", "x4"],
            "criterion": criterion,
        },
    )

    removed = {s["removed"] for s in result.raw_summary["steps"]}
    assert removed == {"x3", "x4"}
    assert result.effect_size["r_squared"] > 0.85


def test_backward_elimination_r_squared_keeps_true_predictors(tool_ctx, rng):
    df = _make_known_structure_data(rng)
    handle = tool_ctx.data_manager.register(df, source="upload")

    result = REGISTRY.dispatch(
        tool_ctx,
        "backward_elimination",
        {
            "dataset_id": handle.dataset_id,
            "target": "y",
            "predictors": ["x1", "x2", "x3", "x4"],
            "criterion": "r_squared",
        },
    )

    removed = {s["removed"] for s in result.raw_summary["steps"]}
    assert "x1" not in removed and "x2" not in removed
    # each successive removal should improve (or hold steady) adjusted R-squared
    scores = [s["r_squared"] for s in result.raw_summary["steps"]]
    assert scores == sorted(scores)


def test_backward_elimination_raises_when_nothing_improves(tool_ctx, rng):
    n = 300
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 5 + 4 * x1 + 5 * x2 + rng.normal(scale=0.3, size=n)  # both strongly predictive
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    handle = tool_ctx.data_manager.register(df, source="upload")

    with pytest.raises(ValueError, match="did not remove any"):
        REGISTRY.dispatch(
            tool_ctx,
            "backward_elimination",
            {
                "dataset_id": handle.dataset_id,
                "target": "y",
                "predictors": ["x1", "x2"],
                "criterion": "bic",
            },
        )


def test_backward_elimination_rejects_empty_predictors(tool_ctx):
    df = pd.DataFrame({"x1": [1, 2, 3], "y": [1, 2, 3]})
    handle = tool_ctx.data_manager.register(df, source="upload")

    with pytest.raises(ValueError, match="must not be empty"):
        REGISTRY.dispatch(
            tool_ctx,
            "backward_elimination",
            {"dataset_id": handle.dataset_id, "target": "y", "predictors": []},
        )


def test_forward_and_backward_selection_agree_on_known_structure_data(tool_ctx, rng):
    # A meaningful cross-check: both procedures, started from opposite ends, should
    # converge on the same predictor set for data with a clear signal/noise split.
    df = _make_known_structure_data(rng)
    handle = tool_ctx.data_manager.register(df, source="upload")

    forward = REGISTRY.dispatch(
        tool_ctx,
        "forward_selection",
        {
            "dataset_id": handle.dataset_id,
            "target": "y",
            "candidate_predictors": ["x1", "x2", "x3", "x4"],
            "criterion": "bic",
        },
    )
    backward = REGISTRY.dispatch(
        tool_ctx,
        "backward_elimination",
        {
            "dataset_id": handle.dataset_id,
            "target": "y",
            "predictors": ["x1", "x2", "x3", "x4"],
            "criterion": "bic",
        },
    )

    forward_selected = {s["added"] for s in forward.raw_summary["steps"]}
    backward_kept = {"x1", "x2", "x3", "x4"} - {s["removed"] for s in backward.raw_summary["steps"]}
    assert forward_selected == backward_kept == {"x1", "x2"}
