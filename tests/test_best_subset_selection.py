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


@pytest.mark.parametrize("criterion", ["aic", "bic", "r_squared", "p_value", "mallows_cp"])
def test_best_subset_selection_finds_true_predictors(tool_ctx, rng, criterion):
    df = _make_known_structure_data(rng)
    handle = tool_ctx.data_manager.register(df, source="upload")

    result = REGISTRY.dispatch(
        tool_ctx,
        "best_subset_selection",
        {
            "dataset_id": handle.dataset_id,
            "target": "y",
            "candidate_predictors": ["x1", "x2", "x3", "x4"],
            "criterion": criterion,
        },
    )

    assert set(result.raw_summary["selected_predictors"]) == {"x1", "x2"}
    assert result.effect_size["r_squared"] > 0.85


def test_best_subset_selection_mallows_cp_matches_theory(tool_ctx, rng):
    df = _make_known_structure_data(rng)
    handle = tool_ctx.data_manager.register(df, source="upload")

    result = REGISTRY.dispatch(
        tool_ctx,
        "best_subset_selection",
        {
            "dataset_id": handle.dataset_id,
            "target": "y",
            "candidate_predictors": ["x1", "x2", "x3", "x4"],
            "criterion": "mallows_cp",
        },
    )

    assert set(result.raw_summary["selected_predictors"]) == {"x1", "x2"}


def test_best_subset_selection_respects_max_predictors(tool_ctx, rng):
    df = _make_known_structure_data(rng)
    handle = tool_ctx.data_manager.register(df, source="upload")

    result = REGISTRY.dispatch(
        tool_ctx,
        "best_subset_selection",
        {
            "dataset_id": handle.dataset_id,
            "target": "y",
            "candidate_predictors": ["x1", "x2", "x3", "x4"],
            "criterion": "aic",
            "max_predictors": 1,
        },
    )

    assert len(result.raw_summary["selected_predictors"]) == 1


def test_best_subset_selection_deduplicates_candidates(tool_ctx, rng):
    df = _make_known_structure_data(rng)
    handle = tool_ctx.data_manager.register(df, source="upload")

    result = REGISTRY.dispatch(
        tool_ctx,
        "best_subset_selection",
        {
            "dataset_id": handle.dataset_id,
            "target": "y",
            "candidate_predictors": ["x1", "x1", "x2", "x3"],
            "criterion": "aic",
        },
    )
    assert "x1" in result.raw_summary["selected_predictors"]


def test_best_subset_selection_rejects_excessive_search(tool_ctx, rng):
    n = 200
    many = {f"z{i}": rng.normal(size=n) for i in range(18)}
    df = pd.DataFrame({**many, "y": rng.normal(size=n)})
    handle = tool_ctx.data_manager.register(df, source="upload")

    with pytest.raises(ValueError, match="safety cap"):
        REGISTRY.dispatch(
            tool_ctx,
            "best_subset_selection",
            {
                "dataset_id": handle.dataset_id,
                "target": "y",
                "candidate_predictors": list(many.keys()),
                "criterion": "aic",
                "max_predictors": 9,
            },
        )


def test_best_subset_selection_rejects_empty_candidates(tool_ctx):
    df = pd.DataFrame({"x1": [1, 2, 3], "y": [1, 2, 3]})
    handle = tool_ctx.data_manager.register(df, source="upload")

    with pytest.raises(ValueError, match="must not be empty"):
        REGISTRY.dispatch(
            tool_ctx,
            "best_subset_selection",
            {"dataset_id": handle.dataset_id, "target": "y", "candidate_predictors": []},
        )


def test_best_subset_selection_rejects_invalid_max_predictors(tool_ctx):
    df = pd.DataFrame({"x1": [1, 2, 3], "y": [1, 2, 3]})
    handle = tool_ctx.data_manager.register(df, source="upload")

    with pytest.raises(ValueError, match="max_predictors"):
        REGISTRY.dispatch(
            tool_ctx,
            "best_subset_selection",
            {
                "dataset_id": handle.dataset_id,
                "target": "y",
                "candidate_predictors": ["x1"],
                "max_predictors": 0,
            },
        )


def test_best_subset_selection_rejects_unsatisfiable_p_value_alpha(tool_ctx, rng):
    df = _make_known_structure_data(rng)
    handle = tool_ctx.data_manager.register(df, source="upload")

    with pytest.raises(ValueError, match="did not select any"):
        REGISTRY.dispatch(
            tool_ctx,
            "best_subset_selection",
            {
                "dataset_id": handle.dataset_id,
                "target": "y",
                "candidate_predictors": ["x1", "x2", "x3", "x4"],
                "criterion": "p_value",
                "alpha": 1e-100,
            },
        )
