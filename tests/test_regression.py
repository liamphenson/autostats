import numpy as np
import pandas as pd
import pytest

from autostats.core.tools.registry import REGISTRY


def test_linear_regression_recovers_known_coefficients(tool_ctx, rng):
    x = rng.normal(size=300)
    y = 3.0 + 2.0 * x + rng.normal(scale=0.1, size=300)
    df = pd.DataFrame({"x": x, "y": y})
    handle = tool_ctx.data_manager.register(df, source="upload")

    result = REGISTRY.dispatch(
        tool_ctx, "linear_regression", {"dataset_id": handle.dataset_id, "target": "y", "predictors": ["x"]}
    )

    assert result.effect_size["r_squared"] > 0.95
    assert abs(result.raw_summary["params"]["x"] - 2.0) < 0.1


def test_logistic_regression_separates_classes(tool_ctx, rng):
    x = rng.normal(size=400)
    logits = 4 * x
    p = 1 / (1 + np.exp(-logits))
    y = (rng.uniform(size=400) < p).astype(int)
    df = pd.DataFrame({"x": x, "y": y})
    handle = tool_ctx.data_manager.register(df, source="upload")

    result = REGISTRY.dispatch(
        tool_ctx, "logistic_regression", {"dataset_id": handle.dataset_id, "target": "y", "predictors": ["x"]}
    )

    assert result.effect_size["mcfadden_pseudo_r_squared"] > 0.3
    assert result.p_value < 0.05


def test_weighted_linear_regression_recovers_coefficients_and_passes_diagnostics(tool_ctx, rng):
    # Heteroscedastic noise (variance grows with x) with correctly-specified inverse-variance
    # weights: a properly weighted fit should both recover the true slope AND show no residual
    # heteroscedasticity -- unlike the raw (unweighted) residuals, which still would.
    x = rng.uniform(1, 10, size=300)
    noise_sd = 0.5 * x
    y = 3.0 + 2.0 * x + rng.normal(scale=noise_sd)
    weights = 1 / (noise_sd**2)
    df = pd.DataFrame({"x": x, "y": y, "w": weights})
    handle = tool_ctx.data_manager.register(df, source="upload")

    result = REGISTRY.dispatch(
        tool_ctx,
        "weighted_linear_regression",
        {"dataset_id": handle.dataset_id, "target": "y", "predictors": ["x"], "weights_column": "w"},
    )

    assert abs(result.raw_summary["params"]["x"] - 2.0) < 0.2
    bp_check = next(a for a in result.assumptions if a.name == "homoscedasticity_breusch_pagan")
    assert bp_check.passed, "weighted residuals should not show heteroscedasticity with correct weights"


def test_weighted_linear_regression_rejects_nonpositive_weights(tool_ctx, rng):
    x = rng.normal(size=50)
    y = 3.0 + 2.0 * x + rng.normal(scale=0.1, size=50)
    weights = np.ones(50)
    weights[0] = -1.0
    df = pd.DataFrame({"x": x, "y": y, "w": weights})
    handle = tool_ctx.data_manager.register(df, source="upload")

    with pytest.raises(ValueError, match="positive"):
        REGISTRY.dispatch(
            tool_ctx,
            "weighted_linear_regression",
            {"dataset_id": handle.dataset_id, "target": "y", "predictors": ["x"], "weights_column": "w"},
        )


def test_irls_regression_corrects_heteroscedasticity_without_known_weights(tool_ctx, rng):
    # Same heteroscedastic setup as the WLS test, but with NO weights column at all --
    # irls_regression must discover suitable weights on its own.
    x = rng.uniform(1, 10, size=300)
    noise_sd = 0.5 * x
    y = 3.0 + 2.0 * x + rng.normal(scale=noise_sd)
    df = pd.DataFrame({"x": x, "y": y})
    handle = tool_ctx.data_manager.register(df, source="upload")

    result = REGISTRY.dispatch(
        tool_ctx, "irls_regression", {"dataset_id": handle.dataset_id, "target": "y", "predictors": ["x"]}
    )

    assert abs(result.raw_summary["params"]["x"] - 2.0) < 0.2
    bp_check = next(a for a in result.assumptions if a.name == "homoscedasticity_breusch_pagan")
    assert bp_check.passed, "IRLS-derived weights should correct the known heteroscedasticity"
    assert result.raw_summary["irls_iterations"] >= 1
    assert "IRLS" in result.interpretation


def test_irls_regression_converges_quickly_on_homoscedastic_data(tool_ctx, rng):
    x = rng.normal(size=200)
    y = 3.0 + 2.0 * x + rng.normal(scale=0.5, size=200)
    df = pd.DataFrame({"x": x, "y": y})
    handle = tool_ctx.data_manager.register(df, source="upload")

    result = REGISTRY.dispatch(
        tool_ctx, "irls_regression", {"dataset_id": handle.dataset_id, "target": "y", "predictors": ["x"]}
    )

    assert result.raw_summary["irls_converged"] is True
    assert abs(result.raw_summary["params"]["x"] - 2.0) < 0.2


def test_irls_regression_rejects_invalid_max_iterations(tool_ctx, rng):
    x = rng.normal(size=50)
    y = 3.0 + 2.0 * x + rng.normal(scale=0.1, size=50)
    df = pd.DataFrame({"x": x, "y": y})
    handle = tool_ctx.data_manager.register(df, source="upload")

    with pytest.raises(ValueError, match="max_iterations"):
        REGISTRY.dispatch(
            tool_ctx,
            "irls_regression",
            {"dataset_id": handle.dataset_id, "target": "y", "predictors": ["x"], "max_iterations": 0},
        )
