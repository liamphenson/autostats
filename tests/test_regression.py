import numpy as np
import pandas as pd

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
