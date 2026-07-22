import numpy as np
import pandas as pd

from autostats.core.tools.registry import REGISTRY


def test_mann_whitney_u_detects_shift(tool_ctx, rng):
    df = pd.DataFrame(
        {
            "value": np.concatenate([rng.exponential(1, 40), rng.exponential(4, 40)]),
            "group": ["a"] * 40 + ["b"] * 40,
        }
    )
    handle = tool_ctx.data_manager.register(df, source="upload")
    result = REGISTRY.dispatch(
        tool_ctx, "mann_whitney_u_test", {"dataset_id": handle.dataset_id, "group_col": "group", "value_col": "value"}
    )
    assert result.p_value < 0.05


def test_chi_square_independence_on_associated_categoricals(tool_ctx, rng):
    n = 200
    x = rng.choice(["yes", "no"], size=n)
    y = np.where(x == "yes", rng.choice(["A", "B"], size=n, p=[0.9, 0.1]), rng.choice(["A", "B"], size=n, p=[0.1, 0.9]))
    df = pd.DataFrame({"x": x, "y": y})
    handle = tool_ctx.data_manager.register(df, source="upload")

    result = REGISTRY.dispatch(
        tool_ctx, "chi_square_test_independence", {"dataset_id": handle.dataset_id, "column_a": "x", "column_b": "y"}
    )
    assert result.p_value < 0.001
    assert result.effect_size["cramers_v"] > 0.5


def test_kruskal_wallis_detects_group_difference(tool_ctx, rng):
    df = pd.DataFrame(
        {
            "value": np.concatenate([rng.exponential(1, 30), rng.exponential(5, 30), rng.exponential(10, 30)]),
            "group": ["a"] * 30 + ["b"] * 30 + ["c"] * 30,
        }
    )
    handle = tool_ctx.data_manager.register(df, source="upload")
    result = REGISTRY.dispatch(
        tool_ctx, "kruskal_wallis_test", {"dataset_id": handle.dataset_id, "group_col": "group", "value_col": "value"}
    )
    assert result.p_value < 0.01
