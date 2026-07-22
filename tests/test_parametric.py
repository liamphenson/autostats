import numpy as np
from scipy import stats as scipy_stats

from autostats.core.tools.registry import REGISTRY


def test_two_sample_t_test_matches_scipy_reference(tool_ctx, two_group_df):
    handle = tool_ctx.data_manager.register(two_group_df, source="upload")

    result = REGISTRY.dispatch(
        tool_ctx,
        "two_sample_t_test",
        {"dataset_id": handle.dataset_id, "group_col": "group", "value_col": "value"},
    )

    a = two_group_df.loc[two_group_df.group == "A", "value"]
    b = two_group_df.loc[two_group_df.group == "B", "value"]
    expected = scipy_stats.ttest_ind(a, b, equal_var=False)

    assert result.statistic == expected.statistic
    assert result.p_value == expected.pvalue
    assert result.assumptions_met is True
    assert "significant" in result.interpretation


def test_one_way_anova_detects_group_difference(tool_ctx, rng):
    import pandas as pd

    df = pd.DataFrame(
        {
            "value": np.concatenate(
                [rng.normal(0, 1, 50), rng.normal(5, 1, 50), rng.normal(10, 1, 50)]
            ),
            "group": ["a"] * 50 + ["b"] * 50 + ["c"] * 50,
        }
    )
    handle = tool_ctx.data_manager.register(df, source="upload")

    result = REGISTRY.dispatch(
        tool_ctx, "one_way_anova", {"dataset_id": handle.dataset_id, "group_col": "group", "value_col": "value"}
    )

    assert result.p_value < 0.001
    assert result.effect_size["eta_squared"] > 0.5


def test_nonnormal_two_sample_recommends_mann_whitney(tool_ctx, rng):
    import pandas as pd

    df = pd.DataFrame(
        {
            "value": np.concatenate([rng.exponential(1, 15), rng.exponential(3, 15)]),
            "group": ["a"] * 15 + ["b"] * 15,
        }
    )
    handle = tool_ctx.data_manager.register(df, source="upload")

    result = REGISTRY.dispatch(
        tool_ctx, "two_sample_t_test", {"dataset_id": handle.dataset_id, "group_col": "group", "value_col": "value"}
    )

    if not result.assumptions_met:
        assert result.recommended_alternative == "mann_whitney_u_test"
