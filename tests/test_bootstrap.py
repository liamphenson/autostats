import numpy as np
import pandas as pd
import pytest

from autostats.core.tools.registry import REGISTRY


def _register(tool_ctx, df, **kwargs):
    return tool_ctx.data_manager.register(df, source="upload", **kwargs)


@pytest.mark.parametrize("statistic", ["mean", "median", "std", "var"])
def test_bootstrap_point_estimate_matches_full_sample_statistic(tool_ctx, rng, statistic):
    x = rng.exponential(scale=5, size=100)
    df = pd.DataFrame({"x": x})
    handle = _register(tool_ctx, df)

    result = REGISTRY.dispatch(
        tool_ctx, "bootstrap", {"dataset_id": handle.dataset_id, "column": "x", "statistic": statistic, "n_boot": 500}
    )

    stat_func = {"mean": np.mean, "median": np.median, "std": np.std, "var": np.var}[statistic]
    assert result.effect_size["point_estimate"] == pytest.approx(float(stat_func(x)))


def test_bootstrap_is_reproducible_with_random_state(tool_ctx, rng):
    x = rng.normal(size=100)
    df = pd.DataFrame({"x": x})
    handle = _register(tool_ctx, df)

    r1 = REGISTRY.dispatch(
        tool_ctx, "bootstrap", {"dataset_id": handle.dataset_id, "column": "x", "n_boot": 500, "random_state": 7}
    )
    r2 = REGISTRY.dispatch(
        tool_ctx, "bootstrap", {"dataset_id": handle.dataset_id, "column": "x", "n_boot": 500, "random_state": 7}
    )

    assert r1.confidence_interval == r2.confidence_interval
    assert r1.effect_size["standard_error"] == r2.effect_size["standard_error"]


def test_bootstrap_does_not_flag_median_as_unreliable(tool_ctx, rng):
    # The key theoretical distinction from jackknife: bootstrap IS valid for the
    # median (a non-smooth statistic), so it should not carry an analogous warning.
    x = rng.normal(size=100)
    df = pd.DataFrame({"x": x})
    handle = _register(tool_ctx, df)

    result = REGISTRY.dispatch(
        tool_ctx, "bootstrap", {"dataset_id": handle.dataset_id, "column": "x", "statistic": "median", "n_boot": 500}
    )

    assert result.assumptions_met is True
    assert result.warnings == []


def test_bootstrap_does_not_return_raw_replicates(tool_ctx, rng):
    x = rng.normal(size=200)
    df = pd.DataFrame({"x": x})
    handle = _register(tool_ctx, df)

    result = REGISTRY.dispatch(tool_ctx, "bootstrap", {"dataset_id": handle.dataset_id, "column": "x", "n_boot": 500})

    assert "bootstrap_estimates" not in result.raw_summary
    assert len(result.raw_summary) <= 2


def test_bootstrap_rejects_non_numeric_column(tool_ctx):
    df = pd.DataFrame({"cat": ["a", "b", "c", "d"]})
    handle = _register(tool_ctx, df)

    with pytest.raises(ValueError, match="must be numeric"):
        REGISTRY.dispatch(tool_ctx, "bootstrap", {"dataset_id": handle.dataset_id, "column": "cat"})


def test_bootstrap_rejects_missing_column(tool_ctx):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    handle = _register(tool_ctx, df)

    with pytest.raises(ValueError, match="not found"):
        REGISTRY.dispatch(tool_ctx, "bootstrap", {"dataset_id": handle.dataset_id, "column": "missing"})


def test_bootstrap_rejects_too_few_values(tool_ctx):
    df = pd.DataFrame({"x": [1.0]})
    handle = _register(tool_ctx, df)

    with pytest.raises(ValueError, match="at least 2"):
        REGISTRY.dispatch(tool_ctx, "bootstrap", {"dataset_id": handle.dataset_id, "column": "x"})


def test_bootstrap_rejects_non_positive_n_boot(tool_ctx, rng):
    df = pd.DataFrame({"x": rng.normal(size=20)})
    handle = _register(tool_ctx, df)

    with pytest.raises(ValueError, match="positive"):
        REGISTRY.dispatch(tool_ctx, "bootstrap", {"dataset_id": handle.dataset_id, "column": "x", "n_boot": 0})


def test_bootstrap_rejects_excessive_n_boot(tool_ctx, rng):
    df = pd.DataFrame({"x": rng.normal(size=20)})
    handle = _register(tool_ctx, df)

    with pytest.raises(ValueError, match="n_boot must be"):
        REGISTRY.dispatch(tool_ctx, "bootstrap", {"dataset_id": handle.dataset_id, "column": "x", "n_boot": 50_000})


def test_bootstrap_rejects_excessive_total_draws(tool_ctx, rng):
    df = pd.DataFrame({"x": rng.normal(size=100_000)})
    handle = _register(tool_ctx, df)

    with pytest.raises(ValueError, match="safety cap"):
        REGISTRY.dispatch(tool_ctx, "bootstrap", {"dataset_id": handle.dataset_id, "column": "x", "n_boot": 20_000})


def test_bootstrap_flags_small_sample_size(tool_ctx, rng):
    df = pd.DataFrame({"x": rng.normal(size=5)})
    handle = _register(tool_ctx, df)

    result = REGISTRY.dispatch(tool_ctx, "bootstrap", {"dataset_id": handle.dataset_id, "column": "x", "n_boot": 200})

    size_check = next(a for a in result.assumptions if a.name.startswith("sample_size"))
    assert size_check.passed is False
    assert result.assumptions_met is False
