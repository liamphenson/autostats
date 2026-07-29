import numpy as np
import pandas as pd
import pytest

from autostats.core.tools.registry import REGISTRY


def _register(tool_ctx, df, **kwargs):
    return tool_ctx.data_manager.register(df, source="upload", **kwargs)


def test_jackknife_mean_matches_classical_standard_error(tool_ctx, rng):
    # For the mean specifically, the correctly-computed jackknife SE has a known-exact
    # closed form: SE_jack = s/sqrt(n), the classical standard error of the mean. This
    # gives a precise, independent check on the CI width (the bug this was rewritten to
    # fix understated the margin by a factor of exactly (n-1)).
    n = 100
    x = rng.normal(loc=50, scale=10, size=n)
    df = pd.DataFrame({"x": x})
    handle = _register(tool_ctx, df)

    result = REGISTRY.dispatch(tool_ctx, "jackknife", {"dataset_id": handle.dataset_id, "column": "x", "statistic": "mean"})

    s = np.std(x, ddof=1)
    correct_margin = 1.96 * s / np.sqrt(n)
    ci_low, ci_high = result.confidence_interval
    reported_margin = (ci_high - ci_low) / 2

    assert result.effect_size["point_estimate"] == pytest.approx(np.mean(x))
    assert reported_margin == pytest.approx(correct_margin, rel=1e-6)


def test_jackknife_bias_is_exactly_zero_for_mean(tool_ctx, rng):
    # Known identity: the jackknife bias estimate for the mean is exactly zero, since
    # the mean of the leave-one-out replicates equals the true sample mean exactly.
    x = rng.normal(size=80)
    df = pd.DataFrame({"x": x})
    handle = _register(tool_ctx, df)

    result = REGISTRY.dispatch(tool_ctx, "jackknife", {"dataset_id": handle.dataset_id, "column": "x", "statistic": "mean"})

    assert result.effect_size["bias"] == pytest.approx(0.0, abs=1e-9)
    assert result.effect_size["bias_corrected_estimate"] == pytest.approx(result.effect_size["point_estimate"])


@pytest.mark.parametrize("statistic", ["mean", "median", "std", "var"])
def test_jackknife_point_estimate_matches_full_sample_statistic(tool_ctx, rng, statistic):
    # The core bug this rewrite fixes: the tool must report the actual statistic
    # computed on the full sample, not just the mean of the leave-one-out replicates
    # (which is a materially different, systematically biased quantity for anything
    # other than the mean).
    x = rng.exponential(scale=5, size=60)
    df = pd.DataFrame({"x": x})
    handle = _register(tool_ctx, df)

    result = REGISTRY.dispatch(tool_ctx, "jackknife", {"dataset_id": handle.dataset_id, "column": "x", "statistic": statistic})

    stat_func = {"mean": np.mean, "median": np.median, "std": np.std, "var": np.var}[statistic]
    assert result.effect_size["point_estimate"] == pytest.approx(float(stat_func(x)))


def test_jackknife_flags_median_as_unreliable(tool_ctx, rng):
    x = rng.normal(size=50)
    df = pd.DataFrame({"x": x})
    handle = _register(tool_ctx, df)

    result = REGISTRY.dispatch(tool_ctx, "jackknife", {"dataset_id": handle.dataset_id, "column": "x", "statistic": "median"})

    assert result.assumptions_met is False
    assert result.recommended_alternative == "bootstrap"
    assert any("median" in w.lower() for w in result.warnings)


def test_jackknife_does_not_return_raw_replicates(tool_ctx, rng):
    x = rng.normal(size=200)
    df = pd.DataFrame({"x": x})
    handle = _register(tool_ctx, df)

    result = REGISTRY.dispatch(tool_ctx, "jackknife", {"dataset_id": handle.dataset_id, "column": "x", "statistic": "mean"})

    assert "jackknife_estimates" not in result.raw_summary
    assert len(result.raw_summary) <= 2  # just a small min/max summary, not n values


def test_jackknife_rejects_non_numeric_column(tool_ctx):
    df = pd.DataFrame({"cat": ["a", "b", "c", "d"]})
    handle = _register(tool_ctx, df)

    with pytest.raises(ValueError, match="must be numeric"):
        REGISTRY.dispatch(tool_ctx, "jackknife", {"dataset_id": handle.dataset_id, "column": "cat"})


def test_jackknife_rejects_missing_column(tool_ctx):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    handle = _register(tool_ctx, df)

    with pytest.raises(ValueError, match="not found"):
        REGISTRY.dispatch(tool_ctx, "jackknife", {"dataset_id": handle.dataset_id, "column": "missing"})


def test_jackknife_rejects_too_few_values(tool_ctx):
    df = pd.DataFrame({"x": [1.0]})
    handle = _register(tool_ctx, df)

    with pytest.raises(ValueError, match="at least 2"):
        REGISTRY.dispatch(tool_ctx, "jackknife", {"dataset_id": handle.dataset_id, "column": "x"})


def test_jackknife_rejects_excessive_sample_size(tool_ctx, rng):
    df = pd.DataFrame({"x": rng.normal(size=25_000)})
    handle = _register(tool_ctx, df)

    with pytest.raises(ValueError, match="safety cap"):
        REGISTRY.dispatch(tool_ctx, "jackknife", {"dataset_id": handle.dataset_id, "column": "x"})


def test_jackknife_flags_small_sample_size(tool_ctx, rng):
    df = pd.DataFrame({"x": rng.normal(size=5)})
    handle = _register(tool_ctx, df)

    result = REGISTRY.dispatch(tool_ctx, "jackknife", {"dataset_id": handle.dataset_id, "column": "x", "statistic": "mean"})

    size_check = next(a for a in result.assumptions if a.name.startswith("sample_size"))
    assert size_check.passed is False
    assert result.assumptions_met is False


# --- jackknife_regression ----------------------------------------------------

def test_jackknife_regression_matches_full_model_coefficients(tool_ctx, rng):
    n = 100
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 5 + 2 * x1 + 3 * x2 + rng.normal(scale=1.0, size=n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    handle = _register(tool_ctx, df)

    result = REGISTRY.dispatch(
        tool_ctx, "jackknife_regression", {"dataset_id": handle.dataset_id, "target": "y", "predictors": ["x1", "x2"]}
    )
    ols_result = REGISTRY.dispatch(
        tool_ctx, "linear_regression", {"dataset_id": handle.dataset_id, "target": "y", "predictors": ["x1", "x2"]}
    )

    coef_by_term = {row[0]: row[1] for row in result.tables[0].rows}
    for term, ols_coef in ols_result.raw_summary["params"].items():
        assert coef_by_term[term] == pytest.approx(ols_coef, abs=1e-3)


def test_jackknife_regression_works_with_single_predictor(tool_ctx, rng):
    n = 60
    x1 = rng.normal(size=n)
    y = 5 + 2 * x1 + rng.normal(scale=1.0, size=n)
    df = pd.DataFrame({"y": y, "x1": x1})
    handle = _register(tool_ctx, df)

    result = REGISTRY.dispatch(
        tool_ctx, "jackknife_regression", {"dataset_id": handle.dataset_id, "target": "y", "predictors": ["x1"]}
    )
    assert {row[0] for row in result.tables[0].rows} == {"const", "x1"}


def test_jackknife_regression_drops_missing_values_cleanly(tool_ctx, rng):
    n = 80
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 5 + 2 * x1 + 3 * x2 + rng.normal(scale=1.0, size=n)
    y[10] = np.nan
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    handle = _register(tool_ctx, df)

    result = REGISTRY.dispatch(
        tool_ctx, "jackknife_regression", {"dataset_id": handle.dataset_id, "target": "y", "predictors": ["x1", "x2"]}
    )
    assert result.sample_sizes["observations"] == n - 1
    assert not any(np.isnan(row[1]) for row in result.tables[0].rows)


def test_jackknife_regression_deduplicates_predictors(tool_ctx, rng):
    n = 60
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 5 + 2 * x1 + 3 * x2 + rng.normal(scale=1.0, size=n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    handle = _register(tool_ctx, df)

    result = REGISTRY.dispatch(
        tool_ctx, "jackknife_regression",
        {"dataset_id": handle.dataset_id, "target": "y", "predictors": ["x1", "x1", "x2"]},
    )
    assert [row[0] for row in result.tables[0].rows] == ["const", "x1", "x2"]


def test_jackknife_regression_reports_vif_and_autocorrelation_checks(tool_ctx, rng):
    n = 100
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 5 + 2 * x1 + 3 * x2 + rng.normal(scale=1.0, size=n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    handle = _register(tool_ctx, df)

    result = REGISTRY.dispatch(
        tool_ctx, "jackknife_regression", {"dataset_id": handle.dataset_id, "target": "y", "predictors": ["x1", "x2"]}
    )
    names = {a.name for a in result.assumptions}
    assert "jackknife_independence" in names
    assert "multicollinearity_x1" in names
    assert "multicollinearity_x2" in names


def test_jackknife_regression_rejects_non_numeric_predictor(tool_ctx, rng):
    df = pd.DataFrame({"y": rng.normal(size=20), "cat": ["a", "b"] * 10})
    handle = _register(tool_ctx, df)

    with pytest.raises(ValueError, match="must be numeric"):
        REGISTRY.dispatch(
            tool_ctx, "jackknife_regression", {"dataset_id": handle.dataset_id, "target": "y", "predictors": ["cat"]}
        )


def test_jackknife_regression_rejects_empty_predictors(tool_ctx, rng):
    df = pd.DataFrame({"y": rng.normal(size=20)})
    handle = _register(tool_ctx, df)

    with pytest.raises(ValueError, match="must not be empty"):
        REGISTRY.dispatch(
            tool_ctx, "jackknife_regression", {"dataset_id": handle.dataset_id, "target": "y", "predictors": []}
        )


def test_jackknife_regression_rejects_excessive_sample_size(tool_ctx, rng):
    n = 6000
    x1 = rng.normal(size=n)
    y = 5 + 2 * x1 + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "x1": x1})
    handle = _register(tool_ctx, df)

    with pytest.raises(ValueError, match="safety cap"):
        REGISTRY.dispatch(
            tool_ctx, "jackknife_regression", {"dataset_id": handle.dataset_id, "target": "y", "predictors": ["x1"]}
        )
