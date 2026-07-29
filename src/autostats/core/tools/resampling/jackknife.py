from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pydantic import BaseModel
from scipy.stats import norm
from statsmodels.stats.stattools import durbin_watson

from autostats.core.schemas.stat_result import AssumptionCheck, StatResult, TableArtifact
from autostats.core.tools.base import BaseTool, ToolContext, ToolInput
from autostats.core.tools.registry import REGISTRY
from autostats.core.tools.resampling.shared import STAT_FUNCS
from autostats.core.tools.stats.assumptions import check_sample_size, check_vif

_Z_95 = 1.96
# Each leave-one-out replicate recomputes the statistic from scratch (O(n) per
# replicate, O(n log n) for median), so the whole procedure is at least O(n^2) --
# 20,000 rows already takes ~1s for the median statistic in testing, with no
# closed-form shortcut available for median. Fail fast past that rather than
# hang a tool call.
_MAX_JACKKNIFE_N = 20_000
# Each replicate here refits a full OLS model (a much more expensive operation than
# a mean/median/std/var call) -- timed directly: ~0.5-0.7ms/replicate with 3
# predictors, and it gets slower with more predictors, so this needs its own,
# separately-calibrated (lower) cap rather than reusing _MAX_JACKKNIFE_N.
_MAX_JACKKNIFE_REGRESSION_N = 5_000


class JackknifeInput(ToolInput):
    column: str
    statistic: Literal["mean", "median", "std", "var"] = "mean"


@REGISTRY.register
class JackknifeTool(BaseTool):
    name = "jackknife"
    description = (
        "Jackknife resampling: estimate a statistic (mean, median, std, or var) of a numeric "
        "column, along with its bias and standard error, by leaving out one observation at a "
        "time and recomputing. Reports the actual point estimate (computed on the full sample), "
        "the jackknife bias and bias-corrected estimates, the jackknife standard error, and a "
        "95% confidence interval centered on the point estimate. Note: the jackknife's variance "
        "estimate is known to be inconsistent for the median specifically (a non-smooth "
        "statistic) -- that case is flagged as an unmet assumption, recommending a "
        "bootstrap-based estimate instead."
    )
    category = "resampling"
    input_model = JackknifeInput

    def run(self, ctx: ToolContext, params: JackknifeInput) -> BaseModel:
        df = ctx.data_manager.load(params.dataset_id)
        if params.column not in df.columns:
            raise ValueError(f"Column '{params.column}' not found in dataset.")
        if not pd.api.types.is_numeric_dtype(df[params.column]):
            raise ValueError(f"'{params.column}' must be numeric for jackknife resampling.")

        column_data = df[params.column].dropna().to_numpy()
        n = len(column_data)
        if n < 2:
            raise ValueError("Jackknife resampling requires at least 2 non-missing values.")
        if n > _MAX_JACKKNIFE_N:
            raise ValueError(
                f"Jackknife resampling on {n:,} rows would recompute the statistic {n:,} times, "
                f"which exceeds the {_MAX_JACKKNIFE_N:,}-row safety cap. Use a smaller sample "
                "(e.g. via train_test_split) or an alternative estimate instead."
            )

        stat_func = STAT_FUNCS[params.statistic]

        # The actual point estimate, computed once on the full sample -- distinct from
        # (and, for every statistic but the mean, numerically different from) the mean
        # of the leave-one-out replicates below.
        point_estimate = float(stat_func(column_data))
        jackknife_estimates = np.array([stat_func(np.delete(column_data, i)) for i in range(n)])
        replicate_mean = float(np.mean(jackknife_estimates))

        # Standard jackknife bias/variance estimators (Tukey; see e.g. Efron & Tibshirani,
        # "An Introduction to the Bootstrap", ch. 11). Note the variance estimator's
        # (n-1)/n factor -- omitting it (e.g. via a plain ddof=1 sample variance of the
        # replicates) understates the standard error by a factor of roughly (n-1).
        bias = (n - 1) * (replicate_mean - point_estimate)
        bias_corrected_estimate = point_estimate - bias
        variance_jack = ((n - 1) / n) * float(np.sum((jackknife_estimates - replicate_mean) ** 2))
        standard_error = float(np.sqrt(variance_jack))
        margin = _Z_95 * standard_error
        confidence_interval = (point_estimate - margin, point_estimate + margin)

        smoothness_check = AssumptionCheck(
            name="jackknife_smoothness",
            passed=params.statistic != "median",
            detail=(
                "The jackknife variance estimate is known to be inconsistent for the median "
                "(a non-smooth statistic); treat this standard error and confidence interval "
                "as unreliable."
                if params.statistic == "median" else
                f"The jackknife variance estimate is asymptotically consistent for the "
                f"{params.statistic}, a smooth statistic."
            ),
        )
        size_check = check_sample_size(n, 10, params.column)
        assumptions_met = smoothness_check.passed and size_check.passed

        warnings: list[str] = []
        if not smoothness_check.passed:
            warnings.append(
                "The jackknife standard error/CI is known to be inconsistent for the median; "
                "treat this interval as unreliable and consider a bootstrap-based estimate instead."
            )
        if not size_check.passed:
            warnings.append(
                f"Sample size ({n}) is small for a jackknife estimate; see assumptions."
            )

        return StatResult(
            tool_name=self.name,
            test_name=f"jackknife_{params.statistic}",
            confidence_interval=confidence_interval,
            confidence_level=0.95,
            sample_sizes={params.column: n},
            effect_size={
                "point_estimate": point_estimate,
                "bias": bias,
                "bias_corrected_estimate": bias_corrected_estimate,
                "standard_error": standard_error,
            },
            assumptions=[smoothness_check, size_check],
            assumptions_met=assumptions_met,
            recommended_alternative=None if smoothness_check.passed else "bootstrap",
            interpretation=(
                f"Jackknife estimate of the {params.statistic} of '{params.column}': "
                f"point estimate={point_estimate:.4g}, SE={standard_error:.4g}, bias={bias:.4g}, "
                f"bias-corrected estimate={bias_corrected_estimate:.4g}, "
                f"95% CI=[{confidence_interval[0]:.4g}, {confidence_interval[1]:.4g}]."
            ),
            warnings=warnings,
            # Deliberately not returning all n leave-one-out replicates -- unlike every
            # other tool in this codebase, that would dump a potentially huge array
            # straight into the LLM's context via the tool-call output. A small summary
            # is enough to sanity-check the replicates without that cost.
            raw_summary={
                "jackknife_replicate_min": float(np.min(jackknife_estimates)),
                "jackknife_replicate_max": float(np.max(jackknife_estimates)),
            },
        )

class JackknifeRegressionInput(ToolInput):
    target: str
    predictors: list[str]


@REGISTRY.register
class JackknifeRegressionTool(BaseTool):
    name = "jackknife_regression"
    description = (
        "Jackknife regression: an alternative to OLS's own standard errors for a linear "
        "regression's coefficients, computed by leaving out one observation at a time, "
        "refitting, and using the spread of the resulting leave-one-out coefficient estimates. "
        "Reports the actual coefficients (from the full-sample fit), the jackknife bias and "
        "bias-corrected estimates, jackknife standard errors, t-statistics, p-values, and 95% "
        "CIs per term. This delete-1 case-resampling approach is a legitimate alternative when "
        "OLS's normality or homoscedasticity assumptions are violated, but it is NOT valid when "
        "errors are autocorrelated/dependent (it assumes approximately independent "
        "observations) -- that case is flagged as an unmet assumption rather than silently "
        "trusted."
    )
    category = "resampling"
    input_model = JackknifeRegressionInput

    def run(self, ctx: ToolContext, params: JackknifeRegressionInput) -> BaseModel:
        df = ctx.data_manager.load(params.dataset_id)
        predictors = list(dict.fromkeys(params.predictors))
        if not predictors:
            raise ValueError("predictors must not be empty")

        for col in [params.target, *predictors]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in dataset.")
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"'{col}' must be numeric for jackknife regression.")

        data = df[[params.target, *predictors]].dropna()
        n = len(data)
        # OLS needs at least len(predictors)+1 rows to be identified, and leave-one-out
        # needs n-1 rows to still meet that bar.
        min_n = len(predictors) + 2
        if n < min_n:
            raise ValueError(
                f"Jackknife regression with {len(predictors)} predictor(s) requires at least "
                f"{min_n} non-missing rows; only {n} are available."
            )
        if n > _MAX_JACKKNIFE_REGRESSION_N:
            raise ValueError(
                f"Jackknife regression on {n:,} rows would refit the model {n:,} times, which "
                f"exceeds the {_MAX_JACKKNIFE_REGRESSION_N:,}-row safety cap. Use a smaller "
                "sample (e.g. via train_test_split) or an alternative estimate instead."
            )

        y = data[params.target]
        X = sm.add_constant(data[predictors])
        full_model = sm.OLS(y, X).fit()
        # The actual point estimates, from the full-sample fit -- distinct from the mean of
        # the leave-one-out replicates below (see JackknifeTool's `point_estimate` for the
        # same distinction; the gap is far smaller for OLS coefficients than for e.g. a
        # median, since coefficients are a much smoother function of the data, but it's
        # still the conceptually correct point estimate to report).
        point_estimates = full_model.params

        jackknife_coefficients = np.zeros((n, len(point_estimates)))
        for i in range(n):
            X_loo = X.drop(X.index[i])
            y_loo = y.drop(y.index[i])
            jackknife_coefficients[i] = sm.OLS(y_loo, X_loo).fit().params.values

        replicate_means = jackknife_coefficients.mean(axis=0)
        # Same jackknife bias/variance estimators as JackknifeTool, applied per coefficient.
        bias = (n - 1) * (replicate_means - point_estimates.values)
        bias_corrected = point_estimates.values - bias
        standard_error = jackknife_coefficients.std(axis=0, ddof=1) * np.sqrt((n - 1) ** 2 / n)
        t_stat = point_estimates.values / standard_error
        p_value = 2 * (1 - norm.cdf(np.abs(t_stat)))
        margin = _Z_95 * standard_error
        ci_lower = point_estimates.values - margin
        ci_upper = point_estimates.values + margin

        vif_checks = check_vif(data[predictors])
        size_check = check_sample_size(n, 10 * len(predictors), "observations")
        dw = float(durbin_watson(full_model.resid))
        autocorrelation_check = AssumptionCheck(
            name="jackknife_independence",
            passed=1.5 <= dw <= 2.5,
            statistic=dw,
            detail=(
                f"Durbin-Watson statistic is {dw:.3f}. Delete-1 case-resampling jackknife "
                "assumes approximately independent observations; a value far from 2 indicates "
                "autocorrelated residuals, which this method does NOT correct for (unlike "
                "heteroscedasticity or non-normality, which it is robust to) -- a block-"
                "resampling or time-series-aware method would be needed instead."
            ),
        )
        assumptions = [*vif_checks, size_check, autocorrelation_check]
        assumptions_met = all(c.passed for c in vif_checks) and size_check.passed and autocorrelation_check.passed

        coef_table = TableArtifact(
            table_id=f"jackknife_regression_coef_{params.dataset_id}",
            title="Jackknife regression coefficients",
            columns=["term", "coef", "jackknife_se", "t", "p_value", "ci_lower", "ci_upper", "bias", "bias_corrected"],
            rows=[
                [
                    term, round(point_estimates.iloc[i], 4), round(standard_error[i], 4), round(t_stat[i], 4),
                    round(p_value[i], 4), round(ci_lower[i], 4), round(ci_upper[i], 4),
                    round(bias[i], 4), round(bias_corrected[i], 4),
                ]
                for i, term in enumerate(point_estimates.index)
            ],
        )

        sig_terms = [t for i, t in enumerate(point_estimates.index) if t != "const" and p_value[i] < 0.05]
        return StatResult(
            tool_name=self.name,
            test_name="jackknife_regression",
            sample_sizes={"observations": n},
            effect_size={"r_squared": float(full_model.rsquared)},
            assumptions=assumptions,
            assumptions_met=assumptions_met,
            tables=[coef_table],
            interpretation=(
                f"Jackknife regression of '{params.target}' on {predictors}: significant "
                f"predictors (jackknife p<0.05): {sig_terms or 'none'}. See the coefficient "
                "table for per-term estimates, jackknife standard errors, and 95% CIs."
            ),
            warnings=[] if assumptions_met else ["Jackknife regression assumption(s) flagged; see assumptions for details."],
            raw_summary={
                "params": point_estimates.to_dict(),
                "jackknife_standard_errors": dict(zip(point_estimates.index, standard_error.tolist())),
            },
        )