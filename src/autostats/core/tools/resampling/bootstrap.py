from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel

from autostats.core.schemas.stat_result import StatResult
from autostats.core.tools.base import BaseTool, ToolContext, ToolInput
from autostats.core.tools.registry import REGISTRY
from autostats.core.tools.resampling.shared import STAT_FUNCS
from autostats.core.tools.stats.assumptions import check_sample_size

# n_boot itself is bounded (typical practice needs at most a few thousand replicates
# for a stable SE/CI; beyond that has diminishing returns) -- but the real cost driver
# is n_boot * n (each replicate resamples the whole column), which the dataset-size
# check alone doesn't bound. Timed directly: n=20,000, n_boot=5,000 (1e8 total draws)
# took ~0.36s: 1e9 total draws keeps the worst case under ~4s.
_MAX_N_BOOT = 20_000
_MAX_TOTAL_DRAWS = 1_000_000_000


class BootstrapInput(ToolInput):
    column: str
    statistic: Literal["mean", "median", "std", "var"] = "mean"
    n_boot: int = 1000
    random_state: int | None = None


@REGISTRY.register
class BootstrapTool(BaseTool):
    name = "bootstrap"
    description = (
        "Nonparametric bootstrap: estimate the sampling distribution of a statistic (mean, "
        "median, std, or var) of a numeric column by resampling the data with replacement "
        "'n_boot' times and recomputing. Reports the actual point estimate (computed on the "
        "full sample), the bootstrap bias and bias-corrected estimates, the bootstrap standard "
        "error, and a 95% percentile confidence interval. Unlike the jackknife, the bootstrap "
        "is valid for non-smooth statistics like the median -- use this instead of jackknife "
        "in that case. Set 'random_state' for a reproducible result."
    )
    category = "resampling"
    input_model = BootstrapInput

    def run(self, ctx: ToolContext, params: BootstrapInput) -> BaseModel:
        if params.n_boot <= 0:
            raise ValueError("n_boot must be a positive integer.")
        if params.n_boot > _MAX_N_BOOT:
            raise ValueError(f"n_boot must be <= {_MAX_N_BOOT:,} to avoid excessive computation time.")

        df = ctx.data_manager.load(params.dataset_id)
        if params.column not in df.columns:
            raise ValueError(f"Column '{params.column}' not found in dataset.")
        if not pd.api.types.is_numeric_dtype(df[params.column]):
            raise ValueError(f"'{params.column}' must be numeric for bootstrap resampling.")

        column_data = df[params.column].dropna().to_numpy()
        n = len(column_data)
        if n < 2:
            raise ValueError("Bootstrap resampling requires at least 2 non-missing values.")

        total_draws = n * params.n_boot
        if total_draws > _MAX_TOTAL_DRAWS:
            raise ValueError(
                f"Bootstrapping {n:,} rows with n_boot={params.n_boot:,} would require "
                f"{total_draws:,} total resampled draws, which exceeds the {_MAX_TOTAL_DRAWS:,} "
                "safety cap. Reduce n_boot or use a smaller sample (e.g. via train_test_split)."
            )

        stat_func = STAT_FUNCS[params.statistic]
        rng = np.random.default_rng(params.random_state)

        # The actual point estimate, computed once on the full sample -- see JackknifeTool
        # for the same distinction between this and the mean of the replicates below.
        point_estimate = float(stat_func(column_data))
        bootstrap_estimates = np.array(
            [stat_func(rng.choice(column_data, size=n, replace=True)) for _ in range(params.n_boot)]
        )
        replicate_mean = float(np.mean(bootstrap_estimates))

        # Standard bootstrap bias/SE estimators (Efron & Tibshirani, "An Introduction to the
        # Bootstrap", ch. 10) -- note this is a *different* bias formula than the jackknife's:
        # no (n-1) inflation factor, since bootstrap replicates directly approximate the
        # sampling distribution rather than a leave-one-out linear approximation of it.
        bias = replicate_mean - point_estimate
        bias_corrected_estimate = point_estimate - bias
        standard_error = float(np.std(bootstrap_estimates, ddof=1))
        ci_lower = float(np.percentile(bootstrap_estimates, 2.5))
        ci_upper = float(np.percentile(bootstrap_estimates, 97.5))

        size_check = check_sample_size(n, 10, params.column)

        return StatResult(
            tool_name=self.name,
            test_name=f"bootstrap_{params.statistic}",
            confidence_interval=(ci_lower, ci_upper),
            confidence_level=0.95,
            sample_sizes={params.column: n},
            effect_size={
                "point_estimate": point_estimate,
                "bias": bias,
                "bias_corrected_estimate": bias_corrected_estimate,
                "standard_error": standard_error,
            },
            assumptions=[size_check],
            assumptions_met=size_check.passed,
            interpretation=(
                f"Bootstrap estimate of the {params.statistic} of '{params.column}' "
                f"({params.n_boot:,} resamples): point estimate={point_estimate:.4g}, "
                f"SE={standard_error:.4g}, bias={bias:.4g}, "
                f"bias-corrected estimate={bias_corrected_estimate:.4g}, "
                f"95% CI=[{ci_lower:.4g}, {ci_upper:.4g}]."
            ),
            warnings=[] if size_check.passed else [f"Sample size ({n}) is small for a bootstrap estimate; see assumptions."],
            # Deliberately not returning all n_boot replicate values -- same rationale as
            # JackknifeTool: that would dump a potentially huge array into LLM context via
            # the tool-call output. A small summary is enough to sanity-check the replicates.
            raw_summary={
                "bootstrap_replicate_min": float(np.min(bootstrap_estimates)),
                "bootstrap_replicate_max": float(np.max(bootstrap_estimates)),
            },
        )