import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy import stats as scipy_stats

from autostats.core.tools.base import BaseTool, ToolContext, ToolInput
from autostats.core.tools.preprocessing.shared import register_derived, require_columns
from autostats.core.tools.registry import REGISTRY

# Common, interpretable lambda values preferred over an arbitrary continuous MLE when
# they're statistically justified -- i.e. they fall inside the 95% CI.
_INTERPRETABLE_LAMBDAS = [-2, -1, -0.5, 0, 0.5, 1, 2]


class BoxCoxTransformInput(ToolInput):
    column: str
    alpha: float = 0.05


@REGISTRY.register
class BoxCoxTransformTool(BaseTool):
    name = "box_cox_transform"
    description = (
        "Box-Cox transform a strictly positive numeric column to correct non-normality/"
        "heteroscedasticity -- another way to address OLS inadequacy, alongside "
        "weighted_linear_regression/irls_regression. Finds the maximum-likelihood lambda "
        "and its 95% confidence interval by sweeping the profile log-likelihood; if a "
        "well-known, interpretable lambda (e.g. 0 for log, 0.5 for sqrt, 1 for no "
        "transform, -1 for inverse) falls within that interval, applies that one instead "
        "of the raw MLE estimate for interpretability. Registers the transformed column "
        "as a new dataset."
    )
    category = "preprocessing"
    input_model = BoxCoxTransformInput

    def run(self, ctx: ToolContext, params: BoxCoxTransformInput) -> BaseModel:
        df = ctx.data_manager.load(params.dataset_id)
        require_columns(df, [params.column])
        if not pd.api.types.is_numeric_dtype(df[params.column]):
            raise ValueError(f"'{params.column}' must be numeric for a Box-Cox transform")
        series = df[params.column].dropna()
        if (series <= 0).any():
            raise ValueError(
                f"Box-Cox requires strictly positive values; '{params.column}' has "
                f"{int((series <= 0).sum())} value(s) <= 0."
            )

        _, lambda_mle, (ci_low, ci_high) = scipy_stats.boxcox(series.to_numpy(), alpha=params.alpha)

        chosen_lambda = lambda_mle
        if candidates := [
            c for c in _INTERPRETABLE_LAMBDAS if ci_low <= c <= ci_high
        ]:
            chosen_lambda = min(candidates, key=lambda c: abs(c - lambda_mle))
            note = (
                f"lambda={chosen_lambda:g} falls within the {round((1 - params.alpha) * 100)}% CI "
                f"[{ci_low:.3f}, {ci_high:.3f}] for the MLE estimate ({lambda_mle:.3f}) and was used "
                "in place of the raw MLE for interpretability."
            )
        else:
            note = (
                f"No common interpretable lambda falls within the {round((1 - params.alpha) * 100)}% CI "
                f"[{ci_low:.3f}, {ci_high:.3f}]; using the MLE estimate ({lambda_mle:.3f}) directly."
            )

        new_df = df.copy()
        new_df[params.column] = self._transform(df[params.column], chosen_lambda)

        return register_derived(
            ctx,
            params.dataset_id,
            new_df,
            method="box_cox",
            extra_metadata={
                "transformed_column": params.column,
                "lambda": chosen_lambda,
                "lambda_mle": float(lambda_mle),
                "lambda_ci": [float(ci_low), float(ci_high)],
            },
            warnings=[note],
        )

    @staticmethod
    def _transform(column: pd.Series, lam: float) -> pd.Series:
        return np.log(column) if lam == 0 else (column**lam - 1) / lam
