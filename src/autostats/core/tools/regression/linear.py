import numpy as np
import statsmodels.api as sm
from pydantic import BaseModel
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

from autostats.core.schemas.stat_result import AssumptionCheck, StatResult, TableArtifact
from autostats.core.tools.base import BaseTool, ToolContext, ToolInput
from autostats.core.tools.registry import REGISTRY
from autostats.core.tools.stats.assumptions import check_sample_size, check_vif


class LinearRegressionInput(ToolInput):
    target: str
    predictors: list[str]
    alpha: float = 0.05


@REGISTRY.register
class LinearRegressionTool(BaseTool):
    name = "linear_regression"
    description = "Fit an OLS linear regression of a numeric target on one or more predictors, with diagnostics."
    category = "regression"
    input_model = LinearRegressionInput

    # Overridden by subclasses (e.g. WeightedLinearRegressionTool) that share this
    # implementation but fit a different statsmodels model.
    _table_id_prefix = "ols_coef"
    _label = "OLS regression"

    def run(self, ctx: ToolContext, params: LinearRegressionInput) -> BaseModel:
        data = self._load_data(ctx, params)
        X = sm.add_constant(data[params.predictors])
        y = data[params.target]
        model = self._fit(X, y, data, params)
        return self._build_result(model, data, params)

    def _load_data(self, ctx: ToolContext, params: LinearRegressionInput):
        df = ctx.data_manager.load(params.dataset_id)
        return df[[params.target, *params.predictors]].dropna()

    def _fit(self, X, y, data, params: LinearRegressionInput):
        return sm.OLS(y, X).fit()

    def _extra_interpretation(self, model) -> str:
        """Optional trailing note appended to the interpretation; empty by default."""
        return ""

    def _extra_raw_summary(self, model) -> dict:
        """Optional extra keys merged into raw_summary; empty by default."""
        return {}

    def _build_result(self, model, data, params: LinearRegressionInput) -> StatResult:
        coef_table = TableArtifact(
            table_id=f"{self._table_id_prefix}_{params.dataset_id}",
            title=f"{self._label} coefficients",
            columns=["term", "coef", "std_err", "t", "p_value", "ci_lower", "ci_upper"],
            rows=[
                [term, round(model.params[term], 4), round(model.bse[term], 4), round(model.tvalues[term], 4),
                 round(model.pvalues[term], 4), *[round(v, 4) for v in model.conf_int().loc[term]]]
                for term in model.params.index
            ],
        )

        vif_checks = check_vif(data[params.predictors])
        n_check = check_sample_size(len(data), 10 * len(params.predictors), "observations")
        # `wresid`/`resid_pearson` are the weighted (whitened) residuals -- defined on every
        # RegressionResults, OLS or WLS -- and equal the plain residuals when weights are
        # uniform. Using them (rather than the raw, unweighted `resid`) means these
        # diagnostics are unchanged for ordinary regression but actually test the
        # *weighted* model's assumptions for the WLS/IRLS subclasses, instead of
        # re-flagging the heteroscedasticity the weights were meant to correct.
        dw = float(durbin_watson(model.wresid))
        bp_stat, bp_p, _, _ = het_breuschpagan(model.resid_pearson, model.model.exog)

        assumptions = [
            *vif_checks,
            n_check,
            AssumptionCheck(
                name="durbin_watson", passed=1.5 <= dw <= 2.5, statistic=dw,
                detail=f"Durbin-Watson statistic is {dw:.3f} (close to 2 indicates little autocorrelation).",
            ),
            AssumptionCheck(
                name="homoscedasticity_breusch_pagan", passed=bp_p > 0.05, statistic=float(bp_stat),
                p_value=float(bp_p), detail=f"Breusch-Pagan test p={bp_p:.4f} "
                f"({'no evidence of heteroscedasticity' if bp_p > 0.05 else 'evidence of heteroscedasticity'}).",
            ),
        ]
        assumptions_met = all(c.passed for c in vif_checks) and bp_p > 0.05

        sig_terms = [t for t in params.predictors if model.pvalues[t] < params.alpha]
        return StatResult(
            tool_name=self.name,
            test_name=self.name,
            statistic=float(model.fvalue),
            p_value=float(model.f_pvalue),
            degrees_of_freedom=(float(model.df_model), float(model.df_resid)),
            effect_size={"r_squared": float(model.rsquared), "adj_r_squared": float(model.rsquared_adj)},
            sample_sizes={"observations": len(data)},
            assumptions=assumptions,
            assumptions_met=assumptions_met,
            tables=[coef_table],
            interpretation=(
                f"{self._label} of '{params.target}' on {params.predictors}: R-squared={model.rsquared:.3f}, "
                f"F={model.fvalue:.3f} (p={model.f_pvalue:.4f}). "
                f"Significant predictors (alpha={params.alpha}): {sig_terms or 'none'}."
                f"{self._extra_interpretation(model)}"
            ),
            warnings=[] if assumptions_met else ["Regression diagnostics flagged potential multicollinearity or heteroscedasticity; see assumptions."],
            raw_summary={
                "params": model.params.to_dict(),
                "pvalues": model.pvalues.to_dict(),
                **self._extra_raw_summary(model),
            },
        )


class WeightedLinearRegressionInput(LinearRegressionInput):
    weights_column: str


@REGISTRY.register
class WeightedLinearRegressionTool(LinearRegressionTool):
    name = "weighted_linear_regression"
    description = (
        "Fit a weighted least squares (WLS) regression of a numeric target on one or more "
        "predictors, using a column of known weights (e.g. inverse-variance/precision weights) "
        "to correct for heteroscedasticity. Reports the same diagnostics as linear_regression. "
        "If you don't already have a known set of weights, use irls_regression instead."
    )
    category = "regression"
    input_model = WeightedLinearRegressionInput

    _table_id_prefix = "wls_coef"
    _label = "Weighted least squares (WLS) regression"

    def _load_data(self, ctx: ToolContext, params: WeightedLinearRegressionInput):
        df = ctx.data_manager.load(params.dataset_id)
        return df[[params.target, *params.predictors, params.weights_column]].dropna()

    def _fit(self, X, y, data, params: WeightedLinearRegressionInput):
        weights = data[params.weights_column]
        if not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError(f"'{params.weights_column}' must contain only finite, positive values")
        return sm.WLS(y, X, weights=weights).fit()


class IRLSInput(LinearRegressionInput):
    max_iterations: int = 20
    tol: float = 1e-3


@REGISTRY.register
class IterativelyReweightedLeastSquaresTool(LinearRegressionTool):
    name = "irls_regression"
    description = (
        "Fit a regression that automatically estimates and corrects for heteroscedasticity when you "
        "do NOT already have a known set of weights (use weighted_linear_regression instead if you do). "
        "Iteratively fits OLS, estimates how residual variance depends on the predictors via an "
        "auxiliary regression, derives weights from that estimate, and refits by weighted least squares "
        "-- repeating until the weights stabilize or 'max_iterations' is reached."
    )
    category = "regression"
    input_model = IRLSInput

    _table_id_prefix = "irls_coef"
    _label = "Iteratively reweighted least squares (IRLS) regression"

    def _fit(self, X, y, data, params: IRLSInput):
        if params.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")

        model = sm.OLS(y, X).fit()
        weights = np.ones(len(y))
        converged = False
        iterations_used = 0

        for iterations_used in range(1, params.max_iterations + 1):
            # Estimate how residual variance depends on the predictors by regressing
            # log(residual^2) on X (pooling information across observations, unlike using
            # each residual as its own variance estimate), then exponentiate to recover
            # positive variance estimates and invert them into weights.
            log_sq_resid = np.log(model.resid**2 + 1e-8)
            variance_model = sm.OLS(log_sq_resid, X).fit()
            new_weights = 1 / np.exp(variance_model.fittedvalues)
            if not np.isfinite(new_weights).all():
                raise ValueError(
                    "IRLS produced non-finite weights; the auxiliary variance regression may be "
                    "unstable for this data."
                )

            model = sm.WLS(y, X, weights=new_weights).fit()

            if np.allclose(new_weights, weights, rtol=params.tol):
                weights = new_weights
                converged = True
                break
            weights = new_weights

        # Stashed on the fitted result (not `self`) so this stays safe under concurrent
        # calls -- `_extra_interpretation`/`_extra_raw_summary` read it back per-call.
        model.irls_iterations = iterations_used
        model.irls_converged = converged
        return model

    def _extra_interpretation(self, model) -> str:
        status = "converged" if model.irls_converged else "did not converge (reached the iteration cap)"
        return f" IRLS {status} after {model.irls_iterations} iteration(s) of reweighting."

    def _extra_raw_summary(self, model) -> dict:
        return {"irls_iterations": model.irls_iterations, "irls_converged": model.irls_converged}
