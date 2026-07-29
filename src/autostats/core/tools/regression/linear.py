from typing import Literal
import itertools
import math

import numpy as np
import pandas as pd
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
        predictors = list(dict.fromkeys(params.predictors))
        X = sm.add_constant(data[predictors])
        y = data[params.target]
        model = self._fit(X, y, data, params)
        return self._build_result(model, data, params, predictors=predictors)

    def _load_data(self, ctx: ToolContext, params: LinearRegressionInput):
        df = ctx.data_manager.load(params.dataset_id)
        predictors = list(dict.fromkeys(params.predictors))
        return df[[params.target, *predictors]].dropna()

    def _fit(self, X, y, data, params: LinearRegressionInput):
        return sm.OLS(y, X).fit()

    def _extra_interpretation(self, model) -> str:
        """Optional trailing note appended to the interpretation; empty by default."""
        return ""

    def _extra_raw_summary(self, model) -> dict:
        """Optional extra keys merged into raw_summary; empty by default."""
        return {}

    def _build_result(self, model, data, params: LinearRegressionInput, predictors: list[str] | None = None) -> StatResult:
        # `predictors` lets a subclass whose final predictor set isn't known until after
        # fitting (e.g. ForwardSelectionTool, which searches over candidates) report on
        # the predictors actually used rather than `params.predictors` (which such a
        # subclass's input may not even have).
        predictors = params.predictors if predictors is None else predictors
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

        vif_checks = check_vif(data[predictors])
        n_check = check_sample_size(len(data), 10 * len(predictors), "observations")
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

        sig_terms = [t for t in predictors if model.pvalues[t] < params.alpha]
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
                f"{self._label} of '{params.target}' on {predictors}: R-squared={model.rsquared:.3f}, "
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
        predictors = list(dict.fromkeys(params.predictors))
        return df[[params.target, *predictors, params.weights_column]].dropna()

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


def _mallows_cp(model, mse_full: float, n: int) -> float:
    """Mallows' Cp = RSS_p/MSE_full - n + 2p, where p is the number of parameters
    (including the intercept) in `model`, and MSE_full is the residual mean-square
    error of the *full* model (all candidate predictors) -- an unbiased estimate of
    the true error variance against which every subset is judged. Lower is better;
    for the full model itself Cp always equals p exactly (a useful sanity check),
    and a well-specified subset scores at or below its own p."""
    p = len(model.params)
    return model.ssr / mse_full - n + 2 * p


def _mallows_cp_reference(fit_ols_fn, y: pd.Series, data: pd.DataFrame, candidates: list[str]) -> tuple[float, int]:
    """Fit the full model (all candidates) once, to get the MSE reference Mallow's Cp
    needs -- this is held fixed across every subset's Cp computation regardless of
    which subset is currently being scored, so it's computed here, once, up front."""
    full_model = fit_ols_fn(y, data[candidates])
    if full_model.mse_resid <= 0:
        raise ValueError(
            "Mallow's Cp requires the full model (all candidates) to have positive "
            "residual variance; this full model has none (a perfect fit)."
        )
    return full_model.mse_resid, len(data)


def _forward_selection_score(
    model, criterion: str, candidate: str, mse_full: float | None = None, n: int | None = None
) -> float:
    if criterion == "aic":
        return model.aic
    if criterion == "bic":
        return model.bic
    if criterion == "r_squared":
        return model.rsquared_adj
    if criterion == "p_value":
        return model.pvalues[candidate]
    if criterion == "mallows_cp":
        return _mallows_cp(model, mse_full, n)
    raise ValueError(f"Unknown criterion: {criterion}")  # pragma: no cover -- guarded by pydantic Literal

def _backward_elimination_score(
    current_model, trial_model, criterion: str, candidate: str, mse_full: float | None = None, n: int | None = None
) -> float:
    if criterion == "aic":
        return trial_model.aic
    if criterion == "bic":
        return trial_model.bic
    if criterion == "r_squared":
        return trial_model.rsquared_adj
    if criterion == "p_value":
        # The candidate's own significance BEFORE removal -- `trial_model` excludes it
        # entirely, so its p-value has to come from the current (still-full) model.
        return current_model.pvalues[candidate]
    if criterion == "mallows_cp":
        return _mallows_cp(trial_model, mse_full, n)
    raise ValueError(f"Unknown criterion: {criterion}")  # pragma: no cover -- guarded by pydantic Literal

def _forward_selection_is_better(score: float, best_score: float | None, criterion: str) -> bool:
    """Compare two candidates from the SAME step; used to pick that step's winner."""
    if best_score is None:
        return True
    return score > best_score if criterion == "r_squared" else score < best_score

def _backward_elimination_is_better(score: float, best_score: float | None, criterion: str) -> bool:
    """Compare two candidates from the SAME step; used to pick that step's loser (the
    predictor safest/most justified to remove)."""
    if best_score is None:
        return True
    if criterion in {"aic", "bic", "mallows_cp"}:
        return score < best_score  # lower resulting AIC/BIC/Cp after removal -> safer to remove
    # r_squared: higher resulting adjusted R-squared after removal -> safer to remove.
    # p_value: higher (less significant) p-value in the current model -> safer to remove.
    return score > best_score

def _forward_selection_improves(
    best_score: float, current_score: float | None, criterion: str, p_value_threshold: float
) -> bool:
    """Whether adding the step's winning candidate is actually worth it."""
    if criterion == "p_value":
        return best_score < p_value_threshold
    if criterion == "r_squared":
        return current_score is None or best_score > current_score
    return current_score is None or best_score < current_score  # aic/bic/mallows_cp: lower is better

def _backward_elimination_improves(
    best_score: float, current_score: float | None, criterion: str, p_value_threshold: float
) -> bool:
    """Whether removing the step's losing candidate is actually worth it."""
    if criterion == "p_value":
        return best_score > p_value_threshold
    if criterion == "r_squared":
        return current_score is None or best_score > current_score
    return current_score is None or best_score < current_score  # aic/bic/mallows_cp: lower is better


class ForwardSelectionInput(ToolInput):
    target: str
    candidate_predictors: list[str]
    criterion: Literal["aic", "bic", "r_squared", "p_value", "mallows_cp"] = "aic"
    p_value_threshold: float = 0.05
    alpha: float = 0.05
    max_predictors: int | None = None


@REGISTRY.register
class ForwardSelectionTool(LinearRegressionTool):
    name = "forward_selection"
    description = (
        "Forward stepwise selection for a linear regression: starting from no predictors, "
        "repeatedly adds whichever remaining candidate most improves the model under "
        "'criterion', stopping once no remaining candidate improves it further. Criteria: "
        "'aic'/'bic' add the candidate giving the lowest resulting AIC/BIC; 'r_squared' uses "
        "*adjusted* R-squared (plain R-squared never decreases as predictors are added, so it "
        "gives no natural stopping point) and adds the candidate giving the highest resulting "
        "adjusted R-squared; 'p_value' adds the candidate whose own coefficient is most "
        "significant, provided it is below 'p_value_threshold'; 'mallows_cp' adds the candidate "
        "giving the lowest Mallow's Cp, computed against the residual variance of the full model "
        "(all candidate_predictors) -- a well-specified subset scores at or below its own number "
        "of parameters. Reports the same diagnostics as linear_regression for the final selected "
        "model, plus the order predictors were added in."
    )
    category = "regression"
    input_model = ForwardSelectionInput

    _table_id_prefix = "forward_selection_coef"
    _label = "Forward-selected OLS regression"

    def _load_data(self, ctx: ToolContext, params: ForwardSelectionInput):
        df = ctx.data_manager.load(params.dataset_id)
        candidates = list(dict.fromkeys(params.candidate_predictors))
        return df[[params.target, *candidates]].dropna()

    def run(self, ctx: ToolContext, params: ForwardSelectionInput) -> BaseModel:
        if not params.candidate_predictors:
            raise ValueError("candidate_predictors must not be empty")
        if params.max_predictors is not None and params.max_predictors < 1:
            raise ValueError("max_predictors must be >= 1")

        data = self._load_data(ctx, params)
        y = data[params.target]
        # De-duplicate while preserving order, in case the model lists a candidate twice.
        remaining = list(dict.fromkeys(params.candidate_predictors))
        max_steps = params.max_predictors if params.max_predictors is not None else len(remaining)

        mse_full = n = None
        if params.criterion == "mallows_cp":
            mse_full, n = _mallows_cp_reference(self._fit_ols, y, data, remaining)

        model = self._fit_ols(y, data[[]])
        current_score = (
            None if params.criterion == "p_value"
            else _forward_selection_score(model, params.criterion, None, mse_full, n)
        )
        selected: list[str] = []
        steps: list[dict] = []

        while remaining and len(selected) < max_steps:
            best_candidate, best_score, best_model = None, None, None
            for candidate in remaining:
                trial_model = self._fit_ols(y, data[[*selected, candidate]])
                score = _forward_selection_score(trial_model, params.criterion, candidate, mse_full, n)
                if _forward_selection_is_better(score, best_score, params.criterion):
                    best_candidate, best_score, best_model = candidate, score, trial_model

            if not _forward_selection_improves(best_score, current_score, params.criterion, params.p_value_threshold):
                break

            selected.append(best_candidate)
            remaining.remove(best_candidate)
            current_score = best_score
            model = best_model
            steps.append({"step": len(selected), "added": best_candidate, params.criterion: best_score})

        if not selected:
            raise ValueError(
                f"Forward selection under criterion '{params.criterion}' did not add any of "
                f"{params.candidate_predictors} -- none improved on the intercept-only baseline."
            )

        model.forward_selection_criterion = params.criterion
        model.forward_selection_steps = steps
        return self._build_result(model, data, params, predictors=selected)

    @staticmethod
    def _fit_ols(y: pd.Series, predictors_df: pd.DataFrame):
        X = sm.add_constant(predictors_df, has_constant="add")
        return sm.OLS(y, X).fit()

    def _extra_interpretation(self, model) -> str:
        order = ", ".join(f"{s['added']}" for s in model.forward_selection_steps)
        return f" Forward selection ({model.forward_selection_criterion}) added, in order: {order}."

    def _extra_raw_summary(self, model) -> dict:
        return {"criterion": model.forward_selection_criterion, "steps": model.forward_selection_steps}

class BackwardEliminationInput(ToolInput):
    target: str
    predictors: list[str]
    criterion: Literal["aic", "bic", "r_squared", "p_value", "mallows_cp"] = "aic"
    p_value_threshold: float = 0.05
    alpha: float = 0.05

@REGISTRY.register
class BackwardEliminationTool(LinearRegressionTool):
    name = "backward_elimination"
    description = (
        "Backward stepwise elimination for a linear regression: starting from all predictors, "
        "repeatedly removes whichever remaining predictor least worsens the model under "
        "'criterion', stopping once removing any remaining predictor would worsen it. Criteria: "
        "'aic'/'bic' remove the predictor giving the lowest resulting AIC/BIC; 'r_squared' uses "
        "*adjusted* R-squared (plain R-squared never decreases as predictors are added, so it "
        "gives no natural stopping point) and removes the predictor giving the highest resulting "
        "adjusted R-squared; 'p_value' removes the predictor whose own coefficient is least "
        "significant, provided it is above 'p_value_threshold'; 'mallows_cp' removes the predictor "
        "giving the lowest Mallow's Cp, computed against the residual variance of the full model "
        "(all predictors) -- a well-specified subset scores at or below its own number of "
        "parameters. Reports the same diagnostics as linear_regression for the final selected "
        "model, plus the order predictors were removed in."
    )
    category = "regression"
    input_model = BackwardEliminationInput

    _table_id_prefix = "backward_elimination_coef"
    _label = "Backward-eliminated OLS regression"

    def _load_data(self, ctx: ToolContext, params: BackwardEliminationInput):
        df = ctx.data_manager.load(params.dataset_id)
        predictors = list(dict.fromkeys(params.predictors))
        return df[[params.target, *predictors]].dropna()

    def run(self, ctx: ToolContext, params: BackwardEliminationInput) -> BaseModel:
        if not params.predictors:
            raise ValueError("predictors must not be empty")

        data = self._load_data(ctx, params)
        y = data[params.target]
        remaining = list(dict.fromkeys(params.predictors))
        model = self._fit_ols(y, data[remaining])

        # The initial full-predictor fit above already *is* the full model Mallow's Cp
        # needs for its variance reference -- no separate fit required here, unlike
        # forward_selection/best_subset_selection which don't otherwise fit one.
        mse_full = n = None
        if params.criterion == "mallows_cp":
            if model.mse_resid <= 0:
                raise ValueError(
                    "Mallow's Cp requires the full model (all predictors) to have positive "
                    "residual variance; this full model has none (a perfect fit)."
                )
            mse_full, n = model.mse_resid, len(data)

        current_score = (
            None if params.criterion == "p_value"
            else _backward_elimination_score(model, model, params.criterion, None, mse_full, n)
        )
        removed: list[str] = []
        steps: list[dict] = []

        while remaining:
            best_candidate, best_score, best_model = None, None, None
            for candidate in remaining:
                trial_model = self._fit_ols(y, data[[c for c in remaining if c != candidate]])
                score = _backward_elimination_score(model, trial_model, params.criterion, candidate, mse_full, n)
                if _backward_elimination_is_better(score, best_score, params.criterion):
                    best_candidate, best_score, best_model = candidate, score, trial_model

            if not _backward_elimination_improves(best_score, current_score, params.criterion, params.p_value_threshold):
                break

            remaining.remove(best_candidate)
            removed.append(best_candidate)
            current_score = best_score
            model = best_model
            steps.append({"step": len(removed), "removed": best_candidate, params.criterion: best_score})

        if not removed:
            raise ValueError(
                f"Backward elimination under criterion '{params.criterion}' did not remove any of "
                f"{params.predictors} -- none improved on the full-model baseline."
            )

        model.backward_elimination_criterion = params.criterion
        model.backward_elimination_steps = steps
        return self._build_result(model, data, params, predictors=remaining)

    @staticmethod
    def _fit_ols(y: pd.Series, predictors_df: pd.DataFrame):
        X = sm.add_constant(predictors_df, has_constant="add")
        return sm.OLS(y, X).fit()

    def _extra_interpretation(self, model) -> str:
        order = ", ".join(f"{s['removed']}" for s in model.backward_elimination_steps)
        return f" Backward elimination ({model.backward_elimination_criterion}) removed, in order: {order}."

    def _extra_raw_summary(self, model) -> dict:
        return {"criterion": model.backward_elimination_criterion, "steps": model.backward_elimination_steps}

# Best-subset search is exhaustive: sum_{k=1}^{max_predictors} C(len(candidates), k) models.
# That grows explosively (e.g. 18 candidates with max_predictors=9 is ~155k models, which
# took 87s in testing) -- fail fast with a clear message instead of hanging a tool call.
_MAX_SUBSETS_TO_EVALUATE = 50_000


def _total_subset_count(n_candidates: int, max_predictors: int) -> int:
    k_max = min(max_predictors, n_candidates)
    return sum(math.comb(n_candidates, k) for k in range(1, k_max + 1))


def _best_subset_score(
    model, criterion: str, predictors: list[str], mse_full: float | None = None, n: int | None = None
) -> float:
    if criterion == "aic":
        return model.aic
    if criterion == "bic":
        return model.bic
    if criterion == "r_squared":
        return model.rsquared_adj
    if criterion == "p_value":
        # The subset's least-significant predictor's p-value (the one "provided it is
        # below 'alpha'" in the tool description refers to) -- NOT a single named
        # candidate the way forward/backward selection score p_value, since best-subset
        # scores a whole subset at once, not one addition/removal.
        return max(model.pvalues[p] for p in predictors)
    if criterion == "mallows_cp":
        return _mallows_cp(model, mse_full, n)
    raise ValueError(f"Unknown criterion: {criterion}")  # pragma: no cover -- guarded by pydantic Literal


class BestSubsetSelectionInput(ToolInput):
    target: str
    candidate_predictors: list[str]
    criterion: Literal["aic", "bic", "r_squared", "p_value", "mallows_cp"] = "aic"
    max_predictors: int = 10
    alpha: float = 0.05

@REGISTRY.register
class BestSubsetSelectionTool(LinearRegressionTool):
    name = "best_subset_selection"
    description = (
        "Best subset selection for a linear regression: fits all possible models with up to "
        "'max_predictors' predictors from 'candidate_predictors', and selects the best model "
        "under 'criterion'. Criteria: 'aic'/'bic' select the model with the lowest AIC/BIC; "
        "'r_squared' selects the model with the highest *adjusted* R-squared; 'p_value' selects "
        "the model whose least significant predictor is most significant, provided it is below "
        "'alpha' (subsets with any less-significant predictor are excluded); 'mallows_cp' selects "
        "the model with the lowest Mallow's Cp, computed against the residual variance of the "
        "full model (all candidate_predictors) -- a well-specified subset scores at or below its "
        "own number of parameters. This is exhaustive and can be slow for a large candidate pool "
        "-- it fails fast with a clear error if the number of models to fit would be excessive; "
        "use forward_selection or backward_elimination instead in that case. Reports the same "
        "diagnostics as linear_regression for the final selected model, plus the predictors in "
        "that model."
    )
    category = "regression"
    input_model = BestSubsetSelectionInput

    _table_id_prefix = "best_subset_selection_coef"
    _label = "Best-subset-selected OLS regression"

    def _load_data(self, ctx: ToolContext, params: BestSubsetSelectionInput):
        df = ctx.data_manager.load(params.dataset_id)
        # De-duplicated here too (not just in run()'s search loop): otherwise a
        # repeated candidate name makes `data` itself carry a genuinely duplicate-named
        # column, and `data[["x1"]]` on that returns *both* matching columns instead of
        # one -- the actual root cause behind the "truth value of a Series is ambiguous"
        # crash, since a duplicate-column dataframe then propagates into the fitted
        # model's `pvalues` index too.
        candidates = list(dict.fromkeys(params.candidate_predictors))
        return df[[params.target, *candidates]].dropna()

    def run(self, ctx: ToolContext, params: BestSubsetSelectionInput) -> BaseModel:
        if not params.candidate_predictors:
            raise ValueError("candidate_predictors must not be empty")
        if params.max_predictors < 1:
            raise ValueError("max_predictors must be >= 1")

        # De-duplicate while preserving order, same as forward_selection/backward_elimination --
        # otherwise a repeated name produces a subset with a duplicate column label downstream.
        candidates = list(dict.fromkeys(params.candidate_predictors))

        total_subsets = _total_subset_count(len(candidates), params.max_predictors)
        if total_subsets > _MAX_SUBSETS_TO_EVALUATE:
            raise ValueError(
                f"Best subset selection would need to fit {total_subsets:,} models (from "
                f"{len(candidates)} candidates, max_predictors={params.max_predictors}), which "
                f"exceeds the {_MAX_SUBSETS_TO_EVALUATE:,}-model safety cap. Reduce "
                "candidate_predictors or max_predictors, or use forward_selection/"
                "backward_elimination instead for a large candidate pool."
            )

        data = self._load_data(ctx, params)
        y = data[params.target]

        mse_full = n = None
        if params.criterion == "mallows_cp":
            mse_full, n = _mallows_cp_reference(self._fit_ols, y, data, candidates)

        best_model, best_score, best_predictors = None, None, None

        for k in range(1, min(params.max_predictors, len(candidates)) + 1):
            for subset in itertools.combinations(candidates, k):
                trial_model = self._fit_ols(y, data[list(subset)])
                score = _best_subset_score(trial_model, params.criterion, list(subset), mse_full, n)
                if params.criterion == "p_value" and score >= params.alpha:
                    continue  # at least one predictor in this subset isn't significant enough
                if _forward_selection_is_better(score, best_score, params.criterion):
                    best_model, best_score, best_predictors = trial_model, score, list(subset)

        if best_model is None:
            reason = (
                f"no subset had every predictor significant below alpha={params.alpha}"
                if params.criterion == "p_value"
                else "none improved on the initial baseline"
            )
            raise ValueError(
                f"Best subset selection under criterion '{params.criterion}' did not select any "
                f"model from {candidates} -- {reason}."
            )

        best_model.best_subset_selection_criterion = params.criterion
        return self._build_result(best_model, data, params, predictors=best_predictors)

    @staticmethod
    def _fit_ols(y: pd.Series, predictors_df: pd.DataFrame):
        X = sm.add_constant(predictors_df, has_constant="add")
        return sm.OLS(y, X).fit()

    def _extra_interpretation(self, model) -> str:
        predictors = model.model.exog_names[1:]  # skip the constant
        return f" Best subset selection ({model.best_subset_selection_criterion}) selected predictors: {predictors}."

    def _extra_raw_summary(self, model) -> dict:
        predictors = model.model.exog_names[1:]  # skip the constant
        return {"selected_predictors": predictors}