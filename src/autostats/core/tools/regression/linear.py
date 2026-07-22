import statsmodels.api as sm
from pydantic import BaseModel
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

from autostats.core.schemas.stat_result import StatResult, TableArtifact
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

    def run(self, ctx: ToolContext, params: LinearRegressionInput) -> BaseModel:
        df = ctx.data_manager.load(params.dataset_id)
        data = df[[params.target, *params.predictors]].dropna()
        X = sm.add_constant(data[params.predictors])
        y = data[params.target]
        model = sm.OLS(y, X).fit()

        coef_table = TableArtifact(
            table_id=f"ols_coef_{params.dataset_id}",
            title="OLS coefficients",
            columns=["term", "coef", "std_err", "t", "p_value", "ci_lower", "ci_upper"],
            rows=[
                [term, round(model.params[term], 4), round(model.bse[term], 4), round(model.tvalues[term], 4),
                 round(model.pvalues[term], 4), *[round(v, 4) for v in model.conf_int().loc[term]]]
                for term in X.columns
            ],
        )

        vif_checks = check_vif(data[params.predictors])
        dw = float(durbin_watson(model.resid))
        bp_stat, bp_p, _, _ = het_breuschpagan(model.resid, model.model.exog)
        n_check = check_sample_size(len(data), 10 * len(params.predictors), "observations")

        assumptions = [
            *vif_checks,
            n_check,
            {"name": "durbin_watson", "passed": 1.5 <= dw <= 2.5, "statistic": dw,
             "detail": f"Durbin-Watson statistic is {dw:.3f} (close to 2 indicates little autocorrelation)."},
            {"name": "homoscedasticity_breusch_pagan", "passed": bp_p > 0.05, "statistic": float(bp_stat),
             "p_value": float(bp_p), "detail": f"Breusch-Pagan test p={bp_p:.4f} "
             f"({'no evidence of heteroscedasticity' if bp_p > 0.05 else 'evidence of heteroscedasticity'})."},
        ]
        from autostats.core.schemas.stat_result import AssumptionCheck

        assumptions = [a if isinstance(a, AssumptionCheck) else AssumptionCheck(**a) for a in assumptions]
        assumptions_met = all(c.passed for c in vif_checks) and bp_p > 0.05

        sig_terms = [t for t in params.predictors if model.pvalues[t] < params.alpha]
        return StatResult(
            tool_name=self.name,
            test_name="linear_regression",
            statistic=float(model.fvalue),
            p_value=float(model.f_pvalue),
            degrees_of_freedom=(float(model.df_model), float(model.df_resid)),
            effect_size={"r_squared": float(model.rsquared), "adj_r_squared": float(model.rsquared_adj)},
            sample_sizes={"observations": len(data)},
            assumptions=assumptions,
            assumptions_met=assumptions_met,
            tables=[coef_table],
            interpretation=(
                f"OLS regression of '{params.target}' on {params.predictors}: R-squared={model.rsquared:.3f}, "
                f"F={model.fvalue:.3f} (p={model.f_pvalue:.4f}). "
                f"Significant predictors (alpha={params.alpha}): {sig_terms or 'none'}."
            ),
            warnings=[] if assumptions_met else ["Regression diagnostics flagged potential multicollinearity or heteroscedasticity; see assumptions."],
            raw_summary={"params": model.params.to_dict(), "pvalues": model.pvalues.to_dict()},
        )
