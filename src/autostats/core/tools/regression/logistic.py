import numpy as np
import statsmodels.api as sm
from pydantic import BaseModel

from autostats.core.schemas.stat_result import StatResult, TableArtifact
from autostats.core.tools.base import BaseTool, ToolContext, ToolInput
from autostats.core.tools.registry import REGISTRY
from autostats.core.tools.stats.assumptions import check_sample_size, check_vif


class LogisticRegressionInput(ToolInput):
    target: str
    predictors: list[str]
    alpha: float = 0.05


@REGISTRY.register
class LogisticRegressionTool(BaseTool):
    name = "logistic_regression"
    description = "Fit a logistic regression of a binary target on one or more predictors, reporting odds ratios."
    category = "regression"
    input_model = LogisticRegressionInput

    def run(self, ctx: ToolContext, params: LogisticRegressionInput) -> BaseModel:
        df = ctx.data_manager.load(params.dataset_id)
        data = df[[params.target, *params.predictors]].dropna()
        if data[params.target].nunique() != 2:
            raise ValueError(f"'{params.target}' must be binary for logistic regression")
        y = data[params.target].astype("category").cat.codes
        X = sm.add_constant(data[params.predictors])
        model = sm.Logit(y, X).fit(disp=0)

        odds_ratios = np.exp(model.params)
        coef_table = TableArtifact(
            table_id=f"logit_coef_{params.dataset_id}",
            title="Logistic regression coefficients",
            columns=["term", "coef", "odds_ratio", "p_value"],
            rows=[
                [term, round(model.params[term], 4), round(odds_ratios[term], 4), round(model.pvalues[term], 4)]
                for term in X.columns
            ],
        )

        vif_checks = check_vif(data[params.predictors])
        n_check = check_sample_size(len(data), 10 * len(params.predictors), "observations")
        assumptions = [*vif_checks, n_check]
        assumptions_met = all(c.passed for c in vif_checks)

        pseudo_r2 = float(model.prsquared)
        sig_terms = [t for t in params.predictors if model.pvalues[t] < params.alpha]
        return StatResult(
            tool_name=self.name,
            test_name="logistic_regression",
            statistic=float(model.llr),
            p_value=float(model.llr_pvalue),
            degrees_of_freedom=float(model.df_model),
            effect_size={"mcfadden_pseudo_r_squared": pseudo_r2},
            sample_sizes={"observations": len(data)},
            assumptions=assumptions,
            assumptions_met=assumptions_met,
            tables=[coef_table],
            interpretation=(
                f"Logistic regression of '{params.target}' on {params.predictors}: "
                f"McFadden pseudo-R-squared={pseudo_r2:.3f}, LR chi2 p={model.llr_pvalue:.4f}. "
                f"Significant predictors (alpha={params.alpha}): {sig_terms or 'none'}."
            ),
            warnings=[] if assumptions_met else ["Potential multicollinearity among predictors; see VIF checks."],
            raw_summary={"params": model.params.to_dict(), "odds_ratios": odds_ratios.to_dict()},
        )
