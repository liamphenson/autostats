from pydantic import BaseModel
from statsmodels.tsa.stattools import adfuller, kpss

from autostats.core.schemas.stat_result import AssumptionCheck, StatResult
from autostats.core.tools.base import BaseTool, ToolContext, ToolInput
from autostats.core.tools.registry import REGISTRY
from autostats.core.tools.stats.assumptions import check_sample_size


class CheckStationarityInput(ToolInput):
    column: str
    alpha: float = 0.05


@REGISTRY.register
class CheckStationarityTool(BaseTool):
    name = "check_stationarity"
    description = "Test a time series column for stationarity using both ADF (H0: unit root) and KPSS (H0: stationary)."
    category = "timeseries"
    input_model = CheckStationarityInput

    def run(self, ctx: ToolContext, params: CheckStationarityInput) -> BaseModel:
        df = ctx.data_manager.load(params.dataset_id)
        series = df[params.column].dropna()

        adf_stat, adf_p, *_ = adfuller(series)
        kpss_stat, kpss_p, *_ = kpss(series, nlags="auto")

        adf_stationary = adf_p < params.alpha
        kpss_stationary = kpss_p >= params.alpha

        checks = [
            AssumptionCheck(
                name="adf_test", passed=adf_stationary, statistic=float(adf_stat), p_value=float(adf_p),
                detail=f"ADF test: {'rejects' if adf_stationary else 'fails to reject'} the unit-root null (p={adf_p:.4f}).",
            ),
            AssumptionCheck(
                name="kpss_test", passed=kpss_stationary, statistic=float(kpss_stat), p_value=float(kpss_p),
                detail=f"KPSS test: {'fails to reject' if kpss_stationary else 'rejects'} the stationarity null (p={kpss_p:.4f}).",
            ),
            check_sample_size(len(series), 30, params.column),
        ]
        agree = adf_stationary == kpss_stationary
        conclusion = (
            f"Both tests agree the series is {'stationary' if adf_stationary else 'non-stationary'}."
            if agree else
            "ADF and KPSS disagree -- this is informative on its own (e.g. trend-stationary series) and warrants closer inspection."
        )
        return StatResult(
            tool_name=self.name,
            test_name="check_stationarity",
            statistic=float(adf_stat),
            p_value=float(adf_p),
            sample_sizes={params.column: len(series)},
            assumptions=checks,
            assumptions_met=agree,
            recommended_alternative=None if adf_stationary else "decompose_time_series (then difference before ARIMA)",
            interpretation=f"Stationarity check on '{params.column}': {conclusion}",
        )
