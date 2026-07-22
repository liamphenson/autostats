import matplotlib.pyplot as plt
from pydantic import BaseModel
from statsmodels.tsa.seasonal import STL

from autostats.core.schemas.stat_result import StatResult
from autostats.core.tools.base import BaseTool, ToolContext, ToolInput
from autostats.core.tools.plotting import save_current_figure
from autostats.core.tools.registry import REGISTRY


class DecomposeTimeSeriesInput(ToolInput):
    column: str
    period: int


@REGISTRY.register
class DecomposeTimeSeriesTool(BaseTool):
    name = "decompose_time_series"
    description = "Decompose a time series into trend, seasonal, and residual components using STL."
    category = "timeseries"
    input_model = DecomposeTimeSeriesInput

    def run(self, ctx: ToolContext, params: DecomposeTimeSeriesInput) -> BaseModel:
        df = ctx.data_manager.load(params.dataset_id)
        series = df[params.column].dropna()
        result = STL(series, period=params.period).fit()

        fig, axes = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
        axes[0].plot(series.values); axes[0].set_ylabel("observed")
        axes[1].plot(result.trend); axes[1].set_ylabel("trend")
        axes[2].plot(result.seasonal); axes[2].set_ylabel("seasonal")
        axes[3].plot(result.resid); axes[3].set_ylabel("residual")
        plot = save_current_figure(ctx.plots_dir, f"STL decomposition of {params.column}")

        seasonal_strength = float(
            max(0.0, 1 - result.resid.var() / (result.seasonal + result.resid).var())
        )
        trend_strength = float(
            max(0.0, 1 - result.resid.var() / (result.trend + result.resid).var())
        )
        return StatResult(
            tool_name=self.name,
            test_name="stl_decomposition",
            effect_size={"trend_strength": trend_strength, "seasonal_strength": seasonal_strength},
            sample_sizes={params.column: len(series)},
            plots=[plot],
            interpretation=(
                f"STL decomposition of '{params.column}' (period={params.period}): "
                f"trend strength={trend_strength:.3f}, seasonal strength={seasonal_strength:.3f}."
            ),
        )
