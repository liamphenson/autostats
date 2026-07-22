from pydantic import BaseModel
from statsmodels.tsa.arima.model import ARIMA

from autostats.core.schemas.stat_result import StatResult, TableArtifact
from autostats.core.tools.base import BaseTool, ToolContext, ToolInput
from autostats.core.tools.registry import REGISTRY


def _select_order(series) -> tuple[int, int, int]:
    try:
        import pmdarima as pm

        model = pm.auto_arima(series, seasonal=False, suppress_warnings=True, error_action="ignore")
        return model.order
    except ImportError:
        return (1, 1, 1)


class FitArimaModelInput(ToolInput):
    column: str
    order: tuple[int, int, int] | None = None


@REGISTRY.register
class FitArimaModelTool(BaseTool):
    name = "fit_arima_model"
    description = "Fit an ARIMA model to a time series column. Order is auto-selected via pmdarima if not given."
    category = "timeseries"
    input_model = FitArimaModelInput

    def run(self, ctx: ToolContext, params: FitArimaModelInput) -> BaseModel:
        df = ctx.data_manager.load(params.dataset_id)
        series = df[params.column].dropna()
        order = params.order or _select_order(series)
        model = ARIMA(series, order=order).fit()

        ctx.data_manager._arima_models = getattr(ctx.data_manager, "_arima_models", {})
        ctx.data_manager._arima_models[f"{params.dataset_id}:{params.column}"] = model

        coef_table = TableArtifact(
            table_id=f"arima_coef_{params.dataset_id}",
            title=f"ARIMA{order} coefficients",
            columns=["term", "coef", "p_value"],
            rows=[[t, round(model.params[t], 4), round(model.pvalues[t], 4)] for t in model.params.index],
        )
        return StatResult(
            tool_name=self.name,
            test_name="arima_fit",
            statistic=float(model.aic),
            sample_sizes={params.column: len(series)},
            tables=[coef_table],
            interpretation=(
                f"Fit ARIMA{order} to '{params.column}': AIC={model.aic:.2f}, BIC={model.bic:.2f}. "
                f"Use forecast_time_series to generate forecasts from this fit."
            ),
            raw_summary={"order": list(order)},
        )


class ForecastTimeSeriesInput(ToolInput):
    column: str
    steps: int = 10
    alpha: float = 0.05


@REGISTRY.register
class ForecastTimeSeriesTool(BaseTool):
    name = "forecast_time_series"
    description = "Generate a forecast with confidence intervals from a previously fit ARIMA model (call fit_arima_model first)."
    category = "timeseries"
    input_model = ForecastTimeSeriesInput

    def run(self, ctx: ToolContext, params: ForecastTimeSeriesInput) -> BaseModel:
        models = getattr(ctx.data_manager, "_arima_models", {})
        key = f"{params.dataset_id}:{params.column}"
        if key not in models:
            raise ValueError(f"No fitted ARIMA model for '{params.column}'; call fit_arima_model first.")
        model = models[key]
        forecast = model.get_forecast(steps=params.steps)
        mean = forecast.predicted_mean
        ci = forecast.conf_int(alpha=params.alpha)

        table = TableArtifact(
            table_id=f"forecast_{params.dataset_id}",
            title=f"{params.steps}-step forecast for {params.column}",
            columns=["step", "forecast", "ci_lower", "ci_upper"],
            rows=[
                [i + 1, round(mean.iloc[i], 4), round(ci.iloc[i, 0], 4), round(ci.iloc[i, 1], 4)]
                for i in range(params.steps)
            ],
        )
        return StatResult(
            tool_name=self.name,
            test_name="arima_forecast",
            confidence_level=1 - params.alpha,
            tables=[table],
            interpretation=f"Generated a {params.steps}-step forecast for '{params.column}' with {(1 - params.alpha) * 100:.0f}% confidence intervals.",
        )
