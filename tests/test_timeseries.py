import numpy as np
import pandas as pd

from autostats.core.tools.registry import REGISTRY


def test_check_stationarity_on_stationary_noise(tool_ctx, rng):
    df = pd.DataFrame({"x": rng.normal(size=200)})
    handle = tool_ctx.data_manager.register(df, source="upload")
    result = REGISTRY.dispatch(tool_ctx, "check_stationarity", {"dataset_id": handle.dataset_id, "column": "x"})
    assert result.p_value < 0.05


def test_fit_and_forecast_arima(tool_ctx, rng):
    n = 150
    trend = np.linspace(0, 10, n)
    noise = rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"x": trend + noise})
    handle = tool_ctx.data_manager.register(df, source="upload")

    fit_result = REGISTRY.dispatch(
        tool_ctx, "fit_arima_model", {"dataset_id": handle.dataset_id, "column": "x", "order": [1, 1, 0]}
    )
    assert fit_result.statistic is not None

    forecast_result = REGISTRY.dispatch(
        tool_ctx, "forecast_time_series", {"dataset_id": handle.dataset_id, "column": "x", "steps": 5}
    )
    assert forecast_result.tables[0].rows.__len__() == 5


def test_decompose_time_series_produces_plot(tool_ctx, rng):
    n = 120
    t = np.arange(n)
    seasonal = 5 * np.sin(2 * np.pi * t / 12)
    df = pd.DataFrame({"x": seasonal + rng.normal(scale=0.2, size=n)})
    handle = tool_ctx.data_manager.register(df, source="upload")

    result = REGISTRY.dispatch(
        tool_ctx, "decompose_time_series", {"dataset_id": handle.dataset_id, "column": "x", "period": 12}
    )
    assert len(result.plots) == 1
    assert result.effect_size["seasonal_strength"] > 0.5
