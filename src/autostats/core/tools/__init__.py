from autostats.core.tools.base import BaseTool, ToolContext, ToolInput
from autostats.core.tools.registry import REGISTRY, ToolRegistry

__all__ = ["REGISTRY", "BaseTool", "ToolContext", "ToolInput", "ToolRegistry", "load_all_tools"]


def load_all_tools() -> None:
    """Import every tool module for its `@REGISTRY.register` side effect.

    Adding a new tool only ever requires adding its import here -- no other
    agent-loop or prompt changes are needed.
    """
    import autostats.core.tools.data.file_upload  # noqa: F401
    import autostats.core.tools.data.store  # noqa: F401
    import autostats.core.tools.preprocessing.encoding  # noqa: F401
    import autostats.core.tools.preprocessing.split  # noqa: F401  -- FIX 1: was missing entirely
    import autostats.core.tools.preprocessing.transforms  # noqa: F401
    import autostats.core.tools.resampling.jackknife  # noqa: F401
    import autostats.core.tools.resampling.bootstrap  # noqa: F401
    import autostats.core.tools.stats.descriptive  # noqa: F401
    import autostats.core.tools.stats.parametric  # noqa: F401
    import autostats.core.tools.stats.nonparametric  # noqa: F401
    import autostats.core.tools.regression.linear  # noqa: F401
    import autostats.core.tools.regression.logistic  # noqa: F401
    import autostats.core.tools.timeseries.stationarity  # noqa: F401
    import autostats.core.tools.timeseries.decomposition  # noqa: F401
    import autostats.core.tools.timeseries.arima  # noqa: F401
