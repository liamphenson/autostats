from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autostats.core.data.manager import DataManager
from autostats.core.tools import load_all_tools
from autostats.core.tools.base import ToolContext

load_all_tools()


@pytest.fixture
def tmp_data_manager(tmp_path: Path) -> DataManager:
    return DataManager(session_id="test", storage_dir=tmp_path / "datasets")


@pytest.fixture
def tool_ctx(tmp_data_manager: DataManager, tmp_path: Path) -> ToolContext:
    return ToolContext(session_id="test", data_manager=tmp_data_manager, plots_dir=str(tmp_path / "plots"))


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def two_group_df(rng: np.random.Generator) -> pd.DataFrame:
    a = rng.normal(loc=10, scale=2, size=60)
    b = rng.normal(loc=12, scale=2, size=60)
    return pd.DataFrame(
        {
            "value": np.concatenate([a, b]),
            "group": ["A"] * 60 + ["B"] * 60,
        }
    )
