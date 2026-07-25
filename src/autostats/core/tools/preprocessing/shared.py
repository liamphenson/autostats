"""Shared helpers for preprocessing tools that transform a dataset and register
the result as a new one (encoding, Box-Cox, ...)."""

import pandas as pd

from autostats.core.schemas.dataset import DatasetHandle
from autostats.core.tools.base import ToolContext


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    if missing := [c for c in columns if c not in df.columns]:
        raise ValueError(f"Column(s) not found in dataset: {missing}")


def register_derived(
    ctx: ToolContext,
    dataset_id: str,
    new_df: pd.DataFrame,
    *,
    method: str,
    extra_metadata: dict,
    warnings: list[str] | None = None,
) -> DatasetHandle:
    """Register a transformed dataframe as a new dataset, inheriting the parent's
    trust_level so a caveat on the original (e.g. low-trust scraped data)
    survives the transform."""
    parent_trust = ctx.data_manager.get_meta(dataset_id).trust_level
    return ctx.data_manager.register(
        new_df,
        source="derived",
        source_metadata={"derived_from": dataset_id, "method": method, **extra_metadata},
        trust_level=parent_trust,
        validation_warnings=warnings or [],
    )
