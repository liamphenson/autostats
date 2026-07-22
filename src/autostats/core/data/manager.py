import uuid
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from autostats.core.schemas.dataset import ColumnInfo, DatasetHandle, DatasetMeta

PREVIEW_ROWS = 5


class DataManager:
    """Owns every DataFrame loaded during a session.

    Raw DataFrames never leave this class -- tools resolve `dataset_id`
    to a DataFrame internally via `load()`; only `DatasetMeta` (shape,
    columns, a small preview) is allowed into LLM context.
    """

    def __init__(self, session_id: str, storage_dir: Path):
        self.session_id = session_id
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._frames: dict[str, pd.DataFrame] = {}
        self._meta: dict[str, DatasetMeta] = {}

    def _parquet_path(self, dataset_id: str) -> Path:
        return self.storage_dir / f"{dataset_id}.parquet"

    def register(
        self,
        df: pd.DataFrame,
        source: Literal["upload", "fred", "world_bank", "census", "web_scrape"],
        source_metadata: dict[str, Any] | None = None,
        trust_level: Literal["high", "medium", "low"] = "high",
        validation_warnings: list[str] | None = None,
    ) -> DatasetHandle:
        dataset_id = uuid.uuid4().hex[:12]
        self._frames[dataset_id] = df
        df.to_parquet(self._parquet_path(dataset_id))

        columns = [ColumnInfo(name=c, dtype=str(df[c].dtype)) for c in df.columns]
        preview = df.head(PREVIEW_ROWS).to_dict(orient="records")

        meta = DatasetHandle(
            dataset_id=dataset_id,
            n_rows=len(df),
            n_cols=len(df.columns),
            columns=columns,
            preview=preview,
            source=source,
            source_metadata=source_metadata or {},
            trust_level=trust_level,
            validation_warnings=validation_warnings or [],
        )
        self._meta[dataset_id] = meta
        return meta

    def load(self, dataset_id: str) -> pd.DataFrame:
        if dataset_id in self._frames:
            return self._frames[dataset_id]
        path = self._parquet_path(dataset_id)
        if not path.exists():
            raise KeyError(f"Unknown dataset_id: {dataset_id}")
        df = pd.read_parquet(path)
        self._frames[dataset_id] = df
        return df

    def get_meta(self, dataset_id: str) -> DatasetMeta:
        if dataset_id not in self._meta:
            raise KeyError(f"Unknown dataset_id: {dataset_id}")
        return self._meta[dataset_id]

    def list_datasets(self) -> list[DatasetMeta]:
        return list(self._meta.values())

    def catalog_text(self) -> str:
        """Rendered as part of the system prompt -- the only view of loaded
        datasets the LLM ever gets."""
        if not self._meta:
            return "No datasets are currently loaded."
        lines = ["Loaded datasets:"]
        for meta in self._meta.values():
            cols = ", ".join(f"{c.name} ({c.dtype})" for c in meta.columns)
            lines.append(
                f"- dataset_id={meta.dataset_id!r}: {meta.n_rows} rows x {meta.n_cols} cols "
                f"[{cols}] source={meta.source} trust={meta.trust_level}"
            )
            if meta.validation_warnings:
                lines.append(f"  warnings: {'; '.join(meta.validation_warnings)}")
        return "\n".join(lines)
