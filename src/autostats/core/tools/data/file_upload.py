from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from autostats.core.schemas.dataset import DatasetHandle
from autostats.core.tools.base import BaseTool, ToolContext
from autostats.core.tools.registry import REGISTRY


class UploadDatasetInput(BaseModel):
    file_path: str
    file_type: str | None = None


def _read_any(path: Path, file_type: str | None) -> pd.DataFrame:
    ext = (file_type or path.suffix.lstrip(".")).lower()
    if ext in ("csv",):
        return pd.read_csv(path)
    if ext in ("tsv",):
        return pd.read_csv(path, sep="\t")
    if ext in ("xlsx", "xls"):
        return pd.read_excel(path)
    if ext in ("json",):
        return pd.read_json(path)
    if ext in ("parquet",):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {ext}")


@REGISTRY.register
class UploadDatasetTool(BaseTool):
    name = "upload_dataset"
    description = "Load a user-provided data file (CSV/TSV/Excel/JSON/Parquet) into the session dataset store."
    category = "data_io"
    input_model = UploadDatasetInput

    def run(self, ctx: ToolContext, params: UploadDatasetInput) -> BaseModel:
        path = Path(params.file_path)
        if not path.exists():
            raise FileNotFoundError(f"No such file: {path}")
        df = _read_any(path, params.file_type)
        handle: DatasetHandle = ctx.data_manager.register(
            df, source="upload", source_metadata={"file_name": path.name}
        )
        return handle
