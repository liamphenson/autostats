from pydantic import BaseModel

from autostats.core.schemas.dataset import DatasetMeta
from autostats.core.tools.base import BaseTool, ToolContext, ToolInput
from autostats.core.tools.registry import REGISTRY


class ListDatasetsInput(BaseModel):
    pass


class DatasetList(BaseModel):
    datasets: list[DatasetMeta]


@REGISTRY.register
class ListDatasetsTool(BaseTool):
    name = "list_datasets"
    description = "List all datasets currently loaded in this session."
    category = "data_io"
    input_model = ListDatasetsInput

    def run(self, ctx: ToolContext, params: ListDatasetsInput) -> BaseModel:
        return DatasetList(datasets=ctx.data_manager.list_datasets())


class GetDatasetPreviewInput(ToolInput):
    n: int = 5


@REGISTRY.register
class GetDatasetPreviewTool(BaseTool):
    name = "get_dataset_preview"
    description = "Return the first N rows and column dtypes of a loaded dataset."
    category = "data_io"
    input_model = GetDatasetPreviewInput

    def run(self, ctx: ToolContext, params: GetDatasetPreviewInput) -> BaseModel:
        df = ctx.data_manager.load(params.dataset_id)
        meta = ctx.data_manager.get_meta(params.dataset_id)
        preview = df.head(params.n).to_dict(orient="records")
        return meta.model_copy(update={"preview": preview})
