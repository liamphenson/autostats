from pydantic import BaseModel

from autostats.core.schemas.stat_result import StatResult, TableArtifact
from autostats.core.tools.base import BaseTool, ToolContext, ToolInput
from autostats.core.tools.registry import REGISTRY


class DescribeDatasetInput(ToolInput):
    columns: list[str] | None = None


@REGISTRY.register
class DescribeDatasetTool(BaseTool):
    name = "describe_dataset"
    description = "Compute descriptive statistics (count, mean, std, min/max, quartiles) for numeric columns."
    category = "descriptive"
    input_model = DescribeDatasetInput

    def run(self, ctx: ToolContext, params: DescribeDatasetInput) -> BaseModel:
        df = ctx.data_manager.load(params.dataset_id)
        cols = params.columns or list(df.select_dtypes("number").columns)
        desc = df[cols].describe().round(4)
        table = TableArtifact(
            table_id=f"describe_{params.dataset_id}",
            title="Descriptive statistics",
            columns=["stat", *cols],
            rows=[[idx, *row.tolist()] for idx, row in desc.iterrows()],
        )
        summary = ", ".join(f"{c}: mean={desc.loc['mean', c]:.3g}, sd={desc.loc['std', c]:.3g}" for c in cols)
        return StatResult(
            tool_name=self.name,
            test_name="descriptive_statistics",
            interpretation=f"Descriptive statistics for {', '.join(cols)}. {summary}.",
            tables=[table],
            sample_sizes={c: int(df[c].count()) for c in cols},
        )


class CorrelationMatrixInput(ToolInput):
    columns: list[str]
    method: str = "pearson"


@REGISTRY.register
class CorrelationMatrixTool(BaseTool):
    name = "correlation_matrix"
    description = "Compute a Pearson or Spearman correlation matrix across the given numeric columns."
    category = "descriptive"
    input_model = CorrelationMatrixInput

    def run(self, ctx: ToolContext, params: CorrelationMatrixInput) -> BaseModel:
        df = ctx.data_manager.load(params.dataset_id)
        corr = df[params.columns].corr(method=params.method).round(4)
        table = TableArtifact(
            table_id=f"corr_{params.dataset_id}",
            title=f"{params.method.title()} correlation matrix",
            columns=["", *params.columns],
            rows=[[idx, *row.tolist()] for idx, row in corr.iterrows()],
        )
        return StatResult(
            tool_name=self.name,
            test_name=f"{params.method}_correlation_matrix",
            interpretation=f"{params.method.title()} correlation matrix computed for: {', '.join(params.columns)}.",
            tables=[table],
        )
