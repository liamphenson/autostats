import pandas as pd
from pydantic import BaseModel

from autostats.core.schemas.dataset import DatasetHandle
from autostats.core.tools.base import BaseTool, ToolContext, ToolInput
from autostats.core.tools.registry import REGISTRY


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    if missing := [c for c in columns if c not in df.columns]:
        raise ValueError(f"Column(s) not found in dataset: {missing}")


def _register_derived(
    ctx: ToolContext,
    dataset_id: str,
    new_df: pd.DataFrame,
    *,
    method: str,
    extra_metadata: dict,
    warnings: list[str] | None = None,
) -> DatasetHandle:
    """Register an encoded dataframe as a new dataset, inheriting the parent's
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


class OneHotEncodeInput(ToolInput):
    columns: list[str]


@REGISTRY.register
class OneHotEncodeTool(BaseTool):
    name = "one_hot_encode"
    description = (
        "One-hot encode one or more categorical columns into a binary column per category "
        "(k columns for k categories) and register the result as a new dataset."
    )
    category = "preprocessing"
    input_model = OneHotEncodeInput

    def run(self, ctx: ToolContext, params: OneHotEncodeInput) -> BaseModel:
        df = ctx.data_manager.load(params.dataset_id)
        _require_columns(df, params.columns)
        encoded = pd.get_dummies(df, columns=params.columns, prefix=params.columns, drop_first=False)
        new_columns = [c for c in encoded.columns if c not in df.columns]
        encoded[new_columns] = encoded[new_columns].astype(int)
        warnings = [
            f"'{c}' has {df[c].nunique()} categories, producing that many new columns."
            for c in params.columns
            if df[c].nunique() > 15
        ]
        return _register_derived(
            ctx,
            params.dataset_id,
            encoded,
            method="one_hot",
            extra_metadata={"encoded_columns": params.columns, "new_columns": new_columns},
            warnings=warnings,
        )


class DummyEncodeInput(ToolInput):
    columns: list[str]


@REGISTRY.register
class DummyEncodeTool(BaseTool):
    name = "dummy_encode"
    description = (
        "Dummy-encode one or more categorical columns: one-hot encoding with the first category "
        "dropped as the reference level. This is the standard encoding for regression predictors -- "
        "it avoids the dummy-variable trap (perfect multicollinearity) that full one-hot encoding "
        "creates alongside an intercept."
    )
    category = "preprocessing"
    input_model = DummyEncodeInput

    def run(self, ctx: ToolContext, params: DummyEncodeInput) -> BaseModel:
        df = ctx.data_manager.load(params.dataset_id)
        _require_columns(df, params.columns)
        encoded = pd.get_dummies(df, columns=params.columns, prefix=params.columns, drop_first=True)
        new_columns = [c for c in encoded.columns if c not in df.columns]
        encoded[new_columns] = encoded[new_columns].astype(int)
        return _register_derived(
            ctx,
            params.dataset_id,
            encoded,
            method="dummy",
            extra_metadata={"encoded_columns": params.columns, "new_columns": new_columns},
        )


class OrdinalEncodeInput(ToolInput):
    column: str
    order: list[str] | None = None


@REGISTRY.register
class OrdinalEncodeTool(BaseTool):
    name = "ordinal_encode"
    description = (
        "Encode a single categorical column as integers 0..k-1 following a meaningful order you "
        "provide (e.g. ['low', 'medium', 'high']). Use this only when the categories have a genuine "
        "ranking -- for unordered categories use one_hot_encode/dummy_encode instead."
    )
    category = "preprocessing"
    input_model = OrdinalEncodeInput

    def run(self, ctx: ToolContext, params: OrdinalEncodeInput) -> BaseModel:
        df = ctx.data_manager.load(params.dataset_id)
        _require_columns(df, [params.column])
        categories = {str(c) for c in df[params.column].dropna().unique()}
        warnings = []
        if params.order is None:
            order = sorted(categories)
            warnings.append(
                f"No explicit 'order' was given for '{params.column}'; categories were sorted "
                f"alphabetically ({order}). Pass 'order' explicitly if this doesn't reflect a true ranking."
            )
        elif missing := categories - set(params.order):
            raise ValueError(f"'order' is missing categories present in the data: {sorted(missing)}")
        else:
            order = params.order
        mapping = {cat: i for i, cat in enumerate(order)}
        new_df = df.copy()
        new_df[params.column] = df[params.column].astype(str).map(mapping)
        return _register_derived(
            ctx,
            params.dataset_id,
            new_df,
            method="ordinal",
            extra_metadata={"encoded_column": params.column, "order": order},
            warnings=warnings,
        )


class LabelEncodeInput(ToolInput):
    column: str


@REGISTRY.register
class LabelEncodeTool(BaseTool):
    name = "label_encode"
    description = (
        "Assign each category in a column an arbitrary integer code (0..k-1, sorted alphabetically). "
        "Does not imply any meaningful order or distance between categories -- use ordinal_encode if "
        "the categories have a true ranking, or one_hot_encode/dummy_encode before regression."
    )
    category = "preprocessing"
    input_model = LabelEncodeInput

    def run(self, ctx: ToolContext, params: LabelEncodeInput) -> BaseModel:
        df = ctx.data_manager.load(params.dataset_id)
        _require_columns(df, [params.column])
        categories = sorted(str(c) for c in df[params.column].dropna().unique())
        mapping = {cat: i for i, cat in enumerate(categories)}
        new_df = df.copy()
        new_df[params.column] = df[params.column].astype(str).map(mapping)
        return _register_derived(
            ctx,
            params.dataset_id,
            new_df,
            method="label",
            extra_metadata={"encoded_column": params.column, "mapping": mapping},
            warnings=[
                "Label encoding assigns arbitrary integer codes with no meaningful order or distance "
                "between categories; using them directly in linear/logistic regression implies a false "
                "numeric relationship. Prefer one_hot_encode or dummy_encode for regression predictors."
            ],
        )


class TargetEncodeInput(ToolInput):
    column: str
    target_column: str
    smoothing: float = 0.0


@REGISTRY.register
class TargetEncodeTool(BaseTool):
    name = "target_encode"
    description = (
        "Replace each category in a column with the mean of a numeric target column for that "
        "category (optionally shrunk toward the global mean via 'smoothing'). Computed in-sample on "
        "this dataset -- carries a real risk of target leakage/overfitting, so treat results from "
        "models using this column as exploratory unless a proper train/validation split is used."
    )
    category = "preprocessing"
    input_model = TargetEncodeInput

    def run(self, ctx: ToolContext, params: TargetEncodeInput) -> BaseModel:
        df = ctx.data_manager.load(params.dataset_id)
        _require_columns(df, [params.column, params.target_column])
        if not pd.api.types.is_numeric_dtype(df[params.target_column]):
            raise ValueError(f"target_column '{params.target_column}' must be numeric")
        if params.smoothing < 0:
            raise ValueError("smoothing must be >= 0")

        global_mean = df[params.target_column].mean()
        group_stats = df.groupby(params.column)[params.target_column].agg(["mean", "count"])
        if params.smoothing > 0:
            weight = group_stats["count"] / (group_stats["count"] + params.smoothing)
            mapping = (weight * group_stats["mean"] + (1 - weight) * global_mean).to_dict()
        else:
            mapping = group_stats["mean"].to_dict()

        new_df = df.copy()
        new_df[params.column] = df[params.column].map(mapping)
        return _register_derived(
            ctx,
            params.dataset_id,
            new_df,
            method="target",
            extra_metadata={
                "encoded_column": params.column,
                "target_column": params.target_column,
                "smoothing": params.smoothing,
            },
            warnings=[
                "Target encoding was computed in-sample on this dataset; this risks leakage/overfitting "
                "if the same data is then used to evaluate a model. Treat results from models using this "
                "column as exploratory unless a proper train/validation split was used."
            ],
        )
