import pandas as pd
from pydantic import BaseModel

from autostats.core.tools.base import BaseTool, ToolContext, ToolInput
from autostats.core.tools.preprocessing.shared import register_derived
from autostats.core.tools.registry import REGISTRY


class TrainTestSplitInput(ToolInput):
    test_size: float = 0.2
    random_state: int | None = None
    shuffle: bool = False


class TrainTestSplitResult(BaseModel):
    train_dataset_id: str
    test_dataset_id: str
    train_n_rows: int
    test_n_rows: int


@REGISTRY.register
class TrainTestSplitTool(BaseTool):
    name = "train_test_split"
    description = (
        "Split a dataset into training and testing sets. Specify the proportion of the "
        "dataset to include in the test split with 'test_size' (between 0 and 1). Optionally,"
        " set 'random_state' for reproducibility and 'shuffle' to shuffle the data before splitting."
    )
    category = "preprocessing"
    input_model = TrainTestSplitInput

    def run(self, ctx: ToolContext, params: TrainTestSplitInput) -> BaseModel:
        df = ctx.data_manager.load(params.dataset_id)
        if not (0 < params.test_size < 1):
            raise ValueError("test size must be between 0 and 1.")
        if params.shuffle:
            df = df.sample(frac=1, random_state=params.random_state).reset_index(drop=True)
        split_index = int(len(df) * (1 - params.test_size))
        train_df = df.iloc[:split_index]
        test_df = df.iloc[split_index:]
        if len(train_df) == 0 or len(test_df) == 0:
            raise ValueError(
                f"Resulting split would leave an empty set (train={len(train_df)} rows, "
                f"test={len(test_df)} rows); choose a less extreme test_size or use a larger dataset."
            )
        train_handle = register_derived(ctx, params.dataset_id, train_df, method="train_test_split", extra_metadata={"split": "train"})
        test_handle = register_derived(ctx, params.dataset_id, test_df, method="train_test_split", extra_metadata={"split": "test"})
        return TrainTestSplitResult(
            train_dataset_id=train_handle.dataset_id,
            test_dataset_id=test_handle.dataset_id,
            train_n_rows=train_handle.n_rows,
            test_n_rows=test_handle.n_rows,
        )


class TrainValidationTestSplitInput(ToolInput):
    validation_size: float = 0.1
    test_size: float = 0.2
    random_state: int | None = None
    shuffle: bool = False


class TrainValidationTestSplitResult(BaseModel):
    train_dataset_id: str
    validation_dataset_id: str
    test_dataset_id: str
    train_n_rows: int
    validation_n_rows: int
    test_n_rows: int


@REGISTRY.register
class TrainValidationTestSplitTool(BaseTool):
    name = "train_validation_test_split"
    description = (
        "Split a dataset into training, validation, and testing sets. Specify the proportion of the "
        "dataset to include in the validation and test splits with 'validation_size' and 'test_size' (between 0 and 1). "
        "Optionally, set 'random_state' for reproducibility and 'shuffle' to shuffle the data before splitting."
    )
    category = "preprocessing"
    input_model = TrainValidationTestSplitInput

    def run(self, ctx: ToolContext, params: TrainValidationTestSplitInput) -> BaseModel:
        df = ctx.data_manager.load(params.dataset_id)
        if not (0 < params.validation_size < 1) or not (0 < params.test_size < 1):
            raise ValueError("validation size and test size must be between 0 and 1.")
        if params.validation_size + params.test_size >= 1:
            raise ValueError("The sum of validation size and test size must be less than 1.")
        if params.shuffle:
            df = df.sample(frac=1, random_state=params.random_state).reset_index(drop=True)
        split_index_val = int(len(df) * (1 - params.validation_size - params.test_size))
        split_index_test = int(len(df) * (1 - params.test_size))
        train_df = df.iloc[:split_index_val]
        validation_df = df.iloc[split_index_val:split_index_test]
        test_df = df.iloc[split_index_test:]
        if len(train_df) == 0 or len(validation_df) == 0 or len(test_df) == 0:
            raise ValueError(
                f"Resulting split would leave an empty set (train={len(train_df)} rows, "
                f"validation={len(validation_df)} rows, test={len(test_df)} rows); choose "
                "smaller validation_size/test_size or use a larger dataset."
            )
        train_handle = register_derived(ctx, params.dataset_id, train_df, method="train_validation_test_split", extra_metadata={"split": "train"})
        validation_handle = register_derived(ctx, params.dataset_id, validation_df, method="train_validation_test_split", extra_metadata={"split": "validation"})
        test_handle = register_derived(ctx, params.dataset_id, test_df, method="train_validation_test_split", extra_metadata={"split": "test"})
        return TrainValidationTestSplitResult(
            train_dataset_id=train_handle.dataset_id,
            validation_dataset_id=validation_handle.dataset_id,
            test_dataset_id=test_handle.dataset_id,
            train_n_rows=train_handle.n_rows,
            validation_n_rows=validation_handle.n_rows,
            test_n_rows=test_handle.n_rows,
        )
