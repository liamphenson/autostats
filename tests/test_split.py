import pandas as pd
import pytest

from autostats.core.tools.registry import REGISTRY


def _register(tool_ctx, df, **kwargs):
    return tool_ctx.data_manager.register(df, source="upload", **kwargs)


# --- train_test_split ------------------------------------------------------

def test_train_test_split_row_counts_and_no_overlap(tool_ctx):
    df = pd.DataFrame({"a": range(100)})
    handle = _register(tool_ctx, df)

    result = REGISTRY.dispatch(tool_ctx, "train_test_split", {"dataset_id": handle.dataset_id, "test_size": 0.2})

    assert result.train_n_rows == 80
    assert result.test_n_rows == 20
    train = tool_ctx.data_manager.load(result.train_dataset_id)
    test = tool_ctx.data_manager.load(result.test_dataset_id)
    assert len(train) == 80
    assert len(test) == 20
    # every original row appears exactly once across the two splits
    assert sorted(train["a"].tolist() + test["a"].tolist()) == list(range(100))


def test_train_test_split_without_shuffle_preserves_order(tool_ctx):
    df = pd.DataFrame({"a": range(20)})
    handle = _register(tool_ctx, df)

    result = REGISTRY.dispatch(
        tool_ctx, "train_test_split", {"dataset_id": handle.dataset_id, "test_size": 0.25, "shuffle": False}
    )

    train = tool_ctx.data_manager.load(result.train_dataset_id)
    test = tool_ctx.data_manager.load(result.test_dataset_id)
    assert train["a"].tolist() == list(range(15))
    assert test["a"].tolist() == list(range(15, 20))


def test_train_test_split_shuffle_is_reproducible_with_random_state(tool_ctx):
    df = pd.DataFrame({"a": range(50)})
    handle = _register(tool_ctx, df)

    r1 = REGISTRY.dispatch(
        tool_ctx, "train_test_split",
        {"dataset_id": handle.dataset_id, "test_size": 0.2, "shuffle": True, "random_state": 7},
    )
    r2 = REGISTRY.dispatch(
        tool_ctx, "train_test_split",
        {"dataset_id": handle.dataset_id, "test_size": 0.2, "shuffle": True, "random_state": 7},
    )

    train1 = tool_ctx.data_manager.load(r1.train_dataset_id)["a"].tolist()
    train2 = tool_ctx.data_manager.load(r2.train_dataset_id)["a"].tolist()
    assert train1 == train2
    # a shuffled split of sequential data shouldn't just be the same prefix in order
    assert train1 != list(range(40))


@pytest.mark.parametrize("test_size", [0.0, 1.0, -0.1, 1.5])
def test_train_test_split_rejects_invalid_test_size(tool_ctx, test_size):
    df = pd.DataFrame({"a": range(10)})
    handle = _register(tool_ctx, df)

    with pytest.raises(ValueError, match="between 0 and 1"):
        REGISTRY.dispatch(tool_ctx, "train_test_split", {"dataset_id": handle.dataset_id, "test_size": test_size})


def test_train_test_split_rejects_resulting_empty_split(tool_ctx):
    df = pd.DataFrame({"a": range(3)})
    handle = _register(tool_ctx, df)

    with pytest.raises(ValueError, match="empty set"):
        REGISTRY.dispatch(tool_ctx, "train_test_split", {"dataset_id": handle.dataset_id, "test_size": 0.99})


def test_train_test_split_registers_derived_datasets_inheriting_trust_level(tool_ctx):
    df = pd.DataFrame({"a": range(10)})
    handle = _register(tool_ctx, df, trust_level="low", validation_warnings=["scraped from the web"])

    result = REGISTRY.dispatch(tool_ctx, "train_test_split", {"dataset_id": handle.dataset_id, "test_size": 0.2})

    train_meta = tool_ctx.data_manager.get_meta(result.train_dataset_id)
    test_meta = tool_ctx.data_manager.get_meta(result.test_dataset_id)
    assert train_meta.source == "derived" and train_meta.trust_level == "low"
    assert test_meta.source == "derived" and test_meta.trust_level == "low"


def test_train_test_split_feeds_directly_into_linear_regression(tool_ctx, rng):
    x = rng.normal(size=200)
    y = 3.0 + 2.0 * x + rng.normal(scale=0.1, size=200)
    df = pd.DataFrame({"x": x, "y": y})
    handle = _register(tool_ctx, df)

    split = REGISTRY.dispatch(tool_ctx, "train_test_split", {"dataset_id": handle.dataset_id, "test_size": 0.2})

    result = REGISTRY.dispatch(
        tool_ctx,
        "linear_regression",
        {"dataset_id": split.train_dataset_id, "target": "y", "predictors": ["x"]},
    )
    assert result.effect_size["r_squared"] > 0.95


# --- train_validation_test_split --------------------------------------------

def test_train_validation_test_split_row_counts_and_no_overlap(tool_ctx):
    df = pd.DataFrame({"a": range(100)})
    handle = _register(tool_ctx, df)

    result = REGISTRY.dispatch(
        tool_ctx,
        "train_validation_test_split",
        {"dataset_id": handle.dataset_id, "validation_size": 0.1, "test_size": 0.2},
    )

    assert (result.train_n_rows, result.validation_n_rows, result.test_n_rows) == (70, 10, 20)
    train = tool_ctx.data_manager.load(result.train_dataset_id)
    validation = tool_ctx.data_manager.load(result.validation_dataset_id)
    test = tool_ctx.data_manager.load(result.test_dataset_id)
    all_rows = train["a"].tolist() + validation["a"].tolist() + test["a"].tolist()
    assert sorted(all_rows) == list(range(100))


@pytest.mark.parametrize(
    "validation_size,test_size",
    [(0.0, 0.2), (1.0, 0.2), (0.1, 0.0), (0.1, 1.0), (-0.1, 0.2)],
)
def test_train_validation_test_split_rejects_invalid_sizes(tool_ctx, validation_size, test_size):
    df = pd.DataFrame({"a": range(10)})
    handle = _register(tool_ctx, df)

    with pytest.raises(ValueError, match="between 0 and 1"):
        REGISTRY.dispatch(
            tool_ctx,
            "train_validation_test_split",
            {"dataset_id": handle.dataset_id, "validation_size": validation_size, "test_size": test_size},
        )


def test_train_validation_test_split_rejects_sizes_summing_to_at_least_one(tool_ctx):
    df = pd.DataFrame({"a": range(10)})
    handle = _register(tool_ctx, df)

    with pytest.raises(ValueError, match="must be less than 1"):
        REGISTRY.dispatch(
            tool_ctx,
            "train_validation_test_split",
            {"dataset_id": handle.dataset_id, "validation_size": 0.5, "test_size": 0.5},
        )


def test_train_validation_test_split_rejects_resulting_empty_split(tool_ctx):
    df = pd.DataFrame({"a": range(5)})
    handle = _register(tool_ctx, df)

    with pytest.raises(ValueError, match="empty set"):
        REGISTRY.dispatch(
            tool_ctx,
            "train_validation_test_split",
            {"dataset_id": handle.dataset_id, "validation_size": 0.01, "test_size": 0.01},
        )


def test_train_validation_test_split_shuffle_is_reproducible_with_random_state(tool_ctx):
    df = pd.DataFrame({"a": range(50)})
    handle = _register(tool_ctx, df)

    r1 = REGISTRY.dispatch(
        tool_ctx, "train_validation_test_split",
        {"dataset_id": handle.dataset_id, "validation_size": 0.1, "test_size": 0.2, "shuffle": True, "random_state": 3},
    )
    r2 = REGISTRY.dispatch(
        tool_ctx, "train_validation_test_split",
        {"dataset_id": handle.dataset_id, "validation_size": 0.1, "test_size": 0.2, "shuffle": True, "random_state": 3},
    )

    v1 = tool_ctx.data_manager.load(r1.validation_dataset_id)["a"].tolist()
    v2 = tool_ctx.data_manager.load(r2.validation_dataset_id)["a"].tolist()
    assert v1 == v2


def test_train_validation_test_split_registers_derived_datasets_inheriting_trust_level(tool_ctx):
    df = pd.DataFrame({"a": range(20)})
    handle = _register(tool_ctx, df, trust_level="low", validation_warnings=["scraped from the web"])

    result = REGISTRY.dispatch(
        tool_ctx,
        "train_validation_test_split",
        {"dataset_id": handle.dataset_id, "validation_size": 0.1, "test_size": 0.2},
    )

    for dataset_id in (result.train_dataset_id, result.validation_dataset_id, result.test_dataset_id):
        meta = tool_ctx.data_manager.get_meta(dataset_id)
        assert meta.source == "derived" and meta.trust_level == "low"
