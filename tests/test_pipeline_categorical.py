import pandas as pd
import pytest

from genbench.data.datamodule import TabularDataModule
from genbench.data.schema import TabularSchema
from genbench.data.splits import SplitConfigHoldout, SplitConfigKFold
from genbench.transforms.categorical import CategoricalRepresentationTransform
from genbench.transforms.continuous import ContinuousStandardScaler
from genbench.transforms.missing import DropMissingRows
from genbench.transforms.pipeline import TransformPipeline


def test_kfold_with_categorical_representation_transform():
    df = pd.DataFrame(
        {
            "cat": ["a", "b", "a", "a", "b", "c"],
            "cont": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "disc": [0, 1, 0, 1, 0, 1],
        }
    )
    schema = TabularSchema(
        continuous_cols=["cont"],
        discrete_cols=["disc"],
        categorical_cols=["cat"],
    )

    pipeline = TransformPipeline(
        transforms=[
            DropMissingRows(),
            ContinuousStandardScaler(),
            CategoricalRepresentationTransform(
                representation_name="one_hot_representation",
            ),
        ]
    )
    dm = TabularDataModule(df=df, schema=schema, transforms=pipeline)

    dm.prepare_kfold(SplitConfigKFold(n_splits=2, shuffle=False, random_seed=None))
    fold = dm.get_fold(0)

    assert fold.transforms is not None
    assert "cat" not in fold.train.columns
    assert any(col.startswith("cat__") for col in fold.train.columns)


def test_kfold_with_polynomial_representation_transform():
    pytest.importorskip("category_encoders")

    df = pd.DataFrame(
        {
            "cat": ["a", "b", "a", "c", "b", "c"],
            "cont": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "disc": [0, 1, 0, 1, 0, 1],
        }
    )
    schema = TabularSchema(
        continuous_cols=["cont"],
        discrete_cols=["disc"],
        categorical_cols=["cat"],
    )

    pipeline = TransformPipeline(
        transforms=[
            DropMissingRows(),
            ContinuousStandardScaler(),
            CategoricalRepresentationTransform(
                representation_name="polynomial_representation",
            ),
        ]
    )
    dm = TabularDataModule(df=df, schema=schema, transforms=pipeline)

    dm.prepare_kfold(SplitConfigKFold(n_splits=3, shuffle=True, random_seed=42))
    fold = dm.get_fold(0)

    assert fold.transforms is not None
    assert "cat" not in fold.train.columns
    assert any(col.startswith("cat_") for col in fold.train.columns)


@pytest.mark.parametrize(
    "representation_name",
    [
        "sum_representation",
        "helmert_representation",
        "backward_difference_representation",
        "binary_representation",
    ],
)
def test_kfold_with_other_category_encoder_representations(representation_name: str):
    pytest.importorskip("category_encoders")

    df = pd.DataFrame(
        {
            "cat": ["a", "b", "a", "c", "b", "c"],
            "cont": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "disc": [0, 1, 0, 1, 0, 1],
        }
    )
    schema = TabularSchema(
        continuous_cols=["cont"],
        discrete_cols=["disc"],
        categorical_cols=["cat"],
    )

    pipeline = TransformPipeline(
        transforms=[
            DropMissingRows(),
            ContinuousStandardScaler(),
            CategoricalRepresentationTransform(
                representation_name=representation_name,
            ),
        ]
    )
    dm = TabularDataModule(df=df, schema=schema, transforms=pipeline)

    dm.prepare_kfold(SplitConfigKFold(n_splits=3, shuffle=True, random_seed=42))
    fold = dm.get_fold(0)

    assert fold.transforms is not None
    assert "cat" not in fold.train.columns
    assert any(col.startswith("cat_") for col in fold.train.columns)


def test_kfold_raises_when_test_has_unseen_category():
    df = pd.DataFrame(
        {
            "cat": ["a", "a", "a", "z"],
            "cont": [1.0, 2.0, 3.0, 4.0],
            "disc": [0, 1, 0, 1],
        }
    )
    schema = TabularSchema(
        continuous_cols=["cont"],
        discrete_cols=["disc"],
        categorical_cols=["cat"],
    )

    pipeline = TransformPipeline(
        transforms=[
            DropMissingRows(),
            ContinuousStandardScaler(),
            CategoricalRepresentationTransform(representation_name="one_hot_representation"),
        ]
    )
    dm = TabularDataModule(df=df, schema=schema, transforms=pipeline)

    dm.prepare_kfold(SplitConfigKFold(n_splits=2, shuffle=False, random_seed=None))

    with pytest.raises(ValueError, match="not present in train"):
        dm.get_fold(1)


def test_holdout_raises_when_validation_has_unseen_category():
    df = pd.DataFrame(
        {
            "cat": ["a", "a", "a", "z"],
            "cont": [1.0, 2.0, 3.0, 4.0],
            "disc": [0, 1, 0, 1],
        }
    )
    schema = TabularSchema(
        continuous_cols=["cont"],
        discrete_cols=["disc"],
        categorical_cols=["cat"],
    )

    pipeline = TransformPipeline(
        transforms=[
            DropMissingRows(),
            ContinuousStandardScaler(),
            CategoricalRepresentationTransform(representation_name="one_hot_representation"),
        ]
    )
    dm = TabularDataModule(df=df, schema=schema, transforms=pipeline)

    dm.prepare_holdout(SplitConfigHoldout(val_size=0.25, shuffle=False, random_seed=None))

    with pytest.raises(ValueError, match="not present in train"):
        dm.get_holdout()


def test_missing_rows_are_dropped_split_wise_in_holdout():
    df = pd.DataFrame(
        {
            "cont": [1.0, 2.0, 3.0, None],
            "disc": [0, 1, 0, 1],
        }
    )
    schema = TabularSchema(
        continuous_cols=["cont"],
        discrete_cols=["disc"],
        categorical_cols=[],
    )

    pipeline = TransformPipeline(
        transforms=[
            DropMissingRows(),
            ContinuousStandardScaler(),
        ]
    )
    dm = TabularDataModule(df=df, schema=schema, transforms=pipeline)

    dm.prepare_holdout(SplitConfigHoldout(val_size=0.25, shuffle=False, random_seed=None))
    holdout = dm.get_holdout()

    assert len(holdout.train) == 3
    assert len(holdout.val) == 0
