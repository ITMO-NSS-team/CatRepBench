import pandas as pd

from genbench.data.schema import TabularSchema
from genbench.representations.drop_categoricals.drop_categoricals import (
    DropCategoricalsRepresentation,
)
from genbench.transforms.categorical import list_registered_representations


def test_registered() -> None:
    assert "drop_categoricals_representation" in list_registered_representations()


def test_drops_categorical_features_keeps_continuous_and_target() -> None:
    df = pd.DataFrame(
        {
            "color": ["red", "green", "blue", "red"],
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": ["a", "b", "a", "b"],  # target (categorical)
        }
    )
    schema = TabularSchema.infer_from_dataframe(df, target_col="y")
    rep = DropCategoricalsRepresentation().fit(df, schema)
    t = rep.transform(df)
    assert "color" not in t.columns      # categorical FEATURE dropped
    assert "x" in t.columns              # continuous kept
    assert "y" in t.columns              # target kept (never dropped)
    assert rep.is_invertible() is False


def test_state_roundtrip() -> None:
    df = pd.DataFrame({"color": ["red", "green"], "x": [1.0, 2.0], "y": ["a", "b"]})
    schema = TabularSchema.infer_from_dataframe(df, target_col="y")
    rep = DropCategoricalsRepresentation().fit(df, schema)
    rep2 = DropCategoricalsRepresentation.from_state(rep.get_state())
    pd.testing.assert_frame_equal(rep.transform(df), rep2.transform(df))
