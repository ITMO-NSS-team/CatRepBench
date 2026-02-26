import json

import numpy as np
import pandas as pd
import pytest

from genbench.data.schema import TabularSchema
from genbench.representations.base import RepresentationState
from genbench.representations.binary.binary import BinaryRepresentation
from genbench.representations.backward_difference.backward_difference import (
    BackwardDifferenceRepresentation,
)
from genbench.representations.helmert.helmert import HelmertRepresentation
from genbench.representations.sum.sum import SumRepresentation


pytest.importorskip("category_encoders")


REPRESENTATION_CLASSES = [
    SumRepresentation,
    HelmertRepresentation,
    BackwardDifferenceRepresentation,
    BinaryRepresentation,
]


def _make_data() -> tuple[pd.DataFrame, pd.DataFrame, TabularSchema]:
    train_df = pd.DataFrame(
        {
            "color": ["red", "blue", "green", "red"],
            "size": ["small", "medium", "large", "small"],
            "price": [10.0, 15.0, 20.0, 12.0],
        }
    ).astype({"color": "object", "size": "object"})

    test_df = pd.DataFrame(
        {
            "color": ["purple", "red", None],
            "size": ["tiny", "small", None],
            "price": [11.0, 13.0, 14.0],
        }
    ).astype({"color": "object", "size": "object"})

    schema = TabularSchema.infer_from_dataframe(
        train_df,
        categorical_cols=["color", "size"],
        continuous_cols=["price"],
    )
    return train_df, test_df, schema


@pytest.mark.parametrize("rep_cls", REPRESENTATION_CLASSES)
def test_category_encoder_wrapper_interface(rep_cls):
    train_df, _, schema = _make_data()
    rep = rep_cls()

    assert rep.requires_fit() is True
    assert rep.is_invertible() is False

    with pytest.raises(RuntimeError):
        rep.transform(train_df)

    rep.fit(train_df, schema)
    out = rep.transform(train_df)
    assert isinstance(out, pd.DataFrame)
    assert any(col.startswith("color_") for col in out.columns)
    assert any(col.startswith("size_") for col in out.columns)


@pytest.mark.parametrize("rep_cls", REPRESENTATION_CLASSES)
def test_category_encoder_wrapper_state_roundtrip(rep_cls):
    train_df, test_df, schema = _make_data()
    rep = rep_cls(drop_original_categoricals=False)
    rep.fit(train_df, schema)

    state = rep.get_state()
    restored = rep_cls.from_state(state)
    out_1 = rep.transform(test_df)
    out_2 = restored.transform(test_df)

    assert list(out_1.columns) == list(out_2.columns)
    np.testing.assert_allclose(
        out_1.select_dtypes(include=["number"]).to_numpy(dtype=float),
        out_2.select_dtypes(include=["number"]).to_numpy(dtype=float),
        equal_nan=True,
    )

    payload = {"name": state.name, "params": state.params}
    payload_json = json.loads(json.dumps(payload))
    restored_json = rep_cls.from_state(
        RepresentationState(
            name=payload_json["name"],
            params=payload_json["params"],
        )
    )
    out_3 = restored_json.transform(test_df)
    assert list(out_1.columns) == list(out_3.columns)
    np.testing.assert_allclose(
        out_1.select_dtypes(include=["number"]).to_numpy(dtype=float),
        out_3.select_dtypes(include=["number"]).to_numpy(dtype=float),
        equal_nan=True,
    )
