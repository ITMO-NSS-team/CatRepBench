import json

import numpy as np
import pandas as pd
import pytest

from genbench.data.schema import TabularSchema
from genbench.representations.base import RepresentationState
from genbench.representations.similarity.similarity import SimilarityRepresentation
from genbench.transforms.categorical import list_registered_representations


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
            "color": ["grean", "red", None],
            "size": ["smoll", "small", None],
            "price": [11.0, 13.0, 14.0],
        }
    ).astype({"color": "object", "size": "object"})

    schema = TabularSchema.infer_from_dataframe(
        train_df,
        categorical_cols=["color", "size"],
        continuous_cols=["price"],
    )
    return train_df, test_df, schema


def test_similarity_representation_interface():
    train_df, _, schema = _make_data()
    rep = SimilarityRepresentation()

    assert rep.requires_fit() is True
    assert rep.is_invertible() is False

    with pytest.raises(RuntimeError):
        rep.transform(train_df)

    rep.fit(train_df, schema)
    out = rep.transform(train_df)

    assert isinstance(out, pd.DataFrame)
    assert "color" not in out.columns
    assert "size" not in out.columns
    assert out["price"].tolist() == train_df["price"].tolist()
    assert any(col.startswith("color__sim_") for col in out.columns)
    assert any(col.startswith("size__sim_") for col in out.columns)

    red_idx = rep.prototypes_["color"].index("red")
    red_col = rep.out_cols_["color"][red_idx]
    assert out.loc[0, red_col] == pytest.approx(1.0)


def test_similarity_representation_state_roundtrip():
    train_df, test_df, schema = _make_data()
    rep = SimilarityRepresentation(
        ngram_range=(2, 3),
        max_prototypes=2,
        drop_original_categoricals=False,
    )
    rep.fit(train_df, schema)

    state = rep.get_state()
    restored = SimilarityRepresentation.from_state(state)
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
    restored_json = SimilarityRepresentation.from_state(
        RepresentationState(
            name=payload_json["name"],
            params=payload_json["params"],
        )
    )
    out_3 = restored_json.transform(test_df)
    np.testing.assert_allclose(
        out_1.select_dtypes(include=["number"]).to_numpy(dtype=float),
        out_3.select_dtypes(include=["number"]).to_numpy(dtype=float),
        equal_nan=True,
    )


def test_similarity_representation_prefers_closest_ngram_match():
    train_df, test_df, schema = _make_data()
    rep = SimilarityRepresentation(ngram_range=(2, 3))
    rep.fit(train_df, schema)

    transformed = rep.transform(test_df)
    green_idx = rep.prototypes_["color"].index("green")
    blue_idx = rep.prototypes_["color"].index("blue")
    green_col = rep.out_cols_["color"][green_idx]
    blue_col = rep.out_cols_["color"][blue_idx]

    assert transformed.loc[0, green_col] > transformed.loc[0, blue_col]
    assert transformed.loc[2, green_col] >= 0.0
    assert transformed.loc[2, green_col] <= 1.0


def test_similarity_representation_validates_configuration():
    train_df, _, schema = _make_data()

    with pytest.raises(ValueError, match="ngram_range"):
        SimilarityRepresentation(ngram_range=(3, 2)).fit(train_df, schema)

    with pytest.raises(ValueError, match="max_prototypes"):
        SimilarityRepresentation(max_prototypes=0).fit(train_df, schema)


def test_similarity_representation_is_registered():
    assert "similarity_representation" in list_registered_representations()
