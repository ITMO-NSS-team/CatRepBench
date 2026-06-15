import numpy as np
import pandas as pd

from genbench.data.schema import TabularSchema
from genbench.representations.tabrep.tabrep import TabRepRepresentation
from genbench.transforms.categorical import (
    _REPRESENTATION_REGISTRY,
    list_registered_representations,
)


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "color": ["red", "green", "blue", "red", "green", "blue"],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )


def _schema(df: pd.DataFrame) -> TabularSchema:
    schema = TabularSchema.infer_from_dataframe(df, target_col=None)
    assert "color" in schema.categorical_cols
    return schema


def test_registered() -> None:
    assert "tabrep_representation" in list_registered_representations()
    assert _REPRESENTATION_REGISTRY["tabrep_representation"] is TabRepRepresentation


def test_transform_shape_and_unit_circle() -> None:
    df = _df()
    rep = TabRepRepresentation().fit(df, _schema(df))
    t = rep.transform(df)
    assert "color" not in t.columns          # original categorical dropped
    assert "x" in t.columns                  # continuous preserved
    assert {"color__tabrep_cos", "color__tabrep_sin"} <= set(t.columns)
    r2 = t["color__tabrep_cos"] ** 2 + t["color__tabrep_sin"] ** 2
    assert np.allclose(r2.to_numpy(), 1.0)   # every category lands on the unit circle
    assert rep.is_invertible() is True
    assert rep.requires_fit() is True


def test_roundtrip_decode() -> None:
    df = _df()
    rep = TabRepRepresentation().fit(df, _schema(df))
    inv = rep.inverse_transform(rep.transform(df))
    assert list(inv["color"]) == list(df["color"])   # exact category recovery
    assert list(inv["x"]) == list(df["x"])           # continuous untouched


def test_unknown_level_maps_to_first() -> None:
    df = _df()
    rep = TabRepRepresentation().fit(df, _schema(df))
    probe = pd.DataFrame({"color": ["__never_seen__"], "x": [9.0]})
    t = rep.transform(probe)
    # unseen -> index 0 -> angle 0 -> (cos, sin) = (1, 0)
    assert np.isclose(t["color__tabrep_cos"].iloc[0], 1.0)
    assert np.isclose(t["color__tabrep_sin"].iloc[0], 0.0)


def test_state_roundtrip() -> None:
    df = _df()
    rep = TabRepRepresentation().fit(df, _schema(df))
    rep2 = TabRepRepresentation.from_state(rep.get_state())
    pd.testing.assert_frame_equal(rep.transform(df), rep2.transform(df))


def test_inverse_transform_handles_nonfinite_values_deterministically() -> None:
    import warnings

    df = _df()
    rep = TabRepRepresentation().fit(df, _schema(df))
    cos_col, sin_col = "color__tabrep_cos", "color__tabrep_sin"
    bad = pd.DataFrame(
        {cos_col: [np.nan, np.inf, -np.inf, 0.0], sin_col: [np.nan, 0.0, np.nan, np.inf], "x": [1.0, 2.0, 3.0, 4.0]}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any RuntimeWarning becomes a failure
        out = rep.inverse_transform(bad)
    assert list(out["color"]) == [lvl for lvl in out["color"]]  # no NaN/None
    assert set(out["color"]).issubset(set(df["color"]))         # all valid in-vocab
