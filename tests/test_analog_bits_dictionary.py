import numpy as np
import pandas as pd

from genbench.data.schema import TabularSchema
from genbench.representations.analog_bits.analog_bits import AnalogBitsRepresentation
from genbench.representations.dictionary.dictionary import DictionaryRepresentation
from genbench.transforms.categorical import list_registered_representations


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "color": ["red", "green", "blue", "red", "green", "blue", "yellow"],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        }
    )


def _schema(df: pd.DataFrame) -> TabularSchema:
    s = TabularSchema.infer_from_dataframe(df, target_col=None)
    assert "color" in s.categorical_cols
    return s


def test_registered() -> None:
    regs = list_registered_representations()
    assert "analog_bits_representation" in regs
    assert "dictionary_representation" in regs


def test_analog_bits_roundtrip() -> None:
    df = _df()
    rep = AnalogBitsRepresentation().fit(df, _schema(df))
    t = rep.transform(df)
    assert "color" not in t.columns and "x" in t.columns
    bitcols = [c for c in t.columns if c.startswith("color__bit")]
    assert len(bitcols) == 2                                  # K=4 -> 2 bits
    assert set(np.unique(t[bitcols].to_numpy())) <= {-1.0, 1.0}
    inv = rep.inverse_transform(t)
    assert list(inv["color"]) == list(df["color"])           # exact recovery
    assert list(inv["x"]) == list(df["x"])
    assert rep.is_invertible() is True


def test_dictionary_roundtrip() -> None:
    df = _df()
    rep = DictionaryRepresentation().fit(df, _schema(df))
    t = rep.transform(df)
    assert "color" in t.columns                              # 1-dim, replaced in place
    assert t["color"].between(-1.0, 1.0).all()
    inv = rep.inverse_transform(t)
    assert list(inv["color"]) == list(df["color"])


def test_state_roundtrip() -> None:
    df = _df()
    for cls in (AnalogBitsRepresentation, DictionaryRepresentation):
        rep = cls().fit(df, _schema(df))
        rep2 = cls.from_state(rep.get_state())
        pd.testing.assert_frame_equal(rep.transform(df), rep2.transform(df))


def test_dictionary_inverse_transform_handles_nonfinite_deterministically() -> None:
    import warnings

    df = _df()
    rep = DictionaryRepresentation().fit(df, _schema(df))
    bad = pd.DataFrame({"color": [np.nan, np.inf, -np.inf], "x": [1.0, 2.0, 3.0]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = rep.inverse_transform(bad)
    assert set(out["color"]).issubset(set(df["color"]))
