from __future__ import annotations

import pandas as pd

from genbench.data.schema import TabularSchema
from genbench.transforms.continuous import ContinuousStandardScaler


def test_schema_infers_string_and_bool_as_categorical():
    df = pd.DataFrame(
        {
            "x_string": pd.Series(["a", "b", "a"], dtype="string"),
            "x_bool": [True, False, True],
            "x_cont": [0.1, 0.2, 0.3],
        }
    )

    schema = TabularSchema.infer_from_dataframe(df)

    assert "x_string" in schema.categorical_cols
    assert "x_bool" in schema.categorical_cols
    assert "x_cont" in schema.continuous_cols


def test_schema_infers_numeric_columns_by_fractionality_and_cardinality():
    df = pd.DataFrame(
        {
            "x_float_fractional": [float(i) + 0.5 for i in range(25)],
            "x_float_integer_small": [float(i % 3) for i in range(25)],
            "x_int_large": list(range(25)),
        }
    )

    schema = TabularSchema.infer_from_dataframe(df, discrete_max_unique=20)

    assert "x_float_fractional" in schema.continuous_cols
    assert "x_float_integer_small" in schema.discrete_cols
    assert "x_int_large" in schema.continuous_cols


def test_continuous_standard_scaler_is_noop_when_no_continuous_columns():
    df = pd.DataFrame({"x_disc": [0, 1, 0], "x_cat": ["a", "b", "a"]})
    schema = TabularSchema(continuous_cols=[], discrete_cols=["x_disc"], categorical_cols=["x_cat"])
    scaler = ContinuousStandardScaler()

    scaler.fit(df, schema)
    out = scaler.transform(df)

    pd.testing.assert_frame_equal(out, df)
