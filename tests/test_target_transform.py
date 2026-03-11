from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from genbench.data.schema import TabularSchema
from genbench.transforms.target import TargetTypePreprocessor, infer_is_regression_target


def _schema() -> TabularSchema:
    return TabularSchema(
        continuous_cols=["x_cont"],
        discrete_cols=["x_disc"],
        categorical_cols=["x_cat"],
        target_col="target",
    )


def test_target_preprocessor_label_encodes_categorical_target_for_classification():
    df = pd.DataFrame(
        {
            "x_cont": [0.1, 0.2, 0.3, 0.4],
            "x_disc": [1, 2, 1, 2],
            "x_cat": ["a", "b", "a", "b"],
            "target": ["yes", "no", "yes", "no"],
        }
    )
    schema = _schema()
    tr = TargetTypePreprocessor(task_type="classification")

    tr.fit(df, schema)
    out = tr.transform(df)

    assert tr.did_encode_ is True
    assert tr.did_scale_ is False
    assert pd.api.types.is_integer_dtype(out["target"])


def test_target_preprocessor_raises_on_unseen_target_category():
    train = pd.DataFrame(
        {
            "x_cont": [0.1, 0.2, 0.3],
            "x_disc": [1, 2, 1],
            "x_cat": ["a", "b", "a"],
            "target": ["yes", "no", "yes"],
        }
    )
    val = pd.DataFrame(
        {
            "x_cont": [0.4],
            "x_disc": [2],
            "x_cat": ["b"],
            "target": ["maybe"],
        }
    )
    schema = _schema()
    tr = TargetTypePreprocessor(task_type="classification")
    tr.fit(train, schema)

    with pytest.raises(ValueError, match="not present in train"):
        tr.transform(val)


def test_target_preprocessor_scales_numeric_target_for_regression():
    df = pd.DataFrame(
        {
            "x_cont": [0.1, 0.2, 0.3, 0.4],
            "x_disc": [1, 2, 1, 2],
            "x_cat": ["a", "b", "a", "b"],
            "target": [10.0, 20.0, 30.0, 40.0],
        }
    )
    schema = _schema()
    tr = TargetTypePreprocessor(task_type="regression")
    tr.fit(df, schema)
    out = tr.transform(df)

    assert tr.did_scale_ is True
    assert tr.did_encode_ is False
    assert np.isclose(float(out["target"].mean()), 0.0)
    assert np.isclose(float(out["target"].std(ddof=0)), 1.0)


def test_infer_is_regression_target_uses_threshold_20():
    y_small = pd.Series([float(i % 10) for i in range(100)])
    y_large = pd.Series([float(i) for i in range(100)])

    assert infer_is_regression_target(y_small, discrete_unique_threshold=20) is False
    assert infer_is_regression_target(y_large, discrete_unique_threshold=20) is True
