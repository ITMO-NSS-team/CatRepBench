from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


FeatureType = Literal["continuous", "discrete", "categorical"]


def is_categorical_like_dtype(series: pd.Series) -> bool:
    return bool(
        pd.api.types.is_object_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_string_dtype(series)
        or pd.api.types.is_bool_dtype(series)
    )


def is_integer_valued_numeric(series: pd.Series, *, atol: float = 1e-12) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return False
    if pd.api.types.is_integer_dtype(series):
        return True
    if not pd.api.types.is_numeric_dtype(series):
        return False

    values = series.dropna()
    if values.empty:
        return False

    values_f = values.astype("float64").to_numpy()
    return bool(np.all(np.isclose(values_f, np.round(values_f), atol=atol, rtol=0.0)))


def infer_feature_type(
    series: pd.Series,
    *,
    discrete_max_unique: int = 20,
    treat_bool_as_categorical: bool = True,
) -> FeatureType:
    if pd.api.types.is_bool_dtype(series):
        return "categorical" if treat_bool_as_categorical else "discrete"

    if is_categorical_like_dtype(series):
        return "categorical"

    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError(
            f"Cannot infer feature type for series with dtype '{series.dtype}'. "
            "Please cast it or specify the feature type explicitly."
        )

    values = series.dropna()
    if values.empty:
        return "continuous"

    if is_integer_valued_numeric(series):
        return "discrete" if int(values.nunique()) < discrete_max_unique else "continuous"

    return "continuous"
