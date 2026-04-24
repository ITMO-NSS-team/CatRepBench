from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd

from genbench.data.schema import TabularSchema


def ensure_feature_view(df: pd.DataFrame, schema: TabularSchema) -> pd.DataFrame:
    return ensure_columns_view(df, schema.feature_cols)


def ensure_columns_view(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required feature columns: {missing}")
    return df[list(columns)].copy()


def selected_feature_columns(
    schema: TabularSchema,
    *,
    include_continuous: bool = True,
    include_discrete: bool = True,
    include_categorical: bool = True,
) -> List[str]:
    cols: List[str] = []
    if include_continuous:
        cols.extend(schema.continuous_cols)
    if include_discrete:
        cols.extend(schema.discrete_cols)
    if include_categorical:
        cols.extend(schema.categorical_cols)
    return cols


def mixed_to_numeric_matrix(
    df: pd.DataFrame,
    schema: TabularSchema,
    *,
    columns: Sequence[str] | None = None,
) -> np.ndarray:
    """
    Convert mixed feature columns to numeric arrays for matrix-based metrics.
    """
    data: List[np.ndarray] = []
    selected_cols = list(columns) if columns is not None else schema.feature_cols
    for c in selected_cols:
        col = df[c]
        if c in schema.categorical_cols or pd.api.types.is_object_dtype(col) or isinstance(col.dtype, pd.CategoricalDtype):
            codes, uniques = pd.factorize(col, sort=True)
            codes = codes.astype(float)
            if len(uniques) > 0:
                fill_value = float(codes[codes >= 0].mean()) if (codes >= 0).any() else 0.0
            else:
                fill_value = 0.0
            codes = np.where(codes < 0, fill_value, codes)
            data.append(codes)
        else:
            num = pd.to_numeric(col, errors="coerce").astype(float)
            fill_value = float(np.nanmean(num)) if not np.all(np.isnan(num)) else 0.0
            num = np.nan_to_num(num, nan=fill_value)
            data.append(num)
    if not data:
        return np.empty((0, len(df)), dtype=float)
    return np.vstack(data)
