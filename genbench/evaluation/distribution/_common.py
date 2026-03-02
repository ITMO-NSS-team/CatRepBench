from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from genbench.data.schema import TabularSchema


def ensure_feature_view(df: pd.DataFrame, schema: TabularSchema) -> pd.DataFrame:
    missing = [c for c in schema.feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required feature columns: {missing}")
    return df[schema.feature_cols].copy()


def mixed_to_numeric_matrix(df: pd.DataFrame, schema: TabularSchema) -> np.ndarray:
    """
    Convert mixed feature columns to numeric arrays for matrix-based metrics.
    """
    data: List[np.ndarray] = []
    for c in schema.feature_cols:
        col = df[c]
        if c in schema.categorical_cols or pd.api.types.is_object_dtype(col) or pd.api.types.is_categorical_dtype(col):
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
    return np.vstack(data)
