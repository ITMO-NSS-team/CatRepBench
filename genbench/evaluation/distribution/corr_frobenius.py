from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from genbench.data.schema import TabularSchema
from genbench.evaluation.base import BaseMetric
from genbench.evaluation.distribution._common import (
    ensure_columns_view,
    mixed_to_numeric_matrix,
    selected_feature_columns,
)


def _correlation_matrix(matrix: np.ndarray, *, method: str) -> np.ndarray:
    if matrix.shape[0] == 0:
        return np.empty((0, 0), dtype=float)
    if matrix.shape[1] == 0:
        return np.zeros((matrix.shape[0], matrix.shape[0]), dtype=float)
    if matrix.shape[0] == 1:
        return np.ones((1, 1), dtype=float)
    if method == "pearson":
        return np.asarray(np.corrcoef(matrix), dtype=float)
    if method == "spearman":
        return pd.DataFrame(matrix.T).corr(method="spearman").to_numpy(dtype=float)
    raise ValueError("Correlation method must be 'spearman' or 'pearson'.")


@dataclass
class CorrelationFrobeniusMetric(BaseMetric):
    """
    Frobenius norm of the difference between real and synthetic correlation matrices.

    Uses Spearman correlation by default so discrete-valued features can be compared
    without assuming linear numeric spacing.
    """

    name: str = "corr_frobenius"
    include_continuous: bool = True
    include_discrete: bool = True
    include_categorical: bool = True
    method: str = "spearman"

    def compute(self, real: pd.DataFrame, synth: pd.DataFrame, schema: TabularSchema) -> float:
        cols = selected_feature_columns(
            schema,
            include_continuous=self.include_continuous,
            include_discrete=self.include_discrete,
            include_categorical=self.include_categorical,
        )
        real_feat = ensure_columns_view(real, cols)
        synth_feat = ensure_columns_view(synth, cols)

        real_mat = mixed_to_numeric_matrix(real_feat, schema, columns=cols)
        synth_mat = mixed_to_numeric_matrix(synth_feat, schema, columns=cols)

        if real_mat.shape[0] == 0 or synth_mat.shape[0] == 0:
            return 0.0

        real_corr = _correlation_matrix(real_mat, method=self.method)
        synth_corr = _correlation_matrix(synth_mat, method=self.method)

        real_corr = np.nan_to_num(real_corr, nan=0.0)
        synth_corr = np.nan_to_num(synth_corr, nan=0.0)
        return float(np.linalg.norm(real_corr - synth_corr, ord="fro"))


def compute_corr_frobenius(real: pd.DataFrame, synth: pd.DataFrame, schema: TabularSchema) -> float:
    """
    Backward-compatible function API.
    """

    return CorrelationFrobeniusMetric().compute(real=real, synth=synth, schema=schema)
