from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from genbench.data.schema import TabularSchema
from genbench.evaluation.base import BaseMetric
from genbench.evaluation.distribution._common import ensure_feature_view, mixed_to_numeric_matrix


@dataclass
class CorrelationFrobeniusMetric(BaseMetric):
    """
    Frobenius norm of the difference between real and synthetic correlation matrices.
    """

    name: str = "corr_frobenius"

    def compute(self, real: pd.DataFrame, synth: pd.DataFrame, schema: TabularSchema) -> float:
        real_feat = ensure_feature_view(real, schema)
        synth_feat = ensure_feature_view(synth, schema)

        real_mat = mixed_to_numeric_matrix(real_feat, schema)
        synth_mat = mixed_to_numeric_matrix(synth_feat, schema)

        if real_mat.shape[0] == 0 or synth_mat.shape[0] == 0:
            return 0.0

        real_corr = np.corrcoef(real_mat)
        synth_corr = np.corrcoef(synth_mat)

        real_corr = np.nan_to_num(real_corr, nan=0.0)
        synth_corr = np.nan_to_num(synth_corr, nan=0.0)
        return float(np.linalg.norm(real_corr - synth_corr, ord="fro"))


def compute_corr_frobenius(real: pd.DataFrame, synth: pd.DataFrame, schema: TabularSchema) -> float:
    """
    Backward-compatible function API.
    """

    return CorrelationFrobeniusMetric().compute(real=real, synth=synth, schema=schema)
