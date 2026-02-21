from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler

from genbench.data.schema import TabularSchema
from genbench.evaluation.base import BaseMetric
from genbench.evaluation.distribution._common import ensure_feature_view


@dataclass
class WassersteinDistanceMetric(BaseMetric):
    """
    Mean 1D Wasserstein distance over numeric features in normalized space.
    """

    name: str = "wasserstein_mean"

    def compute(self, real: pd.DataFrame, synth: pd.DataFrame, schema: TabularSchema) -> float:
        cols = list(schema.continuous_cols) + list(schema.discrete_cols)
        if not cols:
            raise ValueError("No numeric columns (continuous+discrete) available for Wasserstein distance.")

        real_num = ensure_feature_view(real, schema)[cols].astype(float)
        synth_num = ensure_feature_view(synth, schema)[cols].astype(float)

        scaler = StandardScaler()
        real_scaled = pd.DataFrame(scaler.fit_transform(real_num), columns=cols)
        synth_scaled = pd.DataFrame(scaler.transform(synth_num), columns=cols)

        distances: List[float] = []
        for c in cols:
            distances.append(float(wasserstein_distance(real_scaled[c], synth_scaled[c])))
        return float(np.mean(distances)) if distances else 0.0


def compute_wasserstein_mean(real: pd.DataFrame, synth: pd.DataFrame, schema: TabularSchema) -> float:
    """
    Backward-compatible function API.
    """

    return WassersteinDistanceMetric().compute(real=real, synth=synth, schema=schema)
