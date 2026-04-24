from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler

from genbench.data.schema import TabularSchema
from genbench.evaluation.base import BaseMetric
from genbench.evaluation.distribution._common import ensure_columns_view, selected_feature_columns


@dataclass
class WassersteinDistanceMetric(BaseMetric):
    """
    Mean 1D Wasserstein distance over selected numeric features in normalized space.
    """

    name: str = "wasserstein_mean"
    include_continuous: bool = True
    include_discrete: bool = True

    def compute(self, real: pd.DataFrame, synth: pd.DataFrame, schema: TabularSchema) -> float:
        cols = selected_feature_columns(
            schema,
            include_continuous=self.include_continuous,
            include_discrete=self.include_discrete,
            include_categorical=False,
        )
        if not cols:
            return 0.0

        real_num = ensure_columns_view(real, cols).astype(float)
        synth_num = ensure_columns_view(synth, cols).astype(float)

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
