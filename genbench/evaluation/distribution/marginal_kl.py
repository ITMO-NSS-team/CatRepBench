from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import entropy

from genbench.data.schema import TabularSchema
from genbench.evaluation.base import BaseMetric
from genbench.evaluation.distribution._common import ensure_columns_view, selected_feature_columns


def _hist_prob(values: np.ndarray, bins: np.ndarray, eps: float) -> np.ndarray:
    hist, _ = np.histogram(values, bins=bins)
    probs = hist.astype(np.float64) + eps
    probs /= probs.sum()
    return probs


@dataclass
class MarginalKLDivergenceMetric(BaseMetric):
    """
    Mean marginal KL divergence across selected feature columns.
    """

    name: str = "marginal_kl_mean"
    n_bins: int = 20
    eps: float = 1e-8
    include_continuous: bool = True
    include_discrete: bool = True
    include_categorical: bool = True

    def compute(self, real: pd.DataFrame, synth: pd.DataFrame, schema: TabularSchema) -> float:
        cols = selected_feature_columns(
            schema,
            include_continuous=self.include_continuous,
            include_discrete=self.include_discrete,
            include_categorical=self.include_categorical,
        )
        real_feat = ensure_columns_view(real, cols)
        synth_feat = ensure_columns_view(synth, cols)

        kl_values: List[float] = []
        for c in cols:
            r = real_feat[c]
            s = synth_feat[c]

            if c in schema.categorical_cols or pd.api.types.is_object_dtype(r) or isinstance(r.dtype, pd.CategoricalDtype):
                categories = sorted(set(r.dropna().unique()).union(set(s.dropna().unique())))
                if not categories:
                    continue
                r_counts = r.value_counts(dropna=False)
                s_counts = s.value_counts(dropna=False)
                r_probs = np.array([r_counts.get(cat, 0) for cat in categories], dtype=float) + self.eps
                s_probs = np.array([s_counts.get(cat, 0) for cat in categories], dtype=float) + self.eps
                r_probs /= r_probs.sum()
                s_probs /= s_probs.sum()
                kl_values.append(float(entropy(r_probs, s_probs)))
                continue

            r_num = pd.to_numeric(r, errors="coerce").dropna()
            s_num = pd.to_numeric(s, errors="coerce").dropna()
            if r_num.empty or s_num.empty:
                continue

            r_min, r_max = float(r_num.min()), float(r_num.max())
            if r_max == r_min:
                continue
            bins = np.linspace(r_min, r_max, self.n_bins + 1)
            p = _hist_prob(r_num.to_numpy(), bins, self.eps)
            q = _hist_prob(s_num.to_numpy(), bins, self.eps)
            kl_values.append(float(entropy(p, q)))

        return float(np.mean(kl_values)) if kl_values else 0.0


def compute_marginal_kl_mean(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    schema: TabularSchema,
    n_bins: int = 20,
    eps: float = 1e-8,
) -> float:
    """
    Backward-compatible function API.
    """

    return MarginalKLDivergenceMetric(n_bins=n_bins, eps=eps).compute(real=real, synth=synth, schema=schema)
