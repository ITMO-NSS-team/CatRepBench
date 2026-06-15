from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from genbench.data.schema import TabularSchema
from genbench.representations.base import RepresentationState


def _safe_str(x: object) -> str:
    if x is None:
        return "__NONE__"
    if isinstance(x, float) and pd.isna(x):
        return "__NAN__"
    return str(x)


@dataclass
class DictionaryRepresentation:
    """
    Dictionary representation for categorical columns.

    Maps each categorical column with K observed levels to a SINGLE continuous
    column of equally-spaced scalar codes in ``[-1, 1]``: level i -> -1 + 2i/(K-1)
    (0.0 if K == 1). Continuous/discrete columns are kept as-is. Unsupervised,
    deterministic, exactly invertible (nearest grid point). Used as a 1-dim
    continuous baseline for diffusion/flow generators (TabRep/TabUnite); differs
    from ``ordinal`` (raw integer index) by being range-normalized for a
    diffusion target. Note: like ordinal, it imposes a (spurious) order on
    nominal levels.
    """

    name: str = "dictionary_representation"

    fitted_: bool = False
    categorical_cols_: List[str] = field(default_factory=list)
    vocab_: Dict[str, List[str]] = field(default_factory=dict)

    @staticmethod
    def requires_fit() -> bool:
        return True

    @staticmethod
    def is_invertible() -> bool:
        return True

    def fit(self, df: pd.DataFrame, schema: TabularSchema) -> "DictionaryRepresentation":
        cat_cols = list(schema.categorical_cols)
        self.categorical_cols_ = cat_cols
        self.vocab_ = {c: sorted({_safe_str(v) for v in df[c].tolist()}) for c in cat_cols}
        self.fitted_ = True
        return self

    @staticmethod
    def _codes(k: int) -> np.ndarray:
        if k <= 1:
            return np.zeros(max(k, 1), dtype=float)
        return np.linspace(-1.0, 1.0, k)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("DictionaryRepresentation must be fitted before transform().")
        out = df.copy()
        for c in self.categorical_cols_:
            if c not in out.columns:
                raise KeyError(f"Categorical column '{c}' not found in DataFrame.")
            levels = self.vocab_[c]
            codes = self._codes(len(levels))
            index = {lvl: i for i, lvl in enumerate(levels)}
            out[c] = [float(codes[index.get(_safe_str(v), 0)]) for v in out[c].tolist()]
        return out

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("DictionaryRepresentation must be fitted before inverse_transform().")
        out = df.copy()
        for c in self.categorical_cols_:
            if c not in out.columns:
                continue
            levels = self.vocab_[c]
            k = len(levels)
            if k == 0:
                continue
            # Sanitize non-finite synthetic values to the code range [-1, 1] so
            # the linear map + int cast below is deterministic and warning-free
            # (nan_to_num's default maps inf to a huge finite that would overflow).
            vals = np.nan_to_num(
                out[c].to_numpy(dtype=float), nan=0.0, posinf=1.0, neginf=-1.0
            )
            if k == 1:
                idx = np.zeros(len(vals), dtype=int)
            else:
                idx = np.clip(np.rint((vals + 1.0) / 2.0 * (k - 1)).astype(int), 0, k - 1)
            out[c] = [levels[i] for i in idx]
        return out

    def get_state(self) -> RepresentationState:
        return RepresentationState(
            name=self.name,
            params={
                "fitted": self.fitted_,
                "categorical_cols": self.categorical_cols_,
                "vocab": self.vocab_,
            },
        )

    @classmethod
    def from_state(cls, state: RepresentationState) -> "DictionaryRepresentation":
        obj = cls()
        obj.fitted_ = bool(state.params.get("fitted", False))
        obj.categorical_cols_ = list(state.params.get("categorical_cols", []))
        obj.vocab_ = {
            str(col): [str(v) for v in levels]
            for col, levels in dict(state.params.get("vocab", {})).items()
        }
        return obj
