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
class AnalogBitsRepresentation:
    """
    Analog-bits representation for categorical columns (bit-diffusion encoding).

    Each categorical column with K observed levels encodes the level index as its
    ``m = max(1, ceil(log2 K))`` binary bits, cast to real **±1** (bit 1 -> +1,
    bit 0 -> -1). Continuous/discrete columns are kept as-is. Unsupervised,
    deterministic, exactly invertible (decode by thresholding at 0). This is the
    standard categorical baseline for continuous-state diffusion generators
    (Chen et al., "Analog Bits"; used in TabUnite/TabRep). Distinct from the
    ``binary`` encoder: bits are real ±1 engineered as a denoising target rather
    than 0/1 integer columns.
    """

    name: str = "analog_bits_representation"
    bit_suffix: str = "__bit"

    fitted_: bool = False
    categorical_cols_: List[str] = field(default_factory=list)
    vocab_: Dict[str, List[str]] = field(default_factory=dict)
    nbits_: Dict[str, int] = field(default_factory=dict)

    @staticmethod
    def requires_fit() -> bool:
        return True

    @staticmethod
    def is_invertible() -> bool:
        return True

    def fit(self, df: pd.DataFrame, schema: TabularSchema) -> "AnalogBitsRepresentation":
        cat_cols = list(schema.categorical_cols)
        self.categorical_cols_ = cat_cols
        vocab: Dict[str, List[str]] = {}
        nbits: Dict[str, int] = {}
        for c in cat_cols:
            levels = sorted({_safe_str(v) for v in df[c].tolist()})
            vocab[c] = levels
            k = len(levels)
            nbits[c] = max(1, int(np.ceil(np.log2(k))) if k > 1 else 1)
        self.vocab_ = vocab
        self.nbits_ = nbits
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("AnalogBitsRepresentation must be fitted before transform().")
        out = df.copy()
        for c in self.categorical_cols_:
            if c not in out.columns:
                raise KeyError(f"Categorical column '{c}' not found in DataFrame.")
            levels = self.vocab_[c]
            index = {lvl: i for i, lvl in enumerate(levels)}
            m = self.nbits_[c]
            idx = np.array([index.get(_safe_str(v), 0) for v in out[c].tolist()], dtype=int)
            for j in range(m):
                bit = (idx >> j) & 1
                out[f"{c}{self.bit_suffix}{j}"] = np.where(bit == 1, 1.0, -1.0)
            out = out.drop(columns=[c])
        return out

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("AnalogBitsRepresentation must be fitted before inverse_transform().")
        out = df.copy()
        for c in self.categorical_cols_:
            levels = self.vocab_[c]
            m = self.nbits_[c]
            cols = [f"{c}{self.bit_suffix}{j}" for j in range(m)]
            if not all(col in out.columns for col in cols):
                continue
            idx = np.zeros(len(out), dtype=int)
            for j, col in enumerate(cols):
                idx = idx | ((out[col].to_numpy(dtype=float) > 0.0).astype(int) << j)
            idx = np.clip(idx, 0, max(len(levels) - 1, 0))
            fallback = levels[0] if levels else "__UNK__"
            out[c] = [levels[i] if 0 <= i < len(levels) else fallback for i in idx]
            out = out.drop(columns=cols)
        return out

    def get_state(self) -> RepresentationState:
        return RepresentationState(
            name=self.name,
            params={
                "bit_suffix": self.bit_suffix,
                "fitted": self.fitted_,
                "categorical_cols": self.categorical_cols_,
                "vocab": self.vocab_,
                "nbits": self.nbits_,
            },
        )

    @classmethod
    def from_state(cls, state: RepresentationState) -> "AnalogBitsRepresentation":
        obj = cls(bit_suffix=str(state.params.get("bit_suffix", "__bit")))
        obj.fitted_ = bool(state.params.get("fitted", False))
        obj.categorical_cols_ = list(state.params.get("categorical_cols", []))
        obj.vocab_ = {
            str(col): [str(v) for v in levels]
            for col, levels in dict(state.params.get("vocab", {})).items()
        }
        obj.nbits_ = {str(col): int(n) for col, n in dict(state.params.get("nbits", {})).items()}
        return obj
