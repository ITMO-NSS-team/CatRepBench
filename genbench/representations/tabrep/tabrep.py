from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from genbench.data.schema import TabularSchema
from genbench.representations.base import RepresentationState


def _safe_str(x: object) -> str:
    # Stable string for categories (handles NaN/None) — matches the other reps.
    if x is None:
        return "__NONE__"
    if isinstance(x, float) and pd.isna(x):
        return "__NAN__"
    return str(x)


@dataclass
class TabRepRepresentation:
    """
    Trigonometric (phase) representation for categorical columns — "TabRep".

    Each categorical column with K observed levels is placed on the unit circle:
    the i-th level -> ``[cos(2*pi*i/K), sin(2*pi*i/K)]`` — exactly TWO continuous
    columns regardless of cardinality. Continuous/discrete columns are kept
    as-is.

    Why it matters for CatRepBench: it is a dense, fixed-low-dimensional,
    *unsupervised* continuous encoding engineered as a denoising target for
    diffusion / flow generative models (Si et al., "TabRep"), which the current
    12 representations lack. Deterministic and exactly invertible (decode by the
    nearest angle), so synthetic samples decode back to valid categories.
    """

    name: str = "tabrep_representation"
    cos_suffix: str = "__tabrep_cos"
    sin_suffix: str = "__tabrep_sin"

    # fitted state
    fitted_: bool = False
    categorical_cols_: List[str] = field(default_factory=list)
    vocab_: Dict[str, List[str]] = field(default_factory=dict)

    @staticmethod
    def requires_fit() -> bool:
        return True

    @staticmethod
    def is_invertible() -> bool:
        return True

    def _cols(self, c: str) -> Tuple[str, str]:
        return c + self.cos_suffix, c + self.sin_suffix

    def fit(self, df: pd.DataFrame, schema: TabularSchema) -> "TabRepRepresentation":
        cat_cols = list(schema.categorical_cols)
        self.categorical_cols_ = cat_cols
        vocab: Dict[str, List[str]] = {}
        for c in cat_cols:
            # Stable, sorted vocabulary of observed levels (train only).
            vocab[c] = sorted({_safe_str(v) for v in df[c].tolist()})
        self.vocab_ = vocab
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("TabRepRepresentation must be fitted before transform().")
        out = df.copy()
        for c in self.categorical_cols_:
            if c not in out.columns:
                raise KeyError(f"Categorical column '{c}' not found in DataFrame.")
            levels = self.vocab_[c]
            k = max(len(levels), 1)
            index = {lvl: i for i, lvl in enumerate(levels)}
            # Unseen levels map to index 0 (the lowest-sorted known level).
            idx = np.array([index.get(_safe_str(v), 0) for v in out[c].tolist()], dtype=float)
            angles = 2.0 * np.pi * idx / k
            cos_col, sin_col = self._cols(c)
            out[cos_col] = np.cos(angles)
            out[sin_col] = np.sin(angles)
            out = out.drop(columns=[c])
        return out

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("TabRepRepresentation must be fitted before inverse_transform().")
        out = df.copy()
        for c in self.categorical_cols_:
            cos_col, sin_col = self._cols(c)
            if cos_col not in out.columns or sin_col not in out.columns:
                continue
            levels = self.vocab_[c]
            k = max(len(levels), 1)
            cos_v = out[cos_col].to_numpy(dtype=float)
            sin_v = out[sin_col].to_numpy(dtype=float)
            ang = np.mod(np.arctan2(sin_v, cos_v), 2.0 * np.pi)  # [0, 2*pi)
            idx = (np.rint(ang / (2.0 * np.pi) * k).astype(int)) % k
            fallback = levels[0] if levels else "__UNK__"
            out[c] = [levels[i] if 0 <= i < len(levels) else fallback for i in idx]
            out = out.drop(columns=[cos_col, sin_col])
        return out

    def get_state(self) -> RepresentationState:
        return RepresentationState(
            name=self.name,
            params={
                "cos_suffix": self.cos_suffix,
                "sin_suffix": self.sin_suffix,
                "fitted": self.fitted_,
                "categorical_cols": self.categorical_cols_,
                "vocab": self.vocab_,
            },
        )

    @classmethod
    def from_state(cls, state: RepresentationState) -> "TabRepRepresentation":
        obj = cls(
            cos_suffix=str(state.params.get("cos_suffix", "__tabrep_cos")),
            sin_suffix=str(state.params.get("sin_suffix", "__tabrep_sin")),
        )
        obj.fitted_ = bool(state.params.get("fitted", False))
        obj.categorical_cols_ = list(state.params.get("categorical_cols", []))
        obj.vocab_ = {
            str(col): [str(v) for v in levels]
            for col, levels in dict(state.params.get("vocab", {})).items()
        }
        return obj
