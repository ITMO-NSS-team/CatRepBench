from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from genbench.data.schema import TabularSchema
from genbench.representations.base import RepresentationState


def _safe_str(x: object) -> str:
    # Stable string for categories (handles NaN/None)
    if x is None:
        return "__NONE__"
    if isinstance(x, float) and pd.isna(x):
        return "__NAN__"
    return str(x)


@dataclass
class LabelRepresentation:
    """
    Label representation for categorical columns.

    Output:
      - Keeps continuous + discrete columns as-is.
      - Replaces each categorical column with a single numerical column
      using label mapping.

    Options:
      - unk_token: bucket for unseen categories at transform-time.
      - include_unk: whether to include an UNK category in mapping (recommended
      True for robustness).

    Inverse transform:
      - Reconstructs each categorical column by applying inverse label mapping.
      - If encountering unknown values -> unk_token (or None if not in
      mapping).
    """

    name: str = "label_representation"
    unk_token: str = "__UNK__"
    include_unk: bool = True

    # fitted state
    fitted_: bool = False
    categorical_cols_: List[str] = field(default_factory=list)
    vocab_: Dict[str, List[str]] = field(
        default_factory=dict)  # col -> list of category strings
    label_map_: Dict[str, Dict[str, int]] = field(
        default_factory=dict)  # col -> dict mapping category to label
    inverse_map_: Dict[str, Dict[int, str]] = field(
        default_factory=dict)  # col -> dict mapping label to category

    @staticmethod
    def requires_fit() -> bool:
        return True

    @staticmethod
    def is_invertible() -> bool:
        return True

    def fit(self, df: pd.DataFrame,
            schema: TabularSchema) -> "LabelRepresentation":
        cat_cols = list(schema.categorical_cols)
        self.categorical_cols_ = cat_cols

        vocab: Dict[str, List[str]] = {}
        label_map: Dict[str, Dict[str, int]] = {}
        inverse_map: Dict[str, Dict[int, str]] = {}

        for c in cat_cols:
            # Build vocab on TRAIN ONLY
            s = df[c].map(_safe_str)
            cats = list(pd.unique(s.dropna()))
            # deterministic ordering
            cats = sorted(cats)

            if self.include_unk and self.unk_token not in cats:
                cats = cats + [self.unk_token]

            vocab[c] = cats
            label_map[c] = {}
            inverse_map[c] = {}

            for idx, cat in enumerate(cats):
                label_map[c][cat] = idx
                inverse_map[c][idx] = cat

        self.vocab_ = vocab
        self.label_map_ = label_map
        self.inverse_map_ = inverse_map
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("LabelRepresentation must be fitted before "
                               "transform().")

        out = df.copy()

        for c in self.categorical_cols_:
            if c not in out.columns:
                raise KeyError(
                    f"Categorical column '{c}' not found in DataFrame.")

            s = out[c].map(_safe_str)

            # Map unseen categories -> UNK (or keep as-is if include_unk=False)
            if self.include_unk:
                known = set(self.vocab_[c])
                s = s.where(s.isin(known), other=self.unk_token)
            else:
                s = s.map(lambda x: x if x in self.label_map_[c] else pd.NA)

            # Apply label mapping
            out[c] = s.map(self.label_map_[c]).astype(
                "Int64")  # Int64 supports NA

        return out

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("LabelRepresentation must be fitted before "
                               "inverse_transform().")

        out = df.copy()

        for c in self.categorical_cols_:
            recovered = out[c].map(self.inverse_map_[c])

            out[c] = recovered

        return out

    def get_state(self) -> RepresentationState:
        return RepresentationState(
            name=self.name,
            params={
                "unk_token": self.unk_token,
                "include_unk": self.include_unk,
                "fitted": self.fitted_,
                "categorical_cols": self.categorical_cols_,
                "vocab": self.vocab_,
                "label_map": self.label_map_,
                "inverse_map": self.inverse_map_
            },
        )

    @classmethod
    def from_state(cls, state: RepresentationState) -> "LabelRepresentation":
        obj = cls(
            unk_token=str(state.params.get("unk_token", "__UNK__")),
            include_unk=bool(state.params.get("include_unk", True)),
        )
        obj.fitted_ = bool(state.params.get("fitted", False))
        obj.categorical_cols_ = list(state.params.get("categorical_cols", []))
        obj.vocab_ = dict(state.params.get("vocab", {}))
        obj.label_map_ = dict(state.params.get("label_map", {}))
        obj.inverse_map_ = dict(state.params.get("inverse_map", {}))
        return obj
