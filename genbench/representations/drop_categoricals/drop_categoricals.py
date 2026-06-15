from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from genbench.data.schema import TabularSchema
from genbench.representations.base import RepresentationState


@dataclass
class DropCategoricalsRepresentation:
    """
    Drop-categoricals baseline: removes all categorical *feature* columns and
    keeps continuous/discrete columns (and the target) as-is.

    This is the "no encoding" lower-bound baseline — it discards categorical
    information entirely so the generative model is trained on the continuous
    features only. Not invertible (dropped columns cannot be recovered), so it is
    only meaningful for datasets that have continuous features; on all/almost-
    categorical datasets it removes nearly everything (a known limitation worth
    reporting, cf. the "Drop encoder" in Clerici & Nobani 2026).

    The schema target column is never dropped (only categorical *features*).
    """

    name: str = "drop_categoricals_representation"

    fitted_: bool = False
    categorical_cols_: List[str] = field(default_factory=list)
    target_col_: Optional[str] = None

    @staticmethod
    def requires_fit() -> bool:
        return True

    @staticmethod
    def is_invertible() -> bool:
        return False

    def fit(self, df: pd.DataFrame, schema: TabularSchema) -> "DropCategoricalsRepresentation":
        self.target_col_ = schema.target_col
        # Drop categorical FEATURES only — never the target.
        self.categorical_cols_ = [c for c in schema.categorical_cols if c != schema.target_col]
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("DropCategoricalsRepresentation must be fitted before transform().")
        drop = [c for c in self.categorical_cols_ if c in df.columns]
        return df.drop(columns=drop)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError(
            "DropCategoricalsRepresentation is not invertible (categorical columns are discarded)."
        )

    def get_state(self) -> RepresentationState:
        return RepresentationState(
            name=self.name,
            params={
                "fitted": self.fitted_,
                "categorical_cols": self.categorical_cols_,
                "target_col": self.target_col_,
            },
        )

    @classmethod
    def from_state(cls, state: RepresentationState) -> "DropCategoricalsRepresentation":
        obj = cls()
        obj.fitted_ = bool(state.params.get("fitted", False))
        obj.categorical_cols_ = list(state.params.get("categorical_cols", []))
        tc = state.params.get("target_col")
        obj.target_col_ = None if tc is None else str(tc)
        return obj
