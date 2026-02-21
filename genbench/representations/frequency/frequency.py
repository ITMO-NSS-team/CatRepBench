from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, cast

import pandas as pd
import numpy as np

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
class FrequencyRepresentation:
    """
    Frequency representation for categorical columns.

    Output:
      - Keeps continuous + discrete columns as-is.
      - Replaces each categorical column with a single numerical column
      containing frequency/count values.

    Options:
      - method: type of frequency encoding (count or normalized)
      - unk_token: bucket for unseen categories at transform-time.
      - include_unk: whether to include an UNK category (recommended True
      for robustness).
      - unk_strategy: how to handle unseen categories (zero, min, mean)

    Not invertible
    """

    name: str = "frequency_representation"
    method: Literal["count", "normalized"] = "count"
    unk_token: str = "__UNK__"
    include_unk: bool = True
    unk_strategy: Literal["zero", "min", "mean"] = "zero"

    # fitted state
    fitted_: bool = False
    categorical_cols_: List[str] = field(default_factory=list)
    freq_maps_: Dict[str, Dict[str, float]] = field(default_factory=dict)
    col_stats_: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @staticmethod
    def requires_fit() -> bool:
        return True

    @staticmethod
    def is_invertible() -> bool:
        return False

    def fit(self, df: pd.DataFrame,
            schema: TabularSchema) -> "FrequencyRepresentation":
        cat_cols = list(schema.categorical_cols)
        self.categorical_cols_ = cat_cols

        freq_maps: Dict[str, Dict[str, float]] = {}
        col_stats: Dict[str, Dict[str, float]] = {}

        for c in cat_cols:
            # Build frequency map on TRAIN ONLY
            s = df[c].map(_safe_str)
            value_counts = s.value_counts()
            total = len(s.dropna())

            # Create frequency dictionary
            freq_map: Dict[str, float] = {}
            for cat, count in value_counts.items():
                if self.method == "count":
                    freq_map[cat] = float(count)
                elif self.method == "normalized":
                    freq_map[cat] = count / total

            if self.include_unk and self.unk_token not in freq_map:
                freq_map[self.unk_token] = 0.0  # UNK gets frequency 0

            freq_maps[c] = freq_map

            # Calculate statistics for unknown strategy
            known_values = [
                freq for cat, freq in freq_map.items()
                if cat != self.unk_token  # Exclude UNK
            ]

            if known_values:
                col_stats[c] = {
                    "min": min(known_values),
                    "mean": np.mean(known_values),
                    "max": max(known_values),
                }
            else:
                col_stats[c] = {
                    "min": 0.0,
                    "mean": 0.0,
                    "max": 0.0,
                }

        self.freq_maps_ = freq_maps
        self.col_stats_ = col_stats
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("FrequencyRepresentation must be fitted before "
                               "transform().")

        out = df.copy()

        for c in self.categorical_cols_:
            if c not in out.columns:
                raise KeyError(
                    f"Categorical column '{c}' not found in DataFrame.")

            s = out[c].map(_safe_str)

            freq_series = s.map(self.freq_maps_[c])

            if self.include_unk:
                unk_freq = self.freq_maps_[c].get(self.unk_token, 0.0)
                freq_series = freq_series.fillna(unk_freq)
            else:
                if self.unk_strategy == "zero":
                    freq_series = freq_series.fillna(0.0)
                elif self.unk_strategy == "min":
                    freq_series = freq_series.fillna(self.col_stats_[c]["min"])
                elif self.unk_strategy == "mean":
                    freq_series = freq_series.fillna(
                        self.col_stats_[c]["mean"])
                else:
                    freq_series = freq_series.fillna(0.0)

            out[c] = freq_series

        return out

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError(
            "FrequencyRepresentation is not invertible."
        )

    def get_state(self) -> RepresentationState:
        return RepresentationState(
            name=self.name,
            params={
                "method": self.method,
                "unk_token": self.unk_token,
                "include_unk": self.include_unk,
                "unk_strategy": self.unk_strategy,
                "fitted": self.fitted_,
                "categorical_cols": self.categorical_cols_,
                "freq_maps": self.freq_maps_,
                "col_stats": self.col_stats_
            },
        )

    @classmethod
    def from_state(cls,
                   state: RepresentationState) -> "FrequencyRepresentation":
        # Helper functions to validate Literal values
        def validate_method(value: str) -> Literal["count", "normalized"]:
            if value in ("count", "normalized"):
                return cast(Literal["count", "normalized"], value)
            return "count"

        def validate_unk_strategy(value: str) -> Literal[
            "zero", "min", "mean"]:
            if value in ("zero", "min", "mean"):
                return cast(Literal["zero", "min", "mean"], value)
            return "zero"

        obj = cls(
            method=validate_method(str(state.params.get("method", "count"))),
            unk_token=str(state.params.get("unk_token", "__UNK__")),
            include_unk=bool(state.params.get("include_unk", True)),
            unk_strategy=validate_unk_strategy(
                str(state.params.get("unk_strategy", "zero"))),
        )
        obj.fitted_ = bool(state.params.get("fitted", False))
        obj.categorical_cols_ = list(state.params.get("categorical_cols", []))
        obj.freq_maps_ = dict(state.params.get("freq_maps", {}))
        obj.col_stats_ = dict(state.params.get("col_stats", {}))
        return obj
