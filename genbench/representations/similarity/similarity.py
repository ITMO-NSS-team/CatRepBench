from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

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


def _build_char_ngrams(value: str, ngram_range: Tuple[int, int], add_boundaries: bool) -> Dict[str, int]:
    text = f" {value} " if add_boundaries else value
    min_n, max_n = ngram_range
    grams: Dict[str, int] = {}

    for n in range(min_n, max_n + 1):
        if len(text) < n:
            continue
        for idx in range(len(text) - n + 1):
            gram = text[idx: idx + n]
            grams[gram] = grams.get(gram, 0) + 1

    return grams


def _ngram_similarity(left: Dict[str, int], right: Dict[str, int]) -> float:
    left_size = sum(left.values())
    right_size = sum(right.values())

    if left_size == 0 and right_size == 0:
        return 1.0
    if left_size == 0 or right_size == 0:
        return 0.0

    overlap = 0
    for gram, count in left.items():
        overlap += min(count, right.get(gram, 0))

    return float((2.0 * overlap) / (left_size + right_size))


@dataclass
class SimilarityRepresentation:
    """
    Similarity representation for string-like categorical columns.

    Each categorical value is encoded as a vector of n-gram similarities against
    prototypes learned from training categories. This is useful when categories
    are noisy or morphologically similar.

    Output:
      - Keeps continuous + discrete columns as-is.
      - Replaces each categorical column with similarity columns.

    Options:
      - ngram_range: inclusive (min_n, max_n) range for character n-grams.
      - add_boundaries: whether to pad strings before extracting n-grams.
      - max_prototypes: optional cap on the number of train categories used as
        prototypes per column. If set, the most frequent categories are kept.
      - drop_original_categoricals: drop original categorical columns.

    Not invertible.
    """

    name: str = "similarity_representation"
    ngram_range: Tuple[int, int] = (2, 4)
    add_boundaries: bool = True
    max_prototypes: int | None = None
    drop_original_categoricals: bool = True

    # fitted state
    fitted_: bool = False
    categorical_cols_: List[str] = field(default_factory=list)
    prototypes_: Dict[str, List[str]] = field(default_factory=dict)
    out_cols_: Dict[str, List[str]] = field(default_factory=dict)

    # cached, reconstructed on fit/from_state
    prototype_ngrams_: Dict[str, List[Dict[str, int]]] = field(default_factory=dict, repr=False)

    @staticmethod
    def requires_fit() -> bool:
        return True

    @staticmethod
    def is_invertible() -> bool:
        return False

    def fit(self, df: pd.DataFrame, schema: TabularSchema) -> "SimilarityRepresentation":
        min_n, max_n = self.ngram_range
        if min_n <= 0 or max_n <= 0 or min_n > max_n:
            raise ValueError("ngram_range must contain positive integers with min_n <= max_n.")
        if self.max_prototypes is not None and self.max_prototypes <= 0:
            raise ValueError("max_prototypes must be > 0 when provided.")

        cat_cols = list(schema.categorical_cols)
        self.categorical_cols_ = cat_cols

        prototypes: Dict[str, List[str]] = {}
        out_cols: Dict[str, List[str]] = {}

        for c in cat_cols:
            values = df[c].map(_safe_str)
            counts = values.value_counts()

            if self.max_prototypes is None:
                selected = sorted(str(cat) for cat in counts.index.tolist())
            else:
                ordered = sorted(
                    ((str(cat), int(count)) for cat, count in counts.items()),
                    key=lambda item: (-item[1], item[0]),
                )
                selected = [cat for cat, _ in ordered[: self.max_prototypes]]

            prototypes[c] = selected
            out_cols[c] = [f"{c}__sim_{idx}" for idx in range(len(selected))]

        self.prototypes_ = prototypes
        self.out_cols_ = out_cols
        self._rebuild_cache()
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("SimilarityRepresentation must be fitted before transform().")

        out = df.copy()

        for c in self.categorical_cols_:
            if c not in out.columns:
                raise KeyError(f"Categorical column '{c}' not found in DataFrame.")

            encoded_cache: Dict[str, List[float]] = {}
            encoded_rows: List[List[float]] = []
            for raw_value in out[c].tolist():
                key = _safe_str(raw_value)
                if key not in encoded_cache:
                    encoded_cache[key] = self._encode_value(c, key)
                encoded_rows.append(encoded_cache[key])

            mat = (
                np.asarray(encoded_rows, dtype=float)
                if encoded_rows
                else np.empty((len(out), len(self.out_cols_[c])), dtype=float)
            )
            for idx, col_name in enumerate(self.out_cols_[c]):
                out[col_name] = mat[:, idx]

            if self.drop_original_categoricals:
                out = out.drop(columns=[c])

        return out

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("SimilarityRepresentation is not invertible.")

    def get_state(self) -> RepresentationState:
        return RepresentationState(
            name=self.name,
            params={
                "ngram_range": list(self.ngram_range),
                "add_boundaries": self.add_boundaries,
                "max_prototypes": self.max_prototypes,
                "drop_original_categoricals": self.drop_original_categoricals,
                "fitted": self.fitted_,
                "categorical_cols": self.categorical_cols_,
                "prototypes": self.prototypes_,
                "out_cols": self.out_cols_,
            },
        )

    @classmethod
    def from_state(cls, state: RepresentationState) -> "SimilarityRepresentation":
        raw_ngram_range = state.params.get("ngram_range", [2, 4])
        if isinstance(raw_ngram_range, tuple):
            ngram_range = tuple(int(value) for value in raw_ngram_range)
        else:
            ngram_range = tuple(int(value) for value in list(raw_ngram_range))

        if len(ngram_range) != 2:
            ngram_range = (2, 4)

        obj = cls(
            ngram_range=(int(ngram_range[0]), int(ngram_range[1])),
            add_boundaries=bool(state.params.get("add_boundaries", True)),
            max_prototypes=(
                None
                if state.params.get("max_prototypes", None) is None
                else int(state.params["max_prototypes"])
            ),
            drop_original_categoricals=bool(state.params.get("drop_original_categoricals", True)),
        )
        obj.fitted_ = bool(state.params.get("fitted", False))
        obj.categorical_cols_ = list(state.params.get("categorical_cols", []))
        obj.prototypes_ = {
            str(col): [str(value) for value in values]
            for col, values in dict(state.params.get("prototypes", {})).items()
        }
        obj.out_cols_ = {
            str(col): [str(value) for value in values]
            for col, values in dict(state.params.get("out_cols", {})).items()
        }
        obj._rebuild_cache()
        return obj

    def _encode_value(self, column: str, value: str) -> List[float]:
        value_ngrams = _build_char_ngrams(value, self.ngram_range, self.add_boundaries)
        return [
            _ngram_similarity(value_ngrams, prototype_ngrams)
            for prototype_ngrams in self.prototype_ngrams_.get(column, [])
        ]

    def _rebuild_cache(self) -> None:
        self.prototype_ngrams_ = {
            column: [
                _build_char_ngrams(value, self.ngram_range, self.add_boundaries)
                for value in values
            ]
            for column, values in self.prototypes_.items()
        }
