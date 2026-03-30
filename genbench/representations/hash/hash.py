from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List

import pandas as pd

from genbench.data.schema import TabularSchema
from genbench.representations.base import RepresentationState


@dataclass
class HashRepresentation:
    """
    Hash representation for categorical columns.

    Wraps category_encoders.HashingEncoder to map categorical values to
    a fixed number of columns using a hash function.

    This is particularly useful for:
      - High-cardinality categorical features
      - Online learning scenarios where new categories appear
      - Memory-constrained environments

    Output:
      - Keeps continuous + discrete columns as-is.
      - Replaces each categorical column with hash columns.

    Note:
      HashingEncoder does NOT support handle_unknown/handle_missing parameters
      like other encoders because it handles all values uniformly via hashing.

    Not invertible: hash functions are one-way, so original values cannot
    be recovered from hash codes.
    """

    name: str = "hash_representation"
    n_components: int = 8
    drop_original_categoricals: bool = True

    # fitted state
    fitted_: bool = False
    categorical_cols_: List[str] = field(default_factory=list)
    out_cols_: Dict[str, List[str]] = field(default_factory=dict)
    encoders_: Dict[str, Any] = field(default_factory=dict, repr=False)

    ENCODER_CLS_NAME: ClassVar[str] = "HashingEncoder"

    @classmethod
    def _encoder_cls(cls) -> Any:
        """Get the encoder class from category_encoders library."""
        try:
            import category_encoders as ce
        except Exception as exc:
            raise ImportError(
                "category_encoders is required for HashRepresentation. "
                "Install it with `pip install category_encoders`."
            ) from exc

        try:
            return getattr(ce, cls.ENCODER_CLS_NAME)
        except AttributeError as exc:
            raise ImportError(
                f"category_encoders does not expose '{cls.ENCODER_CLS_NAME}'."
            ) from exc

    @staticmethod
    def requires_fit() -> bool:
        return True

    @staticmethod
    def is_invertible() -> bool:
        return False

    def fit(self, df: pd.DataFrame,
            schema: TabularSchema) -> "HashRepresentation":
        """Fit the hash encoder."""
        encoder_cls = self._encoder_cls()
        cat_cols = list(schema.categorical_cols)
        self.categorical_cols_ = cat_cols

        out_cols: Dict[str, List[str]] = {}
        encoders: Dict[str, Any] = {}

        for c in cat_cols:
            enc = encoder_cls(
                cols=[c],
                n_components=self.n_components,
                return_df=True,
            )

            col_df = df[[c]].copy()
            enc.fit(col_df)
            transformed = enc.transform(col_df)

            out_cols[c] = list(transformed.columns)
            encoders[c] = enc

        self.out_cols_ = out_cols
        self.encoders_ = encoders
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical columns to hash codes."""
        if not self.fitted_:
            raise RuntimeError(
                "HashRepresentation must be fitted before transform()."
            )

        out = df.copy()

        for c in self.categorical_cols_:
            if c not in out.columns:
                raise KeyError(
                    f"Categorical column '{c}' not found in DataFrame.")

            enc = self.encoders_[c]
            transformed = enc.transform(out[[c]])

            for col_name in self.out_cols_[c]:
                out[col_name] = transformed[col_name]

            if self.drop_original_categoricals:
                out = out.drop(columns=[c])

        return out

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hash encoding is not invertible."""
        raise NotImplementedError(
            "HashRepresentation is not invertible. "
            "Hash functions are one-way functions."
        )

    def get_state(self) -> RepresentationState:
        """Get serializable state."""
        return RepresentationState(
            name=self.name,
            params={
                "n_components": self.n_components,
                "drop_original_categoricals": self.drop_original_categoricals,
                "fitted": self.fitted_,
                "categorical_cols": self.categorical_cols_,
                "out_cols": self.out_cols_,
            },
        )

    @classmethod
    def from_state(cls, state: RepresentationState) -> "HashRepresentation":
        """Restore from serialized state."""
        obj = cls(
            n_components=int(state.params.get("n_components", 8)),
            drop_original_categoricals=bool(
                state.params.get("drop_original_categoricals", True)
            ),
        )
        obj.fitted_ = bool(state.params.get("fitted", False))
        obj.categorical_cols_ = list(state.params.get("categorical_cols", []))
        obj.out_cols_ = dict(state.params.get("out_cols", {}))
        obj.encoders_ = {}
        return obj
