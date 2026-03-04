from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Literal, cast

import numpy as np
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


def _to_float_list(values: List[Any]) -> List[float]:
    out: List[float] = []
    for value in values:
        if value is None or pd.isna(value):
            out.append(float("nan"))
        else:
            out.append(float(value))
    return out


@dataclass
class CategoryEncodersRepresentationBase:
    """
    Generic wrapper for category_encoders tabular encoders.

    Subclasses must define:
      - `name` (representation id)
      - `ENCODER_CLS_NAME` (class name from category_encoders)
    """

    name: str = "category_encoder_representation"
    handle_unknown: Literal["value", "return_nan", "error", "indicator"] = "value"
    handle_missing: Literal["value", "return_nan", "error", "indicator"] = "value"
    drop_original_categoricals: bool = True

    # fitted state
    fitted_: bool = False
    categorical_cols_: List[str] = field(default_factory=list)
    out_cols_: Dict[str, List[str]] = field(default_factory=dict)
    value_map_: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    unknown_vec_: Dict[str, List[float] | None] = field(default_factory=dict)
    missing_vec_: Dict[str, List[float] | None] = field(default_factory=dict)
    encoders_: Dict[str, Any] = field(default_factory=dict, repr=False)

    ENCODER_CLS_NAME: ClassVar[str] = ""

    @classmethod
    def _encoder_cls(cls) -> Any:
        if not cls.ENCODER_CLS_NAME:
            raise NotImplementedError("ENCODER_CLS_NAME must be set in subclass.")
        try:
            import category_encoders as ce  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "category_encoders is required for category-encoder-based representations. "
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

    def fit(self, df: pd.DataFrame, schema: TabularSchema) -> "CategoryEncodersRepresentationBase":
        encoder_cls = self.__class__._encoder_cls()
        cat_cols = list(schema.categorical_cols)
        self.categorical_cols_ = cat_cols

        out_cols: Dict[str, List[str]] = {}
        value_map: Dict[str, Dict[str, List[float]]] = {}
        unknown_vec: Dict[str, List[float] | None] = {}
        missing_vec: Dict[str, List[float] | None] = {}
        encoders: Dict[str, Any] = {}

        for c in cat_cols:
            enc = encoder_cls(
                cols=[c],
                return_df=True,
                handle_unknown=self.handle_unknown,
                handle_missing=self.handle_missing,
            )

            col_df = df[[c]].copy()
            enc.fit(col_df)
            transformed = enc.transform(col_df)

            out_cols[c] = list(transformed.columns)
            encoders[c] = enc

            col_map: Dict[str, List[float]] = {}
            for raw_value in pd.unique(df[c]):
                probe = pd.DataFrame({c: [raw_value]})
                vec = _to_float_list(enc.transform(probe).iloc[0].tolist())
                col_map[_safe_str(raw_value)] = vec
            value_map[c] = col_map

            unknown_probe = "__GB_CE_UNKNOWN__"
            try:
                unknown_vec[c] = _to_float_list(
                    enc.transform(pd.DataFrame({c: [unknown_probe]})).iloc[0].tolist()
                )
            except Exception:
                unknown_vec[c] = None

            try:
                missing_vec[c] = _to_float_list(
                    enc.transform(pd.DataFrame({c: [np.nan]})).iloc[0].tolist()
                )
            except Exception:
                missing_vec[c] = None

        self.out_cols_ = out_cols
        self.value_map_ = value_map
        self.unknown_vec_ = unknown_vec
        self.missing_vec_ = missing_vec
        self.encoders_ = encoders
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError(f"{self.__class__.__name__} must be fitted before transform().")

        out = df.copy()

        for c in self.categorical_cols_:
            if c not in out.columns:
                raise KeyError(f"Categorical column '{c}' not found in DataFrame.")

            encoded_rows: List[List[float]] = []
            mapping = self.value_map_[c]
            for raw_value in out[c].tolist():
                key = _safe_str(raw_value)
                vec = mapping.get(key)

                if vec is None and (raw_value is None or pd.isna(raw_value)):
                    vec = self.missing_vec_.get(c)
                    if vec is None:
                        raise ValueError(
                            f"Missing value found in column '{c}' but handle_missing='error'."
                        )

                if vec is None:
                    vec = self.unknown_vec_.get(c)
                    if vec is None:
                        raise ValueError(
                            f"Unknown category '{raw_value}' found in column '{c}' "
                            "but handle_unknown='error'."
                        )

                encoded_rows.append(vec)

            mat = np.asarray(encoded_rows, dtype=float) if encoded_rows else np.empty((len(out), 0))
            for idx, col_name in enumerate(self.out_cols_[c]):
                out[col_name] = mat[:, idx]

            if self.drop_original_categoricals:
                out = out.drop(columns=[c])

        return out

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError(f"{self.__class__.__name__} is not invertible.")

    def get_state(self) -> RepresentationState:
        return RepresentationState(
            name=self.name,
            params={
                "handle_unknown": self.handle_unknown,
                "handle_missing": self.handle_missing,
                "drop_original_categoricals": self.drop_original_categoricals,
                "fitted": self.fitted_,
                "categorical_cols": self.categorical_cols_,
                "out_cols": self.out_cols_,
                "value_map": self.value_map_,
                "unknown_vec": self.unknown_vec_,
                "missing_vec": self.missing_vec_,
            },
        )

    @classmethod
    def from_state(cls, state: RepresentationState) -> "CategoryEncodersRepresentationBase":
        def validate_handle(value: str) -> Literal["value", "return_nan", "error", "indicator"]:
            if value in ("value", "return_nan", "error", "indicator"):
                return cast(Literal["value", "return_nan", "error", "indicator"], value)
            return "value"

        obj = cls(  # type: ignore[call-arg]
            handle_unknown=validate_handle(str(state.params.get("handle_unknown", "value"))),
            handle_missing=validate_handle(str(state.params.get("handle_missing", "value"))),
            drop_original_categoricals=bool(state.params.get("drop_original_categoricals", True)),
        )
        obj.fitted_ = bool(state.params.get("fitted", False))
        obj.categorical_cols_ = list(state.params.get("categorical_cols", []))
        obj.out_cols_ = dict(state.params.get("out_cols", {}))

        raw_value_map = dict(state.params.get("value_map", {}))
        obj.value_map_ = {
            str(col): {str(cat): _to_float_list(list(vec)) for cat, vec in dict(col_map).items()}
            for col, col_map in raw_value_map.items()
        }

        raw_unknown_vec = dict(state.params.get("unknown_vec", {}))
        obj.unknown_vec_ = {
            str(col): None if vec is None else _to_float_list(list(vec))
            for col, vec in raw_unknown_vec.items()
        }

        raw_missing_vec = dict(state.params.get("missing_vec", {}))
        obj.missing_vec_ = {
            str(col): None if vec is None else _to_float_list(list(vec))
            for col, vec in raw_missing_vec.items()
        }

        obj.encoders_ = {}
        return obj
