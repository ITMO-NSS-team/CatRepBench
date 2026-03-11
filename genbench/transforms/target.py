from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from genbench.data.schema import TabularSchema
from .base import TransformState


def infer_is_regression_target(
    y: pd.Series,
    *,
    task_type: Optional[str] = None,
    discrete_unique_threshold: int = 20,
) -> bool:
    """
    Infer regression/classification mode for target by task hint + target properties.
    """
    if task_type is not None:
        normalized = task_type.strip().lower()
        if normalized not in {"classification", "regression"}:
            raise ValueError("task_type must be 'classification' or 'regression'.")
        return normalized == "regression"
    return bool(pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) > discrete_unique_threshold)


def _safe_str(x: object) -> str:
    if x is None:
        return "__NONE__"
    if isinstance(x, float) and pd.isna(x):
        return "__NAN__"
    return str(x)


@dataclass
class TargetTypePreprocessor:
    """
    Train-only target preprocessing:
    - classification + categorical y -> label encoding
    - regression + numeric y -> standard scaling
    """

    name: str = "target_type_preprocessor"
    task_type: Optional[str] = None
    discrete_unique_threshold: int = 20
    scale_numeric_for_regression: bool = True
    encode_categorical_for_classification: bool = True
    eps: float = 1e-12

    fitted_: bool = False
    target_col_: Optional[str] = None
    is_regression_: Optional[bool] = None
    did_encode_: bool = False
    did_scale_: bool = False
    classes_: List[str] = field(default_factory=list)
    class_to_index_: Dict[str, int] = field(default_factory=dict)
    index_to_class_: Dict[int, str] = field(default_factory=dict)
    target_mean_: float = 0.0
    target_std_: float = 1.0

    def requires_fit(self) -> bool:
        return True

    def is_invertible(self) -> bool:
        return True

    def fit(self, df: pd.DataFrame, schema: TabularSchema) -> "TargetTypePreprocessor":
        self.fitted_ = True
        self.target_col_ = schema.target_col
        self.is_regression_ = None
        self.did_encode_ = False
        self.did_scale_ = False
        self.classes_ = []
        self.class_to_index_ = {}
        self.index_to_class_ = {}
        self.target_mean_ = 0.0
        self.target_std_ = 1.0

        if self.target_col_ is None or self.target_col_ not in df.columns:
            return self

        y = df[self.target_col_]
        self.is_regression_ = infer_is_regression_target(
            y,
            task_type=self.task_type,
            discrete_unique_threshold=self.discrete_unique_threshold,
        )

        if self.is_regression_:
            if self.scale_numeric_for_regression and pd.api.types.is_numeric_dtype(y):
                y_num = y.astype(float)
                mean = float(y_num.mean(skipna=True))
                std = float(y_num.std(ddof=0, skipna=True))
                self.target_mean_ = mean
                self.target_std_ = std if np.isfinite(std) and std > self.eps else 1.0
                self.did_scale_ = True
            return self

        if self.encode_categorical_for_classification and (
            pd.api.types.is_object_dtype(y)
            or isinstance(y.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(y)
            or pd.api.types.is_string_dtype(y)
        ):
            classes = sorted({_safe_str(v) for v in y.dropna().tolist()})
            self.classes_ = classes
            self.class_to_index_ = {v: i for i, v in enumerate(classes)}
            self.index_to_class_ = {i: v for v, i in self.class_to_index_.items()}
            self.did_encode_ = True

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("TargetTypePreprocessor must be fitted before transform().")

        out = df.copy()
        if self.target_col_ is None or self.target_col_ not in out.columns:
            return out

        if self.did_encode_:
            raw = out[self.target_col_]
            safe = raw.map(_safe_str)
            encoded = safe.map(self.class_to_index_)
            unseen_mask = raw.notna() & encoded.isna()
            if unseen_mask.any():
                unseen_values = sorted({str(v) for v in raw[unseen_mask].tolist()})
                raise ValueError(
                    f"Found target categories not present in train: {unseen_values}. "
                    "Target label encoding is fitted on train only."
                )
            out[self.target_col_] = encoded.astype("Int64")

        if self.did_scale_:
            out[self.target_col_] = (out[self.target_col_].astype(float) - self.target_mean_) / self.target_std_

        return out

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            return df

        out = df.copy()
        if self.target_col_ is None or self.target_col_ not in out.columns:
            return out

        if self.did_scale_:
            out[self.target_col_] = out[self.target_col_].astype(float) * self.target_std_ + self.target_mean_

        if self.did_encode_:
            raw = out[self.target_col_]
            out[self.target_col_] = raw.map(
                lambda x: self.index_to_class_.get(int(x)) if pd.notna(x) else pd.NA
            )

        return out

    def get_state(self) -> TransformState:
        return TransformState(
            name=self.name,
            params={
                "task_type": self.task_type,
                "discrete_unique_threshold": self.discrete_unique_threshold,
                "scale_numeric_for_regression": self.scale_numeric_for_regression,
                "encode_categorical_for_classification": self.encode_categorical_for_classification,
                "eps": self.eps,
                "fitted": self.fitted_,
                "target_col": self.target_col_,
                "is_regression": self.is_regression_,
                "did_encode": self.did_encode_,
                "did_scale": self.did_scale_,
                "classes": self.classes_,
                "class_to_index": self.class_to_index_,
                "index_to_class": {str(k): v for k, v in self.index_to_class_.items()},
                "target_mean": self.target_mean_,
                "target_std": self.target_std_,
            },
        )

    @classmethod
    def from_state(cls, state: TransformState) -> "TargetTypePreprocessor":
        obj = cls(
            task_type=state.params.get("task_type"),
            discrete_unique_threshold=int(state.params.get("discrete_unique_threshold", 20)),
            scale_numeric_for_regression=bool(state.params.get("scale_numeric_for_regression", True)),
            encode_categorical_for_classification=bool(state.params.get("encode_categorical_for_classification", True)),
            eps=float(state.params.get("eps", 1e-12)),
        )
        obj.fitted_ = bool(state.params.get("fitted", False))
        obj.target_col_ = state.params.get("target_col")
        obj.is_regression_ = state.params.get("is_regression")
        obj.did_encode_ = bool(state.params.get("did_encode", False))
        obj.did_scale_ = bool(state.params.get("did_scale", False))
        obj.classes_ = list(state.params.get("classes", []))
        obj.class_to_index_ = {str(k): int(v) for k, v in dict(state.params.get("class_to_index", {})).items()}
        raw_inverse = dict(state.params.get("index_to_class", {}))
        obj.index_to_class_ = {int(k): str(v) for k, v in raw_inverse.items()}
        obj.target_mean_ = float(state.params.get("target_mean", 0.0))
        obj.target_std_ = float(state.params.get("target_std", 1.0))
        return obj
