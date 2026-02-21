from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Type

import pandas as pd

from genbench.data.schema import TabularSchema
from .base import TransformState, BaseTransform

from genbench.representations.base import BaseRepresentation, RepresentationState
from genbench.representations.one_hot.one_hot import OneHotRepresentation
from genbench.representations.ordinal.ordinal import OrdinalRepresentation
from genbench.representations.frequency.frequency import FrequencyRepresentation


# Registry for representations usable via this transform.
# Add new representations here when you expand the benchmark.
_REPRESENTATION_REGISTRY: Dict[str, Type[BaseRepresentation]] = {
    "one_hot_representation": OneHotRepresentation,
    "ordinal_representation": OrdinalRepresentation,
    "frequency_representation": FrequencyRepresentation
}


@dataclass
class CategoricalRepresentationTransform:
    """
    Transform wrapper that applies a categorical representation as part of the TransformPipeline.

    Why it's a Transform (not used directly in DataModule)?
      - Your current pipeline machinery expects BaseTransform with TransformState,
        and supports cloning + fold-wise refit. :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}

    Important:
      - This should be placed AFTER missing handling, and usually AFTER continuous scaling,
        unless your representation depends on scaled values (rare).
      - Fit is done on TRAIN only per fold/holdout (DataModule ensures that).
    """

    name: str = "categorical_representation_transform"

    # representation config
    representation_name: str = "one_hot_representation"
    representation_kwargs: Dict[str, Any] = field(default_factory=dict)

    # fitted state
    fitted_: bool = False
    repr_: BaseRepresentation | None = None

    def requires_fit(self) -> bool:
        return True

    def is_invertible(self) -> bool:
        return True

    def fit(self, df: pd.DataFrame, schema: TabularSchema) -> "CategoricalRepresentationTransform":
        if self.representation_name not in _REPRESENTATION_REGISTRY:
            raise KeyError(
                f"Unknown representation '{self.representation_name}'. "
                f"Register it in _REPRESENTATION_REGISTRY."
            )

        rep_cls = _REPRESENTATION_REGISTRY[self.representation_name]
        rep = rep_cls(**self.representation_kwargs)  # type: ignore[arg-type]
        rep.fit(df, schema)

        self.repr_ = rep
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_ or self.repr_ is None:
            raise RuntimeError("CategoricalRepresentationTransform must be fitted before transform().")
        return self.repr_.transform(df)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_ or self.repr_ is None:
            # Identity by convention (but for this transform we prefer strictness)
            return df
        return self.repr_.inverse_transform(df)

    def get_state(self) -> TransformState:
        # We store representation state inside transform params (JSON-serializable).
        if self.repr_ is None:
            rep_state = None
        else:
            rs: RepresentationState = self.repr_.get_state()
            rep_state = {"name": rs.name, "params": rs.params}

        return TransformState(
            name=self.name,
            params={
                "representation_name": self.representation_name,
                "representation_kwargs": self.representation_kwargs,
                "fitted": self.fitted_,
                "rep_state": rep_state,
            },
        )

    @classmethod
    def from_state(cls, state: TransformState) -> "CategoricalRepresentationTransform":
        obj = cls(
            representation_name=str(state.params.get("representation_name", "one_hot_representation")),
            representation_kwargs=dict(state.params.get("representation_kwargs", {})),
        )
        obj.fitted_ = bool(state.params.get("fitted", False))

        rep_state = state.params.get("rep_state", None)
        if rep_state is not None:
            rep_name = str(rep_state.get("name"))
            rep_params = dict(rep_state.get("params", {}))
            if rep_name not in _REPRESENTATION_REGISTRY:
                raise KeyError(
                    f"Unknown representation '{rep_name}' in saved state. "
                    f"Register it in _REPRESENTATION_REGISTRY."
                )
            rep_cls = _REPRESENTATION_REGISTRY[rep_name]
            obj.repr_ = rep_cls.from_state(RepresentationState(name=rep_name, params=rep_params))
        return obj


def register_representation(name: str, cls: Type[BaseRepresentation]) -> None:
    """
    Optional helper for downstream extensions.
    Usage:
        from sbtab.transforms.categorical import register_representation
        register_representation("hashing_representation", HashingRepresentation)
    """
    _REPRESENTATION_REGISTRY[name] = cls
