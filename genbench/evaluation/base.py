from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol, runtime_checkable

import pandas as pd

from genbench.data.schema import TabularSchema


@dataclass
class EvaluationResult:
    """
    Unified evaluation output used across metrics and evaluators.
    """

    scores: Dict[str, float]
    details: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class BaseMetric(Protocol):
    """
    Contract for a single distribution-quality metric.
    """

    name: str

    def compute(
        self,
        real: pd.DataFrame,
        synth: pd.DataFrame,
        schema: TabularSchema,
    ) -> float:
        ...


@runtime_checkable
class BaseEvaluator(Protocol):
    """
    Contract for multi-metric or task-specific evaluators.
    """

    name: str

    def evaluate(self, **kwargs: Any) -> EvaluationResult:
        ...
