from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from genbench.data.schema import TabularSchema
from genbench.evaluation.base import BaseMetric, EvaluationResult


@dataclass
class DistributionEvaluationPipeline:
    """
    Run a list of distribution metrics in a single real-vs-synthetic evaluation.
    """

    metrics: List[BaseMetric] = field(default_factory=list)
    name: str = "distribution_pipeline"

    def evaluate(self, real: pd.DataFrame, synth: pd.DataFrame, schema: TabularSchema) -> EvaluationResult:
        scores: Dict[str, float] = {}
        for metric in self.metrics:
            scores[metric.name] = float(metric.compute(real=real, synth=synth, schema=schema))
        return EvaluationResult(scores=scores, meta={"pipeline": self.name})
