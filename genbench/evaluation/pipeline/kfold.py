from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Dict, List

from genbench.evaluation.base import EvaluationResult


@dataclass
class KFoldAggregateResult:
    """
    Aggregate scores across fold-level evaluation results.
    """

    per_fold: List[EvaluationResult] = field(default_factory=list)
    mean_scores: Dict[str, float] = field(default_factory=dict)
    std_scores: Dict[str, float] = field(default_factory=dict)


def aggregate_kfold_results(results: List[EvaluationResult]) -> KFoldAggregateResult:
    if not results:
        return KFoldAggregateResult()

    metric_names = sorted({name for r in results for name in r.scores.keys()})
    mean_scores: Dict[str, float] = {}
    std_scores: Dict[str, float] = {}

    for name in metric_names:
        values = [r.scores[name] for r in results if name in r.scores]
        if not values:
            continue
        mean_scores[name] = float(mean(values))
        std_scores[name] = float(pstdev(values)) if len(values) > 1 else 0.0

    return KFoldAggregateResult(per_fold=results, mean_scores=mean_scores, std_scores=std_scores)
