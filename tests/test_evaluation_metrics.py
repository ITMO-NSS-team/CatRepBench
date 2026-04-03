import numpy as np
import pandas as pd
import pytest

from genbench.data.schema import TabularSchema
from genbench.evaluation.distribution.corr_frobenius import CorrelationFrobeniusMetric
from genbench.evaluation.distribution.marginal_kl import MarginalKLDivergenceMetric
from genbench.evaluation.distribution.wasserstein import WassersteinDistanceMetric
from genbench.evaluation.metrics import (
    compute_corr_frobenius,
    compute_marginal_kl_mean,
    compute_wasserstein_mean,
)
from genbench.evaluation.pipeline.single_run import DistributionEvaluationPipeline
from genbench.evaluation.tstr import tstr_catboost
from genbench.evaluation.utility.tstr_catboost import TSTRCatBoostEvaluator


@pytest.fixture
def sample_data():
    real = pd.DataFrame(
        {
            "x": np.random.normal(0, 1, size=200),
            "y": np.random.randint(0, 5, size=200),
            "cat": np.random.choice(["a", "b", "c"], size=200),
            "target": np.random.normal(0, 1, size=200),
        }
    )
    synth = pd.DataFrame(
        {
            "x": np.random.normal(0.5, 1.2, size=200),
            "y": np.random.randint(0, 5, size=200),
            "cat": np.random.choice(["a", "b", "c"], size=200),
            "target": np.random.normal(0.1, 1.1, size=200),
        }
    )
    schema = TabularSchema(
        continuous_cols=["x"],
        discrete_cols=["y"],
        categorical_cols=["cat"],
        target_col="target",
    )
    return real, synth, schema


def test_wasserstein_non_negative(sample_data):
    real, synth, schema = sample_data
    value = compute_wasserstein_mean(real, synth, schema)
    assert value >= 0.0


def test_marginal_kl_mean(sample_data):
    real, synth, schema = sample_data
    value = compute_marginal_kl_mean(real, synth, schema, n_bins=10)
    assert value >= 0.0


def test_corr_frobenius(sample_data):
    real, synth, schema = sample_data
    value = compute_corr_frobenius(real, synth, schema)
    assert value >= 0.0


def test_distribution_pipeline(sample_data):
    real, synth, schema = sample_data
    pipeline = DistributionEvaluationPipeline(
        metrics=[
            WassersteinDistanceMetric(),
            MarginalKLDivergenceMetric(n_bins=10),
            CorrelationFrobeniusMetric(),
        ]
    )
    result = pipeline.evaluate(real=real, synth=synth, schema=schema)
    assert "wasserstein_mean" in result.scores
    assert "marginal_kl_mean" in result.scores
    assert "corr_frobenius" in result.scores


def test_tstr_catboost_runs(sample_data):
    pytest.importorskip("catboost")
    real, synth, schema = sample_data
    # Use small subset for speed
    real_small = real.head(50).copy()
    synth_small = synth.head(50).copy()
    scores = tstr_catboost(real_small, real_small, synth_small, schema, random_seed=0)
    for key in ["task_type", "r2_real", "r2_synth", "r2_pct_diff"]:
        assert key in scores
    assert scores["task_type"] == "regression"


def test_tstr_catboost_uses_weighted_f1_for_classification():
    pytest.importorskip("catboost")
    df = pd.DataFrame(
        {
            "x": [0, 1, 0, 1, 0, 1, 0, 1],
            "cat": ["a", "a", "b", "b", "a", "b", "a", "b"],
            "target": [0, 0, 1, 1, 0, 1, 0, 1],
        }
    )
    schema = TabularSchema(
        continuous_cols=[],
        discrete_cols=["x"],
        categorical_cols=["cat"],
        target_col="target",
    )

    scores = tstr_catboost(
        train_real=df.iloc[:6].copy(),
        test_real=df.iloc[6:].copy(),
        synth_train=df.iloc[:6].copy(),
        schema=schema,
        random_seed=0,
        task_type="classification",
    )

    for key in ["task_type", "f1_weighted_real", "f1_weighted_synth", "f1_weighted_pct_diff"]:
        assert key in scores
    assert scores["task_type"] == "classification"
    assert "r2_real" not in scores


def test_tstr_evaluator_class(sample_data):
    pytest.importorskip("catboost")
    real, synth, schema = sample_data
    evaluator = TSTRCatBoostEvaluator(random_seed=0)
    result = evaluator.evaluate(
        train_real=real.head(50),
        test_real=real.head(50),
        synth_train=synth.head(50),
        schema=schema,
    )
    assert "r2_real" in result.scores
    assert result.meta["evaluator"] == "tstr_catboost"
