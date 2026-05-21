from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tabpfgen")

from genbench.data.schema import TabularSchema
from genbench.generative.tabpfgen.tabpfgen import (
    TabPFGenGenerative,
    _infer_task_type,
)


class DummyTabPFGen:
    """Mock TabPFGen that returns deterministic samples."""

    def __init__(self, **kwargs):
        self.params = dict(kwargs)
        self.fit_calls = []
        self.gen_class_calls = []
        self.gen_reg_calls = []

    def generate_classification(self, X, y, n_samples, balance_classes=False):
        self.gen_class_calls.append(
            {"n_samples": n_samples, "balance_classes": balance_classes}
        )
        X_synth = np.tile(X[:1], (n_samples, 1)).astype(float)
        y_synth = np.full((n_samples,), y[0])
        return X_synth, y_synth

    def generate_regression(self, X, y, n_samples, use_quantiles=True):
        self.gen_reg_calls.append(
            {"n_samples": n_samples, "use_quantiles": use_quantiles}
        )
        X_synth = np.tile(X[:1], (n_samples, 1)).astype(float)
        y_synth = np.full((n_samples,), float(y[0]))
        return X_synth, y_synth


@pytest.fixture
def classification_data():
    np.random.seed(0)
    df = pd.DataFrame({
        "x1": np.random.randn(20),
        "x2": np.random.randn(20),
        "target": np.random.randint(0, 3, 20),
    })
    schema = TabularSchema(
        continuous_cols=["x1", "x2"],
        discrete_cols=[],
        categorical_cols=["target"],
        target_col="target",
    )
    return df, schema


@pytest.fixture
def regression_data():
    np.random.seed(0)
    df = pd.DataFrame({
        "x1": np.random.randn(20),
        "x2": np.random.randn(20),
        "target": np.random.randn(20),
    })
    schema = TabularSchema(
        continuous_cols=["x1", "x2", "target"],
        discrete_cols=[],
        categorical_cols=[],
        target_col="target",
    )
    return df, schema


def test_model_creation():
    model = TabPFGenGenerative()
    assert model.name == "tabpfgen"
    assert model.requires_fit() is True
    assert model.is_conditional() is False


def test_model_state_round_trip():
    model = TabPFGenGenerative(
        n_sgld_steps=123,
        sgld_step_size=0.05,
        sgld_noise_scale=0.02,
        device="cpu",
        balance_classes=False,
        use_quantiles=False,
        seed=7,
    )
    state = model.get_state()
    restored = TabPFGenGenerative.from_state(state)

    assert restored.n_sgld_steps == 123
    assert restored.sgld_step_size == 0.05
    assert restored.sgld_noise_scale == 0.02
    assert restored.device == "cpu"
    assert restored.balance_classes is False
    assert restored.use_quantiles is False
    assert restored.seed == 7


def test_defaults_match_reference():
    model = TabPFGenGenerative()
    assert model.n_sgld_steps == 1000
    assert model.sgld_step_size == 0.01
    assert model.sgld_noise_scale == 0.01
    assert model.device == "auto"
    assert model.balance_classes is True
    assert model.use_quantiles is True


def test_infer_task_type_categorical_target():
    schema = TabularSchema(
        continuous_cols=["x"],
        discrete_cols=[],
        categorical_cols=["t"],
        target_col="t",
    )
    assert _infer_task_type(np.array([0, 1, 0]), schema, "t") == \
        "classification"


def test_infer_task_type_discrete_low_card():
    schema = TabularSchema(
        continuous_cols=["x"],
        discrete_cols=["t"],
        categorical_cols=[],
        target_col="t",
    )
    assert _infer_task_type(np.array([0, 1, 2, 1, 0]), schema, "t") == \
        "classification"


def test_infer_task_type_continuous():
    schema = TabularSchema(
        continuous_cols=["x", "t"],
        discrete_cols=[],
        categorical_cols=[],
        target_col="t",
    )
    assert _infer_task_type(np.linspace(0, 1, 100), schema, "t") == \
        "regression"


@patch("genbench.generative.tabpfgen.tabpfgen.TabPFGen", DummyTabPFGen)
def test_fit_sample_classification(classification_data):
    df, schema = classification_data
    model = TabPFGenGenerative(device="cpu")
    model.fit(df, schema)

    assert model.fitted_ is True
    assert model.task_type_ == "classification"

    synth = model.sample(7)
    assert isinstance(synth, pd.DataFrame)
    assert len(synth) == 7
    assert set(synth.columns) == set(schema.feature_cols + [schema.target_col])


@patch("genbench.generative.tabpfgen.tabpfgen.TabPFGen", DummyTabPFGen)
def test_fit_sample_regression(regression_data):
    df, schema = regression_data
    model = TabPFGenGenerative(device="cpu")
    model.fit(df, schema)

    assert model.fitted_ is True
    assert model.task_type_ == "regression"

    synth = model.sample(5)
    assert isinstance(synth, pd.DataFrame)
    assert len(synth) == 5


def test_sample_without_fit_raises():
    model = TabPFGenGenerative()
    with pytest.raises(RuntimeError, match="Model is not fitted"):
        model.sample(10)


@patch("genbench.generative.tabpfgen.tabpfgen.TabPFGen", DummyTabPFGen)
def test_balance_classes_passed_through(classification_data):
    df, schema = classification_data
    model = TabPFGenGenerative(device="cpu", balance_classes=False)
    model.fit(df, schema)
    model.sample(3)
    # Pull the dummy out via the model_ attribute and check the call kwargs
    assert model.model_.gen_class_calls[0]["balance_classes"] is False


@patch("genbench.generative.tabpfgen.tabpfgen.TabPFGen", DummyTabPFGen)
def test_sample_preserves_column_order(classification_data):
    df, schema = classification_data
    # Reorder columns so target is in the middle
    df_reordered = df[["x1", "target", "x2"]]
    schema2 = TabularSchema(
        continuous_cols=["x1", "x2"],
        discrete_cols=[],
        categorical_cols=["target"],
        target_col="target",
    )
    model = TabPFGenGenerative(device="cpu")
    model.fit(df_reordered, schema2)
    synth = model.sample(4)
    assert list(synth.columns) == ["x1", "target", "x2"]


@patch("genbench.generative.tabpfgen.tabpfgen.TabPFGen", DummyTabPFGen)
def test_use_quantiles_passed_through(regression_data):
    df, schema = regression_data
    model = TabPFGenGenerative(device="cpu", use_quantiles=False)
    model.fit(df, schema)
    model.sample(3)
    assert model.model_.gen_reg_calls[0]["use_quantiles"] is False


def test_fit_without_target_raises():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    schema = TabularSchema(
        continuous_cols=["x"],
        discrete_cols=[],
        categorical_cols=[],
        target_col=None,
    )
    model = TabPFGenGenerative()
    with pytest.raises(ValueError, match="requires a target column"):
        model.fit(df, schema)
