from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from genbench.data.schema import TabularSchema
from genbench.generative.tabddpm.tabddpm import TabDdpmGenerative


class DummyMLPDiffusion(nn.Module):
    """Dummy MLP diffusion model for testing."""

    def __init__(self, d_in=10, num_classes=0, is_y_cond=False,
                 rtdl_params=None, dim_t=128):
        super().__init__()
        self.d_in = d_in
        self.dim_t = dim_t
        self.linear = nn.Linear(d_in, d_in)

    def forward(self, x, timesteps, y=None):
        return self.linear(x)


class DummyDiffusion(nn.Module):
    """Dummy diffusion model for testing."""

    def __init__(self, num_classes, num_numerical_features, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_numerical_features = num_numerical_features
        self._denoise_fn = kwargs.get('denoise_fn')
        # Create a simple linear layer for loss computation
        self.linear = nn.Linear(10, 1)

    def mixed_loss(self, x, out_dict):
        # Create a real computation graph for backward pass
        dummy_input = torch.randn(1, 10)
        dummy_out = self.linear(dummy_input)
        loss_multi = dummy_out.sum()
        loss_gauss = dummy_out.sum() * 0.5
        return loss_multi, loss_gauss

    def sample_all(self, num_samples, batch_size, y_dist, ddim=False):
        # Generate dummy samples
        n_feat = self.num_numerical_features
        if self.num_classes[0] > 0:
            n_feat += len(self.num_classes)  # categorical columns

        X = torch.randn(num_samples, n_feat)
        y = torch.zeros(num_samples, dtype=torch.long)
        return X, {'y': y}

    def eval(self):
        return self

    def train(self):
        return self

    def to(self, device):
        return self

    def parameters(self):
        return self.linear.parameters()


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    df = pd.DataFrame({
        "cont1": np.random.randn(100),
        "cont2": np.random.randn(100),
        "disc1": np.random.randint(0, 10, 100),
        "cat1": np.random.choice(["a", "b", "c"], 100),
        "cat2": np.random.choice(["x", "y"], 100),
    })

    schema = TabularSchema(
        continuous_cols=["cont1", "cont2"],
        discrete_cols=["disc1"],
        categorical_cols=["cat1", "cat2"],
    )

    return df, schema


@pytest.fixture
def minimal_data():
    """Create minimal data for testing."""
    df = pd.DataFrame({
        "cont": [0.1, 0.2, 0.3, 0.4, 0.5],
        "cat": ["a", "b", "a", "b", "a"],
    })

    schema = TabularSchema(
        continuous_cols=["cont"],
        discrete_cols=[],
        categorical_cols=["cat"],
    )

    return df, schema


def test_model_creation():
    """Test that TabDdpmGenerative can be instantiated."""
    model = TabDdpmGenerative()
    assert model.name == "tabddpm"
    assert model.requires_fit() is True
    assert model.is_conditional() is False


def test_model_state_round_trip():
    """Test get_state and from_state methods."""
    model = TabDdpmGenerative(
        num_timesteps=500,
        num_epochs=50,
        batch_size=512,
        lr=0.001,
    )

    state = model.get_state()
    restored = TabDdpmGenerative.from_state(state)

    assert restored.num_timesteps == 500
    assert restored.num_epochs == 50
    assert restored.batch_size == 512
    assert restored.lr == 0.001


def test_preprocess_data(sample_data):
    """Test data preprocessing."""
    df, schema = sample_data
    model = TabDdpmGenerative()

    X = model._preprocess_data(df, schema)

    # Check that preprocessing sets internal attributes
    assert model.num_numerical_features_ == 3  # 2 continuous + 1 discrete
    assert len(model.num_classes_) == 2  # 2 categorical columns
    assert X.shape[0] == len(df)


def test_preprocess_data_numerical_only():
    """Test preprocessing with numerical features only."""
    df = pd.DataFrame({
        "cont1": [1.0, 2.0, 3.0],
        "cont2": [4.0, 5.0, 6.0],
    })

    schema = TabularSchema(
        continuous_cols=["cont1", "cont2"],
        discrete_cols=[],
        categorical_cols=[],
    )

    model = TabDdpmGenerative()
    X = model._preprocess_data(df, schema)

    assert model.num_numerical_features_ == 2
    assert model.num_classes_[0] == 0  # No categorical features


def test_preprocess_data_categorical_only():
    """Test preprocessing with categorical features only."""
    df = pd.DataFrame({
        "cat1": ["a", "b", "c"],
        "cat2": ["x", "y", "x"],
    })

    schema = TabularSchema(
        continuous_cols=[],
        discrete_cols=[],
        categorical_cols=["cat1", "cat2"],
    )

    model = TabDdpmGenerative()
    X = model._preprocess_data(df, schema)

    assert model.num_numerical_features_ == 0
    assert len(model.num_classes_) == 2


@patch('genbench.generative.tabddpm.tabddpm.MLPDiffusion', DummyMLPDiffusion)
@patch('genbench.generative.tabddpm.tabddpm.GaussianMultinomialDiffusion',
       DummyDiffusion)
def test_fit_minimal(minimal_data):
    """Test model fitting with minimal data."""
    df, schema = minimal_data

    model = TabDdpmGenerative(
        num_epochs=2,
        batch_size=2,
        device='cpu',
    )

    model.fit(df, schema)

    assert model.fitted_ is True
    assert model.model_ is not None
    assert model.diffusion_ is not None


@patch('genbench.generative.tabddpm.tabddpm.MLPDiffusion', DummyMLPDiffusion)
@patch('genbench.generative.tabddpm.tabddpm.GaussianMultinomialDiffusion',
       DummyDiffusion)
def test_sample_after_fit(minimal_data):
    """Test sampling after model is fitted."""
    df, schema = minimal_data

    model = TabDdpmGenerative(
        num_epochs=1,
        batch_size=2,
        device='cpu',
    )

    model.fit(df, schema)

    # Mock the diffusion model for sampling
    model.diffusion_ = DummyDiffusion(
        num_classes=model.num_classes_,
        num_numerical_features=model.num_numerical_features_,
    )

    samples = model.sample(5)

    assert isinstance(samples, pd.DataFrame)
    assert len(samples) == 5


def test_sample_without_fit_raises():
    """Test that sampling without fit raises an error."""
    model = TabDdpmGenerative()

    with pytest.raises(RuntimeError, match="Model is not fitted"):
        model.sample(10)


def test_conditional_sampling_raises():
    """Test that conditional sampling raises NotImplementedError."""
    model = TabDdpmGenerative()

    with pytest.raises(NotImplementedError,
                       match="does not support conditional"):
        model.sample(10, conditions=pd.DataFrame({"x": [1]}))


@patch('genbench.generative.tabddpm.tabddpm.MLPDiffusion', DummyMLPDiffusion)
@patch('genbench.generative.tabddpm.tabddpm.GaussianMultinomialDiffusion',
       DummyDiffusion)
def test_get_loss_history(minimal_data):
    """Test loss history retrieval."""
    df, schema = minimal_data

    model = TabDdpmGenerative(
        num_epochs=2,
        batch_size=2,
        device='cpu',
    )

    model.fit(df, schema)

    history = model.get_loss_history()

    assert history is not None
    assert 'loss' in history
    assert 'multinomial_loss' in history
    assert 'gaussian_loss' in history
    assert len(history['loss']) == 2  # 2 epochs


@patch('genbench.generative.tabddpm.tabddpm.MLPDiffusion', DummyMLPDiffusion)
@patch('genbench.generative.tabddpm.tabddpm.GaussianMultinomialDiffusion',
       DummyDiffusion)
def test_save_and_load_artifacts(tmp_path, minimal_data):
    """Test saving and loading model artifacts."""
    df, schema = minimal_data

    model = TabDdpmGenerative(
        num_epochs=1,
        batch_size=2,
        device='cpu',
    )

    model.fit(df, schema)

    # Save artifacts
    artifacts_dir = tmp_path / "artifacts"
    model.save_artifacts(artifacts_dir)

    # Check files exist
    assert (artifacts_dir / "tabddpm_artifacts.pkl").exists()
    assert (artifacts_dir / "model.pt").exists()

    # Load only the preprocessing artifacts (not the model weights)
    import pickle
    with open(artifacts_dir / "tabddpm_artifacts.pkl", "rb") as f:
        payload = pickle.load(f)

    assert payload["fitted"] is True
    assert payload["num_numerical_features"] == model.num_numerical_features_
    np.testing.assert_array_equal(payload["num_classes"], model.num_classes_)


def test_postprocess_data(sample_data):
    """Test postprocessing generated data."""
    df, schema = sample_data

    model = TabDdpmGenerative()
    model._preprocess_data(df, schema)

    # Create dummy generated data
    n_samples = 10
    n_num = model.num_numerical_features_
    n_cat = len(model.num_classes_) if model.num_classes_[0] > 0 else 0

    X_gen = np.random.randn(n_samples, n_num + n_cat)

    result = model._postprocess_data(X_gen)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == n_samples
    assert list(result.columns) == model.column_order_


def test_custom_parameters():
    """Test model with custom parameters."""
    model = TabDdpmGenerative(
        num_timesteps=500,
        num_epochs=10,
        batch_size=256,
        lr=0.001,
        weight_decay=1e-3,
        dim_t=64,
        d_layers=[128, 128],
        dropout=0.1,
        scheduler='linear',
        gaussian_loss_type='kl',
    )

    assert model.num_timesteps == 500
    assert model.num_epochs == 10
    assert model.batch_size == 256
    assert model.lr == 0.001
    assert model.weight_decay == 1e-3
    assert model.dim_t == 64
    assert model.d_layers == [128, 128]
    assert model.dropout == 0.1
    assert model.scheduler == 'linear'
    assert model.gaussian_loss_type == 'kl'
