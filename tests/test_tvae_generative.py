import sys
import types

import pandas as pd
import pytest

from genbench.data.schema import TabularSchema
from genbench.generative.tvae import tvae as tvae_module
from genbench.generative.tvae.tvae import TvaeGenerative


class DummyTVAE:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fit_called = None
        self.loss_values = None

    def fit(self, df, discrete_columns=()):
        # store a copy to ensure DataFrame is handled
        self.fit_called = (df.copy(), list(discrete_columns))

    def sample(self, n):
        return pd.DataFrame({"synth": list(range(n))})


@pytest.fixture(autouse=True)
def dummy_tvae_module(monkeypatch):
    """Inject a dummy TVAE directly into the wrapper module."""
    mod = types.ModuleType("ctgan")
    mod.TVAE = DummyTVAE
    monkeypatch.setitem(sys.modules, "ctgan", mod)
    monkeypatch.setattr(tvae_module, "TVAE", DummyTVAE)
    return DummyTVAE


def test_fit_uses_schema_discrete_columns(dummy_tvae_module):
    df = pd.DataFrame(
        {
            "cat": ["a", "b"],
            "disc": [1, 2],
            "cont": [0.1, 0.2],
        }
    )
    schema = TabularSchema(
        continuous_cols=["cont"],
        discrete_cols=["disc"],
        categorical_cols=["cat"],
    )

    model = TvaeGenerative()
    model.fit(df, schema)

    assert model.fitted_ is True
    assert model.used_discrete_cols_ == ["cat", "disc"]
    assert model.model_.fit_called is not None
    _, passed_discrete = model.model_.fit_called
    assert passed_discrete == ["cat", "disc"]


def test_sample_returns_dataframe_after_fit(dummy_tvae_module):
    df = pd.DataFrame({"cat": ["a"], "disc": [1]})
    schema = TabularSchema(continuous_cols=[], discrete_cols=["disc"], categorical_cols=["cat"])

    model = TvaeGenerative()
    model.fit(df, schema)
    out = model.sample(3)

    assert isinstance(out, pd.DataFrame)
    assert len(out) == 3
    assert list(out.columns) == ["synth"]


def test_get_loss_history_formats_dict(dummy_tvae_module):
    model = TvaeGenerative()
    dummy = dummy_tvae_module()
    dummy.loss_values = pd.DataFrame(
        {
            "Epoch": [0, 1],
            "Loss": [1.0, 0.5],
        }
    )
    model.model_ = dummy
    model.fitted_ = True

    hist = model.get_loss_history()
    assert hist == {"loss": [1.0, 0.5]}


def test_save_and_load_artifacts_round_trip(tmp_path, dummy_tvae_module):
    model = TvaeGenerative()
    dummy = dummy_tvae_module()
    dummy.some_attr = "keep_me"
    model.model_ = dummy
    model.used_discrete_cols_ = ["cat"]
    model.fitted_ = True

    artifacts_dir = tmp_path / "artifacts"
    model.save_artifacts(artifacts_dir)

    loaded = TvaeGenerative.load_artifacts(artifacts_dir)
    assert isinstance(loaded.model_, dummy_tvae_module)
    assert loaded.used_discrete_cols_ == ["cat"]
    assert loaded.fitted_ is True


def test_state_round_trip():
    model = TvaeGenerative(discrete_cols=["a", "b"], tvae_kwargs={"epochs": 10})
    state = model.get_state()
    restored = TvaeGenerative.from_state(state)

    assert restored.discrete_cols == ["a", "b"]
    assert restored.tvae_kwargs == {"epochs": 10}
