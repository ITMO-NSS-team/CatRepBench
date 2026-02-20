import sys
import types

import pandas as pd
import pytest

from genbench.data.schema import TabularSchema
from genbench.generative.ctgan.ctgan import CtganGenerative


class DummyCTGAN:
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
def dummy_ctgan_module(monkeypatch):
    """Inject a dummy ctgan.CTGAN so tests don't require the real package."""
    mod = types.ModuleType("ctgan")
    mod.CTGAN = DummyCTGAN
    monkeypatch.setitem(sys.modules, "ctgan", mod)
    return DummyCTGAN


def test_fit_uses_schema_discrete_columns(dummy_ctgan_module):
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

    model = CtganGenerative()
    model.fit(df, schema)

    assert model.fitted_ is True
    assert model.used_discrete_cols_ == ["cat", "disc"]
    assert model.model_.fit_called is not None
    _, passed_discrete = model.model_.fit_called
    assert passed_discrete == ["cat", "disc"]


def test_sample_returns_dataframe_after_fit(dummy_ctgan_module):
    df = pd.DataFrame({"cat": ["a"], "disc": [1]})
    schema = TabularSchema(continuous_cols=[], discrete_cols=["disc"], categorical_cols=["cat"])

    model = CtganGenerative()
    model.fit(df, schema)
    out = model.sample(3)

    assert isinstance(out, pd.DataFrame)
    assert len(out) == 3
    assert list(out.columns) == ["synth"]


def test_get_loss_history_formats_dict(dummy_ctgan_module):
    model = CtganGenerative()
    dummy = dummy_ctgan_module()
    dummy.loss_values = pd.DataFrame(
        {
            "Epoch": [0, 1],
            "Generator Loss": [1.0, 0.5],
            "Discriminator Loss": [0.8, 0.4],
        }
    )
    model.model_ = dummy
    model.fitted_ = True

    hist = model.get_loss_history()
    assert hist == {
        "generator_loss": [1.0, 0.5],
        "discriminator_loss": [0.8, 0.4],
    }


def test_save_and_load_artifacts_round_trip(tmp_path, dummy_ctgan_module):
    model = CtganGenerative()
    dummy = dummy_ctgan_module()
    dummy.some_attr = "keep_me"
    model.model_ = dummy
    model.used_discrete_cols_ = ["cat"]
    model.fitted_ = True

    artifacts_dir = tmp_path / "artifacts"
    model.save_artifacts(artifacts_dir)

    loaded = CtganGenerative.load_artifacts(artifacts_dir)
    assert isinstance(loaded.model_, dummy_ctgan_module)
    assert loaded.used_discrete_cols_ == ["cat"]
    assert loaded.fitted_ is True


def test_state_round_trip():
    model = CtganGenerative(discrete_cols=["a", "b"], ctgan_kwargs={"epochs": 10})
    state = model.get_state()
    restored = CtganGenerative.from_state(state)

    assert restored.discrete_cols == ["a", "b"]
    assert restored.ctgan_kwargs == {"epochs": 10}
