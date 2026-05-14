import pytest

from experiments.ctgan.experiment_models import get_experiment_model, list_experiment_models
from genbench.generative.ctgan.ctgan import CtganGenerative
from genbench.generative.tvae.tvae import TvaeGenerative


def test_registry_lists_ctgan_and_tvae():
    assert list_experiment_models() == ("ctgan", "tvae")


def test_registry_resolves_ctgan_contract():
    spec = get_experiment_model("ctgan")

    assert spec.model_id == "ctgan"
    assert spec.display_name == "CTGAN"
    assert spec.artifact_filename == "ctgan.pkl"

    model = spec.create_generative(discrete_cols=["cat"], model_kwargs={"epochs": 1, "cuda": False})

    assert isinstance(model, CtganGenerative)
    assert model.discrete_cols == ["cat"]
    assert model.ctgan_kwargs["epochs"] == 1


def test_registry_resolves_tvae_contract():
    spec = get_experiment_model("tvae")

    assert spec.model_id == "tvae"
    assert spec.display_name == "TVAE"
    assert spec.artifact_filename == "tvae.pkl"

    model = spec.create_generative(discrete_cols=["cat"], model_kwargs={"epochs": 1, "cuda": False})

    assert isinstance(model, TvaeGenerative)
    assert model.discrete_cols == ["cat"]
    assert model.tvae_kwargs["epochs"] == 1


def test_registry_normalizes_model_id_and_rejects_unknown():
    assert get_experiment_model(" TVAE ").model_id == "tvae"
    with pytest.raises(ValueError, match="Unknown model_id"):
        get_experiment_model("unknown")
