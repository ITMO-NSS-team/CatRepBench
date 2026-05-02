from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("optuna")

from experiments.tvae import tvae_tuning as tune_mod
from genbench.data.schema import TabularSchema


class DummyTvaeGenerative:
    created: list["DummyTvaeGenerative"] = []

    def __init__(self, *, discrete_cols=None, tvae_kwargs=None):
        self.discrete_cols = list(discrete_cols or [])
        self.tvae_kwargs = dict(tvae_kwargs or {})
        self.train_df: pd.DataFrame | None = None
        self.fit_schema: TabularSchema | None = None
        self.sample_sizes: list[int] = []
        DummyTvaeGenerative.created.append(self)

    def fit(self, df: pd.DataFrame, schema: TabularSchema) -> "DummyTvaeGenerative":
        self.train_df = df.reset_index(drop=True).copy()
        self.fit_schema = schema
        return self

    def sample(self, n: int, conditions: pd.DataFrame | None = None) -> pd.DataFrame:
        if self.train_df is None:
            raise RuntimeError("Dummy model is not fitted.")
        self.sample_sizes.append(int(n))
        return self.train_df.sample(n=n, replace=True, random_state=0).reset_index(drop=True)

    def save_artifacts(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "tvae.pkl").write_bytes(b"dummy-tvae")


def _build_df(n: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x_cont": [float(i) / 10.0 for i in range(n)],
            "x_disc": [i % 5 for i in range(n)],
            "x_cat": ["a" if i % 2 == 0 else "b" for i in range(n)],
            "target": [i % 2 for i in range(n)],
        }
    )


def _build_schema() -> TabularSchema:
    return TabularSchema(
        continuous_cols=["x_cont"],
        discrete_cols=["x_disc"],
        categorical_cols=["x_cat"],
        target_col="target",
    )


@pytest.fixture(autouse=True)
def fast_score(monkeypatch):
    monkeypatch.setattr(
        tune_mod,
        "_score_synthetic",
        lambda **kwargs: (0.25, {"objective_score": 0.25, "wasserstein_mean": 0.25}),
    )


def test_tune_tvae_saves_outputs_and_returns_params(tmp_path, monkeypatch):
    monkeypatch.setattr(tune_mod, "TvaeGenerative", DummyTvaeGenerative)
    DummyTvaeGenerative.created = []

    result = tune_mod.tune_tvae(
        df=_build_df(),
        schema=_build_schema(),
        dataset="adult sample",
        encoding_method="one_hot_representation",
        n_trials=2,
        epochs=1,
        seed=7,
        output_root=tmp_path / "optuna_results",
        device="cpu",
    )

    assert isinstance(result.best_params, dict)
    assert result.best_params
    assert result.output_dir == tmp_path / "optuna_results" / "tvae" / "adult_sample" / "one_hot_representation"
    assert result.summary_path.exists()
    assert result.best_params_path.exists()
    assert result.trials_path.exists()

    payload = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert payload["dataset"] == "adult sample"
    assert payload["encoding_method"] == "one_hot_representation"
    assert payload["n_trials"] == 2
    assert payload["task_type"] == "classification"
    assert payload["objective_metric"] == "wasserstein_mean"
    assert payload["objective_direction"] == "minimize"

    assert DummyTvaeGenerative.created
    assert "target" in DummyTvaeGenerative.created[0].discrete_cols
    assert DummyTvaeGenerative.created[0].sample_sizes[0] == 8
    assert "compress_dims" in DummyTvaeGenerative.created[0].tvae_kwargs
    assert "decompress_dims" in DummyTvaeGenerative.created[0].tvae_kwargs


def test_tune_tvae_uses_300_epochs_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(tune_mod, "TvaeGenerative", DummyTvaeGenerative)
    DummyTvaeGenerative.created = []

    tune_mod.tune_tvae(
        df=_build_df(),
        schema=_build_schema(),
        dataset="adult sample",
        encoding_method="one_hot_representation",
        n_trials=1,
        output_root=tmp_path / "optuna_results",
        device="cpu",
    )

    assert DummyTvaeGenerative.created
    assert DummyTvaeGenerative.created[0].tvae_kwargs["epochs"] == 300


def test_tune_tvae_save_model_uses_wrapper_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(tune_mod, "TvaeGenerative", DummyTvaeGenerative)
    DummyTvaeGenerative.created = []

    result = tune_mod.tune_tvae(
        df=_build_df(),
        schema=_build_schema(),
        dataset="adult",
        encoding_method="ordinal_representation",
        n_trials=1,
        epochs=1,
        seed=11,
        output_root=tmp_path / "optuna_results",
        save_model=True,
        device="cpu",
    )

    assert result.model_artifacts_dir is not None
    assert (result.model_artifacts_dir / "tvae.pkl").exists()


def test_select_tvae_best_params_wrapper_returns_only_selection_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(tune_mod, "TvaeGenerative", DummyTvaeGenerative)
    DummyTvaeGenerative.created = []

    out = tune_mod.select_tvae_best_params(
        df=_build_df(),
        schema=_build_schema(),
        dataset="my ds",
        encoding_method="one_hot_representation",
        n_trials=1,
        epochs=1,
        seed=13,
        output_root=tmp_path / "optuna_results",
        device="cpu",
    )

    assert set(out.keys()) == {"best_params", "best_value", "best_source"}
    assert isinstance(out["best_params"], dict)


def test_estimate_tvae_runtime_projects_sampled_fit_to_full_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(tune_mod, "TvaeGenerative", DummyTvaeGenerative)
    DummyTvaeGenerative.created = []

    monotonic_values = iter([10.0, 22.0, 22.0, 25.0])
    monkeypatch.setattr(tune_mod.time, "monotonic", lambda: next(monotonic_values))

    result = tune_mod.estimate_tvae_runtime(
        df=_build_df(n=50),
        schema=_build_schema(),
        dataset="adult sample",
        encoding_method="one_hot_representation",
        sample_epochs=10,
        projected_epochs=300,
        projected_total_runs=35,
        output_dir=tmp_path / "estimate",
        device="cpu",
    )

    assert result.summary_path.exists()
    assert DummyTvaeGenerative.created
    assert DummyTvaeGenerative.created[0].tvae_kwargs["epochs"] == 10
    assert result.projected_trial_seconds == pytest.approx(363.0)
    assert result.projected_full_pipeline_seconds == pytest.approx(12705.0)

    payload = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "runtime_estimate"
    assert payload["sample_epochs"] == 10
    assert payload["projected_epochs"] == 300
    assert payload["projected_total_runs"] == 35
    assert payload["projected_full_pipeline_hours"] == pytest.approx(12705.0 / 3600.0)
