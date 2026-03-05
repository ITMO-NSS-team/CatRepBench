from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("optuna")

from experiments import ctgan_tuning as tune_mod
from genbench.evaluation.base import EvaluationResult
from genbench.data.schema import TabularSchema


class DummyCtganGenerative:
    created: list["DummyCtganGenerative"] = []

    def __init__(self, discrete_cols=None, ctgan_kwargs=None):
        self.discrete_cols = list(discrete_cols or [])
        self.ctgan_kwargs = dict(ctgan_kwargs or {})
        self.train_df: pd.DataFrame | None = None
        self.fitted_ = False
        DummyCtganGenerative.created.append(self)

    def fit(self, df: pd.DataFrame, schema: TabularSchema) -> "DummyCtganGenerative":
        self.train_df = df.reset_index(drop=True).copy()
        self.fitted_ = True
        return self

    def sample(self, n: int, conditions: pd.DataFrame | None = None) -> pd.DataFrame:
        if self.train_df is None:
            raise RuntimeError("Dummy model is not fitted.")
        return self.train_df.sample(n=n, replace=True, random_state=0).reset_index(drop=True)

    def save_artifacts(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "ctgan.pkl").write_bytes(b"dummy-ctgan")


class DummyTSTRCatBoostEvaluator:
    calls = 0

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    def evaluate(self, **kwargs):
        DummyTSTRCatBoostEvaluator.calls += 1
        return EvaluationResult(scores={"r2_synth": 0.42})


def _build_df(n: int = 40) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "x_cont": float(i) / 10.0,
                "x_disc": int(i % 5),
                "x_cat": "a" if i % 2 == 0 else "b",
                "target": int(i % 2),
            }
        )
    return pd.DataFrame(rows)


def _build_regression_df(n: int = 60) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "x_cont": float(i) / 10.0,
                "x_disc": int(i % 7),
                "x_cat": "a" if i % 2 == 0 else "b",
                "target": float(i),  # many unique numeric values -> regression inference
            }
        )
    return pd.DataFrame(rows)


def _build_schema() -> TabularSchema:
    return TabularSchema(
        continuous_cols=["x_cont"],
        discrete_cols=["x_disc"],
        categorical_cols=["x_cat"],
        target_col="target",
    )


def test_tune_ctgan_saves_outputs_and_returns_params(tmp_path, monkeypatch):
    monkeypatch.setattr(tune_mod, "CtganGenerative", DummyCtganGenerative)
    DummyCtganGenerative.created = []

    result = tune_mod.tune_ctgan(
        df=_build_df(),
        schema=_build_schema(),
        dataset="adult sample",
        encoding_method="one hot",
        n_trials=2,
        epochs=1,
        finetune_top_k=0,
        seed=7,
        output_root=tmp_path / "optuna_results",
        device="cpu",
    )

    assert isinstance(result.best_params, dict)
    assert result.best_params
    assert result.output_dir == tmp_path / "optuna_results" / "ctgan" / "adult_sample" / "one_hot"
    assert result.summary_path.exists()
    assert result.best_params_path.exists()
    assert result.trials_path.exists()

    payload = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert payload["dataset"] == "adult sample"
    assert payload["encoding_method"] == "one hot"
    assert payload["n_trials"] == 2
    assert payload["task_type"] == "classification"

    # In classification mode target is included in default discrete columns for CTGAN wrapper.
    assert DummyCtganGenerative.created
    assert "target" in DummyCtganGenerative.created[0].discrete_cols


def test_tune_ctgan_save_model_uses_wrapper_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(tune_mod, "CtganGenerative", DummyCtganGenerative)
    DummyCtganGenerative.created = []

    result = tune_mod.tune_ctgan(
        df=_build_df(),
        schema=_build_schema(),
        dataset="adult",
        encoding_method="binary",
        n_trials=1,
        epochs=1,
        finetune_top_k=0,
        seed=11,
        output_root=tmp_path / "optuna_results",
        save_model=True,
        device="cpu",
    )

    assert result.model_artifacts_dir is not None
    assert result.model_artifacts_dir.exists()
    assert (result.model_artifacts_dir / "ctgan.pkl").exists()


def test_tune_ctgan_and_return_params_wrapper(tmp_path, monkeypatch):
    monkeypatch.setattr(tune_mod, "CtganGenerative", DummyCtganGenerative)
    DummyCtganGenerative.created = []

    out = tune_mod.tune_ctgan_and_return_params(
        df=_build_df(),
        schema=_build_schema(),
        dataset="my ds",
        encoding_method="method x",
        n_trials=1,
        epochs=1,
        finetune_top_k=0,
        seed=13,
        output_root=tmp_path / "optuna_results",
        device="cpu",
    )

    assert set(out.keys()) == {"best_params", "best_value", "best_source", "summary_path"}
    assert isinstance(out["best_params"], dict)
    assert Path(out["summary_path"]).exists()


def test_tune_ctgan_regression_branch_uses_utility_evaluator(tmp_path, monkeypatch):
    monkeypatch.setattr(tune_mod, "CtganGenerative", DummyCtganGenerative)
    monkeypatch.setattr(tune_mod, "TSTRCatBoostEvaluator", DummyTSTRCatBoostEvaluator)
    DummyCtganGenerative.created = []
    DummyTSTRCatBoostEvaluator.calls = 0

    result = tune_mod.tune_ctgan(
        df=_build_regression_df(),
        schema=_build_schema(),
        dataset="housing",
        encoding_method="original",
        n_trials=1,
        epochs=1,
        finetune_top_k=0,
        seed=17,
        output_root=tmp_path / "optuna_results",
        device="cpu",
    )

    payload = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert payload["task_type"] == "regression"
    assert DummyTSTRCatBoostEvaluator.calls > 0
    assert DummyCtganGenerative.created
    assert "target" not in DummyCtganGenerative.created[0].discrete_cols
