from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("optuna")

from experiments import ctgan_tuning as tune_mod
from genbench.data.schema import TabularSchema


class DummyCtganGenerative:
    created: list["DummyCtganGenerative"] = []

    def __init__(self, discrete_cols=None, ctgan_kwargs=None):
        self.discrete_cols = list(discrete_cols or [])
        self.ctgan_kwargs = dict(ctgan_kwargs or {})
        self.train_df: pd.DataFrame | None = None
        self.fit_schema: TabularSchema | None = None
        self.sample_sizes: list[int] = []
        self.fitted_ = False
        DummyCtganGenerative.created.append(self)

    def fit(self, df: pd.DataFrame, schema: TabularSchema) -> "DummyCtganGenerative":
        self.train_df = df.reset_index(drop=True).copy()
        self.fit_schema = schema
        self.fitted_ = True
        return self

    def sample(self, n: int, conditions: pd.DataFrame | None = None) -> pd.DataFrame:
        if self.train_df is None:
            raise RuntimeError("Dummy model is not fitted.")
        self.sample_sizes.append(int(n))
        return self.train_df.sample(n=n, replace=True, random_state=0).reset_index(drop=True)

    def save_artifacts(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "ctgan.pkl").write_bytes(b"dummy-ctgan")


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


def _build_schema_with_high_cardinality_discrete() -> TabularSchema:
    return TabularSchema(
        continuous_cols=["x_cont"],
        discrete_cols=["x_disc_high_card"],
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
        encoding_method="one_hot_representation",
        n_trials=2,
        epochs=1,
        seed=7,
        output_root=tmp_path / "optuna_results",
        device="cpu",
    )

    assert isinstance(result.best_params, dict)
    assert result.best_params
    assert result.output_dir == tmp_path / "optuna_results" / "ctgan" / "adult_sample" / "one_hot_representation"
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
    assert payload["holdout"]["random_seed"] == 5
    assert payload["preprocessing"]["representation_name"] == "one_hot_representation"
    assert payload["preprocessing"]["discrete_unique_threshold"] == 20

    # In classification mode target is included in default discrete columns for CTGAN wrapper.
    assert DummyCtganGenerative.created
    assert "target" in DummyCtganGenerative.created[0].discrete_cols
    assert DummyCtganGenerative.created[0].sample_sizes
    assert DummyCtganGenerative.created[0].sample_sizes[0] == 8  # len(val) for 80/20 split on n=40
    assert DummyCtganGenerative.created[0].train_df is not None
    assert "x_cat" not in DummyCtganGenerative.created[0].train_df.columns
    assert any(col.startswith("x_cat__") for col in DummyCtganGenerative.created[0].train_df.columns)


def test_tune_ctgan_save_model_uses_wrapper_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(tune_mod, "CtganGenerative", DummyCtganGenerative)
    DummyCtganGenerative.created = []

    result = tune_mod.tune_ctgan(
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
    assert result.model_artifacts_dir.exists()
    assert (result.model_artifacts_dir / "ctgan.pkl").exists()


def test_tune_ctgan_and_return_params_wrapper(tmp_path, monkeypatch):
    monkeypatch.setattr(tune_mod, "CtganGenerative", DummyCtganGenerative)
    DummyCtganGenerative.created = []

    out = tune_mod.tune_ctgan_and_return_params(
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

    assert set(out.keys()) == {"best_params", "best_value", "best_source", "summary_path"}
    assert isinstance(out["best_params"], dict)
    assert Path(out["summary_path"]).exists()


def test_tune_ctgan_regression_branch_scales_target_and_keeps_it_continuous(tmp_path, monkeypatch):
    monkeypatch.setattr(tune_mod, "CtganGenerative", DummyCtganGenerative)
    DummyCtganGenerative.created = []

    result = tune_mod.tune_ctgan(
        df=_build_regression_df(),
        schema=_build_schema(),
        dataset="housing",
        encoding_method="one_hot_representation",
        n_trials=1,
        epochs=1,
        seed=17,
        output_root=tmp_path / "optuna_results",
        device="cpu",
    )

    payload = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert payload["task_type"] == "regression"
    assert DummyCtganGenerative.created
    assert "target" not in DummyCtganGenerative.created[0].discrete_cols
    assert payload["preprocessing"]["target_processing"]["target_scaled"] is True


def test_tune_ctgan_holdout_split_is_independent_from_optuna_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(tune_mod, "CtganGenerative", DummyCtganGenerative)
    DummyCtganGenerative.created = []

    tune_mod.tune_ctgan(
        df=_build_df(),
        schema=_build_schema(),
        dataset="seed-a",
        encoding_method="one_hot_representation",
        n_trials=1,
        epochs=1,
        seed=1,
        output_root=tmp_path / "optuna_results",
        device="cpu",
    )
    tune_mod.tune_ctgan(
        df=_build_df(),
        schema=_build_schema(),
        dataset="seed-b",
        encoding_method="one_hot_representation",
        n_trials=1,
        epochs=1,
        seed=999,
        output_root=tmp_path / "optuna_results",
        device="cpu",
    )

    assert len(DummyCtganGenerative.created) >= 2
    train_a = DummyCtganGenerative.created[0].train_df
    train_b = DummyCtganGenerative.created[1].train_df
    assert train_a is not None
    assert train_b is not None
    pd.testing.assert_frame_equal(train_a, train_b)


def test_tune_ctgan_raises_on_invalid_encoding_method(tmp_path, monkeypatch):
    monkeypatch.setattr(tune_mod, "CtganGenerative", DummyCtganGenerative)
    DummyCtganGenerative.created = []

    with pytest.raises(ValueError, match="Unsupported encoding_method"):
        tune_mod.tune_ctgan(
            df=_build_df(),
            schema=_build_schema(),
            dataset="adult",
            encoding_method="one hot",
            n_trials=1,
            epochs=1,
            output_root=tmp_path / "optuna_results",
            device="cpu",
        )


def test_tune_ctgan_retypes_high_cardinality_integer_feature_as_continuous(tmp_path, monkeypatch):
    monkeypatch.setattr(tune_mod, "CtganGenerative", DummyCtganGenerative)
    DummyCtganGenerative.created = []

    n = 80
    df = pd.DataFrame(
        {
            "x_cont": [float(i) / 10.0 for i in range(n)],
            "x_disc_high_card": list(range(n)),  # high cardinality integer feature
            "x_cat": ["a" if i % 2 == 0 else "b" for i in range(n)],
            "target": [i % 2 for i in range(n)],
        }
    )

    tune_mod.tune_ctgan(
        df=df,
        schema=_build_schema_with_high_cardinality_discrete(),
        dataset="high-card",
        encoding_method="ordinal_representation",
        n_trials=1,
        epochs=1,
        output_root=tmp_path / "optuna_results",
        device="cpu",
    )

    assert DummyCtganGenerative.created
    used_discrete = DummyCtganGenerative.created[0].discrete_cols
    assert "x_disc_high_card" not in used_discrete


def test_tune_ctgan_label_encodes_categorical_target_for_classification(tmp_path, monkeypatch):
    monkeypatch.setattr(tune_mod, "CtganGenerative", DummyCtganGenerative)
    DummyCtganGenerative.created = []

    n = 60
    df = pd.DataFrame(
        {
            "x_cont": [float(i) / 10.0 for i in range(n)],
            "x_disc": [i % 5 for i in range(n)],
            "x_cat": ["a" if i % 2 == 0 else "b" for i in range(n)],
            "target": ["yes" if i % 2 == 0 else "no" for i in range(n)],
        }
    )
    schema = TabularSchema(
        continuous_cols=["x_cont"],
        discrete_cols=["x_disc"],
        categorical_cols=["x_cat"],
        target_col="target",
    )

    result = tune_mod.tune_ctgan(
        df=df,
        schema=schema,
        dataset="categorical-target",
        encoding_method="one_hot_representation",
        n_trials=1,
        epochs=1,
        output_root=tmp_path / "optuna_results",
        device="cpu",
    )

    payload = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert payload["preprocessing"]["target_processing"]["target_encoded"] is True
    assert DummyCtganGenerative.created
    fitted_train = DummyCtganGenerative.created[0].train_df
    assert fitted_train is not None
    assert pd.api.types.is_integer_dtype(fitted_train["target"])
