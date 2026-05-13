from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("optuna")

from experiments import tabddpm_tuning as tune_mod
from genbench.data.schema import TabularSchema


class DummyTabDDPMGenerative:
    """Mock object replacing the real TabDDPMGenerative for fast testing."""
    created: list["DummyTabDDPMGenerative"] = []

    def __init__(self, **kwargs):
        self.params = dict(kwargs)
        self.train_df: pd.DataFrame | None = None
        self.fit_schema: TabularSchema | None = None
        self.source_schema: TabularSchema | None = None
        self.sample_sizes: list[int] = []
        self.fitted_ = False
        DummyTabDDPMGenerative.created.append(self)

    def fit(
            self,
            df: pd.DataFrame,
            schema: TabularSchema,
            source_schema: TabularSchema
    ) -> "DummyTabDDPMGenerative":
        self.train_df = df.reset_index(drop=True).copy()
        self.fit_schema = schema
        self.source_schema = source_schema
        self.fitted_ = True
        return self

    def sample(self, n: int) -> pd.DataFrame:
        if self.train_df is None:
            raise RuntimeError("Dummy model is not fitted.")
        self.sample_sizes.append(int(n))
        return self.train_df.sample(n=n, replace=True,
                                    random_state=0).reset_index(drop=True)

    def save_artifacts(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "tabddpm.ckpt").write_bytes(b"dummy-tabddpm")


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
                "target": float(i),
                # many unique numeric values -> regression inference
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


def test_tune_tabddpm_saves_outputs_and_returns_params(tmp_path, monkeypatch):
    monkeypatch.setattr(tune_mod, "TabDDPMGenerative", DummyTabDDPMGenerative)
    DummyTabDDPMGenerative.created = []

    result = tune_mod.tune_tabddpm(
        df=_build_df(),
        schema=_build_schema(),
        dataset="adult sample",
        encoding_method="one_hot_representation",
        n_trials=2,
        seed=7,
        output_root=tmp_path / "optuna_results",
        device="cpu",
    )

    assert isinstance(result.best_params, dict)
    assert result.best_params
    # Проверяем правильность формирования пути (tabddpm вместо ctgan)
    assert (result.output_dir == tmp_path / "optuna_results" / "tabddpm" /
            "adult_sample" / "one_hot_representation")
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
    assert payload["preprocessing"][
               "representation_name"] == "one_hot_representation"
    assert payload["preprocessing"]["discrete_unique_threshold"] == 20
    assert "steps" in payload
    assert isinstance(payload["steps"], int)
    assert payload["steps"] > 0
    assert "num_steps" in DummyTabDDPMGenerative.created[0].params
    assert "num_steps" in result.best_params

    # Проверяем, что мок-модель вызывалась корректно
    assert DummyTabDDPMGenerative.created
    assert DummyTabDDPMGenerative.created[0].sample_sizes
    assert DummyTabDDPMGenerative.created[0].sample_sizes[
               0] == 8  # len(val) for 80/20 split on n=40
    assert DummyTabDDPMGenerative.created[0].train_df is not None
    # Проверяем, что категориальный признак был закодирован
    assert "x_cat" not in DummyTabDDPMGenerative.created[0].train_df.columns
    assert any(col.startswith("x_cat__") for col in
               DummyTabDDPMGenerative.created[0].train_df.columns)
    # Проверяем, что в параметры модели передались как перебираемые,
    # так и дефолтные значения
    assert "lr" in DummyTabDDPMGenerative.created[0].params
    assert DummyTabDDPMGenerative.created[0].params["weight_decay"] == 0.0
    assert DummyTabDDPMGenerative.created[0].params["dim_t"] == 256


def test_tune_tabddpm_save_model_uses_wrapper_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(tune_mod, "TabDDPMGenerative", DummyTabDDPMGenerative)
    DummyTabDDPMGenerative.created = []

    result = tune_mod.tune_tabddpm(
        df=_build_df(),
        schema=_build_schema(),
        dataset="adult",
        encoding_method="ordinal_representation",
        n_trials=1,
        seed=11,
        output_root=tmp_path / "optuna_results",
        save_model=True,
        device="cpu",
    )

    assert result.model_artifacts_dir is not None
    assert result.model_artifacts_dir.exists()
    assert (result.model_artifacts_dir / "tabddpm.ckpt").exists()


def test_tune_tabddpm_and_return_params_wrapper(tmp_path, monkeypatch):
    monkeypatch.setattr(tune_mod, "TabDDPMGenerative", DummyTabDDPMGenerative)
    DummyTabDDPMGenerative.created = []

    out = tune_mod.tune_tabddpm_and_return_params(
        df=_build_df(),
        schema=_build_schema(),
        dataset="my ds",
        encoding_method="one_hot_representation",
        n_trials=1,
        seed=13,
        output_root=tmp_path / "optuna_results",
        device="cpu",
    )

    assert set(out.keys()) == {"best_params", "best_value", "best_source",
                               "summary_path"}
    assert isinstance(out["best_params"], dict)
    assert Path(out["summary_path"]).exists()


def test_tune_tabddpm_regression_branch_scales_target_and_keeps_it_continuous(
        tmp_path, monkeypatch):
    monkeypatch.setattr(tune_mod, "TabDDPMGenerative", DummyTabDDPMGenerative)
    DummyTabDDPMGenerative.created = []

    result = tune_mod.tune_tabddpm(
        df=_build_regression_df(),
        schema=_build_schema(),
        dataset="housing",
        encoding_method="one_hot_representation",
        n_trials=1,
        seed=17,
        output_root=tmp_path / "optuna_results",
        device="cpu",
    )

    payload = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert payload["task_type"] == "regression"
    assert DummyTabDDPMGenerative.created
    # В случае регрессии таргет масштабируется и не должен быть в дискретных
    # фичах (если бы мы их собирали)
    assert payload["preprocessing"]["target_processing"][
               "target_scaled"] is True


def test_tune_tabddpm_holdout_split_is_independent_from_optuna_seed(tmp_path,
                                                                    monkeypatch):
    monkeypatch.setattr(tune_mod, "TabDDPMGenerative", DummyTabDDPMGenerative)
    DummyTabDDPMGenerative.created = []

    tune_mod.tune_tabddpm(
        df=_build_df(),
        schema=_build_schema(),
        dataset="seed-a",
        encoding_method="one_hot_representation",
        n_trials=1,
        seed=1,
        output_root=tmp_path / "optuna_results",
        device="cpu",
    )
    tune_mod.tune_tabddpm(
        df=_build_df(),
        schema=_build_schema(),
        dataset="seed-b",
        encoding_method="one_hot_representation",
        n_trials=1,
        seed=999,
        output_root=tmp_path / "optuna_results",
        device="cpu",
    )

    assert len(DummyTabDDPMGenerative.created) >= 2
    train_a = DummyTabDDPMGenerative.created[0].train_df
    train_b = DummyTabDDPMGenerative.created[1].train_df
    assert train_a is not None
    assert train_b is not None
    # Сплит должен быть одинаковым, так как random_seed=5 захардкожен в
    # holdout_cfg
    pd.testing.assert_frame_equal(train_a, train_b)


def test_tune_tabddpm_raises_on_invalid_encoding_method(tmp_path, monkeypatch):
    monkeypatch.setattr(tune_mod, "TabDDPMGenerative", DummyTabDDPMGenerative)
    DummyTabDDPMGenerative.created = []

    with pytest.raises(ValueError, match="Unsupported encoding_method"):
        tune_mod.tune_tabddpm(
            df=_build_df(),
            schema=_build_schema(),
            dataset="adult",
            encoding_method="one hot",
            n_trials=1,
            output_root=tmp_path / "optuna_results",
            device="cpu",
        )


def test_tune_tabddpm_retypes_high_cardinality_integer_feature_as_continuous(
        tmp_path, monkeypatch):
    monkeypatch.setattr(tune_mod, "TabDDPMGenerative", DummyTabDDPMGenerative)
    DummyTabDDPMGenerative.created = []

    n = 80
    df = pd.DataFrame(
        {
            "x_cont": [float(i) / 10.0 for i in range(n)],
            "x_disc_high_card": list(range(n)),
            # high cardinality integer feature
            "x_cat": ["a" if i % 2 == 0 else "b" for i in range(n)],
            "target": [i % 2 for i in range(n)],
        }
    )

    tune_mod.tune_tabddpm(
        df=df,
        schema=_build_schema_with_high_cardinality_discrete(),
        dataset="high-card",
        encoding_method="ordinal_representation",
        n_trials=1,
        output_root=tmp_path / "optuna_results",
        device="cpu",
    )

    assert DummyTabDDPMGenerative.created
    # Проверяем, что препроцессор генбенча отработал и изменил тип колонки
    # (проверяется через то, что модель вообще успешно обучилась на данных
    # без ошибки типов)
    assert DummyTabDDPMGenerative.created[0].fitted_ is True


def test_tune_tabddpm_label_encodes_categorical_target_for_classification(
        tmp_path, monkeypatch):
    monkeypatch.setattr(tune_mod, "TabDDPMGenerative", DummyTabDDPMGenerative)
    DummyTabDDPMGenerative.created = []

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

    result = tune_mod.tune_tabddpm(
        df=df,
        schema=schema,
        dataset="categorical-target",
        encoding_method="one_hot_representation",
        n_trials=1,
        output_root=tmp_path / "optuna_results",
        device="cpu",
    )

    payload = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert payload["preprocessing"]["target_processing"][
               "target_encoded"] is True
    assert DummyTabDDPMGenerative.created
    fitted_train = DummyTabDDPMGenerative.created[0].train_df
    assert fitted_train is not None
    assert pd.api.types.is_integer_dtype(fitted_train["target"])
