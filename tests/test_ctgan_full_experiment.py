from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pandas as pd

import experiments.ctgan.ctgan_full_experiment as full_mod
from genbench.data.schema import TabularSchema


def progress_stages(progress_stream: StringIO) -> list[str]:
    return [json.loads(line)["stage"] for line in progress_stream.getvalue().splitlines() if line.strip()]


def first_occurrence_order(stages: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for stage in stages:
        if stage in seen:
            continue
        seen.add(stage)
        ordered.append(stage)
    return ordered


def write_runner_manifest(
    tmp_path: Path,
    *,
    dataset_id: str = "openml_adult",
    label: str = "adult",
    target_col: str = "target",
    encoding_label: str = "one-hot",
    encoding_id: str = "one_hot_representation",
) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "datasets": [
                    {"label": label, "dataset_id": dataset_id, "target_col": target_col, "id_col": None}
                ],
                "encodings": [
                    {"label": encoding_label, "encoding_id": encoding_id}
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def write_runner_csv(
    tmp_path: Path,
    *,
    dataset_id: str = "openml_adult",
    target_col: str = "target",
    n_rows: int = 10,
) -> None:
    data_dir = tmp_path / "datasets" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "x_cont": [0.1 * i for i in range(n_rows)],
            "x_disc": [i % 2 for i in range(n_rows)],
            "x_cat": ["a" if i % 2 == 0 else "b" for i in range(n_rows)],
            target_col: [0.15 * i + 0.03 for i in range(n_rows)],
        }
    ).to_csv(data_dir / f"{dataset_id}.csv", index=False)


def write_best_params_file(tmp_path: Path) -> Path:
    path = tmp_path / "best_params.json"
    path.write_text(
        json.dumps(
            {
                "best_params": {
                    "embedding_dim": 256,
                    "gen_dim": 512,
                    "disc_dim": 256,
                    "batch_size": 1024,
                    "discriminator_steps": 3,
                    "generator_lr": 5e-4,
                    "lr_ratio": 1.2,
                }
            }
        ),
        encoding="utf-8",
    )
    return path


class DummyCtganGenerative:
    created: list["DummyCtganGenerative"] = []

    def __init__(self, *args, **kwargs):
        self._train_df: pd.DataFrame | None = None
        self.ctgan_kwargs = dict(kwargs.get("ctgan_kwargs", {}))
        DummyCtganGenerative.created.append(self)

    def fit(self, train_df: pd.DataFrame, *args, **kwargs) -> "DummyCtganGenerative":
        self._train_df = train_df.copy()
        return self

    def sample(self, n: int):
        if self._train_df is None:
            raise AssertionError("fit() must be called before sample()")
        return self._train_df.head(n).reset_index(drop=True)


def fake_select_ctgan_best_params(*, output_dir, **kwargs):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps({"best_value": 0.123}), encoding="utf-8")
    (output_dir / "trials.csv").write_text("number,value\n0,0.123\n", encoding="utf-8")
    (output_dir / "best_params.json").write_text(
        json.dumps(
            {
                "embedding_dim": 128,
                "gen_dim": 256,
                "disc_dim": 256,
                "batch_size": 64,
                "discriminator_steps": 1,
                "generator_lr": 1e-3,
                "lr_ratio": 1.0,
            }
        ),
        encoding="utf-8",
    )
    return {
        "best_params": {
            "embedding_dim": 128,
            "gen_dim": 256,
            "disc_dim": 256,
            "batch_size": 64,
            "discriminator_steps": 1,
            "generator_lr": 1e-3,
            "lr_ratio": 1.0,
        },
        "best_value": 0.123,
        "best_source": "stage1",
    }


def fake_select_ctgan_best_params_with_progress(*, output_dir, progress_callback, **kwargs):
    progress_callback(
        "trial 1/30 | Gen. (-1.39) | Discrim. (-0.58):   5%|▌         | 15/300 [00:54<17:02,  3.59s/it] | tuning eta 1h 26m"
    )
    return fake_select_ctgan_best_params(output_dir=output_dir, **kwargs)


def make_task_type_capturing_select(captured: dict[str, object]):
    def _fake_select(*, output_dir, **kwargs):
        captured["task_type"] = kwargs.get("task_type")
        return fake_select_ctgan_best_params(output_dir=output_dir, **kwargs)

    return _fake_select


def fake_tstr(*args, **kwargs):
    task_type = kwargs.get("task_type", "regression")
    if task_type == "classification":
        return {
            "task_type": "classification",
            "f1_weighted_real": 0.91,
            "f1_weighted_synth": 0.87,
            "f1_weighted_pct_diff": 4.4,
        }
    return {
        "task_type": "regression",
        "r2_real": 0.91,
        "r2_synth": 0.87,
        "r2_pct_diff": 4.4,
    }


def fake_manifest_without_target(*args, **kwargs):
    class _Manifest:
        encodings = [
            type("EncodingEntry", (), {"label": "one-hot", "encoding_id": "one_hot_representation"})()
        ]

        def resolve_dataset_label(self, label):
            return type(
                "DatasetEntry",
                (),
                {"label": label, "dataset_id": "openml_no_target", "target_col": None, "id_col": None},
            )()

        def resolve_encoding_label(self, label):
            return self.encodings[0]

    return _Manifest()


def fake_run_full_experiment(
    *,
    output_root,
    dataset_id,
    encoding_method,
    progress_stream=None,
    **kwargs,
):
    output_dir = Path(output_root) / "ctgan" / dataset_id / encoding_method
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({"event": "progress", "stage": "tuning", "message": "started"}),
        json.dumps({"event": "progress", "stage": "saving", "message": "done"}),
    ]
    for line in lines:
        if progress_stream is None:
            print(line)
        else:
            progress_stream.write(line + "\n")
            progress_stream.flush()
    return SimpleNamespace(output_dir=output_dir)


def fake_run_runtime_estimate(
    *,
    output_root,
    dataset_id,
    encoding_method,
    progress_stream=None,
    **kwargs,
):
    output_dir = Path(output_root) / "ctgan" / dataset_id / encoding_method / "runtime_estimate"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps({"mode": "runtime_estimate"}), encoding="utf-8")
    lines = [
        json.dumps({"event": "progress", "stage": "tuning", "message": "estimating runtime"}),
        json.dumps({"event": "progress", "stage": "saving", "message": "writing estimate summary"}),
    ]
    for line in lines:
        if progress_stream is None:
            print(line)
        else:
            progress_stream.write(line + "\n")
            progress_stream.flush()
    return SimpleNamespace(output_dir=output_dir, summary_path=summary_path)


def test_compute_distribution_scores_adds_unencoded_original_numeric_metrics():
    test_df = pd.DataFrame(
        {
            "x_cont": [0.0, 1.0, 2.0, 3.0],
            "x_disc": [0, 0, 1, 1],
            "x_cat_a": [1, 0, 1, 0],
            "x_cat_b": [0, 1, 0, 1],
        }
    )
    synth_df = pd.DataFrame(
        {
            "x_cont": [0.0, 1.0, 2.0, 3.0],
            "x_disc": [1, 1, 0, 0],
            "x_cat_a": [0, 0, 0, 0],
            "x_cat_b": [1, 1, 1, 1],
        }
    )
    transformed_schema = TabularSchema(
        continuous_cols=["x_cont", "x_cat_a", "x_cat_b"],
        discrete_cols=["x_disc"],
        categorical_cols=[],
    )
    original_schema = TabularSchema(
        continuous_cols=["x_cont"],
        discrete_cols=["x_disc"],
        categorical_cols=["x_cat"],
    )

    scores = full_mod._compute_distribution_scores(
        test_df=test_df,
        synth_df=synth_df,
        transformed_schema=transformed_schema,
        original_schema=original_schema,
    )

    assert "wasserstein_mean" in scores
    assert "marginal_kl_mean" in scores
    assert "wasserstein_mean_unencoded" in scores
    assert "marginal_kl_mean_unencoded" in scores
    assert "corr_frobenius_unencoded" in scores
    assert scores["wasserstein_mean_unencoded"] == 0.0


def test_run_full_experiment_writes_expected_artifacts(tmp_path, monkeypatch):
    manifest_path = write_runner_manifest(tmp_path)
    write_runner_csv(tmp_path)
    DummyCtganGenerative.created = []
    monkeypatch.setattr(full_mod, "CtganGenerative", DummyCtganGenerative)
    monkeypatch.setattr(full_mod, "select_ctgan_best_params", fake_select_ctgan_best_params)
    monkeypatch.setattr(full_mod, "tstr_catboost", fake_tstr)

    result = full_mod.run_full_ctgan_experiment(
        manifest_path=manifest_path,
        dataset_id="openml_adult",
        dataset_label="adult",
        encoding_method="one_hot_representation",
        output_root=tmp_path / "results",
        device="cpu",
    )

    assert result.output_dir == tmp_path / "results" / "ctgan" / "openml_adult" / "one_hot_representation"
    assert (result.output_dir / "run_summary.json").exists()
    assert (result.output_dir / "tuning" / "summary.json").exists()
    assert (result.output_dir / "metrics" / "aggregate.json").exists()
    assert (result.output_dir / "crossval" / "per_fold" / "fold_0.json").exists()
    fold_payload = json.loads((result.output_dir / "crossval" / "per_fold" / "fold_0.json").read_text(encoding="utf-8"))
    assert "wasserstein_mean" in fold_payload["distribution"]
    assert "marginal_kl_mean" in fold_payload["distribution"]
    assert "wasserstein_mean_unencoded" in fold_payload["distribution"]
    assert "marginal_kl_mean_unencoded" in fold_payload["distribution"]
    assert "corr_frobenius_unencoded" in fold_payload["distribution"]
    assert "corr_frobenius_transformed" in fold_payload["distribution"]
    assert "corr_frobenius_original" in fold_payload["distribution"]
    assert fold_payload["distribution"]["corr_frobenius_original_status"] == "ok"
    assert fold_payload["utility"]["task_type"] == "regression"
    assert "r2_real" in fold_payload["utility"]
    aggregate_payload = json.loads((result.output_dir / "metrics" / "aggregate.json").read_text(encoding="utf-8"))
    assert set(aggregate_payload["distribution"]) == {
        "wasserstein_mean",
        "marginal_kl_mean",
        "wasserstein_mean_unencoded",
        "marginal_kl_mean_unencoded",
        "corr_frobenius_unencoded",
        "corr_frobenius_transformed",
        "corr_frobenius_original",
    }
    assert set(aggregate_payload["tstr"]["metrics"]) == {"r2_real", "r2_synth", "r2_pct_diff"}
    assert DummyCtganGenerative.created
    assert DummyCtganGenerative.created[0].ctgan_kwargs["epochs"] == 300


def test_run_full_experiment_passes_explicit_task_type_to_tuning(tmp_path, monkeypatch):
    manifest_path = write_runner_manifest(tmp_path)
    write_runner_csv(tmp_path)
    captured: dict[str, object] = {}
    monkeypatch.setattr(full_mod, "CtganGenerative", DummyCtganGenerative)
    monkeypatch.setattr(full_mod, "select_ctgan_best_params", make_task_type_capturing_select(captured))
    monkeypatch.setattr(full_mod, "tstr_catboost", fake_tstr)

    full_mod.run_full_ctgan_experiment(
        manifest_path=manifest_path,
        dataset_id="openml_adult",
        dataset_label="adult",
        encoding_method="one_hot_representation",
        output_root=tmp_path / "results",
        device="cpu",
    )

    assert captured["task_type"] == "regression"


def test_run_full_experiment_uses_weighted_f1_for_classification(tmp_path, monkeypatch):
    manifest_path = write_runner_manifest(tmp_path)
    write_runner_csv(tmp_path)
    monkeypatch.setattr(full_mod, "CtganGenerative", DummyCtganGenerative)
    monkeypatch.setattr(full_mod, "select_ctgan_best_params", fake_select_ctgan_best_params)
    monkeypatch.setattr(full_mod, "tstr_catboost", fake_tstr)
    monkeypatch.setattr(full_mod, "infer_is_regression_target", lambda *args, **kwargs: False)
    progress_stream = StringIO()

    result = full_mod.run_full_ctgan_experiment(
        manifest_path=manifest_path,
        dataset_id="openml_adult",
        dataset_label="adult",
        encoding_method="one_hot_representation",
        output_root=tmp_path / "results",
        progress_stream=progress_stream,
        device="cpu",
    )

    payload = json.loads((result.output_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert payload["tstr"]["task_type"] == "classification"
    assert "f1_weighted_real" in payload["tstr"]["metrics"]
    assert "f1_weighted_synth" in payload["tstr"]["metrics"]
    assert first_occurrence_order(progress_stages(progress_stream)) == [
        "launching",
        "tuning",
        "crossval",
        "metrics",
        "saving",
    ]


def test_run_full_experiment_forwards_tuning_progress_messages(tmp_path, monkeypatch):
    manifest_path = write_runner_manifest(tmp_path)
    write_runner_csv(tmp_path)
    monkeypatch.setattr(full_mod, "CtganGenerative", DummyCtganGenerative)
    monkeypatch.setattr(full_mod, "select_ctgan_best_params", fake_select_ctgan_best_params_with_progress)
    monkeypatch.setattr(full_mod, "tstr_catboost", fake_tstr)
    progress_stream = StringIO()

    full_mod.run_full_ctgan_experiment(
        manifest_path=manifest_path,
        dataset_id="openml_adult",
        dataset_label="adult",
        encoding_method="one_hot_representation",
        output_root=tmp_path / "results",
        progress_stream=progress_stream,
        device="cpu",
    )

    tuning_messages = [
        json.loads(line)["message"]
        for line in progress_stream.getvalue().splitlines()
        if line.strip() and json.loads(line)["stage"] == "tuning"
    ]

    assert "tuning ctgan" in tuning_messages
    assert any("trial 1/30" in message for message in tuning_messages)
    assert any("15/300" in message for message in tuning_messages)
    assert any("tuning eta" in message for message in tuning_messages)
    assert first_occurrence_order(progress_stages(progress_stream)) == [
        "launching",
        "tuning",
        "crossval",
        "metrics",
        "saving",
    ]


def test_run_full_experiment_marks_tstr_unsupported_without_target(tmp_path, monkeypatch):
    manifest_path = write_runner_manifest(tmp_path, dataset_id="openml_no_target")
    write_runner_csv(tmp_path, dataset_id="openml_no_target", target_col="value")
    monkeypatch.setattr(full_mod, "CtganGenerative", DummyCtganGenerative)
    monkeypatch.setattr(full_mod, "select_ctgan_best_params", fake_select_ctgan_best_params)
    monkeypatch.setattr(full_mod, "tstr_catboost", fake_tstr)
    monkeypatch.setattr(full_mod, "load_ctgan_manifest", fake_manifest_without_target)
    progress_stream = StringIO()

    # Defensive branch only: the v1 manifest contract still requires target_col.
    result = full_mod.run_full_ctgan_experiment(
        manifest_path=manifest_path,
        dataset_id="openml_no_target",
        dataset_label="adult",
        encoding_method="one_hot_representation",
        output_root=tmp_path / "results",
        progress_stream=progress_stream,
        device="cpu",
    )

    payload = json.loads((result.output_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert payload["tstr"]["status"] == "unsupported_no_target"
    assert "skipping tstr utility: no target column in schema" in progress_stream.getvalue()
    assert first_occurrence_order(progress_stages(progress_stream)) == [
        "launching",
        "tuning",
        "crossval",
        "metrics",
        "saving",
    ]


def test_run_full_experiment_validates_encoding_method_against_manifest(tmp_path, monkeypatch):
    manifest_path = write_runner_manifest(tmp_path)
    write_runner_csv(tmp_path)
    monkeypatch.setattr(full_mod, "CtganGenerative", DummyCtganGenerative)
    monkeypatch.setattr(full_mod, "select_ctgan_best_params", fake_select_ctgan_best_params)
    monkeypatch.setattr(full_mod, "tstr_catboost", fake_tstr)

    try:
        full_mod.run_full_ctgan_experiment(
            manifest_path=manifest_path,
            dataset_id="openml_adult",
            dataset_label="adult",
            encoding_method="ordinal_representation",
            output_root=tmp_path / "results",
            device="cpu",
        )
    except ValueError as exc:
        assert "encoding_method mismatch" in str(exc)
    else:
        raise AssertionError("Expected run_full_ctgan_experiment() to validate encoding_method against the manifest")


def test_run_full_experiment_marks_original_corr_unsupported_for_non_invertible_representation(tmp_path, monkeypatch):
    manifest_path = write_runner_manifest(
        tmp_path,
        encoding_label="hash",
        encoding_id="hash_representation",
    )
    write_runner_csv(tmp_path)
    monkeypatch.setattr(full_mod, "CtganGenerative", DummyCtganGenerative)
    monkeypatch.setattr(full_mod, "select_ctgan_best_params", fake_select_ctgan_best_params)
    monkeypatch.setattr(full_mod, "tstr_catboost", fake_tstr)

    result = full_mod.run_full_ctgan_experiment(
        manifest_path=manifest_path,
        dataset_id="openml_adult",
        dataset_label="adult",
        encoding_method="hash_representation",
        output_root=tmp_path / "results",
        device="cpu",
    )

    fold_payload = json.loads((result.output_dir / "crossval" / "per_fold" / "fold_0.json").read_text(encoding="utf-8"))
    assert fold_payload["distribution"]["corr_frobenius_original"] is None
    assert fold_payload["distribution"]["corr_frobenius_original_status"] == "unsupported_not_invertible"


def test_infer_project_root_finds_repo_root_for_nested_manifest_path(tmp_path):
    project_root = tmp_path / "repo"
    manifest_path = project_root / "experiments" / "ctgan" / "orchestrator_staff" / "ctgan_orchestrator_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    (project_root / "genbench").mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}", encoding="utf-8")

    resolved = full_mod._infer_project_root(manifest_path)

    assert resolved == project_root


def test_run_full_experiment_uses_best_params_file_and_skips_tuning(tmp_path, monkeypatch):
    manifest_path = write_runner_manifest(tmp_path)
    write_runner_csv(tmp_path)
    best_params_file = write_best_params_file(tmp_path)
    DummyCtganGenerative.created = []
    monkeypatch.setattr(full_mod, "CtganGenerative", DummyCtganGenerative)
    monkeypatch.setattr(full_mod, "tstr_catboost", fake_tstr)
    progress_stream = StringIO()

    def fail_select(**kwargs):
        raise AssertionError("tuning should not run when best_params_file is provided")

    monkeypatch.setattr(full_mod, "select_ctgan_best_params", fail_select)

    result = full_mod.run_full_ctgan_experiment(
        manifest_path=manifest_path,
        dataset_id="openml_adult",
        dataset_label="adult",
        encoding_method="one_hot_representation",
        output_root=tmp_path / "results",
        best_params_file=best_params_file,
        skip_tuning=True,
        progress_stream=progress_stream,
        device="cpu",
    )

    payload = json.loads((result.output_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert payload["tuning"]["best_source"] == "provided_file"
    assert payload["tuning"]["best_params"]["batch_size"] == 1024
    assert payload["tuning"]["best_params_file"] == str(best_params_file)
    assert DummyCtganGenerative.created
    assert DummyCtganGenerative.created[0].ctgan_kwargs["batch_size"] == 1024
    assert first_occurrence_order(progress_stages(progress_stream)) == [
        "launching",
        "crossval",
        "metrics",
        "saving",
    ]


def test_run_full_experiment_poster_fast_uses_single_holdout_and_caps_rows(tmp_path, monkeypatch):
    manifest_path = write_runner_manifest(tmp_path)
    write_runner_csv(tmp_path, n_rows=20)
    best_params_file = write_best_params_file(tmp_path)
    DummyCtganGenerative.created = []
    monkeypatch.setattr(full_mod, "CtganGenerative", DummyCtganGenerative)
    monkeypatch.setattr(full_mod, "tstr_catboost", fake_tstr)
    monkeypatch.setattr(full_mod, "select_ctgan_best_params", fake_select_ctgan_best_params)

    result = full_mod.run_full_ctgan_experiment(
        manifest_path=manifest_path,
        dataset_id="openml_adult",
        dataset_label="adult",
        encoding_method="one_hot_representation",
        output_root=tmp_path / "results",
        best_params_file=best_params_file,
        skip_tuning=True,
        poster_fast=True,
        max_rows=5,
        device="cpu",
    )

    payload = json.loads((result.output_dir / "run_summary.json").read_text(encoding="utf-8"))
    aggregate_payload = json.loads((result.output_dir / "metrics" / "aggregate.json").read_text(encoding="utf-8"))
    fold_payload = json.loads((result.output_dir / "crossval" / "per_fold" / "fold_0.json").read_text(encoding="utf-8"))
    assert payload["poster_fast"]["enabled"] is True
    assert payload["poster_fast"]["max_rows"] == 5
    assert payload["poster_fast"]["effective_rows"] == 5
    assert payload["crossval"]["n_folds"] == 1
    assert aggregate_payload["n_folds"] == 1
    assert fold_payload["n_train"] + fold_payload["n_test"] == 5
    assert aggregate_payload["distribution"]["wasserstein_mean"]["std"] == 0.0
    assert aggregate_payload["distribution"]["marginal_kl_mean"]["std"] == 0.0
    assert aggregate_payload["distribution"]["wasserstein_mean_unencoded"]["std"] == 0.0
    assert aggregate_payload["distribution"]["marginal_kl_mean_unencoded"]["std"] == 0.0
    assert aggregate_payload["distribution"]["corr_frobenius_unencoded"]["std"] == 0.0
    assert aggregate_payload["distribution"]["corr_frobenius_transformed"]["std"] == 0.0
    assert aggregate_payload["distribution"]["corr_frobenius_original"]["std"] == 0.0


def test_run_full_experiment_rejects_skip_tuning_without_best_params_file(tmp_path):
    manifest_path = write_runner_manifest(tmp_path)
    write_runner_csv(tmp_path)

    try:
        full_mod.run_full_ctgan_experiment(
            manifest_path=manifest_path,
            dataset_id="openml_adult",
            dataset_label="adult",
            encoding_method="one_hot_representation",
            output_root=tmp_path / "results",
            skip_tuning=True,
            device="cpu",
        )
    except ValueError as exc:
        assert "best_params_file" in str(exc)
    else:
        raise AssertionError("Expected skip_tuning without best_params_file to raise ValueError")


def test_cli_emits_progress_jsonl(capsys, monkeypatch, tmp_path):
    manifest_path = write_runner_manifest(tmp_path)
    monkeypatch.setattr(full_mod, "run_full_ctgan_experiment", fake_run_full_experiment)

    exit_code = full_mod.main(
        [
            "--manifest",
            str(manifest_path),
            "--dataset-id",
            "openml_adult",
            "--dataset-label",
            "adult",
            "--encoding-method",
            "one_hot_representation",
            "--output-root",
            str(tmp_path / "results"),
            "--progress-format",
            "jsonl",
            "--device",
            "cpu",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert '"stage": "tuning"' in captured.out
    assert '"stage": "saving"' in captured.out


def test_cli_passes_best_params_file_and_skip_tuning(monkeypatch, tmp_path):
    manifest_path = write_runner_manifest(tmp_path)
    best_params_file = write_best_params_file(tmp_path)
    captured: dict[str, object] = {}

    def fake_runner(**kwargs):
        captured.update(kwargs)
        return fake_run_full_experiment(**kwargs)

    monkeypatch.setattr(full_mod, "run_full_ctgan_experiment", fake_runner)

    exit_code = full_mod.main(
        [
            "--manifest",
            str(manifest_path),
            "--dataset-id",
            "openml_adult",
            "--dataset-label",
            "adult",
            "--encoding-method",
            "one_hot_representation",
            "--output-root",
            str(tmp_path / "results"),
            "--best-params-file",
            str(best_params_file),
            "--skip-tuning",
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 0
    assert captured["best_params_file"] == best_params_file
    assert captured["skip_tuning"] is True


def test_cli_passes_poster_fast_and_max_rows(monkeypatch, tmp_path):
    manifest_path = write_runner_manifest(tmp_path)
    captured: dict[str, object] = {}

    def fake_runner(**kwargs):
        captured.update(kwargs)
        return fake_run_full_experiment(**kwargs)

    monkeypatch.setattr(full_mod, "run_full_ctgan_experiment", fake_runner)

    exit_code = full_mod.main(
        [
            "--manifest",
            str(manifest_path),
            "--dataset-id",
            "openml_adult",
            "--dataset-label",
            "adult",
            "--encoding-method",
            "one_hot_representation",
            "--output-root",
            str(tmp_path / "results"),
            "--poster-fast",
            "--max-rows",
            "5",
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 0
    assert captured["poster_fast"] is True
    assert captured["max_rows"] == 5


def test_cli_routes_to_runtime_estimate_mode(monkeypatch, tmp_path):
    manifest_path = write_runner_manifest(tmp_path)
    captured: dict[str, object] = {}

    def fake_runner(**kwargs):
        captured.update(kwargs)
        return fake_run_runtime_estimate(**kwargs)

    monkeypatch.setattr(full_mod, "run_ctgan_runtime_estimate", fake_runner)

    exit_code = full_mod.main(
        [
            "--manifest",
            str(manifest_path),
            "--dataset-id",
            "openml_adult",
            "--dataset-label",
            "adult",
            "--encoding-method",
            "one_hot_representation",
            "--output-root",
            str(tmp_path / "results"),
            "--estimate-runtime",
            "--estimate-sample-epochs",
            "12",
            "--estimate-total-runs",
            "35",
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 0
    assert captured["estimate_sample_epochs"] == 12
    assert captured["estimate_total_runs"] == 35
    assert captured["device"] == "cpu"


def test_full_experiment_cli_help_runs_as_script():
    project_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "experiments/ctgan/ctgan_full_experiment.py", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "usage:" in completed.stdout
