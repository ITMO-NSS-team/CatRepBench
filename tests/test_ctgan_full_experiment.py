from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pandas as pd

import experiments.ctgan_full_experiment as full_mod


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
) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "datasets": [
                    {"label": label, "dataset_id": dataset_id, "target_col": target_col, "id_col": None}
                ],
                "encodings": [
                    {"label": "one-hot", "encoding_id": "one_hot_representation"}
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
) -> None:
    data_dir = tmp_path / "datasets" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "x_cont": [0.1 * i for i in range(10)],
            "x_disc": [i % 2 for i in range(10)],
            "x_cat": ["a" if i % 2 == 0 else "b" for i in range(10)],
            target_col: [float(i) for i in range(10)],
        }
    ).to_csv(data_dir / f"{dataset_id}.csv", index=False)


class DummyCtganGenerative:
    def __init__(self, *args, **kwargs):
        self._train_df: pd.DataFrame | None = None

    def fit(self, train_df: pd.DataFrame, *args, **kwargs) -> "DummyCtganGenerative":
        self._train_df = train_df.copy()
        return self

    def sample(self, n: int):
        if self._train_df is None:
            raise AssertionError("fit() must be called before sample()")
        return self._train_df.head(n).reset_index(drop=True)


def fake_tune_ctgan(*, output_dir, **kwargs):
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
    return SimpleNamespace(
        output_dir=output_dir,
        best_params={
            "embedding_dim": 128,
            "gen_dim": 256,
            "disc_dim": 256,
            "batch_size": 64,
            "discriminator_steps": 1,
            "generator_lr": 1e-3,
            "lr_ratio": 1.0,
        },
    )


def fake_tstr(*args, **kwargs):
    return {
        "status": "ok",
        "r2_real": 0.91,
        "r2_synth": 0.87,
        "rmse_real": 0.14,
        "rmse_synth": 0.19,
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


def test_run_full_experiment_writes_expected_artifacts(tmp_path, monkeypatch):
    manifest_path = write_runner_manifest(tmp_path)
    write_runner_csv(tmp_path)
    monkeypatch.setattr(full_mod, "CtganGenerative", DummyCtganGenerative)
    monkeypatch.setattr(full_mod, "tune_ctgan", fake_tune_ctgan)
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


def test_run_full_experiment_marks_tstr_unsupported_for_classification(tmp_path, monkeypatch):
    manifest_path = write_runner_manifest(tmp_path)
    write_runner_csv(tmp_path)
    monkeypatch.setattr(full_mod, "CtganGenerative", DummyCtganGenerative)
    monkeypatch.setattr(full_mod, "tune_ctgan", fake_tune_ctgan)
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
    assert payload["tstr"]["status"] == "unsupported_classification"
    assert "skipping tstr utility: classification target unsupported" in progress_stream.getvalue()
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
    monkeypatch.setattr(full_mod, "tune_ctgan", fake_tune_ctgan)
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
    monkeypatch.setattr(full_mod, "tune_ctgan", fake_tune_ctgan)
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


def test_full_experiment_cli_help_runs_as_script():
    project_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "experiments/ctgan_full_experiment.py", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "usage:" in completed.stdout
