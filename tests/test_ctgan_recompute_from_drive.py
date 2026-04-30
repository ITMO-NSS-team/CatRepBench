from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import experiments.ctgan.ctgan_full_experiment as full_mod
import experiments.ctgan.ctgan_recompute_from_drive as recompute_mod
from experiments.ctgan.orchestrator_staff.ctgan_drive import DriveFileRecord
from genbench.data.schema import TabularSchema


class SampleOnlyModel:
    def __init__(self, sample_df: pd.DataFrame):
        self.sample_df = sample_df.reset_index(drop=True)
        self.device = None

    def sample(self, n: int) -> pd.DataFrame:
        repeats = (n // len(self.sample_df)) + 1
        return pd.concat([self.sample_df] * repeats, ignore_index=True).head(n).copy()

    def set_device(self, device: str) -> None:
        self.device = device


class FakeDriveClient:
    def __init__(self, files: dict[str, bytes]):
        self.files = files
        self.downloads: list[str] = []

    def find_folder_path(self, *path_parts: str, root_id: str | None = None):
        if path_parts == ("CTGAN", "openml_adult", "one_hot_representation"):
            return "drive-folder-id"
        return None

    def list_files_recursive(self, folder_id: str):
        assert folder_id == "drive-folder-id"
        return [
            DriveFileRecord(
                file_id="summary",
                name="run_summary.json",
                mime_type="application/json",
                modified_time="2026-04-24T09:00:00Z",
                relative_path="run_summary.json",
                parent_id=folder_id,
            ),
            DriveFileRecord(
                file_id="broken-new-model",
                name="ctgan.pkl",
                mime_type="application/octet-stream",
                modified_time="2026-04-24T10:00:00Z",
                relative_path="artifacts/fold_0/ctgan.pkl",
                parent_id="fold-folder",
            ),
            DriveFileRecord(
                file_id="valid-old-model",
                name="ctgan.pkl",
                mime_type="application/octet-stream",
                modified_time="2026-04-23T10:00:00Z",
                relative_path="artifacts/fold_0/ctgan.pkl",
                parent_id="fold-folder",
            ),
        ]

    def download_file(self, file_id: str, destination: Path) -> None:
        self.downloads.append(file_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.files[file_id])

    def folder_web_url(self, folder_id: str) -> str:
        return f"https://drive.google.com/drive/folders/{folder_id}"

    def ensure_folder_path(self, *path_parts: str, root_id: str | None = None) -> str:
        return f"fake-folder-{'_'.join(path_parts)}"

    def upload_file(self, local_path: Path, parent_id: str, *, overwrite: bool = True) -> str:
        return f"fake-file-id-{local_path.name}"


def _write_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "label": "adult",
                        "dataset_id": "openml_adult",
                        "target_col": "target",
                        "id_col": None,
                    }
                ],
                "encodings": [
                    {
                        "label": "one-hot",
                        "encoding_id": "one_hot_representation",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_csv(tmp_path: Path) -> None:
    data_dir = tmp_path / "datasets" / "raw"
    data_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "x_cont": [0.1 * i for i in range(12)],
            "x_disc": [i % 2 for i in range(12)],
            "x_cat": ["a" if i % 2 == 0 else "b" for i in range(12)],
            "target": [i % 2 for i in range(12)],
        }
    ).to_csv(data_dir / "openml_adult.csv", index=False)


def _ctgan_pickle(sample_df: pd.DataFrame) -> bytes:
    return pickle.dumps(
        {
            "model": SampleOnlyModel(sample_df),
            "used_discrete_cols": [],
            "fitted": True,
        }
    )


def test_local_drive_records_are_grouped_newest_first():
    records = [
        recompute_mod.DriveFileRecord(
            file_id="old",
            name="ctgan.pkl",
            mime_type="application/octet-stream",
            modified_time="2026-04-23T10:00:00Z",
            relative_path="artifacts/fold_0/ctgan.pkl",
            parent_id="fold-folder",
        ),
        recompute_mod.DriveFileRecord(
            file_id="new",
            name="ctgan.pkl",
            mime_type="application/octet-stream",
            modified_time="2026-04-24T10:00:00Z",
            relative_path="artifacts/fold_0/ctgan.pkl",
            parent_id="fold-folder",
        ),
    ]

    grouped = recompute_mod.drive_records_by_relative_path(records)

    assert [record.file_id for record in grouped["artifacts/fold_0/ctgan.pkl"]] == ["new", "old"]


def test_local_results_worksheet_refresh_archives_and_writes_full_schema():
    class FakeWorksheet:
        def __init__(self, title):
            self.title = title
            self.updated_title = None
            self.update_calls = []
            self.freeze_calls = []

        def update_title(self, title):
            self.updated_title = title
            self.title = title

        def update(self, *args, **kwargs):
            self.update_calls.append((args, kwargs))

        def freeze(self, rows=0):
            self.freeze_calls.append(rows)

    class FakeSpreadsheet:
        def __init__(self):
            self.old = FakeWorksheet("Results")
            self.created = []

        def worksheet(self, title):
            if title == "Results":
                return self.old
            raise RuntimeError(f"unknown worksheet: {title}")

        def add_worksheet(self, title, rows, cols):
            worksheet = FakeWorksheet(title)
            self.created.append((title, rows, cols, worksheet))
            return worksheet

    row = recompute_mod.build_results_sheet_row(
        model_name="CTGAN",
        dataset_label="adult",
        dataset_id="openml_adult",
        encoding_label="one-hot",
        encoding_id="one_hot_representation",
        aggregate={
            "n_folds": 1,
            "distribution": {
                "wasserstein_mean": {"mean": 1.0, "std": 0.0},
                "wasserstein_mean_unencoded": {"mean": 2.0, "std": 0.0},
            },
            "distribution_status": {"corr_frobenius_original_status": "ok"},
            "tstr": {
                "status": "ok",
                "task_type": "classification",
                "metrics": {"f1_weighted_pct_diff": {"mean": 25.0, "std": 0.0}},
            },
        },
        folder_url="https://drive.google.test/folder",
    )
    spreadsheet = FakeSpreadsheet()

    recompute_mod.refresh_full_results_worksheet(
        spreadsheet=spreadsheet,
        worksheet_name="Results",
        rows=[row],
        archive_existing=True,
        now=datetime(2026, 4, 24, 9, 44, 0, tzinfo=timezone.utc),
    )

    assert spreadsheet.old.updated_title == "Results_legacy_20260424_094400"
    assert spreadsheet.created[0][:3] == ("Results", 1000, len(recompute_mod.FULL_RESULTS_HEADERS))
    assert "n_folds" not in recompute_mod.FULL_RESULTS_HEADERS
    new_worksheet = spreadsheet.created[0][3]
    assert new_worksheet.freeze_calls == [1]
    args, kwargs = new_worksheet.update_calls[0]
    written_values = kwargs.get("values") or args[0]
    assert written_values[0] == recompute_mod.FULL_RESULTS_HEADERS
    assert written_values[1][0:3] == ["CTGAN", "adult", "one-hot"]


def test_recompute_creates_results_worksheet_before_any_valid_pairs(tmp_path, monkeypatch):
    manifest_path = _write_manifest(tmp_path)
    _write_csv(tmp_path)

    class FakeWorksheet:
        def __init__(self, title):
            self.title = title
            self.update_calls = []

        def update(self, *args, **kwargs):
            self.update_calls.append((args, kwargs))

        def freeze(self, rows=0):
            pass

    class FakeSpreadsheet:
        def __init__(self):
            self.created = []

        def worksheet(self, title):
            raise RuntimeError("no existing worksheet")

        def add_worksheet(self, title, rows, cols):
            worksheet = FakeWorksheet(title)
            self.created.append((title, rows, cols, worksheet))
            return worksheet

    spreadsheet = FakeSpreadsheet()

    class EmptyDriveClient:
        def find_folder_path(self, *path_parts: str, root_id: str | None = None):
            assert spreadsheet.created, "Results worksheet should be created before Drive scanning starts"
            return None

    monkeypatch.setattr(recompute_mod, "_build_spreadsheet_from_config", lambda config: spreadsheet)

    result = recompute_mod.recompute_all_from_drive(
        manifest_path=manifest_path,
        output_root=tmp_path / "recomputed",
        drive_client=EmptyDriveClient(),
        sheets_config=object(),
        write_sheet=True,
    )

    assert result.rows == []
    assert spreadsheet.created[0][:3] == ("Results", 1000, len(recompute_mod.FULL_RESULTS_HEADERS))
    worksheet = spreadsheet.created[0][3]
    args, kwargs = worksheet.update_calls[0]
    written_values = kwargs.get("values") or args[0]
    assert written_values == [recompute_mod.FULL_RESULTS_HEADERS]


def test_load_ctgan_artifacts_cpu_moves_model_to_cpu(tmp_path):
    sample_df = pd.DataFrame({"x": [1, 2]})
    artifact_dir = tmp_path / "artifacts" / "fold_0"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "ctgan.pkl").write_bytes(_ctgan_pickle(sample_df))

    model = recompute_mod._load_ctgan_artifacts_cpu(artifact_dir)

    assert model.fitted_ is True
    assert model.model_.device == "cpu"
    assert model.sample(2).equals(sample_df)


def test_recompute_from_drive_uses_saved_models_without_refitting(tmp_path, monkeypatch):
    manifest_path = _write_manifest(tmp_path)
    _write_csv(tmp_path)
    df = pd.read_csv(tmp_path / "datasets" / "raw" / "openml_adult.csv")
    schema = TabularSchema.infer_from_dataframe(df, target_col="target")
    split = full_mod._prepare_holdout_data(
        df=df,
        schema=schema,
        encoding_method="one_hot_representation",
        is_regression=False,
    )
    run_summary = {
        "poster_fast": {"enabled": True, "effective_rows": len(df), "max_rows": None},
        "crossval": {"n_folds": 1},
    }
    fake_drive = FakeDriveClient(
        files={
            "summary": json.dumps(run_summary).encode("utf-8"),
            "broken-new-model": b"not a pickle",
            "valid-old-model": _ctgan_pickle(split.train_transformed),
        }
    )

    def fail_fit(*args, **kwargs):
        raise AssertionError("recompute workflow must not refit CTGAN models")

    monkeypatch.setattr(recompute_mod.CtganGenerative, "fit", fail_fit)
    monkeypatch.setattr(
        recompute_mod.full_mod,
        "tstr_catboost",
        lambda **kwargs: {
            "task_type": "classification",
            "f1_weighted_real": 1.0,
            "f1_weighted_synth": 0.75,
            "f1_weighted_pct_diff": 25.0,
        },
    )

    result = recompute_mod.recompute_all_from_drive(
        manifest_path=manifest_path,
        output_root=tmp_path / "recomputed",
        drive_client=fake_drive,
        model_name="CTGAN",
        write_sheet=False,
    )

    assert fake_drive.downloads == ["summary", "broken-new-model", "valid-old-model"]
    assert len(result.rows) == 1
    aggregate_path = tmp_path / "recomputed" / "openml_adult" / "one_hot_representation" / "metrics" / "aggregate.json"
    fold_path = tmp_path / "recomputed" / "openml_adult" / "one_hot_representation" / "crossval" / "per_fold" / "fold_0.json"
    summary_path = tmp_path / "recomputed" / "openml_adult" / "one_hot_representation" / "run_summary.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    fold = json.loads(fold_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert set(aggregate["distribution"]) >= {
        "wasserstein_mean",
        "marginal_kl_mean",
        "corr_frobenius_transformed",
        "wasserstein_mean_unencoded",
        "marginal_kl_mean_unencoded",
        "corr_frobenius_unencoded",
    }
    assert aggregate["tstr"]["metrics"]["f1_weighted_pct_diff"]["mean"] == 25.0
    assert fold["source_artifacts"]["model_file_id"] == "valid-old-model"
    assert summary["source_drive"]["folder_id"] == "drive-folder-id"
    assert result.rows[0][0:3] == ["CTGAN", "adult", "one-hot"]
