import json
import sys
import types
from datetime import datetime, timezone

import pytest

import experiments.ctgan.orchestrator_staff.ctgan_drive as drive_mod
from experiments.ctgan.orchestrator_staff.ctgan_sheets import SheetsClient, SheetsConfig, retry_call


def test_sheets_config_requires_service_account_env(monkeypatch):
    monkeypatch.delenv("CATREPBENCH_GSHEETS_SERVICE_ACCOUNT_PATH", raising=False)
    monkeypatch.delenv("CATREPBENCH_GSHEETS_SERVICE_ACCOUNT_JSON", raising=False)
    monkeypatch.setenv("CATREPBENCH_GSHEETS_SPREADSHEET_ID", "spreadsheet-id")

    with pytest.raises(
        ValueError,
        match="CATREPBENCH_GSHEETS_SERVICE_ACCOUNT_PATH|CATREPBENCH_GSHEETS_SERVICE_ACCOUNT_JSON",
    ):
        SheetsConfig.from_env()


def test_sheets_config_requires_spreadsheet_id_env(monkeypatch):
    monkeypatch.setenv("CATREPBENCH_GSHEETS_SERVICE_ACCOUNT_PATH", "/tmp/service-account.json")
    monkeypatch.delenv("CATREPBENCH_GSHEETS_SERVICE_ACCOUNT_JSON", raising=False)
    monkeypatch.delenv("CATREPBENCH_GSHEETS_SPREADSHEET_ID", raising=False)

    with pytest.raises(
        ValueError,
        match="CATREPBENCH_GSHEETS_SPREADSHEET_ID",
    ):
        SheetsConfig.from_env()


def test_sheets_config_accepts_inline_service_account_json(monkeypatch):
    monkeypatch.delenv("CATREPBENCH_GSHEETS_SERVICE_ACCOUNT_PATH", raising=False)
    monkeypatch.setenv(
        "CATREPBENCH_GSHEETS_SERVICE_ACCOUNT_JSON",
        json.dumps({"type": "service_account", "project_id": "catrepbench-test"}),
    )
    monkeypatch.setenv("CATREPBENCH_GSHEETS_SPREADSHEET_ID", "spreadsheet-id")
    monkeypatch.setenv("CATREPBENCH_GSHEETS_WORKSHEET", "CTGAN")

    config = SheetsConfig.from_env()

    assert config.service_account_path is None
    assert config.service_account_info == {
        "type": "service_account",
        "project_id": "catrepbench-test",
    }
    assert config.spreadsheet_id == "spreadsheet-id"
    assert config.worksheet_name == "CTGAN"


def test_sheets_client_bootstrap_uses_inline_service_account_json(monkeypatch):
    config = SheetsConfig(
        service_account_path=None,
        service_account_info={"type": "service_account", "project_id": "catrepbench-test"},
        spreadsheet_id="spreadsheet-id",
        worksheet_name="CTGAN",
    )
    calls: dict[str, object] = {}

    class FakeSpreadsheet:
        def worksheet(self, name):
            calls["worksheet_name"] = name
            return "worksheet-object"

    class FakeAuthorizedClient:
        def open_by_key(self, key):
            calls["spreadsheet_id"] = key
            return FakeSpreadsheet()

    class FakeGspread:
        @staticmethod
        def authorize(credentials):
            calls["credentials"] = credentials
            return FakeAuthorizedClient()

    class FakeCredentials:
        @staticmethod
        def from_service_account_file(filename, scopes):  # pragma: no cover - should not be used
            raise AssertionError("file credentials path should not be used when inline JSON is present")

        @staticmethod
        def from_service_account_info(info, scopes):
            calls["service_account_info"] = info
            calls["scopes"] = scopes
            return "inline-creds"

    monkeypatch.setattr(SheetsClient, "_import_gspread", staticmethod(lambda: FakeGspread))
    monkeypatch.setattr(
        SheetsClient,
        "_import_service_account",
        staticmethod(lambda: type("FakeServiceAccountModule", (), {"Credentials": FakeCredentials})),
    )

    worksheet = SheetsClient._bootstrap_worksheet(config)

    assert worksheet == "worksheet-object"
    assert calls["service_account_info"] == {
        "type": "service_account",
        "project_id": "catrepbench-test",
    }
    assert calls["spreadsheet_id"] == "spreadsheet-id"
    assert calls["worksheet_name"] == "CTGAN"


def test_retry_call_uses_full_retry_schedule_before_raising():
    calls = {"n": 0}
    sleeps: list[float] = []

    def flaky():
        calls["n"] += 1
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        retry_call(flaky, sleep=sleeps.append)

    assert calls["n"] == 5
    assert sleeps == [1, 2, 4, 8, 16]


def test_results_sheet_writer_uses_unencoded_distribution_metrics(tmp_path, monkeypatch):
    aggregate_path = tmp_path / "aggregate.json"
    aggregate_path.write_text(
        json.dumps(
            {
                "distribution": {
                    "wasserstein_mean": {"mean": 1.0, "std": 1.1},
                    "marginal_kl_mean": {"mean": 2.0, "std": 2.2},
                    "corr_frobenius_transformed": {"mean": 3.0, "std": 3.3},
                    "wasserstein_mean_unencoded": {"mean": 10.1234567, "std": 10.7654321},
                    "marginal_kl_mean_unencoded": {"mean": 20.1234567, "std": 20.7654321},
                    "corr_frobenius_unencoded": {"mean": 30.1234567, "std": 30.7654321},
                }
            }
        ),
        encoding="utf-8",
    )

    class FakeResultsClient:
        def __init__(self):
            self.writes: dict[str, str] = {}

        def read_matrix(self):
            return [["header"], ["subheader"]]

        def write_cell(self, coord, value):
            self.writes[coord] = value

    fake_client = FakeResultsClient()
    monkeypatch.setattr(
        drive_mod,
        "_build_sheets_client_with_drive_scopes",
        lambda config: fake_client,
    )

    drive_mod.write_results_row(
        sheets_config=SheetsConfig(
            spreadsheet_id="sheet-id",
            service_account_info={"type": "service_account"},
            worksheet_name="CTGAN",
        ),
        aggregate_metrics_path=aggregate_path,
        model_name="CTGAN",
        dataset_label="adult",
        encoding_label="one-hot",
        folder_url="https://example.test/folder",
    )

    assert fake_client.writes["A1"] == "Model"
    assert fake_client.writes["B1"] == "Dataset"
    assert fake_client.writes["C1"] == "Categorical representation"
    assert fake_client.writes["D1"] == "Mean WD"
    assert fake_client.writes["F1"] == "Mean KL"
    assert fake_client.writes["H1"] == "Mean Corr dist"
    assert fake_client.writes["A2"] == "CTGAN"
    assert fake_client.writes["D2"] == "10.123457"
    assert fake_client.writes["F2"] == "20.123457"
    assert fake_client.writes["H2"] == "30.123457"


def test_build_results_sheet_row_does_not_fall_back_to_legacy_distribution_metric_names():
    row = drive_mod.build_results_sheet_row(
        model_name="CTGAN",
        dataset_label="adult",
        dataset_id="openml_adult",
        encoding_label="one-hot",
        encoding_id="one_hot_representation",
        aggregate={
            "distribution": {
                "wasserstein_mean": {"mean": 1.25, "std": 0.1},
                "marginal_kl_mean": {"mean": 2.5, "std": 0.2},
                "corr_frobenius_transformed": {"mean": 3.75, "std": 0.3},
            },
            "tstr": {
                "task_type": "classification",
                "metrics": {"f1_weighted_pct_diff": {"mean": 4.5, "std": 0.4}},
            },
        },
        folder_url="https://example.test/folder",
    )

    assert row[3:9] == ["", "", "", "", "", ""]


def test_download_first_valid_drive_file_uses_newest_non_broken_candidate(tmp_path):
    records = [
        drive_mod.DriveFileRecord(
            file_id="old-valid",
            name="ctgan.pkl",
            mime_type="application/octet-stream",
            modified_time="2026-04-23T10:00:00Z",
            relative_path="artifacts/fold_0/ctgan.pkl",
            parent_id="fold-folder",
        ),
        drive_mod.DriveFileRecord(
            file_id="new-broken",
            name="ctgan.pkl",
            mime_type="application/octet-stream",
            modified_time="2026-04-24T10:00:00Z",
            relative_path="artifacts/fold_0/ctgan.pkl",
            parent_id="fold-folder",
        ),
    ]

    class FakeDriveClient:
        def download_file(self, file_id, destination):
            destination.write_text("broken" if file_id == "new-broken" else "valid", encoding="utf-8")

    selected = drive_mod.download_first_valid_drive_file(
        drive_client=FakeDriveClient(),
        candidates=records,
        destination=tmp_path / "ctgan.pkl",
        validator=lambda path: path.read_text(encoding="utf-8") == "valid",
    )

    assert selected.file_id == "old-valid"
    assert (tmp_path / "ctgan.pkl").read_text(encoding="utf-8") == "valid"


def test_drive_client_upload_file_overwrites_existing_file(tmp_path, monkeypatch):
    local_file = tmp_path / "aggregate.json"
    local_file.write_text('{"ok": true}', encoding="utf-8")
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeMediaFileUpload:
        def __init__(self, filename, *, mimetype=None, resumable=False):
            self.filename = filename
            self.mimetype = mimetype
            self.resumable = resumable

    monkeypatch.setitem(
        sys.modules,
        "googleapiclient.http",
        types.SimpleNamespace(MediaFileUpload=FakeMediaFileUpload),
    )

    class FakeRequest:
        def __init__(self, payload):
            self.payload = payload

        def execute(self):
            return self.payload

    class FakeFiles:
        def list(self, **kwargs):
            calls.append(("list", kwargs))
            return FakeRequest({"files": [{"id": "existing-file"}]})

        def update(self, **kwargs):
            calls.append(("update", kwargs))
            return FakeRequest({"id": kwargs["fileId"]})

        def create(self, **kwargs):
            calls.append(("create", kwargs))
            return FakeRequest({"id": "new-file"})

    class FakeService:
        def files(self):
            return FakeFiles()

    client = object.__new__(drive_mod.DriveClient)
    client._service = FakeService()

    file_id = client.upload_file(local_file, "parent-folder", overwrite=True)

    assert file_id == "existing-file"
    assert [name for name, _kwargs in calls] == ["list", "update"]
    update_kwargs = calls[1][1]
    assert update_kwargs["fileId"] == "existing-file"
    assert update_kwargs["supportsAllDrives"] is True


def test_refresh_full_results_worksheet_archives_existing_results_and_writes_header():
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

    spreadsheet = FakeSpreadsheet()
    rows = [
        drive_mod.build_results_sheet_row(
            model_name="CTGAN",
            dataset_label="adult",
            dataset_id="openml_adult",
            encoding_label="one-hot",
            encoding_id="one_hot_representation",
            aggregate={
                "n_folds": 1,
                "distribution": {
                    "wasserstein_mean_unencoded": {"mean": 0.1, "std": 0.0},
                },
                "tstr": {"status": "ok", "task_type": "classification", "metrics": {}},
            },
            folder_url="https://drive.google.test/folder",
        )
    ]

    drive_mod.refresh_full_results_worksheet(
        spreadsheet=spreadsheet,
        worksheet_name="Results",
        rows=rows,
        archive_existing=True,
        now=datetime(2026, 4, 24, 9, 44, 0, tzinfo=timezone.utc),
    )

    assert spreadsheet.old.updated_title == "Results_legacy_20260424_094400"
    assert spreadsheet.created[0][:3] == ("Results", 1000, len(drive_mod.FULL_RESULTS_HEADERS))
    new_worksheet = spreadsheet.created[0][3]
    assert new_worksheet.update_calls
    args, kwargs = new_worksheet.update_calls[0]
    written_values = kwargs.get("values") or args[0]
    assert written_values[0] == drive_mod.FULL_RESULTS_HEADERS
    assert written_values[1][0:3] == ["CTGAN", "adult", "one-hot"]
