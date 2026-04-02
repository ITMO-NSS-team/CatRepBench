import json

import pytest

from experiments.ctgan_sheets import SheetsClient, SheetsConfig, retry_call


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
