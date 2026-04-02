import pytest

from experiments.ctgan_sheets import SheetsConfig, retry_call


def test_sheets_config_requires_service_account_env(monkeypatch):
    monkeypatch.delenv("CATREPBENCH_GSHEETS_SERVICE_ACCOUNT_PATH", raising=False)
    monkeypatch.setenv("CATREPBENCH_GSHEETS_SPREADSHEET_ID", "spreadsheet-id")

    with pytest.raises(
        ValueError,
        match="CATREPBENCH_GSHEETS_SERVICE_ACCOUNT_PATH",
    ):
        SheetsConfig.from_env()


def test_sheets_config_requires_spreadsheet_id_env(monkeypatch):
    monkeypatch.setenv("CATREPBENCH_GSHEETS_SERVICE_ACCOUNT_PATH", "/tmp/service-account.json")
    monkeypatch.delenv("CATREPBENCH_GSHEETS_SPREADSHEET_ID", raising=False)

    with pytest.raises(
        ValueError,
        match="CATREPBENCH_GSHEETS_SPREADSHEET_ID",
    ):
        SheetsConfig.from_env()


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
