from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

_RETRY_DELAYS = (1, 2, 4, 8, 16)
_GSHEETS_SCOPES = ("https://www.googleapis.com/auth/spreadsheets",)


def retry_call(func: Callable[[], Any], sleep: Callable[[float], None] = time.sleep) -> Any:
    last_error: Exception | None = None
    for delay in _RETRY_DELAYS:
        try:
            return func()
        except Exception as exc:  # noqa: BLE001 - bound the retry policy around any API error
            last_error = exc
            sleep(delay)
    if last_error is not None:
        raise last_error
    raise RuntimeError("retry_call exhausted without executing the function")


@dataclass(frozen=True)
class SheetsConfig:
    service_account_path: Path
    spreadsheet_id: str
    worksheet_name: str | None = None

    @classmethod
    def from_env(cls) -> "SheetsConfig":
        service_account_path = os.getenv("CATREPBENCH_GSHEETS_SERVICE_ACCOUNT_PATH")
        spreadsheet_id = os.getenv("CATREPBENCH_GSHEETS_SPREADSHEET_ID")
        worksheet_name = os.getenv("CATREPBENCH_GSHEETS_WORKSHEET")

        missing = []
        if not service_account_path or not service_account_path.strip():
            missing.append("CATREPBENCH_GSHEETS_SERVICE_ACCOUNT_PATH")
        if not spreadsheet_id or not spreadsheet_id.strip():
            missing.append("CATREPBENCH_GSHEETS_SPREADSHEET_ID")
        if missing:
            missing_labels = ", ".join(missing)
            raise ValueError(
                "CATREPBENCH_GSHEETS configuration is incomplete: "
                f"missing {missing_labels}"
            )

        return cls(
            service_account_path=Path(service_account_path).expanduser(),
            spreadsheet_id=spreadsheet_id.strip(),
            worksheet_name=worksheet_name.strip() if worksheet_name and worksheet_name.strip() else None,
        )


class SheetsClient:
    def __init__(self, config: SheetsConfig, *, worksheet: Any | None = None) -> None:
        self.config = config
        self._worksheet = worksheet if worksheet is not None else self._bootstrap_worksheet(config)

    @classmethod
    def from_env(cls, *, worksheet: Any | None = None) -> "SheetsClient":
        return cls(SheetsConfig.from_env(), worksheet=worksheet)

    @staticmethod
    def _import_gspread() -> Any:
        try:
            import gspread
        except ImportError as exc:  # pragma: no cover - exercised only when runtime deps are absent
            raise RuntimeError(
                "gspread is required to talk to Google Sheets. "
                "Install requirements.txt first."
            ) from exc
        return gspread

    @staticmethod
    def _import_service_account() -> Any:
        try:
            from google.oauth2 import service_account
        except ImportError as exc:  # pragma: no cover - exercised only when runtime deps are absent
            raise RuntimeError(
                "google-auth is required to talk to Google Sheets. "
                "Install requirements.txt first."
            ) from exc
        return service_account

    @classmethod
    def _bootstrap_worksheet(cls, config: SheetsConfig) -> Any:
        gspread = cls._import_gspread()
        service_account = cls._import_service_account()

        credentials = service_account.Credentials.from_service_account_file(
            filename=str(config.service_account_path),
            scopes=_GSHEETS_SCOPES,
        )
        client = gspread.authorize(credentials)
        spreadsheet = retry_call(lambda: client.open_by_key(config.spreadsheet_id))

        if config.worksheet_name:
            return retry_call(lambda: spreadsheet.worksheet(config.worksheet_name))
        return retry_call(lambda: spreadsheet.sheet1)

    def read_matrix(self) -> list[list[str]]:
        return retry_call(self._worksheet.get_all_values)

    def read_cell(self, coord: str) -> Any:
        return retry_call(lambda: self._worksheet.acell(coord).value)

    def write_cell(self, coord: str, payload: str) -> Any:
        return retry_call(lambda: self._worksheet.update_acell(coord, payload))
