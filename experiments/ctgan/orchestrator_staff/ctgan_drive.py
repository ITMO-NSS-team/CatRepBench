"""Google Drive upload + Results sheet writer for CatRepBench experiments.

Environment variables
---------------------
CATREPBENCH_GDRIVE_RESULTS_FOLDER_ID
    ID of the root Google Drive folder where experiment artifacts are stored.
    Artifacts are uploaded to:
        <root_folder>/<model>/<dataset_id>/<encoding_method>/

The Google Sheets credentials are reused from ctgan_sheets.SheetsConfig (same
service account must have Drive API access as well as Sheets API access).

Usage
-----
After a successful run of ``run_full_ctgan_experiment``:

    from experiments.ctgan.orchestrator_staff.ctgan_drive import (
        DriveConfig,
        DriveClient,
        ResultsSheetWriter,
        upload_experiment_artifacts,
        write_results_row,
    )

    drive_config = DriveConfig.from_env()           # raises if env not set
    drive_client = DriveClient(drive_config)
    folder_url = upload_experiment_artifacts(
        drive_client=drive_client,
        run_dir=result.output_dir,
        model_name="CTGAN",
        dataset_id=args.dataset_id,
        encoding_method=args.encoding_method,
    )
    write_results_row(
        sheets_config=sheets_config,
        aggregate_metrics_path=result.aggregate_metrics_path,
        model_name="CTGAN",
        dataset_label=args.dataset_label,
        encoding_label=args.encoding_method,
        folder_url=folder_url,
        results_worksheet_name="Results",  # second sheet name
    )
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Retry helper (mirrors ctgan_sheets)
# ---------------------------------------------------------------------------

_RETRY_DELAYS = (1, 2, 4, 8, 16)
_DRIVE_SCOPES = (
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
)

_FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"

FULL_RESULTS_HEADERS = [
    "Model",
    "Dataset",
    "Categorical representation",
    "Mean WD",
    "Std WD",
    "Mean KL",
    "Std KL",
    "Mean Corr dist",
    "Std Corr dist",
    "Mean Utility",
    "Std Utility",
    "Task Type",
    "Drive folder URL",
]


def _retry(func: Callable[[], Any], sleep: Callable[[float], None] = time.sleep) -> Any:
    last_error: Exception | None = None
    for delay in _RETRY_DELAYS:
        try:
            return func()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            sleep(delay)
    if last_error is not None:
        raise last_error
    raise RuntimeError("_retry exhausted without executing the function")


@dataclass(frozen=True)
class DriveFileRecord:
    file_id: str
    name: str
    mime_type: str
    modified_time: str
    relative_path: str
    parent_id: str


# ---------------------------------------------------------------------------
# DriveConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DriveConfig:
    """Configuration for Drive uploads + Sheets credentials."""

    results_folder_id: str
    """Root Google Drive folder ID where experiment runs are stored."""

    service_account_path: Path | None = None
    service_account_info: dict[str, Any] | None = None
    oauth_token_path: Path | None = None
    """Path to OAuth2 token JSON (generated via InstalledAppFlow).
    When set, Drive uploads use this token instead of the service account,
    which avoids the 'Service Accounts do not have storage quota' error."""

    @classmethod
    def from_env(cls) -> "DriveConfig":
        """Build config from environment variables.

        Required:
            CATREPBENCH_GDRIVE_RESULTS_FOLDER_ID

        Drive credentials (pick one):
            CATREPBENCH_GDRIVE_OAUTH_TOKEN_PATH   — OAuth2 token (recommended)
            CATREPBENCH_GSHEETS_SERVICE_ACCOUNT_PATH  OR
            CATREPBENCH_GSHEETS_SERVICE_ACCOUNT_JSON  — service account fallback
        """
        results_folder_id = os.getenv("CATREPBENCH_GDRIVE_RESULTS_FOLDER_ID", "").strip()
        oauth_token_path = os.getenv("CATREPBENCH_GDRIVE_OAUTH_TOKEN_PATH", "").strip()
        service_account_path = os.getenv("CATREPBENCH_GSHEETS_SERVICE_ACCOUNT_PATH", "").strip()
        service_account_json = os.getenv("CATREPBENCH_GSHEETS_SERVICE_ACCOUNT_JSON", "").strip()

        missing: list[str] = []
        if not results_folder_id:
            missing.append("CATREPBENCH_GDRIVE_RESULTS_FOLDER_ID")

        inline_info: dict[str, Any] | None = None
        if not oauth_token_path:
            if service_account_json:
                try:
                    inline_info = json.loads(service_account_json)
                except json.JSONDecodeError as exc:
                    raise ValueError("CATREPBENCH_GSHEETS_SERVICE_ACCOUNT_JSON must be valid JSON") from exc
            elif not service_account_path:
                missing.append(
                    "CATREPBENCH_GDRIVE_OAUTH_TOKEN_PATH or "
                    "CATREPBENCH_GSHEETS_SERVICE_ACCOUNT_PATH or "
                    "CATREPBENCH_GSHEETS_SERVICE_ACCOUNT_JSON"
                )

        if missing:
            raise ValueError(
                "DriveConfig is incomplete, missing: " + ", ".join(missing)
            )

        return cls(
            results_folder_id=results_folder_id,
            service_account_path=Path(service_account_path).expanduser() if service_account_path else None,
            service_account_info=inline_info,
            oauth_token_path=Path(oauth_token_path).expanduser() if oauth_token_path else None,
        )

    @classmethod
    def is_configured(cls) -> bool:
        """Return True if all required env variables are present (no exception)."""
        try:
            cls.from_env()
            return True
        except ValueError:
            return False


# ---------------------------------------------------------------------------
# DriveClient
# ---------------------------------------------------------------------------


class DriveClient:
    """Thin wrapper around the Google Drive v3 REST API."""

    def __init__(self, config: DriveConfig) -> None:
        self.config = config
        self._service = self._build_service(config)

    # ------------------------------------------------------------------
    # Internal bootstrap
    # ------------------------------------------------------------------

    @staticmethod
    def _build_credentials(config: DriveConfig) -> Any:
        # --- OAuth2 user credentials (preferred — avoids service-account quota error) ---
        if config.oauth_token_path is not None:
            try:
                from google.oauth2.credentials import Credentials
                from google.auth.transport.requests import Request
            except ImportError as exc:
                raise RuntimeError(
                    "google-auth is required. Install requirements.txt first."
                ) from exc
            token_path = config.oauth_token_path
            # Don't pass scopes here — use whatever scopes the token was granted with.
            # Passing extra scopes causes invalid_scope on refresh.
            creds = Credentials.from_authorized_user_file(str(token_path))
            if creds.expired and creds.refresh_token:
                creds.refresh(Request())
                with open(token_path, "w", encoding="utf-8") as fh:
                    fh.write(creds.to_json())
            return creds

        # --- Service account fallback ---
        try:
            from google.oauth2 import service_account
        except ImportError as exc:
            raise RuntimeError(
                "google-auth is required. Install requirements.txt first."
            ) from exc

        if config.service_account_info is not None:
            return service_account.Credentials.from_service_account_info(
                info=config.service_account_info,
                scopes=_DRIVE_SCOPES,
            )
        if config.service_account_path is not None:
            return service_account.Credentials.from_service_account_file(
                filename=str(config.service_account_path),
                scopes=_DRIVE_SCOPES,
            )
        raise ValueError(
            "DriveConfig must define oauth_token_path, service_account_path, or service_account_info"
        )

    @classmethod
    def _build_service(cls, config: DriveConfig) -> Any:
        try:
            from googleapiclient.discovery import build
        except ImportError as exc:
            raise RuntimeError(
                "google-api-python-client is required for Drive uploads. "
                "Run: pip install google-api-python-client"
            ) from exc
        credentials = cls._build_credentials(config)
        return build("drive", "v3", credentials=credentials, cache_discovery=False)

    # ------------------------------------------------------------------
    # Folder helpers
    # ------------------------------------------------------------------

    def find_folder(self, name: str, parent_id: str) -> str | None:
        """Return an existing folder ID without creating anything."""
        query = (
            f"name = {json.dumps(name)} "
            f"and '{parent_id}' in parents "
            f"and mimeType = '{_FOLDER_MIME_TYPE}' "
            f"and trashed = false"
        )

        def _search() -> list[dict[str, Any]]:
            return (
                self._service.files()
                .list(
                    q=query,
                    fields="files(id,name,modifiedTime)",
                    spaces="drive",
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
                .get("files", [])
            )

        results = _retry(_search)
        if not results:
            return None
        latest = sorted(
            results,
            key=lambda item: _parse_drive_modified_time(str(item.get("modifiedTime", ""))),
            reverse=True,
        )[0]
        return str(latest["id"])

    def find_folder_path(self, *path_parts: str, root_id: str | None = None) -> str | None:
        """Return the leaf folder for an existing path, or None when any part is absent."""
        current_parent = root_id or self.config.results_folder_id
        for part in path_parts:
            found = self.find_folder(part, current_parent)
            if found is None:
                return None
            current_parent = found
        return current_parent

    def _find_or_create_folder(self, name: str, parent_id: str) -> str:
        """Return folder ID, creating it if it doesn't exist."""
        query = (
            f"name = {json.dumps(name)} "
            f"and '{parent_id}' in parents "
            f"and mimeType = '{_FOLDER_MIME_TYPE}' "
            f"and trashed = false"
        )

        def _search() -> list[dict[str, Any]]:
            return (
                self._service.files()
                .list(
                    q=query,
                    fields="files(id,name)",
                    spaces="drive",
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
                .get("files", [])
            )

        results = _retry(_search)
        if results:
            return str(results[0]["id"])

        metadata: dict[str, Any] = {
            "name": name,
            "mimeType": _FOLDER_MIME_TYPE,
            "parents": [parent_id],
        }

        def _create() -> dict[str, Any]:
            return (
                self._service.files()
                .create(body=metadata, fields="id")
                .execute()
            )

        created = _retry(_create)
        return str(created["id"])

    def ensure_folder_path(self, *path_parts: str, root_id: str | None = None) -> str:
        """Recursively ensure a folder hierarchy exists and return the leaf folder ID.

        Example::

            folder_id = client.ensure_folder_path("CTGAN", "openml_eucalyptus", "frequency_representation")
        """
        current_parent = root_id or self.config.results_folder_id
        for part in path_parts:
            current_parent = self._find_or_create_folder(part, current_parent)
        return current_parent

    # ------------------------------------------------------------------
    # File upload
    # ------------------------------------------------------------------

    def upload_file(self, local_path: Path, parent_folder_id: str) -> str:
        """Upload *local_path* to Drive inside *parent_folder_id*. Returns file ID."""
        try:
            from googleapiclient.http import MediaFileUpload
        except ImportError as exc:
            raise RuntimeError(
                "google-api-python-client is required for Drive uploads."
            ) from exc

        mime_type = _guess_mime_type(local_path)
        metadata: dict[str, Any] = {"name": local_path.name, "parents": [parent_folder_id]}
        media = MediaFileUpload(str(local_path), mimetype=mime_type, resumable=True)

        def _upload() -> dict[str, Any]:
            return (
                self._service.files()
                .create(
                    body=metadata,
                    media_body=media,
                    fields="id",
                    supportsAllDrives=True,
                )
                .execute()
            )

        result = _retry(_upload)
        return str(result["id"])

    def download_file(self, file_id: str, destination: Path) -> None:
        """Download one Drive file to *destination*, replacing any existing file."""
        try:
            from googleapiclient.http import MediaIoBaseDownload
        except ImportError as exc:
            raise RuntimeError(
                "google-api-python-client is required for Drive downloads."
            ) from exc

        destination.parent.mkdir(parents=True, exist_ok=True)

        def _download() -> None:
            request = self._service.files().get_media(fileId=file_id, supportsAllDrives=True)
            with destination.open("wb") as handle:
                downloader = MediaIoBaseDownload(handle, request)
                done = False
                while not done:
                    _status, done = downloader.next_chunk()

        _retry(_download)

    def list_files_recursive(self, folder_id: str) -> list[DriveFileRecord]:
        """List non-folder files below *folder_id* with their relative paths."""
        records: list[DriveFileRecord] = []
        self._list_files_recursive_into(folder_id=folder_id, relative_prefix="", records=records)
        return records

    def _list_files_recursive_into(
        self,
        *,
        folder_id: str,
        relative_prefix: str,
        records: list[DriveFileRecord],
    ) -> None:
        page_token: str | None = None
        while True:
            query = f"'{folder_id}' in parents and trashed = false"

            def _list_page() -> dict[str, Any]:
                return (
                    self._service.files()
                    .list(
                        q=query,
                        fields="nextPageToken,files(id,name,mimeType,modifiedTime,parents)",
                        pageToken=page_token,
                        spaces="drive",
                        supportsAllDrives=True,
                        includeItemsFromAllDrives=True,
                    )
                    .execute()
                )

            page = _retry(_list_page)
            for item in page.get("files", []):
                name = str(item["name"])
                mime_type = str(item.get("mimeType", ""))
                item_id = str(item["id"])
                relative_path = f"{relative_prefix}/{name}" if relative_prefix else name
                if mime_type == _FOLDER_MIME_TYPE:
                    self._list_files_recursive_into(
                        folder_id=item_id,
                        relative_prefix=relative_path,
                        records=records,
                    )
                    continue
                records.append(
                    DriveFileRecord(
                        file_id=item_id,
                        name=name,
                        mime_type=mime_type,
                        modified_time=str(item.get("modifiedTime", "")),
                        relative_path=relative_path,
                        parent_id=folder_id,
                    )
                )
            page_token = page.get("nextPageToken")
            if not page_token:
                break

    def folder_web_url(self, folder_id: str) -> str:
        """Return the browser-facing URL for a Drive folder."""
        return f"https://drive.google.com/drive/folders/{folder_id}"

    # ------------------------------------------------------------------
    # Convenience: upload an entire local directory tree
    # ------------------------------------------------------------------

    def upload_directory(
        self,
        local_dir: Path,
        parent_folder_id: str,
        *,
        extensions: tuple[str, ...] | None = None,
    ) -> int:
        """Upload all files in *local_dir* (recursively) to Drive.

        Parameters
        ----------
        local_dir:
            Local directory to upload.
        parent_folder_id:
            Drive folder that will receive the mirrored sub-tree.
        extensions:
            If given, only files with these suffixes are uploaded
            (e.g. ``(".json", ".png", ".pt")``).

        Returns
        -------
        int
            Number of files uploaded.
        """
        if not local_dir.is_dir():
            return 0

        # Map local sub-path → drive folder id so we reuse created folders
        folder_cache: dict[Path, str] = {local_dir: parent_folder_id}
        count = 0

        for local_file in sorted(local_dir.rglob("*")):
            if not local_file.is_file():
                continue
            if extensions is not None and local_file.suffix.lower() not in extensions:
                continue

            # Ensure parent folder exists in Drive
            relative_parent = local_file.parent
            if relative_parent not in folder_cache:
                # Build the chain from the first unknown ancestor
                chain: list[Path] = []
                cursor = relative_parent
                while cursor not in folder_cache:
                    chain.append(cursor)
                    cursor = cursor.parent
                chain.reverse()
                current_drive_id = folder_cache[cursor]
                for folder_path in chain:
                    current_drive_id = self._find_or_create_folder(
                        folder_path.name, current_drive_id
                    )
                    folder_cache[folder_path] = current_drive_id

            drive_parent_id = folder_cache[relative_parent]
            self.upload_file(local_file, drive_parent_id)
            count += 1

        return count


# ---------------------------------------------------------------------------
# High-level helpers
# ---------------------------------------------------------------------------


def upload_experiment_artifacts(
    *,
    drive_client: DriveClient,
    run_dir: Path,
    model_name: str,
    dataset_id: str,
    encoding_method: str,
) -> str:
    """Upload checkpoints and metric plots from *run_dir* to Google Drive.

    Folder structure on Drive::

        <results_root>/<model_name>/<dataset_id>/<encoding_method>/

    Uploads only files that are clearly plots or model checkpoints:
        * PNG / SVG / PDF images (loss curves, metric plots)
        * .pt / .ckpt / .safetensors (model checkpoints)
        * JSON result files (aggregate.json, run_summary.json, fold_*.json)

    Returns
    -------
    str
        Browser URL of the leaf Drive folder.
    """
    leaf_folder_id = drive_client.ensure_folder_path(
        model_name, dataset_id, encoding_method
    )

    _ARTIFACT_EXTENSIONS = (
        # plots
        ".png", ".svg", ".pdf",
        # checkpoints
        ".pt", ".ckpt", ".safetensors", ".pth", ".pkl",
        # results / metrics
        ".json", ".csv",
    )

    n_uploaded = drive_client.upload_directory(
        run_dir,
        leaf_folder_id,
        extensions=_ARTIFACT_EXTENSIONS,
    )
    print(
        f"[ctgan_drive] Uploaded {n_uploaded} artifact(s) to Drive folder: "
        f"{drive_client.folder_web_url(leaf_folder_id)}"
    )
    return drive_client.folder_web_url(leaf_folder_id)


def _parse_drive_modified_time(value: str) -> datetime:
    if not value:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def sort_drive_file_records_newest_first(records: list[DriveFileRecord]) -> list[DriveFileRecord]:
    return sorted(
        records,
        key=lambda record: _parse_drive_modified_time(record.modified_time),
        reverse=True,
    )


def drive_records_by_relative_path(records: list[DriveFileRecord]) -> dict[str, list[DriveFileRecord]]:
    grouped: dict[str, list[DriveFileRecord]] = {}
    for record in records:
        grouped.setdefault(record.relative_path, []).append(record)
    return {
        relative_path: sort_drive_file_records_newest_first(path_records)
        for relative_path, path_records in grouped.items()
    }


def download_first_valid_drive_file(
    *,
    drive_client: Any,
    candidates: list[DriveFileRecord],
    destination: Path,
    validator: Callable[[Path], bool],
) -> DriveFileRecord:
    """Download the newest candidate that passes *validator*.

    If the freshest clone is corrupt or unreadable, the next newest candidate is tried.
    """
    failures: list[str] = []
    for record in sort_drive_file_records_newest_first(candidates):
        try:
            if destination.exists():
                destination.unlink()
            drive_client.download_file(record.file_id, destination)
            if validator(destination):
                return record
            failures.append(f"{record.file_id}: validator rejected file")
        except Exception as exc:  # noqa: BLE001 - corrupt Drive clones are expected here
            failures.append(f"{record.file_id}: {exc}")
    if destination.exists():
        destination.unlink()
    raise FileNotFoundError(
        "No valid Drive file found for "
        f"{candidates[0].relative_path if candidates else '<empty candidates>'}: "
        + "; ".join(failures)
    )


def _metric_value(aggregate: dict[str, Any], metric_name: str, field: str) -> Any:
    entry = aggregate.get("distribution", {}).get(metric_name, {})
    if isinstance(entry, dict):
        return entry.get(field)
    return None


def _tstr_value(aggregate: dict[str, Any], metric_name: str, field: str) -> Any:
    entry = aggregate.get("tstr", {}).get("metrics", {}).get(metric_name, {})
    if isinstance(entry, dict):
        return entry.get(field)
    return None


def _utility_gap_value(aggregate: dict[str, Any], field: str) -> Any:
    tstr = aggregate.get("tstr", {})
    task_type = tstr.get("task_type") if isinstance(tstr, dict) else None
    if task_type == "regression":
        return _tstr_value(aggregate, "r2_pct_diff", field)
    if task_type == "classification":
        return _tstr_value(aggregate, "f1_weighted_pct_diff", field)
    return _tstr_value(aggregate, "r2_pct_diff", field) or _tstr_value(
        aggregate, "f1_weighted_pct_diff", field
    )


def _format_sheet_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(round(float(value), 6))
    return str(value)


def build_results_sheet_row(
    *,
    model_name: str,
    dataset_label: str,
    dataset_id: str,
    encoding_label: str,
    encoding_id: str,
    aggregate: dict[str, Any],
    folder_url: str,
) -> list[str]:
    # WD, KL, Corr dist are taken from the *unencoded* metrics so that values
    # are comparable across encodings regardless of the encoding dimensionality.
    task_type = aggregate.get("tstr", {}).get("task_type") or ""
    row = [
        model_name,
        dataset_label,
        encoding_label,
        _metric_value(aggregate, "wasserstein_mean_unencoded", "mean"),
        _metric_value(aggregate, "wasserstein_mean_unencoded", "std"),
        _metric_value(aggregate, "marginal_kl_mean_unencoded", "mean"),
        _metric_value(aggregate, "marginal_kl_mean_unencoded", "std"),
        _metric_value(aggregate, "corr_frobenius_unencoded", "mean"),
        _metric_value(aggregate, "corr_frobenius_unencoded", "std"),
        _utility_gap_value(aggregate, "mean"),
        _utility_gap_value(aggregate, "std"),
        task_type,
        folder_url,
    ]
    return [_format_sheet_value(value) for value in row]


def _update_worksheet_values(worksheet: Any, values: list[list[str]]) -> None:
    try:
        worksheet.update(values=values, range_name="A1")
    except TypeError:
        try:
            worksheet.update("A1", values)
        except TypeError:
            worksheet.update(values)


def refresh_full_results_worksheet(
    *,
    spreadsheet: Any,
    worksheet_name: str,
    rows: list[list[str]],
    archive_existing: bool,
    now: datetime | None = None,
) -> Any:
    if archive_existing:
        timestamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).strftime("%Y%m%d_%H%M%S")
        try:
            existing = spreadsheet.worksheet(worksheet_name)
        except Exception:  # noqa: BLE001 - absent sheet is fine when creating a fresh one
            existing = None
        if existing is not None:
            existing.update_title(f"{worksheet_name}_legacy_{timestamp}")

    worksheet = spreadsheet.add_worksheet(
        title=worksheet_name,
        rows=max(len(rows) + 1, 1000),
        cols=len(FULL_RESULTS_HEADERS),
    )
    _update_worksheet_values(worksheet, [FULL_RESULTS_HEADERS, *rows])
    if hasattr(worksheet, "freeze"):
        worksheet.freeze(rows=1)
    return worksheet


def _build_spreadsheet_with_drive_scopes(config: Any) -> Any:
    try:
        import gspread
        from google.oauth2 import service_account
    except ImportError as exc:
        raise RuntimeError(
            "gspread and google-auth are required. Install requirements.txt first."
        ) from exc

    if config.service_account_info is not None:
        credentials = service_account.Credentials.from_service_account_info(
            info=config.service_account_info,
            scopes=list(_DRIVE_SCOPES),
        )
    elif config.service_account_path is not None:
        credentials = service_account.Credentials.from_service_account_file(
            filename=str(config.service_account_path),
            scopes=list(_DRIVE_SCOPES),
        )
    else:
        raise ValueError("SheetsConfig must define service_account_path or service_account_info")

    from experiments.ctgan.orchestrator_staff.ctgan_sheets import retry_call

    client = gspread.authorize(credentials)
    return retry_call(lambda: client.open_by_key(config.spreadsheet_id))


def refresh_full_results_worksheet_from_config(
    *,
    sheets_config: Any,
    worksheet_name: str,
    rows: list[list[str]],
    archive_existing: bool,
) -> None:
    spreadsheet = _build_spreadsheet_with_drive_scopes(sheets_config)
    refresh_full_results_worksheet(
        spreadsheet=spreadsheet,
        worksheet_name=worksheet_name,
        rows=rows,
        archive_existing=archive_existing,
    )


# ---------------------------------------------------------------------------
# Results sheet writer
# ---------------------------------------------------------------------------


def write_results_row(
    *,
    sheets_config: Any,  # ctgan_sheets.SheetsConfig
    aggregate_metrics_path: Path,
    model_name: str,
    dataset_label: str,
    dataset_id: str = "",
    encoding_label: str,
    encoding_id: str = "",
    folder_url: str,
    results_worksheet_name: str = "Results",
) -> None:
    """Append a metrics row to the *results_worksheet_name* sheet.

    The function writes the full metrics schema used by recomputed Results.
    It finds the first empty row by checking column A.
    """
    from experiments.ctgan.orchestrator_staff.ctgan_sheets import SheetsClient, SheetsConfig
    from dataclasses import replace

    # Re-use same service account but target the Results worksheet
    config_results: SheetsConfig = replace(
        sheets_config, worksheet_name=results_worksheet_name
    )
    # Build a client with broader scopes for Drive+Sheets
    # (Drive scopes include Sheets so the same credentials work)
    results_client = _build_sheets_client_with_drive_scopes(config_results)

    # Load aggregate metrics
    with aggregate_metrics_path.open("r", encoding="utf-8") as f:
        aggregate = json.load(f)

    row_values = build_results_sheet_row(
        model_name=model_name,
        dataset_label=dataset_label,
        dataset_id=dataset_id,
        encoding_label=encoding_label,
        encoding_id=encoding_id,
        aggregate=aggregate,
        folder_url=folder_url,
    )

    matrix = results_client.read_matrix()
    has_full_header = bool(matrix and matrix[0][: len(FULL_RESULTS_HEADERS)] == FULL_RESULTS_HEADERS)
    if not has_full_header:
        for col_index, value in enumerate(FULL_RESULTS_HEADERS, start=1):
            results_client.write_cell(f"{_column_name(col_index)}1", value)

    if not has_full_header:
        first_empty_row = 2
    else:
        first_empty_row = max(len(matrix) + 1, 2)
        for row_idx, row in enumerate(matrix):
            if row_idx < 1:
                continue
            cell_a = row[0] if row else ""
            if not cell_a.strip():
                first_empty_row = row_idx + 1  # 1-based sheet row
                break

    for col_idx, value in enumerate(row_values, start=1):
        coord = f"{_column_name(col_idx)}{first_empty_row}"
        results_client.write_cell(coord, value)

    print(
        f"[ctgan_drive] Wrote results row to sheet '{results_worksheet_name}' "
        f"at row {first_empty_row}."
    )


def _column_name(index: int) -> str:
    if index < 1:
        raise ValueError("Column index must be 1-based.")
    letters = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return letters


# ---------------------------------------------------------------------------
# Internal: build a SheetsClient using Drive-compatible scopes
# ---------------------------------------------------------------------------


def _build_sheets_client_with_drive_scopes(config: Any) -> Any:
    """Build a gspread client authorized with Drive+Sheets scopes."""
    try:
        import gspread
        from google.oauth2 import service_account
    except ImportError as exc:
        raise RuntimeError(
            "gspread and google-auth are required. Install requirements.txt first."
        ) from exc

    if config.service_account_info is not None:
        credentials = service_account.Credentials.from_service_account_info(
            info=config.service_account_info,
            scopes=list(_DRIVE_SCOPES),
        )
    elif config.service_account_path is not None:
        credentials = service_account.Credentials.from_service_account_file(
            filename=str(config.service_account_path),
            scopes=list(_DRIVE_SCOPES),
        )
    else:
        raise ValueError("SheetsConfig must define service_account_path or service_account_info")

    from experiments.ctgan.orchestrator_staff.ctgan_sheets import SheetsClient, retry_call

    client = gspread.authorize(credentials)
    spreadsheet = retry_call(lambda: client.open_by_key(config.spreadsheet_id))
    if config.worksheet_name:
        worksheet = retry_call(lambda: spreadsheet.worksheet(config.worksheet_name))
    else:
        worksheet = retry_call(lambda: spreadsheet.sheet1)

    return SheetsClient(config, worksheet=worksheet)


# ---------------------------------------------------------------------------
# MIME type helper
# ---------------------------------------------------------------------------


def _guess_mime_type(path: Path) -> str:
    _MAP = {
        ".json": "application/json",
        ".csv": "text/csv",
        ".txt": "text/plain",
        ".png": "image/png",
        ".svg": "image/svg+xml",
        ".pdf": "application/pdf",
        ".pkl": "application/octet-stream",
        ".pt": "application/octet-stream",
        ".pth": "application/octet-stream",
        ".ckpt": "application/octet-stream",
        ".safetensors": "application/octet-stream",
    }
    return _MAP.get(path.suffix.lower(), "application/octet-stream")
