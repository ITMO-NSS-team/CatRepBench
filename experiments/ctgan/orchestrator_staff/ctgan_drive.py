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

    def _find_or_create_folder(self, name: str, parent_id: str) -> str:
        """Return folder ID, creating it if it doesn't exist."""
        query = (
            f"name = {json.dumps(name)} "
            f"and '{parent_id}' in parents "
            f"and mimeType = 'application/vnd.google-apps.folder' "
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
            "mimeType": "application/vnd.google-apps.folder",
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
        ".pt", ".ckpt", ".safetensors", ".pth",
        # results / metrics
        ".json",
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


# ---------------------------------------------------------------------------
# Results sheet writer
# ---------------------------------------------------------------------------


def write_results_row(
    *,
    sheets_config: Any,  # ctgan_sheets.SheetsConfig
    aggregate_metrics_path: Path,
    model_name: str,
    dataset_label: str,
    encoding_label: str,
    folder_url: str,
    results_worksheet_name: str = "Results",
) -> None:
    """Append a metrics row to the *results_worksheet_name* sheet.

    Expected sheet layout (from the screenshot):
        A: Model
        B: Dataset
        C: Categorical representation
        D: Mean WD – Mean
        E: Mean WD – Std
        F: Mean KL – Mean
        G: Mean KL – Std
        H: Corr dist – Mean
        I: Corr dist – Std
        J: Ссылка на материалы

    The function finds the first empty row (by checking column A) and
    writes to that row.
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

    dist = aggregate.get("distribution", {})

    def _mean_std(key: str) -> tuple[str, str]:
        entry = dist.get(key, {})
        if isinstance(entry, dict):
            return str(round(entry.get("mean", ""), 6)), str(round(entry.get("std", ""), 6))
        return "", ""

    wd_mean, wd_std = _mean_std("wasserstein_mean")
    kl_mean, kl_std = _mean_std("marginal_kl_mean")
    # Use transformed corr_frobenius as "Corr dist"
    corr_mean, corr_std = _mean_std("corr_frobenius_transformed")

    row_values = [
        model_name,
        dataset_label,
        encoding_label,
        wd_mean,
        wd_std,
        kl_mean,
        kl_std,
        corr_mean,
        corr_std,
        folder_url,
    ]

    matrix = results_client.read_matrix()
    # Sheet has a 2-row merged header (rows 1-2), data starts at row 3.
    # Find the first empty row at or after row 3 (index 2).
    first_empty_row = max(len(matrix) + 1, 3)  # default: append after all existing rows
    for row_idx, row in enumerate(matrix):
        if row_idx < 2:
            continue  # skip the two header rows
        cell_a = row[0] if row else ""
        if not cell_a.strip():
            first_empty_row = row_idx + 1  # 1-based sheet row
            break

    # Write each cell individually (SheetsClient.write_cell uses A1 notation)
    col_letters = list("ABCDEFGHIJ")
    for col_idx, (col_letter, value) in enumerate(zip(col_letters, row_values)):
        coord = f"{col_letter}{first_empty_row}"
        results_client.write_cell(coord, value)

    print(
        f"[ctgan_drive] Wrote results row to sheet '{results_worksheet_name}' "
        f"at row {first_empty_row}."
    )


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
        ".pt": "application/octet-stream",
        ".pth": "application/octet-stream",
        ".ckpt": "application/octet-stream",
        ".safetensors": "application/octet-stream",
    }
    return _MAP.get(path.suffix.lower(), "application/octet-stream")
