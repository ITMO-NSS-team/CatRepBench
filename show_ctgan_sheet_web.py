from __future__ import annotations

import argparse
import csv
import datetime as dt
import errno
import json
import os
import sqlite3
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "experiments/ctgan/orchestrator_staff/ctgan_orchestrator_manifest.json"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "experiments/results/recomputed_ctgan"


def _load_dotenv() -> None:
    """Load .env into os.environ (only keys not already set). Handles single-quoted JSON values."""
    for path in (SCRIPT_DIR / ".env", Path.cwd() / ".env"):
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8")
        i, lines = 0, content.splitlines()
        while i < len(lines):
            stripped = lines[i].strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                i += 1; continue
            key, _, rest = stripped.partition("=")
            key = key.strip()
            if rest.startswith("'"):
                if rest.endswith("'") and len(rest) > 1:
                    val = rest[1:-1]
                else:
                    parts = [rest[1:]]
                    i += 1
                    while i < len(lines):
                        if lines[i].endswith("'"):
                            parts.append(lines[i][:-1]); break
                        parts.append(lines[i]); i += 1
                    val = "\n".join(parts)
            elif rest.startswith('"') and rest.endswith('"') and len(rest) > 1:
                val = rest[1:-1]
            else:
                val = rest.split("#")[0].strip()
            if key and key not in os.environ:
                os.environ[key] = val
            i += 1
        return


def _repair_invalid_ca_bundle_env() -> None:
    """Replace stale Linux CA bundle paths from .env with certifi when available."""
    ca_keys = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")
    invalid_keys = [key for key in ca_keys if os.environ.get(key) and not Path(os.environ[key]).exists()]
    if not invalid_keys:
        return
    try:
        import certifi
    except Exception:
        for key in invalid_keys:
            os.environ.pop(key, None)
        return
    ca_bundle = certifi.where()
    for key in invalid_keys:
        os.environ[key] = ca_bundle


_load_dotenv()
_repair_invalid_ca_bundle_env()

_SHEETS_TIMEOUT = 12   # seconds before giving up on a Sheets request

from experiments.ctgan.orchestrator_staff.ctgan_manifest import load_ctgan_manifest
from experiments.ctgan.orchestrator_staff.ctgan_sheets import SheetsClient, SheetsConfig
from experiments.ctgan.orchestrator_staff.ctgan_orchestrator_state import (
    parse_cell_payload,
    validate_worksheet_headers,
)


@dataclass(frozen=True)
class WorksheetSnapshot:
    dataset_headers: tuple[str, ...]
    encoding_headers: tuple[str, ...]
    cell_values: dict[str, str | None]
    coord_labels: dict[str, tuple[str, str]]


def _load_snapshot(matrix: Any, *, manifest) -> WorksheetSnapshot:
    dataset_labels = tuple(entry.label for entry in manifest.datasets)
    encoding_labels = tuple(entry.label for entry in manifest.encodings)

    if isinstance(matrix, dict):
        dataset_headers, encoding_headers = validate_worksheet_headers(
            dataset_headers=matrix["dataset_headers"],
            encoding_headers=matrix["encoding_headers"],
            manifest_dataset_labels=dataset_labels,
            manifest_encoding_labels=encoding_labels,
        )
        cell_values = dict(matrix["cell_values"])
        return WorksheetSnapshot(
            dataset_headers=dataset_headers,
            encoding_headers=encoding_headers,
            cell_values=cell_values,
            coord_labels=_build_coord_labels(dataset_headers, encoding_headers),
        )

    if not isinstance(matrix, list) or not matrix:
        raise ValueError("worksheet matrix must be a non-empty grid.")

    header_row = matrix[0]
    if not isinstance(header_row, list):
        raise ValueError("worksheet matrix header row must be a list.")

    dataset_headers, encoding_headers = validate_worksheet_headers(
        dataset_headers=header_row[1:],
        encoding_headers=[row[0] if isinstance(row, list) and row else "" for row in matrix[1:]],
        manifest_dataset_labels=dataset_labels,
        manifest_encoding_labels=encoding_labels,
    )
    cell_values: dict[str, str | None] = {}
    for row_index, _encoding_label in enumerate(encoding_headers, start=1):
        row = matrix[row_index] if row_index < len(matrix) and isinstance(matrix[row_index], list) else []
        for col_index, _dataset_label in enumerate(dataset_headers, start=1):
            coord = f"{_column_name(col_index + 1)}{row_index + 1}"
            cell_values[coord] = row[col_index] if col_index < len(row) else None
    return WorksheetSnapshot(
        dataset_headers=dataset_headers,
        encoding_headers=encoding_headers,
        cell_values=cell_values,
        coord_labels=_build_coord_labels(dataset_headers, encoding_headers),
    )


def _build_coord_labels(
    dataset_headers: tuple[str, ...],
    encoding_headers: tuple[str, ...],
) -> dict[str, tuple[str, str]]:
    coord_labels: dict[str, tuple[str, str]] = {}
    for encoding_offset, encoding_label in enumerate(encoding_headers, start=2):
        for dataset_offset, dataset_label in enumerate(dataset_headers, start=2):
            coord_labels[f"{_column_name(dataset_offset)}{encoding_offset}"] = (
                dataset_label,
                encoding_label,
            )
    return coord_labels

STATUS_ORDER = (
    "done",
    "in-progress",
    "stale-in-progress",
    "failed",
    "skipped",
    "not-started",
)

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iso_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _parse_iso_datetime(value: Any) -> dt.datetime | None:
    if value in {None, ""}:
        return None
    if isinstance(value, dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)
    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        parsed = dt.datetime.fromisoformat(normalized)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)
    return None


def annotate_cell_state(cell: dict[str, Any], *, now: dt.datetime | None = None,
                        stale_after_seconds: int = 60) -> dict[str, Any]:
    annotated = dict(cell)
    raw_status = annotated.get("status", "not-started")
    annotated["effective_status"] = raw_status
    annotated["is_stale"] = False
    annotated["stale_age_seconds"] = None
    if raw_status != "in-progress":
        return annotated
    now = now or dt.datetime.now(dt.timezone.utc)
    last_update = (_parse_iso_datetime(annotated.get("heartbeat_at"))
                   or _parse_iso_datetime(annotated.get("started_at")))
    if last_update is None:
        return annotated
    age_seconds = max(0.0, (now - last_update).total_seconds())
    annotated["stale_age_seconds"] = int(age_seconds)
    if age_seconds > stale_after_seconds:
        annotated["is_stale"] = True
        annotated["effective_status"] = "stale-in-progress"
    return annotated


def _column_name(index: int) -> str:
    name = ""
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        name = chr(ord("A") + remainder) + name
    return name


# ---------------------------------------------------------------------------
# Progress tab payload
# ---------------------------------------------------------------------------

def _sheets_read(worksheet_name: str | None) -> tuple[list, str, str, str]:
    """Fetch matrix from Sheets. Returns (matrix, ws_title, ss_title, ss_url)."""
    config = SheetsConfig.from_env()
    if worksheet_name:
        config = replace(config, worksheet_name=worksheet_name)
    client = SheetsClient(config)
    matrix = client.read_matrix()
    ws_title = client._worksheet.title
    ss_title = client._worksheet.spreadsheet.title
    ss_url = client._worksheet.spreadsheet.url
    return matrix, ws_title, ss_title, ss_url


def build_status_payload(*, manifest_path: Path, worksheet_name: str | None) -> dict[str, Any]:
    manifest = load_ctgan_manifest(manifest_path, project_root=manifest_path.resolve().parents[1])

    matrix, ws_title, ss_title, ss_url = [], "", "", ""
    sheets_error: str | None = None
    try:
        matrix, ws_title, ss_title, ss_url = _sheets_read(worksheet_name)
        print("[sheets] Status sheet loaded", flush=True)
    except Exception as e:
        sheets_error = str(e)
        print(f"[sheets] Status error: {e}", file=sys.stderr, flush=True)

    if not matrix:
        return {"error": sheets_error or "Google Sheets returned an empty matrix.", "rows": [], "counts": {}, "ss_title": "", "ss_url": ""}

    snapshot = _load_snapshot(matrix, manifest=manifest)

    rows: list[dict[str, Any]] = []
    counts = {status: 0 for status in STATUS_ORDER}
    now = dt.datetime.now(dt.timezone.utc)

    for encoding_idx, encoding_label in enumerate(snapshot.encoding_headers, start=2):
        row_cells: list[dict[str, Any]] = []
        for dataset_idx, dataset_label in enumerate(snapshot.dataset_headers, start=2):
            coord = f"{_column_name(dataset_idx)}{encoding_idx}"
            payload = parse_cell_payload(snapshot.cell_values.get(coord))
            cell = annotate_cell_state({
                "coord": coord,
                "dataset": dataset_label,
                "encoding": encoding_label,
                "status": payload.status,
                "stage": payload.stage,
                "note": payload.note,
                "owner": payload.owner,
                "run_id": payload.run_id,
                "started_at": _iso_or_none(payload.started_at),
                "heartbeat_at": _iso_or_none(payload.heartbeat_at),
                "finished_at": _iso_or_none(payload.finished_at),
            }, now=now)
            counts[cell["effective_status"]] = counts.get(cell["effective_status"], 0) + 1
            row_cells.append(cell)
        rows.append({"encoding": encoding_label, "cells": row_cells})

    details: list[dict[str, Any]] = []
    for row in rows:
        for cell in row["cells"]:
            if cell["effective_status"] in {"in-progress", "stale-in-progress", "failed"}:
                details.append(cell)

    return {
        "spreadsheet_title": ss_title,
        "spreadsheet_url": ss_url,
        "worksheet_name": ws_title,
        "dataset_headers": list(snapshot.dataset_headers),
        "rows": rows,
        "counts": counts,
        "details": details,
    }


def skipped_dataset_ids_from_status_payload(status_payload: dict[str, Any], *, manifest) -> list[str]:
    dataset_ids_by_label = {entry.label: entry.dataset_id for entry in manifest.datasets}
    skipped_dataset_ids: set[str] = set()
    for row in status_payload.get("rows", []):
        for cell in row.get("cells", []):
            if cell.get("status") != "skipped" and cell.get("effective_status") != "skipped":
                continue
            dataset_id = dataset_ids_by_label.get(str(cell.get("dataset", "")).strip())
            if dataset_id:
                skipped_dataset_ids.add(dataset_id)
    return [entry.dataset_id for entry in manifest.datasets if entry.dataset_id in skipped_dataset_ids]


# ---------------------------------------------------------------------------
# Results tab payload — reads Results sheet + local JSON artifacts
# ---------------------------------------------------------------------------

def _safe_float(v: Any) -> float | None:
    try:
        f = float(v)
        return f if f == f else None  # NaN check
    except (TypeError, ValueError):
        return None


_DRIVE_SCOPES = (
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/spreadsheets.readonly",
)
_DRIVE_FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
_DRIVE_CACHE_MANIFEST = ".drive_artifacts_manifest.json"


@dataclass(frozen=True)
class DriveFileRecord:
    file_id: str
    relative_path: str
    modified_time: str
    mime_type: str = ""


class DriveArtifactClient:
    def __init__(self) -> None:
        self._service = self._build_service()

    @staticmethod
    def _build_credentials() -> Any:
        oauth_token_path = os.getenv("CATREPBENCH_GDRIVE_OAUTH_TOKEN_PATH", "").strip()
        service_account_path = os.getenv("CATREPBENCH_GSHEETS_SERVICE_ACCOUNT_PATH", "").strip()
        service_account_json = os.getenv("CATREPBENCH_GSHEETS_SERVICE_ACCOUNT_JSON", "").strip()

        if oauth_token_path:
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request

            token_path = Path(oauth_token_path).expanduser()
            credentials = Credentials.from_authorized_user_file(str(token_path))
            if credentials.expired and credentials.refresh_token:
                credentials.refresh(Request())
                token_path.write_text(credentials.to_json(), encoding="utf-8")
            return credentials

        from google.oauth2 import service_account

        if service_account_json:
            return service_account.Credentials.from_service_account_info(
                json.loads(service_account_json),
                scopes=_DRIVE_SCOPES,
            )
        if service_account_path:
            return service_account.Credentials.from_service_account_file(
                str(Path(service_account_path).expanduser()),
                scopes=_DRIVE_SCOPES,
            )
        raise ValueError(
            "Drive artifact sync needs CATREPBENCH_GDRIVE_OAUTH_TOKEN_PATH, "
            "CATREPBENCH_GSHEETS_SERVICE_ACCOUNT_PATH, or "
            "CATREPBENCH_GSHEETS_SERVICE_ACCOUNT_JSON."
        )

    @classmethod
    def _build_service(cls) -> Any:
        from googleapiclient.discovery import build

        return build("drive", "v3", credentials=cls._build_credentials(), cache_discovery=False)

    def list_files_recursive(self, folder_id: str) -> list[DriveFileRecord]:
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
            page = (
                self._service.files()
                .list(
                    q=f"'{folder_id}' in parents and trashed = false",
                    fields="nextPageToken,files(id,name,mimeType,modifiedTime)",
                    pageToken=page_token,
                    spaces="drive",
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )
            for item in page.get("files", []):
                name = str(item["name"])
                mime_type = str(item.get("mimeType", ""))
                file_id = str(item["id"])
                relative_path = f"{relative_prefix}/{name}" if relative_prefix else name
                if mime_type == _DRIVE_FOLDER_MIME_TYPE:
                    self._list_files_recursive_into(
                        folder_id=file_id,
                        relative_prefix=relative_path,
                        records=records,
                    )
                    continue
                records.append(
                    DriveFileRecord(
                        file_id=file_id,
                        relative_path=relative_path,
                        modified_time=str(item.get("modifiedTime", "")),
                        mime_type=mime_type,
                    )
                )
            page_token = page.get("nextPageToken")
            if not page_token:
                break

    def download_file(self, file_id: str, destination: Path) -> None:
        from googleapiclient.http import MediaIoBaseDownload

        destination.parent.mkdir(parents=True, exist_ok=True)
        request = self._service.files().get_media(fileId=file_id, supportsAllDrives=True)
        with destination.open("wb") as handle:
            downloader = MediaIoBaseDownload(handle, request)
            done = False
            while not done:
                _status, done = downloader.next_chunk()


_DRIVE_ARTIFACT_CLIENT: DriveArtifactClient | None = None
_DRIVE_ARTIFACT_CLIENT_FAILED = False


def _get_drive_artifact_client() -> DriveArtifactClient | None:
    global _DRIVE_ARTIFACT_CLIENT, _DRIVE_ARTIFACT_CLIENT_FAILED
    if _DRIVE_ARTIFACT_CLIENT_FAILED:
        return None
    if _DRIVE_ARTIFACT_CLIENT is not None:
        return _DRIVE_ARTIFACT_CLIENT
    if not (
        os.getenv("CATREPBENCH_GDRIVE_OAUTH_TOKEN_PATH")
        or os.getenv("CATREPBENCH_GSHEETS_SERVICE_ACCOUNT_PATH")
        or os.getenv("CATREPBENCH_GSHEETS_SERVICE_ACCOUNT_JSON")
    ):
        _DRIVE_ARTIFACT_CLIENT_FAILED = True
        return None
    try:
        _DRIVE_ARTIFACT_CLIENT = DriveArtifactClient()
        return _DRIVE_ARTIFACT_CLIENT
    except Exception as exc:
        _DRIVE_ARTIFACT_CLIENT_FAILED = True
        print(f"[drive] Artifact sync disabled: {exc}", file=sys.stderr, flush=True)
        return None


class DriveSyncState:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _connection(self):
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sync_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sync_cells (
                    dataset_id TEXT NOT NULL,
                    encoding_id TEXT NOT NULL,
                    folder_url TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'pending',
                    downloaded_files INTEGER NOT NULL DEFAULT 0,
                    error TEXT NOT NULL DEFAULT '',
                    started_at TEXT,
                    finished_at TEXT,
                    PRIMARY KEY (dataset_id, encoding_id)
                )
                """
            )

    @staticmethod
    def _now() -> str:
        return dt.datetime.now(dt.timezone.utc).isoformat()

    def _set_meta(self, conn: sqlite3.Connection, **values: str) -> None:
        for key, value in values.items():
            conn.execute(
                "INSERT OR REPLACE INTO sync_meta (key, value) VALUES (?, ?)",
                (key, value),
            )

    def start_run(self) -> None:
        now = self._now()
        with self._connection() as conn:
            self._set_meta(conn, status="running", started_at=now, heartbeat_at=now, error="")

    def finish_run(self, *, status: str, error: str) -> None:
        now = self._now()
        with self._connection() as conn:
            self._set_meta(conn, status=status, finished_at=now, heartbeat_at=now, error=error)

    def record_cell_start(self, *, dataset_id: str, encoding_id: str, folder_url: str) -> None:
        now = self._now()
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO sync_cells (
                    dataset_id, encoding_id, folder_url, status, downloaded_files,
                    error, started_at, finished_at
                )
                VALUES (?, ?, ?, 'running', 0, '', ?, NULL)
                ON CONFLICT(dataset_id, encoding_id) DO UPDATE SET
                    folder_url=excluded.folder_url,
                    status='running',
                    error='',
                    started_at=excluded.started_at,
                    finished_at=NULL
                """,
                (dataset_id, encoding_id, folder_url, now),
            )
            self._set_meta(conn, heartbeat_at=now)

    def record_cell_finish(
        self,
        *,
        dataset_id: str,
        encoding_id: str,
        status: str,
        downloaded_files: int,
        error: str,
    ) -> None:
        now = self._now()
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE sync_cells
                SET status=?, downloaded_files=downloaded_files + ?, error=?, finished_at=?
                WHERE dataset_id=? AND encoding_id=?
                """,
                (status, int(downloaded_files), error, now, dataset_id, encoding_id),
            )
            self._set_meta(conn, heartbeat_at=now)

    def summary(self) -> dict[str, Any]:
        with self._connection() as conn:
            meta = {row["key"]: row["value"] for row in conn.execute("SELECT key, value FROM sync_meta")}
            cells = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT dataset_id, encoding_id, folder_url, status, downloaded_files,
                           error, started_at, finished_at
                    FROM sync_cells
                    ORDER BY COALESCE(finished_at, started_at) DESC, dataset_id, encoding_id
                    LIMIT 25
                    """
                )
            ]
            total_downloaded = conn.execute(
                "SELECT COALESCE(SUM(downloaded_files), 0) AS total FROM sync_cells"
            ).fetchone()["total"]
        return {
            "status": meta.get("status", "idle"),
            "started_at": meta.get("started_at"),
            "finished_at": meta.get("finished_at"),
            "heartbeat_at": meta.get("heartbeat_at"),
            "error": meta.get("error", ""),
            "total_downloaded_files": int(total_downloaded or 0),
            "cells": cells,
        }


def _drive_folder_id_from_url(url: str) -> str | None:
    parsed = urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    if "folders" in parts:
        index = parts.index("folders")
        if index + 1 < len(parts):
            return parts[index + 1]
    query_id = parse_qs(parsed.query).get("id", [""])[0]
    return query_id or None


def _is_drive_artifact_path(relative_path: str) -> bool:
    normalized = relative_path.strip("/")
    if normalized in {
        "run_summary.json",
        "source_drive/run_summary.json",
        "metrics/aggregate.json",
        "tuning/summary.json",
        "tuning/best_params.json",
    }:
        return True
    if normalized.startswith("crossval/per_fold/") and Path(normalized).name.startswith("fold_"):
        return normalized.endswith(".json")
    if normalized.startswith("fold_details/") and Path(normalized).name.startswith("fold_"):
        return normalized.endswith(".json")
    if normalized.startswith("artifacts/") and normalized.endswith("/loss_history.csv"):
        return True
    return False


def _record_attr(record: Any, name: str) -> str:
    if isinstance(record, dict):
        return str(record.get(name, ""))
    return str(getattr(record, name, ""))


def _latest_drive_records_by_path(records: list[Any]) -> dict[str, Any]:
    latest: dict[str, Any] = {}
    for record in records:
        relative_path = _record_attr(record, "relative_path")
        if not _is_drive_artifact_path(relative_path):
            continue
        current = latest.get(relative_path)
        if current is None or _record_attr(record, "modified_time") > _record_attr(current, "modified_time"):
            latest[relative_path] = record
    return latest


def _read_drive_cache_manifest(run_dir: Path) -> dict[str, dict[str, str]]:
    manifest_path = run_dir / _DRIVE_CACHE_MANIFEST
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_drive_cache_manifest(run_dir: Path, payload: dict[str, dict[str, str]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / _DRIVE_CACHE_MANIFEST).write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _sync_referenced_source_run_summary(
    *,
    drive_client: Any,
    run_dir: Path,
    manifest: dict[str, dict[str, str]],
) -> bool:
    root_summary_path = run_dir / "run_summary.json"
    if not root_summary_path.exists():
        return False
    try:
        root_summary = json.loads(root_summary_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    source_drive = root_summary.get("source_drive") or {}
    if not isinstance(source_drive, dict):
        return False
    file_id = str(source_drive.get("run_summary_file_id") or "").strip()
    if not file_id:
        return False
    modified_time = str(source_drive.get("run_summary_modified_time") or "")
    relative_path = "source_drive/run_summary.json"
    destination = run_dir / relative_path
    cached = manifest.get(relative_path, {})
    if (
        destination.exists()
        and cached.get("file_id") == file_id
        and cached.get("modified_time") == modified_time
    ):
        return False

    drive_client.download_file(file_id, destination)
    manifest[relative_path] = {"file_id": file_id, "modified_time": modified_time}
    return True


def _sync_drive_artifacts(*, drive_client: Any, folder_url: str, run_dir: Path) -> dict[str, int]:
    folder_id = _drive_folder_id_from_url(folder_url)
    if not folder_id:
        return {"downloaded": 0, "seen": 0}
    records = _latest_drive_records_by_path(drive_client.list_files_recursive(folder_id))
    if not records:
        return {"downloaded": 0, "seen": 0}

    manifest = _read_drive_cache_manifest(run_dir)
    changed = False
    downloaded = 0
    for relative_path, record in records.items():
        file_id = _record_attr(record, "file_id")
        modified_time = _record_attr(record, "modified_time")
        destination = run_dir / relative_path
        cached = manifest.get(relative_path, {})
        if (
            destination.exists()
            and cached.get("file_id") == file_id
            and cached.get("modified_time") == modified_time
        ):
            continue
        drive_client.download_file(file_id, destination)
        manifest[relative_path] = {"file_id": file_id, "modified_time": modified_time}
        changed = True
        downloaded += 1
    if _sync_referenced_source_run_summary(
        drive_client=drive_client,
        run_dir=run_dir,
        manifest=manifest,
    ):
        changed = True
        downloaded += 1
    if changed:
        _write_drive_cache_manifest(run_dir, manifest)
    return {"downloaded": downloaded, "seen": len(records)}


def _load_run_summary(path: Path) -> dict[str, Any]:
    """Load run_summary.json from source_drive/ subdir if present."""
    candidates = [
        path / "source_drive" / "run_summary.json",
        path / "run_summary.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
    return {}


def _load_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_tuning_info(run_dir: Path, run_summary: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        run_summary.get("tuning") or {},
        _load_json_file(run_dir / "tuning" / "summary.json"),
        _load_json_file(run_dir / "tuning" / "best_params.json"),
    ]
    for tuning in candidates:
        if not isinstance(tuning, dict):
            continue
        if not any(key in tuning for key in ("best_value", "best_source", "best_params")):
            continue
        return {
            "best_value": _safe_float(tuning.get("best_value")),
            "best_source": tuning.get("best_source", ""),
            "best_params": tuning.get("best_params", {}),
        }
    return {}


def _dataset_schema_info(run_summary: dict[str, Any]) -> dict[str, Any]:
    """Extract cat/cont/disc counts and cat_share from run_summary schema."""
    schema = run_summary.get("schema", {})
    cat_cols = schema.get("categorical_cols", [])
    cont_cols = schema.get("continuous_cols", [])
    disc_cols = schema.get("discrete_cols", [])
    n_cat = len(cat_cols)
    n_cont = len(cont_cols)
    n_disc = len(disc_cols)
    n_total = n_cat + n_cont + n_disc
    cat_share = (n_cat / n_total) if n_total > 0 else 0.0
    return {
        "n_cat": n_cat,
        "n_cont": n_cont,
        "n_disc": n_disc,
        "n_total": n_total,
        "cat_share": round(cat_share, 4),
        "n_rows": run_summary.get("poster_fast", {}).get("effective_rows"),
        "n_folds": run_summary.get("crossval", {}).get("n_folds"),
        "target_col": schema.get("target_col"),
    }


def _parse_sheet_rows(matrix: list) -> tuple[list[str], list[dict[str, Any]]]:
    if not matrix or len(matrix) < 2:
        return [], []
    headers = [str(h) for h in matrix[0]]
    sheet_rows = []
    for row in matrix[1:]:
        padded = list(row) + [""] * max(0, len(headers) - len(row))
        sheet_rows.append(dict(zip(headers, padded)))
    return headers, sheet_rows


def build_results_payload(
    *,
    manifest_path: Path,
    results_worksheet: str,
    output_root: Path,
    sync_drive: bool = True,
    rank_excluded_dataset_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Read Results sheet + local per-fold JSONs, loss CSVs, and run_summary.json."""
    matrix: list = []
    if results_worksheet:
        try:
            matrix, _, _, _ = _sheets_read(results_worksheet)
            print("[sheets] Results sheet loaded", flush=True)
        except Exception as e:
            print(f"[sheets] Results error: {e}", file=sys.stderr, flush=True)

    manifest = load_ctgan_manifest(manifest_path, project_root=manifest_path.resolve().parents[1])
    dataset_order = [d.dataset_id for d in manifest.datasets]
    dataset_labels = {d.dataset_id: d.label for d in manifest.datasets}
    encoding_order = [e.encoding_id for e in manifest.encodings]
    encoding_labels = {e.encoding_id: e.label for e in manifest.encodings}

    headers, sheet_rows = _parse_sheet_rows(matrix)

    # Index by (dataset_label, encoding_label)
    sheet_index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in sheet_rows:
        ds = str(row.get("Dataset", "")).strip()
        enc = str(row.get("Categorical representation", "")).strip()
        if ds and enc:
            sheet_index[(ds, enc)] = row

    drive_client = (
        _get_drive_artifact_client()
        if sync_drive and any(row.get("Drive folder URL") for row in sheet_rows)
        else None
    )

    # Collect per-dataset schema info (from any available run_summary)
    dataset_schema_cache: dict[str, dict[str, Any]] = {}

    # Build per-cell data
    cells: list[dict[str, Any]] = []
    for dataset_id in dataset_order:
        ds_label = dataset_labels.get(dataset_id, dataset_id)
        for enc_id in encoding_order:
            enc_label = encoding_labels.get(enc_id, enc_id)
            sheet_row = sheet_index.get((ds_label, enc_label), {})

            # Local artifact paths
            run_dir = output_root / dataset_id / enc_id
            aggregate_path = run_dir / "metrics" / "aggregate.json"
            per_fold_dir = run_dir / "crossval" / "per_fold"
            fold_details_dir = run_dir / "fold_details"
            artifacts_dir = run_dir / "artifacts"

            drive_url = str(sheet_row.get("Drive folder URL", "")).strip()
            if drive_client is not None and drive_url:
                try:
                    _sync_drive_artifacts(
                        drive_client=drive_client,
                        folder_url=drive_url,
                        run_dir=run_dir,
                    )
                except Exception as exc:
                    print(
                        f"[drive] Artifact sync error for {ds_label} / {enc_label}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )

            # Load run_summary.json (source_drive or local)
            run_summary = _load_run_summary(run_dir)

            # Cache schema info per dataset
            if dataset_id not in dataset_schema_cache and run_summary:
                dataset_schema_cache[dataset_id] = _dataset_schema_info(run_summary)

            aggregate: dict[str, Any] = {}
            if aggregate_path.exists():
                try:
                    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
                except Exception:
                    pass

            # Per-fold metrics
            fold_data: list[dict[str, Any]] = []
            if per_fold_dir.exists():
                for fold_file in sorted(per_fold_dir.glob("fold_*.json")):
                    try:
                        fold_data.append(json.loads(fold_file.read_text(encoding="utf-8")))
                    except Exception:
                        pass

            fold_details_by_id: dict[int, dict[str, Any]] = {}
            if fold_details_dir.exists():
                for detail_file in sorted(fold_details_dir.glob("fold_*.json")):
                    try:
                        detail = json.loads(detail_file.read_text(encoding="utf-8"))
                        fold_id = int(detail.get("fold_id", detail_file.stem.replace("fold_", "")))
                        fold_details_by_id[fold_id] = detail
                    except Exception:
                        pass

            # Loss history per fold
            loss_histories: dict[int, dict[str, list[float]]] = {}
            if artifacts_dir.exists():
                for fold_dir in sorted(artifacts_dir.iterdir()):
                    loss_csv = fold_dir / "loss_history.csv"
                    if not loss_csv.exists():
                        continue
                    try:
                        fold_id = int(fold_dir.name.replace("fold_", ""))
                        gen_losses, disc_losses = [], []
                        with loss_csv.open(encoding="utf-8") as f:
                            reader = csv.DictReader(f)
                            for r in reader:
                                g = _safe_float(r.get("generator_loss"))
                                d = _safe_float(r.get("discriminator_loss"))
                                if g is not None:
                                    gen_losses.append(g)
                                if d is not None:
                                    disc_losses.append(d)
                        loss_histories[fold_id] = {
                            "generator": gen_losses,
                            "discriminator": disc_losses,
                        }
                    except Exception:
                        pass

            # Pull metrics from aggregate or sheet
            def _agg(key: str, field: str = "mean") -> float | None:
                dist = aggregate.get("distribution", {})
                if key in dist:
                    v = dist[key]
                    return _safe_float(v.get(field) if isinstance(v, dict) else v)
                return _safe_float(sheet_row.get(
                    {"mean": "Mean", "std": "Std"}.get(field, field) + " " + {
                        "wasserstein_mean_unencoded": "WD",
                        "marginal_kl_mean_unencoded": "KL",
                        "corr_frobenius_unencoded": "Corr dist",
                    }.get(key, key), ""
                ))

            def _tstr(field: str = "mean") -> float | None:
                # Prefer local aggregate.json
                tstr_agg = aggregate.get("tstr", {})
                metrics_agg = tstr_agg.get("metrics", {})
                for key in ("f1_weighted_pct_diff", "r2_pct_diff"):
                    if key in metrics_agg:
                        v = metrics_agg[key]
                        return _safe_float(v.get(field) if isinstance(v, dict) else v)
                # Fallback: run_summary.json tstr
                tstr_rs = run_summary.get("tstr", {})
                metrics_rs = tstr_rs.get("metrics", {})
                for key in ("f1_weighted_pct_diff", "r2_pct_diff"):
                    if key in metrics_rs:
                        v = metrics_rs[key]
                        return _safe_float(v.get(field) if isinstance(v, dict) else v)
                # Fallback: sheet
                col = ("Mean" if field == "mean" else "Std") + " Utility"
                return _safe_float(sheet_row.get(col))

            def _tstr_real_synth(field: str = "mean") -> tuple[float | None, float | None]:
                """Return (f1_real, f1_synth) or (r2_real, r2_synth) mean values."""
                for src in (aggregate.get("tstr", {}), run_summary.get("tstr", {})):
                    metrics = src.get("metrics", {})
                    if "f1_weighted_real" in metrics and "f1_weighted_synth" in metrics:
                        real = metrics["f1_weighted_real"]
                        synth = metrics["f1_weighted_synth"]
                        r = _safe_float(real.get(field) if isinstance(real, dict) else real)
                        s = _safe_float(synth.get(field) if isinstance(synth, dict) else synth)
                        return r, s
                    if "r2_real" in metrics and "r2_synth" in metrics:
                        real = metrics["r2_real"]
                        synth = metrics["r2_synth"]
                        r = _safe_float(real.get(field) if isinstance(real, dict) else real)
                        s = _safe_float(synth.get(field) if isinstance(synth, dict) else synth)
                        return r, s
                return None, None

            task_type = (
                aggregate.get("tstr", {}).get("task_type")
                or run_summary.get("tstr", {}).get("task_type")
                or sheet_row.get("Task Type") or ""
            )

            tuning_info = _load_tuning_info(run_dir, run_summary)

            # Correlation matrices per fold
            corr_per_fold: list[dict[str, Any]] = []
            for fold in fold_data:
                dist = fold.get("distribution", {})
                util = fold.get("utility") or {}
                corr_per_fold.append({
                    "fold_id": fold.get("fold_id"),
                    "corr_frobenius_unencoded": _safe_float(dist.get("corr_frobenius_unencoded")),
                    "corr_frobenius_transformed": _safe_float(dist.get("corr_frobenius_transformed")),
                    "corr_frobenius_original": _safe_float(dist.get("corr_frobenius_original")),
                    "corr_frobenius_original_status": dist.get("corr_frobenius_original_status"),
                    "wd_unencoded": _safe_float(dist.get("wasserstein_mean_unencoded")),
                    "wd_transformed": _safe_float(dist.get("wasserstein_mean")),
                    "kl_unencoded": _safe_float(dist.get("marginal_kl_mean_unencoded")),
                    "kl_transformed": _safe_float(dist.get("marginal_kl_mean")),
                    "f1_real": _safe_float(util.get("f1_weighted_real")),
                    "f1_synth": _safe_float(util.get("f1_weighted_synth")),
                    "f1_pct_diff": _safe_float(util.get("f1_weighted_pct_diff")),
                    "r2_real": _safe_float(util.get("r2_real")),
                    "r2_synth": _safe_float(util.get("r2_synth")),
                    "r2_pct_diff": _safe_float(util.get("r2_pct_diff")),
                    "utility_status": util.get("status"),
                    "task_type": util.get("task_type"),
                    "n_train": fold.get("n_train"),
                    "n_test": fold.get("n_test"),
                    "all_metrics": fold,
                    "detail": fold_details_by_id.get(int(fold.get("fold_id", -1))),
                })

            f1_real_mean, f1_synth_mean = _tstr_real_synth("mean")
            wd_mean = _agg("wasserstein_mean_unencoded", "mean")
            wd_std = _agg("wasserstein_mean_unencoded", "std")
            kl_mean = _agg("marginal_kl_mean_unencoded", "mean")
            kl_std = _agg("marginal_kl_mean_unencoded", "std")
            corr_mean = _agg("corr_frobenius_unencoded", "mean")
            corr_std = _agg("corr_frobenius_unencoded", "std")
            has_data = wd_mean is not None and kl_mean is not None and corr_mean is not None
            utility_mean = _tstr("mean") if has_data else None
            utility_std = _tstr("std") if has_data else None
            if not has_data:
                f1_real_mean, f1_synth_mean = None, None
                corr_per_fold = []
                loss_histories = {}
                tuning_info = {}
            cells.append({
                "dataset_id": dataset_id,
                "dataset_label": ds_label,
                "encoding_id": enc_id,
                "encoding_label": enc_label,
                "has_data": has_data,
                "task_type": task_type,
                "drive_url": sheet_row.get("Drive folder URL", ""),
                "wd_mean": wd_mean,
                "wd_std": wd_std,
                "kl_mean": kl_mean,
                "kl_std": kl_std,
                "corr_mean": corr_mean,
                "corr_std": corr_std,
                "utility_mean": utility_mean,
                "utility_std": utility_std,
                "f1_real_mean": f1_real_mean,
                "f1_synth_mean": f1_synth_mean,
                "n_folds": aggregate.get("n_folds") or run_summary.get("crossval", {}).get("n_folds"),
                "per_fold": corr_per_fold,
                "loss_histories": {str(k): v for k, v in loss_histories.items()},
                "tuning": tuning_info,
            })

    return {
        "dataset_order": dataset_order,
        "dataset_labels": dataset_labels,
        "encoding_order": encoding_order,
        "encoding_labels": encoding_labels,
        "cells": cells,
        "headers": headers,
        "dataset_schema": dataset_schema_cache,
        "rank_excluded_dataset_ids": rank_excluded_dataset_ids or [],
    }


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

def html_page(refresh_seconds: int) -> str:
    return r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CTGAN Monitor</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    :root {
      --bg: #08111e;
      --panel: rgba(255,255,255,0.04);
      --border: rgba(255,255,255,0.08);
      --card: rgba(255,255,255,0.06);
      --muted: #94a3b8;
      --text: #e2e8f0;
      --accent: #38bdf8;
      --accent2: #a855f7;
      --done: #14532d; --done-b: #22c55e;
      --prog: #1e3a8a; --prog-b: #60a5fa;
      --stale: #78350f; --stale-b: #f59e0b;
      --failed: #7f1d1d; --failed-b: #f87171;
      --skipped: #3f3f46; --skipped-b: #a1a1aa;
      --empty: #111827; --empty-b: #374151;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Inter', system-ui, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }

    /* Tabs */
    .tab-bar { display: flex; gap: 0; border-bottom: 1px solid var(--border); background: rgba(0,0,0,0.3); padding: 0 20px; position: sticky; top: 0; z-index: 20; backdrop-filter: blur(8px); }
    .tab-btn { padding: 14px 20px; font-size: 14px; font-weight: 500; border: none; background: none; color: var(--muted); cursor: pointer; border-bottom: 2px solid transparent; transition: color 150ms, border-color 150ms; }
    .tab-btn.active { color: var(--accent); border-bottom-color: var(--accent); }
    .tab-btn:hover:not(.active) { color: var(--text); }
    .tab-panel { display: none; padding: 16px 20px 80px; }
    .tab-panel.active { display: block; }

    /* Header strip */
    .page-header { padding: 16px 20px; background: rgba(0,0,0,0.2); border-bottom: 1px solid var(--border); display: flex; align-items: center; justify-content: space-between; }
    .page-header h1 { font-size: 18px; font-weight: 700; }
    .page-header .sub { color: var(--muted); font-size: 13px; }

    /* ---- PROGRESS TAB ---- */
    .summary { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 14px; }
    .badge { background: var(--panel); border: 1px solid var(--border); border-radius: 999px; padding: 6px 12px; font-size: 13px; }
    .layout { display: grid; grid-template-columns: minmax(0,1fr) 340px; gap: 14px; }
    .pg-card { background: rgba(17,24,39,0.95); border: 1px solid var(--border); border-radius: 14px; overflow: hidden; }
    .table-wrap { overflow: auto; max-height: calc(100vh - 240px); }
    table { border-collapse: separate; border-spacing: 0; width: 100%; min-width: 900px; }
    th, td { border-bottom: 1px solid #1f2937; border-right: 1px solid #1f2937; padding: 7px 8px; text-align: left; font-size: 12px; vertical-align: top; }
    th { position: sticky; top: 0; background: #0f172a; z-index: 1; }
    th:first-child, td:first-child { position: sticky; left: 0; background: #0f172a; z-index: 2; min-width: 140px; }
    .cell { border-radius: 8px; padding: 7px; min-height: 60px; border: 1px solid var(--empty-b); background: var(--empty); cursor: pointer; }
    .cell.done { background: var(--done); border-color: var(--done-b); }
    .cell.in-progress { background: var(--prog); border-color: var(--prog-b); }
    .cell.stale-in-progress { background: var(--stale); border-color: var(--stale-b); }
    .cell.failed { background: var(--failed); border-color: var(--failed-b); }
    .cell.skipped { background: var(--skipped); border-color: var(--skipped-b); }
    .cell .status { font-weight: 700; margin-bottom: 3px; font-size: 11px; }
    .cell .stage { color: #d1d5db; font-size: 11px; }
    .cell .coord { color: var(--muted); font-size: 10px; margin-top: 3px; }
    .side { padding: 14px; display: flex; flex-direction: column; gap: 12px; }
    .side h2 { font-size: 15px; }
    .detail-list { display: flex; flex-direction: column; gap: 8px; max-height: calc(100vh - 320px); overflow: auto; }
    .detail { border: 1px solid #374151; border-radius: 8px; padding: 9px; background: #0b1220; font-size: 12px; }
    .detail .title { font-weight: 700; margin-bottom: 3px; }
    .detail pre { white-space: pre-wrap; word-break: break-word; font-size: 11px; color: #d1d5db; font-family: ui-monospace, monospace; margin-top: 4px; }
    .action-row { display: flex; gap: 8px; margin-bottom: 8px; }
    button { border: 1px solid #4b5563; background: #111827; color: var(--text); border-radius: 8px; padding: 7px 12px; cursor: pointer; font-size: 12px; font-family: inherit; transition: background 120ms; }
    button:hover { background: #1f2937; }
    button.danger { border-color: #f87171; background: #7f1d1d; }
    button.danger:hover { background: #991b1b; }
    button.pill-btn { border-radius: 999px; padding: 5px 12px; font-size: 12px; }
    button.pill-btn.active { background: rgba(56,189,248,0.2); border-color: rgba(56,189,248,0.6); color: #c8f5ff; }

    /* ---- RESULTS TAB ---- */
    .results-section { margin-bottom: 22px; }
    .results-section-title { font-size: 13px; font-weight: 600; color: var(--muted); letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 10px; }

    /* Toolbar */
    .results-toolbar { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 14px; }
    .results-toolbar label { font-size: 13px; color: var(--muted); display: flex; align-items: center; gap: 6px; }
    .results-toolbar select { background: rgba(255,255,255,0.07); border: 1px solid var(--border); border-radius: 8px; color: var(--text); padding: 5px 8px; font-size: 13px; font-family: inherit; }

    /* Summary rank charts */
    .rank-charts-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-bottom: 20px; }
    .rank-filters { display: flex; flex-wrap: wrap; align-items: center; gap: 12px; margin-bottom: 10px; }
    .rank-filters label { display: inline-flex; align-items: center; gap: 5px; color: var(--muted); font-size: 13px; cursor: pointer; }
    .rank-filters input[type=checkbox] { accent-color: var(--accent); }
    .rank-filters input[type=range] { accent-color: var(--accent); }
    .rank-filters .range-val { color: var(--text); font-weight: 600; min-width: 36px; }
    .rank-filters select { background: rgba(255,255,255,0.07); border: 1px solid var(--border); border-radius: 8px; color: var(--text); padding: 4px 8px; font-size: 12px; font-family: inherit; }

    /* Results matrix */
    .res-matrix { overflow: auto; border: 1px solid var(--border); border-radius: 12px; margin-bottom: 20px; }
    .res-matrix table { border-collapse: collapse; min-width: 800px; width: 100%; font-size: 12px; }
    .res-matrix th, .res-matrix td { border: 1px solid var(--border); padding: 8px 10px; }
    .res-matrix thead th { background: rgba(255,255,255,0.04); font-weight: 600; position: sticky; top: 0; z-index: 1; white-space: nowrap; cursor: pointer; }
    .res-matrix thead th:first-child { cursor: default; }
    .res-matrix thead th:not(:first-child):hover { background: rgba(56,189,248,0.08); }
    .res-matrix tbody tr:hover { background: rgba(255,255,255,0.03); }
    .res-matrix td.ds-col { font-weight: 600; white-space: nowrap; cursor: pointer; }
    .res-matrix td.ds-col:hover { color: var(--accent); }
    .res-matrix td.metric-val { text-align: right; font-variant-numeric: tabular-nums; cursor: pointer; }
    .res-matrix td.metric-val:hover { background: rgba(56,189,248,0.1); }
    .res-matrix td.no-data { color: var(--muted); text-align: center; }
    .rank-pill { display: inline-block; padding: 1px 5px; border-radius: 999px; font-size: 10px; font-weight: 700; margin-left: 4px; }
    .rank-1 { background: rgba(34,197,94,0.2); color: #4ade80; }
    .rank-2 { background: rgba(56,189,248,0.2); color: #7dd3fc; }
    .rank-3 { background: rgba(168,85,247,0.2); color: #d8b4fe; }

    /* Card matrix (dataset rows) */
    .matrix-section { margin-top: 24px; }
    .matrix-row { display: grid; grid-template-columns: 220px 1fr; gap: 12px; margin-bottom: 12px; align-items: stretch; }
    .ds-name-card { padding: 14px 16px; border: 1px solid var(--border); border-radius: 14px;
      background: linear-gradient(120deg, rgba(56,189,248,0.12), rgba(168,85,247,0.09));
      display: flex; flex-direction: column; gap: 8px; }
    .ds-name-card.highlight { box-shadow: 0 0 0 2px rgba(56,189,248,0.5); }
    .ds-title { font-size: 15px; font-weight: 700; cursor: pointer; display: inline-flex; align-items: center; gap: 6px; }
    .ds-title:hover { color: var(--accent); }
    .ds-title::after { content: "↗"; opacity: 0.5; font-size: 12px; }
    .ds-pills { display: flex; flex-wrap: wrap; gap: 5px; }
    .ds-pill { padding: 3px 8px; border-radius: 999px; border: 1px solid var(--border); background: rgba(255,255,255,0.05); font-size: 11px; color: var(--muted); }
    .ds-pill.accent { border-color: rgba(56,189,248,0.4); color: #7dd3fc; }
    .ds-pill.cls { border-color: rgba(34,197,94,0.4); color: #4ade80; }
    .ds-pill.reg { border-color: rgba(251,191,36,0.4); color: #fcd34d; }
    .enc-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 8px; }
    .enc-cell { padding: 10px 12px; border-radius: 12px; border: 1px solid var(--border); background: var(--card);
      cursor: pointer; position: relative; overflow: hidden; transition: transform 150ms, border-color 150ms; }
    .enc-cell::before { content: ""; position: absolute; inset: 0;
      background: radial-gradient(circle at 20% 20%, rgba(56,189,248,0.18), transparent 55%);
      opacity: 0; transition: opacity 150ms; }
    .enc-cell:hover { transform: translateY(-2px); border-color: rgba(56,189,248,0.4); }
    .enc-cell:hover::before { opacity: 1; }
    .enc-cell.empty { background: rgba(255,255,255,0.02); cursor: default; }
    .enc-cell.empty:hover { transform: none; border-color: var(--border); }
    .enc-name { font-size: 11px; font-weight: 700; color: var(--accent); margin-bottom: 6px; cursor: pointer; }
    .enc-name:hover { text-decoration: underline; }
    .enc-metrics { display: flex; flex-direction: column; gap: 2px; font-size: 11px; color: var(--muted); }
    .enc-metrics .kv { display: flex; justify-content: space-between; }
    .enc-metrics .kv span { color: var(--text); font-variant-numeric: tabular-nums; }
    .enc-rank { position: absolute; top: 8px; right: 8px; font-size: 10px; font-weight: 700; padding: 1px 5px; border-radius: 999px; }

    /* Charts section */
    .charts-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-top: 16px; }
    .chart-card { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 14px; }
    .chart-card h3 { font-size: 13px; font-weight: 600; margin-bottom: 10px; color: var(--text); }
    canvas { width: 100% !important; height: 260px !important; }

    /* Modal */
    .modal { position: fixed; inset: 0; display: none; align-items: center; justify-content: center; z-index: 1000; }
    .modal.open { display: flex; }
    .modal-backdrop { position: absolute; inset: 0; background: rgba(0,0,0,0.65); backdrop-filter: blur(6px); }
    .modal-box { position: relative; z-index: 1; background: #0d1b2a; border: 1px solid var(--border); border-radius: 16px; padding: 20px; max-width: 960px; width: 94%; max-height: 90vh; overflow-y: auto; }
    .modal-box h2 { font-size: 18px; margin-bottom: 4px; }
    .modal-box .sub { color: var(--muted); font-size: 13px; margin-bottom: 14px; }
    .modal-close { position: absolute; top: 12px; right: 14px; background: rgba(255,255,255,0.08); border: 1px solid var(--border); color: var(--text); border-radius: 50%; width: 30px; height: 30px; font-size: 17px; cursor: pointer; display: flex; align-items: center; justify-content: center; }
    .fold-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px,1fr)); gap: 8px; margin-bottom: 14px; }
    .fold-card { background: rgba(255,255,255,0.04); border: 1px solid var(--border); border-radius: 10px; padding: 10px; font-size: 12px; }
    .fold-card .fold-title { font-weight: 700; margin-bottom: 6px; color: var(--accent); }
    .fold-card .kv { display: flex; justify-content: space-between; gap: 6px; color: var(--muted); }
    .fold-card .kv span { color: var(--text); font-weight: 500; }
    .modal-charts { display: grid; grid-template-columns: repeat(2,1fr); gap: 10px; }
    .modal-charts canvas { height: 220px !important; }
    .pill-row { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
    .modal-metric-single canvas { height: 340px !important; width: 100% !important; }
    .fold-card { cursor: pointer; transition: border-color 120ms, transform 120ms; }
    .fold-card:hover { border-color: rgba(56,189,248,0.55); transform: translateY(-1px); }
    .metrics-table { width: 100%; min-width: 0; border-collapse: collapse; margin-bottom: 12px; }
    .metrics-table th, .metrics-table td { position: static; background: transparent; border: 1px solid var(--border); padding: 6px 8px; font-size: 12px; }
    .metrics-table td:last-child { text-align: right; font-variant-numeric: tabular-nums; color: var(--text); }
    .matrix-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin-top: 12px; }
    .matrix-card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 10px; overflow: auto; }
    .matrix-card h3 { font-size: 12px; margin-bottom: 8px; }
    .corr-table { border-collapse: collapse; min-width: max-content; }
    .corr-table th, .corr-table td { position: static; min-width: 42px; max-width: 72px; padding: 4px 5px; text-align: center; font-size: 10px; border: 1px solid rgba(255,255,255,0.08); background: transparent; }
    .corr-table th { color: var(--muted); font-weight: 600; }
    .corr-table .row-head { text-align: right; max-width: 110px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

    /* Tuning badge */
    .tuning-row { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 10px; }
    .tuning-badge { font-size: 11px; padding: 3px 8px; border-radius: 8px; background: rgba(168,85,247,0.12); border: 1px solid rgba(168,85,247,0.3); color: #d8b4fe; font-family: ui-monospace, monospace; }

    /* Scroll-to-top */
    .scroll-top { position: fixed; right: 18px; bottom: 18px; z-index: 500; padding: 9px 12px; border-radius: 10px; border: 1px solid var(--border); background: rgba(56,189,248,0.12); color: #c8f5ff; cursor: pointer; display: none; }

    @media (max-width: 900px) {
      .layout, .charts-grid, .modal-charts, .rank-charts-grid { grid-template-columns: 1fr; }
      .matrix-grid { grid-template-columns: 1fr; }
      .table-wrap, .detail-list { max-height: none; }
      .matrix-row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
<div class="page-header">
  <h1>CTGAN Monitor</h1>
  <div class="sub" id="meta">Loading…</div>
</div>
<div class="tab-bar">
  <button class="tab-btn active" data-tab="progress">Progress</button>
  <button class="tab-btn" data-tab="results">Results</button>
</div>

<!-- ==================== PROGRESS TAB ==================== -->
<div class="tab-panel active" id="tab-progress">
  <div class="summary" id="summary"></div>
  <div class="layout">
    <div class="pg-card table-wrap">
      <table id="grid"></table>
    </div>
    <div class="pg-card side">
      <div>
        <h2>Selected Cell</h2>
        <div style="color:var(--muted);font-size:13px;" id="selected">Click a cell to inspect.</div>
      </div>
      <div>
        <h2>Active / Failed</h2>
        <div class="detail-list" id="details"></div>
      </div>
    </div>
  </div>
</div>

<!-- ==================== RESULTS TAB ==================== -->
<div class="tab-panel" id="tab-results">

  <!-- Toolbar -->
  <div class="results-toolbar">
    <label>Metric:
      <select id="metric-select">
        <option value="wd">WD (Wasserstein)</option>
        <option value="kl">KL Divergence</option>
        <option value="corr">Corr dist (Frobenius)</option>
        <option value="utility">Utility gap</option>
      </select>
    </label>
    <label>Show ranks: <input type="checkbox" id="show-ranks" checked></label>
    <span id="results-status" style="color:var(--muted);font-size:12px;"></span>
  </div>

  <!-- Rank summary charts (all 4 metrics) -->
  <div class="results-section">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:10px;">
      <div class="results-section-title" style="margin-bottom:0;">Average rank per encoding</div>
      <div class="rank-filters" id="rank-filters">
        <label><input type="checkbox" id="filter-cls" checked> Classification</label>
        <label><input type="checkbox" id="filter-reg" checked> Regression</label>
        <label>Cat share ≥ <span class="range-val" id="filter-cat-val">0%</span></label>
        <input type="range" id="filter-cat" min="0" max="100" step="5" value="0">
      </div>
    </div>
    <div class="rank-charts-grid" id="rank-charts"></div>
  </div>

  <!-- Metric matrix table -->
  <div class="results-section">
    <div class="results-section-title">Metric matrix</div>
    <div class="res-matrix" id="res-matrix"></div>
  </div>

  <!-- Card matrix (dataset rows) -->
  <div class="results-section matrix-section" id="matrix-section">
    <div class="results-section-title">Per-dataset cards</div>
    <div id="matrix-rows"></div>
  </div>

</div>

<!-- Modal -->
<div class="modal" id="detail-modal">
  <div class="modal-backdrop" id="modal-backdrop"></div>
  <div class="modal-box" id="modal-box">
    <button class="modal-close" id="modal-close">×</button>
    <h2 id="modal-title"></h2>
    <div class="sub" id="modal-sub"></div>
    <div id="modal-body"></div>
  </div>
</div>

<button class="scroll-top" id="scroll-top">↑ Top</button>

<script>
const REFRESH_SEC = """ + str(refresh_seconds) + r""";
let progressData = null;
let resultsData = null;
let summaryCharts = [];
let rankCharts = [];
let modalCharts = [];
let activeMetric = 'wd';
let showRanks = true;
let resultsLoading = false;

// ---- Scroll-to-top ----
const scrollTopBtn = document.getElementById('scroll-top');
window.addEventListener('scroll', () => { scrollTopBtn.style.display = window.scrollY > 300 ? 'block' : 'none'; });
scrollTopBtn.addEventListener('click', () => window.scrollTo({ top: 0, behavior: 'smooth' }));

// ---- Tab switching ----
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
    if (btn.dataset.tab === 'results' && !resultsData) loadResults();
  });
});

// ---- Metric select + rank toggle ----
document.getElementById('metric-select').addEventListener('change', e => {
  activeMetric = e.target.value;
  if (resultsData) { renderResultsMatrix(resultsData); renderCardMatrix(resultsData); }
});
document.getElementById('show-ranks').addEventListener('change', e => {
  showRanks = e.target.checked;
  if (resultsData) { renderResultsMatrix(resultsData); renderCardMatrix(resultsData); }
});

// ---- Rank filter controls ----
const filterCls = document.getElementById('filter-cls');
const filterReg = document.getElementById('filter-reg');
const filterCat = document.getElementById('filter-cat');
const filterCatVal = document.getElementById('filter-cat-val');
[filterCls, filterReg].forEach(el => el.addEventListener('change', () => { if (resultsData) renderRankCharts(resultsData); }));
filterCat.addEventListener('input', () => {
  filterCatVal.textContent = filterCat.value + '%';
  if (resultsData) renderRankCharts(resultsData);
});

// ---- Modal ----
const modal = document.getElementById('detail-modal');
document.getElementById('modal-backdrop').addEventListener('click', closeModal);
document.getElementById('modal-close').addEventListener('click', closeModal);
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });
function closeModal() {
  modal.classList.remove('open');
  modalCharts.forEach(c => c?.destroy?.());
  modalCharts = [];
}

// ============================================================
// HELPERS
// ============================================================
function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
}

const PALETTE = ['#38bdf8','#a855f7','#22c55e','#f59e0b','#f87171','#fb923c','#34d399','#818cf8','#e879f9','#fbbf24','#60a5fa','#4ade80'];

function fmtNum(v, d=4) {
  if (v === null || v === undefined || !isFinite(v)) return '—';
  return Number(v).toFixed(d);
}
function fmtPct(v) {
  if (v === null || v === undefined || !isFinite(v)) return '—';
  return (Number(v) * 100).toFixed(1) + '%';
}

function computeRanks(values) {
  // values: array of {id, val}; lower=better
  const ranked = values.filter(x => x.val !== null && x.val !== undefined && isFinite(x.val));
  ranked.sort((a, b) => a.val - b.val);
  const out = {};
  ranked.forEach((x, i) => { out[x.id] = i + 1; });
  return out;
}

const METRIC_KEYS = {
  wd:      { mean: 'wd_mean',      std: 'wd_std',      label: 'WD ↓',      title: 'Wasserstein Distance (unencoded, lower=better)' },
  kl:      { mean: 'kl_mean',      std: 'kl_std',      label: 'KL ↓',      title: 'KL Divergence (unencoded, lower=better)' },
  corr:    { mean: 'corr_mean',    std: 'corr_std',    label: 'Corr ↓',    title: 'Correlation Frobenius dist (unencoded, lower=better)' },
  utility: { mean: 'utility_mean', std: 'utility_std', label: 'Utility ↓', title: 'Utility gap (lower=better)' },
};
const ALL_METRICS = Object.entries(METRIC_KEYS);

// ============================================================
// PROGRESS TAB
// ============================================================
let selectedCell = null;
function setSelected(cell) {
  selectedCell = cell;
  const target = document.getElementById('selected');
  target.innerHTML = '';
  const actions = el('div', 'action-row');
  const clearBtn = el('button', 'danger', 'Clear Cell');
  clearBtn.onclick = () => clearSelectedCell();
  actions.appendChild(clearBtn);
  target.appendChild(actions);
  ['coord','dataset','encoding','status','effective_status','stage','owner','run_id','started_at','heartbeat_at','finished_at','stale_age_seconds','note'].forEach(k => {
    const d = el('div', 'detail');
    d.appendChild(el('div', 'title', k));
    d.appendChild(el('pre', '', String(cell[k] ?? '')));
    target.appendChild(d);
  });
}

async function clearSelectedCell() {
  if (!selectedCell) return;
  if (!confirm(`Clear ${selectedCell.coord} (${selectedCell.dataset} / ${selectedCell.encoding})?`)) return;
  const r = await fetch(`/api/clear?coord=${encodeURIComponent(selectedCell.coord)}`, { method: 'POST' });
  const d = await r.json();
  if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`);
  document.getElementById('selected').innerHTML = `<div style="color:var(--muted)">Cleared ${d.coord}. Refreshing…</div>`;
  selectedCell = null;
  await refreshProgress();
}

function renderProgress(data) {
  document.getElementById('meta').textContent =
    `${data.spreadsheet_title} / ${data.worksheet_name} · refresh ${REFRESH_SEC}s`;
  const summary = document.getElementById('summary');
  summary.innerHTML = '';
  ['done','in-progress','stale-in-progress','failed','skipped','not-started'].forEach(k => {
    summary.appendChild(el('div', 'badge', `${k}: ${data.counts[k] || 0}`));
  });
  const table = document.getElementById('grid');
  table.innerHTML = '';
  const thead = document.createElement('thead');
  const hr = document.createElement('tr');
  hr.appendChild(el('th', '', 'encoding \\ dataset'));
  data.dataset_headers.forEach(h => hr.appendChild(el('th', '', h)));
  thead.appendChild(hr); table.appendChild(thead);
  const tbody = document.createElement('tbody');
  data.rows.forEach(row => {
    const tr = document.createElement('tr');
    tr.appendChild(el('td', '', row.encoding));
    row.cells.forEach(cell => {
      const td = document.createElement('td');
      const box = el('div', `cell ${cell.effective_status}`);
      box.appendChild(el('div', 'status', cell.effective_status === 'stale-in-progress' ? 'stale' : cell.status));
      box.appendChild(el('div', 'stage', cell.stage || ''));
      box.appendChild(el('div', 'coord', cell.coord));
      box.title = cell.note || '';
      box.onclick = () => setSelected(cell);
      td.appendChild(box); tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  const details = document.getElementById('details');
  details.innerHTML = '';
  if (!data.details.length) {
    const m = el('div', '', 'No active or failed cells.'); m.style.color='var(--muted)'; m.style.fontSize='13px';
    details.appendChild(m); return;
  }
  data.details.forEach(cell => {
    const card = el('div', 'detail');
    card.appendChild(el('div', 'title', `${cell.coord} · ${cell.effective_status} · ${cell.stage||''}`));
    const sub = el('div', '', `${cell.dataset} / ${cell.encoding}`); sub.style.color='var(--muted)';
    card.appendChild(sub);
    if (cell.stale_age_seconds) {
      const s = el('div', '', `last update ${cell.stale_age_seconds}s ago`); s.style.color='var(--muted)';
      card.appendChild(s);
    }
    card.appendChild(el('pre', '', cell.note || ''));
    card.onclick = () => setSelected(cell);
    details.appendChild(card);
  });
}

async function refreshProgress() {
  const r = await fetch('/api/status');
  progressData = await r.json();
  renderProgress(progressData);
}

// ============================================================
// RESULTS TAB
// ============================================================

// --- Build cell index + ranks ---
function buildIndex(data) {
  const idx = {};
  data.cells.forEach(c => { idx[c.dataset_id + '|' + c.encoding_id] = c; });
  return idx;
}
function rankExcludedSet(data) {
  return new Set(data.rank_excluded_dataset_ids || []);
}
function isRankedDataset(data, ds) {
  return !rankExcludedSet(data).has(ds);
}
function buildRanksByDs(data, mk, idx) {
  const ranks = {};
  const excluded = rankExcludedSet(data);
  data.dataset_order.forEach(ds => {
    if (excluded.has(ds)) {
      ranks[ds] = {};
      return;
    }
    const vals = data.encoding_order.map(enc => ({ id: enc, val: (idx[ds+'|'+enc]||{})[mk.mean] ?? null }));
    ranks[ds] = computeRanks(vals);
  });
  return ranks;
}

// ---- Metric matrix table ----
function renderResultsMatrix(data) {
  const { dataset_order, dataset_labels, encoding_order, encoding_labels } = data;
  const mk = METRIC_KEYS[activeMetric];
  const idx = buildIndex(data);
  const ranksByDs = buildRanksByDs(data, mk, idx);

  const wrap = document.getElementById('res-matrix');
  wrap.innerHTML = '';
  const table = document.createElement('table');

  const thead = document.createElement('thead');
  const hr = document.createElement('tr');
  const th0 = document.createElement('th'); th0.textContent = 'Dataset \\ Encoding';
  hr.appendChild(th0);
  encoding_order.forEach(enc => {
    const th = document.createElement('th');
    th.textContent = encoding_labels[enc] || enc;
    th.title = 'Click to see this encoding across datasets';
    th.onclick = () => openEncodingModal(enc, data);
    hr.appendChild(th);
  });
  thead.appendChild(hr); table.appendChild(thead);

  const tbody = document.createElement('tbody');
  dataset_order.forEach(ds => {
    const tr = document.createElement('tr');
    const td0 = document.createElement('td');
    td0.className = 'ds-col';
    td0.textContent = dataset_labels[ds] || ds;
    td0.title = 'Click to see this dataset across encodings';
    td0.onclick = () => openDatasetModal(ds, data);
    tr.appendChild(td0);
    encoding_order.forEach(enc => {
      const cell = idx[ds+'|'+enc];
      const td = document.createElement('td');
      if (!cell || !cell.has_data) {
        td.className = 'no-data'; td.textContent = '—';
      } else {
        td.className = 'metric-val';
        const val = cell[mk.mean], std = cell[mk.std];
        const rank = ranksByDs[ds]?.[enc];
        let html = fmtNum(val);
        if (std !== null && std !== undefined && isFinite(std)) html += `<br><small style="color:var(--muted)">±${fmtNum(std)}</small>`;
        if (showRanks && rank) {
          const cls = rank <= 3 ? `rank-${rank}` : '';
          html += ` <span class="rank-pill ${cls}">#${rank}</span>`;
        }
        td.innerHTML = html;
        td.title = `${dataset_labels[ds]} / ${encoding_labels[enc]}\n${mk.title}`;
        td.onclick = () => openCellModal(cell);
      }
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  wrap.appendChild(table);

  document.getElementById('results-status').textContent =
    `${data.cells.filter(c => c.has_data).length}/${data.cells.length} cells with data`;
}

// ---- Rank summary charts (all 4 metrics) ----
function renderRankCharts(data) {
  const { dataset_order, encoding_order, encoding_labels, dataset_schema } = data;
  const showCls = filterCls.checked, showReg = filterReg.checked;
  const catMin = Number(filterCat.value || 0) / 100;

  rankCharts.forEach(c => c?.destroy?.()); rankCharts = [];
  const wrap = document.getElementById('rank-charts');
  wrap.innerHTML = '';

  const idx = buildIndex(data);

  // Filter datasets
  const filteredDs = dataset_order.filter(ds => {
    if (!isRankedDataset(data, ds)) return false;
    const schema = (dataset_schema || {})[ds] || {};
    const taskType = data.cells.find(c => c.dataset_id === ds && c.task_type)?.task_type || '';
    const isReg = taskType === 'regression';
    if (!showCls && !isReg) return false;
    if (!showReg && isReg) return false;
    const catShare = schema.cat_share ?? 0;
    return catShare >= catMin;
  });

  ALL_METRICS.forEach(([key, mk]) => {
    // Compute ranks per filtered dataset, then average per encoding
    const ranksByDs = {};
    filteredDs.forEach(ds => {
      const vals = encoding_order.map(enc => ({ id: enc, val: (idx[ds+'|'+enc]||{})[mk.mean] ?? null }));
      ranksByDs[ds] = computeRanks(vals);
    });
    const avgRanks = encoding_order.map(enc => {
      const rs = filteredDs.map(ds => ranksByDs[ds]?.[enc]).filter(r => r !== undefined && isFinite(r));
      return rs.length ? rs.reduce((a,b)=>a+b,0)/rs.length : null;
    });
    const labels = encoding_order.map(e => encoding_labels[e] || e);
    const card = document.createElement('div'); card.className = 'chart-card';
    const h = document.createElement('h3'); h.textContent = mk.label + ' — avg rank (↓ better)'; card.appendChild(h);
    const canvas = document.createElement('canvas'); card.appendChild(canvas); wrap.appendChild(card);
    const chart = new Chart(canvas, {
      type: 'bar',
      data: {
        labels,
        datasets: [{ data: avgRanks,
          backgroundColor: labels.map((_,i) => PALETTE[i%PALETTE.length]+'55'),
          borderColor: labels.map((_,i) => PALETTE[i%PALETTE.length]),
          borderWidth: 1.5, borderRadius: 6 }]
      },
      options: {
        maintainAspectRatio: false,
        plugins: { legend: { display: false }, tooltip: { callbacks: {
          label: ctx => {
            const enc = encoding_order[ctx.dataIndex];
            const dsCount = filteredDs.filter(ds => ranksByDs[ds]?.[enc] !== undefined).length;
            return `Rank: ${ctx.parsed.y?.toFixed(2) ?? '—'} · datasets: ${dsCount}`;
          }
        }}},
        scales: {
          x: { ticks: { color: '#94a3b8', maxRotation: 30, font: { size: 11 } }, grid: { color: 'rgba(255,255,255,0.03)' } },
          y: { ticks: { color: '#94a3b8', stepSize: 1 }, grid: { color: 'rgba(255,255,255,0.05)' },
               title: { display: true, text: 'avg rank', color: '#94a3b8', font: { size: 11 } } }
        }
      }
    });
    rankCharts.push(chart);
  });
}

// ---- Card matrix (per-dataset rows) ----
function renderCardMatrix(data) {
  const { dataset_order, dataset_labels, encoding_order, encoding_labels, dataset_schema } = data;
  const mk = METRIC_KEYS[activeMetric];
  const idx = buildIndex(data);
  const ranksByDs = buildRanksByDs(data, mk, idx);

  const wrap = document.getElementById('matrix-rows');
  wrap.innerHTML = '';

  dataset_order.forEach(ds => {
    const dsLabel = dataset_labels[ds] || ds;
    const schema = (dataset_schema || {})[ds] || {};
    const taskType = data.cells.find(c => c.dataset_id === ds && c.task_type)?.task_type || '';

    const row = document.createElement('div'); row.className = 'matrix-row'; row.dataset.dataset = ds;

    // Dataset name card
    const nameCard = document.createElement('div'); nameCard.className = 'ds-name-card'; nameCard.dataset.dataset = ds;
    const title = document.createElement('div'); title.className = 'ds-title'; title.textContent = dsLabel;
    title.onclick = () => openDatasetModal(ds, data);
    nameCard.appendChild(title);

    const pills = document.createElement('div'); pills.className = 'ds-pills';
    if (taskType) {
      const tp = document.createElement('span');
      tp.className = 'ds-pill ' + (taskType === 'regression' ? 'reg' : 'cls');
      tp.textContent = taskType; pills.appendChild(tp);
    }
    if (schema.n_cat !== undefined) {
      const cp = document.createElement('span'); cp.className = 'ds-pill accent';
      cp.textContent = `cat ${schema.n_cat}/${schema.n_total}`; pills.appendChild(cp);
    }
    if (schema.n_rows) {
      const rp = document.createElement('span'); rp.className = 'ds-pill';
      rp.textContent = `n=${schema.n_rows}`; pills.appendChild(rp);
    }
    if (schema.n_folds) {
      const fp = document.createElement('span'); fp.className = 'ds-pill';
      fp.textContent = `${schema.n_folds}-fold`; pills.appendChild(fp);
    }
    nameCard.appendChild(pills);
    row.appendChild(nameCard);

    // Encoding cards grid
    const grid = document.createElement('div'); grid.className = 'enc-grid';
    encoding_order.forEach(enc => {
      const cell = idx[ds+'|'+enc];
      const encLabel = encoding_labels[enc] || enc;
      const card = document.createElement('div');
      if (!cell || !cell.has_data) {
        card.className = 'enc-cell empty';
        card.innerHTML = `<div class="enc-name">${encLabel}</div><div style="color:var(--muted);font-size:11px;">no data</div>`;
      } else {
        card.className = 'enc-cell';
        card.onclick = () => openCellModal(cell);

        const rank = ranksByDs[ds]?.[enc];
        if (showRanks && rank) {
          const rankEl = document.createElement('div'); rankEl.className = 'enc-rank';
          const cls = rank <= 3 ? `rank-${rank}` : ''; rankEl.className = `enc-rank rank-pill ${cls}`;
          rankEl.textContent = '#' + rank; card.appendChild(rankEl);
        }

        const nameEl = document.createElement('div'); nameEl.className = 'enc-name';
        nameEl.textContent = encLabel;
        nameEl.onclick = (e) => { e.stopPropagation(); openEncodingModal(enc, data); };
        card.appendChild(nameEl);

        const metrics = document.createElement('div'); metrics.className = 'enc-metrics';
        const isReg = cell.task_type === 'regression';
        const metricRows = [
          ['WD', fmtNum(cell.wd_mean)],
          ['KL', fmtNum(cell.kl_mean)],
          ['Corr', fmtNum(cell.corr_mean)],
          ['Util gap', fmtNum(cell.utility_mean, 2) + '%'],
          [isReg ? 'R² real' : 'F1 real', fmtNum(cell.f1_real_mean, 3)],
        ];
        metricRows.forEach(([k, v]) => {
          const kv = document.createElement('div'); kv.className = 'kv';
          kv.innerHTML = `${k}: <span>${v}</span>`; metrics.appendChild(kv);
        });
        card.appendChild(metrics);
      }
      grid.appendChild(card);
    });
    row.appendChild(grid);
    wrap.appendChild(row);
  });
}

// ---- Cell detail modal ----
function openCellModal(cell) {
  document.getElementById('modal-title').textContent = `${cell.dataset_label} / ${cell.encoding_label}`;
  document.getElementById('modal-sub').textContent = `Task: ${cell.task_type || '—'} · Folds: ${cell.n_folds || '—'}`;
  const body = document.getElementById('modal-body');
  body.innerHTML = '';

  // Tuning info
  if (cell.tuning && cell.tuning.best_value !== null) {
    const row = document.createElement('div'); row.className = 'tuning-row';
    const bv = document.createElement('span'); bv.className = 'tuning-badge';
    bv.textContent = `tuning score: ${fmtNum(cell.tuning.best_value)} (${cell.tuning.best_source || ''})`;
    row.appendChild(bv);
    if (cell.tuning.best_params) {
      Object.entries(cell.tuning.best_params).forEach(([k, v]) => {
        const b = document.createElement('span'); b.className = 'tuning-badge';
        b.textContent = `${k}: ${typeof v === 'number' ? v.toFixed(4) : v}`;
        row.appendChild(b);
      });
    }
    body.appendChild(row);
  }

  // Per-fold cards
  if (cell.per_fold?.length) {
    const grid = document.createElement('div'); grid.className = 'fold-grid';
    const isReg = cell.task_type === 'regression';
    cell.per_fold.forEach(fold => {
      const card = document.createElement('div'); card.className = 'fold-card';
      card.innerHTML = `<div class="fold-title">Fold ${fold.fold_id}</div>`;
      [
        ['WD', fmtNum(fold.wd_unencoded)],
        ['KL', fmtNum(fold.kl_unencoded)],
        ['Corr', fmtNum(fold.corr_frobenius_unencoded)],
        [isReg ? 'R² real' : 'F1 real', fmtNum(isReg ? fold.r2_real : fold.f1_real, 3)],
        [isReg ? 'R² synth' : 'F1 synth', fmtNum(isReg ? fold.r2_synth : fold.f1_synth, 3)],
        [isReg ? 'R² gap' : 'F1 gap%', fmtNum(isReg ? fold.r2_pct_diff : fold.f1_pct_diff, 2)],
        ['n_train', fold.n_train ?? '—'],
        ['n_test', fold.n_test ?? '—'],
      ].forEach(([k, v]) => {
        const kv = document.createElement('div'); kv.className = 'kv';
        kv.innerHTML = `${k}: <span>${v}</span>`; card.appendChild(kv);
      });
      card.title = 'Open fold details';
      card.onclick = () => openFoldModal(cell, fold);
      grid.appendChild(card);
    });
    body.appendChild(grid);
  }

  // Modal charts
  const chartWrap = document.createElement('div'); chartWrap.className = 'modal-charts'; body.appendChild(chartWrap);
  if (cell.per_fold?.length) {
    const foldIds = cell.per_fold.map(f => `F${f.fold_id}`);
    const isReg = cell.task_type === 'regression';
    addModalLineChart(chartWrap, 'Distribution per fold', foldIds, [
      { label: 'WD', data: cell.per_fold.map(f => f.wd_unencoded), color: '#38bdf8' },
      { label: 'KL', data: cell.per_fold.map(f => f.kl_unencoded), color: '#a855f7' },
      { label: 'Corr', data: cell.per_fold.map(f => f.corr_frobenius_unencoded), color: '#22c55e' },
    ]);
    addModalLineChart(chartWrap, 'Utility per fold', foldIds, [
      { label: isReg ? 'R² real' : 'F1 real', data: cell.per_fold.map(f => isReg ? f.r2_real : f.f1_real), color: '#38bdf8' },
      { label: isReg ? 'R² synth' : 'F1 synth', data: cell.per_fold.map(f => isReg ? f.r2_synth : f.f1_synth), color: '#f59e0b' },
    ]);
  }
  const lossKeys = Object.keys(cell.loss_histories || {});
  if (lossKeys.length) {
    lossKeys.forEach(fk => {
      const h = cell.loss_histories[fk];
      addModalLineChart(chartWrap, `Loss — fold ${fk}`, h.generator.map((_,i)=>i+1), [
        { label: 'Generator', data: h.generator, color: '#38bdf8' },
        { label: 'Discriminator', data: h.discriminator, color: '#f87171' },
      ], { thinLine: true });
    });
  }
  if (cell.drive_url) {
    const link = document.createElement('a'); link.href = cell.drive_url; link.target = '_blank';
    link.textContent = '↗ Open Drive folder';
    link.style.cssText = 'display:inline-block;margin-top:12px;color:var(--accent);font-size:13px;';
    body.appendChild(link);
  }
  modal.classList.add('open');
}

function flattenMetrics(obj, prefix='') {
  const rows = [];
  Object.entries(obj || {}).forEach(([key, val]) => {
    const path = prefix ? `${prefix}.${key}` : key;
    if (val && typeof val === 'object' && !Array.isArray(val)) {
      rows.push(...flattenMetrics(val, path));
    } else if (!Array.isArray(val)) {
      rows.push([path, val]);
    }
  });
  return rows;
}

function fmtMetricValue(v) {
  if (v === null || v === undefined || v === '') return '—';
  if (typeof v === 'number') return isFinite(v) ? Number(v).toFixed(5) : '—';
  return String(v);
}

function matrixColor(v, mode) {
  if (v === null || v === undefined || !isFinite(v)) return 'rgba(255,255,255,0.03)';
  const x = Math.max(-1, Math.min(1, Number(v)));
  if (mode === 'difference') {
    const a = Math.min(0.75, Math.abs(x));
    return x >= 0 ? `rgba(34,197,94,${a})` : `rgba(248,113,113,${a})`;
  }
  const a = Math.min(0.75, Math.abs(x));
  return x >= 0 ? `rgba(56,189,248,${a})` : `rgba(168,85,247,${a})`;
}

function addCorrelationTable(container, title, columns, matrix, mode) {
  const card = document.createElement('div'); card.className = 'matrix-card';
  const h = document.createElement('h3'); h.textContent = title; card.appendChild(h);
  const table = document.createElement('table'); table.className = 'corr-table';
  const thead = document.createElement('thead');
  const hr = document.createElement('tr'); hr.appendChild(document.createElement('th'));
  columns.forEach(col => { const th = document.createElement('th'); th.textContent = col; th.title = col; hr.appendChild(th); });
  thead.appendChild(hr); table.appendChild(thead);
  const tbody = document.createElement('tbody');
  (matrix || []).forEach((row, i) => {
    const tr = document.createElement('tr');
    const th = document.createElement('th'); th.className = 'row-head'; th.textContent = columns[i] || ''; th.title = columns[i] || ''; tr.appendChild(th);
    (row || []).forEach(v => {
      const td = document.createElement('td');
      td.textContent = v === null || v === undefined || !isFinite(v) ? '—' : Number(v).toFixed(2);
      td.style.background = matrixColor(v, mode);
      td.title = fmtMetricValue(v);
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody); card.appendChild(table); container.appendChild(card);
}

function openFoldModal(cell, fold) {
  document.getElementById('modal-title').textContent = `${cell.dataset_label} / ${cell.encoding_label} · Fold ${fold.fold_id}`;
  document.getElementById('modal-sub').textContent = `Task: ${cell.task_type || '—'} · n_train=${fold.n_train ?? '—'} · n_test=${fold.n_test ?? '—'}`;
  const body = document.getElementById('modal-body'); body.innerHTML = '';

  const back = document.createElement('button'); back.className = 'pill-btn'; back.textContent = 'Back to run';
  back.onclick = () => openCellModal(cell);
  body.appendChild(back);

  const table = document.createElement('table'); table.className = 'metrics-table';
  const tbody = document.createElement('tbody');
  flattenMetrics(fold.all_metrics || fold.detail?.metrics || fold).forEach(([k, v]) => {
    const tr = document.createElement('tr');
    const tdK = document.createElement('td'); tdK.textContent = k; tr.appendChild(tdK);
    const tdV = document.createElement('td'); tdV.textContent = fmtMetricValue(v); tr.appendChild(tdV);
    tbody.appendChild(tr);
  });
  table.appendChild(tbody); body.appendChild(table);

  const detail = fold.detail || {};
  const chartWrap = document.createElement('div'); chartWrap.className = 'modal-charts'; body.appendChild(chartWrap);
  const loss = detail.loss_history || null;
  const fallbackLoss = (cell.loss_histories || {})[String(fold.fold_id)];
  if (loss?.length) {
    addModalLineChart(chartWrap, `Loss — fold ${fold.fold_id}`, loss.map(r => r.epoch), [
      { label: 'Generator', data: loss.map(r => r.generator_loss), color: '#38bdf8' },
      { label: 'Discriminator', data: loss.map(r => r.discriminator_loss), color: '#f87171' },
    ], { thinLine: true });
  } else if (fallbackLoss) {
    addModalLineChart(chartWrap, `Loss — fold ${fold.fold_id}`, fallbackLoss.generator.map((_,i)=>i+1), [
      { label: 'Generator', data: fallbackLoss.generator, color: '#38bdf8' },
      { label: 'Discriminator', data: fallbackLoss.discriminator, color: '#f87171' },
    ], { thinLine: true });
  }

  const matrices = detail.correlation_matrices?.transformed;
  if (matrices?.columns?.length) {
    const matrixWrap = document.createElement('div'); matrixWrap.className = 'matrix-grid';
    addCorrelationTable(matrixWrap, 'Real correlation', matrices.columns, matrices.real, 'corr');
    addCorrelationTable(matrixWrap, 'Synthetic correlation', matrices.columns, matrices.synthetic, 'corr');
    addCorrelationTable(matrixWrap, 'Synthetic − real', matrices.columns, matrices.difference, 'difference');
    body.appendChild(matrixWrap);
  } else {
    const missing = document.createElement('div');
    missing.style.cssText = 'color:var(--muted);font-size:13px;margin-top:12px;';
    missing.textContent = 'Correlation matrices are not cached for this fold yet.';
    body.appendChild(missing);
  }

  modal.classList.add('open');
}

// ---- Dataset modal (metric per encoding, switchable) ----
function openDatasetModal(dsId, data) {
  const dsLabel = data.dataset_labels[dsId] || dsId;
  const schema = (data.dataset_schema || {})[dsId] || {};
  const taskType = data.cells.find(c => c.dataset_id === dsId && c.task_type)?.task_type || '';
  document.getElementById('modal-title').textContent = dsLabel;
  document.getElementById('modal-sub').textContent =
    `Task: ${taskType || '—'} · cat ${schema.n_cat ?? '?'}/${schema.n_total ?? '?'} · n=${schema.n_rows ?? '?'}`;
  const body = document.getElementById('modal-body'); body.innerHTML = '';

  const dsCells = data.cells.filter(c => c.dataset_id === dsId && c.has_data);
  const idx = {}; dsCells.forEach(c => { idx[c.encoding_id] = c; });
  const encLabels = data.encoding_order.map(e => data.encoding_labels[e] || e);

  // Metric picker
  const pillRow = document.createElement('div'); pillRow.className = 'pill-row';
  ALL_METRICS.forEach(([key, mk], i) => {
    const btn = document.createElement('button'); btn.className = 'pill-btn' + (i===0?' active':'');
    btn.textContent = mk.label; btn.dataset.key = key; pillRow.appendChild(btn);
  });
  body.appendChild(pillRow);

  const chartWrap = document.createElement('div'); chartWrap.className = 'modal-metric-single'; body.appendChild(chartWrap);
  let curChart = null;

  const renderMetric = (key) => {
    const mk = METRIC_KEYS[key];
    const vals = data.encoding_order.map(e => (idx[e]||{})[mk.mean] ?? null);
    const ranks = isRankedDataset(data, dsId)
      ? computeRanks(data.encoding_order.map(e => ({ id: e, val: (idx[e]||{})[mk.mean] ?? null })))
      : {};
    if (curChart) { curChart.destroy(); curChart = null; chartWrap.innerHTML = ''; }
    const canvas = document.createElement('canvas'); chartWrap.appendChild(canvas);
    curChart = new Chart(canvas, {
      type: 'bar',
      data: { labels: encLabels, datasets: [{ data: vals,
        backgroundColor: encLabels.map((_,i)=>PALETTE[i%PALETTE.length]+'55'),
        borderColor: encLabels.map((_,i)=>PALETTE[i%PALETTE.length]),
        borderWidth: 1.5, borderRadius: 5 }] },
      options: {
        maintainAspectRatio: false,
        plugins: { legend: { display: false }, tooltip: { callbacks: {
          label: ctx => {
            const enc = data.encoding_order[ctx.dataIndex];
            const rank = ranks[enc];
            return `${mk.label}: ${fmtNum(ctx.parsed.y)} · rank #${rank ?? '—'}`;
          }
        }}},
        scales: {
          x: { ticks: { color:'#94a3b8', maxRotation:35, font:{size:11} }, grid:{color:'rgba(255,255,255,0.03)'} },
          y: { ticks:{color:'#94a3b8'}, grid:{color:'rgba(255,255,255,0.05)'},
               title:{display:true,text:mk.label,color:'#94a3b8',font:{size:11}} }
        }
      }
    });
    modalCharts.push(curChart);
  };

  renderMetric('wd');
  pillRow.querySelectorAll('button').forEach(btn => {
    btn.addEventListener('click', () => {
      pillRow.querySelectorAll('button').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      modalCharts = modalCharts.filter(c => c !== curChart);
      renderMetric(btn.dataset.key);
    });
  });
  modal.classList.add('open');
}

// ---- Encoding modal (metric per dataset, switchable) ----
function openEncodingModal(encId, data) {
  const encLabel = data.encoding_labels[encId] || encId;
  document.getElementById('modal-title').textContent = encLabel;
  document.getElementById('modal-sub').textContent = 'Metrics across datasets';
  const body = document.getElementById('modal-body'); body.innerHTML = '';

  const encCells = data.cells.filter(c => c.encoding_id === encId && c.has_data);
  const idx = {}; encCells.forEach(c => { idx[c.dataset_id] = c; });
  const dsOrder = data.dataset_order.filter(ds => idx[ds] && isRankedDataset(data, ds));
  const dsLabels = dsOrder.map(ds => data.dataset_labels[ds] || ds);

  const pillRow = document.createElement('div'); pillRow.className = 'pill-row';
  ALL_METRICS.forEach(([key, mk], i) => {
    const btn = document.createElement('button'); btn.className = 'pill-btn' + (i===0?' active':'');
    btn.textContent = mk.label; btn.dataset.key = key; pillRow.appendChild(btn);
  });
  body.appendChild(pillRow);

  const chartWrap = document.createElement('div'); chartWrap.className = 'modal-metric-single'; body.appendChild(chartWrap);
  let curChart = null;

  const renderMetric = (key) => {
    const mk = METRIC_KEYS[key];
    const vals = dsOrder.map(ds => (idx[ds]||{})[mk.mean] ?? null);
    const ranks = computeRanks(dsOrder.map(ds => ({ id: ds, val: (idx[ds]||{})[mk.mean] ?? null })));
    const catShares = dsOrder.map(ds => ((data.dataset_schema||{})[ds]||{}).cat_share ?? 0);
    if (curChart) { curChart.destroy(); curChart = null; chartWrap.innerHTML = ''; }
    const canvas = document.createElement('canvas'); chartWrap.appendChild(canvas);
    curChart = new Chart(canvas, {
      type: 'bar',
      data: { labels: dsLabels, datasets: [{ data: vals,
        backgroundColor: dsLabels.map((_,i) => PALETTE[i%PALETTE.length]+'55'),
        borderColor: dsLabels.map((_,i) => PALETTE[i%PALETTE.length]),
        borderWidth: 1.5, borderRadius: 5 }] },
      options: {
        maintainAspectRatio: false,
        plugins: { legend:{display:false}, tooltip:{ callbacks:{
          label: ctx => {
            const ds = dsOrder[ctx.dataIndex];
            const rank = ranks[ds];
            const cs = catShares[ctx.dataIndex];
            return `${mk.label}: ${fmtNum(ctx.parsed.y)} · rank #${rank ?? '—'} · cat ${fmtPct(cs)}`;
          }
        }}},
        scales: {
          x: { ticks:{color:'#94a3b8',maxRotation:35,font:{size:11}}, grid:{color:'rgba(255,255,255,0.03)'} },
          y: { ticks:{color:'#94a3b8'}, grid:{color:'rgba(255,255,255,0.05)'},
               title:{display:true,text:mk.label,color:'#94a3b8',font:{size:11}} }
        }
      }
    });
    modalCharts.push(curChart);
  };

  renderMetric('wd');
  pillRow.querySelectorAll('button').forEach(btn => {
    btn.addEventListener('click', () => {
      pillRow.querySelectorAll('button').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      modalCharts = modalCharts.filter(c => c !== curChart);
      renderMetric(btn.dataset.key);
    });
  });
  modal.classList.add('open');
}

function addModalLineChart(container, title, labels, datasets, opts={}) {
  const card = document.createElement('div'); card.className = 'chart-card';
  const h = document.createElement('h3'); h.textContent = title; card.appendChild(h);
  const canvas = document.createElement('canvas'); canvas.style.height = '220px';
  card.appendChild(canvas); container.appendChild(card);
  const chart = new Chart(canvas, {
    type: 'line',
    data: { labels, datasets: datasets.map(ds => ({
      label: ds.label, data: ds.data, borderColor: ds.color, backgroundColor: ds.color+'20',
      borderWidth: 2, tension: 0.2,
      pointRadius: (opts.thinLine || labels.length > 30) ? 0 : 3, fill: false,
    })) },
    options: {
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#94a3b8', boxWidth: 12 } } },
      scales: {
        x: { ticks:{color:'#94a3b8',maxTicksLimit:10}, grid:{color:'rgba(255,255,255,0.03)'} },
        y: { ticks:{color:'#94a3b8'}, grid:{color:'rgba(255,255,255,0.05)'} }
      }
    }
  });
  modalCharts.push(chart);
}

async function loadResults() {
  if (resultsLoading) return;
  resultsLoading = true;
  document.getElementById('results-status').textContent = 'Loading…';
  try {
    const r = await fetch('/api/results');
    resultsData = await r.json();
    renderResultsMatrix(resultsData);
    renderRankCharts(resultsData);
    renderCardMatrix(resultsData);
    renderSyncStatus(resultsData.sync_status);
  } catch(e) {
    document.getElementById('results-status').textContent = `Error: ${e}`;
  } finally {
    resultsLoading = false;
  }
}

function renderSyncStatus(sync) {
  if (!sync) return;
  const status = sync.status || 'idle';
  const downloaded = sync.total_downloaded_files || 0;
  const latest = (sync.cells || [])[0];
  let text = `${resultsData.cells.filter(c => c.has_data).length}/${resultsData.cells.length} cells with data`;
  text += ` · Drive sync: ${status}`;
  if (downloaded) text += ` · downloaded ${downloaded}`;
  if (latest && latest.status === 'running') text += ` · ${latest.dataset_id}/${latest.encoding_id}`;
  if (sync.error) text += ` · ${sync.error}`;
  document.getElementById('results-status').textContent = text;
}

// ---- Init ----
Chart.defaults.color = '#94a3b8';
Chart.defaults.font.family = "'Inter', system-ui, sans-serif";

refreshProgress().catch(e => { document.getElementById('meta').textContent = `Error: ${e}`; });
setInterval(() => refreshProgress().catch(() => {}), REFRESH_SEC * 1000);
setInterval(() => {
  if (document.getElementById('tab-results').classList.contains('active')) {
    loadResults();
  }
}, REFRESH_SEC * 1000);
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# App + HTTP server
# ---------------------------------------------------------------------------

class App:
    def __init__(self, *, manifest_path: Path, worksheet_name: str | None,
                 results_worksheet: str, output_root: Path, refresh_seconds: int,
                 no_sheets: bool = False) -> None:
        self.manifest_path = manifest_path
        self.worksheet_name = worksheet_name
        self.results_worksheet = results_worksheet
        self.output_root = output_root
        self.refresh_seconds = refresh_seconds
        self.no_sheets = no_sheets
        self.sync_state = DriveSyncState(output_root / ".ctgan_monitor_sync.sqlite3")
        self._drive_sync_thread: threading.Thread | None = None
        self._drive_sync_thread_lock = threading.Lock()

    def build_status_payload(self) -> dict[str, Any]:
        if self.no_sheets:
            return {"rows": [], "counts": {}, "spreadsheet_title": "—", "worksheet_name": "—",
                    "spreadsheet_url": "", "error": "Sheets disabled (--no-sheets)"}
        return build_status_payload(manifest_path=self.manifest_path,
                                    worksheet_name=self.worksheet_name)

    def build_results_payload(self) -> dict[str, Any]:
        if not self.no_sheets:
            self._maybe_start_drive_sync()
        rank_excluded_dataset_ids: list[str] = []
        if not self.no_sheets and self.worksheet_name:
            try:
                manifest = load_ctgan_manifest(
                    self.manifest_path,
                    project_root=self.manifest_path.resolve().parents[1],
                )
                status_payload = build_status_payload(
                    manifest_path=self.manifest_path,
                    worksheet_name=self.worksheet_name,
                )
                rank_excluded_dataset_ids = skipped_dataset_ids_from_status_payload(
                    status_payload,
                    manifest=manifest,
                )
            except Exception as exc:
                print(f"[sheets] Rank exclusion status error: {exc}", file=sys.stderr, flush=True)
        payload = build_results_payload(
            manifest_path=self.manifest_path,
            results_worksheet="" if self.no_sheets else self.results_worksheet,
            output_root=self.output_root,
            sync_drive=False,
            rank_excluded_dataset_ids=rank_excluded_dataset_ids,
        )
        payload["sync_status"] = self.sync_state.summary()
        return payload

    def build_sync_status_payload(self) -> dict[str, Any]:
        if not self.no_sheets:
            self._maybe_start_drive_sync()
        return self.sync_state.summary()

    def _maybe_start_drive_sync(self) -> None:
        with self._drive_sync_thread_lock:
            if self._drive_sync_thread and self._drive_sync_thread.is_alive():
                return
            self._drive_sync_thread = threading.Thread(
                target=self._drive_sync_loop,
                name="ctgan-drive-sync",
                daemon=True,
            )
            self._drive_sync_thread.start()

    def _drive_sync_loop(self) -> None:
        interval = max(int(self.refresh_seconds), 60)
        while True:
            self._run_drive_sync_once()
            time.sleep(interval)

    def _run_drive_sync_once(self) -> None:
        if self.no_sheets:
            return
        drive_client = _get_drive_artifact_client()
        if drive_client is None:
            self.sync_state.finish_run(status="disabled", error="Drive credentials unavailable")
            return

        self.sync_state.start_run()
        try:
            matrix, _, _, _ = _sheets_read(self.results_worksheet)
            headers, sheet_rows = _parse_sheet_rows(matrix)
            manifest = load_ctgan_manifest(self.manifest_path, project_root=self.manifest_path.resolve().parents[1])
            dataset_ids_by_label = {entry.label: entry.dataset_id for entry in manifest.datasets}
            encoding_ids_by_label = {entry.label: entry.encoding_id for entry in manifest.encodings}

            for row in sheet_rows:
                ds_label = str(row.get("Dataset", "")).strip()
                enc_label = str(row.get("Categorical representation", "")).strip()
                folder_url = str(row.get("Drive folder URL", "")).strip()
                dataset_id = dataset_ids_by_label.get(ds_label)
                encoding_id = encoding_ids_by_label.get(enc_label)
                if not (dataset_id and encoding_id and folder_url):
                    continue

                self.sync_state.record_cell_start(
                    dataset_id=dataset_id,
                    encoding_id=encoding_id,
                    folder_url=folder_url,
                )
                try:
                    stats = _sync_drive_artifacts(
                        drive_client=drive_client,
                        folder_url=folder_url,
                        run_dir=self.output_root / dataset_id / encoding_id,
                    )
                    self.sync_state.record_cell_finish(
                        dataset_id=dataset_id,
                        encoding_id=encoding_id,
                        status="ok",
                        downloaded_files=stats.get("downloaded", 0),
                        error="",
                    )
                except Exception as exc:
                    self.sync_state.record_cell_finish(
                        dataset_id=dataset_id,
                        encoding_id=encoding_id,
                        status="error",
                        downloaded_files=0,
                        error=str(exc),
                    )
                    print(
                        f"[drive] Artifact sync error for {ds_label} / {enc_label}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )

            self.sync_state.finish_run(status="ok", error="")
        except Exception as exc:
            self.sync_state.finish_run(status="error", error=str(exc))
            print(f"[drive] Sync run error: {exc}", file=sys.stderr, flush=True)

    def clear_cell(self, coord: str) -> dict[str, Any]:
        normalized = coord.strip().upper()
        if not normalized:
            raise ValueError("coord is required")
        status_payload = self.build_status_payload()
        valid_coords = {cell["coord"] for row in status_payload["rows"] for cell in row["cells"]}
        if normalized not in valid_coords:
            raise ValueError(f"unknown coord: {normalized}")
        config = SheetsConfig.from_env()
        if self.worksheet_name:
            config = replace(config, worksheet_name=self.worksheet_name)
        client = SheetsClient(config)
        client.write_cell(normalized, "")
        return {"ok": True, "coord": normalized}

    def make_handler(self) -> type[BaseHTTPRequestHandler]:
        app = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urlparse(self.path)
                if parsed.path == "/":
                    self._send_html(html_page(app.refresh_seconds))
                elif parsed.path == "/api/status":
                    self._send_json(app.build_status_payload())
                elif parsed.path == "/api/results":
                    self._send_json(app.build_results_payload())
                elif parsed.path == "/api/sync-status":
                    self._send_json(app.build_sync_status_payload())
                elif parsed.path == "/healthz":
                    self._send_json({"ok": True})
                elif parsed.path == "/favicon.ico":
                    self._send_no_content()
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)

            def do_POST(self):
                parsed = urlparse(self.path)
                if parsed.path != "/api/clear":
                    self.send_error(HTTPStatus.NOT_FOUND); return
                coord = parse_qs(parsed.query).get("coord", [""])[0]
                try:
                    self._send_json(app.clear_cell(coord))
                except ValueError as e:
                    self._send_json({"ok": False, "error": str(e)}, status=HTTPStatus.BAD_REQUEST)
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

            def log_message(self, fmt, *args):
                print(f"[{self.log_date_time_string()}] {fmt % args}", flush=True)

            def _send_html(self, payload: str):
                body = payload.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers(); self.wfile.write(body)

            def _send_json(self, payload, *, status=HTTPStatus.OK):
                body = json.dumps(payload, default=str).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers(); self.wfile.write(body)

            def _send_no_content(self):
                self.send_response(HTTPStatus.NO_CONTENT)
                self.send_header("Cache-Control", "max-age=86400")
                self.send_header("Content-Length", "0")
                self.end_headers()

        return Handler


def _column_name(index: int) -> str:
    name = ""
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        name = chr(ord("A") + remainder) + name
    return name


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    railway_port = os.getenv("PORT")
    default_port = int(railway_port) if railway_port and railway_port.isdigit() else 8765
    default_host = "0.0.0.0" if railway_port else "127.0.0.1"
    parser = argparse.ArgumentParser(description="CTGAN Monitor — progress + results dashboard")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--worksheet", help="Progress worksheet name (orchestrator matrix)")
    parser.add_argument("--results-worksheet", default="Results", help="Results sheet name (default: Results)")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT),
                        help="Root dir with per-dataset per-encoding artifacts")
    parser.add_argument("--host", default=default_host)
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument("--refresh-seconds", type=int, default=30)
    parser.add_argument("--no-sheets", action="store_true",
                        help="Skip Google Sheets entirely (only local artifacts)")
    args = parser.parse_args(raw_argv)
    port_is_explicit = "--port" in raw_argv or any(arg.startswith("--port=") for arg in raw_argv)

    app = App(
        manifest_path=Path(args.manifest).resolve(),
        worksheet_name=args.worksheet,
        results_worksheet="" if args.no_sheets else args.results_worksheet,
        output_root=Path(args.output_root),
        refresh_seconds=max(int(args.refresh_seconds), 1),
        no_sheets=args.no_sheets,
    )
    handler = app.make_handler()
    server, bound_port = _bind_server(args.host, args.port, handler, allow_port_fallback=not port_is_explicit)
    print(f"CTGAN Monitor: http://{args.host}:{bound_port}", flush=True)
    print(f"  Progress sheet: {'disabled' if args.no_sheets else (args.worksheet or '(from env)')}", flush=True)
    print(f"  Results sheet:  {'disabled' if args.no_sheets else args.results_worksheet}", flush=True)
    print(f"  Artifacts root: {args.output_root}", flush=True)
    server.serve_forever()
    return 0


def _bind_server(
    host: str,
    port: int,
    handler: type[BaseHTTPRequestHandler],
    *,
    allow_port_fallback: bool,
) -> tuple[ThreadingHTTPServer, int]:
    max_attempts = 20 if allow_port_fallback else 1
    for offset in range(max_attempts):
        candidate_port = port + offset
        try:
            server = ThreadingHTTPServer((host, candidate_port), handler)
            bound_port = int(getattr(server, "server_address", (host, candidate_port))[1])
            if offset:
                print(
                    f"Port {port} is busy; using {bound_port} instead.",
                    file=sys.stderr,
                    flush=True,
                )
            return server, bound_port
        except OSError as e:
            if e.errno != errno.EADDRINUSE or offset == max_attempts - 1:
                raise
    raise RuntimeError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
