from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import experiments.ctgan.ctgan_full_experiment as full_mod
from experiments.ctgan.experiment_models import ExperimentModelSpec, get_experiment_model
from experiments.ctgan.orchestrator_staff import ctgan_drive as drive_mod
from experiments.ctgan.orchestrator_staff.ctgan_manifest import (
    CtganManifest,
    DatasetEntry,
    EncodingEntry,
    load_ctgan_manifest,
)
from experiments.ctgan.orchestrator_staff.ctgan_orchestrator_state import (
    parse_cell_payload,
    validate_worksheet_headers,
)
from experiments.ctgan.orchestrator_staff.ctgan_sheets import SheetsConfig
from genbench.data.datamodule import TabularDataModule
from genbench.data.schema import TabularSchema
from genbench.data.splits import SplitConfigHoldout, SplitConfigKFold
from genbench.evaluation.distribution.corr_frobenius import CorrelationFrobeniusMetric
from genbench.evaluation.distribution.marginal_kl import MarginalKLDivergenceMetric
from genbench.evaluation.distribution.wasserstein import WassersteinDistanceMetric
from genbench.evaluation.pipeline.single_run import DistributionEvaluationPipeline
from genbench.generative.ctgan.ctgan import CtganGenerative
from genbench.generative.tvae.tvae import TvaeGenerative


DEFAULT_MANIFEST = Path("experiments/ctgan/orchestrator_staff/ctgan_orchestrator_manifest.json")
DEFAULT_OUTPUT_ROOT = Path("experiments/results/recomputed_ctgan")
_FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
_RETRY_DELAYS = (1, 2, 4, 8, 16)
_DRIVE_SCOPES = (
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
)

# Columns written to the results spreadsheet.
# WD, KL, Corr dist are computed on the *unencoded* (original) feature space
# so results are comparable across encodings regardless of dimensionality.
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


def _retry(func: Callable[[], Any]) -> Any:
    last_error: Exception | None = None
    for delay in _RETRY_DELAYS:
        try:
            return func()
        except Exception as exc:  # noqa: BLE001 - Google APIs raise several transient exception types
            last_error = exc
            import time

            time.sleep(delay)
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


@dataclass(frozen=True)
class LocalDriveConfig:
    results_folder_id: str
    service_account_path: Path | None = None
    service_account_info: dict[str, Any] | None = None
    oauth_token_path: Path | None = None

    @classmethod
    def from_env(cls) -> "LocalDriveConfig":
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
            raise ValueError("Drive configuration is incomplete, missing: " + ", ".join(missing))

        return cls(
            results_folder_id=results_folder_id,
            service_account_path=Path(service_account_path).expanduser() if service_account_path else None,
            service_account_info=inline_info,
            oauth_token_path=Path(oauth_token_path).expanduser() if oauth_token_path else None,
        )


class LocalDriveReader:
    def __init__(self, config: LocalDriveConfig) -> None:
        self.config = config
        self._service = self._build_service(config)

    @staticmethod
    def _build_credentials(config: LocalDriveConfig) -> Any:
        if config.oauth_token_path is not None:
            try:
                from google.auth.transport.requests import Request
                from google.oauth2.credentials import Credentials
            except ImportError as exc:
                raise RuntimeError("google-auth is required for Drive access.") from exc

            creds = Credentials.from_authorized_user_file(str(config.oauth_token_path))
            if creds.expired and creds.refresh_token:
                creds.refresh(Request())
                config.oauth_token_path.write_text(creds.to_json(), encoding="utf-8")
            return creds

        try:
            from google.oauth2 import service_account
        except ImportError as exc:
            raise RuntimeError("google-auth is required for Drive access.") from exc

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
        raise ValueError("LocalDriveConfig must define OAuth or service-account credentials.")

    @classmethod
    def _build_service(cls, config: LocalDriveConfig) -> Any:
        try:
            from googleapiclient.discovery import build
        except ImportError as exc:
            raise RuntimeError("google-api-python-client is required for Drive access.") from exc
        return build("drive", "v3", credentials=cls._build_credentials(config), cache_discovery=False)

    def find_folder(self, name: str, parent_id: str) -> str | None:
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
        current_parent = root_id or self.config.results_folder_id
        for part in path_parts:
            found = self.find_folder(part, current_parent)
            if found is None:
                return None
            current_parent = found
        return current_parent

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
                item_id = str(item["id"])
                name = str(item["name"])
                mime_type = str(item.get("mimeType", ""))
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

    def download_file(self, file_id: str, destination: Path) -> None:
        try:
            from googleapiclient.http import MediaIoBaseDownload
        except ImportError as exc:
            raise RuntimeError("google-api-python-client is required for Drive downloads.") from exc

        destination.parent.mkdir(parents=True, exist_ok=True)

        def _download() -> None:
            request = self._service.files().get_media(fileId=file_id, supportsAllDrives=True)
            with destination.open("wb") as handle:
                downloader = MediaIoBaseDownload(handle, request)
                done = False
                while not done:
                    _status, done = downloader.next_chunk()

        _retry(_download)

    def folder_web_url(self, folder_id: str) -> str:
        return f"https://drive.google.com/drive/folders/{folder_id}"

    def create_folder(self, name: str, parent_id: str) -> str:
        """Create a subfolder and return its ID."""
        metadata = {
            "name": name,
            "mimeType": _FOLDER_MIME_TYPE,
            "parents": [parent_id],
        }

        def _create() -> str:
            result = (
                self._service.files()
                .create(body=metadata, fields="id", supportsAllDrives=True)
                .execute()
            )
            return str(result["id"])

        return _retry(_create)

    def ensure_folder_path(self, *path_parts: str, root_id: str | None = None) -> str:
        """Find or create nested subfolders and return the leaf folder ID."""
        current_parent = root_id or self.config.results_folder_id
        for part in path_parts:
            found = self.find_folder(part, current_parent)
            if found is None:
                found = self.create_folder(part, current_parent)
            current_parent = found
        return current_parent

    def upload_file(self, local_path: Path, parent_id: str, *, overwrite: bool = True) -> str:
        """Upload a local file to Drive, optionally overwriting an existing file with the same name.

        Returns the Drive file ID.
        """
        try:
            from googleapiclient.http import MediaFileUpload
        except ImportError as exc:
            raise RuntimeError("google-api-python-client is required for Drive uploads.") from exc

        name = local_path.name

        # Check if a file with this name already exists in the parent folder
        existing_id: str | None = None
        if overwrite:
            query = (
                f"name = {json.dumps(name)} "
                f"and '{parent_id}' in parents "
                f"and mimeType != '{_FOLDER_MIME_TYPE}' "
                f"and trashed = false"
            )

            def _search() -> list[dict[str, Any]]:
                return (
                    self._service.files()
                    .list(
                        q=query,
                        fields="files(id)",
                        spaces="drive",
                        supportsAllDrives=True,
                        includeItemsFromAllDrives=True,
                    )
                    .execute()
                    .get("files", [])
                )

            existing = _retry(_search)
            if existing:
                existing_id = str(existing[0]["id"])

        media = MediaFileUpload(str(local_path), resumable=False)

        if existing_id is not None:
            def _update() -> str:
                result = (
                    self._service.files()
                    .update(
                        fileId=existing_id,
                        media_body=media,
                        fields="id",
                        supportsAllDrives=True,
                    )
                    .execute()
                )
                return str(result["id"])

            return _retry(_update)
        else:
            metadata = {"name": name, "parents": [parent_id]}

            def _insert() -> str:
                result = (
                    self._service.files()
                    .create(
                        body=metadata,
                        media_body=media,
                        fields="id",
                        supportsAllDrives=True,
                    )
                    .execute()
                )
                return str(result["id"])

            return _retry(_insert)


@dataclass(frozen=True)
class RecomputedPair:
    dataset_id: str
    encoding_id: str
    output_dir: Path
    aggregate_metrics_path: Path
    row: list[str]


@dataclass(frozen=True)
class RecomputeFromDriveResult:
    rows: list[list[str]]
    pairs: list[RecomputedPair] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class UploadedLocalPair:
    dataset_id: str
    encoding_id: str
    run_dir: Path
    aggregate_metrics_path: Path
    folder_url: str
    row: list[str]


@dataclass(frozen=True)
class UploadMissingLocalResult:
    uploaded: list[UploadedLocalPair] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)


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


def sort_drive_file_records_newest_first(records: list[Any]) -> list[Any]:
    return sorted(
        records,
        key=lambda record: _parse_drive_modified_time(str(record.modified_time)),
        reverse=True,
    )


def drive_records_by_relative_path(records: list[Any]) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = {}
    for record in records:
        grouped.setdefault(str(record.relative_path), []).append(record)
    return {
        relative_path: sort_drive_file_records_newest_first(path_records)
        for relative_path, path_records in grouped.items()
    }


def download_first_valid_drive_file(
    *,
    drive_client: Any,
    candidates: list[Any],
    destination: Path,
    validator: Callable[[Path], bool],
) -> Any:
    failures: list[str] = []
    for record in sort_drive_file_records_newest_first(candidates):
        try:
            if destination.exists():
                destination.unlink()
            drive_client.download_file(str(record.file_id), destination)
            if validator(destination):
                return record
            failures.append(f"{record.file_id}: validator rejected file")
        except Exception as exc:  # noqa: BLE001 - corrupt Drive clones should fall through
            failures.append(f"{record.file_id}: {exc}")
    if destination.exists():
        destination.unlink()
    logical_path = candidates[0].relative_path if candidates else "<empty candidates>"
    raise FileNotFoundError(f"No valid Drive file found for {logical_path}: " + "; ".join(failures))


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
        aggregate,
        "f1_weighted_pct_diff",
        field,
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


def _worksheet_values(spreadsheet: Any, worksheet_name: str) -> list[list[str]]:
    worksheet = spreadsheet.worksheet(worksheet_name)
    values = worksheet.get_all_values()
    if not isinstance(values, list):
        raise ValueError(f"worksheet {worksheet_name!r} must return a matrix.")
    return values


def _results_row_keys(matrix: list[list[str]]) -> set[tuple[str, str, str]]:
    if not matrix:
        return set()
    header = matrix[0]
    if not isinstance(header, list):
        return set()
    if all(column in header for column in ("Model", "Dataset", "Categorical representation")):
        model_idx = header.index("Model")
        dataset_idx = header.index("Dataset")
        encoding_idx = header.index("Categorical representation")
    else:
        model_idx, dataset_idx, encoding_idx = 0, 1, 2

    keys: set[tuple[str, str, str]] = set()
    for row in matrix[1:]:
        if not isinstance(row, list):
            continue
        if len(row) <= max(model_idx, dataset_idx, encoding_idx):
            continue
        key = (
            row[model_idx].strip(),
            row[dataset_idx].strip(),
            row[encoding_idx].strip(),
        )
        if all(key):
            keys.add(key)
    return keys


def _done_status_pairs(
    *,
    matrix: list[list[str]],
    manifest: CtganManifest,
    model_id: str,
) -> list[tuple[DatasetEntry, EncodingEntry]]:
    if not matrix:
        return []
    header_row = matrix[0]
    if not isinstance(header_row, list):
        raise ValueError("status worksheet header row must be a list.")

    dataset_headers, encoding_headers = validate_worksheet_headers(
        dataset_headers=header_row[1:],
        encoding_headers=[row[0] if isinstance(row, list) and row else "" for row in matrix[1:]],
        manifest_dataset_labels=[entry.label for entry in manifest.datasets],
        manifest_encoding_labels=[entry.label for entry in manifest.encodings],
    )

    done_pairs: list[tuple[DatasetEntry, EncodingEntry]] = []
    for row_index, encoding_label in enumerate(encoding_headers, start=1):
        row = matrix[row_index] if row_index < len(matrix) and isinstance(matrix[row_index], list) else []
        for col_index, dataset_label in enumerate(dataset_headers, start=1):
            raw_value = row[col_index] if col_index < len(row) else None
            payload = parse_cell_payload(raw_value)
            if payload.status != "done":
                continue
            if payload.model_id not in {None, model_id}:
                continue
            done_pairs.append(
                (
                    manifest.resolve_dataset_label(dataset_label),
                    manifest.resolve_encoding_label(encoding_label),
                )
            )
    return done_pairs


def _missing_local_run_reason(run_dir: Path) -> str | None:
    if not run_dir.is_dir():
        return f"local run directory not found: {run_dir}"
    if not (run_dir / "run_summary.json").is_file():
        return f"run_summary.json not found in {run_dir}"
    if not (run_dir / "metrics" / "aggregate.json").is_file():
        return f"metrics/aggregate.json not found in {run_dir}"
    return None


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
        except Exception:  # noqa: BLE001 - absent sheet is fine
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


def _build_spreadsheet_from_config(config: SheetsConfig) -> Any:
    try:
        import gspread
        from google.oauth2 import service_account
    except ImportError as exc:
        raise RuntimeError("gspread and google-auth are required for Sheets access.") from exc

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

    client = gspread.authorize(credentials)
    from experiments.ctgan.orchestrator_staff.ctgan_sheets import retry_call

    return retry_call(lambda: client.open_by_key(config.spreadsheet_id))


def refresh_full_results_worksheet_from_config(
    *,
    sheets_config: SheetsConfig,
    worksheet_name: str,
    rows: list[list[str]],
    archive_existing: bool,
) -> None:
    spreadsheet = _build_spreadsheet_from_config(sheets_config)
    refresh_full_results_worksheet(
        spreadsheet=spreadsheet,
        worksheet_name=worksheet_name,
        rows=rows,
        archive_existing=archive_existing,
    )


class IncrementalResultsWorksheetWriter:
    def __init__(self, worksheet: Any) -> None:
        self._worksheet = worksheet
        self._rows: list[list[str]] = []

    @classmethod
    def create(
        cls,
        *,
        spreadsheet: Any,
        worksheet_name: str,
        archive_existing: bool,
    ) -> "IncrementalResultsWorksheetWriter":
        worksheet = refresh_full_results_worksheet(
            spreadsheet=spreadsheet,
            worksheet_name=worksheet_name,
            rows=[],
            archive_existing=archive_existing,
        )
        return cls(worksheet)

    def append_row(self, row: list[str]) -> None:
        self._rows.append(row)
        _update_worksheet_values(self._worksheet, [FULL_RESULTS_HEADERS, *self._rows])


def _upload_recomputed_jsons(
    *,
    drive_client: Any,
    run_dir: Path,
    folder_id: str,
) -> None:
    """Upload recomputed JSON result files back to the same Drive folder.

    Uploads metrics/aggregate.json, crossval/per_fold/fold_*.json, and
    run_summary.json — overwriting the old versions on Drive so the Drive
    folder stays in sync with the recomputed results.
    """
    json_files = [
        run_dir / "metrics" / "aggregate.json",
        run_dir / "run_summary.json",
    ]
    per_fold_dir = run_dir / "crossval" / "per_fold"
    if per_fold_dir.exists():
        json_files.extend(sorted(per_fold_dir.glob("fold_*.json")))

    uploaded = 0
    for local_path in json_files:
        if not local_path.exists():
            continue
        # Relative path within the run_dir → mirrors the Drive subfolder structure
        relative = local_path.relative_to(run_dir)
        # Ensure the subfolder exists on Drive
        parts = list(relative.parts)
        if len(parts) > 1:
            parent_id = drive_client.ensure_folder_path(*parts[:-1], root_id=folder_id)
        else:
            parent_id = folder_id
        drive_client.upload_file(local_path, parent_id, overwrite=True)
        uploaded += 1

    print(f"[ctgan_recompute_from_drive] uploaded {uploaded} recomputed JSON(s) back to Drive")


def _compute_distribution_scores(
    *,
    test_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    transformed_schema: TabularSchema,
    original_schema: TabularSchema,
) -> dict[str, float]:
    transformed_pipeline = DistributionEvaluationPipeline(
        metrics=[
            WassersteinDistanceMetric(),
            MarginalKLDivergenceMetric(),
            CorrelationFrobeniusMetric(),
        ]
    )
    transformed_scores = transformed_pipeline.evaluate(
        real=test_df,
        synth=synth_df,
        schema=transformed_schema,
    ).scores
    unencoded_pipeline = DistributionEvaluationPipeline(
        metrics=[
            WassersteinDistanceMetric(
                name="wasserstein_mean_unencoded",
                include_discrete=False,
            ),
            MarginalKLDivergenceMetric(
                name="marginal_kl_mean_unencoded",
                include_categorical=False,
            ),
            CorrelationFrobeniusMetric(
                name="corr_frobenius_unencoded",
                include_categorical=False,
                method="spearman",
            ),
        ]
    )
    unencoded_scores = unencoded_pipeline.evaluate(
        real=test_df,
        synth=synth_df,
        schema=original_schema,
    ).scores
    return {
        "wasserstein_mean": float(transformed_scores["wasserstein_mean"]),
        "marginal_kl_mean": float(transformed_scores["marginal_kl_mean"]),
        "corr_frobenius_transformed": float(transformed_scores["corr_frobenius"]),
        "wasserstein_mean_unencoded": float(unencoded_scores["wasserstein_mean_unencoded"]),
        "marginal_kl_mean_unencoded": float(unencoded_scores["marginal_kl_mean_unencoded"]),
        "corr_frobenius_unencoded": float(unencoded_scores["corr_frobenius_unencoded"]),
    }


def _load_manifest_in_json_order(manifest_path: Path, *, project_root: Path) -> CtganManifest:
    manifest = load_ctgan_manifest(manifest_path, project_root=project_root)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return manifest

    dataset_by_id = {entry.dataset_id: entry for entry in manifest.datasets}
    encoding_by_id = {entry.encoding_id: entry for entry in manifest.encodings}

    ordered_datasets: list[DatasetEntry] = []
    for item in payload.get("datasets", []):
        if isinstance(item, dict):
            dataset_id = item.get("dataset_id")
            if isinstance(dataset_id, str) and dataset_id in dataset_by_id:
                ordered_datasets.append(dataset_by_id[dataset_id])

    ordered_encodings: list[EncodingEntry] = []
    for item in payload.get("encodings", []):
        if isinstance(item, dict):
            encoding_id = item.get("encoding_id")
            if isinstance(encoding_id, str) and encoding_id in encoding_by_id:
                ordered_encodings.append(encoding_by_id[encoding_id])

    return CtganManifest(
        datasets=tuple(ordered_datasets) if ordered_datasets else manifest.datasets,
        encodings=tuple(ordered_encodings) if ordered_encodings else manifest.encodings,
    )


def _load_json_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _valid_json_file(path: Path) -> bool:
    _load_json_file(path)
    return True


def _valid_model_artifact(
    path: Path,
    artifact_filename: str = "ctgan.pkl",
    model_spec: ExperimentModelSpec | None = None,
) -> bool:
    if path.name != artifact_filename or not path.exists() or path.stat().st_size <= 0:
        return False
    _load_model_artifacts_cpu(path.parent, model_spec or get_experiment_model("ctgan"))
    return True


def _valid_ctgan_artifact(path: Path) -> bool:
    return _valid_model_artifact(path, "ctgan.pkl", get_experiment_model("ctgan"))


def _torch_load_cpu_from_bytes(buffer: bytes) -> Any:
    import torch

    try:
        return torch.load(
            io.BytesIO(buffer),
            map_location=torch.device("cpu"),
            weights_only=False,
        )
    except TypeError:
        return torch.load(io.BytesIO(buffer), map_location=torch.device("cpu"))


def _load_ctgan_artifacts_cpu(path: Path) -> CtganGenerative:
    return _load_model_artifacts_cpu(path, get_experiment_model("ctgan"))


def _load_model_artifacts_cpu(path: Path, model_spec: ExperimentModelSpec) -> Any:
    path = path.resolve()
    bundle_path = path / model_spec.artifact_filename
    if not bundle_path.exists():
        raise FileNotFoundError(f"{model_spec.artifact_filename} not found in {path}")

    patched = False
    original_load_from_bytes: Any | None = None
    try:
        import torch

        original_load_from_bytes = getattr(torch.storage, "_load_from_bytes", None)
        if original_load_from_bytes is not None:
            torch.storage._load_from_bytes = _torch_load_cpu_from_bytes
            patched = True
    except ImportError:
        torch = None  # type: ignore[assignment]

    try:
        with bundle_path.open("rb") as handle:
            payload = pickle.load(handle)
    finally:
        if patched:
            torch.storage._load_from_bytes = original_load_from_bytes  # type: ignore[union-attr]

    if model_spec.model_id == "ctgan":
        obj: Any = CtganGenerative()
    elif model_spec.model_id == "tvae":
        obj = TvaeGenerative()
    else:
        raise ValueError(f"Unsupported recompute model_id: {model_spec.model_id}")

    obj.model_ = payload.get("model")
    obj.used_discrete_cols_ = payload.get("used_discrete_cols", [])
    obj.fitted_ = bool(payload.get("fitted", obj.model_ is not None))
    if obj.model_ is not None and hasattr(obj.model_, "set_device"):
        obj.model_.set_device("cpu")
    return obj


def _prepare_holdout_data(
    *,
    df: pd.DataFrame,
    schema: TabularSchema,
    encoding_method: str,
    is_regression: bool | None,
    random_seed: int,
) -> full_mod.PreparedFoldData:
    task_type = full_mod._task_type_from_flag(is_regression)
    pipeline, _ = full_mod.build_preprocess_pipeline(
        schema=schema,
        encoding_method=encoding_method,
        task_type=task_type,
    )
    dm = TabularDataModule(
        df=df,
        schema=schema,
        transforms=pipeline,
        unseen_category_policy="move_to_train",
        validate=True,
    )
    dm.prepare_holdout(SplitConfigHoldout(val_size=0.2, shuffle=True, random_seed=random_seed))
    holdout = dm.get_holdout()
    if holdout.train_raw is None or holdout.val_raw is None:
        raise RuntimeError("HoldoutData must include raw train/val data.")

    train_df = holdout.train.reset_index(drop=True)
    test_df = holdout.val.reset_index(drop=True)
    transformed_schema = TabularSchema.infer_from_dataframe(
        train_df,
        target_col=schema.target_col,
        id_col=schema.id_col,
    )
    transformed_schema.validate(test_df)
    return full_mod.PreparedFoldData(
        train_raw=holdout.train_raw.reset_index(drop=True),
        test_raw=holdout.val_raw.reset_index(drop=True),
        train_transformed=train_df,
        test_transformed=test_df,
        transformed_schema=transformed_schema,
        transforms=holdout.transforms,
    )


def _prepare_split_data(
    *,
    df: pd.DataFrame,
    schema: TabularSchema,
    encoding_method: str,
    is_regression: bool | None,
    poster_fast: bool,
    fold_id: int,
    n_folds: int,
    random_seed: int,
) -> full_mod.PreparedFoldData:
    if poster_fast:
        return _prepare_holdout_data(
            df=df,
            schema=schema,
            encoding_method=encoding_method,
            is_regression=is_regression,
            random_seed=random_seed,
        )
    return full_mod._prepare_fold_data(
        df=df,
        schema=schema,
        encoding_method=encoding_method,
        is_regression=is_regression,
        split_cfg=SplitConfigKFold(n_splits=n_folds, shuffle=True, random_seed=random_seed),
        fold_id=fold_id,
    )


def _evaluate_saved_model_fold(
    *,
    fold_id: int,
    split_data: full_mod.PreparedFoldData,
    schema: TabularSchema,
    model_dir: Path,
    is_regression: bool | None,
    source_model_record: DriveFileRecord,
    model_spec: ExperimentModelSpec | None = None,
) -> dict[str, Any]:
    model = _load_model_artifacts_cpu(model_dir, model_spec or get_experiment_model("ctgan"))
    synth_df = model.sample(len(split_data.train_transformed)).reset_index(drop=True)

    distribution_scores = _compute_distribution_scores(
        test_df=split_data.test_transformed,
        synth_df=synth_df,
        transformed_schema=split_data.transformed_schema,
        original_schema=schema,
    )
    corr_original_value, corr_original_status = full_mod._compute_original_space_corr(
        test_raw=split_data.test_raw,
        synth_df=synth_df,
        schema=schema,
        transforms=split_data.transforms,
    )
    distribution_scores["corr_frobenius_original"] = corr_original_value
    distribution_scores["corr_frobenius_original_status"] = corr_original_status

    tstr_scores = full_mod._compute_tstr_scores(
        train_df=split_data.train_transformed,
        test_df=split_data.test_transformed,
        synth_df=synth_df,
        transformed_schema=split_data.transformed_schema,
        is_regression=is_regression,
    )

    return {
        "fold_id": int(fold_id),
        "n_train": int(len(split_data.train_transformed)),
        "n_test": int(len(split_data.test_transformed)),
        "distribution": distribution_scores,
        "utility": tstr_scores,
        "source_artifacts": {
            "model_file_id": source_model_record.file_id,
            "model_relative_path": source_model_record.relative_path,
            "model_modified_time": source_model_record.modified_time,
        },
    }


def _aggregate_fold_results(
    *,
    dataset: DatasetEntry,
    encoding: EncodingEntry,
    n_folds: int,
    fold_results: list[dict[str, Any]],
) -> dict[str, Any]:
    distribution_records = [fold["distribution"] for fold in fold_results]
    utility_records = [
        {key: value for key, value in fold["utility"].items() if key not in {"status", "task_type"}}
        for fold in fold_results
        if fold["utility"].get("status") == "ok"
    ]
    utility_status = fold_results[0]["utility"]["status"] if fold_results else "unsupported_no_target"
    utility_task_type = (
        fold_results[0]["utility"].get("task_type")
        if fold_results and fold_results[0]["utility"].get("status") == "ok"
        else None
    )
    original_corr_statuses = [
        str(fold["distribution"].get("corr_frobenius_original_status", ""))
        for fold in fold_results
        if fold.get("distribution")
    ]
    return {
        "dataset_id": dataset.dataset_id,
        "dataset_label": dataset.label,
        "encoding_method": encoding.encoding_id,
        "encoding_label": encoding.label,
        "n_folds": n_folds,
        "distribution": full_mod._aggregate_numeric_records(distribution_records),
        "distribution_status": {
            "corr_frobenius_original_status": (
                "ok" if original_corr_statuses and all(status == "ok" for status in original_corr_statuses)
                else (original_corr_statuses[0] if original_corr_statuses else "")
            )
        },
        "tstr": {
            "status": utility_status,
            "task_type": utility_task_type,
            "metrics": full_mod._aggregate_numeric_records(utility_records),
        },
    }


def _effective_recompute_dataframe(
    *,
    df: pd.DataFrame,
    source_run_summary: dict[str, Any],
) -> pd.DataFrame:
    poster_payload = source_run_summary.get("poster_fast", {})
    if not isinstance(poster_payload, dict) or not poster_payload.get("enabled"):
        return df.reset_index(drop=True)

    max_rows = poster_payload.get("max_rows")
    effective_rows = poster_payload.get("effective_rows")
    if isinstance(max_rows, int):
        return full_mod._cap_dataframe_rows(df, max_rows=max_rows)
    if isinstance(effective_rows, int) and 1 < effective_rows < len(df):
        return full_mod._cap_dataframe_rows(df, max_rows=effective_rows)
    return df.reset_index(drop=True)


def _download_source_summary(
    *,
    drive_client: Any,
    records_by_path: dict[str, list[DriveFileRecord]],
    destination: Path,
) -> tuple[dict[str, Any], DriveFileRecord]:
    candidates = records_by_path.get("run_summary.json", [])
    if not candidates:
        raise FileNotFoundError("run_summary.json not found in Drive folder.")
    record = download_first_valid_drive_file(
        drive_client=drive_client,
        candidates=candidates,
        destination=destination,
        validator=_valid_json_file,
    )
    return _load_json_file(destination), record


def _recompute_pair_from_drive(
    *,
    manifest_path: Path,
    project_root: Path,
    dataset: DatasetEntry,
    encoding: EncodingEntry,
    output_root: Path,
    drive_client: Any,
    random_seed: int,
    model_id: str = "ctgan",
    model_name: str | None = None,
) -> RecomputedPair | None:
    model_spec = get_experiment_model(model_id)
    display_name = model_name or model_spec.display_name
    folder_id = drive_client.find_folder_path(display_name, dataset.dataset_id, encoding.encoding_id)
    if folder_id is None:
        return None

    run_dir = output_root / dataset.dataset_id / encoding.encoding_id
    source_dir = run_dir / "source_drive"
    records = drive_client.list_files_recursive(folder_id)
    records_by_path = drive_records_by_relative_path(records)

    source_run_summary, source_summary_record = _download_source_summary(
        drive_client=drive_client,
        records_by_path=records_by_path,
        destination=source_dir / "run_summary.json",
    )

    poster_fast = bool(source_run_summary.get("poster_fast", {}).get("enabled"))
    n_folds = int(source_run_summary.get("crossval", {}).get("n_folds", 1 if poster_fast else 5))

    df = pd.read_csv(project_root / "datasets" / "raw" / f"{dataset.dataset_id}.csv")
    df = _effective_recompute_dataframe(df=df, source_run_summary=source_run_summary)
    schema = TabularSchema.infer_from_dataframe(
        df,
        target_col=dataset.target_col,
        id_col=dataset.id_col,
    )
    is_regression: bool | None = None
    if dataset.target_col is not None:
        is_regression = bool(full_mod.infer_is_regression_target(df[dataset.target_col]))

    fold_results: list[dict[str, Any]] = []
    model_file_records: list[dict[str, str]] = []
    for fold_id in range(n_folds):
        logical_path = f"artifacts/fold_{fold_id}/{model_spec.artifact_filename}"
        candidates = records_by_path.get(logical_path, [])
        if not candidates:
            return None

        local_model_path = run_dir / "artifacts" / f"fold_{fold_id}" / model_spec.artifact_filename
        selected_model_record = download_first_valid_drive_file(
            drive_client=drive_client,
            candidates=candidates,
            destination=local_model_path,
            validator=lambda path: _valid_model_artifact(
                path,
                model_spec.artifact_filename,
                model_spec,
            ),
        )
        model_file_records.append(
            {
                "file_id": selected_model_record.file_id,
                "relative_path": selected_model_record.relative_path,
                "modified_time": selected_model_record.modified_time,
            }
        )

        split_data = _prepare_split_data(
            df=df,
            schema=schema,
            encoding_method=encoding.encoding_id,
            is_regression=is_regression,
            poster_fast=poster_fast,
            fold_id=fold_id,
            n_folds=n_folds,
            random_seed=random_seed,
        )
        fold_payload = _evaluate_saved_model_fold(
            fold_id=fold_id,
            split_data=split_data,
            schema=schema,
            model_dir=local_model_path.parent,
            model_spec=model_spec,
            is_regression=is_regression,
            source_model_record=selected_model_record,
        )
        fold_results.append(fold_payload)
        full_mod._save_json(run_dir / "crossval" / "per_fold" / f"fold_{fold_id}.json", fold_payload)

    aggregate_payload = _aggregate_fold_results(
        dataset=dataset,
        encoding=encoding,
        n_folds=n_folds,
        fold_results=fold_results,
    )
    aggregate_path = run_dir / "metrics" / "aggregate.json"
    full_mod._save_json(aggregate_path, aggregate_payload)

    # Upload recomputed JSON results back to the same Drive folder,
    # overwriting the old aggregate.json and per-fold JSONs.
    _upload_recomputed_jsons(
        drive_client=drive_client,
        run_dir=run_dir,
        folder_id=folder_id,
    )

    folder_url = drive_client.folder_web_url(folder_id)
    local_summary = {
        "dataset_id": dataset.dataset_id,
        "dataset_label": dataset.label,
        "encoding_label": encoding.label,
        "encoding_method": encoding.encoding_id,
        "manifest_path": manifest_path,
        "output_dir": run_dir,
        "poster_fast": source_run_summary.get("poster_fast", {}),
        "crossval": {"n_folds": n_folds, "per_fold_dir": run_dir / "crossval" / "per_fold"},
        "metrics_path": aggregate_path,
        "tstr": aggregate_payload["tstr"],
        "source_drive": {
            "folder_id": folder_id,
            "folder_url": folder_url,
            "run_summary_file_id": source_summary_record.file_id,
            "run_summary_modified_time": source_summary_record.modified_time,
            "model_files": model_file_records,
        },
    }
    full_mod._save_json(run_dir / "run_summary.json", local_summary)

    row = build_results_sheet_row(
        model_name=display_name,
        dataset_label=dataset.label,
        dataset_id=dataset.dataset_id,
        encoding_label=encoding.label,
        encoding_id=encoding.encoding_id,
        aggregate=aggregate_payload,
        folder_url=folder_url,
    )
    return RecomputedPair(
        dataset_id=dataset.dataset_id,
        encoding_id=encoding.encoding_id,
        output_dir=run_dir,
        aggregate_metrics_path=aggregate_path,
        row=row,
    )


def recompute_all_from_drive(
    *,
    manifest_path: Path | str = DEFAULT_MANIFEST,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    drive_client: Any | None = None,
    sheets_config: Any | None = None,
    model_id: str = "ctgan",
    model_name: str = "CTGAN",
    results_worksheet: str = "Results",
    archive_existing_results: bool = True,
    random_seed: int = 42,
    write_sheet: bool = True,
) -> RecomputeFromDriveResult:
    manifest_path = Path(manifest_path).resolve()
    output_root = Path(output_root)
    project_root = full_mod._infer_project_root(manifest_path)
    manifest = _load_manifest_in_json_order(manifest_path, project_root=project_root)
    model_spec = get_experiment_model(model_id)
    effective_model_name = model_name if model_spec.model_id == "ctgan" else model_spec.display_name
    drive = drive_client if drive_client is not None else LocalDriveReader(LocalDriveConfig.from_env())
    sheet_writer: IncrementalResultsWorksheetWriter | None = None
    if write_sheet:
        config = sheets_config if sheets_config is not None else SheetsConfig.from_env()
        spreadsheet = _build_spreadsheet_from_config(config)
        sheet_writer = IncrementalResultsWorksheetWriter.create(
            spreadsheet=spreadsheet,
            worksheet_name=results_worksheet,
            archive_existing=archive_existing_results,
        )
        print(
            f"[ctgan_recompute_from_drive] created fresh worksheet '{results_worksheet}'",
            flush=True,
        )

    pairs: list[RecomputedPair] = []
    skipped: list[dict[str, str]] = []
    for dataset in manifest.datasets:
        for encoding in manifest.encodings:
            print(
                "[ctgan_recompute_from_drive] scanning "
                f"{effective_model_name}/{dataset.dataset_id}/{encoding.encoding_id}",
                flush=True,
            )
            try:
                pair = _recompute_pair_from_drive(
                    manifest_path=manifest_path,
                    project_root=project_root,
                    dataset=dataset,
                    encoding=encoding,
                    output_root=output_root,
                    drive_client=drive,
                    model_id=model_spec.model_id,
                    model_name=effective_model_name,
                    random_seed=random_seed,
                )
            except FileNotFoundError as exc:
                skipped.append(
                    {
                        "dataset_id": dataset.dataset_id,
                        "encoding_id": encoding.encoding_id,
                        "reason": str(exc),
                    }
                )
                print(
                    "[ctgan_recompute_from_drive] skipped "
                    f"{dataset.dataset_id}/{encoding.encoding_id}: {exc}",
                    flush=True,
                )
                continue
            if pair is None:
                skipped.append(
                    {
                        "dataset_id": dataset.dataset_id,
                        "encoding_id": encoding.encoding_id,
                        "reason": "missing valid Drive artifacts",
                    }
                )
                print(
                    "[ctgan_recompute_from_drive] skipped "
                    f"{dataset.dataset_id}/{encoding.encoding_id}: missing valid Drive artifacts",
                    flush=True,
                )
                continue
            pairs.append(pair)
            if sheet_writer is not None:
                sheet_writer.append_row(pair.row)
            print(
                "[ctgan_recompute_from_drive] recomputed "
                f"{dataset.dataset_id}/{encoding.encoding_id}",
                flush=True,
            )

    rows = [pair.row for pair in pairs]
    return RecomputeFromDriveResult(rows=rows, pairs=pairs, skipped=skipped)


def upload_missing_local_results(
    *,
    manifest_path: Path | str = DEFAULT_MANIFEST,
    local_results_root: Path | str = Path("experiments/results"),
    drive_client: Any | None = None,
    sheets_config: Any | None = None,
    spreadsheet: Any | None = None,
    model_id: str = "ctgan",
    model_name: str = "CTGAN",
    status_worksheet: str | None = None,
    results_worksheet: str = "Results",
    dry_run: bool = False,
) -> UploadMissingLocalResult:
    """Upload completed local runs that are missing from the Results worksheet.

    The status worksheet (for example ``CTGAN`` or ``TVAE``) is treated as the
    source of truth for completed experiments. A run is uploaded only when its
    cell is ``done`` and the full Results sheet does not already contain the
    same ``(Model, Dataset, Categorical representation)`` row.
    """
    manifest_path = Path(manifest_path).resolve()
    local_results_root = Path(local_results_root)
    project_root = full_mod._infer_project_root(manifest_path)
    manifest = _load_manifest_in_json_order(manifest_path, project_root=project_root)
    model_spec = get_experiment_model(model_id)
    effective_model_name = model_name if model_spec.model_id == "ctgan" else model_spec.display_name
    effective_status_worksheet = status_worksheet or effective_model_name

    config = sheets_config if sheets_config is not None else SheetsConfig.from_env()
    active_spreadsheet = spreadsheet if spreadsheet is not None else _build_spreadsheet_from_config(config)
    status_matrix = _worksheet_values(active_spreadsheet, effective_status_worksheet)
    try:
        results_matrix = _worksheet_values(active_spreadsheet, results_worksheet)
    except Exception:  # noqa: BLE001 - write_results_row can create a fresh Results sheet/header
        results_matrix = []
    existing_result_keys = _results_row_keys(results_matrix)

    drive = drive_client if drive_client is not None else drive_mod.DriveClient(drive_mod.DriveConfig.from_env())
    uploaded: list[UploadedLocalPair] = []
    skipped: list[dict[str, str]] = []

    for dataset, encoding in _done_status_pairs(
        matrix=status_matrix,
        manifest=manifest,
        model_id=model_spec.model_id,
    ):
        result_key = (effective_model_name, dataset.label, encoding.label)
        skip_base = {
            "dataset_id": dataset.dataset_id,
            "encoding_id": encoding.encoding_id,
        }
        if result_key in existing_result_keys:
            skipped.append({**skip_base, "reason": "already present in Results"})
            continue

        run_dir = local_results_root / model_spec.model_id / dataset.dataset_id / encoding.encoding_id
        missing_reason = _missing_local_run_reason(run_dir)
        if missing_reason is not None:
            skipped.append({**skip_base, "reason": missing_reason})
            continue

        aggregate_path = run_dir / "metrics" / "aggregate.json"
        aggregate = _load_json_file(aggregate_path)
        row = build_results_sheet_row(
            model_name=effective_model_name,
            dataset_label=dataset.label,
            dataset_id=dataset.dataset_id,
            encoding_label=encoding.label,
            encoding_id=encoding.encoding_id,
            aggregate=aggregate,
            folder_url="",
        )
        if dry_run:
            skipped.append({**skip_base, "reason": "dry-run"})
            print(
                "[ctgan_recompute_from_drive] would upload local "
                f"{effective_model_name}/{dataset.dataset_id}/{encoding.encoding_id}",
                flush=True,
            )
            continue

        try:
            folder_url = drive_mod.upload_experiment_artifacts(
                drive_client=drive,
                run_dir=run_dir,
                model_name=effective_model_name,
                dataset_id=dataset.dataset_id,
                encoding_method=encoding.encoding_id,
            )
            drive_mod.write_results_row(
                sheets_config=config,
                aggregate_metrics_path=aggregate_path,
                model_name=effective_model_name,
                dataset_label=dataset.label,
                dataset_id=dataset.dataset_id,
                encoding_label=encoding.label,
                encoding_id=encoding.encoding_id,
                folder_url=folder_url,
                results_worksheet_name=results_worksheet,
            )
        except Exception as exc:  # noqa: BLE001 - continue scanning remaining completed runs
            skipped.append({**skip_base, "reason": str(exc)})
            print(
                "[ctgan_recompute_from_drive] failed to upload local "
                f"{effective_model_name}/{dataset.dataset_id}/{encoding.encoding_id}: {exc}",
                flush=True,
            )
            continue

        row = build_results_sheet_row(
            model_name=effective_model_name,
            dataset_label=dataset.label,
            dataset_id=dataset.dataset_id,
            encoding_label=encoding.label,
            encoding_id=encoding.encoding_id,
            aggregate=aggregate,
            folder_url=folder_url,
        )
        uploaded.append(
            UploadedLocalPair(
                dataset_id=dataset.dataset_id,
                encoding_id=encoding.encoding_id,
                run_dir=run_dir,
                aggregate_metrics_path=aggregate_path,
                folder_url=folder_url,
                row=row,
            )
        )
        existing_result_keys.add(result_key)
        print(
            "[ctgan_recompute_from_drive] uploaded local "
            f"{effective_model_name}/{dataset.dataset_id}/{encoding.encoding_id}",
            flush=True,
        )

    return UploadMissingLocalResult(uploaded=uploaded, skipped=skipped)


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for parent in Path(__file__).resolve().parents:
        env_file = parent / ".env"
        if env_file.exists():
            load_dotenv(env_file, override=False)
            return


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recompute model metrics from saved Google Drive artifacts."
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--model-id", default="ctgan")
    parser.add_argument("--model-name", default="CTGAN")
    parser.add_argument("--status-worksheet")
    parser.add_argument("--results-worksheet", default="Results")
    parser.add_argument("--archive-existing-results", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--local-results-root", default="experiments/results")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--no-write-sheet", action="store_true")
    parser.add_argument("--upload-missing-local", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    args = build_arg_parser().parse_args(argv)
    if args.upload_missing_local:
        result = upload_missing_local_results(
            manifest_path=args.manifest,
            local_results_root=args.local_results_root,
            model_id=args.model_id,
            model_name=args.model_name,
            status_worksheet=args.status_worksheet,
            results_worksheet=args.results_worksheet,
            dry_run=args.dry_run,
        )
        print(
            "[ctgan_recompute_from_drive] "
            f"uploaded {len(result.uploaded)} local pair(s), skipped {len(result.skipped)} pair(s)"
        )
        return 0

    result = recompute_all_from_drive(
        manifest_path=args.manifest,
        output_root=args.output_root,
        model_id=args.model_id,
        model_name=args.model_name,
        results_worksheet=args.results_worksheet,
        archive_existing_results=args.archive_existing_results,
        random_seed=args.random_seed,
        write_sheet=not args.no_write_sheet,
    )
    print(
        "[ctgan_recompute_from_drive] "
        f"recomputed {len(result.rows)} pair(s), skipped {len(result.skipped)} pair(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
