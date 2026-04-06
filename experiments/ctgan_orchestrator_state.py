from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from experiments.ctgan_manifest import load_ctgan_manifest

_SCHEMA_VERSION = 1
_STALE_TIMEOUT = timedelta(hours=4)

_STATUS_NOT_STARTED = "not-started"
_STATUS_IN_PROGRESS = "in-progress"
_STATUS_DONE = "done"
_STATUS_FAILED = "failed"
_STATUS_SKIPPED = "skipped"

_STAGE_LAUNCHING = "launching"
_STAGE_TUNING = "tuning"
_STAGE_TRANSITION = "crossval"
_STAGE_METRICS = "metrics"
_STAGE_SAVING = "saving"
_STAGE_DONE = "done"
_STAGE_FAILED = "failed"
_STAGE_SKIPPED = "skipped"

_IN_PROGRESS_STAGES = {
    _STAGE_LAUNCHING,
    _STAGE_TUNING,
    _STAGE_TRANSITION,
    _STAGE_METRICS,
    _STAGE_SAVING,
}
_STATUS_TO_STAGE = {
    _STATUS_NOT_STARTED: {None},
    _STATUS_IN_PROGRESS: _IN_PROGRESS_STAGES,
    _STATUS_DONE: {_STAGE_DONE},
    _STATUS_FAILED: {_STAGE_FAILED},
    _STATUS_SKIPPED: {_STAGE_SKIPPED},
}


@dataclass(frozen=True)
class CellPayload:
    v: int
    status: str
    run_id: str | None
    owner: str | None
    started_at: datetime | None
    heartbeat_at: datetime | None
    finished_at: datetime | None
    stage: str | None
    note: str

    def is_claimable(self, *, now: datetime) -> bool:
        if self.status == _STATUS_NOT_STARTED:
            return True
        if self.status != _STATUS_IN_PROGRESS or self.heartbeat_at is None:
            return False
        current_time = _ensure_utc_datetime(now, field_name="now")
        return current_time - self.heartbeat_at > _STALE_TIMEOUT


def parse_cell_payload(raw_value: str | None) -> CellPayload:
    if raw_value is None:
        return CellPayload(
            v=_SCHEMA_VERSION,
            status=_STATUS_NOT_STARTED,
            run_id=None,
            owner=None,
            started_at=None,
            heartbeat_at=None,
            finished_at=None,
            stage=None,
            note="",
        )
    if not isinstance(raw_value, str):
        raise ValueError("cell payload must be a string or null.")
    if not raw_value.strip():
        return CellPayload(
            v=_SCHEMA_VERSION,
            status=_STATUS_NOT_STARTED,
            run_id=None,
            owner=None,
            started_at=None,
            heartbeat_at=None,
            finished_at=None,
            stage=None,
            note="",
        )

    try:
        payload = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError("cell payload must be valid JSON.") from exc

    if not isinstance(payload, dict):
        raise ValueError("cell payload must be a JSON object.")

    expected_keys = {
        "v",
        "status",
        "run_id",
        "owner",
        "started_at",
        "heartbeat_at",
        "finished_at",
        "stage",
        "note",
    }
    unexpected_keys = set(payload) - expected_keys
    if unexpected_keys:
        raise ValueError(f"cell payload contains unexpected keys: {sorted(unexpected_keys)!r}")

    missing_keys = expected_keys - set(payload)
    if missing_keys:
        raise ValueError(f"cell payload is missing required keys: {sorted(missing_keys)!r}")

    version = payload["v"]
    if type(version) is not int or version != _SCHEMA_VERSION:
        raise ValueError("cell payload version must be v=1.")

    status = payload["status"]
    if not isinstance(status, str) or status not in _STATUS_TO_STAGE:
        raise ValueError("cell payload status is invalid.")

    run_id = _parse_nullable_string(payload["run_id"], field_name="run_id")
    owner = _parse_nullable_string(payload["owner"], field_name="owner")
    started_at = _parse_nullable_timestamp(payload["started_at"], field_name="started_at")
    heartbeat_at = _parse_nullable_timestamp(payload["heartbeat_at"], field_name="heartbeat_at")
    finished_at = _parse_nullable_timestamp(payload["finished_at"], field_name="finished_at")
    stage = _parse_nullable_string(payload["stage"], field_name="stage")
    note = _parse_required_string(payload["note"], field_name="note")

    _validate_status_stage(status=status, stage=stage)
    _validate_timestamp_ordering(
        started_at=started_at,
        heartbeat_at=heartbeat_at,
        finished_at=finished_at,
    )

    if status == _STATUS_NOT_STARTED:
        _require_all_none(
            {
                "run_id": run_id,
                "owner": owner,
                "started_at": started_at,
                "heartbeat_at": heartbeat_at,
                "finished_at": finished_at,
            }
        )
    elif status == _STATUS_IN_PROGRESS:
        _require_non_empty_string(run_id, field_name="run_id")
        _require_non_empty_string(owner, field_name="owner")
        _require_non_none(started_at, field_name="started_at")
        _require_non_none(heartbeat_at, field_name="heartbeat_at")
        if finished_at is not None:
            raise ValueError("finished_at must be null while status is in-progress.")
    else:
        _require_non_empty_string(run_id, field_name="run_id")
        _require_non_empty_string(owner, field_name="owner")
        _require_non_none(started_at, field_name="started_at")
        _require_non_none(heartbeat_at, field_name="heartbeat_at")
        _require_non_none(finished_at, field_name="finished_at")

    return CellPayload(
        v=version,
        status=status,
        run_id=run_id,
        owner=owner,
        started_at=started_at,
        heartbeat_at=heartbeat_at,
        finished_at=finished_at,
        stage=stage,
        note=note,
    )


def find_first_claimable_cell(
    *,
    dataset_headers: Sequence[str],
    encoding_headers: Sequence[str],
    cell_values: dict[str, str | None],
    manifest_dataset_labels: Sequence[str] | None = None,
    manifest_encoding_labels: Sequence[str] | None = None,
    now: datetime | None = None,
) -> str | None:
    default_dataset_labels: tuple[str, ...] | None = None
    default_encoding_labels: tuple[str, ...] | None = None
    if manifest_dataset_labels is None or manifest_encoding_labels is None:
        default_dataset_labels, default_encoding_labels = _default_manifest_labels()
    if manifest_dataset_labels is None:
        manifest_dataset_labels = default_dataset_labels
    if manifest_encoding_labels is None:
        manifest_encoding_labels = default_encoding_labels

    validated_dataset_headers, validated_encoding_headers = validate_worksheet_headers(
        dataset_headers=dataset_headers,
        encoding_headers=encoding_headers,
        manifest_dataset_labels=manifest_dataset_labels,
        manifest_encoding_labels=manifest_encoding_labels,
    )

    current_time = now or datetime.now(timezone.utc)
    for encoding_offset, _encoding_label in enumerate(validated_encoding_headers, start=2):
        for dataset_offset, _dataset_label in enumerate(validated_dataset_headers, start=2):
            coord = f"{_column_name(dataset_offset)}{encoding_offset}"
            payload = parse_cell_payload(cell_values.get(coord))
            if payload.is_claimable(now=current_time):
                return coord
    return None


def validate_worksheet_headers(
    *,
    dataset_headers: Sequence[str],
    encoding_headers: Sequence[str],
    manifest_dataset_labels: Sequence[str] | None = None,
    manifest_encoding_labels: Sequence[str] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    default_dataset_labels: tuple[str, ...] | None = None
    default_encoding_labels: tuple[str, ...] | None = None
    if manifest_dataset_labels is None or manifest_encoding_labels is None:
        default_dataset_labels, default_encoding_labels = _default_manifest_labels()
    if manifest_dataset_labels is None:
        manifest_dataset_labels = default_dataset_labels
    if manifest_encoding_labels is None:
        manifest_encoding_labels = default_encoding_labels

    validated_dataset_headers = _normalize_worksheet_axis(
        dataset_headers,
        axis_name="dataset",
        manifest_labels=manifest_dataset_labels,
    )
    validated_encoding_headers = _normalize_worksheet_axis(
        encoding_headers,
        axis_name="encoding",
        manifest_labels=manifest_encoding_labels,
    )
    return validated_dataset_headers, validated_encoding_headers


def _normalize_worksheet_axis(
    labels: Sequence[str],
    *,
    axis_name: str,
    manifest_labels: Sequence[str] | None,
) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    trimmed_manifest_labels = None
    if manifest_labels is not None:
        trimmed_manifest_labels = {label.strip() for label in manifest_labels}
        if any(not label.strip() for label in manifest_labels):
            raise ValueError(f"{axis_name} manifest labels cannot be blank.")

    encountered_blank = False
    for raw_label in labels:
        if not isinstance(raw_label, str):
            raise ValueError(f"{axis_name} headers must be strings.")
        label = raw_label.strip()
        if not label:
            encountered_blank = True
            continue
        if encountered_blank:
            raise ValueError(f"blank {axis_name} header is not allowed between non-empty headers.")
        if label in seen:
            raise ValueError(f"duplicate {axis_name} header: {raw_label!r}")
        if trimmed_manifest_labels is not None and label not in trimmed_manifest_labels:
            raise ValueError(f"unknown {axis_name} header: {raw_label!r}")
        seen.add(label)
        normalized.append(label)

    if trimmed_manifest_labels is not None and set(normalized) != trimmed_manifest_labels:
        missing = sorted(trimmed_manifest_labels - set(normalized))
        extra = sorted(set(normalized) - trimmed_manifest_labels)
        if missing:
            raise ValueError(f"incomplete {axis_name} axis; missing headers: {missing!r}")
        if extra:
            raise ValueError(f"unknown {axis_name} header(s): {extra!r}")

    return tuple(normalized)


@lru_cache(maxsize=1)
def _default_manifest_labels() -> tuple[tuple[str, ...], tuple[str, ...]]:
    project_root = Path(__file__).resolve().parents[1]
    manifest = load_ctgan_manifest(
        project_root / "experiments" / "ctgan_orchestrator_manifest.json",
        project_root=project_root,
    )
    return tuple(entry.label for entry in manifest.datasets), tuple(entry.label for entry in manifest.encodings)


def _validate_status_stage(*, status: str, stage: str | None) -> None:
    allowed_stages = _STATUS_TO_STAGE[status]
    if stage not in allowed_stages:
        allowed_text = ", ".join("null" if value is None else value for value in sorted(allowed_stages, key=_sort_stage_key))
        raise ValueError(f"stage {stage!r} is invalid for status {status!r}; allowed: {allowed_text}.")


def _parse_nullable_string(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string or null.")
    return value


def _parse_required_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string.")
    return value


def _parse_nullable_timestamp(value: object, *, field_name: str) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a UTC timestamp string or null.")
    return _ensure_utc_datetime(_parse_timestamp(value, field_name=field_name), field_name=field_name)


def _parse_timestamp(value: str, *, field_name: str) -> datetime:
    iso_value = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(iso_value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a UTC timestamp string.") from exc
    return parsed


def _ensure_utc_datetime(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be a UTC datetime.")
    return value.astimezone(timezone.utc)


def _require_non_none(value: datetime | None, *, field_name: str) -> None:
    if value is None:
        raise ValueError(f"{field_name} is required for this status.")


def _require_non_empty_string(value: str | None, *, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is required for this status.")


def _require_all_none(values: dict[str, object]) -> None:
    for field_name, value in values.items():
        if value is not None:
            raise ValueError(f"{field_name} must be null for not-started payloads.")


def _validate_timestamp_ordering(
    *,
    started_at: datetime | None,
    heartbeat_at: datetime | None,
    finished_at: datetime | None,
) -> None:
    if started_at is not None and heartbeat_at is not None and heartbeat_at < started_at:
        raise ValueError("heartbeat_at must be greater than or equal to started_at.")
    if finished_at is not None:
        if started_at is not None and finished_at < started_at:
            raise ValueError("finished_at must be greater than or equal to started_at.")
        if heartbeat_at is not None and finished_at < heartbeat_at:
            raise ValueError("finished_at must be greater than or equal to heartbeat_at.")


def _sort_stage_key(stage: str | None) -> tuple[int, str]:
    if stage is None:
        return (0, "")
    return (1, stage)


def _column_name(index: int) -> str:
    if index < 1:
        raise ValueError("column index must be positive.")
    name = ""
    current = index
    while current:
        current, remainder = divmod(current - 1, 26)
        name = chr(65 + remainder) + name
    return name
