from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class DatasetEntry:
    label: str
    dataset_id: str
    target_col: str
    id_col: Optional[str]
    csv_path: Path


@dataclass(frozen=True)
class EncodingEntry:
    label: str
    encoding_id: str


@dataclass(frozen=True)
class CtganManifest:
    datasets: tuple[DatasetEntry, ...]
    encodings: tuple[EncodingEntry, ...]

    def resolve_dataset_label(self, label: str) -> DatasetEntry:
        normalized = _normalize_key(label)
        for entry in self.datasets:
            if _normalize_key(entry.label) == normalized:
                return entry
        raise ValueError(f"Unknown dataset label: {label!r}")

    def resolve_encoding_label(self, label: str) -> EncodingEntry:
        normalized = _normalize_key(label)
        for entry in self.encodings:
            if _normalize_key(entry.label) == normalized:
                return entry
        raise ValueError(f"Unknown encoding label: {label!r}")


def _normalize_key(value: str) -> str:
    return value.strip()


def _require_str(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string.")
    if not value.strip():
        raise ValueError(f"{field_name} cannot be empty.")
    return value


def _ensure_unique(entries: list[tuple[str, str]], *, kind: str) -> None:
    seen: dict[str, str] = {}
    for raw_value, normalized_value in entries:
        if normalized_value in seen:
            raise ValueError(
                f"duplicate {kind}: {raw_value!r} conflicts with {seen[normalized_value]!r}"
            )
        seen[normalized_value] = raw_value


def _load_dataset_entries(payload: object, *, project_root: Path) -> tuple[DatasetEntry, ...]:
    if not isinstance(payload, list):
        raise ValueError("datasets must be a list.")

    labels: list[tuple[str, str]] = []
    ids: list[tuple[str, str]] = []
    normalized_rows: list[tuple[str, str, str, Optional[str]]] = []

    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("dataset entries must be objects.")
        label = _require_str(item.get("label"), field_name="datasets.label")
        dataset_id = _require_str(item.get("dataset_id"), field_name="datasets.dataset_id")
        target_col = _require_str(item.get("target_col"), field_name="datasets.target_col")
        id_col_raw = item.get("id_col")
        if id_col_raw is not None:
            id_col = _require_str(id_col_raw, field_name="datasets.id_col")
        else:
            id_col = None

        labels.append((label, _normalize_key(label)))
        ids.append((dataset_id, _normalize_key(dataset_id)))
        normalized_rows.append((label, dataset_id, target_col, id_col))

    _ensure_unique(labels, kind="dataset label")
    _ensure_unique(ids, kind="dataset_id")

    return tuple(
        DatasetEntry(
            label=label,
            dataset_id=dataset_id,
            target_col=target_col,
            id_col=id_col,
            csv_path=project_root / "datasets" / "raw" / f"{dataset_id}.csv",
        )
        for label, dataset_id, target_col, id_col in normalized_rows
    )


def _load_encoding_entries(payload: object) -> tuple[EncodingEntry, ...]:
    if not isinstance(payload, list):
        raise ValueError("encodings must be a list.")

    labels: list[tuple[str, str]] = []
    ids: list[tuple[str, str]] = []
    normalized_rows: list[tuple[str, str]] = []

    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("encoding entries must be objects.")
        label = _require_str(item.get("label"), field_name="encodings.label")
        encoding_id = _require_str(item.get("encoding_id"), field_name="encodings.encoding_id")
        labels.append((label, _normalize_key(label)))
        ids.append((encoding_id, _normalize_key(encoding_id)))
        normalized_rows.append((label, encoding_id))

    _ensure_unique(labels, kind="encoding label")
    _ensure_unique(ids, kind="encoding_id")

    return tuple(EncodingEntry(label=label, encoding_id=encoding_id) for label, encoding_id in normalized_rows)


def load_ctgan_manifest(manifest_path: Path, *, project_root: Path) -> CtganManifest:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest must be a JSON object.")

    datasets = _load_dataset_entries(payload.get("datasets"), project_root=project_root)
    encodings = _load_encoding_entries(payload.get("encodings"))
    return CtganManifest(datasets=datasets, encodings=encodings)
