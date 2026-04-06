from __future__ import annotations

import argparse
import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, TextIO

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.ctgan_manifest import CtganManifest, DatasetEntry, EncodingEntry, load_ctgan_manifest
from experiments.ctgan_orchestrator_state import (
    CellPayload,
    find_first_claimable_cell,
    parse_cell_payload,
    validate_worksheet_headers,
)
from experiments.ctgan_sheets import SheetsClient, SheetsConfig

_DEFAULT_OUTPUT_ROOT = Path("experiments/results")
_HEARTBEAT_FAILURE_TIMEOUT_SECONDS = 15 * 60
_FAILURE_OUTPUT_TAIL_LINES = 8
_EOF = object()
_STREAM_QUEUES: dict[int, queue.Queue[str | object]] = {}


class SheetsClientProtocol(Protocol):
    def read_matrix(self) -> Any:
        ...

    def read_cell(self, coord: str) -> Any:
        ...

    def write_cell(self, coord: str, payload: str) -> Any:
        ...


@dataclass(frozen=True)
class WorksheetSnapshot:
    dataset_headers: tuple[str, ...]
    encoding_headers: tuple[str, ...]
    cell_values: dict[str, str | None]
    coord_labels: dict[str, tuple[str, str]]


@dataclass(frozen=True)
class OrchestratorRunResult:
    exit_code: int
    claimed_coord: str | None
    runner_argv: tuple[str, ...] | None
    claimed_jobs: int = 0


def spawn_runner(argv: list[str], env: dict[str, str], *, cwd: Path) -> subprocess.Popen[str]:
    return subprocess.Popen(
        argv,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def run_once(
    *,
    sheets: SheetsClientProtocol,
    manifest_path: Path | str,
    worksheet_name: str,
    dry_run: bool = False,
    heartbeat_seconds: int = 120,
    output_root: Path | str = _DEFAULT_OUTPUT_ROOT,
    best_params_file: Path | str | None = None,
    skip_tuning: bool = False,
    device: str = "cuda",
    continue_on_failure: bool = False,
) -> OrchestratorRunResult:
    del worksheet_name

    manifest_path = Path(manifest_path).resolve()
    project_root = _infer_project_root(manifest_path)
    manifest = load_ctgan_manifest(manifest_path, project_root=project_root)
    owner = _build_owner()
    first_claimed_coord: str | None = None
    first_runner_argv: tuple[str, ...] | None = None
    claimed_jobs = 0
    had_failures = False

    while True:
        snapshot = _load_snapshot(sheets.read_matrix(), manifest=manifest)
        claim_coord = find_first_claimable_cell(
            dataset_headers=snapshot.dataset_headers,
            encoding_headers=snapshot.encoding_headers,
            cell_values=snapshot.cell_values,
            manifest_dataset_labels=tuple(entry.label for entry in manifest.datasets),
            manifest_encoding_labels=tuple(entry.label for entry in manifest.encodings),
            now=datetime.now(timezone.utc),
        )
        if claim_coord is None:
            return OrchestratorRunResult(
                exit_code=1 if had_failures else 0,
                claimed_coord=first_claimed_coord,
                runner_argv=first_runner_argv,
                claimed_jobs=claimed_jobs,
            )

        dataset_label, encoding_label = snapshot.coord_labels[claim_coord]
        dataset = manifest.resolve_dataset_label(dataset_label)
        encoding = manifest.resolve_encoding_label(encoding_label)
        runner_argv = tuple(
            _build_runner_argv(
                manifest_path=manifest_path,
                dataset=dataset,
                encoding=encoding,
                output_root=Path(output_root),
                best_params_file=Path(best_params_file).resolve() if best_params_file is not None else None,
                skip_tuning=skip_tuning,
                device=device,
            )
        )

        if first_claimed_coord is None:
            first_claimed_coord = claim_coord
            first_runner_argv = runner_argv

        if dry_run:
            print(f"Would claim: {claim_coord}")
            print("Runner argv:", " ".join(runner_argv))
            return OrchestratorRunResult(
                exit_code=0,
                claimed_coord=claim_coord,
                runner_argv=runner_argv,
                claimed_jobs=0,
            )

        run_id = str(uuid.uuid4())
        claimed_at = _utcnow()
        claim_payload = _build_payload(
            status="in-progress",
            run_id=run_id,
            owner=owner,
            started_at=claimed_at,
            heartbeat_at=claimed_at,
            finished_at=None,
            stage="launching",
            note="claimed by orchestrator",
        )
        sheets.write_cell(claim_coord, claim_payload)

        if not _ensure_lease_owner(sheets=sheets, coord=claim_coord, run_id=run_id, owner=owner):
            return OrchestratorRunResult(
                exit_code=1,
                claimed_coord=claim_coord,
                runner_argv=runner_argv,
                claimed_jobs=claimed_jobs,
            )

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            process = spawn_runner(list(runner_argv), env, cwd=project_root)
        except Exception as exc:
            _write_terminal_state(
                sheets=sheets,
                coord=claim_coord,
                run_id=run_id,
                owner=owner,
                status="failed",
                stage="failed",
                note=f"launch failed: {exc}",
            )
            had_failures = True
            claimed_jobs += 1
            if continue_on_failure:
                continue
            return OrchestratorRunResult(
                exit_code=1,
                claimed_coord=claim_coord,
                runner_argv=runner_argv,
                claimed_jobs=claimed_jobs,
            )

        exit_code = _supervise_runner(
            sheets=sheets,
            coord=claim_coord,
            run_id=run_id,
            owner=owner,
            process=process,
            heartbeat_seconds=heartbeat_seconds,
        )
        if exit_code != 0:
            had_failures = True
            if continue_on_failure:
                claimed_jobs += 1
                continue
            return OrchestratorRunResult(
                exit_code=exit_code,
                claimed_coord=claim_coord,
                runner_argv=runner_argv,
                claimed_jobs=claimed_jobs,
            )

        claimed_jobs += 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the CTGAN Google Sheets orchestrator.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--worksheet", required=True)
    parser.add_argument("--output-root", default=str(_DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--heartbeat-seconds", type=int, default=120)
    parser.add_argument("--best-params-file")
    parser.add_argument("--skip-tuning", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config = SheetsConfig.from_env()
    config = replace(config, worksheet_name=args.worksheet)
    sheets = SheetsClient(config)
    result = run_once(
        sheets=sheets,
        manifest_path=args.manifest,
        worksheet_name=args.worksheet,
        dry_run=args.dry_run,
        heartbeat_seconds=args.heartbeat_seconds,
        output_root=args.output_root,
        best_params_file=Path(args.best_params_file).resolve() if args.best_params_file else None,
        skip_tuning=args.skip_tuning,
        device=args.device,
        continue_on_failure=args.continue_on_failure,
    )
    return result.exit_code


def _load_snapshot(raw_matrix: Any, *, manifest: CtganManifest) -> WorksheetSnapshot:
    dataset_labels = tuple(entry.label for entry in manifest.datasets)
    encoding_labels = tuple(entry.label for entry in manifest.encodings)

    if isinstance(raw_matrix, dict):
        dataset_headers, encoding_headers = validate_worksheet_headers(
            dataset_headers=raw_matrix["dataset_headers"],
            encoding_headers=raw_matrix["encoding_headers"],
            manifest_dataset_labels=dataset_labels,
            manifest_encoding_labels=encoding_labels,
        )
        cell_values = dict(raw_matrix["cell_values"])
        return WorksheetSnapshot(
            dataset_headers=dataset_headers,
            encoding_headers=encoding_headers,
            cell_values=cell_values,
            coord_labels=_build_coord_labels(dataset_headers, encoding_headers),
        )

    if not isinstance(raw_matrix, list) or not raw_matrix:
        raise ValueError("worksheet matrix must be a non-empty grid.")

    header_row = raw_matrix[0]
    if not isinstance(header_row, list):
        raise ValueError("worksheet matrix header row must be a list.")

    dataset_headers, encoding_headers = validate_worksheet_headers(
        dataset_headers=header_row[1:],
        encoding_headers=[row[0] if isinstance(row, list) and row else "" for row in raw_matrix[1:]],
        manifest_dataset_labels=dataset_labels,
        manifest_encoding_labels=encoding_labels,
    )
    coord_labels = _build_coord_labels(dataset_headers, encoding_headers)
    cell_values: dict[str, str | None] = {}
    for row_index, _encoding_label in enumerate(encoding_headers, start=1):
        row = raw_matrix[row_index] if row_index < len(raw_matrix) and isinstance(raw_matrix[row_index], list) else []
        for col_index, _dataset_label in enumerate(dataset_headers, start=1):
            coord = f"{_column_name(col_index + 1)}{row_index + 1}"
            cell_values[coord] = row[col_index] if col_index < len(row) else None
    return WorksheetSnapshot(
        dataset_headers=dataset_headers,
        encoding_headers=encoding_headers,
        cell_values=cell_values,
        coord_labels=coord_labels,
    )


def _supervise_runner(
    *,
    sheets: SheetsClientProtocol,
    coord: str,
    run_id: str,
    owner: str,
    process: Any,
    heartbeat_seconds: int,
) -> int:
    current_stage = "launching"
    current_note = "runner started"
    failure_output_tail: list[str] = []
    heartbeat_interval = max(heartbeat_seconds, 0)
    next_heartbeat_at = time.monotonic() + heartbeat_interval
    heartbeat_fail_started_at: float | None = None

    while True:
        timeout = max(0.0, next_heartbeat_at - time.monotonic())
        line = _read_available_line(process.stdout, timeout_seconds=timeout)
        if line:
            progress = _parse_progress_event(line)
            if progress is not None:
                current_stage = progress["stage"]
                current_note = progress["message"]
                if _write_in_progress_state(
                    sheets=sheets,
                    coord=coord,
                    run_id=run_id,
                    owner=owner,
                    stage=current_stage,
                    note=current_note,
                ):
                    heartbeat_fail_started_at = None
                    next_heartbeat_at = time.monotonic() + heartbeat_interval
                else:
                    if _heartbeat_fail_started(heartbeat_fail_started_at):
                        heartbeat_fail_started_at = time.monotonic()
                    if _heartbeat_failure_expired(heartbeat_fail_started_at):
                        _stop_process(process)
                        return 1
            else:
                _append_failure_output_line(failure_output_tail, line)
            continue

        if process.poll() is not None:
            break

        if time.monotonic() >= next_heartbeat_at:
            if _write_in_progress_state(
                sheets=sheets,
                coord=coord,
                run_id=run_id,
                owner=owner,
                stage=current_stage,
                note=current_note,
            ):
                heartbeat_fail_started_at = None
            else:
                if _heartbeat_fail_started(heartbeat_fail_started_at):
                    heartbeat_fail_started_at = time.monotonic()
                if _heartbeat_failure_expired(heartbeat_fail_started_at):
                    _stop_process(process)
                    return 1
            next_heartbeat_at = time.monotonic() + heartbeat_interval

    return_code = process.wait(timeout=5) if hasattr(process, "wait") else getattr(process, "returncode", 1)
    terminal_status = "done" if return_code == 0 else "failed"
    terminal_stage = "done" if return_code == 0 else "failed"
    terminal_note = (
        current_note
        if return_code == 0
        else _classify_runner_failure(return_code=return_code, failure_output_tail=failure_output_tail)
    )

    if not _write_terminal_state(
        sheets=sheets,
        coord=coord,
        run_id=run_id,
        owner=owner,
        status=terminal_status,
        stage=terminal_stage,
        note=terminal_note,
    ):
        return 1
    return int(return_code)


def _heartbeat_fail_started(value: float | None) -> bool:
    return value is None


def _heartbeat_failure_expired(started_at: float | None) -> bool:
    if started_at is None:
        return False
    return time.monotonic() - started_at >= _HEARTBEAT_FAILURE_TIMEOUT_SECONDS


def _write_in_progress_state(
    *,
    sheets: SheetsClientProtocol,
    coord: str,
    run_id: str,
    owner: str,
    stage: str,
    note: str,
) -> bool:
    if not _ensure_lease_owner(sheets=sheets, coord=coord, run_id=run_id, owner=owner):
        return False
    prior = parse_cell_payload(sheets.read_cell(coord))
    payload = _build_payload(
        status="in-progress",
        run_id=run_id,
        owner=owner,
        started_at=prior.started_at or _utcnow(),
        heartbeat_at=_utcnow(),
        finished_at=None,
        stage=stage,
        note=note,
    )
    sheets.write_cell(coord, payload)
    return True


def _write_terminal_state(
    *,
    sheets: SheetsClientProtocol,
    coord: str,
    run_id: str,
    owner: str,
    status: str,
    stage: str,
    note: str,
) -> bool:
    if not _ensure_lease_owner(sheets=sheets, coord=coord, run_id=run_id, owner=owner):
        return False
    prior = parse_cell_payload(sheets.read_cell(coord))
    now = _utcnow()
    payload = _build_payload(
        status=status,
        run_id=run_id,
        owner=owner,
        started_at=prior.started_at or now,
        heartbeat_at=now,
        finished_at=now,
        stage=stage,
        note=note,
    )
    sheets.write_cell(coord, payload)
    return True


def _ensure_lease_owner(
    *,
    sheets: SheetsClientProtocol,
    coord: str,
    run_id: str,
    owner: str,
) -> bool:
    payload = parse_cell_payload(sheets.read_cell(coord))
    return payload.status == "in-progress" and payload.run_id == run_id and payload.owner == owner


def _parse_progress_event(line: str) -> dict[str, str] | None:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or payload.get("event") != "progress":
        return None
    stage = payload.get("stage")
    message = payload.get("message")
    if not isinstance(stage, str) or not isinstance(message, str):
        return None
    return {"stage": stage, "message": message}


def _append_failure_output_line(buffer: list[str], line: str) -> None:
    normalized = " ".join(line.strip().split())
    if not normalized:
        return
    buffer.append(normalized)
    if len(buffer) > _FAILURE_OUTPUT_TAIL_LINES:
        del buffer[: len(buffer) - _FAILURE_OUTPUT_TAIL_LINES]


def _classify_runner_failure(*, return_code: int, failure_output_tail: list[str]) -> str:
    if return_code < 0:
        return f"runner terminated by signal {-return_code}"

    tail_text = " | ".join(failure_output_tail)
    tail_lower = tail_text.lower()
    if any(
        marker in tail_lower
        for marker in (
            "out of memory",
            "oom",
            "mps backend out of memory",
            "cuda out of memory",
            "not enough memory",
        )
    ):
        if tail_text:
            return f"resource_exhausted: {tail_text}"
        return "resource_exhausted: runner reported out of memory"

    if tail_text:
        return f"runner exited with code {return_code}: {tail_text}"
    return f"runner exited with code {return_code}"


def _build_runner_argv(
    *,
    manifest_path: Path,
    dataset: DatasetEntry,
    encoding: EncodingEntry,
    output_root: Path,
    best_params_file: Path | None,
    skip_tuning: bool,
    device: str,
) -> list[str]:
    argv = [
        sys.executable,
        "experiments/ctgan_full_experiment.py",
        "--manifest",
        str(manifest_path),
        "--dataset-id",
        dataset.dataset_id,
        "--dataset-label",
        dataset.label,
        "--encoding-method",
        encoding.encoding_id,
        "--output-root",
        str(output_root),
        "--progress-format",
        "jsonl",
    ]
    if best_params_file is not None:
        argv.extend(["--best-params-file", str(best_params_file)])
    if skip_tuning:
        argv.append("--skip-tuning")
    argv.extend(["--device", device])
    return argv


def _build_payload(
    *,
    status: str,
    run_id: str | None,
    owner: str | None,
    started_at: datetime | None,
    heartbeat_at: datetime | None,
    finished_at: datetime | None,
    stage: str | None,
    note: str,
) -> str:
    return json.dumps(
        {
            "v": 1,
            "status": status,
            "run_id": run_id,
            "owner": owner,
            "started_at": _format_timestamp(started_at),
            "heartbeat_at": _format_timestamp(heartbeat_at),
            "finished_at": _format_timestamp(finished_at),
            "stage": stage,
            "note": note,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _format_timestamp(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _build_coord_labels(
    dataset_headers: tuple[str, ...],
    encoding_headers: tuple[str, ...],
) -> dict[str, tuple[str, str]]:
    mapping: dict[str, tuple[str, str]] = {}
    for encoding_offset, encoding_label in enumerate(encoding_headers, start=2):
        for dataset_offset, dataset_label in enumerate(dataset_headers, start=2):
            mapping[f"{_column_name(dataset_offset)}{encoding_offset}"] = (dataset_label, encoding_label)
    return mapping


def _column_name(index: int) -> str:
    if index < 1:
        raise ValueError("column index must be positive.")
    name = ""
    current = index
    while current:
        current, remainder = divmod(current - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _infer_project_root(manifest_path: Path) -> Path:
    candidates = [manifest_path.parent, manifest_path.parent.parent]
    for candidate in candidates:
        if (candidate / "datasets" / "raw").exists() or (candidate / "genbench").exists():
            return candidate
    if manifest_path.parent.name == "experiments":
        return manifest_path.parent.parent
    return manifest_path.parent


def _build_owner() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:0"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _read_available_line(stream: TextIO | None, *, timeout_seconds: float) -> str | None:
    if stream is None:
        if timeout_seconds > 0:
            time.sleep(timeout_seconds)
        return None

    line_queue = _stream_line_queue(stream)
    try:
        line = line_queue.get(timeout=max(timeout_seconds, 0.0))
    except queue.Empty:
        return None

    if line is _EOF:
        _STREAM_QUEUES.pop(id(stream), None)
        return None

    return str(line)


def _stream_line_queue(stream: TextIO) -> queue.Queue[str | object]:
    stream_id = id(stream)
    if stream_id in _STREAM_QUEUES:
        return _STREAM_QUEUES[stream_id]

    line_queue: queue.Queue[str | object] = queue.Queue()
    _STREAM_QUEUES[stream_id] = line_queue

    def _reader() -> None:
        try:
            for line in stream:
                line_queue.put(line)
        finally:
            line_queue.put(_EOF)

    threading.Thread(target=_reader, name=f"ctgan-runner-stdout-{stream_id}", daemon=True).start()
    return line_queue


def _stop_process(process: Any) -> None:
    if hasattr(process, "terminate"):
        process.terminate()
    try:
        if hasattr(process, "wait"):
            process.wait(timeout=5)
            return
    except Exception:
        pass
    if hasattr(process, "kill"):
        process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
