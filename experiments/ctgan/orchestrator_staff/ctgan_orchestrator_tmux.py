from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

_SESSION_PREFIX = "ctgan-orch"
_DEFAULT_OUTPUT_ROOT = Path("experiments/results")


@dataclass(frozen=True)
class LaunchResult:
    session_name: str
    log_path: Path
    metadata_path: Path
    attach_command: tuple[str, ...]


@dataclass(frozen=True)
class SessionInfo:
    session_name: str
    log_path: Path | None
    metadata_path: Path | None
    worksheet: str | None
    started_at: str | None


def build_session_name(*, index: int, hostname: str | None = None) -> str:
    if index < 0:
        raise ValueError("index must be >= 0.")
    raw_hostname = (hostname or socket.gethostname()).split(".", 1)[0].lower()
    normalized_hostname = re.sub(r"[^a-z0-9-]+", "-", raw_hostname)
    normalized_hostname = re.sub(r"-{2,}", "-", normalized_hostname).strip("-") or "host"
    return f"{_SESSION_PREFIX}-{normalized_hostname}-{index:02d}"


def log_path_for_session(*, output_root: Path, session_name: str) -> Path:
    return Path(output_root) / "orchestrator_logs" / f"{session_name}.log"


def metadata_path_for_session(*, output_root: Path, session_name: str) -> Path:
    return Path(output_root) / "orchestrator_logs" / f"{session_name}.json"


def build_worker_argv(
    *,
    manifest_path: Path,
    worksheet: str,
    output_root: Path,
    heartbeat_seconds: int,
) -> list[str]:
    return [
        sys.executable,
        "experiments/ctgan/orchestrator_staff/ctgan_orchestrator.py",
        "--manifest",
        str(manifest_path),
        "--worksheet",
        worksheet,
        "--output-root",
        str(output_root),
        "--heartbeat-seconds",
        str(heartbeat_seconds),
    ]


def build_tmux_launch_command(
    *,
    session_name: str,
    worker_argv: list[str],
    log_path: Path,
    cwd: Path,
) -> list[str]:
    shell_command = (
        f"cd {shlex.quote(str(cwd))} && "
        f"{shlex.join(worker_argv)} 2>&1 | tee -a {shlex.quote(str(log_path))}"
    )
    return ["tmux", "new-session", "-d", "-s", session_name, shell_command]


def write_session_metadata(
    *,
    session_name: str,
    metadata_path: Path,
    manifest_path: Path,
    worksheet: str,
    log_path: Path,
    worker_argv: list[str],
) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "session": session_name,
        "hostname": socket.gethostname(),
        "started_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "argv": worker_argv,
        "log_path": str(log_path),
        "manifest": str(manifest_path),
        "worksheet": worksheet,
    }
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def launch_session(
    *,
    manifest_path: Path,
    worksheet: str,
    output_root: Path,
    index: int,
    heartbeat_seconds: int = 120,
) -> LaunchResult:
    _require_tmux()
    manifest_path = Path(manifest_path).resolve()
    project_root = _infer_project_root(manifest_path)
    session_name = build_session_name(index=index)
    log_path = log_path_for_session(output_root=Path(output_root), session_name=session_name)
    metadata_path = metadata_path_for_session(output_root=Path(output_root), session_name=session_name)

    if _tmux_session_exists(session_name):
        raise RuntimeError(f"tmux session already exists: {session_name}")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    worker_argv = build_worker_argv(
        manifest_path=manifest_path,
        worksheet=worksheet,
        output_root=Path(output_root),
        heartbeat_seconds=heartbeat_seconds,
    )
    launch_argv = build_tmux_launch_command(
        session_name=session_name,
        worker_argv=worker_argv,
        log_path=log_path,
        cwd=project_root,
    )
    subprocess.run(launch_argv, check=True, cwd=project_root)
    write_session_metadata(
        session_name=session_name,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        worksheet=worksheet,
        log_path=log_path,
        worker_argv=worker_argv,
    )

    return LaunchResult(
        session_name=session_name,
        log_path=log_path,
        metadata_path=metadata_path,
        attach_command=("tmux", "attach", "-t", session_name),
    )


def attach_session(session_name: str) -> None:
    _require_tmux()
    subprocess.run(["tmux", "attach", "-t", session_name], check=True)


def list_sessions(*, output_root: Path) -> list[SessionInfo]:
    _require_tmux()
    result = subprocess.run(
        ["tmux", "ls"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []

    sessions: list[SessionInfo] = []
    for line in result.stdout.splitlines():
        raw_name = line.split(":", 1)[0].strip()
        if not raw_name.startswith(f"{_SESSION_PREFIX}-"):
            continue
        metadata = _read_session_metadata(
            metadata_path_for_session(output_root=Path(output_root), session_name=raw_name)
        )
        sessions.append(
            SessionInfo(
                session_name=raw_name,
                log_path=Path(metadata["log_path"]) if metadata.get("log_path") else None,
                metadata_path=metadata_path_for_session(output_root=Path(output_root), session_name=raw_name),
                worksheet=metadata.get("worksheet"),
                started_at=metadata.get("started_at"),
            )
        )
    return sessions


def tail_log(*, session_name: str, output_root: Path, lines: int = 50) -> str:
    metadata = _read_session_metadata(
        metadata_path_for_session(output_root=Path(output_root), session_name=session_name)
    )
    log_path = Path(metadata["log_path"]) if metadata.get("log_path") else log_path_for_session(
        output_root=Path(output_root),
        session_name=session_name,
    )
    content = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    tail_lines = content.splitlines()[-lines:]
    return "\n".join(tail_lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch and attach to CTGAN orchestrator tmux sessions.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument("--manifest", required=True)
    launch_parser.add_argument("--worksheet", required=True)
    launch_parser.add_argument("--output-root", default=str(_DEFAULT_OUTPUT_ROOT))
    launch_parser.add_argument("--heartbeat-seconds", type=int, default=120)
    launch_parser.add_argument("--index", type=int, default=1)

    attach_parser = subparsers.add_parser("attach")
    attach_parser.add_argument("--session", required=True)

    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--output-root", default=str(_DEFAULT_OUTPUT_ROOT))

    tail_parser = subparsers.add_parser("tail")
    tail_parser.add_argument("--session", required=True)
    tail_parser.add_argument("--output-root", default=str(_DEFAULT_OUTPUT_ROOT))
    tail_parser.add_argument("--lines", type=int, default=50)

    args = parser.parse_args(argv)

    if args.command == "launch":
        result = launch_session(
            manifest_path=Path(args.manifest),
            worksheet=args.worksheet,
            output_root=Path(args.output_root),
            index=int(args.index),
            heartbeat_seconds=int(args.heartbeat_seconds),
        )
        print(f"session: {result.session_name}")
        print(f"attach: {' '.join(result.attach_command)}")
        print(f"log: {result.log_path}")
        return 0

    if args.command == "attach":
        attach_session(args.session)
        return 0

    if args.command == "list":
        for session in list_sessions(output_root=Path(args.output_root)):
            log_text = str(session.log_path) if session.log_path is not None else "-"
            worksheet_text = session.worksheet or "-"
            started_text = session.started_at or "-"
            print(f"{session.session_name}\t{worksheet_text}\t{started_text}\t{log_text}")
        return 0

    if args.command == "tail":
        print(
            tail_log(
                session_name=args.session,
                output_root=Path(args.output_root),
                lines=int(args.lines),
            )
        )
        return 0

    raise RuntimeError(f"unknown command: {args.command}")


def _infer_project_root(manifest_path: Path) -> Path:
    for candidate in manifest_path.parents:
        if (candidate / "datasets" / "raw").exists() or (candidate / "genbench").exists():
            return candidate
    return manifest_path.parent


def _read_session_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _require_tmux() -> str:
    tmux_path = shutil.which("tmux")
    if tmux_path is None:
        raise RuntimeError("tmux is required for ctgan_orchestrator_tmux.")
    return tmux_path


def _tmux_session_exists(session_name: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


if __name__ == "__main__":
    raise SystemExit(main())
