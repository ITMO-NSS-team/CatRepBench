"""Persistent timing storage for experiment runs.

Survives Ctrl+C / kernel restarts / re-runs with ``--skip_existing`` by
accumulating elapsed seconds across invocations into a JSON file. The
contract is:

- Each (dataset, encoding) pair has a separate entry.
- ``tuning_seconds`` and ``cv_seconds`` accumulate every time you call
  ``add(...)`` with that field — so if a run finishes tuning but is killed
  during CV, the tuning time is already saved; the next run only adds CV.
- ``total_seconds`` is always recomputed as the sum of the two.
- ``last_updated`` is an ISO8601 timestamp of the most recent write.

If a run is killed *inside* tuning (before the context manager exits), the
in-progress slice is lost. This is acceptable: Optuna stores trial state in
its own SQLite, so on resume tuning continues — we just under-count the
fraction of tuning time spent in the killed segment.

Usage
-----
::

    from experiments._timing import ExperimentTimings, timed

    timings = ExperimentTimings(Path("experiments/timings_tabddpm.json"))

    with timed() as t:
        tuning_info = ensure_tuning(...)
    timings.add(dataset, encoding, tuning_seconds=t.elapsed)

    with timed() as t:
        metrics = run_cv_for_encoding(...)
    timings.add(dataset, encoding, cv_seconds=t.elapsed)
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class _Stopwatch:
    elapsed: float = 0.0


@contextmanager
def timed():
    """Context manager that measures wall-clock seconds.

    The yielded object exposes ``.elapsed`` after the block ends.
    Uses ``time.monotonic()`` so it's immune to system clock adjustments.
    """
    sw = _Stopwatch()
    start = time.monotonic()
    try:
        yield sw
    finally:
        sw.elapsed = time.monotonic() - start


class ExperimentTimings:
    """Append-only JSON timing log.

    File schema::

        {
          "<dataset>": {
            "<encoding>": {
              "tuning_seconds": float,
              "cv_seconds": float,
              "total_seconds": float,
              "last_updated": "YYYY-MM-DDTHH:MM:SS"
            }
          }
        }
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.data: Dict[str, Dict[str, Dict[str, float]]] = {}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                # Corrupted file (likely from an unclean kill mid-write).
                # Preserve it for inspection, start fresh.
                self.path.rename(self.path.with_suffix(".json.bak"))
                self.data = {}

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: tmp -> rename. Avoids half-written JSON if killed
        # during write.
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def add(
        self,
        dataset: str,
        encoding: str,
        *,
        tuning_seconds: Optional[float] = None,
        cv_seconds: Optional[float] = None,
    ) -> None:
        """Accumulate seconds for (dataset, encoding). Persists immediately."""
        entry = self.data.setdefault(dataset, {}).setdefault(
            encoding,
            {"tuning_seconds": 0.0, "cv_seconds": 0.0},
        )
        if tuning_seconds is not None:
            entry["tuning_seconds"] = float(
                entry.get("tuning_seconds", 0.0)) + float(tuning_seconds)
        if cv_seconds is not None:
            entry["cv_seconds"] = float(
                entry.get("cv_seconds", 0.0)) + float(cv_seconds)
        entry["total_seconds"] = (
            entry.get("tuning_seconds", 0.0) + entry.get("cv_seconds", 0.0)
        )
        entry["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._save()

    def get(self, dataset: str, encoding: str) -> Optional[Dict[str, float]]:
        return self.data.get(dataset, {}).get(encoding)
