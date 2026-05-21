"""Tests for experiments/_timing.py persistent timing storage."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from experiments._timing import ExperimentTimings, timed


def test_timed_measures_seconds():
    with timed() as t:
        time.sleep(0.05)
    assert t.elapsed >= 0.05
    assert t.elapsed < 1.0  # sanity


def test_add_accumulates_across_calls(tmp_path: Path):
    path = tmp_path / "timings.json"
    t = ExperimentTimings(path)

    t.add("ds1", "enc1", tuning_seconds=10.0)
    t.add("ds1", "enc1", cv_seconds=5.0)

    entry = t.get("ds1", "enc1")
    assert entry["tuning_seconds"] == 10.0
    assert entry["cv_seconds"] == 5.0
    assert entry["total_seconds"] == 15.0
    assert "last_updated" in entry


def test_add_is_incremental(tmp_path: Path):
    path = tmp_path / "timings.json"
    t = ExperimentTimings(path)

    t.add("ds1", "enc1", tuning_seconds=10.0)
    t.add("ds1", "enc1", tuning_seconds=3.0)  # add more

    assert t.get("ds1", "enc1")["tuning_seconds"] == 13.0


def test_survives_restart(tmp_path: Path):
    path = tmp_path / "timings.json"

    t1 = ExperimentTimings(path)
    t1.add("ds1", "enc1", tuning_seconds=10.0, cv_seconds=5.0)

    # Simulate process restart
    t2 = ExperimentTimings(path)
    entry = t2.get("ds1", "enc1")
    assert entry["tuning_seconds"] == 10.0
    assert entry["cv_seconds"] == 5.0


def test_can_add_to_existing_pair_on_restart(tmp_path: Path):
    path = tmp_path / "timings.json"

    t1 = ExperimentTimings(path)
    t1.add("ds1", "enc1", tuning_seconds=10.0)

    # Run 2: continue and add CV time
    t2 = ExperimentTimings(path)
    t2.add("ds1", "enc1", cv_seconds=5.0)

    assert t2.get("ds1", "enc1")["total_seconds"] == 15.0


def test_corrupted_file_is_backed_up(tmp_path: Path):
    path = tmp_path / "timings.json"
    path.write_text("not json {", encoding="utf-8")

    t = ExperimentTimings(path)
    assert t.data == {}
    assert path.with_suffix(".json.bak").exists()

    t.add("ds1", "enc1", tuning_seconds=1.0)
    assert t.get("ds1", "enc1")["tuning_seconds"] == 1.0


def test_atomic_write_creates_no_stale_tmp(tmp_path: Path):
    path = tmp_path / "timings.json"
    t = ExperimentTimings(path)
    t.add("ds1", "enc1", tuning_seconds=1.0)
    assert path.exists()
    assert not path.with_suffix(".json.tmp").exists()


def test_multiple_datasets_and_encodings(tmp_path: Path):
    path = tmp_path / "timings.json"
    t = ExperimentTimings(path)

    t.add("ds1", "one_hot", tuning_seconds=10.0, cv_seconds=2.0)
    t.add("ds1", "ordinal", tuning_seconds=12.0, cv_seconds=3.0)
    t.add("ds2", "one_hot", tuning_seconds=20.0, cv_seconds=4.0)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["ds1"]["one_hot"]["total_seconds"] == 12.0
    assert data["ds1"]["ordinal"]["total_seconds"] == 15.0
    assert data["ds2"]["one_hot"]["total_seconds"] == 24.0
