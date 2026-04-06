from datetime import datetime, timezone

import pytest

from experiments.ctgan_orchestrator_state import (
    find_first_claimable_cell,
    parse_cell_payload,
    validate_worksheet_headers,
)


def test_blank_cell_is_implicit_not_started():
    payload = parse_cell_payload("   ")
    assert payload.status == "not-started"
    assert payload.stage is None


def test_in_progress_cell_becomes_claimable_after_four_hours():
    payload = parse_cell_payload(
        '{"v":1,"status":"in-progress","run_id":"r1","owner":"h:p:0","started_at":"2026-04-02T00:00:00Z","heartbeat_at":"2026-04-02T00:00:00Z","finished_at":null,"stage":"tuning","note":""}'
    )
    now = datetime(2026, 4, 2, 4, 1, tzinfo=timezone.utc)
    assert payload.is_claimable(now=now) is True


@pytest.mark.parametrize("raw_value", ["true", "false"])
def test_boolean_schema_version_is_rejected(raw_value: str):
    with pytest.raises(ValueError, match="version"):
        parse_cell_payload(
            f'{{"v":{raw_value},"status":"not-started","run_id":null,"owner":null,"started_at":null,"heartbeat_at":null,"finished_at":null,"stage":null,"note":""}}'
        )


def test_invalid_stage_for_done_status_is_rejected():
    with pytest.raises(ValueError, match="stage"):
        parse_cell_payload(
            '{"v":1,"status":"done","run_id":"r1","owner":"h:p:0","started_at":"2026-04-02T00:00:00Z","heartbeat_at":"2026-04-02T00:10:00Z","finished_at":"2026-04-02T00:11:00Z","stage":"tuning","note":""}'
        )


def test_incomplete_worksheet_axis_is_rejected():
    with pytest.raises(ValueError, match="incomplete"):
        validate_worksheet_headers(
            dataset_headers=["adult"],
            encoding_headers=["one-hot", "ordinal"],
            manifest_dataset_labels=["adult", "bank-marketing"],
            manifest_encoding_labels=["one-hot", "ordinal"],
        )


def test_timestamp_ordering_is_rejected_when_heartbeat_precedes_start():
    with pytest.raises(ValueError, match="heartbeat_at"):
        parse_cell_payload(
            '{"v":1,"status":"in-progress","run_id":"r1","owner":"h:p:0","started_at":"2026-04-02T00:10:00Z","heartbeat_at":"2026-04-02T00:00:00Z","finished_at":null,"stage":"tuning","note":""}'
        )


def test_timestamp_ordering_is_rejected_when_finished_precedes_heartbeat():
    with pytest.raises(ValueError, match="finished_at"):
        parse_cell_payload(
            '{"v":1,"status":"done","run_id":"r1","owner":"h:p:0","started_at":"2026-04-02T00:00:00Z","heartbeat_at":"2026-04-02T00:10:00Z","finished_at":"2026-04-02T00:05:00Z","stage":"done","note":""}'
        )


def test_skipped_cell_is_terminal_and_not_claimable():
    payload = parse_cell_payload(
        '{"v":1,"status":"skipped","run_id":"r1","owner":"h:p:0","started_at":"2026-04-02T00:00:00Z","heartbeat_at":"2026-04-02T00:10:00Z","finished_at":"2026-04-02T00:11:00Z","stage":"skipped","note":"aliased to B2: no categorical features"}'
    )
    now = datetime(2026, 4, 2, 12, 0, tzinfo=timezone.utc)
    assert payload.status == "skipped"
    assert payload.stage == "skipped"
    assert payload.is_claimable(now=now) is False


def test_find_first_claimable_cell_scans_left_to_right_then_top_to_bottom():
    coord = find_first_claimable_cell(
        dataset_headers=["adult", "bank-marketing"],
        encoding_headers=["one-hot", "ordinal"],
        manifest_dataset_labels=["adult", "bank-marketing"],
        manifest_encoding_labels=["one-hot", "ordinal"],
        cell_values={
            "B2": '{"v":1,"status":"done","run_id":"r1","owner":"h:p:0","started_at":"2026-04-02T00:00:00Z","heartbeat_at":"2026-04-02T00:10:00Z","finished_at":"2026-04-02T00:11:00Z","stage":"done","note":""}',
            "C2": " ",
            "B3": " ",
            "C3": " ",
        },
    )
    assert coord == "C2"


def test_unknown_sheet_header_is_rejected_against_manifest():
    with pytest.raises(ValueError, match="unknown dataset header"):
        find_first_claimable_cell(
            dataset_headers=["adult", "not-in-manifest"],
            encoding_headers=["one-hot"],
            cell_values={"B2": " "},
        )
