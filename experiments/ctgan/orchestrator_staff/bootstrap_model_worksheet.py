"""Create an empty per-model status worksheet (tab) in the Google Sheet.

The orchestrator (`ctgan_orchestrator.py --worksheet <tab> --model-id <model>`)
only *writes into* an already-existing tab; it never creates one. The CTGAN and
TVAE status tabs were created by hand, so when a new model (TabDDPM, TabPFGen) is
integrated there is no tab for it and nothing shows up in the table or the web
monitor. This script fills that gap: it builds the dataset x encoder status grid
straight from the orchestrator manifest (same labels the orchestrator validates
against) and creates the tab with empty (= pending) status cells.

Layout matches what `ctgan_orchestrator._load_snapshot` expects:
- row 1:  [corner, <dataset label>, <dataset label>, ...]   (datasets = columns)
- rows 2+: [<encoding label>, "", "", ...]                    (encoders = rows)

Run it where the Google Sheets API is reachable (i.e. your normal terminal — the
same place the orchestrator already talks to Sheets), with the usual env vars:
    CATREPBENCH_GSHEETS_SPREADSHEET_ID
    CATREPBENCH_GSHEETS_SERVICE_ACCOUNT_PATH  (or _JSON)

Examples:
    # preview without touching the sheet
    python -m experiments.ctgan.orchestrator_staff.bootstrap_model_worksheet \
        --worksheet TabDDPM --dry-run

    # create the TabDDPM and TabPFGen tabs
    python -m experiments.ctgan.orchestrator_staff.bootstrap_model_worksheet \
        --worksheet TabDDPM
    python -m experiments.ctgan.orchestrator_staff.bootstrap_model_worksheet \
        --worksheet TabPFGen
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.ctgan.orchestrator_staff.ctgan_manifest import load_ctgan_manifest
from experiments.ctgan.orchestrator_staff.ctgan_sheets import SheetsConfig, retry_call

_MANIFEST_PATH = (
    Path(__file__).resolve().parent / "ctgan_orchestrator_manifest.json"
)


def build_status_grid(manifest_path: Path = _MANIFEST_PATH) -> list[list[str]]:
    """Build the dataset x encoder status grid (empty cells = pending)."""
    project_root = manifest_path.resolve().parents[3]
    manifest = load_ctgan_manifest(manifest_path, project_root=project_root)
    dataset_labels = [entry.label for entry in manifest.datasets]
    encoding_labels = [entry.label for entry in manifest.encodings]

    header_row = ["", *dataset_labels]
    grid = [header_row]
    blanks = [""] * len(dataset_labels)
    for encoding_label in encoding_labels:
        grid.append([encoding_label, *blanks])
    return grid


def _open_spreadsheet(config: SheetsConfig):
    try:
        import gspread
        from google.oauth2 import service_account
    except ImportError as exc:  # pragma: no cover - only when deps absent
        raise RuntimeError(
            "gspread and google-auth are required. Install requirements.txt."
        ) from exc

    scopes = ("https://www.googleapis.com/auth/spreadsheets",)
    if config.service_account_info is not None:
        credentials = service_account.Credentials.from_service_account_info(
            info=config.service_account_info, scopes=scopes
        )
    elif config.service_account_path is not None:
        credentials = service_account.Credentials.from_service_account_file(
            filename=str(config.service_account_path), scopes=scopes
        )
    else:
        raise ValueError("SheetsConfig needs service_account_path or _info.")
    client = gspread.authorize(credentials)
    return retry_call(lambda: client.open_by_key(config.spreadsheet_id))


def create_worksheet(worksheet_name: str, *, grid: list[list[str]]) -> None:
    config = SheetsConfig.from_env()
    spreadsheet = _open_spreadsheet(config)

    existing = {ws.title for ws in retry_call(spreadsheet.worksheets)}
    if worksheet_name in existing:
        raise SystemExit(
            f"Worksheet {worksheet_name!r} already exists; refusing to modify it. "
            "Delete/rename it first if you want a clean rebuild."
        )

    n_rows = len(grid)
    n_cols = len(grid[0])
    worksheet = retry_call(
        lambda: spreadsheet.add_worksheet(
            title=worksheet_name, rows=max(n_rows, 50), cols=max(n_cols, 26)
        )
    )
    last_col = _column_name(n_cols)
    retry_call(lambda: worksheet.update(f"A1:{last_col}{n_rows}", grid))
    if hasattr(worksheet, "freeze"):
        retry_call(lambda: worksheet.freeze(rows=1, cols=1))
    print(
        f"Created worksheet {worksheet_name!r} ({n_rows} rows x {n_cols} cols; "
        f"{n_rows - 1} encoders x {n_cols - 1} datasets, all pending)."
    )


def _column_name(index_1_based: int) -> str:
    name = ""
    n = index_1_based
    while n > 0:
        n, rem = divmod(n - 1, 26)
        name = chr(ord("A") + rem) + name
    return name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worksheet",
        required=True,
        help="Tab title to create, e.g. TabDDPM or TabPFGen.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the grid shape and headers without touching the sheet.",
    )
    args = parser.parse_args(argv)

    grid = build_status_grid()
    n_rows, n_cols = len(grid), len(grid[0])
    if args.dry_run:
        print(f"[dry-run] worksheet {args.worksheet!r}: {n_rows}x{n_cols}")
        print(f"[dry-run] datasets (columns): {grid[0][1:]}")
        print(f"[dry-run] encoders (rows):    {[r[0] for r in grid[1:]]}")
        print("[dry-run] all data cells empty (= pending). No sheet changes.")
        return 0

    create_worksheet(args.worksheet, grid=grid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
