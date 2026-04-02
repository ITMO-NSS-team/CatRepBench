# CTGAN Experiments Runbook

This document explains how to prepare, launch, monitor, and recover CTGAN experiments in CatRepBench.

## What One Job Means

One Google Sheets data cell corresponds to one full CTGAN experiment:

- one dataset
- one categorical encoding

The orchestrator runs the full pipeline for that cell:

1. tuning
2. cross-validation
3. metrics
4. saving results

## Required Files

The main entrypoints are:

- `experiments/ctgan_orchestrator.py`
- `experiments/ctgan_orchestrator_tmux.py`
- `experiments/ctgan_full_experiment.py`
- `experiments/ctgan_orchestrator_manifest.json`

## One-Time Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download datasets

The raw CSV datasets must exist under `datasets/raw/`.

### 3. Create and share Google Sheets credentials

Use a Google service account with access to the experiment spreadsheet.

The service account JSON key should be stored locally, for example:

```bash
mkdir -p ~/.config/catrepbench
mv ~/Downloads/service-account.json ~/.config/catrepbench/gsheets-service-account.json
chmod 600 ~/.config/catrepbench/gsheets-service-account.json
```

### 4. Export environment variables

```bash
export CATREPBENCH_GSHEETS_SERVICE_ACCOUNT_PATH="$HOME/.config/catrepbench/gsheets-service-account.json"
export CATREPBENCH_GSHEETS_SPREADSHEET_ID="<your-spreadsheet-id>"
export CATREPBENCH_GSHEETS_WORKSHEET="CTGAN"
```

To make this persistent, add the same exports to `~/.zshrc` or your shell profile.

## Google Sheets Layout

The worksheet must be a matrix:

- row 1: dataset labels
- column A: encoding labels
- inner cells: experiment state

Important rules:

- dataset labels must match `label` values in `experiments/ctgan_orchestrator_manifest.json`
- encoding labels must match `label` values in `experiments/ctgan_orchestrator_manifest.json`
- matching is case-sensitive after trimming whitespace

### Empty Cells Are Correct

An empty cell means `not-started`.

Do not pre-fill the table with `done`, `in-progress`, or custom text.
The orchestrator writes JSON payloads into cells during execution.

## Before Starting Workers

All machines or users participating in the same batch should use:

- the same git branch or commit
- the same manifest file
- the same worksheet structure

This avoids inconsistent results across workers.

## Dry Run

Always test the setup before a real launch:

```bash
python experiments/ctgan_orchestrator.py \
  --manifest experiments/ctgan_orchestrator_manifest.json \
  --worksheet CTGAN \
  --dry-run
```

This prints the first claimable cell and the runner command it would start.

## Normal Launch

To run a worker in the current terminal:

```bash
python experiments/ctgan_orchestrator.py \
  --manifest experiments/ctgan_orchestrator_manifest.json \
  --worksheet CTGAN
```

Behavior:

- the orchestrator claims the first available job
- starts the full CTGAN runner locally
- updates heartbeat and stage in Google Sheets
- writes `done` or `failed`
- continues to the next claimable job until the queue is empty

## Recommended Launch With tmux

If you want to reconnect later and watch live stdout and stderr, use the `tmux` wrapper.

### Start a worker in tmux

```bash
python experiments/ctgan_orchestrator_tmux.py launch \
  --manifest experiments/ctgan_orchestrator_manifest.json \
  --worksheet CTGAN \
  --index 1
```

This creates a detached `tmux` session, starts the orchestrator inside it, and writes logs to:

- `experiments/results/orchestrator_logs/<session>.log`

### List active orchestrator tmux sessions

```bash
python experiments/ctgan_orchestrator_tmux.py list
```

### Attach to a running session

```bash
python experiments/ctgan_orchestrator_tmux.py attach --session <session-name>
```

### Read the latest log lines without attaching

```bash
python experiments/ctgan_orchestrator_tmux.py tail \
  --session <session-name> \
  --lines 100
```

## Results and Logs

Experiment outputs are stored under:

- `experiments/results/ctgan/<dataset_id>/<encoding_id>/`

Orchestrator `tmux` logs are stored under:

- `experiments/results/orchestrator_logs/`

## Running From Multiple Machines

Multiple workers can use the same Google Sheet at the same time.

Each worker:

1. reads the worksheet
2. claims the first available cell
3. re-checks ownership
4. starts the experiment only if the lease still belongs to that worker

Because of that, several workers can cooperate on one shared queue.

## Onboarding a New User or New Machine

A new contributor needs:

1. repository access
2. the correct git branch or commit
3. the same spreadsheet link
4. a valid Google service account key, or their own service account with spreadsheet access
5. the environment variables from this runbook

Suggested onboarding sequence:

1. clone the repository
2. checkout the agreed branch
3. install `requirements.txt`
4. place the service account JSON locally
5. export the three `CATREPBENCH_GSHEETS_*` variables
6. run `--dry-run`
7. start a normal worker or a `tmux` worker

## Status Semantics

The orchestrator writes structured JSON payloads into cells.

The important fields are:

- `status`
- `stage`
- `note`
- `owner`
- `run_id`
- `heartbeat_at`

Typical `status` values:

- `in-progress`
- `done`
- `failed`

Typical `stage` values:

- `launching`
- `tuning`
- `crossval`
- `metrics`
- `saving`
- `done`
- `failed`

## Failure Notes

The orchestrator classifies failures more precisely than a plain exit code when possible.

Examples:

- `resource_exhausted: ...`
- `runner terminated by signal 9`
- `runner exited with code 1: ...`

This helps distinguish normal Python failures from memory or process-kill issues.

## How To Reset a Job

If you want to make a failed or stale job available again:

- clear the cell content completely in Google Sheets

An empty cell becomes claimable again.

Do not manually edit `run_id`, `owner`, or partial JSON fields.


