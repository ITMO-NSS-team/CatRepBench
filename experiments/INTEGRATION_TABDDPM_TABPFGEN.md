# TabDDPM & TabPFGen — integration into the shared experiment format

Both models are registered as `ExperimentModelSpec` in
`experiments/ctgan/experiment_models.py` and run through the **same**
`experiments/ctgan/ctgan_full_experiment.py` runner as CTGAN/TVAE, producing the
identical artifact tree the web monitor reads:
`metrics/aggregate.json` (distribution `wasserstein_mean[_unencoded]`,
`marginal_kl_mean[_unencoded]`, `corr_frobenius_unencoded` + `tstr`),
`crossval/per_fold/fold_*.json`, `run_summary.json`,
`artifacts/fold_*/{<model>.pkl, loss_history.csv}`.

`list_experiment_models()` → `('ctgan', 'tvae', 'tabddpm', 'tabpfgen')`.

## Per-cell run (one dataset × one encoder)
```bash
PYTHONPATH=$PWD python experiments/ctgan/ctgan_full_experiment.py \
  --manifest experiments/ctgan/orchestrator_staff/ctgan_orchestrator_manifest.json \
  --dataset-id <id> --dataset-label <label> --encoding-method <encoder_id> \
  --model-id tabddpm   # or tabpfgen \
  --device cuda        # cluster GPU (cuda → mps → cpu auto-resolves) \
  --output-root experiments/results
```
Tuning is via the model's `select_*_best_params` (Optuna for TabDDPM, none for
TabPFGen); pass `--skip-tuning --best-params-file <json>` to reuse params.

## TabDDPM
- num_steps API; MPS-safe sampling (float32); `load_artifacts` restores the exact
  architecture from saved params and reloads onto the locally-available device
  (cuda→mps→cpu) — so GPU-trained artifacts reload anywhere. Round-trip test:
  `tests/test_tabddpm_roundtrip.py`.
- NaN handling is transparent: bounded sample retries then raise `FoundNANsError`
  with a reason; fold errors are logged, not silently dropped.
- **Continuous-value clipping to [min,max] is intentionally NOT applied** (rejected
  methodology). The encoded-space WD can be large/unstable on continuous columns —
  this is an open methodology question (more steps / beta schedule / robust WD /
  transparent fold-drop), NOT a code bug.
- Full 26×12 grid needs a CUDA GPU/cluster (days-to-weeks on CPU; M1 MPS is fine
  for smoke only).

## TabPFGen
- Inference-only (SGLD over a frozen pre-trained TabPFN): `get_loss_history()→None`,
  no real tuning (`select_tabpfgen_best_params → {}`), `estimate_runtime → 0`.
- The wrapper imports `tabpfn`/`tabpfgen` **lazily inside `fit()`**, and the
  `_create_tabpfgen` factory imports the wrapper lazily — so the registry/runner
  load fine for CTGAN/TVAE/TabDDPM even when those packages are absent.
- **Running it requires gated TabPFN weights**: set `HF_TOKEN` (with the Prior-Labs
  model-card gates accepted) and network access; otherwise the run is blocked at
  weight download. Not runnable unattended without credentials.
- **Dependency caveat (cluster env):** `tabpfgen==0.1.4` hard-pins
  `scikit-learn==1.5.2`, which breaks `category_encoders` (needs `sklearn>=1.6`,
  uses `sklearn.utils.Tags`). Use **`scikit-learn>=1.6`** (we run 1.8.0): the
  encoder stack works and TabPFGen only uses stable sklearn APIs. Expect a
  non-fatal pip version-conflict warning.

## Verified
- TabDDPM smoke through the shared runner on MPS **and** CPU (openml_zoo × one-hot ×
  poster-fast) → correct monitor artifact tree, TSTR ok.
- `tests/test_tabddpm_roundtrip.py` passes (non-default arch reload).
- Registry imports with `tabpfn` absent; all four model specs resolve.

## Making the model tabs appear in the table + web monitor
The orchestrator (`ctgan_orchestrator.py --worksheet <tab> --model-id <model>`) is
already model-agnostic, but it only *writes into* an existing tab — it never
creates one. The CTGAN/TVAE status tabs were made by hand, so a new model has no
tab and nothing shows up in the table or the monitor. Create the tab first:
```bash
# preview (no sheet changes)
python -m experiments.ctgan.orchestrator_staff.bootstrap_model_worksheet \
    --worksheet TabDDPM --dry-run
# create the empty dataset x encoder grid (all cells = pending)
python -m experiments.ctgan.orchestrator_staff.bootstrap_model_worksheet --worksheet TabDDPM
python -m experiments.ctgan.orchestrator_staff.bootstrap_model_worksheet --worksheet TabPFGen
```
The grid is built straight from the manifest (22 datasets x 12 encoders) and is
verified to pass the orchestrator's own `validate_worksheet_headers`. Then run the
orchestrator with `--worksheet TabDDPM --model-id tabddpm` to populate cells.

The **web monitor** (separate repo `ctgan-monitor-railway`) reads these tabs; it
needs the new worksheet names + model keys registered so the new model views
appear on the site. That is a code change in that repo (not auto-discovered).

## Not done (needs you / a GPU cluster / credentials)
- The full grid run (compute) — launch on the cluster with `--device cuda`.
- TabPFGen actual run — needs `HF_TOKEN` + accepted gates.
- **Create the `TabDDPM`/`TabPFGen` tabs** by running `bootstrap_model_worksheet`
  above from a host where the Sheets API is reachable. NOTE: this Claude session
  cannot do it — `sheets.googleapis.com` is blocked at the network level here
  (confirmed: times out with and without the sandbox; `oauth2.googleapis.com`
  resolves so auth succeeds then gspread hangs on the values call).
- **Web monitor model views** for TabDDPM/TabPFGen — blocked from this session by
  macOS TCC on the monitor repo (`Operation not permitted`).
- Methodology call on continuous-WD instability for TabDDPM.
