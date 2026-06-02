# TabDDPM run — analysis findings & caveats

Observations from the full TabDDPM grid run (21 datasets × 12 encoders), to keep
in mind when writing up results. Append as more surface.

## Forest_Fires utility (TSTR) is uninformative — flag or exclude

`uci_Forest_Fires` target = `area` (burned area, **regression**, ~48% zeros,
heavily skewed). The target is essentially **unpredictable from the features** —
this is a well-known property of the Forest Fires dataset.

Evidence: TSTR R² of a model trained on **REAL** data (the baseline) is **negative
for every encoder** (worse than predicting the mean), so there is no learnable
signal to compare against:

| encoder | r2_real | r2_synth |
|---|---|---|
| one-hot | −1.27 | −8.2 |
| ordinal | −1.37 | −0.02 |
| sum | −0.98 | −0.27 |
| binary | −0.92 | −56.9 |
| gel | −0.95 | −69.9 |
| similarity | −0.97 | −97.7 |
| gumbel-softmax | −0.35 | −68.1 |
| frequency | −1.11 | −29.0 |
| polynomial | −0.93 | −34.4 |
| helmert | tstr=failed (target collapsed to ~constant) | — |

Consequences:
- `r2_synth` swings wildly (−0.02 … −97.7) and the utility-gap `r2_pct_diff`
  is huge/erratic (0.87 … 212) because it divides two negative R² values.
- **This is the dataset's nature, not a TabDDPM/pipeline bug** — CTGAN/TVAE show
  the same negative real-R² pattern on Forest_Fires.

Recommendation:
- Treat Forest_Fires **utility** as unreliable: exclude it from the utility
  comparison, or report a clamped/absolute R² rather than `r2_pct_diff`.
- Forest_Fires **distribution** metrics (WD, KL, correlation) remain valid.
- More generally, any dataset whose real-data TSTR R² is ≤ 0 has no utility
  signal and should be handled the same way.

## Polynomial encoding: predictive champion vs generative liability

In the predictive encoding benchmark of **Clerici & Nobani (2026, IJDSA
s41060-025-00886-w)**, the **polynomial** (trend) contrast is the single best
unsupervised encoder — top F1 for MLP/SVM/LR/DT and top RMSE for RF/K-NN/SVM/DT
(though its Wilcoxon advantage over one-hot is often not significant). In OUR
**generative** setting it is the opposite — a liability:
- On high-cardinality categoricals it **overflows numerically** (patsy
  `scores ** arange(n)`, `RuntimeWarning: overflow encountered in power`).
  `Seoul_Bike_Sharing_Demand` has a ~365-level `Date` column → polynomial
  expands to 379 columns with inf/NaN → no Optuna trial completes → the cell
  fails. **TVAE fails on the same cell too**, confirming it is the encoding, not
  the model.
- Takeaway for the paper: encoder quality does **not** transfer between
  predictive and generative tasks. A contrast designed to expose monotone trends
  for a downstream predictor is numerically ill-posed as a generative target on
  high-cardinality nominal features.

## `frequency` encoding diverges to NaN on large datasets (it is `count`, not relative)

`FrequencyRepresentation` defaults to `method="count"` — **absolute counts**, not
the standard relative frequency. On large datasets these are huge (e.g.
bank-marketing encoded columns reach **35 496**). Even after standardization +
quantile-normal transform, the Gaussian diffusion **sampling diverges to NaN**
(`FoundNANsError`: 10 consecutive all-NaN batches), so every Optuna trial is
pruned and the cell fails (now cleanly, after the tuning-loop cap). The model
trains fine — the failure is in sampling.
- The class already supports `method="normalized"` (count ÷ total → values in
  [0,1]). Switching to it would both **match the textbook definition** of
  frequency encoding and **remove the divergence** (small magnitudes).
- **Decision for Ilya** (changes all `frequency` results, so not done
  automatically): adopt `normalized`, or keep `count` and report these large
  datasets as known frequency failures.

## Loss graphs missing for TVAE in the dashboard — data was never uploaded to Drive

TVAE loss curves do not render because the monitor has **0 `loss_series` for
tvae**: tvae's Drive folders contain only `crossval/ metrics/ tuning/
run_summary.json` — **no `artifacts/` directory** (where `loss_history.csv`
lives), whereas tabddpm/ctgan uploaded `artifacts/`. The loss CSVs **do exist
locally on the cluster** (`experiments/results/tvae/*/*/artifacts/fold_*/loss_history.csv`,
815 files), so the data is real — it was just never uploaded to Drive for tvae.
- Fix options (need Ilya's ok, both touch infra): (a) backfill the monitor's
  Postgres `loss_series` for tvae from the local CSVs, or (b) upload tvae's
  `artifacts/` to Drive and let the normal sync pull them. The monitor code
  path is already fixed (`loss_histories` is sourced from `loss_series` →
  commit `c377c44`); it just has nothing to show for tvae.
