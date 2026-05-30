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
