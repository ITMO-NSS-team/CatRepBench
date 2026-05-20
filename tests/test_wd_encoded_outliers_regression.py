"""
Regression test for the Mean WD inflation bug.

When TabDDPM produces outliers on encoded-categorical columns
(frequency/polynomial/GEL/similarity/... representations can yield
arbitrary unbounded values whose scale ContinuousStandardScaler never
normalized, because it runs BEFORE the categorical encoder), the
Wasserstein distance applied over those columns blows up by orders of
magnitude — we observed WD up to ~1e9 in our CSVs.

The fix in experiments/tabddpm_metrics.py and
experiments/tabddpm_tuning.py restricts WD to the ORIGINAL continuous
columns from the source schema. This test keeps a separate module so it
does not import the catboost-dependent TSTR helpers at collection time.
"""
import numpy as np
import pandas as pd

from genbench.data.schema import TabularSchema
from genbench.evaluation.distribution.wasserstein import (
    WassersteinDistanceMetric,
)


def test_wd_isolated_from_encoded_categorical_outliers():
    rng = np.random.default_rng(42)
    n = 1000

    real = pd.DataFrame({
        "x_cont": rng.normal(0.0, 1.0, size=n),
        "freq_enc_x": rng.uniform(0.0, 1.0, size=n),
    })
    synth = pd.DataFrame({
        "x_cont": rng.normal(0.05, 1.05, size=n),
        # synth ran away on the encoded column (this is what we see in
        # practice for many representations).
        "freq_enc_x": np.concatenate([
            rng.normal(0.0, 1e3, size=n - 1),
            [1e6],
        ]),
    })

    schema_with_encoded = TabularSchema(
        continuous_cols=["x_cont", "freq_enc_x"],
        discrete_cols=[],
        categorical_cols=[],
    )
    schema_continuous_only = TabularSchema(
        continuous_cols=["x_cont"],
        discrete_cols=[],
        categorical_cols=[],
    )

    metric = WassersteinDistanceMetric()
    wd_with_encoded = metric.compute(real, synth, schema_with_encoded)
    wd_continuous_only = metric.compute(real, synth, schema_continuous_only)

    # The bug manifests as WD orders of magnitude larger when encoded
    # columns are included; the fix keeps WD in the natural 0-3 range.
    assert wd_continuous_only < 1.0
    assert wd_with_encoded > 100.0 * wd_continuous_only
