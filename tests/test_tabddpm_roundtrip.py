"""TabDDPM save/load round-trip with a NON-default architecture.

Regression test for the load_artifacts bug where the architecture hyperparameters
(d_layers/dim_t/num_timesteps/...) were not restored from the saved params, so
load_state_dict raised a size-mismatch for any non-default config. We deliberately
fit with d_layers=[64, 64] (default is [256, 256, 256]) so the bug would surface.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from genbench.data.schema import TabularSchema
from genbench.generative.tabddpm.tabddpm import TabDDPMGenerative


def _tiny_frame(n: int = 120) -> pd.DataFrame:
    # TabDDPM fits on ALREADY-PREPROCESSED data (BaseGenerative contract): the
    # encoder has converted categoricals to integer codes before fit, so 'cat'
    # is integer-coded here (the model does df[cat_cols].astype(int64)).
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(5.0, 2.0, size=n),
            "cat": rng.integers(0, 3, size=n),
            "type": rng.integers(0, 2, size=n),  # classification target
        }
    )


def test_tabddpm_save_load_roundtrip_nondefault_arch() -> None:
    df = _tiny_frame()
    schema = TabularSchema.infer_from_dataframe(
        df, target_col="type", categorical_cols=["cat"]
    )

    # NON-default architecture so the size-mismatch bug would trigger on reload.
    model = TabDDPMGenerative(
        num_steps=30,
        num_timesteps=50,
        d_layers=[64, 64],
        dim_t=32,
        batch_size=64,
        device="cpu",
    )
    model.fit(df, schema)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "art"
        model.save_artifacts(out)
        loaded = TabDDPMGenerative.load_artifacts(out)

        # Architecture hyperparameters must be restored from the saved params.
        assert loaded.d_layers == [64, 64]
        assert loaded.dim_t == 32
        assert loaded.num_timesteps == 50

        # Sampling must work after reload (state_dict loaded with no size mismatch).
        sampled = loaded.sample(16)
        assert len(sampled) == 16
        for col in ("x1", "x2", "cat", "type"):
            assert col in sampled.columns
