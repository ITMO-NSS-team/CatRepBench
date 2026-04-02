from __future__ import annotations

import pandas as pd
import pytest

from experiments.ctgan_common import (
    build_ctgan_kwargs,
    build_preprocess_pipeline,
    default_discrete_cols,
)
from genbench.data.schema import TabularSchema


def test_build_ctgan_kwargs_materializes_dims_for_cpu():
    out = build_ctgan_kwargs(
        {
            "embedding_dim": 128,
            "gen_dim": 256,
            "disc_dim": 512,
            "batch_size": 256,
            "discriminator_steps": 1,
            "generator_lr": 1e-3,
            "lr_ratio": 1.0,
        },
        epochs=5,
        device="cpu",
    )
    assert out["generator_dim"] == (256, 256)
    assert out["discriminator_dim"] == (512, 512)
    assert out["cuda"] is False


@pytest.mark.parametrize("device", ["gpu", "CUDA"])
def test_build_ctgan_kwargs_raises_on_invalid_device(device: str):
    with pytest.raises(ValueError, match="device must be 'cpu' or 'cuda'"):
        build_ctgan_kwargs(
            {
                "embedding_dim": 128,
                "gen_dim": 256,
                "disc_dim": 512,
                "batch_size": 256,
                "discriminator_steps": 1,
                "generator_lr": 1e-3,
                "lr_ratio": 1.0,
            },
            epochs=5,
            device=device,
        )


def test_build_preprocess_pipeline_includes_representation_and_target_steps():
    schema = TabularSchema(
        continuous_cols=["x_cont"],
        discrete_cols=["x_disc"],
        categorical_cols=["x_cat"],
        target_col="target",
    )
    pipeline, representation_name = build_preprocess_pipeline(
        schema=schema,
        encoding_method="one_hot_representation",
        task_type=None,
    )
    names = [type(step).__name__ for step in pipeline.transforms]
    assert representation_name == "one_hot_representation"
    assert names == [
        "DropMissingRows",
        "ContinuousStandardScaler",
        "CategoricalRepresentationTransform",
        "TargetTypePreprocessor",
    ]


def test_default_discrete_cols_includes_target_only_for_classification():
    schema = TabularSchema(
        continuous_cols=["x_cont"],
        discrete_cols=["x_disc"],
        categorical_cols=["x_cat"],
        target_col="target",
    )
    train_df = pd.DataFrame(
        {
            "x_cont": [0.1, 0.2],
            "x_disc": [1, 2],
            "x_cat": ["a", "b"],
            "target": [0, 1],
        }
    )

    assert default_discrete_cols(
        schema,
        train_df,
        include_target_for_classification=True,
    ) == ["x_disc", "x_cat", "target"]
    assert default_discrete_cols(
        schema,
        train_df,
        include_target_for_classification=False,
    ) == ["x_disc", "x_cat"]
