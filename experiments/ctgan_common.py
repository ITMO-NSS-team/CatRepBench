from __future__ import annotations

from typing import Any

import pandas as pd

from genbench.data.schema import TabularSchema
from genbench.transforms.categorical import (
    CategoricalRepresentationTransform,
    list_registered_representations,
)
from genbench.transforms.continuous import ContinuousStandardScaler
from genbench.transforms.missing import DropMissingRows
from genbench.transforms.pipeline import TransformPipeline
from genbench.transforms.target import TargetTypePreprocessor


def _validate_device(device: str) -> str:
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'.")
    return device


def _validate_encoding_method(encoding_method: str) -> str:
    supported = list_registered_representations()
    if encoding_method not in supported:
        raise ValueError(
            "Unsupported encoding_method. "
            f"Use one of {supported}. Got: '{encoding_method}'."
        )
    return encoding_method


def build_preprocess_pipeline(
    schema: TabularSchema,
    encoding_method: str,
    task_type: str | None,
) -> tuple[TransformPipeline, str | None]:
    representation_name = _validate_encoding_method(encoding_method)
    transforms = [DropMissingRows(), ContinuousStandardScaler()]
    if schema.categorical_cols:
        transforms.append(
            CategoricalRepresentationTransform(
                representation_name=representation_name,
            )
        )
    else:
        representation_name = None
    transforms.append(TargetTypePreprocessor(task_type=task_type))
    return TransformPipeline(transforms=transforms), representation_name


def default_discrete_cols(
    schema: TabularSchema,
    train_df: pd.DataFrame,
    *,
    include_target_for_classification: bool,
) -> list[str]:
    cols = [
        col
        for col in list(schema.discrete_cols) + list(schema.categorical_cols)
        if col in train_df.columns
    ]
    if (
        include_target_for_classification
        and schema.target_col
        and schema.target_col in train_df.columns
        and schema.target_col not in cols
    ):
        cols.append(schema.target_col)
    return cols


def build_ctgan_kwargs(
    best_params: dict[str, object],
    *,
    epochs: int,
    device: str,
) -> dict[str, Any]:
    device = _validate_device(device)
    embedding_dim = int(best_params["embedding_dim"])
    gen_dim = int(best_params["gen_dim"])
    disc_dim = int(best_params["disc_dim"])
    batch_size = int(best_params["batch_size"])
    discriminator_steps = int(best_params["discriminator_steps"])
    generator_lr = float(best_params["generator_lr"])
    lr_ratio = float(best_params["lr_ratio"])

    return {
        "epochs": int(epochs),
        "embedding_dim": embedding_dim,
        "generator_dim": (gen_dim, gen_dim),
        "discriminator_dim": (disc_dim, disc_dim),
        "batch_size": batch_size,
        "discriminator_steps": discriminator_steps,
        "generator_lr": generator_lr,
        "discriminator_lr": generator_lr * lr_ratio,
        "pac": 1,
        "verbose": False,
        "cuda": device == "cuda",
    }
