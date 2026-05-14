from __future__ import annotations

from typing import Any

DEFAULT_TVAE_EPOCHS = 300


def _validate_device(device: str) -> str:
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'.")
    return device


def build_tvae_kwargs(
    best_params: dict[str, object],
    *,
    epochs: int,
    device: str,
    verbose: bool = False,
) -> dict[str, Any]:
    device = _validate_device(device)
    compress_dim = int(best_params["compress_dim"])
    decompress_dim = int(best_params["decompress_dim"])
    return {
        "epochs": int(epochs),
        "embedding_dim": int(best_params["embedding_dim"]),
        "compress_dims": (compress_dim, compress_dim),
        "decompress_dims": (decompress_dim, decompress_dim),
        "batch_size": int(best_params["batch_size"]),
        "l2scale": float(best_params["l2scale"]),
        "loss_factor": float(best_params["loss_factor"]),
        "cuda": device == "cuda",
        "verbose": bool(verbose),
    }
