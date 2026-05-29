from __future__ import annotations

from typing import Any

# TabDDPM is trained for a fixed number of optimizer *steps* (not epochs).
# Matches TabDDPMGenerative.num_steps default.
DEFAULT_TABDDPM_NUM_STEPS = 5000


def _validate_device(device: str) -> str:
    if device not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be 'cpu', 'cuda' or 'mps'.")
    return device


def build_tabddpm_kwargs(
    best_params: dict[str, object],
    *,
    epochs: int,
    device: str,
) -> dict[str, Any]:
    """Map tuner/best_params output to TabDDPMGenerative.__init__ kwargs.

    The shared runner calls every model spec the same way
    (``build_model_kwargs(best_params, epochs=default_epochs, device=device)``).
    For TabDDPM ``epochs`` carries the default training-step budget; the
    sampled ``num_steps`` (when present in ``best_params``) takes precedence.
    All other keys map straight onto the model's flat dataclass fields.
    """
    device = _validate_device(device)
    params = dict(best_params)
    num_steps = int(params.get("num_steps", epochs))
    return {
        "num_timesteps": int(params.get("num_timesteps", 1000)),
        "num_steps": num_steps,
        "batch_size": int(params.get("batch_size", 1024)),
        "lr": float(params.get("lr", 0.002)),
        "weight_decay": float(params.get("weight_decay", 1e-4)),
        "dim_t": int(params.get("dim_t", 128)),
        "d_layers": [int(d) for d in params.get("d_layers", [256, 256, 256])],
        "dropout": float(params.get("dropout", 0.0)),
        "scheduler": str(params.get("scheduler", "cosine")),
        "gaussian_loss_type": str(params.get("gaussian_loss_type", "mse")),
        "device": device,
    }
