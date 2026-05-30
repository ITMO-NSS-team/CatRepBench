from __future__ import annotations

from typing import Any

# TabPFGen is inference-only: it runs SGLD on top of a frozen pre-trained
# TabPFN, so there is no per-dataset training loop. The shared runner still
# passes an ``epochs`` budget to every model's ``build_model_kwargs``; for
# TabPFGen that argument is ignored (kept for signature compatibility). The
# only "epochs-like" knob is the number of SGLD sampler steps, which defaults
# to TabPFGenGenerative.n_sgld_steps.
DEFAULT_TABPFGEN_SGLD_STEPS = 1000


def _validate_device(device: str) -> str:
    # TabPFGen accepts the same backends as TabPFN plus "auto" for autodetect.
    if device not in {"cpu", "cuda", "mps", "auto"}:
        raise ValueError("device must be 'cpu', 'cuda', 'mps' or 'auto'.")
    return device


def build_tabpfgen_kwargs(
    best_params: dict[str, object],
    *,
    epochs: int,
    device: str,
) -> dict[str, Any]:
    """Map tuner/best_params output to TabPFGenGenerative.__init__ kwargs.

    The shared runner calls every model spec the same way
    (``build_model_kwargs(best_params, epochs=default_epochs, device=device)``).
    TabPFGen does not train, so ``epochs`` is accepted but unused; the sampler's
    ``n_sgld_steps`` (when present in ``best_params``) controls SGLD length.
    All keys map straight onto the wrapper's flat dataclass fields.
    """
    device = _validate_device(device)
    params = dict(best_params)
    return {
        "n_sgld_steps": int(params.get("n_sgld_steps", DEFAULT_TABPFGEN_SGLD_STEPS)),
        "sgld_step_size": float(params.get("sgld_step_size", 0.01)),
        "sgld_noise_scale": float(params.get("sgld_noise_scale", 0.01)),
        "balance_classes": bool(params.get("balance_classes", True)),
        "use_quantiles": bool(params.get("use_quantiles", True)),
        "seed": params.get("seed", None),
        "device": device,
    }
