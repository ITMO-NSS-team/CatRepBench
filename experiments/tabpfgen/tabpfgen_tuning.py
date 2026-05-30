"""Runner-facing TabPFGen tuning/runtime-estimate adapters.

The shared CTGAN/TVAE runner expects every model to expose a
``select_best_params(...)`` and ``estimate_runtime(...)`` with a fixed keyword
signature (same shape as ``experiments/tvae/tvae_tuning.py`` and
``experiments/tabddpm/tabddpm_tuning.py``).

TabPFGen is inference-only: it runs SGLD on top of a frozen pre-trained TabPFN,
so there is no training loss and no meaningful per-dataset hyperparameter
search in this pipeline. Both adapters are therefore trivial:

  - ``select_tabpfgen_best_params`` performs no tuning and returns an empty
    ``best_params`` dict (the wrapper falls back to its defaults).
  - ``estimate_tabpfgen_runtime`` reports a zero projected runtime; there is no
    training phase to time.

Both functions are pure-python (no ``tabpfn``/``tabpfgen`` import) so the
registry can import them eagerly without requiring the heavy deps.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pandas as pd

from genbench.data.schema import TabularSchema


def select_tabpfgen_best_params(
    *,
    df: pd.DataFrame,
    schema: TabularSchema,
    dataset: str,
    encoding_method: str,
    task_type: Optional[str] = None,
    output_dir: Path | str | None = None,
    device: str = "cuda",
    progress_callback: Optional[Callable[[str], None]] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """No-op tuner: TabPFGen has no per-dataset training to tune.

    Mirrors ``select_tvae_best_params`` / ``select_tabddpm_best_params`` and
    returns ``{"best_params", "best_value", "best_source"}``. ``best_params``
    is empty, so ``build_tabpfgen_kwargs`` uses the wrapper defaults.
    """
    if progress_callback is not None:
        progress_callback("tabpfgen is inference-only; skipping tuning")
    return {
        "best_params": {},
        "best_value": None,
        "best_source": "no_tuning",
    }


def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(name).strip()).strip("_") or "unknown"


def estimate_tabpfgen_runtime(
    *,
    df: pd.DataFrame,
    schema: TabularSchema,
    dataset: str,
    encoding_method: str,
    sample_epochs: int = 10,
    projected_epochs: int = 1000,
    projected_total_runs: int = 35,
    task_type: Optional[str] = None,
    output_dir: Path | str | None = None,
    output_root: Path | str = Path("experiments/optuna_results"),
    device: str = "cuda",
    progress_callback: Optional[Callable[[str], None]] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Trivial runtime estimate: TabPFGen has no training phase to project.

    Matches ``estimate_tabddpm_runtime``'s keyword signature so the runner can
    call it uniformly, but always reports a zero projected pipeline runtime.
    """
    out_dir = Path(output_dir) if output_dir is not None else (
        Path(output_root) / "tabpfgen_runtime_estimate" / _slug(dataset) / _slug(encoding_method)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    payload: Dict[str, Any] = {
        "mode": "runtime_estimate",
        "dataset": dataset,
        "encoding_method": encoding_method,
        "model": "tabpfgen",
        "inference_only": True,
        "note": "TabPFGen has no per-dataset training; runtime estimate is trivial.",
        "projected_total_runs": int(projected_total_runs),
        "projected_fit_seconds": 0.0,
        "projected_trial_seconds": 0.0,
        "projected_full_pipeline_seconds": 0.0,
        "projected_full_pipeline_hours": 0.0,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    if progress_callback is not None:
        progress_callback("tabpfgen is inference-only; projected runtime 0s")

    payload["summary_path"] = str(summary_path)
    payload["output_dir"] = str(out_dir)
    return payload
