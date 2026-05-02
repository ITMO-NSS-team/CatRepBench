from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from experiments.ctgan.ctgan_common import DEFAULT_CTGAN_EPOCHS, build_ctgan_kwargs
from experiments.tvae.tvae_common import DEFAULT_TVAE_EPOCHS, build_tvae_kwargs
from genbench.generative.base import BaseGenerative
from genbench.generative.ctgan.ctgan import CtganGenerative
from genbench.generative.tvae.tvae import TvaeGenerative


@dataclass(frozen=True)
class ExperimentModelSpec:
    model_id: str
    display_name: str
    artifact_filename: str
    default_epochs: int
    build_model_kwargs: Callable[..., dict[str, Any]]
    create_generative: Callable[[list[str], dict[str, Any]], BaseGenerative]
    select_best_params: Callable[..., dict[str, Any]]
    estimate_runtime: Callable[..., Any]


def _create_ctgan(discrete_cols: list[str], model_kwargs: dict[str, Any]) -> BaseGenerative:
    return CtganGenerative(discrete_cols=discrete_cols, ctgan_kwargs=model_kwargs)


def _create_tvae(discrete_cols: list[str], model_kwargs: dict[str, Any]) -> BaseGenerative:
    return TvaeGenerative(discrete_cols=discrete_cols, tvae_kwargs=model_kwargs)


def _select_ctgan_best_params(**kwargs: Any) -> dict[str, Any]:
    from experiments.ctgan.ctgan_tuning import select_ctgan_best_params

    return select_ctgan_best_params(**kwargs)


def _estimate_ctgan_runtime(**kwargs: Any) -> Any:
    from experiments.ctgan.ctgan_tuning import estimate_ctgan_runtime

    return estimate_ctgan_runtime(**kwargs)


def _select_tvae_best_params(**kwargs: Any) -> dict[str, Any]:
    from experiments.tvae.tvae_tuning import select_tvae_best_params

    return select_tvae_best_params(**kwargs)


def _estimate_tvae_runtime(**kwargs: Any) -> Any:
    from experiments.tvae.tvae_tuning import estimate_tvae_runtime

    return estimate_tvae_runtime(**kwargs)


_MODELS: dict[str, ExperimentModelSpec] = {
    "ctgan": ExperimentModelSpec(
        model_id="ctgan",
        display_name="CTGAN",
        artifact_filename="ctgan.pkl",
        default_epochs=DEFAULT_CTGAN_EPOCHS,
        build_model_kwargs=build_ctgan_kwargs,
        create_generative=_create_ctgan,
        select_best_params=_select_ctgan_best_params,
        estimate_runtime=_estimate_ctgan_runtime,
    ),
    "tvae": ExperimentModelSpec(
        model_id="tvae",
        display_name="TVAE",
        artifact_filename="tvae.pkl",
        default_epochs=DEFAULT_TVAE_EPOCHS,
        build_model_kwargs=build_tvae_kwargs,
        create_generative=_create_tvae,
        select_best_params=_select_tvae_best_params,
        estimate_runtime=_estimate_tvae_runtime,
    ),
}


def list_experiment_models() -> tuple[str, ...]:
    return tuple(_MODELS)


def get_experiment_model(model_id: str) -> ExperimentModelSpec:
    normalized = str(model_id).strip().lower()
    try:
        return _MODELS[normalized]
    except KeyError as exc:
        available = ", ".join(list_experiment_models())
        raise ValueError(f"Unknown model_id {model_id!r}. Use one of: {available}") from exc
