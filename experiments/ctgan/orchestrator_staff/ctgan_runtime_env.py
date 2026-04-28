from __future__ import annotations

import os
from pathlib import Path


_RELATIVE_PATH_ENV_KEYS = (
    "CATREPBENCH_GDRIVE_OAUTH_TOKEN_PATH",
    "CATREPBENCH_GSHEETS_SERVICE_ACCOUNT_PATH",
)
_CA_BUNDLE_ENV_KEYS = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")


def prepare_runtime_env(*, dotenv_path: Path | None = None) -> None:
    loaded_dotenv = _load_dotenv(dotenv_path=dotenv_path)
    if loaded_dotenv is not None:
        _resolve_relative_env_paths(loaded_dotenv.parent)
    _repair_invalid_ca_bundle_env()


def _load_dotenv(*, dotenv_path: Path | None = None) -> Path | None:
    if dotenv_path is not None:
        env_file = dotenv_path
        if not env_file.exists():
            return None
        _load_dotenv_file(env_file)
        return env_file

    for parent in Path(__file__).resolve().parents:
        env_file = parent / ".env"
        if env_file.exists():
            _load_dotenv_file(env_file)
            return env_file
    return None


def _load_dotenv_file(env_file: Path) -> None:
    _load_simple_dotenv(env_file)


def _load_simple_dotenv(env_file: Path) -> None:
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        else:
            value = value.split("#", 1)[0].strip()
        os.environ[key] = value


def _resolve_relative_env_paths(base_dir: Path) -> None:
    for key in _RELATIVE_PATH_ENV_KEYS:
        raw_value = os.environ.get(key, "").strip()
        if not raw_value:
            continue
        path = Path(raw_value).expanduser()
        if path.is_absolute():
            continue
        os.environ[key] = str((base_dir / path).resolve())


def _repair_invalid_ca_bundle_env() -> None:
    invalid_keys = [
        key
        for key in _CA_BUNDLE_ENV_KEYS
        if os.environ.get(key) and not Path(os.environ[key]).exists()
    ]
    if not invalid_keys:
        return
    try:
        import certifi
    except Exception:
        for key in invalid_keys:
            os.environ.pop(key, None)
        return
    ca_bundle = certifi.where()
    for key in invalid_keys:
        os.environ[key] = ca_bundle
