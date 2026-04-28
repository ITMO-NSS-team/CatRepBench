import os
import sys
from pathlib import Path
from types import SimpleNamespace

from experiments.ctgan.orchestrator_staff.ctgan_runtime_env import prepare_runtime_env


def test_prepare_runtime_env_resolves_drive_paths_and_repairs_missing_ca_bundles(monkeypatch, tmp_path):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "\n".join(
            [
                "CATREPBENCH_GDRIVE_OAUTH_TOKEN_PATH=oauth_token.json",
                "CATREPBENCH_GSHEETS_SERVICE_ACCOUNT_PATH=service-account.json",
                f"SSL_CERT_FILE={tmp_path / 'missing-ssl.pem'}",
                f"REQUESTS_CA_BUNDLE={tmp_path / 'missing-requests.pem'}",
                f"CURL_CA_BUNDLE={tmp_path / 'missing-curl.pem'}",
            ]
        ),
        encoding="utf-8",
    )
    ca_bundle = tmp_path / "certifi.pem"
    ca_bundle.write_text("test-ca", encoding="utf-8")

    monkeypatch.setitem(sys.modules, "certifi", SimpleNamespace(where=lambda: str(ca_bundle)))
    monkeypatch.delenv("CATREPBENCH_GDRIVE_OAUTH_TOKEN_PATH", raising=False)
    monkeypatch.delenv("CATREPBENCH_GSHEETS_SERVICE_ACCOUNT_PATH", raising=False)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
    monkeypatch.delenv("CURL_CA_BUNDLE", raising=False)

    prepare_runtime_env(dotenv_path=dotenv_path)

    assert Path(os.environ["CATREPBENCH_GDRIVE_OAUTH_TOKEN_PATH"]) == tmp_path / "oauth_token.json"
    assert Path(os.environ["CATREPBENCH_GSHEETS_SERVICE_ACCOUNT_PATH"]) == tmp_path / "service-account.json"
    assert os.environ["SSL_CERT_FILE"] == str(ca_bundle)
    assert os.environ["REQUESTS_CA_BUNDLE"] == str(ca_bundle)
    assert os.environ["CURL_CA_BUNDLE"] == str(ca_bundle)
