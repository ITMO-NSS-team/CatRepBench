"""
Optional workaround for environments where api.priorlabs.ai is unreachable
(403 Forbidden / geo-blocked / firewalled) but the user already has accepted
the TabPFN licenses on https://ux.priorlabs.ai/account/licenses.

When the environment variable TABPFN_PRIORLABS_BYPASS is set to a truthy
value, this module monkey-patches tabpfn.browser_auth.ensure_license_accepted
so it short-circuits without hitting PriorLabs. Weights are still downloaded
from HuggingFace (Prior-Labs/* repos), so a valid HF_TOKEN with accepted
model-card gates is still required.

Usage: import this module BEFORE importing tabpfn or tabpfgen.
"""
from __future__ import annotations

import os


def _truthy(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def apply_bypass_if_requested() -> bool:
    """Patch tabpfn.browser_auth if TABPFN_PRIORLABS_BYPASS is set.

    Returns True iff the patch was applied.
    """
    if not _truthy(os.environ.get("TABPFN_PRIORLABS_BYPASS", "")):
        return False

    try:
        import tabpfn.browser_auth as ba
    except ImportError:
        return False

    def _bypass(hf_repo_id: str, **_kw):
        ba._accepted_repos.add(hf_repo_id)
        return True

    ba.ensure_license_accepted = _bypass
    print(
        "[priorlabs_bypass] TABPFN_PRIORLABS_BYPASS active — skipping "
        "PriorLabs license verification. Weights are still fetched from "
        "HuggingFace; ensure HF_TOKEN is configured and gates are accepted "
        "for Prior-Labs/* repos."
    )
    return True


apply_bypass_if_requested()
