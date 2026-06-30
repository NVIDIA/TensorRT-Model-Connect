"""CPU profile matrix defaults owned by the Mamba family."""

from __future__ import annotations


def cpu_profile_matrix_specs() -> list[dict]:
    return [{
        "order": 30,
        "strategy": "ssm_recurrent",
        "label": "ssm_recurrent\n(mamba-130m)",
        "hf_id": "state-spaces/mamba-130m-hf",
        "bundle": "mamba-130m.trtfb",
        "runner": "family",
    }]
