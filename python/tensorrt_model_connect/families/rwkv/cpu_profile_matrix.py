"""CPU profile matrix defaults owned by the RWKV family."""

from __future__ import annotations


def cpu_profile_matrix_specs() -> list[dict]:
    return [{
        "order": 40,
        "strategy": "rwkv_recurrent",
        "label": "rwkv_recurrent\n(rwkv-169m)",
        "hf_id": "RWKV/rwkv-4-169m-pile",
        "bundle": "rwkv-169m.trtfb",
        "runner": "family",
    }]
