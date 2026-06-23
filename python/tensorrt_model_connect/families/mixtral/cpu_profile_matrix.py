"""CPU profile matrix defaults owned by the Mixtral family."""

from __future__ import annotations


def cpu_profile_matrix_specs() -> list[dict]:
    return [{
        "order": 20,
        "strategy": "decoder_moe",
        "label": "decoder_moe\n(mixtral-stories-15m)",
        "hf_id": "RichardErkhov/mistralai_-_Mixtral-8x7B-v0.1-Stories-15M",
        "bundle": "mixtral-stories-15m.trtfb",
        "runner": "decoder",
    }]
