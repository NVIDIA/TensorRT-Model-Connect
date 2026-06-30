"""CPU profile matrix defaults owned by the Qwen3.5 family."""

from __future__ import annotations


def cpu_profile_matrix_specs() -> list[dict]:
    return [{
        "order": 50,
        "strategy": "qwen3_5_hybrid_mamba_attention",
        "label": "hybrid_mamba_attn\n(qwen35-9b)",
        "hf_id": "Qwen/Qwen3-5B",
        "bundle": "qwen35-9b.trtfb",
        "runner": "decoder",
    }]
