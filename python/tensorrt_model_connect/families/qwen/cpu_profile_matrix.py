"""CPU profile matrix defaults owned by the Qwen family."""

from __future__ import annotations


def cpu_profile_matrix_specs() -> list[dict]:
    return [{
        "order": 10,
        "strategy": "qwen_decoder_kv_cache",
        "label": "qwen_decoder_kv_cache\n(qwen3-0.6b)",
        "hf_id": "Qwen/Qwen3-0.6B",
        "bundle": "qwen3-0.6b.trtfb",
        "runner": "decoder",
    }]
