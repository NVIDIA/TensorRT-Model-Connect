"""Qwen-VL-owned Hugging Face reference tests."""

from __future__ import annotations

import inspect

from tests.e2e.models.qwen_vl.e2e_plugins.references import (
    hf_transformers as qwen_vl_hf_transformers,
)


def test_owner_reference_uses_image_pad_fallback() -> None:
    source = inspect.getsource(
        qwen_vl_hf_transformers.HfTransformersReference._run_vl_full_generation
    )
    assert 'fallback_text = f"<|vision_start|><|image_pad|><|vision_end|>{prompt}"' in source
