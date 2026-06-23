"""InternVL-owned Hugging Face reference tests."""

from __future__ import annotations

import inspect

from tests.e2e.models.internvl.e2e_plugins.references import (
    hf_transformers as internvl_hf_transformers,
)


def test_owner_reference_uses_image_placeholder_fallback() -> None:
    source = inspect.getsource(
        internvl_hf_transformers.HfTransformersReference._run_vl_full_generation
    )
    assert 'fallback_text = f"<IMG_CONTEXT>\\n{prompt}"' in source
