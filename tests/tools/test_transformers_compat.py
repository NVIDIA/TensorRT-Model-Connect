"""Tests for compatibility shims around external Transformers model code."""

from __future__ import annotations


def test_patch_legacy_dynamic_cache_api_restores_internlm_remote_code_hooks() -> None:
    from transformers.cache_utils import DynamicCache

    from tensorrt_model_connect.transformers_compat import (
        patch_legacy_dynamic_cache_api,
    )

    patch_legacy_dynamic_cache_api()

    assert hasattr(DynamicCache, "from_legacy_cache")
    assert hasattr(DynamicCache, "get_max_length")
    assert isinstance(DynamicCache.from_legacy_cache(None), DynamicCache)
