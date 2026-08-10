# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep Transformers from treating Lance's local fallback as flash-attn."""

from __future__ import annotations

import transformers.utils as transformers_utils
import transformers.cache_utils as cache_utils
import transformers.utils.import_utils as import_utils


def _compiled_flash_attn_is_unavailable() -> bool:
    return False


for name in (
    "is_flash_attn_2_available",
    "is_flash_attn_3_available",
    "is_flash_attn_4_available",
):
    setattr(import_utils, name, _compiled_flash_attn_is_unavailable)
    setattr(transformers_utils, name, _compiled_flash_attn_is_unavailable)

Cache = cache_utils.Cache
DynamicCache = cache_utils.DynamicCache
if not hasattr(Cache, "get_usable_length"):
    def _get_usable_length(self, new_seq_length, layer_idx=0):
        previous = self.get_seq_length(layer_idx)
        maximum = self.get_max_cache_shape()
        if maximum is None or maximum < 0:
            return previous
        if previous + new_seq_length > maximum:
            return max(0, maximum - new_seq_length)
        return previous

    Cache.get_usable_length = _get_usable_length
if not hasattr(DynamicCache, "to_legacy_cache"):
    def _to_legacy_cache(self):
        return tuple((layer.keys, layer.values) for layer in self.layers)

    DynamicCache.to_legacy_cache = _to_legacy_cache
if not hasattr(DynamicCache, "from_legacy_cache"):
    @classmethod
    def _from_legacy_cache(cls, past_key_values=None):
        cache = cls()
        for layer_idx, (key_states, value_states) in enumerate(
            past_key_values or ()
        ):
            cache.update(key_states, value_states, layer_idx)
        return cache

    DynamicCache.from_legacy_cache = _from_legacy_cache
if not hasattr(cache_utils, "SlidingWindowCache"):
    class _LegacySlidingWindowCache:
        pass

    cache_utils.SlidingWindowCache = _LegacySlidingWindowCache
