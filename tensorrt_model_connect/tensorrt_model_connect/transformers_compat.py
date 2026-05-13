"""Compatibility shims for external Transformers model code."""

from __future__ import annotations


def patch_legacy_dynamic_cache_api() -> None:
    """Restore cache methods still used by some trusted remote-code models.

    Transformers 5.x removed a few legacy ``DynamicCache`` helpers that older
    remote-code repositories still call during generation.  The shim is small
    and idempotent so reference subprocesses can install it before loading a
    trusted remote-code model.
    """
    try:
        from transformers.cache_utils import DynamicCache
    except Exception:
        return

    if not hasattr(DynamicCache, "from_legacy_cache"):

        @classmethod
        def from_legacy_cache(cls, past_key_values=None, *args, **kwargs):
            if past_key_values is None:
                return cls()
            return cls(past_key_values)

        DynamicCache.from_legacy_cache = from_legacy_cache

    if (
        not hasattr(DynamicCache, "get_max_length")
        and hasattr(DynamicCache, "get_max_cache_shape")
    ):

        def get_max_length(self):
            return self.get_max_cache_shape()

        DynamicCache.get_max_length = get_max_length
