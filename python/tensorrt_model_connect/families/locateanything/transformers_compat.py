# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transformers compatibility for the pinned LocateAnything remote code."""

from __future__ import annotations

import sys
from typing import Any, Callable


_ATTENTION_METHOD = "_check_and_adjust_attn_implementation"


def install_remote_attention_compat(model_class: type[Any]) -> int:
    """Accept the Transformers 5.5 attention keyword in pinned remote classes.

    The pinned LocateAnything overrides predate ``allow_all_kernels``.  Keep
    their original behavior, including the conservative default for remote
    kernels, while accepting the new optional keyword passed by Transformers.
    """
    module = sys.modules.get(model_class.__module__)
    if module is None:
        raise RuntimeError(f"LocateAnything remote module {model_class.__module__!r} is not loaded")

    remote_package = model_class.__module__.rsplit(".", 1)[0]
    roots = (
        model_class,
        getattr(module, "Qwen2ForCausalLM", None),
        getattr(module, "Qwen3ForCausalLM", None),
    )
    patched = 0
    seen: set[type[Any]] = set()
    for root in roots:
        if not isinstance(root, type):
            continue
        for candidate in root.__mro__:
            if candidate in seen or not candidate.__module__.startswith(remote_package):
                continue
            seen.add(candidate)
            original = candidate.__dict__.get(_ATTENTION_METHOD)
            if original is None or getattr(original, "_trtmc_transformers_55", False):
                continue
            setattr(candidate, _ATTENTION_METHOD, _accept_allow_all_kernels(original))
            patched += 1
    return patched


def _accept_allow_all_kernels(original: Callable[..., Any]) -> Callable[..., Any]:
    def compatible(
        self: Any,
        attn_implementation: str | None,
        is_init_check: bool = False,
        allow_all_kernels: bool = False,
    ) -> str:
        # The pinned implementation cannot opt into unverified kernels.  Ignore
        # the new opt-in instead of broadening its existing security behavior.
        del allow_all_kernels
        return original(self, attn_implementation, is_init_check)

    compatible._trtmc_transformers_55 = True  # type: ignore[attr-defined]
    return compatible
