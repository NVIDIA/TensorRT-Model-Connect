# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned Edge offload and native TensorRT decoder composition."""


def build(request, writer) -> None:
    """Build the selected task through this family's reviewed Edge route.

    Args:
        request: Original Model Connect request.
        writer: Unpublished output bundle writer.

    Raises:
        Exception: Unsupported request, unavailable dependency, or build failure.
    """
    from .build_request import coerce_request
    from .edge_llm.dispatch import build as dispatch_build

    request = coerce_request(request)

    if request.dynamic_kv_cache:
        raise NotImplementedError("qwen3_5_moe does not support dynamic_kv_cache")
    dispatch_build(request, writer, _build_native)


def _build_native(request, writer) -> None:
    """Preserve the original target, media and speculative request on fallback."""
    from .native_model import _build_native as native_build

    execution = getattr(request, "execution", None)
    options = {}
    if execution is not None:
        if execution.variant == "mtp":
            options["require_mtp"] = True
        else:
            options.update(draft_dir=execution.checkpoints[0].model_dir,
                           draft_variant=execution.variant)
    writer.set_header(family="qwen3_5_moe", task=request.task, backend=request.backend)
    native_build(request, writer, **options)
