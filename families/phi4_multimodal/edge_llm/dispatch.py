# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned complete-network route map; native is the default."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import tempfile
import traceback

from . import builder as edge_llm

_LOG = logging.getLogger(__name__)

# Complete native offload only; no cross compilation or guessed platform aliases.
EDGE_DISPATCH = {
    ("linux", arch, sm, weights): edge_llm.prepare
    for arch, sms in (("x86_64", (80, 120)),)
    for sm in sms
    for weights in (("fp16", "int4_awq") if sm == 80 else ("fp16", "fp8", "nvfp4"))
}


def mapped_request(request) -> bool:
    """Preserve native-only controls before reading any checkpoint metadata."""
    return (
        request.backend == "trt"
        and request.task == "vision_language_generation"
        and request.precision.lower() == "fp16"
        and request.quantization in {None, "none", "fp8", "nvfp4", "int4_awq"}
        and request.max_batch_size in {1, 2}
        and request.tensor_parallel_size
        == request.context_parallel_size
        == 1
        and not request.dynamic_kv_cache
        and not request.fp32_layers
        and getattr(request, "graph_transform", None) is None
        and all(
            value is None
            for value in (request.image_height, request.image_width, request.video_num_frames)
        )
    )


def candidate(request, raw: dict) -> bool:
    """Admit original vision-LoRA or already merged ModelOpt image checkpoints."""
    vision = raw.get("vision_config") or {}
    if not isinstance(vision, dict):
        return False
    lora = raw.get("vision_lora")
    merged = lora is None and (raw.get("quantization_config") or {}).get("quant_method") == "modelopt"
    if merged:
        if request.max_batch_size != 2 or request.max_sequence_length != 8192:
            return False
    elif not (isinstance(lora, dict) and lora.get("r") == 256
              and lora.get("lora_alpha") == 512 and request.max_batch_size == 1):
        return False
    shape = tuple(
        raw.get(key)
        for key in (
            "num_hidden_layers",
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "vocab_size",
        )
    )
    return (
        raw.get("model_type") in {"phi4mm", "phi4_multimodal"}
        and raw.get("architectures") == ["Phi4MMForCausalLM"]
        and all(type(value) is int for value in shape)
        and shape == (32, 3072, 8192, 24, 8, 200064)
        and raw.get("partial_rotary_factor") == 0.75
        and raw.get("eos_token_id") == 199999
        and raw.get("hidden_act", "silu") == "silu"
        and not any(
            raw.get(key)
            for key in (
                "num_experts",
                "num_local_experts",
                "draft_config",
                "dflash_config",
                "dspark_config",
                "jetspec_config",
                "eagle_config",
                "speculative_config",
                "text_config",
            )
        )
        and all(
            vision.get(key, expected) == expected
            for key, expected in (
                ("hidden_size", 1152),
                ("intermediate_size", 4304),
                ("num_hidden_layers", 27),
                ("num_attention_heads", 16),
                ("image_size", 448),
                ("patch_size", 14),
            )
        )
        and mapped_request(request)
    )


def build(request, writer, native) -> None:
    """Dispatch locally or warn and retry native once with the original request.

    Args:
        request: Unmodified Model Connect build request.
        writer: Unpublished bundle writer.
        native: This family's original native builder callback.

    Raises:
        Exception: Common input/publication error, or native build error with
            Edge cause after a failed preparation. Cancellation never retries.
    """
    if not mapped_request(request) or not (Path(request.model_dir) / "config.json").is_file():
        native(request, writer)
        return
    raw = json.loads((Path(request.model_dir) / "config.json").read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("checkpoint config.json must contain an object")
    config = raw
    if not candidate(request, raw):
        native(request, writer)
        return
    capacity = config.get("max_position_embeddings")
    if type(capacity) is not int or capacity <= 0:
        raise ValueError("checkpoint max_position_embeddings must be a positive integer")
    if request.max_sequence_length and request.max_sequence_length > capacity:
        raise ValueError("max_sequence_length exceeds checkpoint context capacity")
    if not edge_llm.package_present():
        native(request, writer)
        return
    failure = None
    descriptor, name = tempfile.mkstemp(
        prefix=f".{request.output_path.name}.edge-", suffix=".log", dir=request.output_path.parent
    )
    os.close(descriptor)
    log_path = Path(name)
    with tempfile.TemporaryDirectory(prefix="trtmc-phi4_multimodal-edge-") as directory:
        try:
            target = edge_llm.local_target()
            weight_format = edge_llm.request_weight_format(request, raw)
            key = (target["os"], target["arch"], target["sm"], weight_format)
            adapter = EDGE_DISPATCH.get(key)
            if adapter is not None:
                files, marker = adapter(request, raw, target, Path(directory), log_path)
        except Exception as error:
            failure = error
            with log_path.open("a", encoding="utf-8") as log:
                traceback.print_exception(error, file=log)
            _LOG.warning(
                "phi4_multimodal Edge build failed: %s. Diagnostics: %s. "
                "Retrying native once with the unchanged request.",
                error,
                log_path,
                exc_info=True,
            )
        else:
            # Edge preparation did not touch writer; publication cannot fallback.
            if adapter is not None:
                edge_llm.publish(request, writer, files, marker)
                return
            log_path.unlink()  # A platform non-match is not an Edge failure.
    try:
        native(request, writer)
    except Exception as error:
        if failure is not None:
            raise error from failure
        raise
