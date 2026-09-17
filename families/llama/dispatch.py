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

from . import edge_llm

_LOG = logging.getLogger(__name__)

# Only the two native platforms exercised by the recorded FP16 profiles.
EDGE_DISPATCH = {("linux", "x86_64", sm, "fp16"): edge_llm.prepare for sm in (80, 120)}

# Documented Llama3.x dense shapes: layers, hidden, intermediate, heads, KV heads.
LLAMA3_CONFIGS = {
    (16, 2048, 8192, 32, 8),  # Llama3.2 1B
    (28, 3072, 8192, 24, 8),  # Llama3.2 3B
    (32, 4096, 14336, 32, 8),  # Llama3 / Llama3.1 8B
}


def candidate(request, raw: dict) -> bool:
    """Return whether the documented Llama3 config and request can use Edge."""
    shape = tuple(
        raw.get(name)
        for name in (
            "num_hidden_layers",
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
        )
    )
    return (
        raw.get("model_type") == "llama"
        and all(type(value) is int for value in shape)
        and shape in LLAMA3_CONFIGS
        and raw.get("vocab_size") == 128256
        and not raw.get("num_experts")
        and not raw.get("text_config")
        and raw.get("architectures", ["LlamaForCausalLM"]) == ["LlamaForCausalLM"]
        and raw.get("hidden_act", "silu") == "silu"
        and not raw.get("attention_bias", False)
        and not raw.get("mlp_bias", False)
        and request.backend == "trt"
        and request.task == "text_generation"
        and request.precision.lower() == "fp16"
        and request.quantization in {None, "none"}
        and request.max_batch_size
        == request.tensor_parallel_size
        == request.context_parallel_size
        == 1
        and not request.dynamic_kv_cache
        and not request.fp32_layers
        and request.graph_transform is None
        and all(
            value is None
            for value in (request.image_height, request.image_width, request.video_num_frames)
        )
    )


def build(request, writer, native, *, draft_dir: Path | None = None) -> None:
    """Dispatch locally or warn and retry native once with the original request.

    Args:
        request: Unmodified Model Connect build request.
        writer: Unpublished bundle writer.
        native: This family's original native builder callback.

    Raises:
        Exception: Common input/publication error, or native build error with
            Edge cause after a failed preparation. Cancellation never retries.
    """
    raw = json.loads((Path(request.model_dir) / "config.json").read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("checkpoint config.json must contain an object")
    config = raw.get("text_config", raw)
    if not isinstance(config, dict):
        raise ValueError("checkpoint text_config must contain an object")
    if not candidate(request, raw):
        native(request, writer)
        return
    capacity = config.get("max_position_embeddings")
    if type(capacity) is not int or capacity <= 0:
        raise ValueError("checkpoint max_position_embeddings must be a positive integer")
    if request.max_sequence_length and request.max_sequence_length > capacity:
        raise ValueError("max_sequence_length exceeds checkpoint context capacity")
    failure = None
    descriptor, name = tempfile.mkstemp(
        prefix=f".{request.output_path.name}.edge-", suffix=".log", dir=request.output_path.parent
    )
    os.close(descriptor)
    log_path = Path(name)
    with tempfile.TemporaryDirectory(prefix="trtmc-llama-edge-") as directory:
        try:
            target = edge_llm.local_target()
            weight_format = edge_llm.request_weight_format(request, raw)
            key = (target["os"], target["arch"], target["sm"], weight_format)
            adapter = EDGE_DISPATCH.get(key)
            # Ordinary profiles were qualified on SM80; the EAGLE pair on SM120.
            if target["sm"] != (120 if draft_dir is not None else 80):
                adapter = None
            if adapter is not None:
                if draft_dir is None:
                    files, marker = adapter(request, raw, target, Path(directory), log_path)
                else:
                    files, marker = adapter(
                        request, raw, target, Path(directory), log_path, draft_dir=draft_dir
                    )
        except Exception as error:
            failure = error
            with log_path.open("a", encoding="utf-8") as log:
                traceback.print_exception(error, file=log)
            _LOG.warning(
                "llama Edge build failed: %s. Diagnostics: %s. "
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
