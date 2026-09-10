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

# Explicit native platform routes supported by pinned Edge. No version-name
# comparison, nearest-GPU matching, shared registry, or legacy-Qwen entries.
EDGE_DISPATCH = {
    ("linux", architecture, sm, "fp16"): edge_llm.prepare
    for architecture, sms in (
        ("x86_64", (80, 86, 100, 120)),
        ("aarch64", (87, 110, 121)),
    )
    for sm in sms
}


def candidate(request, raw: dict) -> bool:
    """Return whether this family's model/request contract can delegate to Edge."""
    config = raw.get("text_config", raw)
    return (
        isinstance(config, dict)
        and raw.get("model_type") == "qwen3_5"
        and ("output_gate_type" not in config or "mlp_only_layers" in config)
        and config.get("linear_key_head_dim") == config.get("linear_value_head_dim") == 128
        and not config.get("num_experts")
        and not raw.get("quantization_config") and not config.get("quantization_config")
        and not any(
            (Path(request.model_dir) / name).exists()
            for name in ("hf_quant_config.json", "quantize_config.json", "quant_config.json")
        )
        and request.backend == "trt" and request.task == "text_generation"
        and request.precision.lower() == "fp16"
        and request.quantization in {None, "none"}
        and request.max_batch_size == request.tensor_parallel_size == request.context_parallel_size == 1
        and not request.dynamic_kv_cache and not request.fp32_layers and request.graph_transform is None
        and all(value is None for value in (request.image_height, request.image_width, request.video_num_frames))
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
    descriptor, name = tempfile.mkstemp(prefix=f".{request.output_path.name}.edge-", suffix=".log",
                                        dir=request.output_path.parent)
    os.close(descriptor)
    log_path = Path(name)
    with tempfile.TemporaryDirectory(prefix="trtmc-qwen3_5-edge-") as directory:
        try:
            target = edge_llm.local_target()
            key = (target["os"], target["arch"], target["sm"], request.precision.lower())
            adapter = EDGE_DISPATCH.get(key)
            if adapter is not None:
                files, marker = adapter(request, raw, target, Path(directory), log_path)
        except Exception as error:
            failure = error
            with log_path.open("a", encoding="utf-8") as log:
                traceback.print_exception(error, file=log)
            _LOG.warning("qwen3_5 Edge build failed: %s. Diagnostics: %s. "
                         "Retrying native once with the unchanged request.", error, log_path, exc_info=True)
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
