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

# Only native platforms exercised by the recorded original-source profiles.
EDGE_DISPATCH = {("linux", "x86_64", sm, "fp16"): edge_llm.prepare for sm in (80, 120)}

# Pinned documented text topologies, not repository-name heuristics.
# type, layers, hidden, intermediate, attention heads, KV heads, vocabulary.
INTERNVL_CONFIGS = {
    ("qwen2", 24, 896, 4864, 14, 2, 151674),
    ("qwen2", 28, 1536, 8960, 12, 2, 151674),
    ("qwen2", 28, 3584, 18944, 28, 4, 151674),
    ("qwen2", 48, 5120, 13824, 40, 8, 151674),
}


def mapped_request(request) -> bool:
    """Preserve native-only controls before reading any checkpoint metadata."""
    return (
        request.backend == "trt"
        and request.task == "vision_language_generation"
        and request.precision.lower() == "fp16"
        and request.quantization in {None, "none"}
        and request.max_batch_size
        == request.tensor_parallel_size
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
    """Admit documented dense InternVL text/vision configurations and mapped controls."""
    config = raw.get("text_config", raw.get("llm_config", {}))
    vision = raw.get("vision_config", {})
    if not isinstance(config, dict) or not isinstance(vision, dict):
        return False
    keys = (
        "model_type",
        "num_hidden_layers",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
    )
    shape = tuple(config.get(key) for key in keys)
    return (
        raw.get("model_type") in {"internvl", "internvl_chat"}
        and not ("text_config" in raw and "llm_config" in raw)
        and all(type(value) is int for value in shape[1:])
        and shape in INTERNVL_CONFIGS
        and not any(
            config.get(key) for key in ("num_experts", "num_local_experts", "moe_intermediate_size")
        )
        and not any(
            raw.get(key)
            for key in ("audio_config", "draft_config", "eagle_config", "speculative_config")
        )
        and config.get("hidden_act", "silu") == "silu"
        and vision.get("image_size") in (448, [448, 448])
        and vision.get("patch_size") in (14, [14, 14])
        and vision.get("num_channels", 3) == 3
        and vision.get("model_type") in {"internvl_vision", "intern_vit_6b"}
        and tuple(
            vision.get(key)
            for key in (
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
            )
        )
        == (1024, 4096, 24, 16)
        and not vision.get("use_moe", False)
        and raw.get("downsample_ratio", 0.5) == 0.5
        and raw.get("image_seq_length", 256) == 256
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
    config = raw.get("text_config", raw.get("llm_config", raw))
    if not isinstance(config, dict):
        raise ValueError("checkpoint text_config must contain an object")
    if not candidate(request, raw):
        native(request, writer)
        return
    if not edge_llm.package_present():
        native(request, writer)
        return
    failure = None
    descriptor, name = tempfile.mkstemp(
        prefix=f".{request.output_path.name}.edge-", suffix=".log", dir=request.output_path.parent
    )
    os.close(descriptor)
    log_path = Path(name)
    with tempfile.TemporaryDirectory(
        prefix=f".{request.output_path.name}.edge-", dir=request.output_path.parent
    ) as directory:
        try:
            target = edge_llm.local_target()
            weight_format = edge_llm.request_weight_format(request, raw)
            key = (target["os"], target["arch"], target["sm"], weight_format)
            adapter = EDGE_DISPATCH.get(key)
            # The recorded 14B profile uses SM120; 1B/2B/8B use SM80.
            if target["sm"] != (120 if config["hidden_size"] == 5120 else 80):
                adapter = None
            if adapter is not None:
                # This is an Edge profile restriction, not a native build limit.
                capacity = config.get("max_position_embeddings")
                if type(capacity) is not int or capacity <= 0:
                    raise ValueError(
                        "checkpoint max_position_embeddings must be a positive integer"
                    )
                if request.max_sequence_length and request.max_sequence_length > capacity:
                    raise ValueError("max_sequence_length exceeds checkpoint context capacity")
                files, marker = adapter(request, raw, target, Path(directory), log_path)
        except Exception as error:
            failure = error
            with log_path.open("a", encoding="utf-8") as log:
                traceback.print_exception(error, file=log)
            _LOG.warning(
                "internvl Edge build failed: %s. Diagnostics: %s. "
                "Retrying native once with the unchanged request.",
                error,
                log_path,
                exc_info=True,
            )
        except BaseException:
            log_path.unlink(missing_ok=True)
            raise
        else:
            # Edge preparation did not touch writer; publication cannot fallback.
            if adapter is not None:
                edge_llm.publish(request, writer, files, marker)
                log_path.unlink()
                return
            log_path.unlink()  # A platform non-match is not an Edge failure.
    try:
        native(request, writer)
    except Exception as error:
        if failure is not None:
            raise error from failure
        raise
