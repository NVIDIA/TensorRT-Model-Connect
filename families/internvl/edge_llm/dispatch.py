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

# Native SDK identity is checked on each complete-network route.
EDGE_DISPATCH = {
    ("linux", "x86_64", sm, weight_format): edge_llm.prepare
    for sm in (80, 86, 120)
    for weight_format in ("fp16", "int4_awq")
}

# Pinned documented text topologies, not repository-name heuristics.
# type, layers, hidden, intermediate, attention heads, KV heads, vocabulary.
INTERNVL_CONFIGS = {
    ("qwen2", 24, 896, 4864, 14, 2, 151674),
    ("qwen2", 28, 1536, 8960, 12, 2, 151674),
    ("qwen2", 28, 3584, 18944, 28, 4, 151674),
    ("qwen2", 48, 5120, 13824, 40, 8, 151674),
    # InternVL3.5 uses its explicit head dimension, not a hidden/heads inference.
    ("qwen3", 28, 1024, 3072, 16, 8, 151936),
    ("qwen3", 28, 2048, 6144, 16, 8, 151936),
    ("qwen3", 36, 2560, 9728, 32, 8, 151936),
    ("qwen3", 36, 4096, 12288, 32, 8, 151936),
    ("qwen3", 40, 5120, 17408, 40, 8, 151936),
}


def mapped_request(request) -> bool:
    """Preserve native-only controls before reading any checkpoint metadata."""
    return (
        request.backend == "trt"
        and request.task == "vision_language_generation"
        and request.precision.lower() == "fp16"
        and request.quantization in {None, "none", "int4_awq"}
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
        and (config.get("model_type") != "qwen3" or config.get("head_dim") == 128)
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


UnavailableEdgeConfiguration = edge_llm.UnavailableEdgeConfiguration


def device_memory_bytes() -> int:
    """Read native device capacity for this family's complete-network offload."""
    from cuda.bindings import runtime

    status, device = runtime.cudaGetDevice()
    if int(status) != 0:
        raise RuntimeError(f"InternVL CUDA device query failed: {status}")
    status, properties = runtime.cudaGetDeviceProperties(device)
    if int(status) != 0:
        raise RuntimeError(f"InternVL CUDA capacity query failed: {status}")
    return properties.totalGlobalMem


def validate_execution_capacity(request, config: dict, target: dict, weight_format: str) -> None:
    """Reject the recorded 3.5-14B build OOM without rejecting larger GPUs.

    The pinned ONNX builder requires this single allocation for the recorded
    FP16/context384 profile. It is only a necessary lower bound, not a promise
    that the remaining build allocations fit. Quality-only failures are not
    part of execution admission.
    """
    minimum = {80: 49_913_047_296, 86: 52_849_060_096}.get(target.get("sm"))
    if (minimum is not None and target.get("tensorrt_version") == "11.1.0.106"
            and weight_format == "fp16"
            and config.get("model_type") == "qwen3"
            and config.get("num_hidden_layers") == 40
            and config.get("hidden_size") == 5120
            and request.max_sequence_length == 384
            and device_memory_bytes() < minimum):
        raise UnavailableEdgeConfiguration(
            "InternVL3.5-14B Edge 0.11 FP16/context384 is unavailable on this GPU: "
            f"the recorded ONNX build requires a {minimum:,}-byte allocation. "
            "The recorded 40 GiB SM80 and 48 GiB SM86 builds both exhausted memory."
        )


def validate_native_capacity(request, raw: dict) -> None:
    """Avoid the recorded native OOM even when the optional Edge SDK is absent."""
    config = raw.get("text_config", raw.get("llm_config", raw))
    if not (
        candidate(request, raw) and config.get("model_type") == "qwen3"
        and config.get("num_hidden_layers") == 40 and config.get("hidden_size") == 5120
        and request.max_sequence_length == 384
        and edge_llm.request_weight_format(request, raw) == "fp16"
    ):
        return
    target = edge_llm.local_target()
    # TP1's native factory retains independent prefill and decode engines.
    # The recorded decode plan is 29.54 GB and the next weight allocation is
    # 29.54 GB: neither available 40/48 GiB device can hold this pair. This is
    # an admission estimate, not an observed native SM86 execution result.
    if (target.get("tensorrt_version") == "11.1.0.106"
            and target.get("sm") in {80, 86}
            and device_memory_bytes() <= 48 * 1024 ** 3):
        raise UnavailableEdgeConfiguration(
            "InternVL3.5-14B native FP16/context384 is unavailable on this GPU: "
            "the recorded runtime exhausted memory while loading two resident "
            "roughly 29.5 GB decoder plans. A 40/48 GiB device cannot hold this pair."
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
        if getattr(request, "int4_gemm_plugin_version", None) is not None:
            raise UnavailableEdgeConfiguration("INT4 plugin selection requires a mapped Edge AWQ request")
        native(request, writer)
        return
    raw = json.loads((Path(request.model_dir) / "config.json").read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("checkpoint config.json must contain an object")
    config = raw.get("text_config", raw.get("llm_config", raw))
    if not isinstance(config, dict):
        raise ValueError("checkpoint text_config must contain an object")
    def run_native():
        validate_native_capacity(request, raw)
        native(request, writer)

    if not candidate(request, raw):
        if getattr(request, "int4_gemm_plugin_version", None) is not None:
            raise UnavailableEdgeConfiguration("INT4 plugin selection requires a mapped Edge AWQ request")
        run_native()
        return
    if not edge_llm.package_present():
        if getattr(request, "int4_gemm_plugin_version", None) is not None:
            raise UnavailableEdgeConfiguration("INT4 plugin selection requires an Edge SDK")
        run_native()
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
            edge_llm.int4_plugin_version(request, config, target, weight_format)
            key = (target["os"], target["arch"], target["sm"], weight_format)
            adapter = EDGE_DISPATCH.get(key)
            # Original Qwen2-backed 14B uses its recorded SM120 profile.
            # Qwen3-backed InternVL3.5 uses the native SM80 ONNX profile.
            profile_sm = (
                120 if config.get("model_type") == "qwen2" and config["hidden_size"] == 5120 else 80
            )
            larger_memory_retry = (
                config.get("model_type") == "qwen3"
                and config.get("num_hidden_layers") == 40
                and config.get("hidden_size") == 5120
                and target["sm"] == 86
                and weight_format == "fp16"
            )
            awq_sm120_control = (
                weight_format == "int4_awq" and target["sm"] == 120
                and config.get("model_type") == "qwen2"
                and config.get("hidden_size") == 1536
                and config.get("num_hidden_layers") == 28
            )
            if target["sm"] != profile_sm and not (larger_memory_retry or awq_sm120_control):
                adapter = None
            if adapter is None and getattr(request, "int4_gemm_plugin_version", None) is not None:
                raise UnavailableEdgeConfiguration("No Edge INT4 route is mapped for this target")
            if adapter is not None:
                validate_execution_capacity(request, config, target, weight_format)
                # This is an Edge profile restriction, not a native build limit.
                capacity = config.get("max_position_embeddings")
                if type(capacity) is not int or capacity <= 0:
                    raise ValueError(
                        "checkpoint max_position_embeddings must be a positive integer"
                    )
                if request.max_sequence_length and request.max_sequence_length > capacity:
                    raise ValueError("max_sequence_length exceeds checkpoint context capacity")
                files, marker = adapter(request, raw, target, Path(directory), log_path)
        except UnavailableEdgeConfiguration:
            # Do not silently rerun a recorded failing configuration via fallback.
            log_path.unlink(missing_ok=True)
            raise
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
        run_native()
    except Exception as error:
        if failure is not None:
            raise error from failure
        raise
