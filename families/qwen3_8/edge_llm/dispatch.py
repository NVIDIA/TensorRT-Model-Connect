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

# Exact qualified native DSpark route; no unvalidated platform or ordinary offload.
EDGE_DISPATCH = {
    ("linux", "x86_64", 120, "fp16"): edge_llm.prepare_dspark,
}


def request_matches(request) -> bool:
    """Validate mapped request controls independently of checkpoint metadata."""
    return (
        request.backend == "trt" and request.task == "text_generation"
        and request.precision.lower() == "fp16"
        and request.quantization in {None, "nvfp4"}
        and request.max_batch_size == request.tensor_parallel_size == request.context_parallel_size == 1
        and not request.dynamic_kv_cache and not request.fp32_layers and request.graph_transform is None
        and all(value is None for value in (request.image_height, request.image_width, request.video_num_frames))
    )


def candidate(request, raw: dict) -> bool:
    """Return whether this family's model/request contract can delegate to Edge."""
    config = raw.get("text_config", raw)
    source_quantization = edge_llm.checkpoint_quantization(Path(request.model_dir), raw)
    return (
        isinstance(config, dict)
        and raw.get("model_type") == "qwen3_5"
        and ("output_gate_type" in config and "mlp_only_layers" not in config)
        and config.get("linear_key_head_dim") == config.get("linear_value_head_dim") == 128
        and not config.get("num_experts")
        and source_quantization == "nvfp4"
        and request_matches(request)
    )


def build(request, writer, native, *, draft_dir: Path) -> None:
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
    with tempfile.TemporaryDirectory(
        prefix=f".{request.output_path.name}.edge-", dir=request.output_path.parent
    ) as directory:
        try:
            target = edge_llm.local_target()
            key = (target["os"], target["arch"], target["sm"], request.precision.lower())
            adapter = EDGE_DISPATCH.get(key)
            if adapter is not None:
                files, marker = adapter(request, raw, target, Path(directory), log_path, draft_dir)
        except Exception as error:
            failure = error
            with log_path.open("a", encoding="utf-8") as log:
                traceback.print_exception(error, file=log)
            _LOG.warning("qwen3_8 Edge build failed: %s. Diagnostics: %s. "
                         "Retrying native once with the unchanged request.", error, log_path, exc_info=True)
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


def build_paired(request, writer, execution) -> None:
    """Build the retained mixed-NVFP4 Qwen3.8 / DSpark block7 pair."""
    execution.validate_local()

    if execution.variant != "dspark" or tuple(x.role for x in execution.checkpoints) != ("draft",):
        raise ValueError("Qwen3.8 paired execution requires variant=dspark and one draft checkpoint")
    if request.precision.lower() != "fp16":
        raise ValueError("Qwen3.8 DSpark requires --precision fp16; the native default is unchanged")
    if not request_matches(request):
        raise ValueError(
            "Qwen3.8 DSpark requires backend=trt, text_generation, batch/TP/CP=1, "
            "quantization unset or nvfp4, and no dynamic KV, FP32 layers, graph or media overrides"
        )
    draft_dir = execution.checkpoints[0].model_dir
    raw = json.loads((request.model_dir / "config.json").read_text())
    draft = json.loads((draft_dir / "config.json").read_text())
    if not candidate(request, raw):
        raise ValueError("The retained Qwen3.8 DSpark pair requires a matching mixed-NVFP4 base")
    base = raw.get("text_config", raw)
    if not isinstance(draft, dict) or draft.get("architectures") != ["DSparkDraftModel"]:
        raise ValueError("Expected a DSparkDraftModel companion")
    for name in ("hidden_size", "vocab_size"):
        if type(draft.get(name)) is not int or draft[name] != base.get(name):
            raise ValueError(f"Qwen3.8 DSpark base and draft disagree on {name}")
    if draft.get("num_target_layers") != base.get("num_hidden_layers"):
        raise ValueError("Qwen3.8 DSpark target layer count differs from base")
    config = draft.get("dspark_config")
    if not isinstance(config, dict) or config.get("block_size", draft.get("block_size")) != 7:
        raise ValueError("Qwen3.8 DSpark maps the upstream block7 / verify8 profile")
    layers = config.get("target_layer_ids", draft.get("target_layer_ids"))
    if (not isinstance(layers, list) or not layers
            or any(type(i) is not int or not 0 <= i < base["num_hidden_layers"] for i in layers)
            or len(set(layers)) != len(layers)):
        raise ValueError("Invalid Qwen3.8 DSpark target layer IDs")
    mask = config.get("mask_token_id", draft.get("mask_token_id"))
    if type(mask) is not int or not 0 <= mask < base["vocab_size"]:
        raise ValueError("Invalid Qwen3.8 DSpark mask token")
    limit = request.max_sequence_length or min(base["max_position_embeddings"], 256)
    if not 8 < limit <= 1024:
        raise ValueError("Qwen3.8 DSpark requires max_sequence_length above 8 and at most 1024")
    capacity = draft.get("max_position_embeddings")
    if type(capacity) is not int or not 8 < limit <= capacity:
        raise ValueError("Requested context exceeds DSpark draft capacity or block minimum")
    if draft.get("quantization_config") or any(
        (draft_dir / name).exists()
        for name in ("hf_quant_config.json", "quantize_config.json", "quant_config.json")
    ):
        raise ValueError("This Qwen3.8 DSpark profile requires unquantized draft weights")

    def native_pair(original_request, original_writer):
        # A failure must never replace the requested pair with base-only decoding.
        raise NotImplementedError(
            "Native Qwen3.8 does not implement the requested DSpark variant; "
            "the qualified Edge route requires Linux x86_64, SM120 and FP16"
        )

    build(request, writer, native_pair, draft_dir=draft_dir)
