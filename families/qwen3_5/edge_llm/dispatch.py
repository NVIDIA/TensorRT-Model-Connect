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

# Only the native platform exercised by the recorded original-source profiles.
EDGE_DISPATCH = {("linux", "x86_64", 80, "fp16"): edge_llm.prepare}

# Layers, hidden, intermediate, attention heads, KV heads, vocabulary.
# These configurations are verified from the retained per-model engine metadata.
DENSE_CONFIGS = {
    (24, 1024, 3584, 8, 2, 248320),
    (24, 2048, 6144, 8, 2, 248320),
    (32, 2560, 9216, 16, 4, 248320),
    (32, 4096, 12288, 16, 4, 248320),
}


def candidate(request, raw: dict) -> bool:
    """Return whether this family's model/request contract can delegate to Edge."""
    config = raw.get("text_config", raw)
    return (
        isinstance(config, dict)
        and raw.get("model_type") == "qwen3_5"
        and tuple(
            config.get(key)
            for key in (
                "num_hidden_layers",
                "hidden_size",
                "intermediate_size",
                "num_attention_heads",
                "num_key_value_heads",
                "vocab_size",
            )
        )
        in DENSE_CONFIGS
        and ("output_gate_type" not in config or "mlp_only_layers" in config)
        and config.get("linear_key_head_dim") == config.get("linear_value_head_dim") == 128
        and not config.get("num_experts")
        and not raw.get("quantization_config")
        and not config.get("quantization_config")
        and not any(
            (Path(request.model_dir) / name).exists()
            for name in ("hf_quant_config.json", "quantize_config.json", "quant_config.json")
        )
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
    if not (Path(request.model_dir) / "config.json").is_file():
        native(request, writer)
        return
    raw = json.loads((Path(request.model_dir) / "config.json").read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("checkpoint config.json must contain an object")
    config = raw.get("text_config", raw)
    if not isinstance(config, dict):
        raise ValueError("checkpoint text_config must contain an object")
    if not candidate(request, raw):
        native(request, writer)
        return
    if draft_dir is None and not edge_llm.package_present():
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
    with tempfile.TemporaryDirectory(
        prefix=f".{request.output_path.name}.edge-", dir=request.output_path.parent
    ) as directory:
        try:
            target = edge_llm.local_target()
            key = (target["os"], target["arch"], target["sm"], request.precision.lower())
            adapter = EDGE_DISPATCH.get(key)
            if adapter is not None:
                options = {} if draft_dir is None else {"draft_dir": draft_dir}
                files, marker = adapter(request, raw, target, Path(directory), log_path, **options)
        except Exception as error:
            failure = error
            with log_path.open("a", encoding="utf-8") as log:
                traceback.print_exception(error, file=log)
            _LOG.warning(
                "qwen3_5 Edge build failed: %s. Diagnostics: %s. "
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


def build_paired(request, writer, execution) -> None:
    """Build an explicit Qwen3.5 DFlash pair, never a base-only replacement."""
    execution.validate_local()

    if execution.variant != "dflash" or tuple(x.role for x in execution.checkpoints) != ("draft",):
        raise ValueError(
            "Qwen3.5 paired execution requires variant=dflash and one draft checkpoint"
        )
    draft_dir = execution.checkpoints[0].model_dir
    raw = json.loads((request.model_dir / "config.json").read_text())
    draft = json.loads((draft_dir / "config.json").read_text())
    if not candidate(request, raw):
        raise ValueError("Qwen3.5 DFlash requires an admitted dense base request")
    base = raw.get("text_config", raw)
    if (base.get("hidden_size"), base.get("num_hidden_layers")) not in {(2560, 32), (4096, 32)}:
        raise ValueError("Qwen3.5 DFlash is qualified only for the recorded 4B/9B base profiles")
    if not isinstance(draft, dict) or draft.get("architectures") != ["DFlashDraftModel"]:
        raise ValueError("Expected a DFlashDraftModel companion")
    for name in ("hidden_size", "vocab_size"):
        if type(draft.get(name)) is not int or draft[name] != base.get(name):
            raise ValueError(f"Qwen3.5 DFlash base and draft disagree on {name}")
    if draft.get("num_target_layers") != base.get("num_hidden_layers"):
        raise ValueError("Qwen3.5 DFlash target layer count differs from base")
    config = draft.get("dflash_config")
    if not isinstance(config, dict) or config.get("block_size") != 16:
        raise ValueError("Qwen3.5 DFlash currently maps the upstream linear block16 profile")
    layers = config.get("target_layer_ids")
    if (
        not isinstance(layers, list)
        or not layers
        or any(type(i) is not int or not 0 <= i < base["num_hidden_layers"] for i in layers)
        or len(set(layers)) != len(layers)
    ):
        raise ValueError("Invalid Qwen3.5 DFlash target layer IDs")
    mask = config.get("mask_token_id")
    if type(mask) is not int or not 0 <= mask < base["vocab_size"]:
        raise ValueError("Invalid Qwen3.5 DFlash mask token")
    capacity = draft.get("max_position_embeddings")
    limit = request.max_sequence_length or min(base["max_position_embeddings"], 256)
    if type(capacity) is not int or not 16 < limit <= capacity:
        raise ValueError("Requested context exceeds DFlash draft capacity or block minimum")
    if draft.get("quantization_config") or any(
        (draft_dir / name).exists()
        for name in ("hf_quant_config.json", "quantize_config.json", "quant_config.json")
    ):
        raise ValueError("This Qwen3.5 DFlash profile requires unquantized draft weights")

    def native_pair(original_request, original_writer):
        # Preserve the requested variant on fallback; native has no DFlash decoder.
        raise NotImplementedError("Native Qwen3.5 does not implement the requested DFlash variant")

    build(request, writer, native_pair, draft_dir=draft_dir)
