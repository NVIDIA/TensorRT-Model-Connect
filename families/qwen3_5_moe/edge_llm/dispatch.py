# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned complete-network offload with a native TensorRT fallback."""

from __future__ import annotations
import logging
import os
from pathlib import Path
import tempfile
import traceback
from ..draft_policy import block_policy
from . import builder as edge_llm, precision

_LOG = logging.getLogger(__name__)
EDGE_DISPATCH = {
    ("linux", arch, sm, weights): edge_llm.prepare
    for arch, sms in (("x86_64", (80, 86, 100, 120)), ("aarch64", (87, 110, 121)))
    for sm in sms
    for weights in ("int4", "nvfp4")
    if weights != "nvfp4" or sm in (100, 110, 120, 121)
}


def validate_request(request) -> None:
    """Reject every unsupported control rather than silently modifying new-family policy."""
    if request.family != "qwen3_5_moe" or request.task not in {
        "text_generation",
        "vision_language_generation",
    }:
        raise ValueError("Unsupported Qwen3.5-MoE task/family")
    if request.backend != "trt" or request.precision.lower() != "fp16":
        raise ValueError("Qwen3.5-MoE Edge requires TensorRT FP16 compute")
    if (
        request.max_batch_size != 1
        or request.tensor_parallel_size != 1
        or request.context_parallel_size != 1
    ):
        raise ValueError("Qwen3.5-MoE requires batch/TP/CP1")
    if (
        request.dynamic_kv_cache
        or request.fp32_layers
        or getattr(request, "mtp_seq_len", None) is not None
        or getattr(request, "graph_transform", None) is not None
    ):
        raise ValueError("Unmapped dynamic KV, MTP, FP32 layer or graph callback control")
    if any(
        value is not None
        for value in (request.image_height, request.image_width, request.video_num_frames)
    ):
        raise ValueError(
            "Qwen3.5-MoE owns image preprocessing; spatial/video overrides are unsupported"
        )
    if request.max_sequence_length is not None and (
        type(request.max_sequence_length) is not int or request.max_sequence_length <= 1
    ):
        raise ValueError("Context capacity must be an integer greater than 1")


def validate_execution(request, raw: dict) -> Path | None:
    """Admit documented paired drafts or embedded NVFP4 MTP; never fetch a draft."""
    execution = getattr(request, "execution", None)
    if execution is None:
        return None
    execution.validate_local()
    if execution.variant == "mtp":
        if execution.checkpoints:
            raise ValueError("Qwen3.6-MoE MTP uses embedded weights, not a companion")
        if precision.source_policy(request, raw)["llm"] != "mixed_nvfp4_experts_shared_head_fp8_attention_gdn":
            raise ValueError("The documented MoE MTP recipe requires NVFP4")
        if raw["text_config"].get("mtp_num_hidden_layers") != 1:
            raise ValueError("Qwen3.6-MoE MTP requires one embedded prediction layer")
        if (request.max_sequence_length or 4096) <= 4:
            raise ValueError("Qwen3.6-MoE MTP context must exceed the verify size")
        return None
    if execution.variant not in {"dflash", "dflash_ddtree", "jetspec"} or tuple(
        checkpoint.role for checkpoint in execution.checkpoints
    ) != ("draft",):
        raise ValueError("Qwen3.5-MoE DFlash requires exactly one explicit draft checkpoint")
    if execution.variant == "jetspec":
        policy = precision.source_policy(request, raw)
        if policy["weights"] != "none" and policy["llm"] != "mixed_nvfp4_experts_shared_head_fp8_attention_gdn":
            raise ValueError("Qwen3.5-MoE JetSpec requires an original or Qwen3.6 mixed NVFP4 target")
    draft_dir = execution.checkpoints[0].model_dir
    draft = precision.object_file(draft_dir / "config.json")
    base = raw["text_config"]
    if draft.get("architectures") != ["DFlashDraftModel"]:
        raise ValueError("Qwen3.5-MoE requires its matched DFlashDraftModel")
    for name in ("hidden_size", "vocab_size"):
        if type(draft.get(name)) is not int or draft[name] != base[name]:
            raise ValueError(f"Qwen3.5-MoE DFlash base/draft mismatch: {name}")
    if draft.get("num_target_layers") != base["num_hidden_layers"]:
        raise ValueError("Qwen3.5-MoE DFlash target layer count differs")
    config = block_policy(draft, execution.variant)
    layers = config.get("target_layer_ids")
    if (not isinstance(layers, list) or not layers or len(set(layers)) != len(layers)
        or any(type(layer) is not int or not 0 <= layer < base["num_hidden_layers"] for layer in layers)):
        raise ValueError("Invalid Qwen3.5-MoE DFlash target layer IDs")
    mask = config.get("mask_token_id")
    if type(mask) is not int or not 0 <= mask < base["vocab_size"]:
        raise ValueError("Invalid Qwen3.5-MoE DFlash mask token")
    limit = request.max_sequence_length or min(base["max_position_embeddings"], 4096)
    capacity = draft.get("max_position_embeddings")
    if type(capacity) is not int or not 16 < limit <= capacity:
        raise ValueError("Qwen3.5-MoE DFlash context exceeds the draft capacity")
    # Both listed pairs use an original BF16 draft, including the NVFP4 target.
    if draft.get("quantization_config") or (draft_dir / "hf_quant_config.json").exists():
        raise ValueError("Qwen3.5-MoE DFlash currently requires an original draft checkpoint")
    return draft_dir


def build(request, writer, native_build) -> None:
    """Prepare Edge assets or build the same request through the owning native graph.

    Publication and cancellation errors are outside preparation fallback handling.
    """
    validate_request(request)
    if request.verbose:
        _LOG.info("Selecting family-owned Qwen3.5-MoE Edge builder for %s", request.task)
    raw = precision.object_file(Path(request.model_dir) / "config.json")
    precision.validate_config(raw)
    policy = precision.source_policy(request, raw)
    validate_execution(request, raw)
    if (
        request.max_sequence_length
        and request.max_sequence_length > raw["text_config"]["max_position_embeddings"]
    ):
        raise ValueError("Requested context exceeds source capacity")
    original = policy["weights"] == "none"
    routed_only = policy["llm"] == "nvfp4_routed_experts_original_shared_head_attention"
    options = {}
    if original or routed_only:
        from .checkpoint import validate_checkpoint

        # An absent quantizer in config.json alone cannot prove unquantized weights.
        # Verify before native fallback as well, and reuse this result during offload.
        options["verified_layout"] = validate_checkpoint(Path(request.model_dir), policy["weights"], allow_snapshot_blobs=True)
    target = edge_llm.local_target()
    key = (target["os"], target["arch"], target["sm"], policy["weights"])
    adapter = EDGE_DISPATCH.get(key)
    if routed_only:
        adapter = None
        if edge_llm.selected_release() == "0.12.0" and (
                key in EDGE_DISPATCH or key == ("linux", "aarch64", 103, "nvfp4")):
            adapter = edge_llm.prepare
    # New original-weight profiles must not opt into the legacy SDK.
    original_target = original and (
        key[:3] + ("int4",) in EDGE_DISPATCH or key[:3] == ("linux", "aarch64", 103)
    )
    if (adapter is None and (original_target or key in {("linux", "aarch64", 103, "int4"),
                                                      ("linux", "aarch64", 103, "nvfp4")})
            and edge_llm.selected_release() == "0.12.0"):
        # These routes need the 0.12 SDK; prepare verifies the installed
        # architectures and exact native CUDA/TensorRT tuple before using it.
        adapter = edge_llm.prepare
    execution = getattr(request, "execution", None)
    if execution is not None and execution.variant == "jetspec" and edge_llm.selected_release() != "0.12.0":
        adapter = None  # JetSpec is a 0.12 Edge ABI; keep the requested native draft.
    if adapter is None:
        native_build(request, writer)
        return
    descriptor, name = tempfile.mkstemp(
        prefix=f".{request.output_path.name}.edge-", suffix=".log", dir=request.output_path.parent
    )
    os.close(descriptor)
    log = Path(name)
    edge_error = None
    with tempfile.TemporaryDirectory(
        prefix=f".{request.output_path.name}.edge-stage-", dir=request.output_path.parent
    ) as directory:
        try:
            files, marker = adapter(request, raw, target, Path(directory), log, **options)
        except Exception as error:
            with log.open("a", encoding="utf-8") as output:
                traceback.print_exception(error, file=output)
            _LOG.warning(
                "Qwen3.5-MoE Edge preparation failed: %s. Diagnostics: %s. Falling back to native TensorRT.",
                error,
                log,
                exc_info=True,
            )
            edge_error = error
        else:
            edge_llm.publish(request, writer, files, marker)
            log.unlink(missing_ok=True)
            return
    # Release staged checkpoint copies before native graph construction starts.
    try:
        native_build(request, writer)
    except Exception as native_error:
        raise native_error from edge_error
