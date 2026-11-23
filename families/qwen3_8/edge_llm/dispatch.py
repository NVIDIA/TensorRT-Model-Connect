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

# Retain the separate DSpark route; historical evidence does not qualify a new release.
EDGE_DISPATCH = {
    ("linux", "x86_64", 120, "fp16"): edge_llm.prepare_dspark,
}


DFLASH_EDGE_DISPATCH = {
    ("linux", "x86_64", 120, "fp16"): edge_llm.prepare_dflash,
}


MTP_EDGE_DISPATCH = {
    ("linux", "x86_64", 120, "fp16"): edge_llm.prepare_mtp,
}


ORDINARY_EDGE_DISPATCH = {
    ("linux", "x86_64", 80, "fp16", 2, "none"): edge_llm.prepare_tp2,
    ("linux", "x86_64", 120, "fp16", 1, "nvfp4"): edge_llm.prepare_nvfp4,
}


def ordinary_candidate(request, raw: dict, *, embedded_mtp: bool = False, current: bool = False) -> bool:
    """Admit dense Swish 27B FP16 TP2 or original mixed-NVFP4 TP1."""
    config = raw.get("text_config", raw)
    if not isinstance(config, dict):
        return False
    source = Path(request.model_dir)
    quantization = (edge_llm.prepared_checkpoint_quantization(source, raw) if embedded_mtp else
                    edge_llm.ordinary_quantization(source, raw, current=current))
    return (
        edge_llm.dense_swish_27b(raw)
        and ((quantization, request.tensor_parallel_size) in {("none", 2), ("nvfp4", 1)}
             or (current and not embedded_mtp and (quantization, request.tensor_parallel_size) == ("none", 1))
             or (embedded_mtp and (quantization, request.tensor_parallel_size) == ("int4_awq", 1)))
        and request.quantization in {None, quantization}
        and request.backend == "trt"
        and (request.task == "text_generation" or (
            request.task == "vision_language_generation" and isinstance(raw.get("vision_config"), dict)
        ))
        and request.precision.lower() == "fp16"
        and request.max_batch_size == request.context_parallel_size == 1
        and not request.dynamic_kv_cache and not request.fp32_layers and request.graph_transform is None
        and all(value is None for value in (request.image_height, request.image_width, request.video_num_frames))
    )


def request_matches(request, execution_variant: str = "dspark") -> bool:
    """Validate mapped request controls independently of checkpoint metadata."""
    quantizations = {None, "nvfp4"}
    if execution_variant in {"dflash", "dflash_tree"}:
        quantizations |= {"none", "int4_awq", "int4_gptq"}
    if execution_variant in {"domino", "domino_tree"}:
        quantizations = {None, "none"}
    return (
        request.backend == "trt" and request.task == "text_generation"
        and request.precision.lower() == "fp16"
        and request.quantization in quantizations
        and request.max_batch_size == request.tensor_parallel_size == request.context_parallel_size == 1
        and not request.dynamic_kv_cache and not request.fp32_layers and request.graph_transform is None
        and all(value is None for value in (request.image_height, request.image_width, request.video_num_frames))
    )


def candidate(request, raw: dict, execution_variant: str = "dspark", *, include_current: bool = False) -> bool:
    """Return whether this family's model/request contract can delegate to Edge."""
    config = raw.get("text_config", raw)
    source_quantization = edge_llm.paired_checkpoint_quantization(
        Path(request.model_dir), raw, execution_variant, include_current=include_current)
    return (
        isinstance(config, dict)
        and raw.get("model_type") == "qwen3_5"
        and ("output_gate_type" in config and "mlp_only_layers" not in config)
        and config.get("linear_key_head_dim") == config.get("linear_value_head_dim") == 128
        and not config.get("num_experts")
        and source_quantization is not None
        and request.quantization in {None, source_quantization}
        and request_matches(request, execution_variant)
    )


def build(request, writer, native, *, draft_dir: Path | None = None,
          execution_variant: str = "dspark") -> None:
    """Dispatch locally or warn and retry native once with the original request.

    Args:
        request: Unmodified Model Connect build request.
        writer: Unpublished bundle writer.
        native: This family's original native builder callback.

    Raises:
        Exception: Common input/publication error, or native build error with
            Edge cause after a failed preparation. Cancellation never retries.
    """
    # Edge does not expose the native family's exact chunk-size contract.
    # An explicit choice (including zero) must not be silently discarded.
    if getattr(request, "prefill_chunk_len", None) is not None:
        native(request, writer)
        return
    raw = json.loads((Path(request.model_dir) / "config.json").read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("checkpoint config.json must contain an object")
    config = raw.get("text_config", raw)
    if not isinstance(config, dict):
        raise ValueError("checkpoint text_config must contain an object")
    current = edge_llm.selected_release() == "0.12.0"
    admitted = (candidate(request, raw, execution_variant, include_current=current) if draft_dir is not None else
                ordinary_candidate(request, raw, embedded_mtp=execution_variant == "mtp", current=current))
    if (current and admitted and draft_dir is not None and execution_variant in {"dflash", "dflash_tree"}
            and edge_llm.implicit_packed_zero_points(Path(request.model_dir), raw)):
        _LOG.warning("Edge 0.12 cannot export this packed INT4 target with implicit zero points; "
                     "using the native DFlash2 route with the same checkpoint and companion")
        native(request, writer)
        return
    if not admitted:
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
            paired_routes = DFLASH_EDGE_DISPATCH if execution_variant == "dflash" else EDGE_DISPATCH
            routes = (MTP_EDGE_DISPATCH if execution_variant == "mtp" else
                      paired_routes if draft_dir is not None else ORDINARY_EDGE_DISPATCH)
            ordinary = draft_dir is None and execution_variant != "mtp"
            if ordinary:
                quant = edge_llm.ordinary_quantization(Path(request.model_dir), raw, current=current)
                key += (request.tensor_parallel_size, quant)
            if current and ordinary and target["os"] == "linux" and request.tensor_parallel_size == 1:
                # installed_package validates the actual target against the SDK's compiled architectures.
                adapter = {"none": edge_llm.prepare_fp16, "nvfp4": edge_llm.prepare_nvfp4}.get(quant)
            elif current and draft_dir is not None and target["os"] == "linux":
                adapter = {"dflash": edge_llm.prepare_dflash, "dflash_tree": edge_llm.prepare_dflash_tree,
                           "dspark": edge_llm.prepare_dspark,
                           "domino": edge_llm.prepare_domino,
                           "domino_tree": edge_llm.prepare_domino_tree}.get(execution_variant)
            else:
                adapter = routes.get(key)
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
                _LOG.info("Qwen3.8 Edge build diagnostics: %s", log_path)
                return
            log_path.unlink()  # A platform non-match is not an Edge failure.
    try:
        native(request, writer)
    except Exception as error:
        if failure is not None:
            raise error from failure
        raise


def build_paired(request, writer, execution) -> None:
    """Build the admitted paired checkpoint or embedded MTP request."""
    execution.validate_local()

    if execution.variant == "mtp":
        if execution.checkpoints:
            raise ValueError("Qwen3.6/3.8 MTP uses embedded draft layers, not a companion")
        raw = json.loads((request.model_dir / "config.json").read_text())
        base = raw.get("text_config", raw)
        if (not ordinary_candidate(request, raw, embedded_mtp=True)
                or request.tensor_parallel_size != 1 or base.get("mtp_num_hidden_layers") != 1):
            raise ValueError("MTP requires a uniform NVFP4 or INT4-AWQ dense 27B checkpoint with one embedded draft layer")
        limit = request.max_sequence_length or min(base["max_position_embeddings"], 256)
        if limit <= 4:
            raise ValueError("MTP capacity must exceed its four verification positions")
        if request.mtp_seq_len not in (None, 4):
            raise ValueError("The MTP variant requires four verification positions")

        def native_mtp(original_request, original_writer):
            from dataclasses import replace
            from ..model import _build_native

            # Preserve the source's quantized arithmetic even when the public
            # request inferred its format. Never replace MTP with base decoding.
            source_format = edge_llm.prepared_checkpoint_quantization(
                Path(original_request.model_dir), raw)
            native_request = replace(
                original_request, quantization=source_format, mtp_seq_len=4)
            _build_native(native_request, original_writer, require_mtp=True)

        build(request, writer, native_mtp, execution_variant="mtp")
        return
    if execution.variant not in {"dspark", "dflash", "dflash_tree", "domino", "domino_tree"} or tuple(x.role for x in execution.checkpoints) != ("draft",):
        raise ValueError("Qwen3.8 paired execution requires DSpark, DFlash2 or Domino and one draft checkpoint")
    if request.precision.lower() != "fp16":
        raise ValueError("Qwen3.8 paired execution requires --precision fp16; the native default is unchanged")
    if not request_matches(request, execution.variant):
        raise ValueError(
            "Qwen3.8 paired execution requires backend=trt, text_generation, batch/TP/CP=1, "
            "quantization unset or nvfp4 for mixed targets, none for original targets, "
            "or int4_awq/int4_gptq for matching DFlash2 targets, with no dynamic KV, FP32 layers, graph or media overrides"
        )
    draft_dir = execution.checkpoints[0].model_dir
    raw = json.loads((request.model_dir / "config.json").read_text())
    draft = json.loads((draft_dir / "config.json").read_text())
    if not candidate(request, raw, execution.variant, include_current=True):
        raise ValueError("Qwen3.8 requires a matching mixed-NVFP4 base, or an original 27B or matched prepared INT4 base for DFlash2")
    base = raw.get("text_config", raw)
    variant = execution.variant
    if variant in {"domino", "domino_tree"}:
        from ..domino import DominoConfig
        DominoConfig.from_dict(draft, base)
    else:
        architecture = "DSparkDraftModel" if variant == "dspark" else "DFlash2DraftModel"
        if not isinstance(draft, dict) or draft.get("architectures") != [architecture]:
            raise ValueError(f"Expected a {architecture} companion")
        for name in ("hidden_size", "vocab_size"):
            if type(draft.get(name)) is not int or draft[name] != base.get(name):
                raise ValueError(f"Qwen3.8 paired execution base and draft disagree on {name}")
        if draft.get("num_target_layers") != base.get("num_hidden_layers"):
            raise ValueError("Qwen3.8 paired execution target layer count differs from base")
        config = draft.get("dspark_config" if variant == "dspark" else "dflash_config")
        block = 7 if variant == "dspark" else 8
        if not isinstance(config, dict) or config.get("block_size", draft.get("block_size")) != block:
            raise ValueError(f"Qwen3.8 {variant} requires its documented block{block} profile")
        layers = config.get("target_layer_ids", draft.get("target_layer_ids"))
        if (not isinstance(layers, list) or not layers
                or any(type(i) is not int or not 0 <= i < base["num_hidden_layers"] for i in layers)
                or len(set(layers)) != len(layers)):
            raise ValueError("Invalid Qwen3.8 paired execution target layer IDs")
        mask = config.get("mask_token_id", draft.get("mask_token_id"))
        if type(mask) is not int or not 0 <= mask < base["vocab_size"]:
            raise ValueError("Invalid Qwen3.8 paired execution mask token")
    limit = request.max_sequence_length or min(base["max_position_embeddings"], 256)
    maximum = {"dspark": 1024, "dflash": 4096, "dflash_tree": 4096, "domino": 2048, "domino_tree": 2048}[variant]
    minimum = 16 if variant in {"domino", "domino_tree"} else 9
    if not minimum <= limit <= maximum:
        raise ValueError(f"Qwen3.8 {variant} requires max_sequence_length above {minimum - 1} and at most {maximum}")
    capacity = draft.get("max_position_embeddings")
    if type(capacity) is not int or not 8 < limit <= capacity:
        raise ValueError("Requested context exceeds companion draft capacity or block minimum")
    if draft.get("quantization_config") or any(
        (draft_dir / name).exists()
        for name in ("hf_quant_config.json", "quantize_config.json", "quant_config.json")
    ):
        raise ValueError("This Qwen3.8 paired execution profile requires unquantized draft weights")

    def native_pair(original_request, original_writer):
        from dataclasses import replace
        from ..model import _build_native

        # The companion was validated against this target. Preserve both the
        # requested decoding algorithm and the actual source arithmetic.
        source_format = edge_llm.paired_checkpoint_quantization(
            Path(original_request.model_dir), raw, variant, include_current=True)
        if source_format is None:
            raise ValueError("Qwen3.8 paired target format no longer matches its build request")
        options = {"draft_dir": draft_dir}
        quant = raw.get("quantization_config") or {}
        scheme = quant.get("kv_cache_scheme") or {}
        # The 0.12 ONNX grouped-NVFP4 reader honors declared FP8 KV even
        # without per-attention scales (unlike its direct builder and 0.11).
        if (source_format == "nvfp4" and edge_llm.selected_release() != "0.11.0"
                and quant.get("config_groups") and isinstance(scheme, dict)
                and scheme.get("type") == "float" and scheme.get("num_bits") == 8):
            options["paired_fp8_kv"] = True
        if variant != "dspark":
            options["draft_variant"] = variant
        _build_native(replace(original_request, quantization=None if source_format == "none" else source_format),
                      original_writer, **options)

    if variant == "dspark":
        build(request, writer, native_pair, draft_dir=draft_dir)
    else:
        build(request, writer, native_pair, draft_dir=draft_dir, execution_variant=variant)
