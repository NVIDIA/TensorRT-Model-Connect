# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned Qwen3.8 checkpoint-to-bundle build."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .config import ModelConfig
from .engine_builder import Qwen38Model, _owned
from .weight_spill import host_weight_bytes, spill_if_large


_BUNDLE_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.model",
)


_DEFAULT_PREFILL_SEQ_LEN = 128


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _native_quantization(requested, model_dir: Path, raw: dict):
    """Keep prepared ModelOpt arithmetic when the caller leaves it inferred."""
    if requested is not None:
        return requested
    metadata = raw.get("quantization_config") or raw.get("text_config", {}).get("quantization_config")
    sidecar = model_dir / "hf_quant_config.json"
    prepared = {}
    if sidecar.is_file():
        prepared = json.loads(sidecar.read_text(encoding="utf-8")).get("quantization", {})
    if not metadata and not prepared:
        return None
    metadata = metadata or {}
    if metadata.get("quant_method") not in (None, "modelopt"):
        return None
    algorithm = prepared.get("quant_algo", metadata.get("quant_algo"))
    if algorithm == "NVFP4":
        return "nvfp4"
    if algorithm == "W4A16_AWQ":
        return "int4_awq"
    groups = metadata.get("config_groups", {}).values()
    if any(group.get("weights", {}).get("num_bits") == 4
           and group.get("weights", {}).get("type") == "float" for group in groups):
        return "nvfp4"
    layers = prepared.get("quantized_layers", {}).values()
    if algorithm == "MIXED_PRECISION" and any(layer.get("quant_algo") == "NVFP4" for layer in layers):
        return "nvfp4"
    if metadata.get("quant_method") == "modelopt" or algorithm:
        raise NotImplementedError("Native inferred ModelOpt format requires supported NVFP4 or W4A16_AWQ metadata")
    return None


def _runtime_config(model_dir: Path, config: ModelConfig, model: Qwen38Model, **updates) -> dict:
    runtime = model.get_bundle_config_overrides(config)
    eos_token_ids = [config.eos_token_id]
    generation_path = model_dir / "generation_config.json"
    if generation_path.is_file():
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
        if not isinstance(generation, dict):
            raise ValueError("generation_config.json must contain one JSON object")
        if "eos_token_id" in generation:
            eos = generation["eos_token_id"]
            eos_token_ids = eos if isinstance(eos, list) else [eos]
    if not eos_token_ids or any(
        isinstance(value, bool) or not isinstance(value, int) for value in eos_token_ids
    ):
        raise ValueError("Qwen3.8 eos_token_id must contain one or more integer IDs")
    runtime["eos_token_ids"] = eos_token_ids
    runtime.update(updates)
    runtime.setdefault("prefill_chunk_len", 0)
    fields = {
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "bos_token_id",
        "eos_token_ids",
        "num_attention_layers",
        "num_mamba_layers",
        "d_inner",
        "mamba_d_state",
        "mamba_d_conv",
        "mamba_nheads",
        "mamba_head_dim",
        "conv_dim",
        "max_cache_length",
        "precision",
        "layer_types",
        "decoder_engine_layout",
        "has_mtp",
        "mtp_num_draft_tokens",
        "prefill_chunk_len",
    }
    if "kv_cache_quantization" in runtime:
        fields.add("kv_cache_quantization")
    if "tensor_parallel_size" in runtime:
        fields.add("tensor_parallel_size")
    missing = fields - runtime.keys()
    if missing:
        raise ValueError(f"Qwen3.8 runtime config is missing {sorted(missing)}")
    return {name: runtime[name] for name in fields}


def build(request, writer) -> None:
    """Build one Qwen3.8 hybrid text-generation bundle."""
    from .build_request import coerce_request

    request = coerce_request(request)

    from .edge_llm.config import Qwen38BuildRequest
    from .edge_llm.dispatch import build_paired

    if isinstance(request, Qwen38BuildRequest) and request.execution is not None:
        build_paired(request, writer, request.execution)
        return
    from .edge_llm.dispatch import build as edge_build
    edge_build(request, writer, _build_native)


def _mtp_enabled(request, weights, quant_ctx=None, *, require_mtp=False) -> bool:
    """MTP engines are built only when the user asks for them with `--mtp-seq-len`."""
    if request.mtp_seq_len is None and not require_mtp:
        return False
    if "mtp_layer.fc" not in weights and not _owned(quant_ctx, "mtp_layer.fc"):
        raise ValueError("Requested native MTP has no MTP weights; embedded draft weights are required")
    return True


def _publish_native_header(request, writer) -> None:
    """Normalize the explicit legacy vision task to its current public Task name."""
    from dataclasses import replace

    if request.task == "vision_language_generation":
        request = replace(request, task="images_text_to_text")
    writer.set_header(family="qwen3_8", task=request.task, backend=request.backend)


def _native_prefill_chunk_len(request, capacity, *, visual=False, external_draft=False,
                             fp8_cache=False, prepared_mtp=False):
    """Retain chunked prefill where its existing engine/state contract applies.

    Media, external features, TP2, and FP8/MTP source storage need per-token
    inputs not exposed by the existing chunked graph. They retain their exact
    single-token path; an explicit incompatible chunk request is never ignored.
    """
    requested = request.prefill_chunk_len
    single_token = (visual or external_draft or fp8_cache or prepared_mtp
                    or request.tensor_parallel_size != 1)
    if single_token:
        if requested not in (None, 0):
            raise NotImplementedError(
                "prefill_chunk_len is not supported with native media, external drafts, "
                "TP2, FP8 KV storage, or prepared MTP projections")
        return 0
    chunk = int(requested) if requested is not None else _DEFAULT_PREFILL_SEQ_LEN
    if chunk > capacity:
        chunk = ((capacity // 64) * 64 if capacity >= 64
                 else 1 << (capacity.bit_length() - 1))
    return chunk if chunk >= 2 else 0


# Share of GPU memory that the decode and prefill engines' weights may take together.
_TWO_ENGINE_MEMORY_FRACTION = 0.8


def _fit_prefill_engine(request, chunk: int, weight_bytes: int, gpu_bytes: int | None) -> int:
    """Drop the default prefill engine when a second copy of the weights does not fit on the GPU.

    The prefill engine carries its own copy of every weight. A 27B FP16 checkpoint needs about 108 GB
    for both engines, more than a 128 GB unified-memory GPU leaves after the runtime; prompts then run
    through the decode engine one token at a time. An explicit prefill_chunk_len is always kept.
    """
    if chunk == 0 or request.prefill_chunk_len is not None or gpu_bytes is None:
        return chunk
    needed = 2 * weight_bytes
    if needed <= gpu_bytes * _TWO_ENGINE_MEMORY_FRACTION:
        return chunk
    print(f"Skipping the prefill engine: the decode and prefill engines need about {needed / 1024**3:.0f} GiB "
          f"of weights and the GPU has {gpu_bytes / 1024**3:.0f} GiB; prompts run through the decode engine",
          file=sys.stderr)
    return 0


def _gpu_memory_bytes() -> int | None:
    """Total memory of the current CUDA device, or None when no device can be queried."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory)


def _build_native(request, writer, *, require_mtp: bool = False, draft_dir: Path | None = None,
                  draft_variant: str = "dspark") -> None:
    """Build the requested family-native topology without precision or task downgrade."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("qwen3_8 does not support dynamic_kv_cache")
    if request.graph_transform is not None:
        raise NotImplementedError("qwen3_8 does not support graph_transform")

    visual = request.task == "vision_language_generation"
    if request.task not in {"text_generation", "vision_language_generation"}:
        raise ValueError("qwen3_8 supports text_generation and vision_language_generation")
    if visual and draft_dir is not None:
        raise ValueError("Qwen native media does not combine with external drafts")
    if request.image_height is not None or request.image_width is not None:
        raise NotImplementedError("qwen3_8 does not support image dimensions")
    if request.video_num_frames is not None:
        raise NotImplementedError("qwen3_8 does not support video_num_frames")
    if request.max_batch_size != 1:
        raise NotImplementedError("qwen3_8 requires max_batch_size=1")
    if request.tensor_parallel_size not in (1, 2) or request.context_parallel_size != 1:
        raise NotImplementedError("qwen3_8 native supports TP1/TP2 and CP1")
    if request.quantization not in {None, "none", "nvfp4", "fp8", "int4_awq"}:
        raise NotImplementedError(
            f"qwen3_8 does not support quantization={request.quantization!r}")

    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    quantization = _native_quantization(request.quantization, model_dir, config.raw)
    quantized = quantization in {"nvfp4", "fp8", "int4_awq"}
    text_config = config.raw.get("text_config", config.raw)
    if (
        not isinstance(text_config, dict)
        or "output_gate_type" not in text_config
        or ("mlp_only_layers" in text_config)
    ):
        raise ValueError("checkpoint is not a Qwen3.8 model")
    precision = str(request.precision).lower()
    if precision not in {"fp16", "fp32", "bf16"}:
        raise ValueError("qwen3_8 precision must be fp16, bf16, or fp32")
    max_sequence_length = _positive_int(
        request.max_sequence_length or min(config.max_position_embeddings, 4096 if visual else 256),
        "max_sequence_length",
    )
    if max_sequence_length > config.max_position_embeddings:
        raise ValueError("qwen3_8 max_sequence_length exceeds checkpoint context capacity")

    if request.tensor_parallel_size == 2 and (
        precision != "fp16" or quantized or require_mtp or request.mtp_seq_len is not None
        or draft_dir is not None
    ):
        raise ValueError("Qwen native TP2 requires original FP16 ordinary generation")

    visual_config = visual_metadata = None
    if visual:
        from .native_media import native_media_metadata
        visual_config, visual_metadata = native_media_metadata(
            model_dir, config.raw, config.hidden_size, max_sequence_length, precision)

    draft_config = None
    draft_implementation = None
    if draft_dir is not None:
        if require_mtp or request.mtp_seq_len is not None or precision != "fp16":
            raise ValueError("Native external drafts require FP16 without embedded MTP")
        draft_raw = json.loads((Path(draft_dir) / "config.json").read_text())
        if draft_variant == "dspark":
            from . import draft as draft_implementation
            draft_config = draft_implementation.DraftConfig.from_dict(draft_raw, text_config)
            maximum = 1024
        elif draft_variant == "dflash":
            from . import dflash2 as draft_implementation
            draft_config = draft_implementation.DFlash2Config.from_dict(draft_raw, text_config)
            maximum = 4096
        else:
            raise ValueError("Unknown Qwen3.8 native external draft variant")
        if not 8 < max_sequence_length <= min(maximum, draft_raw.get("max_position_embeddings", maximum)):
            raise ValueError("Native external draft context exceeds target/draft contract")

    model = Qwen38Model()
    config.raw["_model_dir"] = str(model_dir)
    config.raw["_fp32_layers"] = list(request.fp32_layers)
    config.raw["_resolved_build_precision"] = precision
    config.raw["_quantized_build_requested"] = quantized

    quant_ctx = None
    readers = None
    if quantized:
        from . import graph_ops
        from .checkpoint_mapper import _open_safetensors

        # Share one set of safetensors readers between calibration and
        # load_weights() so the checkpoint's shard index is only built once.
        readers = _open_safetensors(model_dir)
        if quantization == "nvfp4":
            from .quantization import calibrate_qwen3_8_nvfp4
            quant_ctx = calibrate_qwen3_8_nvfp4(model_dir, config, graph_ops, readers=readers)
        elif quantization == "int4_awq":
            from .quantization import load_qwen3_8_awq
            quant_ctx = load_qwen3_8_awq(model_dir, config, graph_ops, readers=readers)
        else:
            from .quantization import calibrate_qwen3_8_fp8
            quant_ctx = calibrate_qwen3_8_fp8(model_dir, config, graph_ops, readers=readers)

    weights = model.load_weights(
        str(model_dir), config, precision=precision,
        quant_ctx=quant_ctx, readers=readers)
    # A 27B checkpoint's host weights plus TensorRT's device copy exceed a 128 GB unified-memory GPU.
    spill_if_large(weights, request.output_path)
    # Embedded draft weights alone do not request speculative execution.
    # Ordinary Edge fallback must not add duplicate verification engines.
    has_mtp = _mtp_enabled(request, weights, quant_ctx, require_mtp=require_mtp)
    prefill_chunk_len = _native_prefill_chunk_len(
        request, max_sequence_length, visual=visual, external_draft=draft_dir is not None,
        fp8_cache=bool(weights.get("_fp8_kv_cache", False)),
        prepared_mtp=has_mtp and _owned(quant_ctx, "mtp_layer.fc"),
    )
    if request.tensor_parallel_size == 1:
        prefill_chunk_len = _fit_prefill_engine(
            request, prefill_chunk_len, host_weight_bytes(weights), _gpu_memory_bytes())
    if request.tensor_parallel_size == 2:
        _build_tensor_parallel(request, writer, model_dir, config, model, weights,
                               max_sequence_length, visual_config, visual_metadata, readers)
        return
    # Every plan embeds the full weight set: write each one to the bundle as soon as it is built, so a
    # 27B build never holds two serialized plans (plus the host weights) in memory at once.
    _publish_native_header(request, writer)
    writer.add_bytes("engine.plan", model.build_engine(
        config,
        weights,
        max_sequence_length,
        precision=precision,
        quant_ctx=quant_ctx,
        verbose=bool(request.verbose),
        debug_layer_outputs=False,
        **({"target_feature_layers": draft_config.feature_layers} if draft_config else {}),
        **({"media_inputs": True} if visual else {}),
    ))

    if prefill_chunk_len >= 2:
        writer.add_bytes("prefill_engine.plan", model.build_engine_prefill(
            config, weights, max_sequence_length, prefill_chunk_len,
            precision=precision, quant_ctx=quant_ctx, verbose=bool(request.verbose),
        ))
    else:
        prefill_chunk_len = 0

    # Built only with --mtp-seq-len. mtp_seq_len = 1 real token + (mtp_seq_len-1) drafted tokens verified
    # per speculative-decode round. Default of 2 (1 draft) matches the
    # only configuration validated end-to-end on real hardware so far;
    # larger values draft more tokens per round through C++-orchestrated
    # single-step engines with isolated speculative KV state, at the cost of
    # a more expensive recovery path on any non-full-accept round (see
    # Qwen38MtpScheduler's class-level comment in mtp_scheduler.h).
    mtp_seq_len = int(request.mtp_seq_len) if request.mtp_seq_len is not None else 2
    if has_mtp and prefill_chunk_len >= 2:
        writer.add_bytes("mtp_prefill_engine.plan", model.build_mtp_prefill_engine(
            config, weights, max_sequence_length + mtp_seq_len - 2, prefill_chunk_len,
            precision=precision, quant_ctx=quant_ctx, verbose=bool(request.verbose),
        ))
    if visual:
        from .vision import build_vision_engine, load_vision_weights
        visual_weights, visual_quant = load_vision_weights(model_dir, visual_config, readers=readers)
        writer.add_bytes("native_visual.plan", build_vision_engine(
            visual_config, visual_weights, visual_metadata["max_tokens"] * visual_config.merge ** 2,
            quant_ctx=visual_quant))
        del visual_weights, visual_quant
        writer.add_json("native_visual.json", visual_metadata)
    if has_mtp:
        writer.add_bytes("mtp_engine.plan", model.build_mtp_engine(
            # Scratch draft suffixes must fit even at the target's last
            # admitted position, without evicting the confirmed prefix.
            config, weights, max_sequence_length + mtp_seq_len - 2,
            precision=precision, quant_ctx=quant_ctx, verbose=bool(request.verbose),
            media_inputs=visual,
        ))
        writer.add_bytes("multi_token_engine.plan", model.build_engine_multi_token(
            config, weights, max_sequence_length, mtp_seq_len,
            precision=precision, quant_ctx=quant_ctx, verbose=bool(request.verbose),
            media_inputs=visual,
        ))
    writer.add_json(
        "runtime.json",
        _runtime_config(
            model_dir,
            config,
            model,
            precision=precision,
            max_cache_length=max_sequence_length,
            decoder_engine_layout="single",
            has_mtp=has_mtp,
            mtp_num_draft_tokens=(mtp_seq_len - 1) if has_mtp else 0,
            prefill_chunk_len=prefill_chunk_len,
            **({"kv_cache_quantization": "fp8"} if weights.get("_fp8_kv_cache", False) else {}),
        ),
    )
    if draft_config is not None:
        import numpy as np

        draft_weights = draft_implementation.load_draft_weights(Path(draft_dir), draft_config)
        prefix = "native_dspark" if draft_variant == "dspark" else "native_dflash2"
        writer.add_bytes(f"{prefix}.plan", draft_implementation.build_draft_engine(
            draft_config, draft_weights, weights.get("w_lm_head"), max_sequence_length,
            quant_ctx=quant_ctx, verbose=bool(request.verbose),
        ))
        metadata = {
            "schema_version": 1, "hidden_size": draft_config.hidden,
            "vocab_size": draft_config.vocab, "max_cache_length": max_sequence_length,
            "feature_layers": list(draft_config.feature_layers), "rows": draft_config.query_rows,
            "mask_token_id": draft_config.mask_token,
        }
        if draft_variant == "dspark":
            writer.add_bytes(f"{prefix}_heads.plan", draft_implementation.build_markov_engine(
                draft_config, draft_weights, verbose=bool(request.verbose)))
            metadata.update(sample_from_anchor=draft_config.sample_from_anchor,
                            causal=draft_config.causal, confidence=draft_config.confidence)
        else:
            writer.add_bytes(f"{prefix}_selector.plan", draft_implementation.build_selector_engine(
                draft_config, draft_weights, verbose=bool(request.verbose)))
            metadata.update(window=draft_config.window, rank=draft_config.selector_rank,
                            top_k=draft_config.selector_top_k)
        # Host-resident lookup avoids another vocabulary-sized GPU table/engine.
        writer.add_bytes(f"{prefix}_embeddings.bin",
                         np.asarray(weights["embedding"], dtype=np.float16).tobytes())
        writer.add_json(f"{prefix}.json", metadata)
    for filename in _BUNDLE_FILES:
        path = model_dir / filename
        if path.is_file():
            writer.add_bytes(filename, path.read_bytes())


def _build_tensor_parallel(request, writer, model_dir, config, model, weights,
                           capacity, visual_config, visual_metadata, readers):
    from cuda.bindings import runtime as cuda
    from .parallel import shard_weights

    def check(result):
        status, *values = result
        if status != cuda.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Qwen native TP CUDA operation failed: {status}")
        return values[0] if values else None

    count = check(cuda.cudaGetDeviceCount())
    if count < 2:
        raise RuntimeError("Qwen native TP2 build requires two visible CUDA devices")
    previous = check(cuda.cudaGetDevice())
    visual = visual_config is not None
    _publish_native_header(request, writer)
    visual_weights = visual_quant = None
    if visual:
        from .vision import build_vision_engine, load_vision_weights
        visual_weights, visual_quant = load_vision_weights(model_dir, visual_config, readers=readers)
    try:
        for rank in range(2):
            check(cuda.cudaSetDevice(rank))
            local, shard = shard_weights(config, weights, rank)
            writer.add_bytes(f"engine.rank{rank}.plan", model.build_engine(
                local, shard, capacity, precision="fp16", media_inputs=visual,
                verbose=bool(request.verbose)))
            if visual:
                writer.add_bytes(f"native_visual.rank{rank}.plan", build_vision_engine(
                    visual_config, visual_weights, visual_metadata["max_tokens"] * visual_config.merge ** 2,
                    quant_ctx=visual_quant))
    finally:
        check(cuda.cudaSetDevice(previous))
    if visual:
        writer.add_json("native_visual.json", visual_metadata)
    writer.add_json("runtime.json", _runtime_config(
        model_dir, config, model, precision="fp16", max_cache_length=capacity,
        decoder_engine_layout="single", has_mtp=False, mtp_num_draft_tokens=0,
        tensor_parallel_size=2))
    for filename in _BUNDLE_FILES:
        path = model_dir / filename
        if path.is_file():
            writer.add_bytes(filename, path.read_bytes())
