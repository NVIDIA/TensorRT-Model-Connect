# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned Qwen3.8 checkpoint-to-bundle build."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .config import ModelConfig
from .engine_builder import Qwen38Model, native_kv_available


_BUNDLE_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.model",
)


# KV cache length when --max-sequence-length is not given (capped by the checkpoint's context).
_DEFAULT_MAX_SEQUENCE_LENGTH = 4096
_DEFAULT_PREFILL_SEQ_LEN = 128
# Dynamic prefill engine: largest call and the token count the engine is tuned for.
_DEFAULT_DYNAMIC_PREFILL_MAX_LEN = 1024
_DEFAULT_DYNAMIC_PREFILL_OPT_LEN = 256


def _resolve_lengths(
    max_position_embeddings: int,
    max_sequence_length: int | None,
    prefill_chunk_len: int | None,
    prefill_dynamic: bool,
) -> tuple[int, int]:
    """KV cache length and prefill call size (tokens) for one build.

    Without --max-sequence-length the cache is min(checkpoint context, 4096); a dynamic
    prefill engine takes up to prefill_chunk_len tokens in one call, so then the default
    cache is raised to at least that. An explicit cache that cannot hold one dynamic call
    is an error, not a silent clamp.
    """
    if prefill_chunk_len is None:
        prefill_chunk_len = (
            _DEFAULT_DYNAMIC_PREFILL_MAX_LEN if prefill_dynamic else _DEFAULT_PREFILL_SEQ_LEN)
    if max_sequence_length:
        cache_len = _positive_int(max_sequence_length, "max_sequence_length")
    else:
        cache_len = min(
            max_position_embeddings,
            max(_DEFAULT_MAX_SEQUENCE_LENGTH, prefill_chunk_len if prefill_dynamic else 0))
    if cache_len > max_position_embeddings:
        raise ValueError("qwen3_8 max_sequence_length exceeds checkpoint context capacity")
    if prefill_dynamic and prefill_chunk_len > cache_len:
        raise ValueError(
            f"the dynamic prefill engine takes up to --prefill-chunk-len = {prefill_chunk_len} tokens in "
            f"one call, but the KV cache holds {cache_len} tokens; raise --max-sequence-length "
            f"(at most {max_position_embeddings}) or lower --prefill-chunk-len")
    return cache_len, prefill_chunk_len


def _choose_prefill_engine(
    engine: str,
    *,
    precision: str,
    mtp: bool,
    prefill_chunk_len: int | None,
    plugin_unavailable,
) -> tuple[bool, str | None]:
    """(use the dynamic prefill engine, why the static one was chosen instead).

    `auto` takes the dynamic engine when it can be built and falls back to the static one,
    returning the reason; `dynamic` raises instead of falling back. `plugin_unavailable` is
    called last and returns None or the reason the cuTile GDN prefill plugin cannot be used.
    """
    if engine == "static":
        return False, None
    if prefill_chunk_len == 0:  # no prefill engine at all (build_request rejects 'dynamic' here)
        return False, None
    if precision == "fp32":
        reason = "the dynamic prefill engine needs fp16 or bf16 (cuTile GDN plugin)"
    elif mtp:
        reason = "MTP engines are not supported with the dynamic prefill engine yet"
    else:
        reason = plugin_unavailable()
    if reason is None:
        return True, None
    if engine == "dynamic":
        raise ValueError(f"--prefill-engine dynamic is not available: {reason}")
    return False, reason


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
    if runtime.get("prefill_dynamic"):
        fields.add("prefill_dynamic")
    if runtime.get("kv_native"):
        fields.add("kv_native")
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


def _mtp_enabled(request, weights) -> bool:
    """MTP engines are built only when the user asks for them with `--mtp-seq-len`."""
    if request.mtp_seq_len is None:
        return False
    if "mtp_layer.fc" not in weights:
        raise ValueError("--mtp-seq-len was given but the checkpoint has no MTP weights")
    return True


def _build_native(request, writer) -> None:
    """Preserve the original native graph and unsupported-request behavior."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("qwen3_8 does not support dynamic_kv_cache")

    if request.task != "text_generation":
        raise ValueError("qwen3_8 supports only task=text_generation")
    if request.image_height is not None or request.image_width is not None:
        raise NotImplementedError("qwen3_8 does not support image dimensions")
    if request.video_num_frames is not None:
        raise NotImplementedError("qwen3_8 does not support video_num_frames")
    if request.max_batch_size != 1:
        raise NotImplementedError("qwen3_8 requires max_batch_size=1")
    if request.tensor_parallel_size != 1 or request.context_parallel_size != 1:
        raise NotImplementedError("qwen3_8 supports only single-device builds")
    quantized = request.quantization in {"nvfp4", "fp8"}
    if request.quantization not in {None, "none", "nvfp4", "fp8"}:
        raise NotImplementedError(
            f"qwen3_8 does not support quantization={request.quantization!r}")

    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
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
    from .native_plugins.gdn_plugin_builder import prefill_plugin_unavailable_reason

    prefill_dynamic, static_reason = _choose_prefill_engine(
        request.prefill_engine, precision=precision, mtp=request.mtp_seq_len is not None,
        prefill_chunk_len=request.prefill_chunk_len,
        plugin_unavailable=prefill_plugin_unavailable_reason)
    if static_reason is not None:
        print(f"[trtmc build] qwen3_8: building the static prefill engine: {static_reason}",
              file=sys.stderr)
    max_sequence_length, prefill_chunk_len = _resolve_lengths(
        config.max_position_embeddings, request.max_sequence_length,
        request.prefill_chunk_len, prefill_dynamic)

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
        if request.quantization == "nvfp4":
            from .quantization import calibrate_qwen3_8_nvfp4
            quant_ctx = calibrate_qwen3_8_nvfp4(model_dir, config, graph_ops, readers=readers)
        else:
            from .quantization import calibrate_qwen3_8_fp8
            quant_ctx = calibrate_qwen3_8_fp8(model_dir, config, graph_ops, readers=readers)

    weights = model.load_weights(
        str(model_dir), config, precision=precision,
        quant_ctx=quant_ctx, readers=readers)
    has_mtp = _mtp_enabled(request, weights)
    # Native KV cache: the engines write the cache in place and attention reads only the filled
    # rows. The MTP engines share the cache buffers with the old layout, so they keep it.
    native_kv = (not has_mtp) and native_kv_available()

    # Prompt tokens per multi-token prefill call. 0 keeps the old
    # one-decode-step-per-prompt-token prefill (no prefill_engine.plan).
    if prefill_chunk_len > max_sequence_length:
        # Static engine: largest valid chunk that still fits the cache.
        prefill_chunk_len = (
            (max_sequence_length // 64) * 64 if max_sequence_length >= 64
            else 1 << (max_sequence_length.bit_length() - 1))
    prefill_plan = None
    prefill_opt_len = 0
    if prefill_dynamic:
        if prefill_chunk_len < 2:
            raise ValueError("the dynamic prefill engine needs a multi-token prefill engine")
        prefill_opt_len = min(
            int(request.prefill_opt_len or _DEFAULT_DYNAMIC_PREFILL_OPT_LEN), prefill_chunk_len)
    if prefill_chunk_len >= 2:
        prefill_plan = model.build_engine_prefill(
            config, weights, max_sequence_length, prefill_chunk_len,
            precision=precision, quant_ctx=quant_ctx, verbose=bool(request.verbose),
            dynamic=prefill_dynamic, opt_len=prefill_opt_len, dynamic_cache=True,
            native_kv=native_kv,
        )
    else:
        prefill_chunk_len = 0

    plan = model.build_engine(
        config,
        weights,
        max_sequence_length,
        precision=precision,
        quant_ctx=quant_ctx,
        verbose=bool(request.verbose),
        debug_layer_outputs=False,
        native_kv=native_kv,
        cache_pad_rows=prefill_chunk_len,
    )

    mtp_plan = None
    draft_chain_plan = None
    multi_token_plan = None
    # Built only with --mtp-seq-len. mtp_seq_len = 1 real token + (mtp_seq_len-1) drafted tokens verified
    # per speculative-decode round. Default of 2 (1 draft) matches the
    # only configuration validated end-to-end on real hardware so far;
    # larger values draft more tokens per round via build_mtp_draft_chain_engine's
    # in-graph self-chaining (see that function's docstring) at the cost of
    # a more expensive recovery path on any non-full-accept round (see
    # Qwen38MtpScheduler's class-level comment in mtp_scheduler.h).
    mtp_seq_len = int(request.mtp_seq_len) if request.mtp_seq_len is not None else 2
    mtp_prefill_plan = None
    if has_mtp and prefill_chunk_len >= 2:
        mtp_prefill_plan = model.build_mtp_prefill_engine(
            config, weights, max_sequence_length, prefill_chunk_len,
            precision=precision, quant_ctx=quant_ctx, verbose=bool(request.verbose),
        )
    if has_mtp:
        mtp_plan = model.build_mtp_engine(
            config, weights, max_sequence_length,
            precision=precision, quant_ctx=quant_ctx, verbose=bool(request.verbose),
        )
        draft_chain_plan = model.build_mtp_draft_chain_engine(
            config, weights, max_sequence_length, mtp_seq_len - 1,
            precision=precision, quant_ctx=quant_ctx, verbose=bool(request.verbose),
        )
        multi_token_plan = model.build_engine_multi_token(
            config, weights, max_sequence_length, mtp_seq_len,
            precision=precision, quant_ctx=quant_ctx, verbose=bool(request.verbose),
        )

    writer.set_header(family="qwen3_8", task=request.task, backend=request.backend)
    writer.add_bytes("engine.plan", plan)
    if prefill_plan is not None:
        writer.add_bytes("prefill_engine.plan", prefill_plan)
    if mtp_prefill_plan is not None:
        writer.add_bytes("mtp_prefill_engine.plan", mtp_prefill_plan)
    if mtp_plan is not None:
        writer.add_bytes("mtp_engine.plan", mtp_plan)
    if draft_chain_plan is not None:
        writer.add_bytes("mtp_draft_chain_engine.plan", draft_chain_plan)
    if multi_token_plan is not None:
        writer.add_bytes("multi_token_engine.plan", multi_token_plan)
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
            prefill_dynamic=prefill_dynamic,
            kv_native=native_kv,
        ),
    )
    for filename in _BUNDLE_FILES:
        path = model_dir / filename
        if path.is_file():
            writer.add_bytes(filename, path.read_bytes())
