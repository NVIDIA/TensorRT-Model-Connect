# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ELF Flow family-owned build implementation.

ELF is implemented from the GitHub source at https://github.com/lillian039/ELF.
The weight names below mirror the Flax module tree in ``src/modules/model.py``.
"""

from __future__ import annotations

import pickle
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoint_mapper import WeightDict
from .model_config import ModelConfig
from .config import resolve_elf_config


class _TensorStore:
    def __init__(self, model_dir: str | Path):
        from .checkpoint_mapper import _has_tensor, _load_tensor, _open_safetensors

        self._has_tensor = _has_tensor
        self._load_tensor = _load_tensor
        self._readers = None
        self._arrays: dict[str, np.ndarray] | None = None
        model_path = Path(model_dir)
        try:
            self._readers = _open_safetensors(model_path)
        except FileNotFoundError:
            self._arrays = _load_local_elf_arrays(model_path)
            if self._arrays is None:
                raise FileNotFoundError(
                    f"No ELF safetensors, npz, or local GitHub checkpoint found in {model_path}"
                )

    def has(self, name: str) -> bool:
        if self._arrays is not None:
            return name in self._arrays
        return bool(self._has_tensor(self._readers, name))

    def get(self, name: str) -> np.ndarray:
        if self._arrays is not None:
            return np.asarray(self._arrays[name], dtype=np.float32)
        return self._load_tensor(self._readers, name)


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def _flatten_arrays(value: Any, prefix: str = "") -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            arrays.update(_flatten_arrays(item, name))
        return arrays
    if prefix:
        try:
            arrays[prefix] = np.asarray(value)
        except (TypeError, ValueError):
            pass
    return arrays


def _select_upstream_params(payload: Any) -> Any:
    if isinstance(payload, Mapping):
        if "ema_params1" in payload:
            return payload["ema_params1"]
        if "params" in payload:
            return payload["params"]
    return payload


def _load_npz_arrays(path: Path) -> dict[str, np.ndarray] | None:
    try:
        loaded = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return None
    if not hasattr(loaded, "files"):
        return None
    return {key: loaded[key] for key in loaded.files}


def _load_pickle_arrays(path: Path) -> dict[str, np.ndarray] | None:
    try:
        with path.open("rb") as f:
            payload = pickle.load(f)
    except Exception:
        return None
    arrays = _flatten_arrays(_select_upstream_params(payload))
    return arrays or None


def _load_flax_arrays(path: Path) -> dict[str, np.ndarray] | None:
    try:
        from flax import serialization
    except ImportError:
        return None

    if path.is_file():
        try:
            payload = serialization.msgpack_restore(path.read_bytes())
        except Exception:
            return None
    else:
        try:
            from flax.training import checkpoints

            payload = checkpoints.restore_checkpoint(str(path.resolve()), target=None)
        except Exception:
            return None
    arrays = _flatten_arrays(_select_upstream_params(payload))
    return arrays or None


def _load_orbax_arrays(path: Path) -> dict[str, np.ndarray] | None:
    if not path.is_dir() or not (
        (path / "_CHECKPOINT_METADATA").exists() or (path / "manifest.ocdbt").exists()
    ):
        return None
    try:
        import orbax.checkpoint as ocp

        payload = ocp.PyTreeCheckpointer().restore(str(path.resolve()))
    except Exception:
        return None
    arrays = _flatten_arrays(_select_upstream_params(payload))
    return arrays or None


def _load_checkpoint_arrays(path: Path) -> dict[str, np.ndarray] | None:
    if path.suffix == ".npz":
        arrays = _load_npz_arrays(path)
        if arrays:
            return arrays
    arrays = _load_orbax_arrays(path)
    if arrays:
        return arrays
    arrays = _load_pickle_arrays(path)
    if arrays:
        return arrays
    return _load_flax_arrays(path)


def _local_checkpoint_candidates(model_path: Path) -> list[Path]:
    if model_path.is_file():
        return [model_path]
    if not model_path.is_dir():
        return []

    candidates: list[Path] = []
    for name in ("model.npz", "elf_params.npz"):
        candidate = model_path / name
        if candidate.exists():
            candidates.append(candidate)

    checkpoints = sorted(
        model_path.glob("checkpoint_*"),
        key=lambda item: (_checkpoint_step(item), item.name),
        reverse=True,
    )
    candidates.extend(checkpoints)
    return candidates


def _load_local_elf_arrays(model_path: Path) -> dict[str, np.ndarray] | None:
    for candidate in _local_checkpoint_candidates(model_path):
        arrays = _load_checkpoint_arrays(candidate)
        if arrays:
            return arrays
    return None


def _name_variants(name: str) -> list[str]:
    variants = [name]
    if "." in name:
        variants.append(name.replace(".", "/"))
    if "/" in name:
        variants.append(name.replace("/", "."))
    prefixed: list[str] = []
    for item in variants:
        prefixed.append(f"params.{item}")
        prefixed.append(f"params/{item}")
    out: list[str] = []
    for item in variants + prefixed:
        if item not in out:
            out.append(item)
    return out


def _load(store: _TensorStore, *names: str, dtype: np.dtype = np.float32) -> np.ndarray:
    for name in names:
        for candidate in _name_variants(name):
            if store.has(candidate):
                return np.ascontiguousarray(store.get(candidate), dtype=dtype)
    joined = ", ".join(names)
    raise KeyError(f"ELF tensor not found; tried: {joined}")


def _find_encoder_checkpoint(model_dir: str | Path) -> Path | None:
    model_path = Path(model_dir)
    for name in (
        "t5_small_encoder_jax.pkl",
        "encoder_checkpoint.pkl",
        "text_encoder.pkl",
        "t5_encoder.pkl",
    ):
        candidate = model_path / name
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _elf_encoder_pad_token_id(config: ModelConfig) -> int:
    raw = config.raw or {}
    explicit = raw.get("elf_encoder_pad_token_id", raw.get("encoder_pad_token_id"))
    if explicit is not None:
        return int(explicit)
    pad_token_id = raw.get("pad_token_id", config.pad_token_id)
    if isinstance(pad_token_id, int) and pad_token_id >= 0:
        return int(pad_token_id)
    if str(raw.get("pad_token", "")).lower() == "eos":
        eos_token_id = raw.get("eos_token_id", config.eos_token_id)
        return int(eos_token_id) if isinstance(eos_token_id, int) and eos_token_id >= 0 else 1
    return 0


name = "elf_flow"
runtime_strategy = "elf_flow"


def matches(config) -> bool:
    model_type = str(getattr(config, "model_type", config))
    mt = (model_type or "").lower()
    return mt in ("elf", "embedded_language_flow", "embedded-language-flow")


def load_weights(model_dir: str, config: ModelConfig, *, precision: str = "fp32") -> WeightDict:
    cfg = resolve_elf_config(config)
    store = _TensorStore(model_dir)
    # Keep source weights in FP32 so fp32_layers can preserve individual
    # blocks without first rounding their constants through FP16.
    target_dtype = np.float32
    weights = WeightDict()
    config.raw["_elf_model_dir"] = str(Path(model_dir).resolve())
    encoder_checkpoint = _find_encoder_checkpoint(model_dir)
    if encoder_checkpoint is not None:
        weights["_elf_encoder_checkpoint"] = str(encoder_checkpoint.resolve())
        config.raw["_elf_encoder_checkpoint"] = str(encoder_checkpoint.resolve())

    def proj(name: str, *aliases: str) -> np.ndarray:
        return _load(store, name, *aliases, dtype=target_dtype)

    def vec(name: str, *aliases: str) -> np.ndarray:
        return _load(store, name, *aliases, dtype=np.float32)

    if cfg["input_dim"] == 2 * cfg["text_encoder_dim"]:
        weights["self_cond_proj.w"] = proj("self_cond_proj.kernel")
        weights["self_cond_proj.b"] = proj("self_cond_proj.bias")

    weights["text_proj.proj1.w"] = proj("text_proj.proj1.kernel")
    weights["text_proj.proj2.w"] = proj("text_proj.proj2.kernel")
    weights["text_proj.proj2.b"] = proj("text_proj.proj2.bias")

    weights["t_embedder.mlp_0.w"] = proj("t_embedder.mlp_0.kernel")
    weights["t_embedder.mlp_0.b"] = proj("t_embedder.mlp_0.bias")
    weights["t_embedder.mlp_2.w"] = proj("t_embedder.mlp_2.kernel")
    weights["t_embedder.mlp_2.b"] = proj("t_embedder.mlp_2.bias")
    weights["t_emb_tokens"] = proj("t_emb_tokens")

    if cfg["num_self_cond_cfg_tokens"] > 0:
        weights["self_cond_cfg_embedder.mlp_0.w"] = proj("self_cond_cfg_embedder.mlp_0.kernel")
        weights["self_cond_cfg_embedder.mlp_0.b"] = proj("self_cond_cfg_embedder.mlp_0.bias")
        weights["self_cond_cfg_embedder.mlp_2.w"] = proj("self_cond_cfg_embedder.mlp_2.kernel")
        weights["self_cond_cfg_embedder.mlp_2.b"] = proj("self_cond_cfg_embedder.mlp_2.bias")
        weights["self_cond_cfg_tokens"] = proj("self_cond_cfg_tokens")

    if cfg["num_model_mode_tokens"] > 0:
        weights["mode_tokens"] = proj("mode_tokens")

    for layer_idx in range(cfg["depth"]):
        src = f"blocks_{layer_idx}"
        dst = f"layer.{layer_idx}"
        weights[f"{dst}.norm1"] = vec(f"{src}.norm1.weight")
        weights[f"{dst}.attn.qkv.w"] = proj(f"{src}.attn.qkv.kernel")
        weights[f"{dst}.attn.qkv.b"] = proj(f"{src}.attn.qkv.bias")
        weights[f"{dst}.attn.q_norm"] = vec(f"{src}.attn.q_norm.weight")
        weights[f"{dst}.attn.k_norm"] = vec(f"{src}.attn.k_norm.weight")
        weights[f"{dst}.attn.proj.w"] = proj(f"{src}.attn.proj.kernel")
        weights[f"{dst}.attn.proj.b"] = proj(f"{src}.attn.proj.bias")
        weights[f"{dst}.norm2"] = vec(f"{src}.norm2.weight")
        weights[f"{dst}.mlp.w12.w"] = proj(f"{src}.mlp.w12.kernel")
        weights[f"{dst}.mlp.w12.b"] = proj(f"{src}.mlp.w12.bias")
        weights[f"{dst}.mlp.w3.w"] = proj(f"{src}.mlp.w3.kernel")
        weights[f"{dst}.mlp.w3.b"] = proj(f"{src}.mlp.w3.bias")

    weights["decoder.proj.w"] = proj("proj_kernel")
    weights["decoder.proj.b"] = proj("proj_bias")
    weights["decoder.unembed.w"] = proj("unembed_kernel")
    weights["decoder.unembed.b"] = proj("unembed_bias")
    if config.vocab_size <= 0 and weights["decoder.unembed.w"].ndim == 2:
        config.vocab_size = int(weights["decoder.unembed.w"].shape[1])
        config.raw["vocab_size"] = config.vocab_size
    weights["final.norm"] = vec("final_layer.norm_final.weight")
    weights["final.linear.w"] = proj("final_layer.linear.kernel")
    weights["final.linear.b"] = proj("final_layer.linear.bias")
    return weights


def build_engine(
    config: ModelConfig,
    weights: WeightDict,
    max_cache_length: int,
    *,
    precision: str = "fp32",
    quant_ctx=None,
    verbose: bool = False,
    debug_layer_outputs: bool = False,
    parallel_config=None,
) -> bytes:
    del quant_ctx
    from .builder import build_elf_flow_engine

    return build_elf_flow_engine(
        config,
        weights,
        max_cache_length,
        precision=precision,
        verbose=verbose,
        debug_layer_outputs=debug_layer_outputs,
    )


def build_extra_engines(
    config: ModelConfig,
    weights: WeightDict,
    max_cache_length: int,
    *,
    precision: str = "fp32",
    verbose: bool = False,
    build_timing: dict | None = None,
) -> dict | None:
    del max_cache_length
    encoder_checkpoint = weights.get("_elf_encoder_checkpoint")
    if not encoder_checkpoint:
        return {}

    from ...build_timing import timed_trt_compile, timed_weight_loading
    from .t5_encoder_builder import (
        build_t5_encoder_engine,
        load_jax_t5_encoder_weights,
    )

    cfg = resolve_elf_config(config)
    with timed_weight_loading(build_timing, "elf_t5_encoder"):
        t5_weights = load_jax_t5_encoder_weights(
            str(encoder_checkpoint), precision=precision, num_layers=6
        )
    with timed_trt_compile(build_timing, "elf_t5_encoder"):
        t5_plan = build_t5_encoder_engine(
            t5_weights,
            d_model=cfg["text_encoder_dim"],
            num_heads=8,
            d_kv=64,
            d_ff=2048,
            num_layers=6,
            vocab_size=32128,
            max_seq_len=cfg["max_length"],
            eps=1e-6,
            verbose=verbose,
        )
    return {"elf_text_encoder_plan": t5_plan}


def get_bundle_config_overrides(config: ModelConfig) -> dict:
    cfg = resolve_elf_config(config)
    raw = config.raw or {}
    return {
        "runtime_strategy": runtime_strategy,
        "model_type": "elf",
        "hidden_size": cfg["hidden_size"],
        "num_hidden_layers": cfg["depth"],
        "num_attention_heads": cfg["num_heads"],
        "head_dim": cfg["head_dim"],
        "max_position_embeddings": cfg["max_length"],
        "vocab_size": cfg["vocab_size"],
        "elf_variant": cfg["variant"],
        "elf_max_length": cfg["max_length"],
        "elf_max_input_length": cfg["max_input_length"],
        "elf_text_encoder_dim": cfg["text_encoder_dim"],
        "elf_input_dim": cfg["input_dim"],
        "elf_bottleneck_dim": cfg["bottleneck_dim"],
        "elf_num_time_tokens": cfg["num_time_tokens"],
        "elf_num_self_cond_cfg_tokens": cfg["num_self_cond_cfg_tokens"],
        "elf_num_model_mode_tokens": cfg["num_model_mode_tokens"],
        "elf_denoiser_noise_scale": cfg["denoiser_noise_scale"],
        "elf_denoiser_p_mean": cfg["denoiser_p_mean"],
        "elf_denoiser_p_std": cfg["denoiser_p_std"],
        "elf_t_eps": cfg["t_eps"],
        "elf_latent_mean": float(raw.get("latent_mean", 0.0)),
        "elf_latent_std": float(raw.get("latent_std", 0.2)),
        "elf_encoder_model_name": raw.get("encoder_model_name", "t5-small"),
        "elf_encoder_max_length": cfg["max_length"],
        "elf_encoder_pad_token_id": _elf_encoder_pad_token_id(config),
        "elf_has_text_encoder": int(bool(raw.get("_elf_encoder_checkpoint"))),
        "elf_runtime_contract": "api_path_denoise_or_decode_logits",
        "elf_user_contract": "diffusion_text_generation",
        "elf_output_schema": "jsonl_id_generated_after_sampler_decode",
    }


# Model-owned build entry
requires_tokenizer = True
embed_input = False


def _build_local_engine(
    config,
    weights,
    max_cache_length,
    precision,
    quant_ctx,
    verbose,
    parallel,
):
    from tensorrt_model_connect.tvm_ffi.graph_build import engine_role, inspection_role

    def build_role(role: str):
        previous = config.raw.get("_decoder_engine_role")
        config.raw["_decoder_engine_role"] = role
        try:
            with engine_role(role):
                return build_engine(
                    config,
                    weights,
                    max_cache_length,
                    precision=precision,
                    quant_ctx=quant_ctx,
                    verbose=verbose,
                )
        finally:
            if previous is None:
                config.raw.pop("_decoder_engine_role", None)
            else:
                config.raw["_decoder_engine_role"] = previous

    target_role = inspection_role()
    if target_role is not None:
        build_role(target_role)
        raise RuntimeError("graph inspection did not reach TensorRT serialization")
    selected_role = (
        "dual_profile"
        if str(config.raw.get("_decoder_engine_layout") or "split") == "dual_profile"
        else "decode"
    )
    return build_role(selected_role), (
        "dual_profile" if selected_role == "dual_profile" else "single"
    )


def build(model_dir: str, output_path: str, **options) -> None:
    """Build the complete elf_flow bundle in the owning family module."""
    import json
    import time
    from datetime import datetime, timezone
    from pathlib import Path

    from tensorrt_model_connect.build_timing import (
        add_build_timing,
        build_timing_phase,
        new_build_timing,
        untracked_phase_time,
        write_build_timing,
    )
    from tensorrt_model_connect.bundle_writer import (
        BundleInfo,
        BundleSection,
        gpu_name,
        tensorrt_abi,
        tensorrt_version,
        write_bundle,
    )
    from tensorrt_model_connect.parallel_config import (
        normalize_parallel_config,
    )
    from tensorrt_model_connect.tokenizer_conversion import (
        detect_tokenizer_add_special_tokens,
        prepare_tokenizer_special_frame,
    )

    model_path = Path(model_dir)
    decoder_engine_layout = str(options.get("decoder_engine_layout") or "split")
    if decoder_engine_layout not in {"split", "dual_profile"}:
        raise ValueError(
            "decoder_engine_layout must be 'split' or 'dual_profile', "
            f"got {decoder_engine_layout!r}"
        )
    parallel = normalize_parallel_config(options.get("parallel_config"))
    if parallel.cp_enabled:
        raise NotImplementedError(f"{name} does not support context-parallel builds")
    if options.get("dynamic_kv_cache") or options.get("triattention_stats_path"):
        raise ValueError(f"{name} does not use a decoder KV-cache runtime")

    config = ModelConfig.from_dir(model_path)
    config.raw["_model_dir"] = str(model_path)
    config.raw["_decoder_engine_layout"] = decoder_engine_layout
    config.raw["_fp32_layers"] = sorted(set(options.get("fp32_layers") or ()))
    config.raw["_family_build_options"] = dict(options.get("family_build_options") or {})
    config.raw["_parallel_build_enabled"] = bool(parallel.enabled)
    config.raw["_rtx_build_requested"] = bool(options.get("rtx"))
    config.raw["_runtime_dynamic_kv_requested"] = False
    config.raw["_quantized_build_requested"] = bool(options.get("quantize"))
    precision = str(options.get("precision") or "fp32").lower()
    config.raw["_resolved_build_precision"] = precision
    requested_cache_length = options.get("max_cache_length")
    max_cache_length = int(256 if requested_cache_length is None else requested_cache_length)
    if max_cache_length < 1:
        raise ValueError("max_cache_length must be >= 1")

    timing = new_build_timing(options.get("build_timing_path"))
    timing["model_dir"] = str(model_path)
    timing["output_path"] = str(output_path)
    started = time.monotonic()
    write_build_timing(timing)

    weights_started = time.monotonic()
    weights = load_weights(str(model_path), config, precision=precision)
    add_build_timing(timing, "weights_loading_s", time.monotonic() - weights_started)
    write_build_timing(timing)

    quantize = options.get("quantize")
    quant_ctx = None
    quant_plan = None
    if quantize:
        from dataclasses import replace
        from tensorrt_model_connect.quantization import QuantPlan, build_quant_context
        from . import graph_ops as family_graph_ops

        quant_plan = QuantPlan.from_build_args(
            precision=precision,
            quantize=str(quantize),
            quant_scales=options.get("quant_scales"),
            quant_calibration_samples=int(options.get("quant_calibration_samples") or 512),
        )
        quant_method = str(
            config.raw.get("quantization_config", {}).get("quant_method", "")
        ).lower()
        if quant_plan.scale_source == "modelopt" and quant_method in {
            "awq",
            "gptq",
            "compressed-tensors",
            "compressed_tensors",
        }:
            quant_plan = replace(quant_plan, scale_source="prequantized")
        quant_ctx = build_quant_context(
            format_name=quant_plan.quant_format,
            model_dir=str(model_path),
            config=config,
            scales_json=options.get("quant_scales"),
            num_calibration_samples=int(options.get("quant_calibration_samples") or 512),
            quant_plan=quant_plan,
            graph_ops=family_graph_ops,
        )

    if parallel.enabled:
        raise ValueError(f"{name} does not support tensor-parallel builds")

    verbose = bool(options.get("verbose"))
    compile_started = time.monotonic()

    sections = []
    plan, decoder_layout = _build_local_engine(
        config, weights, max_cache_length, precision, quant_ctx, verbose, parallel
    )
    sections.append(BundleSection("engine_plan", plan))

    compile_elapsed = time.monotonic() - compile_started
    add_build_timing(timing, "trt_compile_s", compile_elapsed)
    add_build_timing(timing, "trt_compile_main_engine_s", compile_elapsed)
    write_build_timing(timing)

    vision_plan = None

    extra_started = time.monotonic()
    compile_before_extra = build_timing_phase(timing, "trt_compile_s")
    extra_engines = (
        build_extra_engines(
            config,
            weights,
            max_cache_length,
            precision=precision,
            verbose=verbose,
            build_timing=timing,
        )
        or {}
    )
    extra_elapsed = time.monotonic() - extra_started
    add_build_timing(
        timing,
        "trt_compile_s",
        untracked_phase_time(extra_elapsed, compile_before_extra, timing, "trt_compile_s"),
    )
    add_build_timing(timing, "trt_compile_extra_engines_s", extra_elapsed)
    write_build_timing(timing)
    sections.extend(
        BundleSection(section_name, section_data)
        for section_name, section_data in extra_engines.items()
    )

    tokenizer_frame = prepare_tokenizer_special_frame(
        model_path,
        source_model_id_or_path=str(options.get("tokenizer_source_model_id_or_path") or model_path),
        source_revision=(
            str(options["tokenizer_source_revision"])
            if options.get("tokenizer_source_revision")
            else None
        ),
    )

    if tokenizer_frame is None:
        prefix_ids, suffix_ids = [], []
        add_special_tokens = detect_tokenizer_add_special_tokens(model_path)
    else:
        prefix_ids, suffix_ids = tokenizer_frame
        add_special_tokens = bool(prefix_ids or suffix_ids)

    trt_version = tensorrt_version()
    trt_abi_value = tensorrt_abi(trt_version)
    info = BundleInfo(
        model_id=model_path.name,
        model_type=config.model_type,
        family=name,
        trt_version=trt_version,
        trt_abi=trt_abi_value,
        gpu_name=gpu_name(),
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        max_cache_length=max_cache_length,
        runtime_strategy=runtime_strategy,
        precision=precision,
        quantization=(quant_plan.quant_format if quant_plan else "none"),
        tokenizer_add_special_tokens=add_special_tokens,
    )

    source_config = model_path / "config.json"
    runtime_config = (
        json.loads(source_config.read_text(encoding="utf-8"))
        if source_config.is_file()
        else {key: value for key, value in config.raw.items() if not str(key).startswith("_")}
    )
    generation_config = model_path / "generation_config.json"
    if generation_config.is_file():
        generation = json.loads(generation_config.read_text(encoding="utf-8"))
        if "eos_token_id" in generation:
            runtime_config["eos_token_id"] = generation["eos_token_id"]
    runtime_config.update(
        {
            "runtime_strategy": runtime_strategy,
            "engine_backend": "trt_rtx" if options.get("rtx") else "trt",
            "trt_version": trt_version,
            "precision": precision,
            "tokenizer_add_special_tokens": int(add_special_tokens),
            "decoder_engine_layout": decoder_layout,
        }
    )
    if trt_abi_value:
        runtime_config["trt_abi"] = trt_abi_value
    if tokenizer_frame is not None:
        runtime_config["tokenizer_special_prefix_ids"] = prefix_ids
        runtime_config["tokenizer_special_suffix_ids"] = suffix_ids
    if options.get("fp32_layers"):
        runtime_config["fp32_layers"] = sorted(set(options["fp32_layers"]))
    if quant_plan is not None:
        runtime_config["quantization"] = quant_plan.as_config_dict()
    runtime_config.update(parallel.to_bundle_config_fields())
    if vision_plan is not None:
        runtime_config["has_vision_engine"] = True
    overrides = get_bundle_config_overrides(config)
    if overrides is not None:
        merged = dict(overrides)
        merged.update(runtime_config)
        merged.update(overrides)
        runtime_config = merged

    from tensorrt_model_connect.tvm_ffi.graph_build import kernel_slots_section

    slot_section = kernel_slots_section()
    if slot_section is not None:
        sections.append(BundleSection("kernel_slots.json", slot_section))

    sections.append(
        BundleSection("config.json", json.dumps(runtime_config, indent=2).encode("utf-8"))
    )
    for filename in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.model",
        "preprocessor_config.json",
        "processor_config.json",
    ):
        path = model_path / filename
        if path.is_file():
            sections.append(BundleSection(filename, path.read_bytes()))

    kernel_manifest = []
    for global_name, library in options.get("kernel_artifacts") or ():
        section_name = f"kernel_{global_name.replace('.', '_')}.so"
        sections.append(BundleSection(section_name, Path(library).read_bytes()))
        kernel_manifest.append(
            {
                "global_name": global_name,
                "func_name": "run",
                "section": section_name,
            }
        )
    if kernel_manifest:
        sections.append(
            BundleSection(
                "kernel_manifest.json",
                json.dumps({"kernels": kernel_manifest}).encode("utf-8"),
            )
        )

    write_started = time.monotonic()
    write_bundle(output_path, info, sections)
    add_build_timing(timing, "bundle_write_s", time.monotonic() - write_started)
    timing["total_s"] = time.monotonic() - started
    write_build_timing(timing)
