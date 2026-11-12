# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build one Stable Audio Open text-to-audio bundle.

Three engines go into the bundle — the conditioning network (T5 encoder plus the
duration conditioners), the DiT denoiser and the Oobleck decoder — plus the T5
tokenizer and the scheduler constants the runtime needs to run the sampling loop.
"""

from __future__ import annotations

from pathlib import Path

import tensorrt as trt
from safetensors import safe_open

from . import conditioner_builder, dit_builder, vae_builder
from . import config as config_module

_PRECISIONS = {"fp32": trt.float32, "fp16": trt.float16}
# StableAudioPipeline.__call__ defaults.
_DEFAULT_STEPS = 100
_DEFAULT_GUIDANCE = 7.0


def _load_weights(path: Path, prefix: str = "") -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"stable_audio checkpoint is missing {path}")
    weights = {}
    with safe_open(str(path), framework="numpy") as reader:
        for key in reader.keys():
            if key.startswith(prefix):
                weights[key] = reader.get_tensor(key)
    return weights


def _compile(populate, *, verbose: bool, profile: dict | None = None, tf32: bool = False) -> bytes:
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    builder_config = builder.create_builder_config()
    builder_config.builder_optimization_level = 3
    if tf32:
        # FP16 engines multiply FP16 operands as FP32 (Graph.linear / conv1d /
        # conv_transpose1d): TF32 tensor cores form those products exactly and accumulate
        # them in FP32.
        builder_config.set_flag(trt.BuilderFlag.TF32)
    else:
        # FP32 engines stay FP32 throughout.
        builder_config.clear_flag(trt.BuilderFlag.TF32)
    graph = populate(network)
    if profile:
        optimization_profile = builder.create_optimization_profile()
        for name, shapes in profile.items():
            optimization_profile.set_shape(name, *shapes)
        builder_config.add_optimization_profile(optimization_profile)
    plan = builder.build_serialized_network(network, builder_config)
    del graph
    if plan is None:
        raise RuntimeError("stable_audio TensorRT engine build failed")
    return bytes(plan)


def _denoiser_profile(cfg: dict) -> dict:
    """Batch 1 (no guidance) to 2 (the unconditional and conditional branches)."""
    dit = cfg["transformer"]
    tokens = cfg["tokenizer"]["max_length"] + 2

    def shapes(*rest):
        return tuple((batch, *rest) for batch in (1, 2, 2))

    return {
        "hidden_states": shapes(dit["in_channels"], dit["sample_size"]),
        "encoder_hidden_states": shapes(tokens, dit["cross_attention_input_dim"]),
        "global_hidden_states": shapes(1, dit["global_states_input_dim"]),
    }


def _conditioner_plan(model_dir: Path, cfg: dict, verbose: bool) -> bytes:
    text = _load_weights(model_dir / "text_encoder" / "model.safetensors")
    projection = _load_weights(model_dir / "projection_model" / "diffusion_pytorch_model.safetensors")
    return _compile(
        lambda network: conditioner_builder.populate(network, text, projection, cfg), verbose=verbose
    )


def _denoiser_plan(model_dir: Path, cfg: dict, work: trt.DataType, verbose: bool) -> bytes:
    weights = _load_weights(model_dir / "transformer" / "diffusion_pytorch_model.safetensors")
    return _compile(
        lambda network: dit_builder.populate(network, weights, cfg, work),
        verbose=verbose,
        profile=_denoiser_profile(cfg),
        tf32=work == trt.float16,
    )


def _decoder_plan(model_dir: Path, cfg: dict, work: trt.DataType, verbose: bool) -> bytes:
    weights = _load_weights(model_dir / "vae" / "diffusion_pytorch_model.safetensors", "decoder.")
    return _compile(lambda network: vae_builder.populate(network, weights, cfg, work), verbose=verbose,
                    tf32=work == trt.float16)


def _runtime_config(cfg: dict, precision: str) -> dict:
    dit, vae, tokenizer, projection = cfg["transformer"], cfg["vae"], cfg["tokenizer"], cfg["projection"]
    return {
        "precision": precision,
        "sample_rate": vae["sampling_rate"],
        "audio_channels": vae["audio_channels"],
        "audio_samples": cfg["audio_samples"],
        "hop_length": vae["hop_length"],
        "latent_channels": dit["in_channels"],
        "latent_length": dit["sample_size"],
        "text_length": tokenizer["max_length"],
        "pad_token_id": tokenizer["pad_id"],
        "eos_token_id": tokenizer["eos_id"],
        "encoder_tokens": tokenizer["max_length"] + 2,
        "conditioning_dim": projection["conditioning_dim"],
        "global_dim": dit["global_states_input_dim"],
        # The duration conditioners' range, which StableAudioPipeline.check_inputs enforces.
        "duration_min": projection["min_value"],
        "duration_max": projection["max_value"],
        "default_num_inference_steps": _DEFAULT_STEPS,
        "default_guidance_scale": _DEFAULT_GUIDANCE,
        "scheduler": cfg["scheduler"],
    }


def build(request, writer) -> None:
    """Build one Stable Audio Open ``text_to_audio`` bundle."""
    if request.task != "text_to_audio":
        raise ValueError("stable_audio supports only task=text_to_audio")
    if request.backend != "trt":
        raise ValueError("stable_audio supports only backend=trt")
    precision = str(request.precision).lower()
    if precision not in _PRECISIONS:
        raise ValueError(f"stable_audio supports precision fp32 or fp16, got {request.precision!r}")
    if request.max_batch_size != 1:
        raise NotImplementedError("stable_audio generates one waveform per request (max_batch_size=1)")
    if request.tensor_parallel_size != 1 or request.context_parallel_size != 1:
        raise NotImplementedError("stable_audio runs on one GPU without tensor or context parallelism")
    if request.quantization not in (None, "none"):
        raise NotImplementedError("stable_audio does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("stable_audio does not support per-layer FP32 overrides")
    if request.dynamic_kv_cache:
        raise NotImplementedError("stable_audio does not support dynamic_kv_cache")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("stable_audio does not support MTP speculative decoding")
    if request.image_height is not None or request.image_width is not None:
        raise NotImplementedError("stable_audio does not take an image size")
    if request.video_num_frames is not None:
        raise NotImplementedError("stable_audio does not take a frame count")

    model_dir = Path(request.model_dir)
    cfg = config_module.resolve(model_dir)
    if request.max_sequence_length not in (None, cfg["tokenizer"]["max_length"]):
        # The text length is the tokenizer's padded length the denoiser was trained on.
        raise NotImplementedError(
            f"stable_audio text length is fixed at {cfg['tokenizer']['max_length']} tokens")
    verbose = bool(request.verbose)
    work = _PRECISIONS[precision]

    writer.set_header(family="stable_audio", task=request.task, backend=request.backend)
    writer.add_bytes("conditioner.plan", _conditioner_plan(model_dir, cfg, verbose))
    writer.add_bytes("transformer.plan", _denoiser_plan(model_dir, cfg, work, verbose))
    writer.add_bytes("vae_decoder.plan", _decoder_plan(model_dir, cfg, work, verbose))
    writer.add_bytes("tokenizer.json", (model_dir / "tokenizer" / "tokenizer.json").read_bytes())
    writer.add_json("runtime.json", _runtime_config(cfg, precision))
