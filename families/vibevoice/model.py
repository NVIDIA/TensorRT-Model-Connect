# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a VibeVoice text-to-speech bundle from the original checkpoint layout."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import tensorrt as trt

from . import config as config_module
from .codec_engines import build_feedback_engine, build_voice_encoder_engine, stream_state_size
from .head_engine import build_head_engine
from .lm_engine import build_embed_engine, build_lm_engine
from .weights import Checkpoint

_TASK = "text_to_speech"
_PRECISIONS = {"bf16": trt.bfloat16, "fp32": trt.float32}
# Positions per sequence when the build does not state max_sequence_length:
# room for a long prompt and about half an hour of speech at 7.5 frames/s.
DEFAULT_CAPACITY = 16384
MIN_CAPACITY = 64
PREFILL_CHUNK = 512
MAX_VOICE_SECONDS = 60
# Generation defaults of the reference (generation_config.json of the Transformers conversion).
DEFAULT_GUIDANCE_SCALE = 1.3
DEFAULT_DIFFUSION_STEPS = 10


def _resolve_tokenizer_dir(model_dir: Path) -> Path:
    """The checkpoint ships no tokenizer; its preprocessor config names the Qwen2.5 one."""
    if all((model_dir / name).is_file() for name in config_module.TOKENIZER_FILES):
        return model_dir
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            config_module.TOKENIZER_REPOSITORY,
            revision=config_module.TOKENIZER_REVISION,
            allow_patterns=config_module.TOKENIZER_FILES,
        )
    )


def _special_token_ids(tokenizer_json: dict) -> dict[str, int]:
    added = {
        item["content"]: int(item["id"])
        for item in tokenizer_json.get("added_tokens", [])
        if item.get("special")
    }
    names = {
        "eos": config_module.EOS_TOKEN,
        "audio_bos": config_module.AUDIO_BOS_TOKEN,
        "audio_eos": config_module.AUDIO_EOS_TOKEN,
        "audio_token": config_module.AUDIO_TOKEN,
    }
    missing = [token for token in names.values() if token not in added]
    if missing:
        raise ValueError("vibevoice tokenizer lacks special tokens: " + ", ".join(missing))
    return {key: added[token] for key, token in names.items()}


def rope_inverse_frequencies(cfg) -> np.ndarray:
    """``1 / theta ** (arange(0, d, 2) / d)`` evaluated in float32, as the reference does."""
    exponents = np.arange(0, cfg.head_dim, 2, dtype=np.int64).astype(np.float32) / np.float32(
        cfg.head_dim
    )
    return (np.float32(1.0) / np.power(np.float32(cfg.rope_theta), exponents)).astype(np.float32)


def train_sigmas(num_train_timesteps: int) -> np.ndarray:
    """sqrt((1 - alpha_bar) / alpha_bar) of the squaredcos_cap_v2 schedule, in float32."""

    def alpha_bar(time: float) -> float:
        return math.cos((time + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = [
        min(
            1 - alpha_bar((i + 1) / num_train_timesteps) / alpha_bar(i / num_train_timesteps), 0.999
        )
        for i in range(num_train_timesteps)
    ]
    alphas = np.float32(1.0) - np.asarray(betas, dtype=np.float64).astype(np.float32)
    # torch.cumprod accumulates float32 in double on the CPU; 1 - alpha_bar cancels
    # badly near t = 0, so a float32 running product would move small sigmas by 1e-4.
    cumulative = np.cumprod(alphas.astype(np.float64)).astype(np.float32)
    return np.sqrt((np.float32(1.0) - cumulative) / cumulative).astype(np.float32)


def sequence_geometry(max_sequence_length, max_position_embeddings: int) -> tuple[int, int]:
    """KV positions per sequence and the prefill chunk, which never exceeds them."""
    capacity = DEFAULT_CAPACITY if max_sequence_length is None else int(max_sequence_length)
    if not MIN_CAPACITY <= capacity <= max_position_embeddings:
        raise ValueError(
            f"vibevoice max_sequence_length must lie in [{MIN_CAPACITY}, "
            f"{max_position_embeddings}], not {capacity}"
        )
    return capacity, min(PREFILL_CHUNK, capacity)


def build(request, writer) -> None:
    """Build one vibevoice bundle: five engines, the tokenizer and runtime.json."""
    if request.task != _TASK:
        raise ValueError(f"vibevoice supports only task={_TASK}")
    if request.backend != "trt":
        raise ValueError("vibevoice supports only backend=trt")
    if request.max_batch_size != 1:
        raise NotImplementedError("vibevoice generates one request at a time (max_batch_size=1)")
    if request.tensor_parallel_size != 1 or request.context_parallel_size != 1:
        raise NotImplementedError("vibevoice runs on one GPU without tensor or context parallelism")
    if request.dynamic_kv_cache:
        raise NotImplementedError("vibevoice does not support dynamic_kv_cache")
    if request.image_height is not None or request.image_width is not None:
        raise NotImplementedError("vibevoice takes text, not an image size")
    if request.video_num_frames is not None:
        raise NotImplementedError("vibevoice takes text, not video")
    if request.quantization not in (None, "none"):
        raise NotImplementedError("vibevoice does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("vibevoice does not support per-layer precision overrides")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("vibevoice has no MTP head")
    if request.graph_transform is not None:
        raise NotImplementedError("vibevoice does not accept graph transforms")
    precision = str(request.precision).lower()
    if precision not in _PRECISIONS:
        raise NotImplementedError(
            f"vibevoice builds bf16 (the checkpoint's type) or fp32, not {precision}"
        )
    dtype = _PRECISIONS[precision]
    model_dir = Path(request.model_dir)
    cfg = config_module.load(model_dir)
    capacity, prefill_chunk = sequence_geometry(
        request.max_sequence_length, cfg.max_position_embeddings
    )
    tokenizer_dir = _resolve_tokenizer_dir(model_dir)
    tokenizer_bytes = (tokenizer_dir / "tokenizer.json").read_bytes()
    special = _special_token_ids(json.loads(tokenizer_bytes))
    control = sorted(special.values())
    weights = Checkpoint(model_dir)
    verbose = bool(request.verbose)
    max_voice_samples = (
        MAX_VOICE_SECONDS * cfg.sample_rate // cfg.acoustic.hop_length * cfg.acoustic.hop_length
    )

    writer.set_header(family="vibevoice", task=request.task, backend=request.backend)
    writer.add_bytes(
        "lm.plan",
        build_lm_engine(
            weights,
            cfg,
            dtype=dtype,
            capacity=capacity,
            prefill_chunk=prefill_chunk,
            control_token_ids=control,
            verbose=verbose,
        ),
    )
    writer.add_bytes(
        "embed.plan",
        build_embed_engine(weights, cfg, dtype=dtype, max_tokens=PREFILL_CHUNK, verbose=verbose),
    )
    writer.add_bytes(
        "diffusion_head.plan", build_head_engine(weights, cfg, dtype=dtype, verbose=verbose)
    )
    writer.add_bytes(
        "feedback.plan", build_feedback_engine(weights, cfg, dtype=dtype, verbose=verbose)
    )
    writer.add_bytes(
        "voice_encoder.plan",
        build_voice_encoder_engine(
            weights, cfg, dtype=dtype, max_samples=max_voice_samples, verbose=verbose
        ),
    )
    writer.add_bytes("tokenizer.json", tokenizer_bytes)
    writer.add_json(
        "runtime.json",
        {
            "precision": precision,
            "hidden_size": cfg.hidden_size,
            "num_layers": cfg.num_layers,
            "num_kv_heads": cfg.num_kv_heads,
            "head_dim": cfg.head_dim,
            "capacity": capacity,
            "prefill_chunk": prefill_chunk,
            "rope_inv_freq": [float(value) for value in rope_inverse_frequencies(cfg)],
            "control_token_ids": control,
            "eos_token_id": special["eos"],
            "audio_bos_token_id": special["audio_bos"],
            "audio_eos_token_id": special["audio_eos"],
            "audio_token_id": special["audio_token"],
            "system_prompt": config_module.SYSTEM_PROMPT,
            "voice_input_header": config_module.VOICE_INPUT_HEADER,
            "text_input_header": config_module.TEXT_INPUT_HEADER,
            "speech_output_header": config_module.SPEECH_OUTPUT_HEADER,
            "audio_bos_token": config_module.AUDIO_BOS_TOKEN,
            "audio_eos_token": config_module.AUDIO_EOS_TOKEN,
            "audio_token": config_module.AUDIO_TOKEN,
            "sample_rate": cfg.sample_rate,
            "hop_length": cfg.acoustic.hop_length,
            "latent_size": cfg.latent_size,
            "target_dbfs": cfg.target_dbfs,
            "normalize_eps": cfg.normalize_eps,
            "max_voice_samples": max_voice_samples,
            "stream_state_size": stream_state_size(cfg),
            "train_sigmas": [float(value) for value in train_sigmas(cfg.ddpm_num_steps)],
            "default_guidance_scale": DEFAULT_GUIDANCE_SCALE,
            "default_diffusion_steps": DEFAULT_DIFFUSION_STEPS,
        },
    )
