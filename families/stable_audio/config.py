# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read and check the diffusers ``StableAudioPipeline`` component configs this family builds.

Every value the engines or the runtime depend on is read here, checked against the
other components, and rejected when this family does not implement it. Defaults are
the ones the diffusers / transformers classes apply when a config stays silent.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

_COMPONENTS = {
    "transformer": ("diffusers", "StableAudioDiTModel"),
    "vae": ("diffusers", "AutoencoderOobleck"),
    "text_encoder": ("transformers", "T5EncoderModel"),
    "projection_model": ("stable_audio", "StableAudioProjectionModel"),
    "scheduler": ("diffusers", "CosineDPMSolverMultistepScheduler"),
}
_TOKENIZER_CLASSES = {"T5Tokenizer", "T5TokenizerFast"}
# CosineDPMSolverMultistepScheduler maps these legacy names onto "midpoint".
_MIDPOINT_ALIASES = {"logrho", "bh1", "bh2"}


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"stable_audio checkpoint is missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _integral(value, name: str) -> int:
    """An integer that a config may spell as a float (``sample_size`` is ``1024.0``)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"stable_audio config {name} must be a number, got {value!r}")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"stable_audio config {name} must be integral, got {value!r}")
    return int(value)


def _positive(value, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"stable_audio config {name} must be positive, got {value!r}")
    return number


def _check_index(index: dict) -> None:
    if index.get("_class_name") != "StableAudioPipeline":
        raise ValueError("stable_audio builds only a diffusers StableAudioPipeline checkpoint")
    for component, expected in _COMPONENTS.items():
        declared = index.get(component)
        if not isinstance(declared, list) or tuple(declared) != expected:
            raise ValueError(
                f"stable_audio expects {component}={list(expected)}, the checkpoint declares {declared!r}")
    tokenizer = index.get("tokenizer")
    if (not isinstance(tokenizer, list) or len(tokenizer) != 2 or tokenizer[0] != "transformers"
            or tokenizer[1] not in _TOKENIZER_CLASSES):
        raise ValueError(f"stable_audio expects a T5 tokenizer, the checkpoint declares {tokenizer!r}")


def _transformer(raw: dict) -> dict:
    heads = _integral(raw.get("num_attention_heads", 24), "num_attention_heads")
    kv_heads = _integral(raw.get("num_key_value_attention_heads", 12), "num_key_value_attention_heads")
    head_dim = _integral(raw.get("attention_head_dim", 64), "attention_head_dim")
    cfg = {
        "sample_size": _integral(raw.get("sample_size", 1024), "sample_size"),
        "in_channels": _integral(raw.get("in_channels", 64), "in_channels"),
        "out_channels": _integral(raw.get("out_channels", 64), "out_channels"),
        "num_layers": _integral(raw.get("num_layers", 24), "num_layers"),
        "num_heads": heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "inner_dim": heads * head_dim,
        "cross_attention_dim": _integral(raw.get("cross_attention_dim", 768), "cross_attention_dim"),
        "cross_attention_input_dim": _integral(
            raw.get("cross_attention_input_dim", 768), "cross_attention_input_dim"),
        "global_states_input_dim": _integral(
            raw.get("global_states_input_dim", 1536), "global_states_input_dim"),
        "time_proj_dim": _integral(raw.get("time_proj_dim", 256), "time_proj_dim"),
        # The pipeline rotates the first half of every head (rotary_embed_dim =
        # attention_head_dim // 2) and rotates halves of that, so both must be even.
        "rotary_dim": head_dim // 2,
    }
    if min(cfg.values()) <= 0:
        raise ValueError("stable_audio transformer config has a non-positive dimension")
    if head_dim % 4:
        raise ValueError("stable_audio needs attention_head_dim divisible by 4 for its partial rotary")
    if heads % kv_heads:
        raise ValueError("stable_audio needs num_attention_heads divisible by the key/value heads")
    if cfg["time_proj_dim"] % 2:
        raise ValueError("stable_audio needs an even time_proj_dim")
    if cfg["in_channels"] != cfg["out_channels"]:
        raise ValueError("stable_audio needs a denoiser whose input and output widths agree")
    return cfg


def _vae(raw: dict) -> dict:
    ratios = [_integral(value, "downsampling_ratios") for value in raw.get("downsampling_ratios", [2, 4, 4, 8, 8])]
    multiples = [_integral(value, "channel_multiples") for value in raw.get("channel_multiples", [1, 2, 4, 8, 16])]
    if not ratios or len(ratios) != len(multiples) or min(ratios) <= 0 or min(multiples) <= 0:
        raise ValueError("stable_audio vae downsampling_ratios and channel_multiples must pair up")
    if any(ratio % 2 for ratio in ratios):
        # An odd stride makes ConvTranspose1d(2s, s, ceil(s / 2)) one sample short of s x the
        # input, and the decoder output is then no longer latent_length x hop_length.
        raise NotImplementedError(f"stable_audio supports even Oobleck strides only, got {ratios}")
    cfg = {
        "decoder_channels": _integral(raw.get("decoder_channels", 128), "decoder_channels"),
        "latent_channels": _integral(raw.get("decoder_input_channels", 64), "decoder_input_channels"),
        "audio_channels": _integral(raw.get("audio_channels", 2), "audio_channels"),
        "sampling_rate": _integral(raw.get("sampling_rate", 44100), "sampling_rate"),
        "upsampling_ratios": list(reversed(ratios)),
        "channel_multiples": [1, *multiples],
        "hop_length": int(math.prod(ratios)),
    }
    if min(cfg["decoder_channels"], cfg["latent_channels"], cfg["audio_channels"], cfg["sampling_rate"]) <= 0:
        raise ValueError("stable_audio vae config has a non-positive dimension")
    return cfg


def _text_encoder(raw: dict) -> dict:
    activation = str(raw.get("feed_forward_proj", "relu"))
    if activation != "relu":
        raise NotImplementedError(
            f"stable_audio builds the T5 encoder with a ReLU feed-forward, not {activation!r}")
    if raw.get("model_type", "t5") != "t5":
        raise ValueError(f"stable_audio expects a T5 text encoder, not {raw.get('model_type')!r}")
    cfg = {
        "d_model": _integral(raw.get("d_model", 512), "d_model"),
        "d_kv": _integral(raw.get("d_kv", 64), "d_kv"),
        "d_ff": _integral(raw.get("d_ff", 2048), "d_ff"),
        "num_layers": _integral(raw.get("num_layers", 6), "num_layers"),
        "num_heads": _integral(raw.get("num_heads", 8), "num_heads"),
        "num_buckets": _integral(raw.get("relative_attention_num_buckets", 32), "relative_attention_num_buckets"),
        "max_distance": _integral(raw.get("relative_attention_max_distance", 128),
                                  "relative_attention_max_distance"),
        "eps": _positive(raw.get("layer_norm_epsilon", 1e-6), "layer_norm_epsilon"),
        "vocab_size": _integral(raw.get("vocab_size", 32128), "vocab_size"),
    }
    if min(cfg.values()) <= 0:
        raise ValueError("stable_audio text encoder config has a non-positive dimension")
    return cfg


def _projection(raw: dict) -> dict:
    cfg = {
        "text_dim": _integral(raw["text_encoder_dim"], "text_encoder_dim"),
        "conditioning_dim": _integral(raw["conditioning_dim"], "conditioning_dim"),
        "min_value": float(raw["min_value"]),
        "max_value": float(raw["max_value"]),
    }
    if not cfg["max_value"] > cfg["min_value"]:
        raise ValueError("stable_audio projection max_value must exceed min_value")
    # StableAudioProjectionModel uses an identity text projection when the widths agree.
    cfg["text_projection"] = cfg["text_dim"] != cfg["conditioning_dim"]
    return cfg


def _scheduler(raw: dict) -> dict:
    solver_type = str(raw.get("solver_type", "midpoint"))
    if solver_type in _MIDPOINT_ALIASES:
        solver_type = "midpoint"
    cfg = {
        "sigma_min": _positive(raw.get("sigma_min", 0.3), "sigma_min"),
        "sigma_max": _positive(raw.get("sigma_max", 500), "sigma_max"),
        "sigma_data": _positive(raw.get("sigma_data", 1.0), "sigma_data"),
        "sigma_schedule": str(raw.get("sigma_schedule", "exponential")),
        "solver_order": _integral(raw.get("solver_order", 2), "solver_order"),
        "prediction_type": str(raw.get("prediction_type", "v_prediction")),
        "rho": _positive(raw.get("rho", 7.0), "rho"),
        "solver_type": solver_type,
        "lower_order_final": bool(raw.get("lower_order_final", True)),
        "euler_at_final": bool(raw.get("euler_at_final", False)),
        "final_sigmas_type": str(raw.get("final_sigmas_type", "zero")),
    }
    if cfg["sigma_schedule"] not in {"exponential", "karras"}:
        raise NotImplementedError(f"stable_audio sigma_schedule {cfg['sigma_schedule']!r}")
    if cfg["solver_order"] not in {1, 2}:
        raise NotImplementedError(f"stable_audio solver_order {cfg['solver_order']}")
    if cfg["prediction_type"] not in {"v_prediction", "epsilon"}:
        raise NotImplementedError(f"stable_audio prediction_type {cfg['prediction_type']!r}")
    if cfg["solver_type"] not in {"midpoint", "heun"}:
        raise NotImplementedError(f"stable_audio solver_type {cfg['solver_type']!r}")
    if cfg["final_sigmas_type"] != "zero":
        # "sigma_min" ends on the noise level the schedule already ends with, where the SDE
        # update takes the square root of a non-positive number (diffusers returns NaN).
        raise NotImplementedError(f"stable_audio final_sigmas_type {cfg['final_sigmas_type']!r}")
    if not cfg["sigma_max"] > cfg["sigma_min"]:
        raise ValueError("stable_audio scheduler sigma_max must exceed sigma_min")
    return cfg


def _tokenizer(model_dir: Path) -> dict:
    directory = model_dir / "tokenizer"
    if not (directory / "tokenizer.json").is_file():
        raise FileNotFoundError(f"stable_audio needs {directory / 'tokenizer.json'}")
    settings = _read_json(directory / "tokenizer_config.json")
    length = _integral(settings.get("model_max_length", 0), "model_max_length")
    if not 1 < length <= 4096:
        raise ValueError(f"stable_audio tokenizer model_max_length {length} is not a padded text length")
    tokens = {}
    for entry in _read_json(directory / "tokenizer.json").get("added_tokens", []):
        tokens[str(entry.get("content"))] = int(entry["id"])
    for name in ("<pad>", "</s>"):
        if name not in tokens:
            raise ValueError(f"stable_audio tokenizer.json does not define {name}")
    return {"max_length": length, "pad_id": tokens["<pad>"], "eos_id": tokens["</s>"]}


def resolve(model_dir: str | Path) -> dict:
    """Return the checked subset of the component configs the builders and runtime need."""
    model_dir = Path(model_dir)
    _check_index(_read_json(model_dir / "model_index.json"))
    transformer = _transformer(_read_json(model_dir / "transformer" / "config.json"))
    vae = _vae(_read_json(model_dir / "vae" / "config.json"))
    text = _text_encoder(_read_json(model_dir / "text_encoder" / "config.json"))
    projection = _projection(_read_json(model_dir / "projection_model" / "config.json"))
    scheduler = _scheduler(_read_json(model_dir / "scheduler" / "scheduler_config.json"))
    tokenizer = _tokenizer(model_dir)

    if projection["text_dim"] != text["d_model"]:
        raise ValueError("stable_audio projection text_encoder_dim must equal the T5 d_model")
    if transformer["cross_attention_input_dim"] != projection["conditioning_dim"]:
        raise ValueError("stable_audio denoiser cross-attention input must equal conditioning_dim")
    if transformer["global_states_input_dim"] != 2 * projection["conditioning_dim"]:
        raise ValueError("stable_audio global states must be the start and end duration embeddings")
    if transformer["in_channels"] != vae["latent_channels"]:
        raise ValueError("stable_audio denoiser and vae disagree on the latent width")
    return {
        "transformer": transformer,
        "vae": vae,
        "text_encoder": text,
        "projection": projection,
        "scheduler": scheduler,
        "tokenizer": tokenizer,
        # Every request denoises the whole latent window; the pipeline crops afterwards.
        "audio_samples": transformer["sample_size"] * vae["hop_length"],
    }
