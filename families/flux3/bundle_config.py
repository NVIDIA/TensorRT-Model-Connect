# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint validation, geometry and ``runtime.json`` for flux3 bundles.

The values mirror the Diffusers ``Flux3DistilledModularPipeline`` text-to-image
path: Qwen3-VL text frame without a system turn, a text window padded to a
multiple of 80 tokens, eight stacked hidden states, a constant timestep shift,
16 flow-map steps and BF16 latents between steps.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RUNTIME_FORMAT = "flux3-runtime-v1"
DEFAULT_SIDE = 1024
SIDE_MULTIPLE = 16
MIN_SIDE = 64
MAX_SIDE = 4096
MAX_PIXELS = 4 * 1024 * 1024
PAD_MULTIPLE = 80
MAX_TEXT_TOKENS = 8192
DEFAULT_TEXT_TOKENS = 4000
TEXT_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)
TIMESTEP_CHANNELS = 256
TIMESTEP_MAX_PERIOD = 10000
NUM_TRAIN_TIMESTEPS = 1000
DISTILLED_STEPS = 16
TIMESTEP_SHIFT = 6.93
IMAGE_ROPE_SCALE = 0.5
_PROMPT_SENTINEL = "\u0000flux3-prompt\u0000"

_TRANSFORMER = {
    "attention_head_dim": 128,
    "axes_dims_rope": [32, 32, 32, 32],
    "cond_in_channels": 128,
    "eps": 1e-06,
    "flow_map": True,
    "in_channels": 128,
    "joint_attention_dim": 20480,
    "mlp_ratio": 3.0,
    "num_attention_heads": 24,
    "rope_theta": 10000,
    "timestep_channels": TIMESTEP_CHANNELS,
}
_TEXT = {
    "hidden_act": "silu",
    "hidden_size": 2560,
    "head_dim": 128,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 9728,
    "rms_norm_eps": 1e-06,
    "attention_bias": False,
    "vocab_size": 151936,
}
_VAE = {
    "act_fn": "silu",
    "latent_channels": 32,
    "patch_size": [2, 2],
    "norm_num_groups": 32,
    "layers_per_block": 2,
    "mid_block_add_attention": True,
    "use_post_quant_conv": True,
    "out_channels": 3,
}


@dataclass(frozen=True)
class Flux3Checkpoint:
    """Validated component configs of one FLUX 3 Image Klein snapshot."""

    model_dir: Path
    transformer: dict[str, Any]
    text: dict[str, Any]
    vae: dict[str, Any]
    text_layers: tuple[int, ...]

    @property
    def hidden_size(self) -> int:
        return int(self.transformer["num_attention_heads"]) * int(
            self.transformer["attention_head_dim"]
        )

    @property
    def vae_block_out_channels(self) -> tuple[int, ...]:
        channels = self.vae.get("decoder_block_out_channels") or self.vae["block_out_channels"]
        return tuple(int(value) for value in channels)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"flux3 cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"flux3 expects a JSON object in {path}")
    return value


def _require(config: dict[str, Any], expected: dict[str, Any], where: str) -> None:
    for key, value in expected.items():
        actual = config.get(key)
        if isinstance(value, float):
            matches = isinstance(actual, (int, float)) and math.isclose(actual, value)
        else:
            matches = actual == value
        if not matches:
            raise ValueError(f"flux3 {where}: {key}={actual!r}, expected {value!r}")


def load_checkpoint(model_dir: str | Path) -> Flux3Checkpoint:
    """Read and validate the configs of a Diffusers-layout FLUX 3 Image Klein snapshot."""

    root = Path(model_dir)
    package = _read_json(root / "config.json")
    if package.get("model") != "flux-3-image-klein" or package.get("format_version") != 1:
        raise ValueError("flux3 supports the black-forest-labs/flux-3-image-klein package only")
    dit = package.get("dit") or {}
    sampler = package.get("sampler") or {}
    text_package = package.get("text_encoder") or {}
    if dit.get("flow_map") is not True or sampler.get("guidance") is not None:
        raise ValueError(
            "flux3 supports the step-distilled (flow-map, guidance-free) checkpoint only"
        )
    if not math.isclose(float(dit.get("image_rope_scale", 0.0)), IMAGE_ROPE_SCALE):
        raise ValueError("flux3 expects image_rope_scale 0.5")
    if not math.isclose(float(sampler.get("alpha", 0.0)), TIMESTEP_SHIFT):
        raise ValueError(f"flux3 expects the sampler shift {TIMESTEP_SHIFT}")
    if int(sampler.get("steps", 0)) != DISTILLED_STEPS:
        raise ValueError(f"flux3 expects {DISTILLED_STEPS} distilled steps")
    if int(text_package.get("pad_multiple", 0)) != PAD_MULTIPLE:
        raise ValueError(f"flux3 expects a text pad multiple of {PAD_MULTIPLE}")
    layers = tuple(int(value) for value in text_package.get("layers", ()))
    if layers != TEXT_LAYERS:
        raise ValueError(f"flux3 expects text encoder layers {TEXT_LAYERS}, found {layers}")

    index = _read_json(root / "modular_model_index.json")
    if index.get("_class_name") != "Flux3DistilledModularPipeline":
        raise ValueError("flux3 expects a Flux3DistilledModularPipeline repository")
    transformer = _read_json(root / "transformer" / "config.json")
    if transformer.get("_class_name") != "Flux3Transformer2DModel":
        raise ValueError("flux3 expects transformer class Flux3Transformer2DModel")
    _require(transformer, _TRANSFORMER, "transformer config")
    text_config = _read_json(root / "text_encoder" / "config.json")
    if text_config.get("model_type") != "qwen3_vl":
        raise ValueError("flux3 expects a Qwen3-VL text encoder")
    text = text_config.get("text_config") or {}
    _require(text, _TEXT, "text encoder config")
    rope = text.get("rope_scaling") or {}
    if rope.get("rope_type", "default") != "default" or sum(
        rope.get("mrope_section", [])
    ) * 2 != int(text["head_dim"]):
        raise ValueError("flux3 expects default Qwen3-VL M-RoPE covering the head dimension")
    if int(text["num_hidden_layers"]) < max(TEXT_LAYERS):
        raise ValueError("flux3 text encoder has fewer layers than the stacked hidden states need")
    vae = _read_json(root / "vae" / "config.json")
    if vae.get("_class_name") != "AutoencoderKLFlux2":
        raise ValueError("flux3 expects an AutoencoderKLFlux2 autoencoder")
    _require(vae, _VAE, "VAE config")
    expected_joint = int(text["hidden_size"]) * len(TEXT_LAYERS)
    if int(transformer["joint_attention_dim"]) != expected_joint:
        raise ValueError("flux3 transformer text width does not match the stacked hidden states")
    if int(transformer["in_channels"]) != int(vae["latent_channels"]) * 4:
        raise ValueError("flux3 transformer channels do not match the 2x2-packed VAE latents")
    return Flux3Checkpoint(
        model_dir=root, transformer=transformer, text=text, vae=vae, text_layers=layers
    )


def image_geometry(height: int | None, width: int | None) -> tuple[int, int]:
    """Validate the generated image size (pixels)."""

    height = DEFAULT_SIDE if height is None else int(height)
    width = DEFAULT_SIDE if width is None else int(width)
    for name, value in (("image_height", height), ("image_width", width)):
        if value % SIDE_MULTIPLE or not MIN_SIDE <= value <= MAX_SIDE:
            raise ValueError(
                f"flux3 {name} must be a multiple of {SIDE_MULTIPLE} in [{MIN_SIDE}, {MAX_SIDE}]"
            )
    if height * width > MAX_PIXELS:
        raise ValueError(f"flux3 images are limited to {MAX_PIXELS} pixels")
    return height, width


def text_token_budget(max_sequence_length: int | None) -> int:
    """Largest text window the engines accept (a multiple of the pad multiple)."""

    budget = DEFAULT_TEXT_TOKENS if max_sequence_length is None else int(max_sequence_length)
    if budget % PAD_MULTIPLE or not PAD_MULTIPLE <= budget <= MAX_TEXT_TOKENS:
        raise ValueError(
            f"flux3 max_sequence_length must be a multiple of {PAD_MULTIPLE} "
            f"in [{PAD_MULTIPLE}, {MAX_TEXT_TOKENS}]"
        )
    return budget


def text_window(real_tokens: int, budget: int) -> int:
    """Padded text window length for ``real_tokens`` framed prompt tokens."""

    if real_tokens < 1:
        raise ValueError("flux3 prompt frame is empty")
    if real_tokens > budget:
        raise ValueError(f"flux3 prompt needs {real_tokens} tokens; the bundle holds {budget}")
    return -(-real_tokens // PAD_MULTIPLE) * PAD_MULTIPLE


def render_prompt_frame(model_dir: str | Path) -> tuple[str, str]:
    """Render the checkpoint's own chat template around a user prompt.

    Returns the text before and after the prompt for
    ``apply_chat_template([{"role": "user", "content": prompt}],
    add_generation_prompt=True)``, which is how the reference encodes prompts.
    """

    import jinja2
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    path = Path(model_dir) / "text_encoder" / "chat_template.json"
    template_text = _read_json(path).get("chat_template")
    if not isinstance(template_text, str):
        raise ValueError(f"flux3 expects a chat_template string in {path}")

    def raise_exception(message: str) -> None:
        raise jinja2.exceptions.TemplateError(message)

    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    environment.globals["raise_exception"] = raise_exception
    rendered = environment.from_string(template_text).render(
        messages=[{"role": "user", "content": _PROMPT_SENTINEL}],
        add_generation_prompt=True,
    )
    if rendered.count(_PROMPT_SENTINEL) != 1:
        raise ValueError("flux3 chat template must place the user prompt exactly once")
    prefix, suffix = rendered.split(_PROMPT_SENTINEL)
    return prefix, suffix


def pad_token_id(model_dir: str | Path) -> int:
    """Id of the tokenizer's pad token (the reference right-pads with it)."""

    root = Path(model_dir) / "text_encoder"
    pad = _read_json(root / "tokenizer_config.json").get("pad_token")
    if isinstance(pad, dict):
        pad = pad.get("content")
    tokens = _read_json(root / "tokenizer.json").get("added_tokens", [])
    ids = [token.get("id") for token in tokens if token.get("content") == pad]
    if not isinstance(pad, str) or len(ids) != 1:
        raise ValueError("flux3 cannot resolve the tokenizer pad token")
    return int(ids[0])


def runtime_config(
    *,
    height: int,
    width: int,
    text_budget: int,
    prompt_prefix: str,
    prompt_suffix: str,
    pad_id: int,
    embed_dim: int,
    latent_channels: int,
) -> dict[str, Any]:
    """The ``runtime.json`` section read by the native pipeline."""

    return {
        "format": RUNTIME_FORMAT,
        "height": height,
        "width": width,
        "latent_height": height // SIDE_MULTIPLE,
        "latent_width": width // SIDE_MULTIPLE,
        "latent_channels": latent_channels,
        "text": {
            "prompt_prefix": prompt_prefix,
            "prompt_suffix": prompt_suffix,
            "pad_token_id": pad_id,
            "pad_multiple": PAD_MULTIPLE,
            "max_tokens": text_budget,
            "embed_dim": embed_dim,
        },
        "schedule": {
            "default_steps": DISTILLED_STEPS,
            "shift": TIMESTEP_SHIFT,
            "num_train_timesteps": NUM_TRAIN_TIMESTEPS,
        },
        "timestep_embedding": {"dim": TIMESTEP_CHANNELS, "max_period": TIMESTEP_MAX_PERIOD},
    }
