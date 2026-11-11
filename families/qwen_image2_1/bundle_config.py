# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static geometry, prompt layout and scheduler settings of a Qwen-Image 2.1 bundle.

Everything here is read from the checkpoint or fixed by the reference pipeline
(``QwenImage21Pipeline``); the native runtime reads the result from ``runtime.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .weights import read_json

PIPELINE_CLASS = "QwenImage21Pipeline"

# The pipeline's fixed prompt frame. The transformer never sees the system turn: its rows are
# dropped from the text encoder output.
SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."
PROMPT_PREFIX = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"

DEFAULT_IMAGE_SIZE = 1024
MIN_IMAGE_SIZE = 256
DEFAULT_TEXT_TOKENS = 512
MAX_TEXT_TOKENS = 4096
DEFAULT_NUM_STEPS = 40


@dataclass(frozen=True)
class Geometry:
    image_height: int
    image_width: int
    latent_height: int
    latent_width: int
    text_tokens: int

    @property
    def image_tokens(self) -> int:
        return self.latent_height * self.latent_width


def resolve_geometry(
    image_height: int | None,
    image_width: int | None,
    max_sequence_length: int | None,
    *,
    vae_scale: int,
    max_image_tokens: int,
    frame_tokens: int,
) -> Geometry:
    """Validate the requested static output size and text budget.

    The pipeline works on sizes that are multiples of twice the VAE scale (one vision-language
    slot covers 2 x 2 latent tokens). The family builds up to ``max_image_tokens`` latent tokens,
    the largest image the scheduler's shift is calibrated for. The text budget also holds the
    ``frame_tokens`` tokens that frame the prompt, so it has to be larger than that.
    """
    height = DEFAULT_IMAGE_SIZE if image_height is None else int(image_height)
    width = DEFAULT_IMAGE_SIZE if image_width is None else int(image_width)
    alignment = 2 * vae_scale
    for label, value in (("image_height", height), ("image_width", width)):
        if value < MIN_IMAGE_SIZE or value % alignment:
            raise ValueError(
                f"qwen_image2_1 {label} must be a multiple of {alignment} and at least "
                f"{MIN_IMAGE_SIZE} (got {value})"
            )
    latent_height = height // vae_scale
    latent_width = width // vae_scale
    if latent_height * latent_width > max_image_tokens:
        raise ValueError(
            f"qwen_image2_1 image size {height}x{width} needs {latent_height * latent_width} "
            f"latent tokens; this family builds at most {max_image_tokens}, the checkpoint's "
            "max_image_seq_len"
        )
    text_tokens = DEFAULT_TEXT_TOKENS if max_sequence_length is None else int(max_sequence_length)
    if not frame_tokens < text_tokens <= MAX_TEXT_TOKENS:
        raise ValueError(
            f"qwen_image2_1 max_sequence_length must be in [{frame_tokens + 1}, {MAX_TEXT_TOKENS}]: "
            f"it also holds the {frame_tokens} tokens that frame the prompt (got {text_tokens})"
        )
    return Geometry(height, width, latent_height, latent_width, text_tokens)


def scheduler_settings(model_dir: Path) -> dict:
    """Read the flow-match schedule the pipeline applies; reject variants it does not use."""
    index = read_json(model_dir / "model_index.json")
    if index.get("_class_name") != PIPELINE_CLASS:
        raise ValueError(f"qwen_image2_1 requires a {PIPELINE_CLASS} checkpoint")
    if index.get("sample_sigmas") is not None:
        raise ValueError("qwen_image2_1 does not support checkpoints with configured sample_sigmas")
    raw = read_json(model_dir / "scheduler" / "scheduler_config.json")
    if raw.get("_class_name") != "FlowMatchEulerDiscreteScheduler":
        raise ValueError(f"unsupported scheduler class {raw.get('_class_name')!r}")
    unsupported = (
        not raw.get("use_dynamic_shifting")
        or raw.get("time_shift_type") != "exponential"
        or raw.get("use_karras_sigmas")
        or raw.get("use_exponential_sigmas")
        or raw.get("use_beta_sigmas")
        or raw.get("invert_sigmas")
        or raw.get("stochastic_sampling")
    )
    if unsupported:
        raise ValueError("qwen_image2_1 supports the exponential dynamic-shift Euler schedule only")
    # The pipeline divides the scheduler's timestep by the literal 1000 before the transformer
    # multiplies it by 1000 again; the runtime relies on the two factors being the same.
    if int(raw["num_train_timesteps"]) != 1000:
        raise ValueError("qwen_image2_1 requires a scheduler with num_train_timesteps = 1000")
    return {
        "num_train_timesteps": int(raw["num_train_timesteps"]),
        "base_shift": float(raw["base_shift"]),
        "max_shift": float(raw["max_shift"]),
        "base_image_seq_len": int(raw["base_image_seq_len"]),
        "max_image_seq_len": int(raw["max_image_seq_len"]),
        "shift_terminal": float(raw["shift_terminal"] or 0.0),
    }


def prompt_layout(processor_dir: Path) -> dict:
    """Count the system-turn tokens the pipeline drops, with the checkpoint's own tokenizer.

    The pipeline derives the count from the processor's chat template rather than hard-coding it.
    The frame the runtime tokenizes must start with exactly those tokens. ``frame_tokens`` is
    what the rest of the frame (user header and assistant header) adds to every prompt.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        processor_dir, local_files_only=True, trust_remote_code=False
    )
    system = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    system_ids = tokenizer.encode(tokenizer.apply_chat_template(system, tokenize=False))
    frame = PROMPT_PREFIX + "a photo" + PROMPT_SUFFIX
    frame_ids = tokenizer.encode(frame)
    if frame_ids != tokenizer.encode(frame, add_special_tokens=False):
        raise ValueError("the checkpoint's tokenizer adds special tokens the runtime does not")
    if not system_ids or frame_ids[: len(system_ids)] != system_ids:
        raise ValueError("the checkpoint's chat template does not render the pipeline's system turn")
    if tokenizer.pad_token_id is None:
        raise ValueError("the checkpoint's tokenizer defines no padding token")
    return {
        "prefix_tokens": len(system_ids),
        "frame_tokens": len(tokenizer.encode(PROMPT_PREFIX + PROMPT_SUFFIX)) - len(system_ids),
        "pad_token_id": int(tokenizer.pad_token_id),
        "prompt_prefix": PROMPT_PREFIX,
        "prompt_suffix": PROMPT_SUFFIX,
    }
