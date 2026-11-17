# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned HunYuanVL (HunyuanOCR) image-text-to-text bundle construction."""

from __future__ import annotations

import json
from pathlib import Path

from .checkpoint import CheckpointWeights
from .config import ImageProcessor, ModelConfig
from .graph import build_decoder, build_visual
from .support import SUPPORTED_TASKS

DEFAULT_CAPACITY = 8192
PREFILL_CHUNK = 2048
# The native runtime renders exactly this released chat template; another needs its own renderer.
CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{%"
    " if messages[0]['content'] is string %}{% set system_message = messages[0]['content'] %"
    "}{% else %}{% set system_message = messages[0]['content']['text'] %}{% endif %}<｜hy_begin▁of▁sentence｜>{{ system_message }"
    "}<｜hy_place▁holder▁no▁3｜>{% else %}{% set loop_messages = messages %}<｜hy_begin▁of▁sentence｜>{%"
    " endif %}{% for message in loop_messages %}{% if message['role'] == 'user' %}{% if message['content'] is string %"
    "}{{ message['content'] }}{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %"
    "}<｜hy_place▁holder▁no▁100｜><｜hy_place▁holder▁no▁102｜><｜hy_place▁holder▁no▁101｜>{%"
    " elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}{% endif %"
    "}<｜hy_User｜>{% elif message['role'] == 'assistant' %}{{ message['content'] }}<｜hy_Assistant｜>{%"
    " endif %}{% endfor %}"
)
_ROLE_TOKENS = {
    "bos": "<｜hy_begin▁of▁sentence｜>",
    "system_end": "<｜hy_place▁holder▁no▁3｜>",
    "user": "<｜hy_User｜>",
    "assistant": "<｜hy_Assistant｜>",
}


def _special_tokens(root: Path, config: ModelConfig) -> dict:
    tokenizer = json.loads((root / "tokenizer.json").read_text(encoding="utf-8"))
    added = {item["content"]: item["id"] for item in tokenizer.get("added_tokens", [])}
    tokens = {}
    for role, content in _ROLE_TOKENS.items():
        if content not in added:
            raise ValueError(f"hunyuan_vl tokenizer lacks the {role} token {content!r}")
        tokens[role] = added[content]
    settings = json.loads((root / "tokenizer_config.json").read_text(encoding="utf-8"))
    for name, expected in (
        ("image_token", config.image_token_id),
        ("image_start_token", config.image_start_token_id),
        ("image_end_token", config.image_end_token_id),
    ):
        if added.get(settings.get(name)) != expected:
            raise ValueError(f"hunyuan_vl tokenizer {name} does not match config id {expected}")
    if settings.get("bos_token") != _ROLE_TOKENS["bos"]:
        raise ValueError("hunyuan_vl tokenizer bos_token differs from the chat template")
    generation = json.loads((root / "generation_config.json").read_text(encoding="utf-8"))
    eos = generation.get("eos_token_id", config.raw["text_config"].get("eos_token_id"))
    eos = [eos] if isinstance(eos, int) else eos
    if (
        not isinstance(eos, list)
        or not eos
        or any(type(i) is not int or not 0 <= i < config.vocab_size for i in eos)
    ):
        raise ValueError("hunyuan_vl needs end-of-sequence token ids inside the vocabulary")
    tokens.update(
        image=config.image_token_id,
        image_start=config.image_start_token_id,
        image_end=config.image_end_token_id,
        eos=eos,
    )
    return tokens


def build(request, writer):
    """Own every build argument and write the decoder, vision and host-side sections."""
    if request.backend != "trt":
        raise ValueError("hunyuan_vl supports only the trt backend")
    if request.task not in SUPPORTED_TASKS:
        raise ValueError(f"hunyuan_vl supports only {', '.join(SUPPORTED_TASKS)}")
    if request.dynamic_kv_cache:
        raise NotImplementedError("hunyuan_vl does not support dynamic_kv_cache")
    for name, value, default in (
        ("max_batch_size", request.max_batch_size, 1),
        ("tensor_parallel_size", request.tensor_parallel_size, 1),
        ("context_parallel_size", request.context_parallel_size, 1),
        ("mtp_seq_len", request.mtp_seq_len, None),
        ("fp32_layers", request.fp32_layers, ()),
        ("video_num_frames", request.video_num_frames, None),
    ):
        if value != default:
            raise ValueError(f"hunyuan_vl does not support {name}={value!r}")
    if request.quantization not in (None, "none"):
        raise ValueError("hunyuan_vl builds the released BF16 weights without quantization")
    if (request.image_height is None) != (request.image_width is None):
        raise ValueError("hunyuan_vl image_height and image_width bound the image area together")
    precision = request.precision.lower()
    if precision not in ("fp32", "bf16"):
        raise ValueError("hunyuan_vl precision must be fp32 or bf16")
    root = Path(request.model_dir)
    config = ModelConfig.from_dir(root)
    processor = ImageProcessor.from_dir(root, config)
    merge = config.merge_size**2
    if request.image_height is None:
        max_patches = config.position_edge**2
    else:
        max_patches = (
            request.image_height * request.image_width // config.patch_size**2 // merge * merge
        )
    if max_patches < merge:
        raise ValueError("hunyuan_vl image area bound must hold at least one merged token")
    # Preprocessing scales small images up to min_pixels, so a smaller bound would reject them all.
    if max_patches * config.patch_size**2 < processor.min_pixels:
        raise ValueError(
            f"hunyuan_vl image area bound of {max_patches} patches is below the image "
            f"processor's min_pixels={processor.min_pixels}"
        )
    if config.raw.get("quantization_config") is not None or any(
        (root / name).exists() for name in ("hf_quant_config.json", "quantize_config.json")
    ):
        raise ValueError("hunyuan_vl expects the original, unquantized checkpoint")
    if (root / "chat_template.jinja").read_text(encoding="utf-8") != CHAT_TEMPLATE:
        raise ValueError("hunyuan_vl renders only the released HunyuanOCR chat template")
    tokens = _special_tokens(root, config)
    capacity = (
        DEFAULT_CAPACITY if request.max_sequence_length is None else request.max_sequence_length
    )
    if type(capacity) is not int or not 1 <= capacity <= config.max_position_embeddings:
        raise ValueError("hunyuan_vl max_sequence_length must be between 1 and the context length")
    weights = CheckpointWeights(root)
    max_input = min(capacity, PREFILL_CHUNK)
    writer.set_header(family="hunyuan_vl", task=request.task, backend=request.backend)
    writer.add_bytes(
        "decoder.plan",
        build_decoder(
            config,
            weights,
            capacity,
            max_input_length=max_input,
            precision=precision,
            verbose=request.verbose,
        ),
    )
    writer.add_bytes(
        "visual.plan",
        build_visual(config, weights, max_patches, precision=precision, verbose=request.verbose),
    )
    with writer.open_section("embeddings.f32") as output:
        output.write(weights["model.embed_tokens.weight"].astype("<f4", copy=False).tobytes())
    writer.add_bytes("tokenizer.json", (root / "tokenizer.json").read_bytes())
    writer.add_json(
        "runtime.json",
        {
            "format": 1,
            "precision": precision,
            "max_cache_length": capacity,
            "max_input_length": max_input,
            "max_patches": max_patches,
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "vocab_size": config.vocab_size,
            "rope_inverse_frequencies": config.inverse_frequencies(),
            "rope_sections": list(config.rope_sections),
            "patch_size": config.patch_size,
            "merge_size": config.merge_size,
            "position_edge": config.position_edge,
            "image": {
                "min_pixels": processor.min_pixels,
                "max_pixels": processor.max_pixels,
                "mean": list(processor.mean),
                "std": list(processor.std),
                "rescale_factor": processor.rescale_factor,
            },
            "tokens": tokens,
        },
    )
