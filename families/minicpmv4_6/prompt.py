# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider chat-template recipes and image-prompt metadata for the native runtime.

The runtime formats conversations with a small recipe (fixed prefixes and
suffixes per role) instead of a template engine. The recipe is extracted from
the checkpoint's own chat template with sentinel messages and then checked
against the template on a set of conversations; a template whose formatting
the recipe cannot reproduce is rejected at build time.
"""

from __future__ import annotations

import math
from pathlib import Path

from .config import load_json, same_value, stored_value

# MiniCPMV4_6ImageProcessor options. The native runtime honors these values only; the checkpoint may restate them.
_PROCESSOR_FIXED = {
    "image_processor_type": "MiniCPMV4_6ImageProcessor",
    "processor_class": "MiniCPMV4_6Processor",
    "do_resize": True, "do_rescale": True, "do_normalize": True, "do_convert_rgb": True,
    "rescale_factor": 1 / 255, "resample": 3, "slice_mode": True, "use_image_id": True, "downsample_mode": "16x",
}
# Options the runtime reads from vision.json (patch_size must also match config.json).
_PROCESSOR_SETTINGS = ("patch_size", "max_slice_nums", "scale_resolution", "image_mean", "image_std")
_FP16_MAX = 65504.0
# The runtime plans slice grids in time linear in max_slice_nums and accepts at most this value
# (runtime/image.h, kMaxSliceNums).
MAX_SLICE_NUMS = 4096

IMAGE_PLACEHOLDER = "<|image_pad|>"
# Image-slice markup tokens of the processor, by role.
MARKUP_TOKENS = {
    "image_pad": "<|image_pad|>",
    "video_pad": "<|video_pad|>",
    "image_start": "<image>",
    "image_end": "</image>",
    "slice_start": "<slice>",
    "slice_end": "</slice>",
    "image_id_start": "<image_id>",
    "image_id_end": "</image_id>",
}


def _split(text: str, marker: str) -> list[str]:
    if text.count(marker) != 1:
        raise ValueError("MiniCPM-V 4.6 chat template does not preserve content markers")
    return text.split(marker)


def _without_prefix(text: str, prefix: str) -> str:
    if not text.startswith(prefix):
        raise ValueError("MiniCPM-V 4.6 chat template has context-dependent formatting")
    return text[len(prefix):]


def _without_suffix(text: str, suffix: str) -> str:
    if not text.endswith(suffix):
        raise ValueError("MiniCPM-V 4.6 chat template has context-dependent formatting")
    return text[:-len(suffix)] if suffix else text


def render_recipe(recipe: dict, messages: list[dict], *, generation: bool = True) -> str:
    """Builder-side oracle of the runtime's message formatter (text content only)."""
    last_user = -1
    for index, message in enumerate(messages):
        content = message["content"].strip()
        if message["role"] == "user" and not (
            content.startswith("<tool_response>") and content.endswith("</tool_response>")
        ):
            last_user = index
    if last_user < 0:
        raise ValueError("MiniCPM-V 4.6 requires a user query")
    system = messages[0]["content"].strip() if messages[0]["role"] == "system" else None
    if system is None:
        output = recipe["default_prefix"]
    elif system:
        output = recipe["system"]["prefix"] + system + recipe["system"]["suffix"]
    else:
        output = recipe["empty_system_prefix"]
    for index, message in enumerate(messages):
        role, content = message["role"], message["content"].strip()
        if role == "system":
            if index:
                raise ValueError("System message must be first")
            continue
        if role not in {"user", "assistant"}:
            raise ValueError("MiniCPM-V 4.6 accepts system, user and assistant roles")
        if role == "user":
            output += recipe["user"]["prefix"] + content + recipe["user"]["suffix"]
            continue
        reasoning = message.get("reasoning_content", "").strip()
        if recipe["parse_assistant_thinking"] and "reasoning_content" not in message and "</think>" in content:
            reasoning = content.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n").strip()
            content = content.split("</think>")[-1].lstrip("\n")
        form = recipe["assistant_recent" if index > last_user else "assistant_history"]
        output += form["prefix"]
        if form["include_reasoning"]:
            output += reasoning + form["reasoning_suffix"]
        output += content + form["suffix"]
    return output + (recipe["generation"] if generation else "")


def provider_recipes(tokenizer) -> dict:
    """Extract the no-tool message forms of the checkpoint template, for both thinking modes."""
    a, b, c, reasoning = (f"MINICPMV46_NATIVE_SENTINEL_{index}_END" for index in range(4))

    def message(role, content, **kwargs):
        return {"role": role, "content": content, **kwargs}

    result = {}
    for thinking in (False, True):
        def render(messages, generation=False):
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=generation, enable_thinking=thinking
            )

        single = render([message("user", a)])
        full_prefix, user_suffix = _split(single, a)
        doubled = render([message("user", a), message("user", b)])
        middle = _split(_split(doubled, a)[1], b)[0]
        user_prefix = _without_prefix(middle, user_suffix)
        default = _without_suffix(full_prefix, user_prefix)
        system = _without_suffix(render([message("system", c), message("user", a)]), user_prefix + a + user_suffix)
        system_prefix, system_suffix = _split(system, c)
        recipe = {
            "default_prefix": default,
            "user": {"prefix": user_prefix, "suffix": user_suffix},
            "system": {"prefix": system_prefix, "suffix": system_suffix},
            "empty_system_prefix": _without_suffix(
                render([message("system", ""), message("user", a)]), user_prefix + a + user_suffix
            ),
            "generation": _without_prefix(render([message("user", a)], True), single),
        }
        for historical in (False, True):
            messages = [message("user", a), message("assistant", b, reasoning_content=reasoning)]
            if historical:
                messages.append(message("user", c))
            assistant = _without_prefix(render(messages), single)
            if historical:
                assistant = _without_suffix(assistant, user_prefix + c + user_suffix)
            prefix, suffix = _split(assistant, b)
            includes = reasoning in prefix
            start, between = _split(prefix, reasoning) if includes else (prefix, "")
            recipe["assistant_history" if historical else "assistant_recent"] = {
                "prefix": start, "include_reasoning": includes, "reasoning_suffix": between, "suffix": suffix,
            }
        probe = [message("user", a), message("assistant", f"<think>{reasoning}</think>{b}")]
        recipe["parse_assistant_thinking"] = False
        actual = render(probe)
        if render_recipe(recipe, probe, generation=False) != actual:
            recipe["parse_assistant_thinking"] = True
            if render_recipe(recipe, probe, generation=False) != actual:
                raise ValueError("Unsupported MiniCPM-V 4.6 assistant reasoning formatting")
        cases = [
            [message("user", " \nHello, 世界! ")],
            [message("system", ""), message("user", "")],
            [message("system", " Follow instructions. "), message("user", a)],
            [message("user", a), message("assistant", b)],
            [message("user", a), message("assistant", b), message("user", c)],
            probe,
            probe + [message("user", c)],
            [message("user", a), message("assistant", b, reasoning_content=reasoning), message("user", c)],
            [message("user", a), message("assistant", b, reasoning_content=reasoning)],
            [message("user", a), message("user", "<tool_response>value</tool_response>"), message("assistant", b)],
        ]
        for messages in cases:
            for generation in (False, True):
                if render_recipe(recipe, messages, generation=generation) != render(messages, generation):
                    raise ValueError("MiniCPM-V 4.6 prompt recipe does not match the checkpoint template")
        # Structured image parts render as one placeholder each, joined to the text by newlines.
        structured = [
            message("user", [{"type": "image"}, {"type": "text", "text": "Describe this."}]),
            message("assistant", b),
            message("user", [{"type": "image"}, {"type": "image"}, {"type": "text", "text": " Compare. "}]),
        ]
        expanded = [
            message("user", f"{IMAGE_PLACEHOLDER}\nDescribe this."),
            message("assistant", b),
            message("user", f"{IMAGE_PLACEHOLDER}\n{IMAGE_PLACEHOLDER}\n Compare. "),
        ]
        if render(structured, True) != render_recipe(recipe, expanded):
            raise ValueError("MiniCPM-V 4.6 image placeholders differ from the checkpoint template")
        result[str(thinking).lower()] = recipe
    return result


def normalization_values(values: object, key: str) -> list[float]:
    if (not isinstance(values, list) or len(values) != 3
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)):
        raise ValueError(f"MiniCPM-V 4.6 preprocessor_config.json {key} must hold three finite numbers")
    return [float(v) for v in values]


def check_normalization(mean: list[float], stddev: list[float]) -> None:
    """The runtime normalizes 0..255 pixels as (pixel - mean * 255) / (std * 255) in FP32; the
    results of both ends must be finite and fit FP16, the vision engine's narrowest work type."""
    def f32(value: float) -> float:
        return stored_value(value, "f")

    for m, s in zip(mean, stddev):
        offset, scale = f32(f32(m) * 255.0), f32(f32(s) * 255.0)
        if not (math.isfinite(offset) and math.isfinite(scale) and scale > 0):
            raise ValueError("MiniCPM-V 4.6 image normalization overflows FP32")
        for pixel in (0.0, 255.0):
            value = f32(f32(pixel - offset) / scale)
            if not math.isfinite(value) or abs(value) > _FP16_MAX:
                raise ValueError("MiniCPM-V 4.6 image normalization produces values outside the FP16 range")


def processor_settings(model_dir: Path, patch_size: int, max_patches: int) -> dict:
    """The image-processor settings the runtime implements, from ``preprocessor_config.json``."""
    processor = load_json(model_dir / "preprocessor_config.json")
    if not isinstance(processor, dict):
        raise ValueError("MiniCPM-V 4.6 preprocessor_config.json must contain one JSON object")
    unknown = sorted(set(processor) - set(_PROCESSOR_FIXED) - set(_PROCESSOR_SETTINGS))
    if unknown:
        raise ValueError(f"MiniCPM-V 4.6 native runtime does not implement image processor options {unknown}")
    for key, value in _PROCESSOR_FIXED.items():
        if key in processor and not same_value(processor[key], value):
            raise ValueError(f"MiniCPM-V 4.6 native runtime implements {key}={value!r}, "
                             f"preprocessor_config.json sets {processor[key]!r}")
    for key in _PROCESSOR_SETTINGS:
        if key not in processor:
            raise ValueError(f"MiniCPM-V 4.6 preprocessor_config.json must set {key}")
    if not same_value(processor["patch_size"], patch_size):
        raise ValueError(f"MiniCPM-V 4.6 preprocessor_config.json requires patch_size={patch_size}")
    for key in ("max_slice_nums", "scale_resolution"):
        if type(processor[key]) is not int or not 0 < processor[key] <= 2**31 - 1:
            raise ValueError(f"MiniCPM-V 4.6 preprocessor_config.json {key} must be a positive 32-bit integer")
    if processor["max_slice_nums"] > MAX_SLICE_NUMS:
        raise ValueError(f"MiniCPM-V 4.6 native runtime supports max_slice_nums up to {MAX_SLICE_NUMS}")
    if processor["scale_resolution"] < 4 * patch_size:
        raise ValueError("MiniCPM-V 4.6 scale_resolution must hold at least one 4 x 4 block of patches")
    if (processor["scale_resolution"] // patch_size) ** 2 > max_patches:
        raise ValueError(f"MiniCPM-V 4.6 scale_resolution exceeds the vision engine's {max_patches} patches")
    mean, stddev = (normalization_values(processor[key], key) for key in ("image_mean", "image_std"))
    check_normalization(mean, stddev)
    return {"scale_resolution": processor["scale_resolution"], "max_slice_nums": processor["max_slice_nums"],
            "mean": mean, "stddev": stddev}


def check_vocabulary(vocabulary: dict[str, int], total: int, rows: int) -> None:
    """Every token id must address a row of the embedding table (checked before any engine build)."""
    ids = list(vocabulary.values())
    if (not ids or type(total) is not int or total > rows or len(set(ids)) != len(ids)
            or any(type(i) is not int or not 0 <= i < rows for i in ids)):
        raise ValueError(f"MiniCPM-V 4.6 tokenizer ids must be distinct rows of the {rows}-row embedding table")


# The tokenizer.json layout the native tokenizer loads (runtime/bpe_tokenizer.cpp, validate_schema and parse_*).
_SPLIT_PATTERN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}"
                  r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
_BYTE_LEVEL = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}
_TOKENIZER_LAYOUT = {
    "version": "1.0", "truncation": None, "padding": None, "normalizer": {"type": "NFC"},
    "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
        {"type": "Split", "pattern": {"Regex": _SPLIT_PATTERN}, "behavior": "Isolated", "invert": False},
        _BYTE_LEVEL]},
    "post_processor": _BYTE_LEVEL, "decoder": _BYTE_LEVEL,
}
_BPE_MODEL = {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": "",
              "end_of_word_suffix": "", "fuse_unk": False, "byte_fallback": False, "ignore_merges": False}
_ADDED_FLAGS = ("single_word", "lstrip", "rstrip", "normalized")


def _byte_alphabet() -> list[str]:
    """GPT-2's byte-level characters: one per byte value."""
    direct = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
    characters, displaced = [], 0
    for byte in range(256):
        characters.append(chr(byte) if byte in direct else chr(256 + displaced))
        displaced += byte not in direct
    return characters


def check_tokenizer_layout(root: object) -> None:
    """The parsed ``tokenizer.json`` must be the byte-level BPE the native tokenizer loads (before any engine build)."""
    def require(condition: bool, what: str) -> None:
        if not condition:
            raise ValueError(f"MiniCPM-V 4.6 native tokenizer cannot load this tokenizer.json: {what}")

    require(isinstance(root, dict) and set(root) == {*_TOKENIZER_LAYOUT, "added_tokens", "model"},
            "unexpected top-level fields")
    for key, expected in _TOKENIZER_LAYOUT.items():
        require(same_value(root[key], expected), f"{key} differs from the supported layout")
    model = root["model"]
    require(isinstance(model, dict) and set(model) == {*_BPE_MODEL, "vocab", "merges"}
            and all(same_value(model[key], value) for key, value in _BPE_MODEL.items()), "not the byte-level BPE model")
    vocab, merges = model["vocab"], model["merges"]
    require(isinstance(vocab, dict) and vocab and all(type(i) is int for i in vocab.values())
            and sorted(vocab.values()) == list(range(len(vocab))), "vocabulary ids are not 0 .. n-1")
    require(all(character in vocab for character in _byte_alphabet()), "a byte-level token is missing")
    added = root["added_tokens"]
    require(isinstance(added, list) and added, "no added tokens")
    contents = set()
    for index, token in enumerate(added):
        require(isinstance(token, dict) and set(token) == {"id", "content", "special", *_ADDED_FLAGS}
                and same_value(token["id"], len(vocab) + index) and isinstance(token["content"], str)
                and token["content"] and token["content"] not in vocab and token["content"] not in contents
                and all(token[flag] is False for flag in _ADDED_FLAGS) and type(token["special"]) is bool,
                f"added token {index} is not a verbatim token numbered after the vocabulary")
        contents.add(token["content"])
    require(isinstance(merges, list), "merges are not a list")
    seen = set()
    for merge in merges:
        require(isinstance(merge, list) and len(merge) == 2 and all(isinstance(part, str) for part in merge)
                and merge[0] in vocab and merge[1] in vocab and merge[0] + merge[1] in vocab
                and tuple(merge) not in seen, f"merge {merge!r} is not a new pair of vocabulary tokens")
        seen.add(tuple(merge))


def image_metadata(model_dir: Path, config, capacity: int, precision: str, max_patches: int) -> dict:
    """Validated image-processor settings, markup token ids and prompt recipes (``vision.json``)."""
    from transformers import AutoTokenizer

    processor = processor_settings(model_dir, config.vision.patch_size, max_patches)
    check_tokenizer_layout(load_json(model_dir / "tokenizer.json"))

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    check_vocabulary(tokenizer.get_vocab(), len(tokenizer), config.text.vocab_size)
    token_ids = {}
    for name, text in MARKUP_TOKENS.items():
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"MiniCPM-V 4.6 tokenizer does not keep {text} as one token")
        token_ids[name] = encoded[0]
    if token_ids["image_pad"] != config.image_token_id or token_ids["video_pad"] != config.video_token_id:
        raise ValueError("MiniCPM-V 4.6 tokenizer and config.json disagree on media token ids")
    del token_ids["video_pad"]
    return {
        "schema_version": 1,
        "precision": precision,
        "capacity": capacity,
        "hidden_size": config.text.hidden_size,
        "patch_size": config.vision.patch_size,
        "positions_per_side": config.vision.positions_per_side,
        "max_patches": max_patches,
        "scale_resolution": processor["scale_resolution"],
        "max_slice_nums": processor["max_slice_nums"],
        "mean": processor["mean"],
        "stddev": processor["stddev"],
        "token_ids": token_ids,
        "prompts": provider_recipes(tokenizer),
    }
