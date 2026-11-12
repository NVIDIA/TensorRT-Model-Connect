# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint text and image processing that the native runtime reproduces.

The runtime renders the chat template, tokenizes and resizes images itself, so
the build accepts only the exact processing it implements: the published
Gemma 3n chat template, the SentencePiece-style BPE tokenizer pipeline and a
768x768 bilinear resize with 1/255 rescaling and no normalization. Anything
else is refused here instead of being silently approximated at run time.
Only the standard library is imported.
"""

from __future__ import annotations

import json
from pathlib import Path

_SPACE_MARK = chr(0x2581)  # SentencePiece's visible space, "LOWER ONE EIGHTH BLOCK"
_IMAGE_SIZE = 768
_BILINEAR = 2  # PIL.Image.Resampling.BILINEAR
_RESCALE = 0.00392156862745098

# Conversations on which the checkpoint's template must render exactly what the
# runtime renders: system prefix, alternating turns, several images per turn,
# text around images, and whitespace the template trims (or keeps).
_PROBES = (
    [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe it."}]}],
    [{"role": "system", "content": [{"type": "text", "text": " Be brief. "}]},
     {"role": "user", "content": [{"type": "text", "text": " Q1 "}, {"type": "image"},
                                  {"type": "image"}, {"type": "text", "text": "\tmore\n"}]},
     {"role": "assistant", "content": [{"type": "text", "text": "A1 "}]},
     {"role": "user", "content": [{"type": "text", "text": "\u3000Q2\u00a0"}]}],
    [{"role": "user", "content": [{"type": "text", "text": "a"}]},
     {"role": "assistant", "content": [{"type": "image"}, {"type": "text", "text": "b"}]},
     {"role": "user", "content": [{"type": "image"}]}],
)


def native_chat_text(messages: list) -> str:
    """The prompt the runtime renders (runtime/prompt.cpp), before image expansion."""
    text, first, prefix = "<bos>", 0, ""
    if messages and messages[0]["role"] == "system":
        prefix = messages[0]["content"][0]["text"] + "\n\n"
        first = 1
    for index, message in enumerate(messages[first:]):
        text += "<start_of_turn>" + ("user" if index % 2 == 0 else "model") + "\n"
        if index == 0:
            text += prefix
        for item in message["content"]:
            text += "<image_soft_token>" if item["type"] == "image" else item["text"].strip()
        text += "<end_of_turn>\n"
    return text + "<start_of_turn>model\n"


# Special tokens the runtime emits while rendering a conversation.
SPECIAL_TOKENS = ("<bos>", "<start_of_turn>", "<end_of_turn>", "<start_of_image>",
                  "<image_soft_token>", "<end_of_image>")


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Gemma 3n {path.name} must be a JSON object")
    return value


def chat_template(root: Path) -> str:
    """The checkpoint's chat template from ``chat_template.jinja`` or the tokenizer config."""
    path = root / "chat_template.jinja"
    if path.is_file():
        return path.read_text(encoding="utf-8")
    template = _load(root / "tokenizer_config.json").get("chat_template")
    if not isinstance(template, str):
        raise ValueError("Gemma 3n checkpoint has no chat template")
    return template


def validate_chat_template(template: str) -> None:
    """Render the probes with the checkpoint template as Transformers does; compare."""
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise ValueError(message)

    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                extensions=[loopcontrols])
    environment.globals["raise_exception"] = raise_exception
    compiled = environment.from_string(template)
    for messages in _PROBES:
        rendered = compiled.render(messages=messages, bos_token="<bos>", add_generation_prompt=True)
        if rendered != native_chat_text(messages):
            raise NotImplementedError(
                "gemma3n renders the Gemma 3n chat template natively; this checkpoint's template "
                "renders a conversation differently")


def validate_tokenizer(tokenizer: dict, config) -> dict:
    """Check the tokenizer pipeline the runtime implements; return its special ids."""
    expected = {
        "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": _SPACE_MARK},
        "pre_tokenizer": {"type": "Split", "pattern": {"String": " "},
                          "behavior": "MergedWithPrevious", "invert": False},
        "decoder": {"type": "Sequence", "decoders": [
            {"type": "Replace", "pattern": {"String": _SPACE_MARK}, "content": " "},
            {"type": "ByteFallback"}, {"type": "Fuse"}]},
    }
    for key, value in expected.items():
        if tokenizer.get(key) != value:
            raise NotImplementedError(f"gemma3n does not implement this tokenizer {key}")
    model = tokenizer.get("model") or {}
    settings = {key: model.get(key) for key in (
        "type", "dropout", "unk_token", "continuing_subword_prefix", "end_of_word_suffix",
        "fuse_unk", "byte_fallback", "ignore_merges")}
    if settings != {"type": "BPE", "dropout": None, "unk_token": "<unk>",
                    "continuing_subword_prefix": None, "end_of_word_suffix": None,
                    "fuse_unk": True, "byte_fallback": True, "ignore_merges": False}:
        raise NotImplementedError("gemma3n implements only byte-fallback BPE without dropout")
    added = tokenizer.get("added_tokens")
    if not isinstance(added, list):
        raise ValueError("Gemma 3n tokenizer has no added_tokens list")
    for token in added:
        if any(token.get(flag) for flag in ("single_word", "lstrip", "rstrip", "normalized")):
            raise NotImplementedError("gemma3n matches added tokens only verbatim on raw text")
    vocab, merges = model.get("vocab"), model.get("merges")
    if not isinstance(vocab, dict) or len(vocab) != config.text.vocab_size:
        raise ValueError("Gemma 3n tokenizer vocabulary does not match the text vocabulary")
    if not isinstance(merges, list) or not all(
            isinstance(pair, list) and len(pair) == 2 and all(isinstance(x, str) for x in pair)
            for pair in merges):
        raise NotImplementedError("gemma3n reads BPE merges stored as token pairs")
    for byte in range(256):
        if f"<0x{byte:02X}>" not in vocab:
            raise ValueError("Gemma 3n tokenizer lacks a byte-fallback token")
    ids = {}
    for token in added:
        if vocab.get(token.get("content")) != token.get("id"):
            raise ValueError("Gemma 3n added token ids disagree with the vocabulary")
        ids[token["content"]] = token["id"]
    if not set(SPECIAL_TOKENS) <= set(ids):
        raise ValueError("Gemma 3n tokenizer lacks a chat or image special token")
    for name, token in (("image_token_id", "<image_soft_token>"), ("boi_token_id", "<start_of_image>"),
                        ("eoi_token_id", "<end_of_image>")):
        if ids[token] != getattr(config, name):
            raise ValueError(f"Gemma 3n {token} id disagrees with the model config")
    return {token: ids[token] for token in SPECIAL_TOKENS}


def validate_image_processor(processor: dict) -> float:
    """Check the image processing the runtime implements; return the rescale factor."""
    if processor.get("image_processor_type") not in ("SiglipImageProcessorFast", "SiglipImageProcessor"):
        raise NotImplementedError("gemma3n implements the SigLIP image processor only")
    checks = {
        "do_resize": processor.get("do_resize", True) is True,
        "size": processor.get("size") == {"height": _IMAGE_SIZE, "width": _IMAGE_SIZE},
        "resample": processor.get("resample") == _BILINEAR,
        "do_rescale": processor.get("do_rescale", True) is True,
        "rescale_factor": processor.get("rescale_factor") == _RESCALE,
        "do_normalize": processor.get("do_normalize") is False,
        "do_center_crop": not processor.get("do_center_crop"),
        "do_pad": not processor.get("do_pad"),
        "do_convert_rgb": processor.get("do_convert_rgb") in (None, True),
    }
    failed = sorted(name for name, ok in checks.items() if not ok)
    if failed:
        raise NotImplementedError(f"gemma3n does not implement image processing {', '.join(failed)}")
    return _RESCALE


def validate_processing(root: Path, config) -> dict:
    """Validate every processing input of the checkpoint; return the runtime fields."""
    root = Path(root)
    validate_chat_template(chat_template(root))
    special = validate_tokenizer(_load(root / "tokenizer.json"), config)
    rescale = validate_image_processor(_load(root / "preprocessor_config.json"))
    processor = root / "processor_config.json"
    if processor.is_file() and _load(processor).get("image_seq_length", 256) != config.soft_tokens_per_image:
        raise ValueError("Gemma 3n processor image_seq_length disagrees with the vision tower")
    generation = root / "generation_config.json"
    eos = (_load(generation) if generation.is_file() else {}).get(
        "eos_token_id", config.raw.get("eos_token_id"))
    eos = [eos] if type(eos) is int else eos
    if not isinstance(eos, list) or not eos or any(
            type(token) is not int or not 0 <= token < config.text.vocab_size for token in eos):
        raise ValueError("Gemma 3n generation config needs end-of-sequence token ids")
    if special["<end_of_turn>"] not in eos:
        raise ValueError("Gemma 3n generation must stop at <end_of_turn>")
    return {"bos_token_id": special["<bos>"], "eos_token_ids": eos, "rescale_factor": rescale}
