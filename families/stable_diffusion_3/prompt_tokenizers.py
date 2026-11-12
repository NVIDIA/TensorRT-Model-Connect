# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The three tokenizer files a bundle carries, and the layout the runtime reproduces.

The builder accepts only the tokenizer layouts the native tokenizers are
verified to reproduce; a checkpoint whose tokenizer files differ is refused at
build time rather than tokenized differently.
"""

from __future__ import annotations

import json
from pathlib import Path

# The CLIP pre-tokenizer of openai/clip-vit-large-patch14 and OpenCLIP bigG.
CLIP_SPLIT = (
    r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|"
    r"[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"
)
# The CLIP layout the pipeline's CLIPTokenizerFast converts SD3's vocab.json /
# merges.txt to (tokenizers library defaults filled in for absent keys).
_CLIP_NORMALIZER = {"type": "Sequence", "normalizers": [
    {"type": "NFC"}, {"type": "Replace", "pattern": {"Regex": r"\s+"}, "content": " "},
    {"type": "Lowercase"},
]}
_CLIP_PRE_TOKENIZER = {"type": "Sequence", "pretokenizers": [
    {"type": "Split", "pattern": {"Regex": CLIP_SPLIT}, "behavior": "Removed", "invert": True},
    {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
]}
_CLIP_MODEL = {"type": "BPE", "dropout": None, "end_of_word_suffix": "</w>", "fuse_unk": False,
               "byte_fallback": False, "ignore_merges": False}
_BYTE_LEVEL_DEFAULTS = {"add_prefix_space": True, "trim_offsets": True, "use_regex": True}

# The T5 layout of stabilityai/stable-diffusion-3.5-medium (the charsmap itself
# is read by the runtime).
_T5_NORMALIZERS = [
    {"type": "Precompiled"},
    {"type": "Strip", "strip_left": False, "strip_right": True},
    {"type": "Replace", "pattern": {"Regex": " {2,}"}, "content": "\u2581"},
]
_T5_PRE_TOKENIZER = {"type": "Metaspace", "replacement": "\u2581", "prepend_scheme": "always",
                     "split": True}


def _refuse(component: str, message: str) -> None:
    raise NotImplementedError(f"stable_diffusion_3 {component}: {message}")


def check_clip_layout(data: bytes, component: str) -> None:
    tokenizer = json.loads(data)
    model = tokenizer.get("model") or {}
    defaults = {"dropout": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": False}
    if {key: model.get(key, defaults.get(key)) for key in _CLIP_MODEL} != _CLIP_MODEL or \
            model.get("continuing_subword_prefix") not in (None, ""):
        _refuse(component, "the CLIP tokenizer must be a plain BPE model with the </w> suffix")
    if tokenizer.get("normalizer") != _CLIP_NORMALIZER:
        _refuse(component, f"unsupported CLIP normalizer {json.dumps(tokenizer.get('normalizer'))}")
    pre = json.loads(json.dumps(tokenizer.get("pre_tokenizer") or {}))
    pieces = pre.get("pretokenizers") or []
    if len(pieces) == 2 and pieces[1].get("type") == "ByteLevel":
        pieces[1] = {**_BYTE_LEVEL_DEFAULTS, **pieces[1]}
    if pre != _CLIP_PRE_TOKENIZER:
        _refuse(component, f"unsupported CLIP pre-tokenizer {json.dumps(pre)[:300]}")
    for token in tokenizer.get("added_tokens", []):
        if token.get("single_word"):
            _refuse(component, f"unsupported added token {json.dumps(token)}")


def check_t5_layout(data: bytes, pad_id: int, eos_id: int) -> None:
    tokenizer = json.loads(data)
    model = tokenizer.get("model") or {}
    if model.get("type") != "Unigram" or model.get("byte_fallback"):
        _refuse("tokenizer_3", "the T5 tokenizer must be a Unigram model without byte fallback")
    normalizer = tokenizer.get("normalizer") or {}
    steps = [{key: value for key, value in step.items() if key != "precompiled_charsmap"}
             for step in normalizer.get("normalizers", [])]
    if normalizer.get("type") != "Sequence" or steps != _T5_NORMALIZERS or \
            not normalizer["normalizers"][0].get("precompiled_charsmap"):
        _refuse("tokenizer_3", f"unsupported T5 normalizer {json.dumps(steps)[:300]}")
    if tokenizer.get("pre_tokenizer") != _T5_PRE_TOKENIZER:
        _refuse("tokenizer_3", f"unsupported T5 pre-tokenizer {json.dumps(tokenizer.get('pre_tokenizer'))}")
    post = tokenizer.get("post_processor") or {}
    single = [next(iter(item.values())).get("id") for item in post.get("single", [])]
    if post.get("type") != "TemplateProcessing" or single != ["A", "</s>"]:
        _refuse("tokenizer_3", "the T5 template must append </s> to the sequence")
    for token in tokenizer.get("added_tokens", []):
        # Special tokens are matched in the raw prompt.
        if not token.get("special") or token.get("normalized") or token.get("single_word"):
            _refuse("tokenizer_3", f"unsupported added token {json.dumps(token)}")
    added = {token["content"]: token["id"] for token in tokenizer.get("added_tokens", [])}
    if added.get("<pad>") != pad_id or added.get("</s>") != eos_id:
        _refuse("tokenizer_3", "the tokenizer disagrees with the T5 special-token ids")


def clip_tokenizer(model_dir: Path, component: str) -> tuple[bytes, dict]:
    """The fast-tokenizer JSON the runtime reads, plus the CLIP special-token ids.

    SD3 ships the legacy vocab.json / merges.txt pair; it is converted with
    the pipeline's own tokenizer class so the bundle always carries one
    format. The two CLIP tokenizers pad differently (CLIP-L with the
    end-of-text token, CLIP-G with "!").
    """
    from transformers import CLIPTokenizerFast

    directory = Path(model_dir) / component
    tokenizer = CLIPTokenizerFast.from_pretrained(str(directory))
    ids = {
        "bos": tokenizer.bos_token_id,
        "eos": tokenizer.eos_token_id,
        "pad": tokenizer.pad_token_id,
    }
    if any(value is None for value in ids.values()):
        _refuse(component, f"a CLIP special token is missing: {ids}")
    path = directory / "tokenizer.json"
    data = path.read_bytes() if path.is_file() else tokenizer.backend_tokenizer.to_str().encode()
    check_clip_layout(data, component)
    return data, ids


def t5_tokenizer(model_dir: Path, cfg: dict) -> bytes:
    data = (Path(model_dir) / "tokenizer_3" / "tokenizer.json").read_bytes()
    check_t5_layout(data, cfg["pad_token_id"], cfg["eos_token_id"])
    return data
