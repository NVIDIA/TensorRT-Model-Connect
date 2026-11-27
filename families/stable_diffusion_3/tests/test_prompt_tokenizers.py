# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The builder refuses tokenizer layouts the native tokenizers do not reproduce."""

import copy
import json

import pytest

from families.stable_diffusion_3 import prompt_tokenizers


def _special(token_id: int, content: str) -> dict:
    return {"id": token_id, "content": content, "special": True, "normalized": False,
            "lstrip": False, "rstrip": False, "single_word": False}


# The layout of the checkpoint's tokenizer_3/tokenizer.json, with a stub vocabulary.
T5 = {
    "model": {"type": "Unigram", "unk_id": 2, "byte_fallback": False,
              "vocab": [["<pad>", 0.0], ["</s>", 0.0], ["<unk>", 0.0]]},
    "normalizer": {"type": "Sequence", "normalizers": [
        {"type": "Precompiled", "precompiled_charsmap": "AAAAAA=="},
        {"type": "Strip", "strip_left": False, "strip_right": True},
        {"type": "Replace", "pattern": {"Regex": " {2,}"}, "content": "▁"},
    ]},
    "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always",
                      "split": True},
    "post_processor": {"type": "TemplateProcessing", "single": [
        {"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": "</s>", "type_id": 0}},
    ]},
    "added_tokens": [_special(0, "<pad>"), _special(1, "</s>"), _special(2, "<unk>")],
}


def _check(tokenizer: dict) -> None:
    prompt_tokenizers.check_t5_layout(json.dumps(tokenizer).encode(), 0, 1)


def test_the_checkpoint_t5_layout_is_accepted():
    _check(T5)


def _first(t):
    t["pre_tokenizer"]["prepend_scheme"] = "first"


def _whitespace_split(t):
    t["pre_tokenizer"] = {"type": "Sequence", "pretokenizers": [
        {"type": "WhitespaceSplit"}, copy.deepcopy(T5["pre_tokenizer"])]}


def _lowercase(t):
    t["normalizer"]["normalizers"].append({"type": "Lowercase"})


def _strip_left(t):
    t["normalizer"]["normalizers"][1]["strip_left"] = True


def _normalized_token(t):
    t["added_tokens"].append({**_special(3, "<extra_id_0>"), "normalized": True})


def _byte_fallback(t):
    t["model"]["byte_fallback"] = True


@pytest.mark.parametrize("change", [_first, _whitespace_split, _lowercase, _strip_left,
                                    _normalized_token, _byte_fallback],
                         ids=lambda change: change.__name__.strip("_"))
def test_unverified_t5_layouts_are_refused(change):
    tokenizer = copy.deepcopy(T5)
    change(tokenizer)
    with pytest.raises(NotImplementedError):
        _check(tokenizer)


# The layout CLIPTokenizerFast converts SD3's vocab.json / merges.txt to, with a stub vocabulary.
CLIP = {
    "model": {"type": "BPE", "dropout": None, "unk_token": "<|endoftext|>",
              "continuing_subword_prefix": "", "end_of_word_suffix": "</w>", "fuse_unk": False,
              "byte_fallback": False, "ignore_merges": False, "vocab": {"a</w>": 0}, "merges": []},
    "normalizer": {"type": "Sequence", "normalizers": [
        {"type": "NFC"}, {"type": "Replace", "pattern": {"Regex": r"\s+"}, "content": " "},
        {"type": "Lowercase"},
    ]},
    "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
        {"type": "Split", "pattern": {"Regex": prompt_tokenizers.CLIP_SPLIT}, "behavior": "Removed",
         "invert": True},
        {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
    ]},
    "added_tokens": [{**_special(49406, "<|startoftext|>"), "normalized": True},
                     _special(49407, "<|endoftext|>")],
}


def test_the_checkpoint_clip_layout_is_accepted():
    prompt_tokenizers.check_clip_layout(json.dumps(CLIP).encode(), "tokenizer")


def _byte_level_regex_off(t):
    t["pre_tokenizer"]["pretokenizers"][1]["use_regex"] = False


def _byte_level_prefix_space(t):
    t["pre_tokenizer"]["pretokenizers"][1]["add_prefix_space"] = True


def _ignore_merges(t):
    t["model"]["ignore_merges"] = True


def _no_lowercase(t):
    t["normalizer"]["normalizers"].pop()


def _single_word_token(t):
    t["added_tokens"][1]["single_word"] = True


@pytest.mark.parametrize("change", [_byte_level_regex_off, _byte_level_prefix_space, _ignore_merges,
                                    _no_lowercase, _single_word_token],
                         ids=lambda change: change.__name__.strip("_"))
def test_unverified_clip_layouts_are_refused(change):
    tokenizer = copy.deepcopy(CLIP)
    change(tokenizer)
    with pytest.raises(NotImplementedError):
        prompt_tokenizers.check_clip_layout(json.dumps(tokenizer).encode(), "tokenizer")
