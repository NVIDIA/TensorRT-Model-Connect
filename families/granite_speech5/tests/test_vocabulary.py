# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The bundle's CTC vocabulary decodes as the reference ParakeetTokenizer does."""

from __future__ import annotations

import copy
import json
import re
from itertools import groupby
from pathlib import Path

import numpy as np
import pytest

from families.granite_speech5 import config as config_module
from families.granite_speech5 import vocabulary as vocabulary_module
from families.granite_speech5.tests._snapshot import model_dir


def _tiny_tokenizer(tokens: list[str]):
    """A tokenizers BPE tokenizer over ``tokens`` with the checkpoint's decoder."""
    from tokenizers import Tokenizer

    content = {
        "version": "1.0", "truncation": None, "padding": None, "added_tokens": [], "normalizer": None,
        "pre_tokenizer": None, "post_processor": None, "decoder": vocabulary_module.DECODER,
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                  "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": True, "ignore_merges": False,
                  "vocab": {token: index for index, token in enumerate(tokens)}, "merges": []},
    }
    return Tokenizer.from_str(json.dumps(content))


def test_byte_tokens_are_read_as_the_byte_fallback_decoder_reads_them():
    """Each candidate decodes alone to its byte (as a character) exactly when byte_value accepts it."""
    candidates = ["<0x41>", "<0x0f>", "<0x+F>", "<0xFG>", "<0x4>", "<0x041>", "<0x-1>", "<0xé>", "<0X41>",
                  "<0x 1>", "<0x_1>", "▁<0x41>", "<0x7F>"]
    tokenizer = _tiny_tokenizer(candidates)
    for index, token in enumerate(candidates):
        decoded = tokenizer.decode([index], skip_special_tokens=False)
        value = vocabulary_module.byte_value(token.replace("▁", " "))
        expected = None if value is None else bytes([value]).decode("utf-8", errors="replace")
        if value is None:
            assert decoded == token.replace("▁", " ").removeprefix(" "), token
        else:
            assert decoded == expected, token


@pytest.fixture(scope="module")
def checkpoint() -> Path:
    return model_dir()


@pytest.fixture(scope="module")
def vocabulary(checkpoint):
    return vocabulary_module.read_vocabulary(checkpoint, 16384, 0)


def test_every_id_has_one_decoding_entry(vocabulary):
    kinds = [kind for kind, _ in vocabulary.entries]
    assert len(kinds) == 16384 and vocabulary.blank_id == 0
    # 255 special tokens are skipped, 256 byte-fallback tokens are bytes, every other id is text.
    assert kinds.count(vocabulary_module.SKIP) == 255
    assert kinds.count(vocabulary_module.BYTE) == 256
    assert sorted(value for kind, value in vocabulary.entries if kind == vocabulary_module.BYTE) == list(range(256))
    assert not any("▁" in value for kind, value in vocabulary.entries if kind == vocabulary_module.TEXT)
    assert vocabulary.to_json()["entries"][0] == [vocabulary_module.SKIP, ""]


def _frames(rng: np.random.Generator, pools: list[list[int]], length: int) -> list[int]:
    ids = []
    while len(ids) < length:
        pool = pools[int(rng.integers(len(pools)))]
        ids.extend([int(pool[int(rng.integers(len(pool)))])] * int(rng.integers(1, 4)))
    return ids[:length]


def test_decoding_matches_the_reference_tokenizer(checkpoint, vocabulary):
    """Random frame-id sequences (repeats, blanks, specials, byte runs, CJK and Latin pieces)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    assert type(tokenizer).__name__ == "ParakeetTokenizer" and tokenizer.pad_token_id == 0
    vocab = tokenizer.get_vocab()
    by_kind = {kind: [index for index, (entry, _) in enumerate(vocabulary.entries) if entry == kind]
               for kind in (vocabulary_module.SKIP, vocabulary_module.TEXT, vocabulary_module.BYTE)}
    cjk = [index for piece, index in vocab.items() if any(ord(char) > 0x3000 for char in piece)]
    accented = [index for piece, index in vocab.items() if any(0xC0 <= ord(char) <= 0x17F for char in piece)]
    # Byte runs that are valid UTF-8 (Japanese, an emoji) and runs that are not.
    runs = [list("あいう".encode()), list("😀".encode()), [0xE3, 0x81], [0xFF], [0xC0, 0xAF], [0xED, 0xA0, 0x80]]
    byte_ids = {value: index for index, (kind, value) in enumerate(vocabulary.entries) if kind == vocabulary_module.BYTE}
    rng = np.random.default_rng(2026)
    pools = [[0], by_kind[vocabulary_module.SKIP], by_kind[vocabulary_module.TEXT], cjk, accented,
             by_kind[vocabulary_module.BYTE]]
    checked = 0
    for _ in range(3000):
        frames = _frames(rng, pools, int(rng.integers(0, 60)))
        if rng.random() < 0.3:  # splice whole byte runs, sometimes with a special token inside
            run = [byte_ids[value] for value in runs[int(rng.integers(len(runs)))]]
            if rng.random() < 0.5 and len(run) > 1:
                run.insert(1, int(rng.choice(by_kind[vocabulary_module.SKIP])))
            at = int(rng.integers(len(frames) + 1))
            frames[at:at] = run
        expected = tokenizer.decode(frames, skip_special_tokens=True)
        collapsed = vocabulary_module.collapse(frames, 0)
        assert collapsed == [key for key, _ in groupby(frames) if key != 0]
        assert vocabulary_module.decode(vocabulary, collapsed) == expected, frames
        checked += 1
    assert checked == 3000


def _tokenizer_json(checkpoint: Path) -> dict:
    return json.loads((checkpoint / "tokenizer.json").read_text(encoding="utf-8"))


def _write(directory: Path, content: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "tokenizer.json").write_text(json.dumps(content), encoding="utf-8")
    return directory


def test_another_decoder_is_refused(checkpoint, tmp_path):
    content = _tokenizer_json(checkpoint)
    changed = copy.deepcopy(content)
    changed["decoder"]["decoders"][-1]["start"] = 2
    with pytest.raises(NotImplementedError, match="decoder"):
        vocabulary_module.read_vocabulary(_write(tmp_path / "strip", changed), 16384, 0)
    changed = copy.deepcopy(content)
    changed["model"]["type"] = "WordPiece"
    with pytest.raises(NotImplementedError, match="BPE"):
        vocabulary_module.read_vocabulary(_write(tmp_path / "model", changed), 16384, 0)


def test_ids_must_be_unique_and_complete(checkpoint, tmp_path):
    content = _tokenizer_json(checkpoint)
    changed = copy.deepcopy(content)
    changed["model"]["vocab"]["▁duplicate"] = 5000
    with pytest.raises(ValueError, match="two pieces"):
        vocabulary_module.read_vocabulary(_write(tmp_path / "vocab", changed), 16384, 0)
    with pytest.raises(ValueError, match="fewer than the 16385 ids"):
        vocabulary_module.read_vocabulary(checkpoint, 16385, 0)
    changed = copy.deepcopy(content)
    piece = next(piece for piece, index in changed["model"]["vocab"].items() if index == 9000)
    changed["model"]["vocab"][piece] = 20000  # as many tokens as ids, but id 9000 has none
    with pytest.raises(ValueError, match=r"no token for the ids \[9000\]"):
        vocabulary_module.read_vocabulary(_write(tmp_path / "gap", changed), 16384, 0)


def _added(token_id: int, content: str, special: bool = True) -> dict:
    return {"id": token_id, "content": content, "single_word": False, "lstrip": False, "rstrip": False,
            "normalized": False, "special": special}


def _small(vocab: dict[str, int], added: list[dict]) -> dict:
    return {"version": "1.0", "truncation": None, "padding": None, "added_tokens": added, "normalizer": None,
            "pre_tokenizer": None, "post_processor": None, "decoder": vocabulary_module.DECODER,
            "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                      "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": True, "ignore_merges": False,
                      "vocab": vocab, "merges": []}}


def test_added_token_ids_resolve_as_the_tokenizer_resolves_them(checkpoint, tmp_path):
    """Existing content keeps its model id, new content follows the vocabulary; stored ids must agree."""
    from tokenizers import Tokenizer

    real = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    pieces = vocabulary_module.read_pieces(checkpoint)
    assert all(pieces[item["content"]] == real.token_to_id(item["content"])
               for item in _tokenizer_json(checkpoint)["added_tokens"])
    good = _small({"a": 0, "b": 1, "c": 2}, [_added(1, "b"), _added(3, "<|x|>"), _added(4, "<|y|>", False)])
    library = Tokenizer.from_str(json.dumps(good))
    pieces = vocabulary_module.read_pieces(_write(tmp_path / "good", good))
    assert {content: pieces[content] for content in ("b", "<|x|>", "<|y|>")} == {
        content: library.token_to_id(content) for content in ("b", "<|x|>", "<|y|>")}
    vocabulary = vocabulary_module.read_vocabulary(tmp_path / "good", 5, 0)
    assert vocabulary_module.decode(vocabulary, [0, 3, 4]) == library.decode([0, 3, 4], skip_special_tokens=True)
    refused = {
        "existing content stored under another id": [_added(2, "a")],
        "new content stored under another id": [_added(7, "<|x|>")],
        "a repeated added token": [_added(3, "<|x|>"), _added(3, "<|x|>", False)],
    }
    for index, (case, added) in enumerate(refused.items()):
        with pytest.raises(ValueError, match="stores the added token|repeats the added token"):
            vocabulary_module.read_pieces(_write(tmp_path / f"refused-{index}", _small({"a": 0, "b": 1, "c": 2}, added)))
    # A vocabulary with a gap: its size is 3, so new content takes id 3, which the piece "d" already holds.
    with pytest.raises(ValueError, match="would take the id 3 of the piece 'd'"):
        vocabulary_module.read_pieces(_write(tmp_path / "gappy", _small({"a": 0, "b": 1, "d": 3}, [_added(3, "<|x|>")])))


@pytest.mark.parametrize(("change", "message"), [
    (lambda content: content["model"].pop("merges"), "Missing vocab/merges"),
    (lambda content: content["model"].__setitem__("merges", [["▁the", "▁unknown-piece"]]), "out of vocabulary"),
    (lambda content: content["model"].__setitem__("dropout", 2.0), "Dropout"),
])
def test_a_tokenizer_the_library_refuses_is_refused(checkpoint, tmp_path, change, message):
    """The reference cannot load these files; the bundle must not build from them either."""
    content = _tokenizer_json(checkpoint)
    change(content)
    with pytest.raises(ValueError, match=f"does not load in the tokenizers library: .*{message}"):
        vocabulary_module.read_vocabulary(_write(tmp_path / "broken", content), 16384, 0)


def test_the_table_is_checked_against_the_library(checkpoint, tmp_path, monkeypatch):
    """A decoding table that disagrees with the library for one id is refused, not built."""
    real = vocabulary_module._token_table

    def shifted(tokenizer):
        by_id, pieces, special = real(tokenizer)
        by_id = dict(by_id)
        by_id[300], by_id[301] = by_id[301], by_id[300]
        return by_id, pieces, special

    monkeypatch.setattr(vocabulary_module, "_token_table", shifted)
    with pytest.raises(ValueError, match="token 300 differs from the tokenizers library"):
        vocabulary_module.read_vocabulary(checkpoint, 16384, 0)


_PROBES = [[2236, 2043, 2043, 3], [2043, 2236], [320, 2043, 320], [482, 384, 385, 2043], [2043], [320], [3, 1, 2236],
           [3091, 728]]  # the last is "word .", which Transformers' clean-up would turn into "word."


def _reference_decodes(directory: Path) -> list[str]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(directory))
    return [tokenizer.decode(ids, skip_special_tokens=True) for ids in _PROBES]


def _variant(checkpoint: Path, directory: Path, config: dict, files: dict | None = None) -> Path:
    """The checkpoint's configuration and tokenizer files, ``config`` merged into tokenizer_config.json, ``files``
    added."""
    _write(directory, _tokenizer_json(checkpoint))
    for name in ("config.json", "processor_config.json"):
        (directory / name).write_text((checkpoint / name).read_text(encoding="utf-8"), encoding="utf-8")
    content = json.loads((checkpoint / "tokenizer_config.json").read_text(encoding="utf-8"))
    content.update(config)
    (directory / "tokenizer_config.json").write_text(json.dumps(content), encoding="utf-8")
    for name, value in (files or {}).items():
        (directory / name).write_text(json.dumps(value), encoding="utf-8")
    return directory


def _entry(content: str, special: bool, normalized: bool = False) -> dict:
    return {"content": content, "lstrip": False, "normalized": normalized, "rstrip": False, "single_word": False,
            "special": special}


_AS_ADDED_TOKEN = {"__type": "AddedToken", "content": "▁the", "special": True}


@pytest.mark.parametrize(("config", "files", "message"), [
    ({"eos_token": "▁the"}, {}, "makes ['▁the'] special"),
    ({"additional_special_tokens": ["<|nospeech|>", "▁the"]}, {}, "makes ['▁the'] special"),
    ({"extra_special_tokens": {"task": "▁the"}}, {}, "makes ['▁the'] special"),
    ({"image_token": "▁the"}, {}, "makes ['▁the'] special"),
    ({"image_token": _AS_ADDED_TOKEN}, {}, "makes ['▁the'] special"),
    ({"model_specific_special_tokens": {"audio_token": "▁the"}}, {}, "makes ['▁the'] special"),
    ({}, {"special_tokens_map.json": {"additional_special_tokens": ["▁the"]}}, "makes ['▁the'] special"),
    ({}, {"added_tokens.json": {"▁the": 2043}}, "does not read added_tokens.json"),
    ({"added_tokens_decoder": {"2043": _entry("▁the", True)}}, {}, "added_tokens_decoder entry"),
    ({"added_tokens_decoder": {"2043": _entry("▁the", False, normalized=True)}}, {}, "added_tokens_decoder entry"),
    ({"added_tokens_decoder": {"320": _entry("<0x41>", True)}}, {}, "added_tokens_decoder entry"),
], ids=["eos", "additional", "extra", "custom", "custom-added-token", "model-specific", "special-tokens-map",
        "added-tokens-json", "decoder-special", "decoder-normalized", "decoder-byte"])
def test_configuration_that_changes_the_reference_is_refused(checkpoint, tmp_path, config, files, message):
    """Transformers applies these on load, so the reference transcript changes; the bundle refuses them."""
    directory = _variant(checkpoint, tmp_path / "variant", config, files)
    assert _reference_decodes(directory) != _reference_decodes(checkpoint)
    with pytest.raises(NotImplementedError, match=re.escape(message)):
        vocabulary_module.read_vocabulary(directory, 16384, 0)


def _decoder_entries(checkpoint: Path) -> dict:
    fields = vocabulary_module._ADDED_TOKEN_FIELDS
    return {str(item["id"]): {field: item[field] for field in fields}
            for item in _tokenizer_json(checkpoint)["added_tokens"]}


@pytest.mark.parametrize("config", [
    {"eos_token": {"__type": "AddedToken", "content": "<|endoftext|>"}, "additional_special_tokens": ["<|nospeech|>"]},
    {"image_token": "<|nospeech|>", "add_bos_token": True, "pad_token": None},
    "added_tokens_decoder",  # the entries a Transformers 4 checkpoint writes: every added token, unchanged
], ids=["special", "custom-special", "decoder"])
def test_configuration_that_tokenizer_json_holds_decodes_like_the_reference(checkpoint, tmp_path, config):
    if config == "added_tokens_decoder":
        config = {"added_tokens_decoder": _decoder_entries(checkpoint)}
    directory = _variant(checkpoint, tmp_path / "variant", config)
    vocabulary = vocabulary_module.read_vocabulary(directory, 16384, 0)
    ours = [vocabulary_module.decode(vocabulary, vocabulary_module.collapse(ids, 0)) for ids in _PROBES]
    assert ours == _reference_decodes(directory) == _reference_decodes(checkpoint)


@pytest.mark.parametrize("entries", [None, []])
def test_an_added_tokens_decoder_that_is_no_object_is_refused(checkpoint, tmp_path, entries):
    """The reference fails to load it (a null is not an absent key)."""
    directory = _variant(checkpoint, tmp_path / "variant", {"added_tokens_decoder": entries})
    with pytest.raises(AttributeError):
        _reference_decodes(directory)
    with pytest.raises(ValueError, match="added_tokens_decoder must be an object"):
        vocabulary_module.read_vocabulary(directory, 16384, 0)


def test_added_tokens_decoder_keys_are_canonical_ids(checkpoint, tmp_path):
    entry = _decoder_entries(checkpoint)["3"]
    for key in ("x", "2.0", "03", "+3", " 3"):
        directory = _variant(checkpoint, tmp_path / "key", {"added_tokens_decoder": {key: entry}})
        with pytest.raises(ValueError, match="is not a canonical decimal id"):
            vocabulary_module.read_vocabulary(directory, 16384, 0)


_CLEAN_UP = {"clean_up_tokenization_spaces": True,
             "clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output": True}


def test_a_legacy_special_tokens_map_that_turns_on_the_clean_up_is_refused(checkpoint, tmp_path):
    """Without an added_tokens_decoder, Transformers merges special_tokens_map.json into its settings."""
    directory = _variant(checkpoint, tmp_path / "map", {}, {"special_tokens_map.json": _CLEAN_UP})
    assert _reference_decodes(directory)[-1] == "word." != _reference_decodes(checkpoint)[-1]
    with pytest.raises(NotImplementedError, match="clean_up_tokenization_spaces"):
        config_module.resolve(directory)


@pytest.mark.parametrize("case", ["special-tokens", "ignored-map"])
def test_a_legacy_special_tokens_map_the_reference_reads_alike_is_accepted(checkpoint, tmp_path, case):
    """A map that repeats the special tokens, or one Transformers skips because the configuration has an
    added_tokens_decoder, leaves the reference's decoding as it is."""
    if case == "special-tokens":
        config, files = {}, {"special_tokens_map.json": {"pad_token": "<unk>", "unk_token": "<unk>"}}
    else:
        ignored = {**_CLEAN_UP, "eos_token": "▁the"}  # would be refused if Transformers read it
        config, files = {"added_tokens_decoder": _decoder_entries(checkpoint)}, {"special_tokens_map.json": ignored}
    directory = _variant(checkpoint, tmp_path / case, config, files)
    assert config_module.resolve(directory).blank_id == 0
    vocabulary = vocabulary_module.read_vocabulary(directory, 16384, 0)
    ours = [vocabulary_module.decode(vocabulary, vocabulary_module.collapse(ids, 0)) for ids in _PROBES]
    assert ours == _reference_decodes(directory) == _reference_decodes(checkpoint)


def test_a_versioned_tokenizer_file_the_reference_would_read_is_refused(checkpoint, tmp_path):
    """fast_tokenizer_files makes Transformers load tokenizer.<version>.json instead of tokenizer.json."""
    versioned = _tokenizer_json(checkpoint)
    versioned["decoder"]["decoders"][-1]["start"] = 0  # keeps the leading space
    directory = _variant(checkpoint, tmp_path / "versioned", {"fast_tokenizer_files": ["tokenizer.5.0.0.json"]},
                         {"tokenizer.5.0.0.json": versioned})
    assert _reference_decodes(directory) != _reference_decodes(checkpoint)
    for read in (vocabulary_module.read_pieces, lambda path: vocabulary_module.read_vocabulary(path, 16384, 0)):
        with pytest.raises(NotImplementedError, match="fast_tokenizer_files"):
            read(directory)


def _added_token(content: str) -> dict:
    return {"__type": "AddedToken", "content": content, "lstrip": False, "rstrip": False, "normalized": False,
            "single_word": False, "special": True}


# tokenizer_config.json changes and special_tokens_map.json, each naming "▁the" (an ordinary piece) or a token
# tokenizer.json already holds as special, in the ways Transformers 5.22.2 merges, overrides or refuses them.
_MERGES = {
    "config-named": ({"eos_token": "▁the"}, None),
    "map-replaces-named": ({"eos_token": "▁the"}, {"eos_token": "<|endoftext|>"}),
    "map-names": ({"eos_token": "<|endoftext|>"}, {"eos_token": "▁the"}),
    "map-removes-named": ({"eos_token": "▁the"}, {"eos_token": None}),
    "config-specific-wins": ({"image_token": "<|endoftext|>"}, {"image_token": "▁the"}),
    "config-specific-kept": ({"image_token": "▁the"}, {"image_token": "<|endoftext|>"}),
    "map-specific": ({}, {"image_token": "▁the"}),
    "map-additional-dropped": ({"additional_special_tokens": ["<|nospeech|>"]}, {"additional_special_tokens": ["▁the"]}),
    "map-additional": ({}, {"additional_special_tokens": ["▁the"]}),
    "map-extra-appended": ({"extra_special_tokens": ["<|nospeech|>"]}, {"extra_special_tokens": ["▁the"]}),
    "map-extra-dict-replaces": ({"extra_special_tokens": {"x": "▁the"}}, {"extra_special_tokens": {"x": "<|endoftext|>"}}),
    "map-named-dict": ({}, {"eos_token": {"content": "▁the"}}),
    "map-specific-dict-refused": ({}, {"model_specific_special_tokens": {"audio_token": "▁the"}}),
    "config-specific-added-token": ({"image_token": _added_token("▁the")}, None),
    "config-specific-plain-dict": ({"image_token": {"content": "▁the"}}, None),
    "config-named-plain-dict": ({"eos_token": {"content": "▁the"}}, None),
    "map-extra-dict-item": ({}, {"extra_special_tokens": [{"content": "<|nospeech|>"}]}),
    "map-extra-dict-item-named": ({}, {"extra_special_tokens": [{"content": "▁the"}]}),
    "map-extra-dict-item-special": ({}, {"extra_special_tokens": [{"content": "<|nospeech|>", "special": True}]}),
    "config-extra-plain-dict": ({"extra_special_tokens": [{"content": "<|nospeech|>"}]}, None),
    "map-additional-plain-dict": ({}, {"additional_special_tokens": [{"content": "▁the"}]}),
    "map-extra-dict-of-dicts": ({}, {"extra_special_tokens": {"x": {"content": "▁the"}}}),
    "config-extra-added-token": ({"extra_special_tokens": [_added_token("▁the")]}, None),
    "map-replaces-specific-added-token": ({"image_token": _added_token("<|endoftext|>")}, {"image_token": "▁the"}),
    "map-replaces-specific-added-token-named": ({"image_token": _added_token("▁the")}, {"image_token": "<|endoftext|>"}),
}


@pytest.mark.parametrize("case", sorted(_MERGES))
def test_special_tokens_merge_as_transformers_merges_them(checkpoint, tmp_path, case):
    """Refused exactly when the reference skips "▁the" or cannot load; otherwise decoded like the reference."""
    config, legacy = _MERGES[case]
    files = {} if legacy is None else {"special_tokens_map.json": legacy}
    directory = _variant(checkpoint, tmp_path / case, config, files)
    try:
        reference = _reference_decodes(directory)
    except (TypeError, ValueError):
        reference = None
    if reference is None or reference != _reference_decodes(checkpoint):
        with pytest.raises((NotImplementedError, ValueError)):
            vocabulary_module.read_vocabulary(directory, 16384, 0)
        return
    vocabulary = vocabulary_module.read_vocabulary(directory, 16384, 0)
    assert [vocabulary_module.decode(vocabulary, vocabulary_module.collapse(ids, 0)) for ids in _PROBES] == reference
