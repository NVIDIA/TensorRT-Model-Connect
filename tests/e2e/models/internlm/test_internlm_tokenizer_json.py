# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
import types

from tensorrt_model_connect.families.internlm.tokenizer_json import (
    ensure_tokenizer_json,
)


def _module(name: str, **members):
    module = types.ModuleType(name)
    for key, value in members.items():
        setattr(module, key, value)
    return module


def test_ensure_tokenizer_json_builds_internlm_bpe(monkeypatch, tmp_path):
    calls = {}

    class FakePiece:
        def __init__(self, piece, score):
            self.piece = piece
            self.score = score

    class FakeModelProto:
        def __init__(self):
            self.trainer_spec = types.SimpleNamespace(
                model_type=2,
                unk_piece="<unk>",
            )
            self.normalizer_spec = types.SimpleNamespace(add_dummy_prefix=True)
            self.pieces = [
                FakePiece("<unk>", 0.0),
                FakePiece("<s>", 0.0),
                FakePiece("</s>", 0.0),
                FakePiece("ab", -1.0),
            ]

        def ParseFromString(self, payload):
            calls["serialized_model"] = payload

    class FakeAddedToken:
        def __init__(self, **spec):
            self.content = spec["content"]

    class FakeBPE:
        def __init__(self, vocab, merges, **kwargs):
            calls["bpe"] = {
                "vocab": vocab,
                "merges": merges,
                **kwargs,
            }

    class FakeTokenizer:
        def __init__(self, model):
            calls["model"] = model

        def add_special_tokens(self, tokens):
            calls["special_tokens"] = [token.content for token in tokens]

        def save(self, path):
            calls["save_path"] = path
            (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")

    def operation(name):
        return lambda *args, **kwargs: (name, args, kwargs)

    def generate_merges(vocab, vocab_scores):
        calls["merge_vocab"] = vocab
        calls["vocab_scores"] = vocab_scores
        return [("a", "b")]

    sentencepiece_model_pb2 = types.SimpleNamespace(ModelProto=FakeModelProto)
    decoders = types.SimpleNamespace(
        Replace=operation("decoder_replace"),
        ByteFallback=operation("byte_fallback"),
        Fuse=operation("fuse"),
        Strip=operation("strip"),
        Sequence=operation("decoder_sequence"),
    )
    normalizers = types.SimpleNamespace(
        Prepend=operation("prepend"),
        Replace=operation("normalizer_replace"),
        Sequence=operation("normalizer_sequence"),
    )
    processors = types.SimpleNamespace(
        TemplateProcessing=operation("template_processing"),
    )

    monkeypatch.setitem(
        sys.modules,
        "sentencepiece",
        _module(
            "sentencepiece",
            sentencepiece_model_pb2=sentencepiece_model_pb2,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tokenizers",
        _module(
            "tokenizers",
            AddedToken=FakeAddedToken,
            Tokenizer=FakeTokenizer,
            decoders=decoders,
            normalizers=normalizers,
            processors=processors,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tokenizers.models",
        _module("tokenizers.models", BPE=FakeBPE),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        _module("transformers"),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers.convert_slow_tokenizer",
        _module(
            "transformers.convert_slow_tokenizer",
            generate_merges=generate_merges,
        ),
    )

    (tmp_path / "tokenizer.model").write_bytes(b"sentencepiece")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "added_tokens_decoder": {
                    str(token_id): {
                        "content": content,
                        "single_word": False,
                        "lstrip": False,
                        "rstrip": False,
                        "normalized": False,
                        "special": True,
                    }
                    for token_id, content in enumerate(
                        ("<unk>", "<s>", "</s>", "ab")
                    )
                },
            }
        ),
        encoding="utf-8",
    )

    assert ensure_tokenizer_json(tmp_path)
    assert calls["serialized_model"] == b"sentencepiece"
    assert calls["bpe"] == {
        "vocab": {"<unk>": 0, "<s>": 1, "</s>": 2, "ab": 3},
        "merges": [("a", "b")],
        "unk_token": "<unk>",
        "fuse_unk": True,
        "byte_fallback": True,
    }
    assert calls["special_tokens"] == ["<unk>", "<s>", "</s>", "ab"]
    assert calls["save_path"] == str(tmp_path / "tokenizer.json")
