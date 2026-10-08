# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A built DeBERTa bundle must carry its native byte-level BPE input contract."""

import json
from pathlib import Path
import struct

import pytest

pytest.importorskip("tensorrt")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")


def _sections(path: Path) -> dict[str, bytes]:
    data = path.read_bytes()
    assert data[:8] == b"BUNDLE\x01\x00"
    header_size = struct.unpack_from("<Q", data, 8)[0]
    header = json.loads(data[16 : 16 + header_size])
    start = 16 + header_size
    return {
        name: data[start + section["offset"] : start + section["offset"] + section["length"]]
        for name, section in header["sections"].items()
    }


@pytest.mark.parametrize("existing_json", [False, True])
def test_build_embeds_consumable_bpe_without_mutating_checkpoint(
    tmp_path, monkeypatch, existing_json
):
    from families.deberta import model
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    tokenizer = tokenizers.ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        ["hello world", "HELLO worlds", "hello hello"],
        vocab_size=300,
        special_tokens=["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"],
    )
    tokenizer.save_model(str(checkpoint))
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deberta",
                "vocab_size": tokenizer.get_vocab_size(),
                "hidden_size": 8,
                "num_attention_heads": 2,
                "num_hidden_layers": 1,
                "intermediate_size": 16,
                "max_position_embeddings": 32,
            }
        )
    )
    reference = transformers.AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    if existing_json:
        reference.backend_tokenizer.save(str(checkpoint / "tokenizer.json"))
    original = {path.name: path.read_bytes() for path in checkpoint.iterdir()}
    monkeypatch.setattr(model._DebertaModel, "load_weights", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        model._DebertaModel, "build_engine", lambda *args, **kwargs: b"engine-boundary"
    )
    bundle = tmp_path / "deberta.bundle"
    writer = BundleWriter(bundle)
    model.build(
        BuildRequest(
            model_dir=checkpoint,
            output_path=bundle,
            family="deberta",
            task="encoding",
            precision="fp32",
            max_sequence_length=16,
        ),
        writer,
    )
    writer.finish()

    sections = _sections(bundle)
    assert json.loads(sections["tokenizer.json"])["model"]["type"] == "BPE"
    decoded = tokenizers.Tokenizer.from_str(sections["tokenizer.json"].decode())
    runtime = json.loads(sections["runtime.json"])
    for text in ("hello world", "HELLO worlds", "unrecognized"):
        actual = (
            runtime["tokenizer_prefix_ids"]
            + decoded.encode(text, add_special_tokens=runtime["tokenizer_add_special_tokens"]).ids
            + runtime["tokenizer_suffix_ids"]
        )
        assert actual == reference.encode(text)
    assert {path.name: path.read_bytes() for path in checkpoint.iterdir()} == original
