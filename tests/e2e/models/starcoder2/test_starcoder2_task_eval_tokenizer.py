# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""StarCoder2-owned task-eval tokenizer contract regression."""

from __future__ import annotations

import json
import struct
from pathlib import Path

from tokenizers import Tokenizer

from tools.validation.engine import (
    _effective_bundle_tokenizer_payload,
    _load_text_input_contract,
)


def _published_backend_payload() -> bytes:
    return json.dumps(
        {
            "model": {
                "type": "BPE",
                "vocab": {
                    "\u010a": 0,
                    "\u0120": 1,
                    "0": 2,
                    ".": 3,
                    "5": 4,
                    "\u010a\u0120": 5,
                    "\u010a\u0120\u0120": 6,
                    "\u010a\u0120\u0120\u0120": 7,
                    "\u010a\u0120\u0120\u0120\u0120": 8,
                },
                "merges": [
                    "\u010a \u0120",
                    "\u010a\u0120 \u0120",
                    "\u010a\u0120\u0120 \u0120",
                    "\u010a\u0120\u0120\u0120 \u0120",
                ],
            },
            "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [
                    {"type": "Digits", "individual_digits": True},
                    {
                        "type": "ByteLevel",
                        "add_prefix_space": False,
                        "trim_offsets": True,
                        "use_regex": True,
                    },
                ],
            },
            "decoder": {
                "type": "ByteLevel",
                "add_prefix_space": True,
                "trim_offsets": True,
                "use_regex": True,
            },
        },
        separators=(",", ":"),
    ).encode("utf-8")


def test_gpt2_wrapper_uses_effective_byte_level_contract() -> None:
    source = _published_backend_payload()
    raw_ids = Tokenizer.from_str(source.decode("utf-8")).encode("\n    0.5").ids

    effective = _effective_bundle_tokenizer_payload(
        source,
        {
            "tokenizer_class": "GPT2Tokenizer",
            "add_prefix_space": False,
        },
    )
    effective_ids = Tokenizer.from_str(effective.decode("utf-8")).encode(
        "\n    0.5"
    ).ids

    assert raw_ids == [8, 2, 3, 4]
    assert effective_ids == [7, 1, 2, 3, 4]
    assert json.loads(source)["pre_tokenizer"]["type"] == "Sequence"
    assert json.loads(effective)["pre_tokenizer"]["type"] == "ByteLevel"


def test_non_gpt2_wrapper_does_not_refine_backend() -> None:
    source = _published_backend_payload()

    assert (
        _effective_bundle_tokenizer_payload(
            source,
            {"tokenizer_class": "CodeLlamaTokenizer"},
        )
        == source
    )


def _write_bundle(path: Path, sections: dict[str, bytes]) -> None:
    offset = 0
    metadata = {}
    for name, payload in sections.items():
        metadata[name] = {"offset": offset, "size": len(payload)}
        offset += len(payload)
    header = json.dumps({"sections": metadata}).encode("utf-8")
    path.write_bytes(
        b"BUNDLE\x01\x00"
        + struct.pack("<Q", len(header))
        + header
        + b"".join(sections.values())
    )


def test_input_contract_loads_effective_bundled_wrapper(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle = tmp_path / "starcoder2.bundle"
    _write_bundle(
        bundle,
        {
            "config.json": b"{}",
            "tokenizer.json": _published_backend_payload(),
            "tokenizer_config.json": json.dumps(
                {
                    "tokenizer_class": "GPT2Tokenizer",
                    "add_prefix_space": False,
                }
            ).encode("utf-8"),
        },
    )
    hf_tokenizer = object()
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: hf_tokenizer,
    )

    loaded_hf, bundled, config = _load_text_input_contract(
        model={"hf_id": "bigcode/starcoder2-3b"},
        bundle_path=bundle,
        local_files_only=True,
        trust_remote_code=False,
    )

    assert loaded_hf is hf_tokenizer
    assert bundled.encode("\n    0.5").ids == [7, 1, 2, 3, 4]
    assert config == {}
