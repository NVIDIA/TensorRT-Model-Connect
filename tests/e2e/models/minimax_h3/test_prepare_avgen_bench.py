# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tests.e2e.models.minimax_h3 import prepare_avgen_bench as prepare


class _WhitespaceTokenizer:
    def encode(self, prompt: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(range(len(prompt.split())))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    source = tmp_path / "source"
    prompts = source / "prompts"
    prompts.mkdir(parents=True)
    rows = {
        "longs": [{"content": "long title", "prompt": "one two three"}],
        "shorts": [{"content": "short title", "prompt": "one"}],
        "medians": [{"content": "median title", "prompt": "one two"}],
    }
    source_hashes = {}
    for category, values in rows.items():
        path = prompts / f"{category}.json"
        path.write_text(json.dumps(values), encoding="utf-8")
        source_hashes[category] = _sha256(path)
    license_path = source / "LICENSE"
    license_path.write_text("MIT fixture\n", encoding="utf-8")
    monkeypatch.setattr(prepare, "CATEGORY_COUNTS", {name: 1 for name in rows})
    monkeypatch.setattr(prepare, "SOURCE_SHA256", source_hashes)
    monkeypatch.setattr(prepare, "AVGEN_LICENSE_SHA256", _sha256(license_path))
    monkeypatch.setattr(
        prepare,
        "REPRESENTATIVES",
        (
            {
                "label": "short",
                "category": "shorts",
                "source_index": 0,
                "source_title": "short title",
                "token_count": 1,
            },
            {
                "label": "median",
                "category": "medians",
                "source_index": 0,
                "source_title": "median title",
                "token_count": 2,
            },
            {
                "label": "long",
                "category": "longs",
                "source_index": 0,
                "source_title": "long title",
                "token_count": 3,
            },
        ),
    )
    return source


def _tokenizer_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    tokenizer = tmp_path / "snapshots" / prepare.MINIMAX_H3_REVISION / "tokenizer"
    tokenizer.mkdir(parents=True)
    tokenizer_json = tokenizer / "tokenizer.json"
    tokenizer_json.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(prepare, "TOKENIZER_JSON_SHA256", _sha256(tokenizer_json))
    return tokenizer


def test_prepare_avgen_bench_preserves_source_order_and_records_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_fixture(tmp_path, monkeypatch)
    tokenizer = _tokenizer_fixture(tmp_path, monkeypatch)
    output = tmp_path / "prepared"

    dataset_path = prepare.prepare_avgen_bench(
        source,
        tokenizer,
        output,
        tokenizer_loader=lambda _path: _WhitespaceTokenizer(),
        source_verifier=lambda _path: None,
    )

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    assert dataset["request_count"] == 3
    assert dataset["token_count"] == {
        "allowed_maximum": 537,
        "maximum": 3,
        "minimum": 1,
    }
    assert [row["token_count"] for row in dataset["requests"]] == [3, 1, 2]
    assert [row["category"].split(",")[-1] for row in dataset["requests"]] == [
        "representative-long",
        "representative-short",
        "representative-median",
    ]
    assert [row["sample_id"] for row in dataset["requests"]] == [
        "longs-000",
        "shorts-000",
        "medians-000",
    ]
    first_prompt = output / dataset["requests"][0]["inputs"]["prompt_file"]
    assert json.loads(first_prompt.read_text(encoding="utf-8")) == {
        "prompt": "one two three",
        "seed": 0,
    }
    assert dataset["requests"][0]["inputs"]["validation_mode"] == "avgen_vis"
    manifest = json.loads((output / "DATASET_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["source"]["revision"] == prepare.AVGEN_REVISION
    assert manifest["source"]["prompts_tree"] == prepare.AVGEN_PROMPTS_TREE
    assert manifest["tokenizer"]["revision"] == prepare.MINIMAX_H3_REVISION
    assert manifest["path_policy"] == "manifest_relative"
    assert manifest["request_count"] == 3
    assert {row["path"] for row in manifest["files"]} == {
        "dataset.json",
        "licenses/AVGEN_BENCH_LICENSE",
        "prompts/longs-000.json",
        "prompts/medians-000.json",
        "prompts/shorts-000.json",
        "provenance/SOURCE.json",
        "upstream/prompts/longs.json",
        "upstream/prompts/medians.json",
        "upstream/prompts/shorts.json",
    }
    provenance = json.loads((output / "provenance" / "SOURCE.json").read_text(encoding="utf-8"))
    assert provenance["source_prompts_tree"] == prepare.AVGEN_PROMPTS_TREE
    assert provenance["tokenizer_file"]["sha256"] == prepare.TOKENIZER_JSON_SHA256


def test_prepare_avgen_bench_rejects_unpinned_tokenizer_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_fixture(tmp_path, monkeypatch)
    tokenizer = tmp_path / "wrong-revision" / "tokenizer"
    tokenizer.mkdir(parents=True)

    with pytest.raises(ValueError, match="pinned MiniMax-H3 snapshot"):
        prepare.prepare_avgen_bench(
            source,
            tokenizer,
            tmp_path / "prepared",
            tokenizer_loader=lambda _path: _WhitespaceTokenizer(),
        )


def test_prepare_avgen_bench_rejects_prompt_outside_dynamic_token_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_fixture(tmp_path, monkeypatch)
    tokenizer = _tokenizer_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(prepare, "MAX_PROMPT_TOKENS", 2)

    with pytest.raises(ValueError, match="outside MiniMax-H3"):
        prepare.prepare_avgen_bench(
            source,
            tokenizer,
            tmp_path / "prepared",
            tokenizer_loader=lambda _path: _WhitespaceTokenizer(),
            source_verifier=lambda _path: None,
        )


def test_prepare_avgen_bench_refuses_to_overwrite_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_fixture(tmp_path, monkeypatch)
    tokenizer = _tokenizer_fixture(tmp_path, monkeypatch)
    output = tmp_path / "prepared"
    output.mkdir()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare.prepare_avgen_bench(
            source,
            tokenizer,
            output,
            tokenizer_loader=lambda _path: _WhitespaceTokenizer(),
        )
