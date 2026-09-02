# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tests.e2e.models.minimax_h3 import prepare_vbench_siglip as prepare


class _WhitespaceTokenizer:
    def encode(self, prompt: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(range(len(prompt.split())))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    source_info = tmp_path / "VBench_full_info.json"
    rows = [
        {"prompt_en": "shared prompt", "dimension": ["motion", "style"]},
        {"prompt_en": "motion only", "dimension": ["motion"]},
        {"prompt_en": "style only words", "dimension": ["style"]},
    ]
    source_info.write_text(json.dumps(rows), encoding="utf-8")
    source_license = tmp_path / "LICENSE"
    source_license.write_text("Apache-2.0 fixture\n", encoding="utf-8")
    monkeypatch.setattr(prepare, "EXPECTED_SOURCE_COUNT", len(rows))
    monkeypatch.setattr(prepare, "SELECTION_DIMENSIONS", ("motion", "style"))
    monkeypatch.setattr(prepare, "PROMPTS_PER_DIMENSION", 1)
    monkeypatch.setattr(prepare, "EXPECTED_PROMPT_COUNT", 2)
    monkeypatch.setattr(prepare, "VBENCH_INFO_SHA256", _sha256(source_info))
    monkeypatch.setattr(prepare, "VBENCH_LICENSE_SHA256", _sha256(source_license))
    return source_info, source_license


def _tokenizer_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    tokenizer = tmp_path / "snapshots" / prepare.MINIMAX_H3_REVISION / "tokenizer"
    tokenizer.mkdir(parents=True)
    tokenizer_json = tokenizer / "tokenizer.json"
    tokenizer_json.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(prepare, "TOKENIZER_JSON_SHA256", _sha256(tokenizer_json))
    return tokenizer


def test_prepare_vbench_siglip_selects_unique_prompts_and_records_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_info, source_license = _source_fixture(tmp_path, monkeypatch)
    tokenizer = _tokenizer_fixture(tmp_path, monkeypatch)
    output = tmp_path / "prepared"

    dataset_path = prepare.prepare_vbench_siglip(
        source_info,
        source_license,
        tokenizer,
        output,
        tokenizer_loader=lambda _path: _WhitespaceTokenizer(),
    )

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    assert dataset["request_count"] == 2
    assert dataset["selection_dimensions"] == ["motion", "style"]
    assert dataset["token_count"] == {
        "allowed_maximum": 537,
        "maximum": 3,
        "minimum": 2,
    }
    assert [row["prompt"] for row in dataset["requests"]] == [
        "shared prompt",
        "style only words",
    ]
    assert [row["selection_dimension"] for row in dataset["requests"]] == [
        "motion",
        "style",
    ]
    assert dataset["requests"][0]["inputs"]["validation_mode"] == "vbench_siglip"
    first_prompt = output / dataset["requests"][0]["inputs"]["prompt_file"]
    assert json.loads(first_prompt.read_text(encoding="utf-8")) == {
        "prompt": "shared prompt",
        "seed": 0,
    }

    manifest = json.loads((output / "DATASET_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["source"] == {
        "repository": prepare.VBENCH_REPOSITORY,
        "revision": prepare.VBENCH_REVISION,
        "info_sha256": prepare.VBENCH_INFO_SHA256,
        "license": "Apache-2.0",
        "license_sha256": prepare.VBENCH_LICENSE_SHA256,
    }
    assert manifest["tokenizer"]["revision"] == prepare.MINIMAX_H3_REVISION
    assert manifest["path_policy"] == "manifest_relative"
    assert manifest["request_count"] == 2
    assert {row["path"] for row in manifest["files"]} == {
        "dataset.json",
        "licenses/VBENCH_LICENSE",
        "prompts/000-motion.json",
        "prompts/001-style.json",
        "provenance/SOURCE.json",
        "upstream/VBench_full_info.json",
    }


def test_prepare_vbench_siglip_rejects_unpinned_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_info, source_license = _source_fixture(tmp_path, monkeypatch)
    tokenizer = _tokenizer_fixture(tmp_path, monkeypatch)
    source_info.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="pinned revision"):
        prepare.prepare_vbench_siglip(
            source_info,
            source_license,
            tokenizer,
            tmp_path / "prepared",
            tokenizer_loader=lambda _path: _WhitespaceTokenizer(),
        )


def test_prepare_vbench_siglip_rejects_unpinned_tokenizer_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_info, source_license = _source_fixture(tmp_path, monkeypatch)
    tokenizer = tmp_path / "wrong-revision" / "tokenizer"
    tokenizer.mkdir(parents=True)

    with pytest.raises(ValueError, match="pinned MiniMax-H3 snapshot"):
        prepare.prepare_vbench_siglip(
            source_info,
            source_license,
            tokenizer,
            tmp_path / "prepared",
            tokenizer_loader=lambda _path: _WhitespaceTokenizer(),
        )


def test_prepare_vbench_siglip_rejects_prompt_outside_dynamic_token_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_info, source_license = _source_fixture(tmp_path, monkeypatch)
    tokenizer = _tokenizer_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(prepare, "MAX_PROMPT_TOKENS", 2)

    with pytest.raises(ValueError, match="outside MiniMax-H3"):
        prepare.prepare_vbench_siglip(
            source_info,
            source_license,
            tokenizer,
            tmp_path / "prepared",
            tokenizer_loader=lambda _path: _WhitespaceTokenizer(),
        )


def test_prepare_vbench_siglip_refuses_to_overwrite_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_info, source_license = _source_fixture(tmp_path, monkeypatch)
    tokenizer = _tokenizer_fixture(tmp_path, monkeypatch)
    output = tmp_path / "prepared"
    output.mkdir()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare.prepare_vbench_siglip(
            source_info,
            source_license,
            tokenizer,
            output,
            tokenizer_loader=lambda _path: _WhitespaceTokenizer(),
        )
