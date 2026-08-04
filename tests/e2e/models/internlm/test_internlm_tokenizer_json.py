# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import sys
import types
from pathlib import Path

import pytest

from tensorrt_model_connect.engine_builder import _ensure_tokenizer_json
from tensorrt_model_connect.families import (
    family_hf_warm_file_specs,
    family_hf_warm_files,
)

tokenizer_json = importlib.import_module(
    "tensorrt_model_connect.families.internlm.tokenizer_json"
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _install_fake_hub(
    monkeypatch: pytest.MonkeyPatch,
    download,
) -> None:
    module = types.ModuleType("huggingface_hub")
    module.hf_hub_download = download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)


class _InternLMTokenizerPlugin:
    @staticmethod
    def ensure_tokenizer_json(model_dir, *, previous_error=None):
        return tokenizer_json.ensure_tokenizer_json(
            model_dir,
            previous_error=previous_error,
        )


def test_internlm_declares_revisioned_official_tokenizer_warm_file() -> None:
    expected_spec = (
        "internlm-official-tokenizer-json",
        tokenizer_json.PINNED_TOKENIZER_REPO_ID,
        tokenizer_json.PINNED_TOKENIZER_FILENAME,
        tokenizer_json.PINNED_TOKENIZER_REVISION,
    )

    assert family_hf_warm_file_specs("internlm") == [expected_spec]
    assert family_hf_warm_files("internlm") == [expected_spec[:3]]


def test_ensure_tokenizer_json_installs_verified_local_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_payload = b"source tokenizer model"
    official_payload = b'{"official": true}'
    artifact = tmp_path / "cached-official-tokenizer.json"
    artifact.write_bytes(official_payload)
    (tmp_path / "tokenizer.model").write_bytes(source_payload)
    calls: list[dict[str, object]] = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(artifact)

    monkeypatch.setattr(
        tokenizer_json,
        "SOURCE_TOKENIZER_MODEL_SHA256",
        _sha256(source_payload),
    )
    monkeypatch.setattr(
        tokenizer_json,
        "PINNED_TOKENIZER_SHA256",
        _sha256(official_payload),
    )
    _install_fake_hub(monkeypatch, fake_download)

    assert tokenizer_json.ensure_tokenizer_json(tmp_path)
    assert (tmp_path / "tokenizer.json").read_bytes() == official_payload
    assert not list(tmp_path.glob(".trtmc-internlm-tokenizer-*.json"))
    assert calls == [
        {
            "repo_id": tokenizer_json.PINNED_TOKENIZER_REPO_ID,
            "filename": tokenizer_json.PINNED_TOKENIZER_FILENAME,
            "revision": tokenizer_json.PINNED_TOKENIZER_REVISION,
            "local_files_only": True,
        }
    ]


def test_non_target_source_without_json_preserves_generic_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "tokenizer.model").write_bytes(b"wrong source")
    downloads: list[dict[str, object]] = []
    _install_fake_hub(
        monkeypatch,
        lambda **kwargs: downloads.append(kwargs),
    )

    assert not tokenizer_json.ensure_tokenizer_json(tmp_path)
    assert downloads == []
    assert not (tmp_path / "tokenizer.json").exists()


def test_non_target_source_with_existing_json_remains_supported(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    existing_payload = b'{"version": "unrelated-valid-contract"}'
    (tmp_path / "tokenizer.model").write_bytes(b"another InternLM source")
    (tmp_path / "tokenizer.json").write_bytes(existing_payload)
    downloads: list[dict[str, object]] = []
    _install_fake_hub(
        monkeypatch,
        lambda **kwargs: downloads.append(kwargs),
    )

    _ensure_tokenizer_json(tmp_path, plugin=_InternLMTokenizerPlugin())

    assert (tmp_path / "tokenizer.json").read_bytes() == existing_payload
    assert downloads == []


def test_pinned_tokenizer_hash_mismatch_is_gating(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_payload = b"expected source"
    artifact = tmp_path / "wrong-official-tokenizer.json"
    artifact.write_bytes(b"wrong official JSON")
    (tmp_path / "tokenizer.model").write_bytes(source_payload)
    monkeypatch.setattr(
        tokenizer_json,
        "SOURCE_TOKENIZER_MODEL_SHA256",
        _sha256(source_payload),
    )
    _install_fake_hub(
        monkeypatch,
        lambda **_kwargs: str(artifact),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "generic conversion failed; pinned family tokenizer install failed: "
            "pinned official InternLM tokenizer.json SHA256 mismatch"
        ),
    ):
        tokenizer_json.ensure_tokenizer_json(
            tmp_path,
            previous_error="generic conversion failed",
        )

    assert not (tmp_path / "tokenizer.json").exists()


def test_builder_revalidates_existing_tokenizer_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_payload = b"expected source"
    official_payload = b"expected official JSON"
    artifact = tmp_path / "cached-official-tokenizer.json"
    artifact.write_bytes(official_payload)
    (tmp_path / "tokenizer.model").write_bytes(source_payload)
    (tmp_path / "tokenizer.json").write_bytes(b"stale tokenizer JSON")
    monkeypatch.setattr(
        tokenizer_json,
        "SOURCE_TOKENIZER_MODEL_SHA256",
        _sha256(source_payload),
    )
    monkeypatch.setattr(
        tokenizer_json,
        "PINNED_TOKENIZER_SHA256",
        _sha256(official_payload),
    )
    _install_fake_hub(monkeypatch, lambda **_kwargs: str(artifact))

    with pytest.raises(
        RuntimeError,
        match="installed InternLM tokenizer.json SHA256 mismatch",
    ):
        _ensure_tokenizer_json(
            tmp_path,
            plugin=_InternLMTokenizerPlugin(),
        )


def test_existing_target_json_validates_without_pinned_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_payload = b"expected source"
    official_payload = b"expected official JSON"
    (tmp_path / "tokenizer.model").write_bytes(source_payload)
    (tmp_path / "tokenizer.json").write_bytes(official_payload)
    monkeypatch.setattr(
        tokenizer_json,
        "SOURCE_TOKENIZER_MODEL_SHA256",
        _sha256(source_payload),
    )
    monkeypatch.setattr(
        tokenizer_json,
        "PINNED_TOKENIZER_SHA256",
        _sha256(official_payload),
    )

    def unexpected_download(**_kwargs):
        raise AssertionError("validated installed JSON must not require cache lookup")

    _install_fake_hub(monkeypatch, unexpected_download)

    _ensure_tokenizer_json(tmp_path, plugin=_InternLMTokenizerPlugin())


def test_missing_pinned_tokenizer_never_falls_back_to_network_or_conversion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_payload = b"expected source"
    (tmp_path / "tokenizer.model").write_bytes(source_payload)
    monkeypatch.setattr(
        tokenizer_json,
        "SOURCE_TOKENIZER_MODEL_SHA256",
        _sha256(source_payload),
    )

    def missing(**kwargs):
        assert kwargs["local_files_only"] is True
        raise FileNotFoundError("not warmed")

    _install_fake_hub(monkeypatch, missing)

    with pytest.raises(RuntimeError, match="unavailable in the local Hugging Face cache"):
        tokenizer_json.ensure_tokenizer_json(tmp_path)

    assert not (tmp_path / "tokenizer.json").exists()


@pytest.mark.skipif(
    "TRTMC_MODEL_PROOF_GPU_ID" not in os.environ,
    reason="real checkpoint artifact proof runs in the isolated model profile",
)
def test_real_checkpoint_uses_pinned_official_tokenizer_contract(
    tmp_path: Path,
) -> None:
    """Validate official JSON plus math tokenizer config, not the broken slow path.

    The official slow tokenizer cannot load this model's NUL piece. The oracle
    here is the independently published, revision-pinned official JSON combined
    with the exact math checkpoint's tokenizer_config contract.
    """
    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    manifest = json.loads(
        (
            Path(__file__).with_name("manifests")
            / "internlm2-1.8b.json"
        ).read_text(encoding="utf-8")
    )
    snapshot = Path(
        snapshot_download(
            repo_id=manifest["hf_id"],
            local_files_only=True,
        )
    )
    for source in snapshot.iterdir():
        if source.name == "model.safetensors" or source.is_dir():
            continue
        shutil.copy2(source, tmp_path / source.name, follow_symlinks=True)

    _ensure_tokenizer_json(tmp_path, plugin=_InternLMTokenizerPlugin())

    tokenizer_path = tmp_path / "tokenizer.json"
    assert tokenizer_json._sha256_file(tokenizer_path) == (
        tokenizer_json.PINNED_TOKENIZER_SHA256
    )
    backend = Tokenizer.from_file(str(tokenizer_path))
    assert backend.get_vocab_size(with_added_tokens=True) == 92_544
    assert {
        alias: backend.token_to_id(alias)
        for alias in (
            "<|plugin|>",
            "<|interpreter|>",
            "<|action_end|>",
            "<|action_start|>",
            "<|im_end|>",
            "<|im_start|>",
        )
    } == {
        "<|plugin|>": 92_538,
        "<|interpreter|>": 92_539,
        "<|action_end|>": 92_540,
        "<|action_start|>": 92_541,
        "<|im_end|>": 92_542,
        "<|im_start|>": 92_543,
    }

    config = json.loads(
        (tmp_path / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path),
        bos_token=config.get("bos_token"),
        eos_token=config.get("eos_token"),
        unk_token=config.get("unk_token"),
        pad_token=config.get("pad_token"),
        chat_template=config.get("chat_template"),
        clean_up_tokenization_spaces=config.get(
            "clean_up_tokenization_spaces",
            False,
        ),
    )

    expected_single_ids = {
        "The capital of France is": [1, 918, 6872, 446, 9760, 505],
        "Hello, how are you?": [1, 9843, 328, 1392, 657, 629, 345],
        "你好，世界！": [1, 77230, 60353, 68339, 60477],
        "  leading  and\tinternal\nwhitespace  ": [
            1,
            387,
            20858,
            387,
            568,
            33437,
            364,
            1458,
            36024,
            387,
        ],
        "café naïve — 🚀": [
            1,
            1062,
            283,
            63000,
            4477,
            67656,
            717,
            262,
            60656,
            262,
            243,
            162,
            157,
            131,
        ],
        "": [1],
    }
    for text, expected_ids in expected_single_ids.items():
        assert tokenizer.encode(text) == expected_ids
        assert tokenizer.decode(
            expected_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ) == text

    pair = tokenizer(
        "Hello, how are you?",
        "你好，世界！",
        return_token_type_ids=True,
    )
    assert pair["input_ids"] == [
        1,
        9843,
        328,
        1392,
        657,
        629,
        345,
        1,
        77230,
        60353,
        68339,
        60477,
    ]
    assert pair["token_type_ids"] == [0] * 7 + [1] * 5

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "The capital of France is"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert rendered == (
        "<s><|im_start|>user\nThe capital of France is<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert tokenizer.encode(rendered, add_special_tokens=False) == [
        1,
        92543,
        1008,
        364,
        918,
        6872,
        446,
        9760,
        505,
        92542,
        364,
        92543,
        525,
        11353,
        364,
    ]
