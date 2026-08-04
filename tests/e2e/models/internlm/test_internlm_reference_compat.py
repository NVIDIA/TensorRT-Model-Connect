# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib
import inspect
import sys
import types
from pathlib import Path

import pytest

from tests.e2e.models.internlm.e2e_plugins.references import hf_transformers


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _install_fake_hub(monkeypatch: pytest.MonkeyPatch, download) -> None:
    module = types.ModuleType("huggingface_hub")
    module.hf_hub_download = download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)


def test_reference_retries_chat_template_without_thinking_kwarg() -> None:
    source = inspect.getsource(
        hf_transformers.HfTransformersReference._run_full_generation
    )
    assert "except TypeError:" in source
    assert 'chat_kwargs.pop("enable_thinking", None)' in source


def test_reference_uses_independently_pinned_fast_tokenizer() -> None:
    source = inspect.getsource(
        hf_transformers.HfTransformersReference._run_full_generation
    )
    module_source = inspect.getsource(hf_transformers)

    assert "_resolve_reference_tokenizer_json(model_ref)" in source
    assert "PreTrainedTokenizerFast" in source
    assert "tokenizer_file=tokenizer_path" in source
    assert 'tokenizer_dir / "tokenizer.json"' not in source
    assert "if tokenizer_path is None:" in source
    assert "AutoTokenizer.from_pretrained" in source
    assert "families.internlm.tokenizer_json" not in module_source


def test_reference_resolver_is_independent_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_payload = b"target source model"
    official_payload = b'{"official": true}'
    (tmp_path / "tokenizer.model").write_bytes(source_payload)
    artifact = tmp_path / "official-tokenizer.json"
    artifact.write_bytes(official_payload)
    monkeypatch.setattr(
        hf_transformers,
        "_REFERENCE_SOURCE_MODEL_SHA256",
        _sha256(source_payload),
    )
    monkeypatch.setattr(
        hf_transformers,
        "_REFERENCE_TOKENIZER_SHA256",
        _sha256(official_payload),
    )
    calls: list[dict[str, object]] = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(artifact)

    _install_fake_hub(monkeypatch, fake_download)
    production_tokenizer = importlib.import_module(
        "tensorrt_model_connect.families.internlm.tokenizer_json"
    )

    def product_resolver_must_not_run(*_args, **_kwargs):
        raise AssertionError("reference must not call the production resolver")

    monkeypatch.setattr(
        production_tokenizer,
        "resolve_pinned_tokenizer_json",
        product_resolver_must_not_run,
    )

    assert hf_transformers._resolve_reference_tokenizer_json(tmp_path) == artifact
    assert calls == [
        {
            "repo_id": hf_transformers._REFERENCE_TOKENIZER_REPO_ID,
            "filename": hf_transformers._REFERENCE_TOKENIZER_FILENAME,
            "revision": hf_transformers._REFERENCE_TOKENIZER_REVISION,
            "local_files_only": True,
        }
    ]

    artifact.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="reference tokenizer SHA256 mismatch"):
        hf_transformers._resolve_reference_tokenizer_json(tmp_path)


def test_reference_resolver_preserves_non_target_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "tokenizer.model").write_bytes(b"another InternLM source")

    def unexpected_download(**_kwargs):
        raise AssertionError("non-target reference must preserve AutoTokenizer fallback")

    _install_fake_hub(monkeypatch, unexpected_download)

    assert hf_transformers._resolve_reference_tokenizer_json(tmp_path) is None
