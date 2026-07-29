# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the SAM3 Hugging Face reference contract."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tests.e2e.models.sam3.e2e_plugins import reference
from tests.e2e.models.sam3.e2e_plugins.references import hf_transformers
from tests.e2e_harness.manifest_loader import load_manifest


SAM3_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"


def test_sam3_manifest_pins_gated_snapshot() -> None:
    manifest_path = Path(__file__).parent / "manifests" / "sam3.json"
    case = load_manifest(manifest_path)

    assert case.hf_id == "facebook/sam3"
    assert case.hf_revision == SAM3_REVISION


def test_cached_model_resolution_honors_sam3_revision(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_snapshot_download(repo_id: str, **kwargs: object) -> str:
        calls.append((repo_id, kwargs))
        return "/cache/pinned-sam3-snapshot"

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    resolved = hf_transformers._resolve_cached_model_ref(
        "facebook/sam3",
        SAM3_REVISION,
    )

    assert resolved == "/cache/pinned-sam3-snapshot"
    assert calls == [
        (
            "facebook/sam3",
            {
                "local_files_only": True,
                "revision": SAM3_REVISION,
            },
        )
    ]


def test_missing_pinned_sam3_snapshot_fails_without_network(monkeypatch) -> None:
    def missing_snapshot(*_args: object, **_kwargs: object) -> str:
        raise FileNotFoundError("processor_config.json is not cached")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            snapshot_download=missing_snapshot,
            try_to_load_from_cache=lambda *_args, **_kwargs: None,
        ),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "pinned HF snapshot is unavailable offline: "
            f"facebook/sam3@{SAM3_REVISION}"
        ),
    ):
        hf_transformers._resolve_cached_model_ref(
            "facebook/sam3",
            SAM3_REVISION,
        )


def test_incomplete_hub_snapshot_uses_pinned_runtime_files(
    monkeypatch,
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / SAM3_REVISION
    snapshot.mkdir()
    for name in (
        "config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "model.safetensors",
    ):
        (snapshot / name).touch()

    def incomplete_snapshot(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("optional LICENSE and sam3.pt are absent")

    def cached_file(
        _repo_id: str,
        filename: str,
        **kwargs: object,
    ) -> str:
        assert filename == "config.json"
        assert kwargs == {"revision": SAM3_REVISION}
        return str(snapshot / filename)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            snapshot_download=incomplete_snapshot,
            try_to_load_from_cache=cached_file,
        ),
    )

    resolved = hf_transformers._resolve_cached_model_ref(
        "facebook/sam3",
        SAM3_REVISION,
    )

    assert resolved == str(snapshot)


def test_sam3_reference_propagates_pinned_snapshot(monkeypatch) -> None:
    captured: dict[str, object] = {}
    expected = object()

    monkeypatch.setattr(
        reference,
        "_resolve_cached_model_ref",
        lambda _hf_id, revision: (
            "/cache/pinned-sam3-snapshot"
            if revision == SAM3_REVISION
            else ""
        ),
    )
    monkeypatch.setattr(reference, "_reference_env", lambda _ctx: {})

    def fake_run_reference_subprocess(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(
        reference,
        "run_reference_subprocess",
        fake_run_reference_subprocess,
    )
    case = SimpleNamespace(
        name="sam3",
        task_strategy="prompted_segmentation",
        hf_id="facebook/sam3",
        hf_revision=SAM3_REVISION,
        inputs={
            "image": "/data/coco.jpg",
            "text_prompt": "car",
        },
        metadata={
            "precision": "fp32",
            "reference_precision": "fp32",
            "trust_remote_code": False,
        },
    )
    stage = SimpleNamespace(name="full_inference")
    ctx = SimpleNamespace(
        artifacts_dir="",
        reference_python_path=lambda: "/opt/venv/bin/python",
        ld_library_path="",
    )

    result = reference.Sam3HfTransformersReference()._run_prompted_segmentation_ref(
        case,
        stage,
        ctx,
    )

    assert result is expected
    command = captured["command"]
    assert isinstance(command, list)
    script = command[2]
    assert "model_ref = '/cache/pinned-sam3-snapshot'" in script
    assert f"model_revision = '{SAM3_REVISION}'" in script
    assert 'load_kwargs["revision"] = model_revision' in script
