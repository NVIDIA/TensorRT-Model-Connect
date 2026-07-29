# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the DeepSeek HF reference backend."""

from __future__ import annotations

from types import SimpleNamespace

from tests.e2e.models.deepseek_v2.e2e_plugins.references import hf_transformers


def test_cached_model_resolution_honors_pinned_revision(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_snapshot_download(repo_id: str, **kwargs: object) -> str:
        calls.append((repo_id, kwargs))
        return "/cache/pinned-snapshot"

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        fake_snapshot_download,
    )

    resolved = hf_transformers._resolve_cached_model_ref(
        "katuni4ka/tiny-random-deepseek-v3",
        "ba144b0d3331a5892aa588d82722d382be2b6e6b",
    )

    assert resolved == "/cache/pinned-snapshot"
    assert calls == [
        (
            "katuni4ka/tiny-random-deepseek-v3",
            {
                "local_files_only": True,
                "revision": "ba144b0d3331a5892aa588d82722d382be2b6e6b",
            },
        )
    ]


def test_full_generation_propagates_model_owned_experts_backend(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    expected = object()

    monkeypatch.setattr(
        hf_transformers,
        "_resolve_cached_model_ref",
        lambda _hf_id, _revision: "/cache/pinned-snapshot",
    )
    monkeypatch.setattr(hf_transformers, "_reference_env", lambda _ctx: {})

    def fake_run_reference_subprocess(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(
        hf_transformers,
        "run_reference_subprocess",
        fake_run_reference_subprocess,
    )
    case = SimpleNamespace(
        name="deepseek-v2-tiny",
        task_strategy="text_generation_causal",
        hf_id="katuni4ka/tiny-random-deepseek-v3",
        hf_revision="ba144b0d3331a5892aa588d82722d382be2b6e6b",
        inputs={
            "prompt": "The capital of France is",
            "max_new_tokens": 10,
        },
        metadata={
            "precision": "fp16",
            "reference_precision": "fp16",
            "trust_remote_code": False,
            "task_eval": {
                "hf_experts_implementation": "batched_mm",
            },
        },
    )
    stage = SimpleNamespace(name="full_generation")
    ctx = SimpleNamespace(
        artifacts_dir="",
        reference_python_path=lambda: "/opt/venv/bin/python",
    )

    result = hf_transformers.HfTransformersReference()._run_full_generation(
        case,
        stage,
        ctx,
    )

    assert result is expected
    command = captured["command"]
    assert isinstance(command, list)
    script = command[2]
    assert "experts_implementation = 'batched_mm'" in script
    assert (
        'load_kwargs["experts_implementation"] = experts_implementation'
        in script
    )
    assert captured["metadata"] == {
        "trust_remote_code": False,
        "experts_implementation": "batched_mm",
    }
