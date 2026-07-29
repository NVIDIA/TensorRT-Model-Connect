# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the DeepSeek HF reference backend."""

from __future__ import annotations

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
