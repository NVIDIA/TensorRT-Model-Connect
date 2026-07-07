# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from tests.e2e.models.pixart.e2e_plugins.references import hf_diffusers


def test_partial_hf_snapshot_is_not_used_as_offline_model(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model_index.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *_args, **_kwargs: str(snapshot),
    )

    resolved = hf_diffusers._resolve_cached_model_ref(
        "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
    )

    assert resolved == "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
