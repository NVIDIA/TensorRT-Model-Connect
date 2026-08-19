# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from tensorrt_model_connect.models.pixart.tests.e2e_plugins.references import hf_diffusers


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


def test_cached_snapshot_resolution_scopes_offline_lookup_to_model_index(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot = tmp_path / "snapshot"
    for relative_path in (
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "tokenizer/tokenizer_config.json",
        "transformer/config.json",
        "vae/config.json",
    ):
        path = snapshot / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    calls: list[dict] = []

    def fake_snapshot_download(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return str(snapshot)

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        fake_snapshot_download,
    )

    resolved = hf_diffusers._resolve_cached_model_ref(
        "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
    )

    assert resolved == str(snapshot)
    assert calls == [{
        "args": ("PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",),
        "kwargs": {
            "allow_patterns": ["model_index.json"],
            "local_files_only": True,
        },
    }]
