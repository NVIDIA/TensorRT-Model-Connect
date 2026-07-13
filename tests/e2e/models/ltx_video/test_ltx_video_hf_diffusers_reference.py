# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LTX Video Hugging Face reference cache contracts."""

from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType

from tensorrt_model_connect.hf_snapshot import hf_snapshot_allow_patterns
from tests.e2e.models.ltx_video.e2e_plugins.references import hf_diffusers


def test_cached_model_ref_uses_the_selective_snapshot_contract(tmp_path: Path, monkeypatch) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    calls: list[tuple[str, dict[str, object]]] = []
    expected_kwargs = {
        "allow_patterns": hf_snapshot_allow_patterns(),
        "local_files_only": True,
    }

    def fake_snapshot_download(repo_id: str, **kwargs: object) -> str:
        calls.append((repo_id, kwargs))
        if kwargs != expected_kwargs:
            raise RuntimeError("selective snapshot rejected without its allowlist")
        return str(snapshot)

    huggingface_hub = ModuleType("huggingface_hub")
    huggingface_hub.snapshot_download = fake_snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub)

    resolved = hf_diffusers._resolve_cached_model_ref("Lightricks/LTX-Video")

    assert resolved == str(snapshot)
    assert calls == [
        (
            "Lightricks/LTX-Video",
            expected_kwargs,
        )
    ]
