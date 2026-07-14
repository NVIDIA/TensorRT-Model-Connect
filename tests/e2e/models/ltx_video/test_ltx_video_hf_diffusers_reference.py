# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the LTX Video Hugging Face diffusers reference."""

from __future__ import annotations

import sys
from types import SimpleNamespace

from tensorrt_model_connect.hf_snapshot import hf_snapshot_allow_patterns

from tests.e2e.models.ltx_video.e2e_plugins.references.hf_diffusers import (
    _resolve_cached_model_ref,
)


def test_resolve_cached_model_ref_uses_selective_snapshot_patterns(
    monkeypatch, tmp_path
) -> None:
    snapshot_path = tmp_path / "snapshot"
    snapshot_path.mkdir()
    expected_patterns = hf_snapshot_allow_patterns()

    def snapshot_download(repo_id: str, **kwargs) -> str:
        assert repo_id == "Lightricks/LTX-Video"
        assert kwargs["local_files_only"] is True
        assert kwargs["allow_patterns"] == expected_patterns
        return str(snapshot_path)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )

    assert _resolve_cached_model_ref("Lightricks/LTX-Video") == str(snapshot_path)
