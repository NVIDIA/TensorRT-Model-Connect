# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed checks for the Magpie checkpoint architecture we implement."""

from __future__ import annotations

from pathlib import Path

import pytest

from tensorrt_model_connect.families.magpie_tts.plugin import (
    _validate_supported_checkpoint_architecture,
)
from tests.e2e_harness import orchestrator
from tests.e2e_harness.contracts import RunContext
from tests.e2e_harness.manifest_loader import load_manifest


def test_latest_upstream_architecture_fails_with_actionable_revision_error() -> None:
    latest_checkpoint = {
        **{f"audio_embeddings.{index}.weight": object() for index in range(16)},
        "local_transformer.position_embeddings.weight": object(),
        "local_transformer.layers.0.norm_self.weight": object(),
        "local_transformer.layers.1.norm_self.weight": object(),
    }

    with pytest.raises(
        ValueError,
        match=r"supports 8 codebooks and one local-transformer layer.*hf_revision",
    ):
        _validate_supported_checkpoint_architecture(latest_checkpoint)


def test_supported_checkpoint_contract_accepts_legacy_architecture() -> None:
    supported_checkpoint = {
        **{f"audio_embeddings.{index}.weight": object() for index in range(8)},
        "local_transformer.position_embeddings.weight": object(),
        "local_transformer_in_projection.weight": object(),
        "local_transformer_in_projection.bias": object(),
        "local_transformer.layers.0.norm_self.weight": object(),
    }

    _validate_supported_checkpoint_architecture(supported_checkpoint)


def test_manifest_revision_reaches_validation_repro_command(tmp_path) -> None:
    manifest = Path(__file__).parent / "manifests" / "magpie-tts-357m.json"
    case = load_manifest(manifest)
    context = RunContext(case=case, engine_dir=str(tmp_path))

    expected = "34d7e40da85cabc97f92198889b65cea27bc7fd1"
    assert case.hf_revision == expected
    repro = orchestrator._build_repro_commands(case, context, None, {})
    assert f"--model-revision {expected}" in repro["build_bundle"]
