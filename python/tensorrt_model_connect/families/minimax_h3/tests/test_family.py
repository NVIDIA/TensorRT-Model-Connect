# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import tomllib

import pytest

from tensorrt_model_connect.families.minimax_h3.config import (
    MiniMaxH3Config,
    SOL_ENGINE_1344X768_124F,
)
from tensorrt_model_connect.families.minimax_h3.plugin import (
    MiniMaxH3Plugin,
    _build_source_revision,
    _fixed_profile,
)


FAMILY_ROOT = Path(__file__).resolve().parents[1]


def test_sol_engine_profile_matches_public_packed_shape() -> None:
    profile = SOL_ENGINE_1344X768_124F
    profile.validate()
    assert profile.sequence_length == 38247
    assert profile.context_parallel_size == 1
    assert profile.padding_rows == 0
    assert profile.padded_sequence_length // profile.context_parallel_size == 38247
    assert profile.attention_size == 7168
    assert profile.video_patch_dim == 96


def test_invalid_single_device_contract_fails_closed() -> None:
    for padded_rows in (38246, 38248):
        with pytest.raises(ValueError, match="no packed-sequence padding"):
            MiniMaxH3Config(padded_sequence_length=padded_rows).validate()
    with pytest.raises(ValueError, match="context_parallel_size=1"):
        MiniMaxH3Config(context_parallel_size=4).validate()


def test_manifest_discovers_both_public_pipeline_names() -> None:
    manifest = tomllib.loads((FAMILY_ROOT / "MODEL.toml").read_text())
    assert manifest["id"] == "minimax_h3"
    assert set(manifest["diffusion_pipeline_classes"]) == {
        "MiniMaxH3ModularPipeline",
        "MiniMaxH3Pipeline",
    }
    assert "transformer/**" in manifest["hf_allow_patterns"]
    assert "vae/**" in manifest["hf_allow_patterns"]


def test_plugin_aliases_are_exact() -> None:
    plugin = MiniMaxH3Plugin()
    for alias in ("minimax-h3", "minimax_h3", "MiniMaxH3Pipeline"):
        assert plugin.matches(alias)
    assert not plugin.matches("minimax-video-01")


def test_plugin_fails_closed_on_unqualified_profile_or_source(monkeypatch) -> None:
    monkeypatch.delenv("TRTMC_MINIMAX_H3_SOURCE_REVISION", raising=False)
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    with pytest.raises(ValueError, match="SOURCE_REVISION"):
        _build_source_revision()
    monkeypatch.setenv("TRTMC_MINIMAX_H3_SOURCE_REVISION", "A" * 40)
    assert _build_source_revision() == "a" * 40
    assert _fixed_profile({}) is SOL_ENGINE_1344X768_124F
    with pytest.raises(ValueError, match="packed-row profile"):
        _fixed_profile({"video_rows": 1})


def test_plugin_bundle_config_preserves_exact_provenance() -> None:
    provenance = {
        "source_revision": "a" * 40,
        "builder_source_sha256": "b" * 64,
        "checkpoint_inventory_sha256": "c" * 64,
        "plan_sha256": {
            "text_encoder.plan": "d" * 64,
            "adaln_precompute.plan": "e" * 64,
            "denoiser.plan": "f" * 64,
            "vae_tile_decoder.plan": "0" * 64,
        },
    }
    config = SimpleNamespace(raw={"seed": 7})
    result = MiniMaxH3Plugin().diffusion_bundle_config(
        config,
        components={"profile": SOL_ENGINE_1344X768_124F, "provenance": provenance},
    )
    assert result | provenance == result
    assert result["seed"] == 7
    assert result["context_parallel_size"] == 1
    with pytest.raises(ValueError, match="runtime profile"):
        MiniMaxH3Plugin().diffusion_bundle_config(
            SimpleNamespace(raw={"video_width": 1}),
            components={"profile": SOL_ENGINE_1344X768_124F, "provenance": provenance},
        )


def test_production_graph_is_native_trt_only() -> None:
    violations = []
    for path in FAMILY_ROOT.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.name for alias in node.names]
                if any(name.startswith(("torch", "torch_tensorrt", "triton")) for name in names):
                    violations.append(f"{path.name}:{node.lineno}: {names}")
            if isinstance(node, ast.Call):
                function = node.func
                name = function.attr if isinstance(function, ast.Attribute) else ""
                if name.startswith("add_plugin") or name == "get_plugin_registry":
                    violations.append(f"{path.name}:{node.lineno}: {name}")
    assert not violations


def test_sol_lossless_optimizations_are_structural() -> None:
    adaln = (FAMILY_ROOT / "adaln_builder.py").read_text()
    dit = (FAMILY_ROOT / "dit_builder.py").read_text()
    ops = (FAMILY_ROOT / "graph_ops.py").read_text()
    assert "block_modulation_" in adaln
    assert "fused_qkv" in dit
    assert "UnaryOperation.NEG" in ops
    assert "add_attention" in ops
    assert "native_attention" in ops
    assert "add_dist_collective" not in ops
    assert 'network.add_input("rank"' not in dit
    assert "layer.mask" not in ops
    assert "SolAttn" not in "\n".join((adaln, dit, ops))
    assert "FirstBlockCache" not in "\n".join((adaln, dit, ops))
