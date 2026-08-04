# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from pathlib import Path
import tomllib

import pytest

from tensorrt_model_connect.families.minimax_h3.config import (
    MiniMaxH3Config,
    SOL_ENGINE_1344X768_124F,
)
from tensorrt_model_connect.families.minimax_h3.plugin import MiniMaxH3Plugin


FAMILY_ROOT = Path(__file__).resolve().parents[1]


def test_sol_engine_profile_matches_public_packed_shape() -> None:
    profile = SOL_ENGINE_1344X768_124F
    profile.validate()
    assert profile.sequence_length == 38247
    assert profile.padding_rows == 25
    assert profile.padded_sequence_length // profile.context_parallel_size == 4784
    assert profile.attention_size == 7168
    assert profile.video_patch_dim == 96


def test_invalid_context_parallel_contract_fails_closed() -> None:
    with pytest.raises(ValueError, match="padded_sequence_length"):
        MiniMaxH3Config(padded_sequence_length=38271).validate()
    with pytest.raises(ValueError, match="num_heads"):
        MiniMaxH3Config(context_parallel_size=4, num_heads=55).validate()


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
    assert "add_rotary_embedding" in ops
    assert "add_attention" in ops
    assert "CollectiveOperation.ALL_TO_ALL" in ops
    assert "SolAttn" not in "\n".join((adaln, dit, ops))
    assert "FirstBlockCache" not in "\n".join((adaln, dit, ops))
