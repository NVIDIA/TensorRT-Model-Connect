# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tomllib

import pytest

from tensorrt_model_connect.models.minimax_h3.config import (
    DEFAULT_WORKSPACE_LIMIT_BYTES,
    FIRST_BLOCK_CACHE_DENOISER_PLAN_FILENAMES,
    MiniMaxH3Config,
    SOL_ENGINE_1344X768_124F,
    default_workspace_limit_bytes,
)
from tensorrt_model_connect.models.minimax_h3 import model as MiniMaxH3Model
from tensorrt_model_connect.models.minimax_h3.model import (
    _build_source_revision,
    _effective_build_config,
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
    assert profile.first_block_cache is False


def test_invalid_single_device_contract_fails_closed() -> None:
    for padded_rows in (38246, 38248):
        with pytest.raises(ValueError, match="no packed-sequence padding"):
            MiniMaxH3Config(padded_sequence_length=padded_rows).validate()
    with pytest.raises(ValueError, match="context_parallel_size=1"):
        MiniMaxH3Config(context_parallel_size=4).validate()
    with pytest.raises(ValueError, match="must be a boolean"):
        MiniMaxH3Config(first_block_cache=1).validate()


def test_manifest_discovers_both_public_pipeline_names() -> None:
    manifest = tomllib.loads((FAMILY_ROOT / "MODEL.toml").read_text())
    assert manifest["id"] == "minimax_h3"
    assert set(manifest["diffusion_pipeline_classes"]) == {
        "MiniMaxH3ModularPipeline",
        "MiniMaxH3Pipeline",
    }
    assert set(manifest["hf_allow_patterns"]) == {
        "model_index.json",
        "modular_model_index.json",
        "processor/**",
        "scheduler/**",
        "audio_scheduler/**",
        "text_encoder/**",
        "tokenizer/**",
        "transformer/**",
        "transformer_ref/**",
        "vae/**",
        "audio_vae/**",
    }


def test_plugin_aliases_are_exact() -> None:
    plugin = MiniMaxH3Model
    for alias in ("minimax-h3", "minimax_h3", "MiniMaxH3Pipeline"):
        assert plugin.matches(SimpleNamespace(model_type=alias, raw={}))
    assert not plugin.matches(SimpleNamespace(model_type="minimax-video-01", raw={}))


def test_plugin_fails_closed_on_unqualified_profile_or_source(monkeypatch) -> None:
    monkeypatch.delenv("TRTMC_MINIMAX_H3_SOURCE_REVISION", raising=False)
    monkeypatch.delenv("TRTMC_ENGINE_BUILD_REVISION", raising=False)
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    with pytest.raises(ValueError, match="SOURCE_REVISION"):
        _build_source_revision()
    monkeypatch.setenv("TRTMC_ENGINE_BUILD_REVISION", "not-a-sha")
    monkeypatch.setenv("GITHUB_SHA", "C" * 40)
    with pytest.raises(ValueError, match="40-character Git SHA"):
        _build_source_revision()
    monkeypatch.setenv("TRTMC_ENGINE_BUILD_REVISION", "B" * 40)
    assert _build_source_revision() == "b" * 40
    monkeypatch.setenv("TRTMC_MINIMAX_H3_SOURCE_REVISION", "A" * 40)
    assert _build_source_revision() == "a" * 40
    assert _fixed_profile({}) is SOL_ENGINE_1344X768_124F
    assert _fixed_profile({"first_block_cache": True}).first_block_cache is True
    assert _fixed_profile({"denoiser_cache_mode": "first_block"}).first_block_cache is True
    with pytest.raises(ValueError, match="disagree"):
        _fixed_profile({"denoiser_cache_mode": "first_block", "first_block_cache": False})
    with pytest.raises(ValueError, match="packed-row profile"):
        _fixed_profile({"video_rows": 1})


def test_namespaced_build_options_select_first_block_cache() -> None:
    raw = _effective_build_config(
        {
            "_family_build_options": {
                "minimax_h3": {
                    "first_block_cache": True,
                    "first_block_cache_threshold": 0.05,
                }
            }
        }
    )

    assert _fixed_profile(raw).first_block_cache is True
    assert raw["first_block_cache_threshold"] == 0.05


def test_plugin_bundle_config_preserves_exact_provenance() -> None:
    provenance = {
        "source_revision": "a" * 40,
        "builder_source_sha256": "b" * 64,
        "checkpoint_inventory_sha256": "c" * 64,
        "workspace_limit_bytes": dict(DEFAULT_WORKSPACE_LIMIT_BYTES),
        "plan_sha256": {
            "text_encoder.plan": "d" * 64,
            "adaln_precompute.plan": "e" * 64,
            "denoiser.plan": "f" * 64,
            "vae_tile_decoder.plan": "0" * 64,
        },
    }
    config = SimpleNamespace(raw={"seed": 7})
    result = MiniMaxH3Model.diffusion_bundle_config(
        config,
        components={"profile": SOL_ENGINE_1344X768_124F, "provenance": provenance},
    )
    assert result | provenance == result
    assert result["seed"] == 7
    assert result["context_parallel_size"] == 1
    assert result["workspace_limit_bytes"] == DEFAULT_WORKSPACE_LIMIT_BYTES
    assert result["first_block_cache"] is False
    assert result["denoiser_cache_mode"] == "monolithic"
    assert result["first_block_cache_threshold"] == 0.025
    assert result["bundle_loading"] == {
        "mode": "staged",
        "eager_sections": ["tokenizer.json", "config.json"],
        "lazy_sections": [
            "text_encoder_plan",
            "adaln_precompute_plan",
            "denoiser_plan",
            "vae_tile_decoder_plan",
        ],
    }
    with pytest.raises(ValueError, match="runtime profile"):
        MiniMaxH3Model.diffusion_bundle_config(
            SimpleNamespace(raw={"video_width": 1}),
            components={"profile": SOL_ENGINE_1344X768_124F, "provenance": provenance},
        )


def test_plugin_emits_first_block_cache_sections_and_profile() -> None:
    profile = replace(SOL_ENGINE_1344X768_124F, first_block_cache=True)
    workspaces = default_workspace_limit_bytes(first_block_cache=True)
    provenance = {
        "source_revision": "a" * 40,
        "builder_source_sha256": "b" * 64,
        "checkpoint_inventory_sha256": "c" * 64,
        "workspace_limit_bytes": workspaces,
        "plan_sha256": {filename: "d" * 64 for filename in workspaces},
    }
    components = {
        "profile": profile,
        "provenance": provenance,
        "text_encoder": b"text",
        "adaln_precompute": b"adaln",
        "denoiser_head": b"head",
        "denoiser_tail": b"tail",
        "denoiser_finish": b"finish",
        "vae_decoder": b"vae",
        "tokenizer_json": b"{}",
    }
    sections = dict(MiniMaxH3Model.diffusion_bundle_sections(components))
    assert tuple(sections) == (
        "text_encoder_plan",
        "adaln_precompute_plan",
        "denoiser_head_plan",
        "denoiser_tail_plan",
        "denoiser_finish_plan",
        "vae_tile_decoder_plan",
        "tokenizer.json",
    )
    config = MiniMaxH3Model.diffusion_bundle_config(
        SimpleNamespace(
            raw={
                "first_block_cache": True,
                "first_block_cache_threshold": 0.08,
            }
        ),
        components=components,
    )
    assert config["first_block_cache"] is True
    assert config["denoiser_cache_mode"] == "first_block"
    assert config["first_block_cache_threshold"] == 0.08
    assert config["bundle_loading"]["lazy_sections"][2:5] == [
        "denoiser_head_plan",
        "denoiser_tail_plan",
        "denoiser_finish_plan",
    ]
    assert set(config["workspace_limit_bytes"]) == {
        "text_encoder.plan",
        "adaln_precompute.plan",
        *FIRST_BLOCK_CACHE_DENOISER_PLAN_FILENAMES,
        "vae_tile_decoder.plan",
    }
    with pytest.raises(ValueError, match="finite and positive"):
        MiniMaxH3Model.diffusion_bundle_config(
            SimpleNamespace(
                raw={
                    "first_block_cache": True,
                    "first_block_cache_threshold": float("nan"),
                }
            ),
            components=components,
        )

    malformed_provenance = dict(provenance)
    malformed_provenance["workspace_limit_bytes"] = {"text_encoder.plan": True}
    with pytest.raises(ValueError, match="workspace_limit_bytes"):
        MiniMaxH3Model.diffusion_bundle_config(
            config,
            components={
                "profile": SOL_ENGINE_1344X768_124F,
                "provenance": malformed_provenance,
            },
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
    assert "build_dit_head_engine" in dit
    assert "build_dit_tail_engine" in dit
    assert "build_dit_finish_engine" in dit
    assert "cache_metric" in dit
    assert "FirstBlockCacheConfig" not in "\n".join((adaln, dit, ops))
