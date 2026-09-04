# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from dataclasses import replace
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace
import tomllib

import pytest

from tensorrt_model_connect.families.minimax_h3.config import (
    DEFAULT_WORKSPACE_LIMIT_BYTES,
    FASTH3_SCHEDULER_GRID_POINTS,
    FASTH3_TRANSFORMER_FORWARDS,
    FASTH3_SEGMENTED_DENOISER_PLAN_FILENAMES,
    FIRST_BLOCK_CACHE_DENOISER_PLAN_FILENAMES,
    MiniMaxH3Config,
    NATIVE_EXPLICIT_CANVAS_SIZES,
    SOL_ENGINE_1344X768_124F,
    SOL_ENGINE_1344X768_124F_FAST_FBC,
    SOL_ENGINE_1344X768_124_TO_345F,
    default_workspace_limit_bytes,
)
from tensorrt_model_connect.families.minimax_h3.plugin import (
    MiniMaxH3Plugin,
    _build_source_revision,
    _default_num_frames,
    _effective_build_config,
    _fast_h3_build_inputs,
    _fixed_profile,
    _resolve_canvas_size,
    _reachable_canvas_sizes,
    _transformer_ref_build_input,
)
from tensorrt_model_connect.families.minimax_h3.fl2va_contract import (
    qwen_vision_patch_rows,
    split_tile_axis,
    video_latent_frames,
)


plugin_module = importlib.import_module("tensorrt_model_connect.families.minimax_h3.plugin")


FAMILY_ROOT = Path(__file__).resolve().parents[1]
AUDIO_VAE_CONFIG = {
    "latents_mean": [0.0] * 32,
    "latents_std": [1.0] * 32,
}
AUDIO_DECODER_PROFILE = SimpleNamespace(
    latent_frames=207,
    min_latent_frames=207,
    max_latent_frames=575,
    sampling_rate=32000,
    hop_length=800,
    batch_size=2,
)


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


def test_fast_fbc_profile_exactly_specializes_qualified_reference_request() -> None:
    profile = SOL_ENGINE_1344X768_124F_FAST_FBC
    profile.validate()
    assert profile.first_block_cache is True
    assert profile.video_row_profile == (37296, 37296, 37296)
    assert profile.audio_row_profile == (414, 414, 414)
    assert profile.text_row_profile == (537, 537, 537)
    assert profile.packed_row_profile == (38247, 38247, 38247)


def test_dynamic_media_profile_covers_released_5_to_15_second_endpoints() -> None:
    profile = SOL_ENGINE_1344X768_124_TO_345F
    profile.validate()
    assert profile.video_row_profile == (18870, 37296, 108576)
    assert profile.audio_row_profile == (414, 414, 1150)
    assert profile.text_row_profile == (1, 128, 2641)
    assert profile.packed_row_profile == (19285, 37838, 112367)
    assert profile.padded_sequence_length == 112367


def test_fast_h3_dynamic_tile_metadata_profiles_cover_public_aspects() -> None:
    profile = SOL_ENGINE_1344X768_124_TO_345F
    assert profile.vsa_prefix_tile_profile == (8, 9, 60)
    assert profile.vsa_video_tile_abi_profile == (360, 660, 2080)
    assert profile.vsa_total_tile_profile == (368, 669, 2140)


def test_public_canvas_resolver_matches_model_card_aspects() -> None:
    assert NATIVE_EXPLICIT_CANVAS_SIZES == ((544, 960), (960, 544))
    assert {
        ratio: _resolve_canvas_size(*ratio)
        for ratio in ((21, 9), (16, 9), (4, 3), (1, 1), (3, 4), (9, 16), (4, 1))
    } == {
        (21, 9): (672, 1536),
        (16, 9): (768, 1344),
        (4, 3): (768, 1024),
        (1, 1): (768, 768),
        (3, 4): (1024, 768),
        (9, 16): (1344, 768),
        (4, 1): (512, 2016),
    }


def test_continuous_canvas_resolver_has_exact_finite_95_canvas_image() -> None:
    reachable = _reachable_canvas_sizes()
    assert len(reachable) == len(set(reachable)) == 95
    assert (576, 1856) in reachable
    assert max(height * width for height, width in reachable) == 1_069_056
    assert {(height // 16) * (width // 16) for height, width in reachable} <= set(range(1, 4177))
    assert max((height // 16) * (width // 16) for height, width in reachable) == 4176


def test_continuous_fl2va_and_vsa_maxima_are_exhaustive() -> None:
    reachable = _reachable_canvas_sizes()
    cases = []
    for height, width in reachable:
        rows_per_frame = (height // 32) * (width // 32)
        for num_frames in range(124, 346, 17):
            latent_frames = video_latent_frames(num_frames)
            for keyframes in (1, 2):
                video_tiles = (
                    -(-(latent_frames + keyframes) // 4)
                    * -(-(height // 32) // 4)
                    * -(-(width // 32) // 4)
                )
                cases.append(
                    (
                        video_tiles,
                        height,
                        width,
                        num_frames,
                        keyframes,
                        rows_per_frame,
                    )
                )
    assert max(case[0] for case in cases) == 2080
    maxima = [case for case in cases if case[0] == 2080]
    assert {(case[1], case[2]) for case in maxima} == {(544, 1952), (1952, 544)}
    assert {case[3] for case in maxima} == {345}
    assert {case[4] for case in maxima} == {1, 2}

    height, width = 576, 1856
    patches = qwen_vision_patch_rows(height, width)
    rows_per_frame = patches // 4
    text_rows = 537 + 2 * (8 + rows_per_frame)
    condition_video_rows = 2 * rows_per_frame
    target_video_rows = video_latent_frames(345) * rows_per_frame
    assert (
        patches,
        rows_per_frame,
        text_rows,
        condition_video_rows,
        target_video_rows,
        text_rows + condition_video_rows + target_video_rows + 1150,
    ) == (4176, 1044, 2641, 2088, 106488, 112367)
    assert (
        max(len(split_tile_axis(h).starts) * len(split_tile_axis(w).starts) for h, w in reachable)
        == 33
    )


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
    plugin = MiniMaxH3Plugin()
    for alias in ("minimax-h3", "minimax_h3", "MiniMaxH3Pipeline"):
        assert plugin.matches(alias)
    assert not plugin.matches("minimax-video-01")


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
    assert _fixed_profile({}) is SOL_ENGINE_1344X768_124_TO_345F
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


def test_dynamic_bundle_default_accepts_only_released_frame_geometries() -> None:
    assert _default_num_frames({}) == 124
    assert _default_num_frames({"video_num_frames": 345}) == 345
    for frames in (123, 125, 360, True):
        with pytest.raises(ValueError, match="valid 5--15 second geometry"):
            _default_num_frames({"video_num_frames": frames})


def test_fast_h3_build_input_is_adapter_only_and_rejects_obsolete_plugin(
    tmp_path: Path,
) -> None:
    adapter = tmp_path / "adapter.safetensors"
    library = tmp_path / "trtmc_model_minimax_h3.dll"
    adapter.write_bytes(b"adapter")
    library.write_bytes(b"plugin")
    assert _fast_h3_build_inputs({"fast_h3_adapter": adapter}) == adapter.resolve()
    assert _fast_h3_build_inputs({}) is None
    assert _fast_h3_build_inputs({"fast_h3_adapter": ""}) is None
    assert _transformer_ref_build_input({"transformer_ref": ""}) is None
    with pytest.raises(ValueError, match="no longer a supported build input"):
        _fast_h3_build_inputs({"vsa_plugin_library": library})


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
            "audio_vae_decoder.plan": "1" * 64,
        },
    }
    config = SimpleNamespace(raw={"seed": 7})
    result = MiniMaxH3Plugin().diffusion_bundle_config(
        config,
        components={
            "profile": SOL_ENGINE_1344X768_124_TO_345F,
            "provenance": provenance,
            "audio_vae_config": AUDIO_VAE_CONFIG,
            "audio_decoder_profile": AUDIO_DECODER_PROFILE,
        },
    )
    assert result | provenance == result
    assert result["seed"] == 7
    assert result["context_parallel_size"] == 1
    assert result["workspace_limit_bytes"] == DEFAULT_WORKSPACE_LIMIT_BYTES
    assert result["first_block_cache"] is False
    assert result["denoiser_cache_mode"] == "monolithic"
    assert result["denoiser_profile_count"] == 1
    assert result["denoiser_profile_layout"] == "public_dynamic"
    assert result["first_block_cache_threshold"] == 0.08
    assert (
        result["vae_tile_batch_min"],
        result["vae_tile_batch_opt"],
        result["vae_tile_batch_max"],
    ) == (15, 28, 33)
    assert (result["num_frames_min"], result["num_frames_opt"], result["num_frames_max"]) == (
        124,
        124,
        345,
    )
    assert (
        result["video_rows_min"],
        result["video_rows_opt"],
        result["video_rows_max"],
    ) == (18870, 37296, 108576)
    assert (
        result["audio_rows_min"],
        result["audio_rows_opt"],
        result["audio_rows_max"],
    ) == (414, 414, 1150)
    assert (
        result["packed_sequence_length_min"],
        result["packed_sequence_length_opt"],
        result["packed_sequence_length_max"],
    ) == (19285, 37838, 112367)
    assert result["explicit_canvas_sizes"] == [[544, 960], [960, 544]]
    assert result["bundle_loading"] == {
        "mode": "staged",
        "eager_sections": ["tokenizer.json", "config.json"],
        "lazy_sections": [
            "text_encoder_plan",
            "vision_encoder_plan",
            "adaln_precompute_plan",
            "denoiser_plan",
            "fl2va_keyframe_vae_encoder_plan",
            "vae_tile_decoder_plan",
            "audio_vae_decoder_plan",
        ],
    }
    with pytest.raises(ValueError, match="canvas aspect"):
        MiniMaxH3Plugin().diffusion_bundle_config(
            SimpleNamespace(raw={"video_width": 1}),
            components={
                "profile": SOL_ENGINE_1344X768_124_TO_345F,
                "provenance": provenance,
                "audio_vae_config": AUDIO_VAE_CONFIG,
                "audio_decoder_profile": AUDIO_DECODER_PROFILE,
            },
        )


def test_fast_h3_bundle_contract_is_four_full_segmented_vsa_forwards() -> None:
    profile = SOL_ENGINE_1344X768_124_TO_345F
    workspaces = default_workspace_limit_bytes(first_block_cache=False, segmented_vsa=True)
    provenance = {
        "source_revision": "a" * 40,
        "builder_source_sha256": "b" * 64,
        "checkpoint_inventory_sha256": "c" * 64,
        "workspace_limit_bytes": workspaces,
        "plan_sha256": {filename: "d" * 64 for filename in workspaces},
    }
    fast_h3 = {
        "adapter_gate_tensor_count": 50,
        "adapter_sha256": "e" * 64,
    }
    components = {
        "profile": profile,
        "provenance": provenance,
        "fast_h3": fast_h3,
        "vsa_implementation": "native_cuda_segmented",
        "audio_vae_config": AUDIO_VAE_CONFIG,
        "audio_decoder_profile": AUDIO_DECODER_PROFILE,
    }

    result = MiniMaxH3Plugin().diffusion_bundle_config(
        SimpleNamespace(raw={"num_inference_steps": 4}), components=components
    )

    assert result["num_inference_steps"] == FASTH3_TRANSFORMER_FORWARDS == 4
    assert result["scheduler_grid_points"] == FASTH3_SCHEDULER_GRID_POINTS == 5
    assert result["transformer_forwards"] == 4
    assert result["guidance_scale"] == 1.0
    assert result["first_block_cache"] is False
    assert result["denoiser_cache_mode"] == "segmented_vsa"
    assert result["denoiser_profile_count"] == 1
    assert result["denoiser_profile_layout"] == "public_dynamic"
    assert result["attention_mode"] == "native_vsa"
    assert result["bundle_loading"]["lazy_sections"][3] == "denoiser_entry_plan"
    assert result["bundle_loading"]["lazy_sections"][4] == ("denoiser_transition_00_plan")
    assert result["vsa"]["implementation"] == "native_cuda_segmented"
    assert result["vsa"]["segment_count"] == 51
    assert result["vsa"]["attention_calls_per_forward"] == 50
    assert result["vsa"]["tensor_abi"]["attention_shape"] == [56, "S", 128]
    assert [item["filename"] for item in result["vsa"]["segments"]] == list(
        FASTH3_SEGMENTED_DENOISER_PLAN_FILENAMES
    )
    assert result["vsa"]["packed_row_to_tile_slot_profile"] == [19285, 37838, 112367]
    assert result["vsa"]["prefix_valid_sizes_profile"] == [8, 9, 60]
    assert result["vsa"]["video_valid_sizes_profile"] == [360, 660, 2080]
    assert result["vsa"]["max_total_tiles"] == 2140
    assert result["vsa"]["runtime_metadata_abi"] == {
        "producer": "modelconnect_cpp",
        "packed_row_to_tile_slot": {
            "dtype": "int32",
            "shape": ["S"],
            "profile": [19285, 37838, 112367],
        },
        "prefix_valid_sizes": {
            "dtype": "int32",
            "shape": ["P"],
            "profile": [8, 9, 60],
        },
        "video_valid_sizes": {
            "dtype": "int32",
            "shape": ["Vtiles"],
            "profile": [360, 660, 2080],
        },
    }
    assert result["padded_sequence_length"] == 112367

    invalid = dict(components)
    invalid["profile"] = replace(profile, first_block_cache=True)
    with pytest.raises(ValueError, match="segmented native CUDA"):
        MiniMaxH3Plugin().diffusion_bundle_config(SimpleNamespace(raw={}), components=invalid)


def test_plugin_emits_first_block_cache_sections_and_profile() -> None:
    profile = replace(SOL_ENGINE_1344X768_124_TO_345F, first_block_cache=True)
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
        "vision_encoder": b"vision",
        "adaln_precompute": b"adaln",
        "denoiser_head": b"head",
        "denoiser_tail": b"tail",
        "denoiser_finish": b"finish",
        "vae_decoder": b"vae",
        "keyframe_vae_encoder": b"keyframe-vae",
        "audio_vae_decoder": b"audio",
        "audio_vae_config": AUDIO_VAE_CONFIG,
        "audio_decoder_profile": AUDIO_DECODER_PROFILE,
        "tokenizer_json": b"{}",
    }
    sections = dict(MiniMaxH3Plugin().diffusion_bundle_sections(components))
    assert tuple(sections) == (
        "text_encoder_plan",
        "vision_encoder_plan",
        "adaln_precompute_plan",
        "denoiser_head_plan",
        "denoiser_tail_plan",
        "denoiser_finish_plan",
        "fl2va_keyframe_vae_encoder_plan",
        "vae_tile_decoder_plan",
        "audio_vae_decoder_plan",
        "tokenizer.json",
    )
    config = MiniMaxH3Plugin().diffusion_bundle_config(
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
    assert config["denoiser_profile_count"] == 2
    assert (
        config["denoiser_profile_layout"]
        == "five_second_reference_then_public_dynamic"
    )
    assert config["first_block_cache_threshold"] == 0.08
    assert config["runtime_memory"] == {
        "mode": "staged",
        "weight_streaming_budget_bytes": 32 << 30,
    }
    assert "adaln_precompute_mode" not in config
    assert "adaln_segmented" not in config
    assert "first_block_cache_abi" not in config
    assert "dense_tail_segment_sections" not in config
    assert config["bundle_loading"]["lazy_sections"] == [
        name for name in sections if name != "tokenizer.json"
    ]
    assert set(config["workspace_limit_bytes"]) == {
        "text_encoder.plan",
        "vision_encoder.plan",
        "adaln_precompute.plan",
        *FIRST_BLOCK_CACHE_DENOISER_PLAN_FILENAMES,
        "fl2va_keyframe_vae_encoder.plan",
        "vae_tile_decoder.plan",
        "audio_vae_decoder.plan",
    }
    default_config = MiniMaxH3Plugin().diffusion_bundle_config(
        SimpleNamespace(raw={"first_block_cache": True}), components=components
    )
    assert default_config["first_block_cache_threshold"] == 0.08
    with pytest.raises(ValueError, match="finite and positive"):
        MiniMaxH3Plugin().diffusion_bundle_config(
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
        MiniMaxH3Plugin().diffusion_bundle_config(
            config,
            components={
                "profile": SOL_ENGINE_1344X768_124_TO_345F,
                "provenance": malformed_provenance,
            },
        )


def test_in_memory_first_block_cache_build_uses_monolithic_native_plans(
    monkeypatch, tmp_path: Path
) -> None:
    root = tmp_path / "model"
    for name in ("transformer", "text_encoder", "vae", "audio_vae", "tokenizer"):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "audio_vae" / "config.json").write_text(json.dumps(AUDIO_VAE_CONFIG))
    (root / "tokenizer" / "tokenizer.json").write_bytes(b"{}")

    builder_calls = []
    partition_groups = []

    def builder(name):
        def build(_weights, *args, **kwargs):
            index = args[1] if len(args) > 1 else None
            builder_calls.append(
                (
                    name,
                    index,
                    kwargs.get("workspace_bytes"),
                    kwargs.get("weight_streaming"),
                )
            )
            return f"{name}:{index}".encode()

        return build

    def module(name, **attributes):
        value = ModuleType(name)
        for attribute, item in attributes.items():
            setattr(value, attribute, item)
        monkeypatch.setitem(sys.modules, name, value)

    family = "tensorrt_model_connect.families.minimax_h3"
    module(
        f"{family}.adaln_builder",
        build_adaln_precompute_engine=builder("adaln_precompute"),
        checkpoint_keys=lambda _profile: ("adaln.monolithic",),
    )
    module(
        f"{family}.dit_builder",
        build_dit_engine=builder("denoiser"),
        build_dit_finish_engine=builder("denoiser_finish"),
        build_dit_head_engine=builder("denoiser_head"),
        build_dit_tail_engine=builder("denoiser_tail"),
        build_dit_vsa_entry_engine=builder("denoiser_entry"),
        build_dit_vsa_finish_engine=builder("denoiser_vsa_finish"),
        build_dit_vsa_transition_engine=builder("denoiser_transition"),
        checkpoint_keys=lambda _profile, **_kwargs: ("denoiser.monolithic",),
        finish_checkpoint_keys=lambda _profile: ("denoiser.finish",),
        head_checkpoint_keys=lambda _profile, **_kwargs: ("denoiser.head",),
        tail_checkpoint_keys=lambda _profile, **_kwargs: ("denoiser.tail",),
        vsa_segment_checkpoint_partitions=lambda _profile: {},
    )
    module(
        f"{family}.multimodal_text_encoder_builder",
        build_multimodal_text_encoder_engine=builder("text_encoder"),
        checkpoint_keys=lambda: ("text",),
    )
    module(
        f"{family}.multimodal_vision_builder",
        build_multimodal_vision_encoder_engine=builder("vision_encoder"),
        checkpoint_keys=lambda: ("vision",),
    )
    module(
        f"{family}.fl2va_vae_encoder_builder",
        build_keyframe_vae_encoder_engine=builder("keyframe_vae_encoder"),
        checkpoint_keys=lambda: ("keyframe_vae",),
    )
    module(
        f"{family}.vae_builder",
        build_vae_tile_decoder_engine=builder("vae_decoder"),
        checkpoint_keys=lambda: ("vae",),
    )
    module(
        f"{family}.audio_vae_builder",
        build_audio_vae_decoder_engine=builder("audio_vae_decoder"),
        checkpoint_keys=lambda _profile: ("audio_vae",),
        decoder_config_from_checkpoint=lambda *_args, **_kwargs: AUDIO_DECODER_PROFILE,
    )

    monkeypatch.setattr(plugin_module, "_build_source_revision", lambda: "a" * 40)
    monkeypatch.setattr(
        plugin_module,
        "checkpoint_snapshot_record",
        lambda _root: {"inventory_sha256": "b" * 64},
    )
    monkeypatch.setattr(plugin_module, "builder_source_sha256", lambda: "c" * 64)
    monkeypatch.setattr(
        plugin_module,
        "validate_component_key_partition",
        lambda _root, groups: partition_groups.extend(groups),
    )
    monkeypatch.setattr(
        plugin_module,
        "load_selected_component_state_dict",
        lambda _root, keys: {key: object() for key in keys},
    )
    monkeypatch.setattr(plugin_module, "numpy_state", dict)

    components = MiniMaxH3Plugin().build_components(
        str(root),
        SimpleNamespace(raw={"first_block_cache": True}),
        {
            "_model_dir": str(root),
            "_transformer_dir": str(root / "transformer"),
            "_text_encoder_dir": str(root / "text_encoder"),
            "_vae_dir": str(root / "vae"),
            "_audio_vae_dir": str(root / "audio_vae"),
            "_tokenizer_dir": str(root / "tokenizer"),
        },
    )

    assert len(partition_groups) == 4
    assert len(set().union(*(set(group) for group in partition_groups))) == 4
    assert {name for name, _index, _workspace, _streaming in builder_calls} >= {
        "adaln_precompute",
        "denoiser_head",
        "denoiser_tail",
        "denoiser_finish",
    }
    dense_calls = [
        call
        for call in builder_calls
        if call[0].startswith("adaln_") or call[0].startswith("denoiser_")
    ]
    assert len(dense_calls) == 4
    assert all(
        workspace is None and streaming is True
        for _name, _index, workspace, streaming in dense_calls
    )
    assert components["adaln_precompute"] == b"adaln_precompute:None"
    assert components["denoiser_tail"] == b"denoiser_tail:None"
    assert set(components["provenance"]["plan_sha256"]) == set(
        default_workspace_limit_bytes(first_block_cache=True)
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
    assert "build_dit_vsa_entry_engine" in dit
    assert "build_dit_vsa_transition_engine" in dit
    assert "build_dit_vsa_finish_engine" in dit
    assert "vsa_attention_output" in dit
    assert "next_residual_hidden" in dit
    assert "cache_metric" in dit
    assert "FirstBlockCacheConfig" not in "\n".join((adaln, dit, ops))
