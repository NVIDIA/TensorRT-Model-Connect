# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import tomllib

import pytest

from tensorrt_model_connect.families.minimax_h3 import audio_vae_builder
from tensorrt_model_connect.families.minimax_h3.audio_vae_builder import (
    AUDIO_LATENT_FRAMES,
    AUDIO_OUTPUT_SAMPLES,
    AUDIO_SAMPLE_RATE,
    AudioVaeDecoderConfig,
    _build_serialized_engine,
    _make_decoder_module,
    _remove_weight_normalization,
    build_audio_vae_decoder_engine,
    validate_audio_vae_decoder_config,
)
from tensorrt_model_connect.families.minimax_h3.checkpoint import (
    load_selected_component_state_dict,
)
from tensorrt_model_connect.families.minimax_h3.config import (
    AUDIO_VAE_DECODER_DEFAULT_WORKSPACE_BYTES,
    DEFAULT_WORKSPACE_LIMIT_BYTES,
    FL2VA_DEFAULT_WORKSPACE_LIMIT_BYTES,
    FL2VA_PLAN_FILENAMES,
    FL2VA_PROCESSOR_ASSET_SECTIONS,
    FIRST_BLOCK_CACHE_DENOISER_PLAN_FILENAMES,
    MiniMaxH3Config,
    REF2VA_DEFAULT_WORKSPACE_LIMIT_BYTES,
    REF2VA_PLAN_FILENAMES,
    SOL_ENGINE_1344X768_124F,
    default_workspace_limit_bytes,
    native_plan_filenames,
)
from tensorrt_model_connect.families.minimax_h3.plugin import (
    MiniMaxH3Plugin,
    _build_source_revision,
    _effective_build_config,
    _fixed_profile,
    _workflow,
)


FAMILY_ROOT = Path(__file__).resolve().parents[1]


def _audio_vae_config() -> dict:
    return {
        "encoder_dim": 64,
        "encoder_rates": [2, 4, 4, 5, 5],
        "latent_dim": 2048,
        "latent_channels": 32,
        "decoder_dim": 1024,
        "decoder_rates": [5, 5, 2, 2, 2, 2, 2],
        "decoder_kernel_sizes": [9, 9, 4, 4, 4, 4, 4],
        "num_attention_heads": 8,
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "sampling_rate": 32000,
        "latents_mean": [float(index) / 32 for index in range(32)],
        "latents_std": [1.0 + float(index) / 32 for index in range(32)],
    }


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


def test_workflow_plan_and_workspace_sets_preserve_t2va_and_add_fl2va() -> None:
    assert _workflow({}) == "t2va"
    assert native_plan_filenames(first_block_cache=False) == tuple(DEFAULT_WORKSPACE_LIMIT_BYTES)
    assert default_workspace_limit_bytes(first_block_cache=False) == (DEFAULT_WORKSPACE_LIMIT_BYTES)
    assert native_plan_filenames(first_block_cache=False, workflow="fl2va") == (
        FL2VA_PLAN_FILENAMES
    )
    assert default_workspace_limit_bytes(first_block_cache=False, workflow="fl2va") == (
        FL2VA_DEFAULT_WORKSPACE_LIMIT_BYTES
    )
    with pytest.raises(ValueError, match="does not support first_block_cache"):
        native_plan_filenames(first_block_cache=True, workflow="fl2va")
    assert native_plan_filenames(first_block_cache=False, workflow="ref2va") == (
        REF2VA_PLAN_FILENAMES
    )
    assert default_workspace_limit_bytes(first_block_cache=False, workflow="ref2va") == (
        REF2VA_DEFAULT_WORKSPACE_LIMIT_BYTES
    )
    with pytest.raises(ValueError, match="does not support first_block_cache"):
        native_plan_filenames(first_block_cache=True, workflow="ref2va")
    with pytest.raises(ValueError, match="workflow must be one of"):
        _workflow({"workflow": "unknown"})


def test_load_weights_routes_exact_workflow_partition_and_processor_assets(tmp_path: Path) -> None:
    root = tmp_path / "model"
    for name in (
        "transformer",
        "transformer_ref",
        "text_encoder",
        "vae",
        "audio_vae",
        "tokenizer",
        "processor",
    ):
        (root / name).mkdir(parents=True)
    transformer_config = {
        "hidden_size": 5376,
        "num_layers": 50,
        "num_attention_heads": 56,
        "attention_head_dim": 128,
        "ffn_dim": 14336,
    }
    for partition in ("transformer", "transformer_ref"):
        (root / partition / "config.json").write_text(json.dumps(transformer_config))
    for relative in FL2VA_PROCESSOR_ASSET_SECTIONS:
        (root / relative).write_text("{}")

    plugin = MiniMaxH3Plugin()
    t2va = plugin.load_weights(str(root), SimpleNamespace(raw={}))
    assert t2va["_workflow"] == "t2va"
    assert t2va["_transformer_subfolder"] == "transformer"
    assert t2va["_transformer_dir"] == str(root / "transformer")

    fl2va = plugin.load_weights(str(root), SimpleNamespace(raw={"workflow": "fl2va"}))
    assert fl2va["_workflow"] == "fl2va"
    assert fl2va["_transformer_subfolder"] == "transformer"
    assert fl2va["_processor_dir"] == str(root / "processor")

    ref2va = plugin.load_weights(str(root), SimpleNamespace(raw={"workflow": "ref2va"}))
    assert ref2va["_workflow"] == "ref2va"
    assert ref2va["_transformer_subfolder"] == "transformer_ref"
    assert ref2va["_transformer_dir"] == str(root / "transformer_ref")


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


def test_audio_vae_decoder_contract_matches_diffusers() -> None:
    config = validate_audio_vae_decoder_config(_audio_vae_config())
    assert config.latent_channels == 32
    assert config.hop_length == 800
    assert config.sampling_rate == AUDIO_SAMPLE_RATE == 32000
    assert AUDIO_LATENT_FRAMES == 207
    assert AUDIO_OUTPUT_SAMPLES == 165600
    assert config.latents_mean[17] == pytest.approx(17 / 32)
    assert config.latents_std[17] == pytest.approx(1 + 17 / 32)

    malformed = _audio_vae_config()
    malformed["decoder_rates"] = [5, 5, 2, 2, 2, 2]
    with pytest.raises(ValueError, match="architecture"):
        validate_audio_vae_decoder_config(malformed)

    malformed = _audio_vae_config()
    malformed["latents_std"][0] = 0.0
    with pytest.raises(ValueError, match="finite positive"):
        validate_audio_vae_decoder_config(malformed)


def test_audio_vae_decoder_builder_exports_then_builds_static_plan(
    tmp_path: Path, monkeypatch
) -> None:
    audio_vae_dir = tmp_path / "audio_vae"
    audio_vae_dir.mkdir()
    (audio_vae_dir / "config.json").write_text(json.dumps(_audio_vae_config()))
    observed = {}

    def export(root, config, verbose):
        observed.update(root=root, config=config, export_verbose=verbose)
        return b"onnx"

    def build(onnx_bytes, *, verbose, workspace_bytes):
        observed.update(
            onnx_bytes=onnx_bytes,
            build_verbose=verbose,
            workspace_bytes=workspace_bytes,
        )
        return b"audio-plan"

    monkeypatch.setattr(audio_vae_builder, "_export_decoder_onnx", export)
    monkeypatch.setattr(audio_vae_builder, "_build_serialized_engine", build)
    assert (
        build_audio_vae_decoder_engine(audio_vae_dir, verbose=True, workspace_bytes=8 << 30)
        == b"audio-plan"
    )
    assert observed == {
        "root": audio_vae_dir,
        "config": validate_audio_vae_decoder_config(_audio_vae_config()),
        "export_verbose": True,
        "onnx_bytes": b"onnx",
        "build_verbose": True,
        "workspace_bytes": 8 << 30,
    }


def test_audio_vae_single_file_checkpoint_loads_only_decoder_keys(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    audio_vae_dir = tmp_path / "audio_vae"
    audio_vae_dir.mkdir()
    safetensors_torch.save_file(
        {
            "dec_in_proj.weight": torch.ones((2, 2, 1), dtype=torch.float32),
            "encoder.block.0.weight": torch.zeros((2, 1, 1), dtype=torch.float32),
        },
        audio_vae_dir / "diffusion_pytorch_model.safetensors",
    )
    selected = load_selected_component_state_dict(audio_vae_dir, ("dec_in_proj.weight",))
    assert tuple(selected) == ("dec_in_proj.weight",)
    assert torch.equal(selected["dec_in_proj.weight"], torch.ones((2, 2, 1)))


def test_build_components_builds_and_provenance_binds_audio_decoder(
    tmp_path: Path, monkeypatch
) -> None:
    plugin_module = sys.modules[MiniMaxH3Plugin.__module__]
    model_dir = tmp_path / "model"
    paths = {}
    for name in ("transformer", "text_encoder", "vae", "audio_vae", "tokenizer"):
        paths[name] = model_dir / name
        paths[name].mkdir(parents=True)
    (paths["tokenizer"] / "tokenizer.json").write_bytes(b"{}")

    def fake_builder(payload: bytes):
        return lambda *_args, **_kwargs: payload

    adaln = SimpleNamespace(
        build_adaln_precompute_engine=fake_builder(b"adaln"),
        checkpoint_keys=lambda _profile: ("adaln",),
    )
    dit = SimpleNamespace(
        build_dit_engine=fake_builder(b"denoiser"),
        build_dit_head_engine=fake_builder(b"head"),
        build_dit_tail_engine=fake_builder(b"tail"),
        build_dit_finish_engine=fake_builder(b"finish"),
        checkpoint_keys=lambda _profile: ("denoiser",),
        head_checkpoint_keys=lambda _profile: ("head",),
        tail_checkpoint_keys=lambda _profile: ("tail",),
        finish_checkpoint_keys=lambda _profile: ("finish",),
    )
    text_encoder = SimpleNamespace(
        build_text_encoder_engine=fake_builder(b"text"),
        checkpoint_keys=lambda: ("text",),
    )
    vae = SimpleNamespace(
        build_vae_tile_decoder_engine=fake_builder(b"video-vae"),
        checkpoint_keys=lambda: ("video-vae",),
    )
    module_prefix = "tensorrt_model_connect.families.minimax_h3"
    monkeypatch.setitem(sys.modules, f"{module_prefix}.adaln_builder", adaln)
    monkeypatch.setitem(sys.modules, f"{module_prefix}.dit_builder", dit)
    monkeypatch.setitem(sys.modules, f"{module_prefix}.text_encoder_builder", text_encoder)
    monkeypatch.setitem(sys.modules, f"{module_prefix}.vae_builder", vae)

    audio_calls = {}

    def build_audio(root, *, verbose, workspace_bytes):
        audio_calls.update(root=root, verbose=verbose, workspace_bytes=workspace_bytes)
        return b"audio-vae"

    monkeypatch.setattr(audio_vae_builder, "build_audio_vae_decoder_engine", build_audio)
    monkeypatch.setattr(plugin_module, "_build_source_revision", lambda: "a" * 40)
    monkeypatch.setattr(
        plugin_module,
        "checkpoint_snapshot_record",
        lambda _root, **_kwargs: {"inventory_sha256": "b" * 64},
    )
    monkeypatch.setattr(plugin_module, "load_selected_component_state_dict", lambda *_args: {})
    monkeypatch.setattr(plugin_module, "numpy_state", lambda _state: {})
    monkeypatch.setattr(plugin_module, "validate_component_key_partition", lambda *_args: None)
    monkeypatch.setattr(plugin_module, "builder_source_sha256", lambda: "c" * 64)

    components = MiniMaxH3Plugin().build_components(
        str(model_dir),
        SimpleNamespace(raw={}),
        {
            "_model_dir": str(model_dir),
            "_transformer_dir": str(paths["transformer"]),
            "_text_encoder_dir": str(paths["text_encoder"]),
            "_vae_dir": str(paths["vae"]),
            "_audio_vae_dir": str(paths["audio_vae"]),
            "_tokenizer_dir": str(paths["tokenizer"]),
        },
        verbose=True,
        parallel_config=SimpleNamespace(mode="single", cp_size=1),
    )
    assert components["audio_vae_decoder"] == b"audio-vae"
    assert audio_calls == {
        "root": str(paths["audio_vae"]),
        "verbose": True,
        "workspace_bytes": AUDIO_VAE_DECODER_DEFAULT_WORKSPACE_BYTES,
    }
    assert (
        components["provenance"]["plan_sha256"]["audio_vae_decoder.plan"]
        == hashlib.sha256(b"audio-vae").hexdigest()
    )


def test_fl2va_build_packages_every_conditioner_plan_asset_and_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    plugin_module = sys.modules[MiniMaxH3Plugin.__module__]
    module_prefix = "tensorrt_model_connect.families.minimax_h3"
    model_dir = tmp_path / "model"
    paths = {}
    for name in ("transformer", "text_encoder", "vae", "audio_vae", "tokenizer", "processor"):
        paths[name] = model_dir / name
        paths[name].mkdir(parents=True)
    (paths["text_encoder"] / "config.json").write_text('{"model_type":"qwen3_vl"}')
    (paths["tokenizer"] / "tokenizer.json").write_bytes(b'{"tokenizer":true}')
    for index, relative in enumerate(FL2VA_PROCESSOR_ASSET_SECTIONS):
        (model_dir / relative).write_bytes(json.dumps({"asset": index}).encode())

    calls = {}

    def payload(name):
        def build(*args, **kwargs):
            calls[name] = {"args": args, "kwargs": kwargs}
            return name.encode()

        return build

    monkeypatch.setitem(
        sys.modules,
        f"{module_prefix}.adaln_builder",
        SimpleNamespace(
            build_adaln_precompute_engine=payload("adaln"),
            checkpoint_keys=lambda _profile: ("adaln.weight",),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        f"{module_prefix}.dit_builder",
        SimpleNamespace(
            build_dit_engine=payload("legacy_denoiser"),
            build_dit_head_engine=payload("head"),
            build_dit_tail_engine=payload("tail"),
            build_dit_finish_engine=payload("finish"),
            build_fl2va_dit_engine=payload("fl2va_denoiser"),
            checkpoint_keys=lambda _profile: ("dit.weight",),
            head_checkpoint_keys=lambda _profile: ("head.weight",),
            tail_checkpoint_keys=lambda _profile: ("tail.weight",),
            finish_checkpoint_keys=lambda _profile: ("finish.weight",),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        f"{module_prefix}.text_encoder_builder",
        SimpleNamespace(
            build_text_encoder_engine=payload("legacy_text"),
            checkpoint_keys=lambda: ("legacy.text.weight",),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        f"{module_prefix}.language_conditioner_builder",
        SimpleNamespace(
            build_language_conditioner_engine=payload("language_conditioner"),
            checkpoint_keys=lambda: ("language.weight",),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        f"{module_prefix}.vision_conditioner_builder",
        SimpleNamespace(
            build_vision_conditioner_engine=payload("vision_conditioner"),
            checkpoint_keys=lambda: ("vision.weight",),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        f"{module_prefix}.vae_builder",
        SimpleNamespace(
            build_vae_tile_decoder_engine=payload("vae_decoder"),
            checkpoint_keys=lambda: ("vae.decoder.weight",),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        f"{module_prefix}.vae_encoder_builder",
        SimpleNamespace(build_vae_encoder_tile_engine=payload("vae_encoder_tile_t1")),
    )
    monkeypatch.setitem(
        sys.modules,
        f"{module_prefix}.audio_vae_builder",
        SimpleNamespace(build_audio_vae_decoder_engine=payload("audio_decoder")),
    )

    snapshot_calls = []
    monkeypatch.setattr(plugin_module, "_build_source_revision", lambda: "a" * 40)
    monkeypatch.setattr(
        plugin_module,
        "checkpoint_snapshot_record",
        lambda root, **kwargs: (
            snapshot_calls.append((root, kwargs)) or {"inventory_sha256": "b" * 64}
        ),
    )
    monkeypatch.setattr(plugin_module, "load_selected_component_state_dict", lambda *_args: {})
    monkeypatch.setattr(plugin_module, "numpy_state", lambda _state: {})
    partition_calls = []
    monkeypatch.setattr(
        plugin_module,
        "validate_component_key_partition",
        lambda *args: partition_calls.append(args),
    )
    monkeypatch.setattr(plugin_module, "builder_source_sha256", lambda: "c" * 64)

    weights = {
        "_workflow": "fl2va",
        "_transformer_subfolder": "transformer",
        "_model_dir": str(model_dir),
        "_transformer_dir": str(paths["transformer"]),
        "_text_encoder_dir": str(paths["text_encoder"]),
        "_vae_dir": str(paths["vae"]),
        "_audio_vae_dir": str(paths["audio_vae"]),
        "_tokenizer_dir": str(paths["tokenizer"]),
        "_processor_dir": str(paths["processor"]),
    }
    components = MiniMaxH3Plugin().build_components(
        str(model_dir),
        SimpleNamespace(raw={"workflow": "fl2va"}),
        weights,
        verbose=True,
        parallel_config=SimpleNamespace(mode="single", cp_size=1),
    )

    assert components["workflow"] == "fl2va"
    assert components["checkpoint_partition"] == "transformer"
    assert components["language_conditioner"] == b"language_conditioner"
    assert components["vision_conditioner"] == b"vision_conditioner"
    assert components["vae_encoder_tile_t1"] == b"vae_encoder_tile_t1"
    assert components["fl2va_denoiser"] == b"fl2va_denoiser"
    assert snapshot_calls == [(model_dir, {"workflow": "fl2va"})]
    assert partition_calls[0][0] == str(paths["transformer"])
    assert calls["language_conditioner"]["kwargs"]["workflow"] == "fl2va"
    assert calls["fl2va_denoiser"]["kwargs"]["checkpoint_subfolder"] == "transformer"
    assert calls["vae_encoder_tile_t1"]["kwargs"] == {
        "num_frames": 1,
        "verbose": True,
        "workspace_bytes": FL2VA_DEFAULT_WORKSPACE_LIMIT_BYTES["vae_encoder_tile_t1.plan"],
    }
    provenance = components["provenance"]
    assert provenance["workflow"] == "fl2va"
    assert provenance["checkpoint_partition"] == "transformer"
    assert provenance["workspace_limit_bytes"] == FL2VA_DEFAULT_WORKSPACE_LIMIT_BYTES
    assert set(provenance["plan_sha256"]) == set(FL2VA_PLAN_FILENAMES)
    assert set(provenance["asset_sha256"]) == {
        "tokenizer.json",
        *FL2VA_PROCESSOR_ASSET_SECTIONS,
    }

    sections = dict(MiniMaxH3Plugin().diffusion_bundle_sections(components))
    assert tuple(sections) == (
        "language_conditioner_plan",
        "vision_conditioner_plan",
        "vae_encoder_tile_t1_plan",
        "adaln_precompute_plan",
        "fl2va_denoiser_plan",
        "vae_tile_decoder_plan",
        "audio_vae_decoder_plan",
        "tokenizer.json",
        *FL2VA_PROCESSOR_ASSET_SECTIONS,
    )
    bundle_config = MiniMaxH3Plugin().diffusion_bundle_config(
        SimpleNamespace(raw={"workflow": "fl2va"}),
        components=components,
    )
    assert bundle_config["workflow"] == "fl2va"
    assert bundle_config["checkpoint_partition"] == "transformer"
    assert bundle_config["fl2va_vae_tile_size"] == 256
    assert bundle_config["fl2va_vae_tile_min_overlap"] == 64
    assert bundle_config["fl2va_vae_temporal_frames"] == [1]
    assert bundle_config["bundle_loading"] == {
        "mode": "staged",
        "eager_sections": [
            "tokenizer.json",
            *FL2VA_PROCESSOR_ASSET_SECTIONS,
            "config.json",
        ],
        "lazy_sections": [
            "language_conditioner_plan",
            "vision_conditioner_plan",
            "vae_encoder_tile_t1_plan",
            "adaln_precompute_plan",
            "fl2va_denoiser_plan",
            "vae_tile_decoder_plan",
            "audio_vae_decoder_plan",
        ],
    }


def test_ref2va_build_packages_dynamic_reference_plans_and_transformer_ref(
    tmp_path: Path, monkeypatch
) -> None:
    plugin_module = sys.modules[MiniMaxH3Plugin.__module__]
    module_prefix = "tensorrt_model_connect.families.minimax_h3"
    model_dir = tmp_path / "model"
    paths = {}
    for name in (
        "transformer_ref",
        "text_encoder",
        "vae",
        "audio_vae",
        "tokenizer",
        "processor",
    ):
        paths[name] = model_dir / name
        paths[name].mkdir(parents=True)
    (paths["text_encoder"] / "config.json").write_text('{"model_type":"qwen3_vl"}')
    (paths["tokenizer"] / "tokenizer.json").write_bytes(b'{"tokenizer":true}')
    for index, relative in enumerate(FL2VA_PROCESSOR_ASSET_SECTIONS):
        (model_dir / relative).write_bytes(json.dumps({"asset": index}).encode())

    calls = {}

    def payload(name):
        def build(*args, **kwargs):
            calls[name] = {"args": args, "kwargs": kwargs}
            return name.encode()

        return build

    monkeypatch.setitem(
        sys.modules,
        f"{module_prefix}.adaln_builder",
        SimpleNamespace(
            build_adaln_precompute_engine=payload("adaln"),
            checkpoint_keys=lambda _profile: ("adaln.weight",),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        f"{module_prefix}.dit_builder",
        SimpleNamespace(
            build_dit_engine=payload("legacy_denoiser"),
            build_dit_head_engine=payload("head"),
            build_dit_tail_engine=payload("tail"),
            build_dit_finish_engine=payload("finish"),
            build_ref2va_dit_engine=payload("ref2va_denoiser"),
            checkpoint_keys=lambda _profile: ("dit.weight",),
            head_checkpoint_keys=lambda _profile: ("head.weight",),
            tail_checkpoint_keys=lambda _profile: ("tail.weight",),
            finish_checkpoint_keys=lambda _profile: ("finish.weight",),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        f"{module_prefix}.text_encoder_builder",
        SimpleNamespace(
            build_text_encoder_engine=payload("legacy_text"),
            checkpoint_keys=lambda: ("legacy.text.weight",),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        f"{module_prefix}.language_conditioner_builder",
        SimpleNamespace(
            build_language_conditioner_engine=payload("language_conditioner"),
            checkpoint_keys=lambda: ("language.weight",),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        f"{module_prefix}.vision_conditioner_builder",
        SimpleNamespace(
            build_vision_conditioner_engine=payload("vision_conditioner"),
            checkpoint_keys=lambda: ("vision.weight",),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        f"{module_prefix}.vae_builder",
        SimpleNamespace(
            build_vae_tile_decoder_engine=payload("vae_decoder"),
            checkpoint_keys=lambda: ("vae.decoder.weight",),
        ),
    )

    def build_vae_encoder_tile(*args, num_frames, **kwargs):
        return payload(f"vae_encoder_tile_t{num_frames}")(*args, num_frames=num_frames, **kwargs)

    monkeypatch.setitem(
        sys.modules,
        f"{module_prefix}.vae_encoder_builder",
        SimpleNamespace(build_vae_encoder_tile_engine=build_vae_encoder_tile),
    )
    monkeypatch.setitem(
        sys.modules,
        f"{module_prefix}.audio_vae_builder",
        SimpleNamespace(
            build_audio_vae_encoder_engine=payload("audio_encoder"),
            build_audio_vae_decoder_engine=payload("audio_decoder"),
        ),
    )
    monkeypatch.setattr(plugin_module, "_build_source_revision", lambda: "a" * 40)
    monkeypatch.setattr(
        plugin_module,
        "checkpoint_snapshot_record",
        lambda _root, **kwargs: {"inventory_sha256": "b" * 64, "workflow": kwargs["workflow"]},
    )
    monkeypatch.setattr(plugin_module, "load_selected_component_state_dict", lambda *_args: {})
    monkeypatch.setattr(plugin_module, "numpy_state", lambda _state: {})
    monkeypatch.setattr(plugin_module, "validate_component_key_partition", lambda *_args: None)
    monkeypatch.setattr(plugin_module, "builder_source_sha256", lambda: "c" * 64)

    weights = {
        "_workflow": "ref2va",
        "_transformer_subfolder": "transformer_ref",
        "_model_dir": str(model_dir),
        "_transformer_dir": str(paths["transformer_ref"]),
        "_text_encoder_dir": str(paths["text_encoder"]),
        "_vae_dir": str(paths["vae"]),
        "_audio_vae_dir": str(paths["audio_vae"]),
        "_tokenizer_dir": str(paths["tokenizer"]),
        "_processor_dir": str(paths["processor"]),
    }
    components = MiniMaxH3Plugin().build_components(
        str(model_dir),
        SimpleNamespace(raw={"workflow": "ref2va"}),
        weights,
        verbose=True,
        parallel_config=SimpleNamespace(mode="single", cp_size=1),
    )
    assert components["workflow"] == "ref2va"
    assert components["checkpoint_partition"] == "transformer_ref"
    assert components["ref2va_denoiser"] == b"ref2va_denoiser"
    assert components["vae_encoder_tile_t1"] == b"vae_encoder_tile_t1"
    assert components["vae_encoder_tile_t17"] == b"vae_encoder_tile_t17"
    assert components["audio_vae_encoder"] == b"audio_encoder"
    assert calls["language_conditioner"]["kwargs"]["workflow"] == "ref2va"
    assert calls["vision_conditioner"]["kwargs"]["workflow"] == "ref2va"
    assert calls["ref2va_denoiser"]["kwargs"]["checkpoint_subfolder"] == "transformer_ref"
    assert calls["vae_encoder_tile_t1"]["kwargs"]["num_frames"] == 1
    assert calls["vae_encoder_tile_t17"]["kwargs"]["num_frames"] == 17
    assert components["provenance"]["workspace_limit_bytes"] == (
        REF2VA_DEFAULT_WORKSPACE_LIMIT_BYTES
    )
    assert set(components["provenance"]["plan_sha256"]) == set(REF2VA_PLAN_FILENAMES)

    sections = dict(MiniMaxH3Plugin().diffusion_bundle_sections(components))
    assert tuple(sections) == (
        "language_conditioner_plan",
        "vision_conditioner_plan",
        "vae_encoder_tile_t1_plan",
        "vae_encoder_tile_t17_plan",
        "audio_vae_encoder_plan",
        "adaln_precompute_plan",
        "ref2va_denoiser_plan",
        "vae_tile_decoder_plan",
        "audio_vae_decoder_plan",
        "tokenizer.json",
        *FL2VA_PROCESSOR_ASSET_SECTIONS,
    )
    bundle_config = MiniMaxH3Plugin().diffusion_bundle_config(
        SimpleNamespace(raw={"workflow": "ref2va"}), components=components
    )
    assert bundle_config["checkpoint_partition"] == "transformer_ref"
    assert bundle_config["max_text_rows"] == 262144
    assert bundle_config["ref2va_min_condition_video_rows"] == 0
    assert bundle_config["ref2va_opt_condition_video_rows"] == 4096
    assert bundle_config["ref2va_min_condition_audio_rows"] == 0
    assert bundle_config["ref2va_opt_condition_audio_rows"] == 0
    assert bundle_config["ref2va_max_images"] == 9
    assert bundle_config["ref2va_max_videos"] == 3
    assert bundle_config["ref2va_max_audios"] == 3
    assert bundle_config["ref2va_max_references"] == 12
    assert bundle_config["bundle_loading"]["lazy_sections"] == [
        "language_conditioner_plan",
        "vision_conditioner_plan",
        "vae_encoder_tile_t1_plan",
        "vae_encoder_tile_t17_plan",
        "audio_vae_encoder_plan",
        "adaln_precompute_plan",
        "ref2va_denoiser_plan",
        "vae_tile_decoder_plan",
        "audio_vae_decoder_plan",
    ]

    with pytest.raises(ValueError, match="FL2VA does not support first_block_cache"):
        MiniMaxH3Plugin().build_components(
            str(model_dir),
            SimpleNamespace(raw={"workflow": "fl2va", "first_block_cache": True}),
            {
                "_workflow": "fl2va",
                "_transformer_subfolder": "transformer",
                "_transformer_dir": str(model_dir / "transformer"),
            },
            parallel_config=SimpleNamespace(mode="single", cp_size=1),
        )


def test_audio_vae_decoder_onnx_contract_and_workspace_are_fail_closed(monkeypatch) -> None:
    observed = {}
    fp32 = object()

    class Tensor:
        def __init__(self, name, shape):
            self.name = name
            self.shape = shape
            self.dtype = fp32

    class Network:
        num_inputs = 1
        num_outputs = 1

        def get_input(self, index):
            assert index == 0
            return Tensor("audio_latents", (2, 32, 207))

        def get_output(self, index):
            assert index == 0
            return Tensor("waveform", (2, 1, 165600))

    class Parser:
        num_errors = 0

        def __init__(self, network, logger):
            del network, logger

        def parse(self, payload):
            observed["onnx"] = payload
            return True

    class BuildConfig:
        def set_memory_pool_limit(self, pool, size):
            observed.update(pool=pool, workspace=size)

        def get_memory_pool_limit(self, pool):
            assert pool == "workspace"
            return observed["workspace"]

        def clear_flag(self, flag):
            observed.setdefault("cleared_flags", []).append(flag)

    class Builder:
        def __init__(self, logger):
            del logger

        def create_network(self, flags):
            observed["flags"] = flags
            return Network()

        def create_builder_config(self):
            return BuildConfig()

        def build_serialized_network(self, network, config):
            del network, config
            return b"audio-plan"

    class Logger:
        INFO = "info"
        WARNING = "warning"

        def __init__(self, severity):
            observed["severity"] = severity

    fake_trt = SimpleNamespace(
        Logger=Logger,
        Builder=Builder,
        OnnxParser=Parser,
        BuilderFlag=SimpleNamespace(TF32="tf32"),
        MemoryPoolType=SimpleNamespace(WORKSPACE="workspace"),
        float32=fp32,
    )
    monkeypatch.setattr(audio_vae_builder.trt_compat, "get_trt", lambda: fake_trt)
    monkeypatch.setattr(audio_vae_builder.trt_compat, "network_creation_flags", lambda **_kwargs: 9)

    assert _build_serialized_engine(b"onnx", verbose=False, workspace_bytes=None) == b"audio-plan"
    assert observed == {
        "severity": "warning",
        "flags": 9,
        "onnx": b"onnx",
        "pool": "workspace",
        "workspace": AUDIO_VAE_DECODER_DEFAULT_WORKSPACE_BYTES,
    }


def test_static_audio_decoder_denormalizes_and_clamps() -> None:
    torch = pytest.importorskip("torch")
    config = AudioVaeDecoderConfig(
        latent_dim=2,
        latent_channels=2,
        decoder_dim=4,
        decoder_rates=(2,),
        decoder_kernel_sizes=(4,),
        resblock_kernel_sizes=(3,),
        resblock_dilation_sizes=((1,),),
        sampling_rate=32000,
        latents_mean=(1.0, -2.0),
        latents_std=(3.0, 4.0),
    )
    module = _make_decoder_module(torch, config, batch=2, latent_frames=3)
    decoded = module(torch.zeros((2, 2, 3), dtype=torch.float32))
    assert tuple(decoded.shape) == (2, 1, 6)
    assert decoded.dtype == torch.float32
    assert torch.all(decoded >= -1.0)
    assert torch.all(decoded <= 1.0)

    class Capture(torch.nn.Module):
        def forward(self, value):
            self.value = value
            return value

    capture = Capture()
    module.dec_in_proj = torch.nn.Identity()
    module.decoder = capture
    source = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    expected = source * torch.tensor([[[3.0], [4.0]]]) + torch.tensor([[[1.0], [-2.0]]])
    assert torch.equal(module(source), expected.reshape(2, 1, 6))
    assert torch.equal(capture.value, expected)


def test_tiny_static_audio_decoder_exports_with_fixed_onnx_contract() -> None:
    torch = pytest.importorskip("torch")
    onnx = pytest.importorskip("onnx")
    config = AudioVaeDecoderConfig(
        latent_dim=2,
        latent_channels=2,
        decoder_dim=4,
        decoder_rates=(2,),
        decoder_kernel_sizes=(4,),
        resblock_kernel_sizes=(3,),
        resblock_dilation_sizes=((1,),),
        sampling_rate=32000,
        latents_mean=(1.0, -2.0),
        latents_std=(3.0, 4.0),
    )
    module = _make_decoder_module(torch, config, batch=2, latent_frames=3)
    _remove_weight_normalization(torch, module)
    buffer = io.BytesIO()
    torch.onnx.export(
        module,
        torch.zeros((2, 2, 3), dtype=torch.float32),
        buffer,
        opset_version=17,
        input_names=["audio_latents"],
        output_names=["waveform"],
        dynamo=False,
    )
    graph = onnx.load_model_from_string(buffer.getvalue()).graph
    assert graph.input[0].name == "audio_latents"
    assert graph.output[0].name == "waveform"
    assert [dim.dim_value for dim in graph.input[0].type.tensor_type.shape.dim] == [2, 2, 3]
    assert [dim.dim_value for dim in graph.output[0].type.tensor_type.shape.dim] == [2, 1, 6]


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
            "audio_vae_decoder.plan": "1" * 64,
        },
    }
    config = SimpleNamespace(raw={"seed": 7})
    result = MiniMaxH3Plugin().diffusion_bundle_config(
        config,
        components={"profile": SOL_ENGINE_1344X768_124F, "provenance": provenance},
    )
    assert result | provenance == result
    assert result["seed"] == 7
    assert result["workflow"] == "t2va"
    assert result["checkpoint_partition"] == "transformer"
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
            "audio_vae_decoder_plan",
        ],
    }
    assert result["audio_sample_rate"] == 32000
    assert result["audio_latent_frames"] == 207
    assert result["audio_output_samples"] == 165600
    with pytest.raises(ValueError, match="runtime profile"):
        MiniMaxH3Plugin().diffusion_bundle_config(
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
        "audio_vae_decoder": b"audio",
        "tokenizer_json": b"{}",
    }
    sections = dict(MiniMaxH3Plugin().diffusion_bundle_sections(components))
    assert tuple(sections) == (
        "text_encoder_plan",
        "adaln_precompute_plan",
        "denoiser_head_plan",
        "denoiser_tail_plan",
        "denoiser_finish_plan",
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
        "audio_vae_decoder.plan",
    }
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
                "profile": SOL_ENGINE_1344X768_124F,
                "provenance": malformed_provenance,
            },
        )


def test_production_runtime_is_native_and_onnx_exports_are_build_only() -> None:
    violations = []
    build_only_torch_modules = {"audio_vae_builder.py", "vae_encoder_builder.py"}
    for path in FAMILY_ROOT.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.name for alias in node.names]
                if path.name not in build_only_torch_modules and any(
                    name.startswith(("torch", "torch_tensorrt", "triton")) for name in names
                ):
                    violations.append(f"{path.name}:{node.lineno}: {names}")
            if isinstance(node, ast.Call):
                function = node.func
                name = function.attr if isinstance(function, ast.Attribute) else ""
                if name.startswith("add_plugin") or name == "get_plugin_registry":
                    violations.append(f"{path.name}:{node.lineno}: {name}")
    assert not violations

    audio_builder = (FAMILY_ROOT / "audio_vae_builder.py").read_text()
    assert "import torch" in audio_builder
    assert "torch.onnx.export" in audio_builder
    assert 'input_names=["audio_latents"]' in audio_builder
    assert 'output_names=["waveform"]' in audio_builder
    assert "python_callback" not in audio_builder
    assert "runtime_bridge" not in audio_builder

    visual_builder = (FAMILY_ROOT / "vae_encoder_builder.py").read_text()
    assert "import torch" in visual_builder
    assert "torch.onnx.export" in visual_builder
    assert 'input_names=["normalized_rgb"]' in visual_builder
    assert 'output_names=["posterior_moments"]' in visual_builder
    assert "python_callback" not in visual_builder
    assert "runtime_bridge" not in visual_builder


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


def test_fixed_and_dynamic_language_rmsnorm_routes_are_explicit() -> None:
    def norm_calls(filename: str) -> list[str]:
        tree = ast.parse((FAMILY_ROOT / filename).read_text(), filename=filename)
        return [
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "op"
            and node.func.attr in {"rms_norm", "qwen_rms_norm"}
        ]

    assert norm_calls("text_encoder_builder.py") == ["rms_norm"] * 3
    assert norm_calls("language_conditioner_builder.py") == ["qwen_rms_norm"] * 3


def test_fixed_and_dynamic_dit_attention_precision_routes_are_explicit() -> None:
    tree = ast.parse((FAMILY_ROOT / "dit_builder.py").read_text(), filename="dit_builder.py")
    attention_block = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_attention_block"
    )
    native_calls = [
        node
        for node in ast.walk(attention_block)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "op"
        and node.func.attr == "native_attention"
    ]
    assert len(native_calls) == 1
    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in native_calls[0].keywords}
    assert keywords["attention_dtype"] == "trt.bfloat16 if rows < 0 else trt.float16"
