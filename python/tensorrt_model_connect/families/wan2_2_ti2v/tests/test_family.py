# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the native Wan2.2-TI2V-5B family."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

from tensorrt_model_connect.families.wan2_2_ti2v.model_config import (
    WAN22_TI2V_5B,
    WAN22_TI2V_5B_L0,
    artifact_profile,
    select_generation_profile,
    validate_native_config,
)
from tensorrt_model_connect.families.wan2_2_ti2v.plugin import (
    WAN22_EAGER_BUNDLE_SECTIONS,
    WAN22_LAZY_BUNDLE_SECTIONS,
    WAN22_MODEL_OWNED_BUNDLE_SECTIONS,
    WAN22_REQUIRED_BUNDLE_SECTIONS,
    Wan22TI2VPlugin,
)


def _artifact_profile(profile=WAN22_TI2V_5B) -> dict:
    return artifact_profile(profile)


def test_hf_snapshot_required_files_are_family_owned() -> None:
    manifest_path = Path(__file__).resolve().parents[1] / "MODEL.toml"
    with manifest_path.open("rb") as manifest_file:
        manifest = tomllib.load(manifest_file)
    required_files = {
        spec.split("|", maxsplit=1)[1]
        for spec in manifest["hf_required_files"]
        if spec.split("|", maxsplit=1)[0] == "Wan-AI/Wan2.2-TI2V-5B"
    }
    assert required_files == {
        "config.json",
        "diffusion_pytorch_model.safetensors.index.json",
        "diffusion_pytorch_model-00001-of-00003.safetensors",
        "diffusion_pytorch_model-00002-of-00003.safetensors",
        "diffusion_pytorch_model-00003-of-00003.safetensors",
        "Wan2.2_VAE.pth",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "google/umt5-xxl/tokenizer.json",
    }


def _plugin_contract() -> dict:
    return {
        "schema": 1,
        "family": "wan2_2_ti2v",
        "semantic_abi": "wan2_2_ti2v.plugins.v1",
        "source_digest": "a" * 64,
        "creator_set": "Wan22DitGelu:1:",
        "runtime_abi": {
            "tensorrt_major": 11,
            "tensorrt_minor": 0,
            "cuda_major": 13,
            "cudnn_major": 9,
        },
        "cuda_architectures": [103, 110],
    }


def _bundle_components(profile=WAN22_TI2V_5B) -> dict:
    components = {
        "plugin_contract": _plugin_contract(),
        "plugin_library": b"wan22-plugins-aot",
        "text_encoders": [("umt5_xxl", b"wan22-t5-plan")],
        "denoiser": b"wan22-dit-plan",
        "vae_decoder": b"wan22-vae-recurrent-plan",
        "vae_decoder_first_frame": b"wan22-vae-initializer-plan",
        "tokenizer_json": b'{"model":{"type":"Unigram"}}',
    }
    section_payloads = {
        "text_encoder_0_plan": components["text_encoders"][0][1],
        "denoiser_plan": components["denoiser"],
        "vae_decoder_plan": components["vae_decoder"],
        "vae_decoder_first_frame_plan": components["vae_decoder_first_frame"],
        "tokenizer.json": components["tokenizer_json"],
        "wan2_2_ti2v_plugins.so": components["plugin_library"],
    }
    sections = {}
    for name, payload in section_payloads.items():
        entry = {"sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}
        if name.endswith("_plan"):
            entry["source_inputs"] = [
                {
                    "name": f"checkpoint/{name}",
                    "sha256": hashlib.sha256(f"input:{name}".encode()).hexdigest(),
                },
                {
                    "name": "plugin/contract.json",
                    "sha256": hashlib.sha256(
                        json.dumps(
                            components["plugin_contract"],
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode()
                    ).hexdigest(),
                },
                {
                    "name": "plugin/elf",
                    "sha256": hashlib.sha256(components["plugin_library"]).hexdigest(),
                },
            ]
            source_document = {
                "family": "wan2_2_ti2v",
                "component": name,
                "profile": _artifact_profile(profile),
                "inputs": entry["source_inputs"],
            }
            entry["source_sha256"] = hashlib.sha256(
                json.dumps(
                    source_document,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode()
            ).hexdigest()
        sections[name] = entry
    components["artifact_manifest"] = {
        "schema": "trtmc.wan2_2_ti2v.bundle-artifacts.v4",
        "family": "wan2_2_ti2v",
        "profile": _artifact_profile(profile),
        "runtime": "native_cpp_cuda_tensorrt",
        "sections": sections,
    }
    return components


def _native_config(**overrides: object) -> dict:
    config = {
        "model_type": "ti2v",
        "in_dim": 48,
        "out_dim": 48,
        "dim": 3072,
        "ffn_dim": 14336,
        "freq_dim": 256,
        "num_heads": 24,
        "num_layers": 30,
        "text_len": 512,
    }
    config.update(overrides)
    return config


def _runtime_config(**overrides: object) -> SimpleNamespace:
    raw = {}
    raw.update(overrides)
    return SimpleNamespace(raw=raw)


def test_official_profile_geometry() -> None:
    arch = WAN22_TI2V_5B
    assert (arch.video_width, arch.video_height, arch.video_num_frames) == (
        1280,
        704,
        121,
    )
    assert (arch.latent_frames, arch.latent_height, arch.latent_width) == (
        31,
        44,
        80,
    )
    assert arch.num_patches == 27280
    assert arch.num_inference_steps == 50
    assert arch.guidance_scale == 5.0


def test_l0_profile_matches_wan21_preview_budget() -> None:
    arch = WAN22_TI2V_5B_L0
    assert (arch.video_width, arch.video_height, arch.video_num_frames) == (
        672,
        384,
        5,
    )
    assert (arch.latent_frames, arch.latent_height, arch.latent_width) == (
        2,
        24,
        42,
    )
    assert arch.num_patches == 504
    assert arch.num_inference_steps == 15
    assert arch.guidance_scale == 5.0


def test_generation_profile_selection_is_exact() -> None:
    assert select_generation_profile({}) is WAN22_TI2V_5B
    assert (
        select_generation_profile(
            {
                "video_width": 672,
                "video_height": 384,
                "video_num_frames": 5,
                "num_inference_steps": 15,
            }
        )
        is WAN22_TI2V_5B_L0
    )
    with pytest.raises(ValueError, match="exact qualified generation profile"):
        select_generation_profile(
            {
                "video_width": 672,
                "video_height": 384,
                "video_num_frames": 5,
                "num_inference_steps": 50,
            }
        )


def test_native_config_validation_is_exact() -> None:
    validate_native_config(_native_config())
    with pytest.raises(ValueError, match="num_layers=40"):
        validate_native_config(_native_config(num_layers=40))


def test_component_source_identity_is_complete_and_family_owned() -> None:
    import tensorrt_model_connect.families.wan2_2_ti2v.trt_builder as trt_builder

    for component, source_files in trt_builder._COMPONENT_SOURCE_FILES.items():
        assert "wan2_2_ti2v/model_config.py" in source_files, component
        assert all(path.startswith("wan2_2_ti2v/") for path in source_files), component

    profile = trt_builder._official_profile()
    assert profile["architecture"] == {
        "model_type": "ti2v",
        "in_channels": 48,
        "out_channels": 48,
        "dim": 3072,
        "ffn_dim": 14336,
        "freq_dim": 256,
        "num_heads": 24,
        "num_layers": 30,
        "head_dim": 128,
        "text_dim": 4096,
        "text_seq_len": 512,
        "eps": 1.0e-6,
        "patch_size": [1, 2, 2],
        "z_dim": 48,
        "scale_factor_temporal": 4,
        "scale_factor_spatial": 16,
        "frame_rate": 24,
        "num_inference_steps": 50,
        "guidance_scale": 5.0,
        "flow_shift": 5.0,
        "train_timesteps": 1000,
    }


def test_unipc_header_tracks_the_current_reproducer() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    generator = (
        repo_root
        / "python/tensorrt_model_connect/families/wan2_2_ti2v/reference"
        / "generate_unipc_coefficients.py"
    )
    header = (
        repo_root / "src/runtime/models/wan2_2_ti2v" / "wan2_2_unipc_coefficients.h"
    ).read_text(encoding="utf-8")
    l0_header = (
        repo_root / "src/runtime/models/wan2_2_ti2v" / "wan2_2_unipc_coefficients_15.h"
    ).read_text(encoding="utf-8")

    generator_sha256 = hashlib.sha256(generator.read_bytes()).hexdigest()
    assert generator_sha256 == "adb1e0a3839924ed4982c872909ab044335fa22f8319a5048b2e896e61e053bb"
    assert generator_sha256 in header
    assert generator_sha256 in l0_header
    assert "742ec7777410d94d73c528432e21c22cb52f021d3fa841b8b942b3f9c51ee2e0" in header
    assert "650f7e64cddb551bd81ee4386857967dcf2a916ea3a04cb97423b74a522cf782" in header
    assert "bcaae1cf25b1b11d35dcb75ab87b8659a4cafa33c63e0a9e68494f1d73060dd4" in (l0_header)

    generator_source = generator.read_text(encoding="utf-8")
    assert "choices=QUALIFIED_INFERENCE_STEPS" in generator_source
    assert 'f"wan2_2_ti2v_5b_unipc_{num_inference_steps}_step_cuda_coefficients"' in (
        generator_source
    )


def test_plugin_matches_only_wan22_aliases() -> None:
    plugin = Wan22TI2VPlugin()
    for alias in (
        "ti2v",
        "ti2v-5b",
        "wan2.2-ti2v-5b",
        "wan2_2_ti2v_5b",
        "WanModel",
    ):
        assert plugin.matches(alias)
    assert not plugin.matches("wan_t2v")
    assert not plugin.matches("wan2.1-t2v-1.3b")


def test_load_weights_requires_complete_native_checkpoint(tmp_path) -> None:
    model = tmp_path / "Wan2.2-TI2V-5B"
    tokenizer = model / "google" / "umt5-xxl"
    tokenizer.mkdir(parents=True)
    (model / "config.json").write_text(json.dumps(_native_config()))
    (model / "Wan2.2_VAE.pth").write_bytes(b"vae")
    (model / "models_t5_umt5-xxl-enc-bf16.pth").write_bytes(b"t5")
    (tokenizer / "tokenizer.json").write_text("{}")
    config = SimpleNamespace(raw={})

    weights = Wan22TI2VPlugin().load_weights(str(model), config)

    assert weights["_transformer_dir"] == str(model)
    assert "_wan2_2_model_dir" not in config.raw
    assert "_wan2_2_source_dir" not in config.raw

    (model / "Wan2.2_VAE.pth").unlink()
    with pytest.raises(FileNotFoundError, match="Wan2.2_VAE.pth"):
        Wan22TI2VPlugin().load_weights(str(model), SimpleNamespace(raw={}))


def test_native_vae_loader_accepts_directory_and_resolved_file(tmp_path, monkeypatch) -> None:
    from tensorrt_model_connect.families.wan2_2_ti2v import checkpoint_mapper

    checkpoint = tmp_path / "Wan2.2_VAE.pth"
    checkpoint.write_bytes(b"native-vae")
    loaded_paths: list[Path] = []
    expected = {"decoder.weight": object()}

    def fake_load(path, *, map_location, weights_only):
        assert map_location == "cpu"
        assert weights_only is True
        loaded_paths.append(Path(path))
        return expected

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=fake_load))

    assert checkpoint_mapper.load_native_vae_state_dict(tmp_path) is expected
    assert checkpoint_mapper.load_native_vae_state_dict(checkpoint) is expected
    assert loaded_paths == [checkpoint, checkpoint]


def test_native_bundle_hooks_emit_exact_seven_section_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    plugin = Wan22TI2VPlugin()
    import tensorrt_model_connect.families.wan2_2_ti2v.trt_builder as trt_builder

    components = _bundle_components()
    monkeypatch.setattr(
        trt_builder,
        "build_wan22_components",
        lambda *_args, **_kwargs: components,
    )

    assert (
        plugin.build_components(
            str(tmp_path), _runtime_config(), {"_transformer_dir": str(tmp_path)}
        )
        == components
    )
    assert plugin.diffusion_bundle_sections(components) == [
        ("text_encoder_0_plan", b"wan22-t5-plan"),
        ("denoiser_plan", b"wan22-dit-plan"),
        ("vae_decoder_plan", b"wan22-vae-recurrent-plan"),
        ("vae_decoder_first_frame_plan", b"wan22-vae-initializer-plan"),
        ("tokenizer.json", b'{"model":{"type":"Unigram"}}'),
        ("wan2_2_ti2v_plugins.so", b"wan22-plugins-aot"),
    ]
    assert len(WAN22_MODEL_OWNED_BUNDLE_SECTIONS) == 6
    assert len(WAN22_REQUIRED_BUNDLE_SECTIONS) == 7
    with pytest.raises(NotImplementedError, match="build_components"):
        plugin.build_engine(_runtime_config(), {}, 256)


def test_public_builder_routes_native_config_to_exact_component_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Native WanModel config without model_index uses build_components()."""
    import tensorrt_model_connect.engine_builder as engine_builder

    model_dir = tmp_path / "Wan2.2-TI2V-5B"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"_class_name": "WanModel", "model_type": "ti2v"})
    )
    calls = {"build_components": 0, "build_engine": 0}
    component_sections = list(WAN22_MODEL_OWNED_BUNDLE_SECTIONS)

    class _NativeWanPlugin:
        name = "wan2_2_ti2v"
        runtime_strategy = "diffusion_wan2_2_ti2v"
        pipeline_classes = ("WanModel", "WanPipeline")

        def load_weights(self, _model_dir, config, **_kwargs):
            assert config.raw["_class_name"] == "WanModel"
            return {}

        def build_engine(self, *_args, **_kwargs):
            calls["build_engine"] += 1
            raise AssertionError("native Wan must not use build_engine")

        def build_components(self, *_args, **_kwargs):
            calls["build_components"] += 1
            return {"payloads": {name: name.encode() for name in component_sections}}

        def diffusion_bundle_sections(self, components, **_kwargs):
            return list(components["payloads"].items())

        def diffusion_bundle_config(self, _config, *, components):
            assert set(components["payloads"]) == set(component_sections)
            return {"_trtmc_wan22_plugin_contract": {"schema": "test"}}

        def diffusion_tokenizer_add_special_tokens(self, *_args, **_kwargs):
            return False

        def diffusion_tokenizer_bundle_sections(self, *_args, **_kwargs):
            return []

    plugin = _NativeWanPlugin()
    captured = {}
    monkeypatch.setattr(engine_builder, "find_plugin", lambda _config: plugin)
    monkeypatch.setattr(engine_builder, "find_diffusion_plugin", lambda _class_name: plugin)
    monkeypatch.setattr(engine_builder, "_setup_trt_import", lambda _rtx: None)
    monkeypatch.setattr(engine_builder.trt_compat, "resolved_summary", lambda: "TensorRT test")
    monkeypatch.setattr(engine_builder, "_get_trt_version", lambda: "10.16.2")
    monkeypatch.setattr(engine_builder, "_get_gpu_name", lambda: "NVIDIA Jetson Thor")
    monkeypatch.setattr(
        engine_builder,
        "write_bundle",
        lambda output, info, sections: captured.update(output=output, info=info, sections=sections),
    )

    output_path = str(tmp_path / "wan22.trtfb")
    revision = "a" * 40
    engine_builder.build_bundle(
        str(model_dir),
        output_path,
        source_model_id="Wan-AI/Wan2.2-TI2V-5B",
        source_revision=revision,
    )

    assert calls == {"build_components": 1, "build_engine": 0}
    assert captured["output"] == output_path
    assert captured["info"].family == "wan2_2_ti2v"
    assert captured["info"].precision == "bf16"
    assert captured["info"].model_id == "Wan-AI/Wan2.2-TI2V-5B"
    assert captured["info"].source_revision == revision
    assert [section.name for section in captured["sections"]] == list(
        WAN22_REQUIRED_BUNDLE_SECTIONS
    )
    config_section = next(
        section for section in captured["sections"] if section.name == "config.json"
    )
    bundled_config = json.loads(config_section.data)
    assert "_trtmc_wan22_plugin_contract" in bundled_config
    assert bundled_config["source_model_id"] == "Wan-AI/Wan2.2-TI2V-5B"
    assert bundled_config["source_revision"] == revision


def test_component_builder_embeds_selected_aot_companion_in_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import tensorrt_model_connect.families.wan2_2_ti2v.trt_builder as trt_builder

    plugin_companion = tmp_path / "libtrtmc_model_wan2_2_ti2v_plugins_trt11_1.so"
    plugin_companion.write_bytes(b"qualified-aot-companion")
    (tmp_path / "config.json").write_text(json.dumps(_native_config()))
    (tmp_path / "models_t5_umt5-xxl-enc-bf16.pth").write_bytes(b"official-t5")
    (tmp_path / "Wan2.2_VAE.pth").write_bytes(b"official-vae")
    (tmp_path / "diffusion_pytorch_model.safetensors").write_bytes(b"official-dit")
    tokenizer = tmp_path / "google" / "umt5-xxl"
    tokenizer.mkdir(parents=True)
    tokenizer_json = b'{"model":{"type":"Unigram"}}'
    (tokenizer / "tokenizer.json").write_bytes(tokenizer_json)
    captured = {}

    monkeypatch.setattr(
        trt_builder,
        "_ensure_plugin_companion",
        lambda **_kwargs: SimpleNamespace(
            load_path=plugin_companion,
            elf_bytes=b"qualified-aot-companion",
            elf_sha256=hashlib.sha256(b"qualified-aot-companion").hexdigest(),
            contract=_plugin_contract(),
        ),
    )

    def build_text_encoder(checkpoint, **kwargs):
        captured["checkpoint"] = checkpoint
        captured.update(kwargs)
        return b"t5"

    monkeypatch.setattr(trt_builder, "build_native_umt5_encoder_engine", build_text_encoder)

    def build_denoiser(model_dir, **kwargs):
        captured["denoiser_model_dir"] = model_dir
        captured["denoiser_kwargs"] = kwargs
        return b"dit"

    monkeypatch.setattr(trt_builder, "build_dit_engine", build_denoiser)
    vae_weights = {"decoder.weight": object()}
    monkeypatch.setattr(
        trt_builder,
        "load_vae_step_weights",
        lambda *_args, **_kwargs: vae_weights,
    )

    def build_vae_step(actual_weights, **kwargs):
        assert actual_weights is vae_weights
        captured.setdefault("vae_builds", []).append(kwargs)
        return b"vae-initializer" if kwargs["first_frame_only"] else b"vae-recurrent"

    monkeypatch.setattr(trt_builder, "build_vae_step_engine", build_vae_step)
    monkeypatch.setattr(trt_builder, "_validate_serialized_engine_contract", lambda *_args: None)
    config = _runtime_config()
    components = trt_builder.build_wan22_components(
        str(tmp_path),
        config=config,
        weights={
            "_text_encoder_checkpoint": str(tmp_path / "models_t5_umt5-xxl-enc-bf16.pth"),
            "_vae_checkpoint": str(tmp_path / "Wan2.2_VAE.pth"),
            "_tokenizer_dir": str(tokenizer),
        },
    )

    assert captured["checkpoint"] == str(tmp_path / "models_t5_umt5-xxl-enc-bf16.pth")
    assert captured["source_gelu_plugin"] == plugin_companion
    assert captured["source_softmax"] is True
    assert captured["source_rmsnorm"] is True
    assert captured["denoiser_model_dir"] == str(tmp_path)
    assert captured["denoiser_kwargs"]["cuda_bf16_plugin"] == str(plugin_companion)
    assert captured["denoiser_kwargs"]["dit_cuda_plugin"] == str(plugin_companion)
    assert captured["denoiser_kwargs"]["source_attention_plugin"] is None
    for exact_flag in (
        "dit_bf16_linear",
        "dit_time_silu",
        "dit_time_linear2",
        "dit_time_projection",
        "dit_block_layer_norm",
        "dit_adaptive_norm",
        "dit_rms_norm",
        "dit_self_gated_residual",
        "dit_ffn_gated_residual",
        "dit_cross_affine_layer_norm",
        "dit_final_projection",
    ):
        assert captured["denoiser_kwargs"][exact_flag] is True
    assert components["plugin_contract"] == _plugin_contract()
    assert components["plugin_library"] == b"qualified-aot-companion"
    assert components["text_encoders"] == [("umt5_xxl", b"t5")]
    assert components["denoiser"] == b"dit"
    assert components["vae_decoder"] == b"vae-recurrent"
    assert components["vae_decoder_first_frame"] == b"vae-initializer"
    assert [build["first_frame_only"] for build in captured["vae_builds"]] == [False, True]
    assert all(
        build["profile"].latent_shape == (1, 48, 1, 44, 80) for build in captured["vae_builds"]
    )
    assert components["tokenizer_json"] == tokenizer_json
    manifest = components["artifact_manifest"]
    assert manifest["profile"] == _artifact_profile()
    assert set(manifest["sections"]) == set(WAN22_MODEL_OWNED_BUNDLE_SECTIONS)
    assert len(manifest["sections"]) == 6
    assert manifest["sections"]["wan2_2_ti2v_plugins.so"] == {
        "sha256": hashlib.sha256(b"qualified-aot-companion").hexdigest(),
        "size": len(b"qualified-aot-companion"),
    }
    for name in (
        "text_encoder_0_plan",
        "denoiser_plan",
        "vae_decoder_plan",
        "vae_decoder_first_frame_plan",
    ):
        assert len(manifest["sections"][name]["source_sha256"]) == 64
    text_sources = {
        source["name"] for source in manifest["sections"]["text_encoder_0_plan"]["source_inputs"]
    }
    assert "source/wan2_2_ti2v/cuda_plugin_companion.py" in text_sources
    assert "plugin/contract.json" in text_sources
    assert "plugin/elf" in text_sources
    assert "plugin/source" in text_sources

    captured.clear()
    l0_components = trt_builder.build_wan22_components(
        str(tmp_path),
        config=_runtime_config(
            video_width=672,
            video_height=384,
            video_num_frames=5,
            num_inference_steps=15,
        ),
        weights={
            "_text_encoder_checkpoint": str(tmp_path / "models_t5_umt5-xxl-enc-bf16.pth"),
            "_vae_checkpoint": str(tmp_path / "Wan2.2_VAE.pth"),
            "_tokenizer_dir": str(tokenizer),
        },
    )
    assert captured["denoiser_kwargs"]["latent_frames"] == 2
    assert captured["denoiser_kwargs"]["latent_height"] == 24
    assert captured["denoiser_kwargs"]["latent_width"] == 42
    assert all(
        build["profile"].latent_shape == (1, 48, 1, 24, 42) for build in captured["vae_builds"]
    )
    assert l0_components["artifact_manifest"]["profile"] == _artifact_profile(WAN22_TI2V_5B_L0)


def test_runtime_bundle_contract_is_official_profile() -> None:
    plugin = Wan22TI2VPlugin()
    config = plugin.get_bundle_config_overrides(_runtime_config())
    assert plugin.runtime_strategy == "diffusion_wan2_2_ti2v"
    assert plugin.requires_tokenizer is True
    assert "checkpoint_path" not in config
    assert "official_source_path" not in config
    assert "runtime_python_module" not in config
    assert config["scheduler"] == "unipc_flow"
    assert config["video_num_frames"] == 121
    assert config["num_inference_steps"] == 50
    assert config["guidance_scale"] == 5.0

    bundle_config = plugin.diffusion_bundle_config(
        _runtime_config(), components=_bundle_components()
    )
    assert bundle_config["runtime_contract"] == {
        "implementation": "native_cpp_cuda_tensorrt",
        "artifact_integrity": "sha256_size_v1",
        "bundle_trust_model": "trusted_executable_artifact",
        "executable_bundle_sections": ["wan2_2_ti2v_plugins.so"],
        "required_bundle_sections": list(WAN22_REQUIRED_BUNDLE_SECTIONS),
        "runtime_dependencies": [
            "trtmc_core",
            "cuda",
            "tensorrt",
            "cudnn",
            "cublaslt",
            "nvrtc",
        ],
        "forbidden_runtime_dependencies": [
            "python",
            "pytorch",
            "libpython",
            "libtorch",
        ],
    }
    assert bundle_config["artifact_manifest"] == _bundle_components()["artifact_manifest"]
    assert bundle_config["_trtmc_wan22_plugin_contract"] == _plugin_contract()
    assert bundle_config["bundle_loading"] == {
        "mode": "staged",
        "eager_sections": list(WAN22_EAGER_BUNDLE_SECTIONS),
        "lazy_sections": list(WAN22_LAZY_BUNDLE_SECTIONS),
    }
    assert set(WAN22_EAGER_BUNDLE_SECTIONS).isdisjoint(WAN22_LAZY_BUNDLE_SECTIONS)
    assert set(WAN22_EAGER_BUNDLE_SECTIONS) | set(WAN22_LAZY_BUNDLE_SECTIONS) == set(
        WAN22_REQUIRED_BUNDLE_SECTIONS
    )
    assert "wan2_2_ti2v_plugins.so" in WAN22_LAZY_BUNDLE_SECTIONS
    assert "wan2_2_ti2v_plugins.so" not in WAN22_EAGER_BUNDLE_SECTIONS


def test_tokenizer_is_source_bound_in_model_owned_sections(tmp_path) -> None:
    tokenizer = tmp_path / "google" / "umt5-xxl"
    tokenizer.mkdir(parents=True)
    tokenizer_json = b'{"model":{"type":"Unigram"}}'
    (tokenizer / "tokenizer.json").write_bytes(tokenizer_json)
    sections = Wan22TI2VPlugin().diffusion_tokenizer_bundle_sections(
        tmp_path, ensure_tokenizer_json=lambda _path: None
    )
    assert sections == []
    bundled = dict(Wan22TI2VPlugin().diffusion_bundle_sections(_bundle_components()))
    assert bundled["tokenizer.json"] == tokenizer_json


def test_runtime_accepts_l0_output_profile() -> None:
    config = Wan22TI2VPlugin().get_diffusion_config(
        _runtime_config(
            video_width=672,
            video_height=384,
            video_num_frames=5,
            num_inference_steps=15,
        )
    )
    assert config["video_width"] == 672
    assert config["video_height"] == 384
    assert config["video_num_frames"] == 5
    assert config["num_inference_steps"] == 15


def test_runtime_rejects_non_qualified_output_profile() -> None:
    plugin = Wan22TI2VPlugin()
    with pytest.raises(ValueError, match="exact qualified generation profile"):
        plugin.get_diffusion_config(_runtime_config(video_num_frames=81))


def test_l0_artifact_manifest_matches_l0_bundle_config() -> None:
    plugin = Wan22TI2VPlugin()
    raw = _runtime_config(
        video_width=672,
        video_height=384,
        video_num_frames=5,
        num_inference_steps=15,
    )
    components = _bundle_components(WAN22_TI2V_5B_L0)
    config = plugin.diffusion_bundle_config(raw, components=components)
    assert config["artifact_manifest"]["profile"] == _artifact_profile(WAN22_TI2V_5B_L0)

    with pytest.raises(ValueError, match="does not match the requested bundle profile"):
        plugin.diffusion_bundle_config(_runtime_config(), components=components)


@pytest.mark.parametrize("seed", [-2, -1, 2_147_483_648])
def test_runtime_rejects_bundle_seed_outside_native_range(seed: int) -> None:
    plugin = Wan22TI2VPlugin()
    with pytest.raises(
        ValueError,
        match="bundle seed must be between 0 and 2147483647",
    ):
        plugin.get_diffusion_config(_runtime_config(seed=seed))


@pytest.mark.parametrize("seed", [0, 42, 2_147_483_647])
def test_runtime_accepts_bundle_seed_in_native_range(seed: int) -> None:
    assert Wan22TI2VPlugin().get_diffusion_config(_runtime_config(seed=seed))["seed"] == seed


def _read_test_prebuilt(
    trt_builder,
    tmp_path: Path,
    *,
    source_sha256: str,
    component: str = "vae_decoder_plan",
) -> bytes | None:
    return trt_builder._read_prebuilt(
        _runtime_config(_test_prebuilt=str(tmp_path / "vae.plan")),
        "_test_prebuilt",
        "WAN22_TEST_PREBUILT_UNUSED",
        component=component,
        source_sha256=source_sha256,
        manifest_config_key="_test_prebuilt_manifest",
        manifest_environment_key="WAN22_TEST_PREBUILT_MANIFEST_UNUSED",
    )


def test_bare_prebuilt_plan_is_rejected_with_manifest_helper(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tensorrt_model_connect.families.wan2_2_ti2v.trt_builder as trt_builder

    monkeypatch.delenv("WAN22_TEST_PREBUILT_UNUSED", raising=False)
    monkeypatch.delenv("WAN22_TEST_PREBUILT_MANIFEST_UNUSED", raising=False)
    (tmp_path / "vae.plan").write_bytes(b"qualified-plan")
    with pytest.raises(FileNotFoundError, match=r"write_wan22_prebuilt_manifest\(\)"):
        _read_test_prebuilt(trt_builder, tmp_path, source_sha256="a" * 64)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("source_sha256", "b" * 64, "mismatched source_sha256"),
        ("component", "denoiser_plan", "mismatched component"),
        (
            "profile",
            {**_artifact_profile(), "video_num_frames": 81},
            "mismatched profile",
        ),
    ),
)
def test_prebuilt_manifest_rejects_stale_source_or_static_contract(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
    message: str,
) -> None:
    import tensorrt_model_connect.families.wan2_2_ti2v.trt_builder as trt_builder

    monkeypatch.delenv("WAN22_TEST_PREBUILT_UNUSED", raising=False)
    monkeypatch.delenv("WAN22_TEST_PREBUILT_MANIFEST_UNUSED", raising=False)
    plan = b"qualified-plan"
    plan_path = tmp_path / "vae.plan"
    plan_path.write_bytes(plan)
    manifest = trt_builder._prebuilt_manifest_payload("vae_decoder_plan", plan, "a" * 64)
    manifest[field] = replacement
    Path(f"{plan_path}.manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=message):
        _read_test_prebuilt(trt_builder, tmp_path, source_sha256="a" * 64)


def test_prebuilt_manifest_rejects_plan_sha256_mismatch(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tensorrt_model_connect.families.wan2_2_ti2v.trt_builder as trt_builder

    monkeypatch.delenv("WAN22_TEST_PREBUILT_UNUSED", raising=False)
    monkeypatch.delenv("WAN22_TEST_PREBUILT_MANIFEST_UNUSED", raising=False)
    plan_path = tmp_path / "vae.plan"
    plan_path.write_bytes(b"qualified-plan")
    manifest = trt_builder._prebuilt_manifest_payload(
        "vae_decoder_plan", b"different-plan", "a" * 64
    )
    Path(f"{plan_path}.manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="plan SHA256 mismatch"):
        _read_test_prebuilt(trt_builder, tmp_path, source_sha256="a" * 64)


@pytest.mark.parametrize(
    "component",
    ("vae_decoder_plan", "vae_decoder_first_frame_plan"),
)
def test_matching_prebuilt_manifest_is_accepted(
    tmp_path, monkeypatch: pytest.MonkeyPatch, component: str
) -> None:
    import tensorrt_model_connect.families.wan2_2_ti2v.trt_builder as trt_builder

    monkeypatch.delenv("WAN22_TEST_PREBUILT_UNUSED", raising=False)
    monkeypatch.delenv("WAN22_TEST_PREBUILT_MANIFEST_UNUSED", raising=False)
    plan = b"qualified-plan"
    plan_path = tmp_path / "vae.plan"
    plan_path.write_bytes(plan)
    manifest = trt_builder._prebuilt_manifest_payload(component, plan, "a" * 64)
    Path(f"{plan_path}.manifest.json").write_text(json.dumps(manifest))

    assert (
        _read_test_prebuilt(
            trt_builder,
            tmp_path,
            source_sha256="a" * 64,
            component=component,
        )
        == plan
    )


@pytest.mark.parametrize(
    ("component", "output_frames"),
    (("vae_decoder_plan", 4), ("vae_decoder_first_frame_plan", 1)),
)
def test_serialized_vae_step_engine_contract_is_exact(
    monkeypatch: pytest.MonkeyPatch, component: str, output_frames: int
) -> None:
    import tensorrt_model_connect.families.wan2_2_ti2v.trt_builder as trt_builder
    from tensorrt_model_connect.families.wan2_2_ti2v.vae_step_builder import (
        OFFICIAL_VAE_STEP_PROFILE,
        VAE_STEP_CACHE_SPECS,
    )

    expected = trt_builder._expected_engine_contract(component)
    assert len(expected) == 66
    assert expected["latent_frame"] == ("input", (1, 48, 1, 44, 80), "float")
    assert expected["video_frame"] == (
        "output",
        (1, 3, output_frames, 704, 1280),
        "float",
    )
    for spec in VAE_STEP_CACHE_SPECS:
        shape = spec.shape(OFFICIAL_VAE_STEP_PROFILE)
        assert expected[f"cache_{spec.index}"] == ("input", shape, "float")
        assert expected[f"cache_out_{spec.index}"] == ("output", shape, "float")
    monkeypatch.setattr(trt_builder, "_inspect_serialized_engine", lambda _plan: expected)
    trt_builder._validate_serialized_engine_contract(b"plan", component)

    wrong = dict(expected)
    wrong["video_frame"] = ("output", (1, 3, 121, 704, 1280), "float")
    monkeypatch.setattr(trt_builder, "_inspect_serialized_engine", lambda _plan: wrong)
    with pytest.raises(ValueError, match="I/O contract mismatch"):
        trt_builder._validate_serialized_engine_contract(b"wrong-plan", component)


def test_l0_serialized_engine_contract_uses_reduced_static_shapes() -> None:
    import tensorrt_model_connect.families.wan2_2_ti2v.trt_builder as trt_builder

    denoiser = trt_builder._expected_engine_contract("denoiser_plan", WAN22_TI2V_5B_L0)
    assert denoiser["latents"] == ("input", (1, 48, 2, 24, 42), "float")
    assert denoiser["noise_prediction"] == (
        "output",
        (1, 48, 2, 24, 42),
        "float",
    )

    recurrent = trt_builder._expected_engine_contract("vae_decoder_plan", WAN22_TI2V_5B_L0)
    initializer = trt_builder._expected_engine_contract(
        "vae_decoder_first_frame_plan", WAN22_TI2V_5B_L0
    )
    assert recurrent["latent_frame"] == ("input", (1, 48, 1, 24, 42), "float")
    assert recurrent["video_frame"] == ("output", (1, 3, 4, 384, 672), "float")
    assert initializer["video_frame"] == ("output", (1, 3, 1, 384, 672), "float")


def test_bundle_section_and_native_runtime_dependency_metadata_are_exact() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    python_manifest = tomllib.loads(
        (repo_root / "python/tensorrt_model_connect/families/wan2_2_ti2v/MODEL.toml").read_text()
    )
    runtime_dir = repo_root / "src/runtime/models/wan2_2_ti2v"
    runtime_manifest = tomllib.loads((runtime_dir / "MODEL.toml").read_text())
    expected_dependencies = [
        "trtmc_core",
        "cuda",
        "tensorrt",
        "cudnn",
        "cublaslt",
        "nvrtc",
    ]
    forbidden_dependencies = ["python", "pytorch", "libpython", "libtorch"]

    assert python_manifest["default_build_precision"] == "bf16"
    assert python_manifest["supported_build_precisions"] == ["bf16"]
    assert python_manifest["build_python_modules"] == ["torch"]
    assert python_manifest["build_dependency_extra"] == "wan"

    for manifest in (python_manifest, runtime_manifest):
        assert manifest["runtime_implementation"] == "native_cpp_cuda_tensorrt"
        assert tuple(manifest["bundle_required_sections"]) == (WAN22_REQUIRED_BUNDLE_SECTIONS)
        assert manifest["runtime_dependencies"] == expected_dependencies
        assert manifest["forbidden_runtime_dependencies"] == forbidden_dependencies

    runtime_plugin_source = (runtime_dir / "plugin.cpp").read_text()
    for section in WAN22_MODEL_OWNED_BUNDLE_SECTIONS:
        assert f'"{section}"' in runtime_plugin_source

    forbidden_source_markers = (
        "Python.h",
        "pybind11",
        "ATen/",
        "#include <torch/",
        '#include "torch/',
        "libpython",
        "libtorch",
    )
    runtime_sources = [
        *runtime_dir.glob("*.cpp"),
        *runtime_dir.glob("*.cu"),
        *runtime_dir.glob("*.h"),
    ]
    for source in runtime_sources:
        text = source.read_text()
        for marker in forbidden_source_markers:
            assert marker not in text, f"{source.name} implies runtime dependency {marker}"


def test_public_build_uses_wan_family_bf16_default(monkeypatch) -> None:
    import tensorrt_model_connect.engine_builder as engine_builder
    import tensorrt_model_connect.runtime_provider.orchestrator as orchestrator

    native_options: list[dict] = []
    monkeypatch.setattr(
        orchestrator,
        "discover_family_implementations_for_model",
        lambda _family, _model: (),
    )
    monkeypatch.setattr(
        engine_builder,
        "preflight_family_build_dependencies",
        lambda _family: None,
    )
    monkeypatch.setattr(
        engine_builder,
        "_try_build_optimized_runtime",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Wan without a capsule must not resolve twice")
        ),
    )
    monkeypatch.setattr(
        engine_builder,
        "_build_native_impl",
        lambda **kwargs: native_options.append(kwargs),
    )

    engine_builder.build(
        "Wan-AI/Wan2.2-TI2V-5B",
        "wan.trtfb",
    )

    assert len(native_options) == 1
    assert native_options[0]["model_id_or_path"] == "Wan-AI/Wan2.2-TI2V-5B"
    assert native_options[0]["precision"] == "bf16"


def test_public_build_rejects_unsupported_wan_precision_before_routing(monkeypatch) -> None:
    import tensorrt_model_connect.engine_builder as engine_builder

    monkeypatch.setattr(
        engine_builder,
        "_try_build_optimized_runtime",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unsupported precision reached optimized routing")
        ),
    )

    with pytest.raises(ValueError, match="does not support build precision 'fp32'"):
        engine_builder.build(
            "Wan-AI/Wan2.2-TI2V-5B",
            "wan.trtfb",
            precision="fp32",
        )


def test_cuda_plugin_dependency_gate_rejects_torch_before_registration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import tensorrt_model_connect.families.wan2_2_ti2v.trt_builder as trt_builder

    plugin_path = tmp_path / "libwan22_cuda_plugin.so"
    plugin_path.write_bytes(b"ELF fixture is mocked through readelf")

    class Result:
        stdout = """
 0x0000000000000001 (NEEDED) Shared library: [libcudart.so.13]
 0x0000000000000001 (NEEDED) Shared library: [libtorch_cuda.so]
 0x0000000000000001 (NEEDED) Shared library: [libc10.so]
"""

    monkeypatch.setattr(trt_builder.subprocess, "run", lambda *_args, **_kwargs: Result())
    with pytest.raises(ValueError, match="forbidden Python/PyTorch runtime dependencies"):
        trt_builder._validate_plugin_runtime_dependencies(plugin_path)


def test_cuda_plugin_dependency_gate_accepts_native_cuda_dependencies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import tensorrt_model_connect.families.wan2_2_ti2v.trt_builder as trt_builder

    plugin_path = tmp_path / "libwan22_cuda_plugin.so"
    plugin_path.write_bytes(b"ELF fixture is mocked through readelf")

    class Result:
        stdout = """
 0x0000000000000001 (NEEDED) Shared library: [libcudart.so.13]
 0x0000000000000001 (NEEDED) Shared library: [libcudnn.so.9]
 0x0000000000000001 (NEEDED) Shared library: [libstdc++.so.6]
"""

    monkeypatch.setattr(trt_builder.subprocess, "run", lambda *_args, **_kwargs: Result())
    assert trt_builder._validate_plugin_runtime_dependencies(plugin_path) == (
        "libcudart.so.13",
        "libcudnn.so.9",
        "libstdc++.so.6",
    )


def test_artifact_manifest_rejects_mutated_section_bytes() -> None:
    plugin = Wan22TI2VPlugin()
    components = _bundle_components()
    components["denoiser"] = b"stale-or-mutated-plan"
    with pytest.raises(ValueError, match="artifact SHA256 mismatch for denoiser_plan"):
        plugin.diffusion_bundle_sections(components)

    components = _bundle_components()
    components["artifact_manifest"]["sections"]["denoiser_plan"]["source_inputs"][0]["sha256"] = (
        "0" * 64
    )
    with pytest.raises(ValueError, match="source identity mismatch for denoiser_plan"):
        plugin.diffusion_bundle_sections(components)

    components = _bundle_components()
    denoiser_inputs = components["artifact_manifest"]["sections"]["denoiser_plan"]["source_inputs"]
    next(source for source in denoiser_inputs if source["name"] == "plugin/elf")["sha256"] = (
        "0" * 64
    )
    with pytest.raises(ValueError, match="bound to a different AOT plugin ELF"):
        plugin.diffusion_bundle_sections(components)

    components = _bundle_components()
    components["vae_decoder_first_frame"] = b"stale-initializer-plan"
    with pytest.raises(
        ValueError,
        match="artifact SHA256 mismatch for vae_decoder_first_frame_plan",
    ):
        plugin.diffusion_bundle_sections(components)

    components = _bundle_components()
    components["artifact_manifest"]["sections"]["tokenizer.json"]["size"] += 1
    with pytest.raises(ValueError, match="artifact size mismatch for tokenizer.json"):
        plugin.diffusion_bundle_sections(components)


@pytest.mark.parametrize("missing_key", ("vae_decoder", "vae_decoder_first_frame"))
def test_bundle_rejects_missing_or_extra_components(missing_key: str) -> None:
    plugin = Wan22TI2VPlugin()
    missing = _bundle_components()
    del missing[missing_key]
    with pytest.raises(ValueError, match="must contain exactly"):
        plugin.diffusion_bundle_sections(missing)

    extra = _bundle_components()
    extra["runtime_python_module"] = b"forbidden"
    with pytest.raises(ValueError, match="must contain exactly"):
        plugin.diffusion_bundle_sections(extra)
