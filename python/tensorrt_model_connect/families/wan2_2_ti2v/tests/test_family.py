# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the native Wan2.2-TI2V-5B family."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

from tensorrt_model_connect.families.wan2_2_ti2v.model_config import (
    WAN22_TI2V_5B,
    official_artifact_profile,
    validate_native_config,
)
from tensorrt_model_connect.families.wan2_2_ti2v.plugin import (
    WAN22_EAGER_BUNDLE_SECTIONS,
    WAN22_LAZY_BUNDLE_SECTIONS,
    WAN22_MODEL_OWNED_BUNDLE_SECTIONS,
    WAN22_REQUIRED_BUNDLE_SECTIONS,
    Wan22TI2VPlugin,
)


def _artifact_profile() -> dict:
    return official_artifact_profile()


def _bundle_components() -> dict:
    components = {
        "umt5_cuda_plugin": b"wan22-umt5-cuda-plugin",
        "dit_cuda_plugin": b"wan22-dit-cuda-plugin",
        "vae_cuda_plugin": b"wan22-vae-cuda-plugin",
        "text_encoders": [("umt5_xxl", b"wan22-t5-plan")],
        "denoiser": b"wan22-dit-plan",
        "vae_decoder": b"wan22-vae-recurrent-plan",
        "vae_decoder_first_frame": b"wan22-vae-initializer-plan",
        "tokenizer_json": b'{"model":{"type":"Unigram"}}',
    }
    section_payloads = {
        "wan2_2_umt5_cuda_plugin_so": components["umt5_cuda_plugin"],
        "wan2_2_dit_cuda_plugin_so": components["dit_cuda_plugin"],
        "wan2_2_vae_cuda_plugin_so": components["vae_cuda_plugin"],
        "text_encoder_0_plan": components["text_encoders"][0][1],
        "denoiser_plan": components["denoiser"],
        "vae_decoder_plan": components["vae_decoder"],
        "vae_decoder_first_frame_plan": components["vae_decoder_first_frame"],
        "tokenizer.json": components["tokenizer_json"],
    }
    sections = {}
    for name, payload in section_payloads.items():
        entry = {"sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}
        if name.endswith("_plan"):
            entry["source_inputs"] = [
                {
                    "name": f"checkpoint/{name}",
                    "sha256": hashlib.sha256(f"input:{name}".encode()).hexdigest(),
                }
            ]
            source_document = {
                "family": "wan2_2_ti2v",
                "component": name,
                "profile": _artifact_profile(),
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
        "schema": "trtmc.wan2_2_ti2v.bundle-artifacts.v2",
        "family": "wan2_2_ti2v",
        "profile": _artifact_profile(),
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

    generator_sha256 = hashlib.sha256(generator.read_bytes()).hexdigest()
    assert generator_sha256 == "f5b513be69c6626b5311f57995da574a2c5bc21a785d722910139bc3fd048de6"
    assert generator_sha256 in header
    assert "742ec7777410d94d73c528432e21c22cb52f021d3fa841b8b942b3f9c51ee2e0" in header


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


def test_native_bundle_hooks_require_all_four_tensorrt_components(
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
        ("wan2_2_umt5_cuda_plugin_so", b"wan22-umt5-cuda-plugin"),
        ("wan2_2_dit_cuda_plugin_so", b"wan22-dit-cuda-plugin"),
        ("wan2_2_vae_cuda_plugin_so", b"wan22-vae-cuda-plugin"),
        ("text_encoder_0_plan", b"wan22-t5-plan"),
        ("denoiser_plan", b"wan22-dit-plan"),
        ("vae_decoder_plan", b"wan22-vae-recurrent-plan"),
        ("vae_decoder_first_frame_plan", b"wan22-vae-initializer-plan"),
        ("tokenizer.json", b'{"model":{"type":"Unigram"}}'),
    ]
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
            return {}

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
    engine_builder.build_bundle(str(model_dir), output_path, precision="bf16")

    assert calls == {"build_components": 1, "build_engine": 0}
    assert captured["output"] == output_path
    assert captured["info"].family == "wan2_2_ti2v"
    assert [section.name for section in captured["sections"]] == list(
        WAN22_REQUIRED_BUNDLE_SECTIONS
    )


def test_component_builder_packages_and_registers_cuda_plugin(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import tensorrt_model_connect.families.wan2_2_ti2v.trt_builder as trt_builder

    umt5_cuda_plugin = tmp_path / "libtrtmc_wan22_umt5_cuda_plugin.so"
    dit_cuda_plugin = tmp_path / "libtrtmc_wan22_dit_cuda_plugin.so"
    vae_cuda_plugin = tmp_path / "libtrtmc_wan22_vae_cuda_plugin.so"
    umt5_cuda_plugin.write_bytes(b"pure-umt5-cuda-plugin")
    dit_cuda_plugin.write_bytes(b"pure-dit-cuda-plugin")
    vae_cuda_plugin.write_bytes(b"pure-vae-cuda-plugin")
    (tmp_path / "config.json").write_text(json.dumps(_native_config()))
    (tmp_path / "models_t5_umt5-xxl-enc-bf16.pth").write_bytes(b"official-t5")
    (tmp_path / "Wan2.2_VAE.pth").write_bytes(b"official-vae")
    (tmp_path / "diffusion_pytorch_model.safetensors").write_bytes(b"official-dit")
    tokenizer = tmp_path / "google" / "umt5-xxl"
    tokenizer.mkdir(parents=True)
    tokenizer_json = b'{"model":{"type":"Unigram"}}'
    (tokenizer / "tokenizer.json").write_bytes(tokenizer_json)
    captured = {}

    monkeypatch.setattr(trt_builder, "ensure_umt5_cuda_plugin", lambda **_kwargs: umt5_cuda_plugin)
    monkeypatch.setattr(trt_builder, "ensure_dit_cuda_plugin", lambda **_kwargs: dit_cuda_plugin)
    monkeypatch.setattr(trt_builder, "ensure_vae_cuda_plugin", lambda **_kwargs: vae_cuda_plugin)
    monkeypatch.setattr(trt_builder, "_validate_plugin_runtime_dependencies", lambda _path: ())
    monkeypatch.setattr(trt_builder, "_register_plugin_library", lambda _path: None)

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
    assert captured["source_gelu_plugin"] == umt5_cuda_plugin
    assert captured["source_softmax"] is True
    assert captured["source_rmsnorm"] is True
    assert captured["denoiser_model_dir"] == str(tmp_path)
    assert captured["denoiser_kwargs"]["cuda_bf16_plugin"] == str(umt5_cuda_plugin)
    assert captured["denoiser_kwargs"]["dit_cuda_plugin"] == str(dit_cuda_plugin)
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
    assert components["umt5_cuda_plugin"] == b"pure-umt5-cuda-plugin"
    assert components["dit_cuda_plugin"] == b"pure-dit-cuda-plugin"
    assert components["vae_cuda_plugin"] == b"pure-vae-cuda-plugin"
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
    assert "source/wan2_2_ti2v/umt5_cuda_plugins/CMakeLists.txt" in text_sources
    assert "source/wan2_2_ti2v/umt5_cuda_plugins/wan22_umt5_gelu_plugin.cu" in text_sources


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
        "required_bundle_sections": list(WAN22_REQUIRED_BUNDLE_SECTIONS),
        "runtime_dependencies": ["trtmc_core", "cuda", "tensorrt", "cudnn"],
        "forbidden_runtime_dependencies": [
            "python",
            "pytorch",
            "libpython",
            "libtorch",
        ],
    }
    assert bundle_config["artifact_manifest"] == _bundle_components()["artifact_manifest"]
    assert bundle_config["bundle_loading"] == {
        "mode": "staged",
        "eager_sections": list(WAN22_EAGER_BUNDLE_SECTIONS),
        "lazy_sections": list(WAN22_LAZY_BUNDLE_SECTIONS),
    }
    assert set(WAN22_EAGER_BUNDLE_SECTIONS).isdisjoint(WAN22_LAZY_BUNDLE_SECTIONS)
    assert set(WAN22_EAGER_BUNDLE_SECTIONS) | set(WAN22_LAZY_BUNDLE_SECTIONS) == set(
        WAN22_REQUIRED_BUNDLE_SECTIONS
    )


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


def test_runtime_rejects_non_official_output_profile() -> None:
    plugin = Wan22TI2VPlugin()
    with pytest.raises(ValueError, match="fixed to the official"):
        plugin.get_diffusion_config(_runtime_config(video_num_frames=81))


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


def test_bundle_section_and_native_runtime_dependency_metadata_are_exact() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    python_manifest = tomllib.loads(
        (repo_root / "python/tensorrt_model_connect/families/wan2_2_ti2v/MODEL.toml").read_text()
    )
    runtime_dir = repo_root / "src/runtime/models/wan2_2_ti2v"
    runtime_manifest = tomllib.loads((runtime_dir / "MODEL.toml").read_text())
    expected_dependencies = ["trtmc_core", "cuda", "tensorrt", "cudnn"]
    forbidden_dependencies = ["python", "pytorch", "libpython", "libtorch"]

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
