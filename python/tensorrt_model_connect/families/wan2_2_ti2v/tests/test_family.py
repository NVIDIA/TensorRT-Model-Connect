# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Customer-facing contracts for the native-only Wan2.2 TI2V-5B family."""

from __future__ import annotations

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
    WAN22_TI2V_5B_L0,
    select_generation_profile,
    validate_native_config,
)
from tensorrt_model_connect.families.wan2_2_ti2v.plugin import (
    WAN22_EAGER_BUNDLE_SECTIONS,
    WAN22_LAZY_BUNDLE_SECTIONS,
    WAN22_MODEL_OWNED_BUNDLE_SECTIONS,
    Wan22TI2VPlugin,
)


def _native_config(**overrides: object) -> dict:
    config = {
        "_class_name": "WanModel",
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
    return SimpleNamespace(raw=dict(overrides))


def _bundle_components() -> dict:
    return {
        "text_encoders": [("umt5_xxl", b"wan22-t5-plan")],
        "denoiser": b"wan22-dit-plan",
        "vae_decoder": b"wan22-vae-recurrent-plan",
        "vae_decoder_first_frame": b"wan22-vae-initializer-plan",
        "tokenizer_json": b'{"model":{"type":"Unigram"}}',
    }


def _write_checkpoint(root: Path) -> Path:
    tokenizer = root / "google" / "umt5-xxl"
    tokenizer.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps(_native_config()), encoding="utf-8")
    (root / "Wan2.2_VAE.pth").write_bytes(b"vae")
    (root / "models_t5_umt5-xxl-enc-bf16.pth").write_bytes(b"t5")
    (root / "diffusion_pytorch_model.safetensors").write_bytes(b"dit")
    (tokenizer / "tokenizer.json").write_bytes(b'{"model":{"type":"Unigram"}}')
    return tokenizer


def test_hf_snapshot_required_files_are_family_owned() -> None:
    manifest_path = Path(__file__).resolve().parents[1] / "MODEL.toml"
    with manifest_path.open("rb") as manifest_file:
        manifest = tomllib.load(manifest_file)
    assert manifest["aliases"] == ["ti2v"]
    assert "prefixes" not in manifest
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


def test_qualified_generation_profiles_are_exact() -> None:
    official = WAN22_TI2V_5B
    assert (official.video_width, official.video_height, official.video_num_frames) == (
        1280,
        704,
        121,
    )
    assert (official.latent_frames, official.latent_height, official.latent_width) == (
        31,
        44,
        80,
    )
    assert official.num_patches == 27280
    assert official.num_inference_steps == 50
    assert official.guidance_scale == 5.0

    l0 = WAN22_TI2V_5B_L0
    assert (l0.video_width, l0.video_height, l0.video_num_frames) == (672, 384, 5)
    assert (l0.latent_frames, l0.latent_height, l0.latent_width) == (2, 24, 42)
    assert l0.num_patches == 504
    assert l0.num_inference_steps == 15

    assert select_generation_profile({}) is official
    assert (
        select_generation_profile(
            {
                "video_width": 672,
                "video_height": 384,
                "video_num_frames": 5,
                "num_inference_steps": 15,
            }
        )
        is l0
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


def test_native_checkpoint_config_is_exact() -> None:
    validate_native_config(_native_config())
    with pytest.raises(ValueError, match="num_layers=40"):
        validate_native_config(_native_config(num_layers=40))


def test_plugin_matches_only_official_model_type_and_family_id() -> None:
    plugin = Wan22TI2VPlugin()
    assert plugin.pipeline_classes == ("WanModel",)
    assert plugin.matches("ti2v")
    assert plugin.matches("wan2_2_ti2v")
    for unrelated in (
        "ti2v-5b",
        "wan2.2-ti2v-5b",
        "wan2_2_ti2v_5b",
        "wan_t2v",
        "wan2.1-t2v-1.3b",
        "WanModel",
        "WanPipeline",
    ):
        assert not plugin.matches(unrelated)


def test_load_weights_requires_complete_native_checkpoint(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)
    config = _runtime_config()

    weights = Wan22TI2VPlugin().load_weights(str(tmp_path), config)

    assert weights["_vae_checkpoint"] == str(tmp_path / "Wan2.2_VAE.pth")
    assert weights["_text_encoder_checkpoint"] == str(tmp_path / "models_t5_umt5-xxl-enc-bf16.pth")
    assert weights["_tokenizer_dir"] == str(tmp_path / "google" / "umt5-xxl")

    (tmp_path / "Wan2.2_VAE.pth").unlink()
    with pytest.raises(FileNotFoundError, match="Wan2.2_VAE.pth"):
        Wan22TI2VPlugin().load_weights(str(tmp_path), _runtime_config())


def test_bundle_hooks_emit_staged_native_bundle() -> None:
    plugin = Wan22TI2VPlugin()
    components = _bundle_components()

    assert plugin.diffusion_bundle_sections(components) == [
        ("text_encoder_0_plan", b"wan22-t5-plan"),
        ("denoiser_plan", b"wan22-dit-plan"),
        ("vae_decoder_plan", b"wan22-vae-recurrent-plan"),
        ("vae_decoder_first_frame_plan", b"wan22-vae-initializer-plan"),
        ("tokenizer.json", b'{"model":{"type":"Unigram"}}'),
    ]
    assert len(WAN22_MODEL_OWNED_BUNDLE_SECTIONS) == 5
    with pytest.raises(NotImplementedError, match="build_components"):
        plugin.build_engine(_runtime_config(), {}, 512)

    bundle_config = plugin.diffusion_bundle_config(_runtime_config(), components=components)
    assert bundle_config["bundle_loading"] == {
        "mode": "staged",
        "eager_sections": list(WAN22_EAGER_BUNDLE_SECTIONS),
        "lazy_sections": list(WAN22_LAZY_BUNDLE_SECTIONS),
    }
    assert set(WAN22_EAGER_BUNDLE_SECTIONS).isdisjoint(WAN22_LAZY_BUNDLE_SECTIONS)
    assert set(WAN22_EAGER_BUNDLE_SECTIONS) | set(WAN22_LAZY_BUNDLE_SECTIONS) == set(
        (*WAN22_MODEL_OWNED_BUNDLE_SECTIONS, "config.json")
    )
    assert not any(
        "plugin" in section for section in (*WAN22_MODEL_OWNED_BUNDLE_SECTIONS, "config.json")
    )


def test_public_builder_routes_wan_to_component_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import tensorrt_model_connect.engine_builder as engine_builder

    _write_checkpoint(tmp_path)
    plugin = Wan22TI2VPlugin()
    components = _bundle_components()
    calls = []
    monkeypatch.setattr(
        plugin,
        "build_components",
        lambda *args, **kwargs: calls.append((args, kwargs)) or components,
    )
    monkeypatch.setattr(engine_builder, "find_plugin", lambda _config: plugin)
    monkeypatch.setattr(engine_builder, "find_diffusion_plugin", lambda _class_name: plugin)
    monkeypatch.setattr(engine_builder, "_setup_trt_import", lambda _rtx: None)
    monkeypatch.setattr(engine_builder.trt_compat, "resolved_summary", lambda: "TensorRT test")
    monkeypatch.setattr(engine_builder, "_get_trt_version", lambda: "11.2.0")
    monkeypatch.setattr(engine_builder, "_get_gpu_name", lambda: "NVIDIA GB300")
    captured = {}
    monkeypatch.setattr(
        engine_builder,
        "write_bundle",
        lambda output, info, sections: captured.update(output=output, info=info, sections=sections),
    )

    output_path = str(tmp_path / "wan22.trtfb")
    engine_builder.build_bundle(
        str(tmp_path),
        output_path,
    )

    assert len(calls) == 1
    assert calls[0][1]["precision"] == "bf16"
    assert captured["output"] == output_path
    assert captured["info"].family == "wan2_2_ti2v"
    assert captured["info"].precision == "bf16"
    assert [section.name for section in captured["sections"]] == [
        *WAN22_MODEL_OWNED_BUNDLE_SECTIONS,
        "config.json",
    ]


def test_component_builder_emits_four_native_plans(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tensorrt_model_connect.families.wan2_2_ti2v import trt_builder

    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer_payload = b'{"model":{"type":"Unigram"}}'
    (tokenizer_dir / "tokenizer.json").write_bytes(tokenizer_payload)
    weights = {
        "_text_encoder_checkpoint": str(tmp_path / "umt5.pth"),
        "_vae_checkpoint": str(tmp_path / "vae.pth"),
        "_tokenizer_dir": str(tokenizer_dir),
    }
    monkeypatch.setattr(
        trt_builder,
        "build_native_umt5_encoder_engine",
        lambda *_args, **_kwargs: b"text-plan",
    )
    denoiser_calls = []
    monkeypatch.setattr(
        trt_builder,
        "build_dit_engine",
        lambda *_args, **kwargs: denoiser_calls.append(kwargs) or b"dit-plan",
    )
    monkeypatch.setattr(
        trt_builder,
        "load_vae_step_weights",
        lambda _checkpoint: object(),
    )
    monkeypatch.setattr(
        trt_builder,
        "build_vae_step_engine",
        lambda _weights, **kwargs: b"vae-first" if kwargs["first_frame_only"] else b"vae-recurrent",
    )
    components = trt_builder.build_wan22_components(
        str(tmp_path),
        config=_runtime_config(),
        weights=weights,
    )

    assert components["text_encoders"] == [("umt5_xxl", b"text-plan")]
    assert components["denoiser"] == b"dit-plan"
    assert components["vae_decoder"] == b"vae-recurrent"
    assert components["vae_decoder_first_frame"] == b"vae-first"
    assert components["tokenizer_json"] == tokenizer_payload
    assert denoiser_calls[0]["profile"] is WAN22_TI2V_5B
    assert denoiser_calls[0]["ffn_fp8_scales"] is None
    assert denoiser_calls[0]["cross_qo_fp8_scales"] is None

    trt_builder.build_wan22_components(
        str(tmp_path),
        config=_runtime_config(
            video_width=672,
            video_height=384,
            video_num_frames=5,
            num_inference_steps=15,
        ),
        weights=weights,
    )
    assert denoiser_calls[1]["profile"] is WAN22_TI2V_5B_L0
    assert denoiser_calls[1]["ffn_fp8_scales"] is None
    assert denoiser_calls[1]["cross_qo_fp8_scales"] is None

    ffn_scales = {
        name: {"input_scale": 0.125}
        for index in range(WAN22_TI2V_5B.num_layers)
        for name in (
            f"blocks.{index}.ffn.net.0.proj",
            f"blocks.{index}.ffn.net.2",
        )
    }
    trt_builder.build_wan22_components(
        str(tmp_path),
        config=_runtime_config(),
        weights=weights,
        fp8_scales=ffn_scales,
    )
    assert denoiser_calls[2]["profile"] is WAN22_TI2V_5B
    assert denoiser_calls[2]["ffn_fp8_scales"] is ffn_scales
    assert denoiser_calls[2]["cross_qo_fp8_scales"] is None

    cross_qo_scales = {
        name: {"input_scale": 0.25}
        for index in range(WAN22_TI2V_5B.num_layers)
        for name in (
            f"blocks.{index}.attn2.to_q",
            f"blocks.{index}.attn2.to_out.0",
        )
    }
    combined_scales = {**ffn_scales, **cross_qo_scales}
    trt_builder.build_wan22_components(
        str(tmp_path),
        config=_runtime_config(),
        weights=weights,
        fp8_scales=combined_scales,
    )
    assert denoiser_calls[3]["ffn_fp8_scales"] == ffn_scales
    assert denoiser_calls[3]["cross_qo_fp8_scales"] == cross_qo_scales

    with pytest.raises(ValueError, match="complete FFN\\+cross-Q/O"):
        trt_builder.build_wan22_components(
            str(tmp_path),
            config=_runtime_config(),
            weights=weights,
            fp8_scales=cross_qo_scales,
        )

    incomplete_scales = dict(combined_scales)
    incomplete_scales.pop("blocks.0.attn2.to_q")
    with pytest.raises(ValueError, match="missing_cross_qo="):
        trt_builder.build_wan22_components(
            str(tmp_path),
            config=_runtime_config(),
            weights=weights,
            fp8_scales=incomplete_scales,
        )

    unqualified_scales = dict(combined_scales)
    unqualified_scales["blocks.0.attn1.to_out.0"] = {"input_scale": 0.25}
    with pytest.raises(ValueError, match="unexpected="):
        trt_builder.build_wan22_components(
            str(tmp_path),
            config=_runtime_config(),
            weights=weights,
            fp8_scales=unqualified_scales,
        )

    with pytest.raises(ValueError, match="FP8 scales are qualified only"):
        trt_builder.build_wan22_components(
            str(tmp_path),
            config=_runtime_config(
                video_width=672,
                video_height=384,
                video_num_frames=5,
                num_inference_steps=15,
            ),
            weights=weights,
            fp8_scales=ffn_scales,
        )


def test_runtime_config_supports_only_qualified_profiles_and_seed_range() -> None:
    plugin = Wan22TI2VPlugin()
    official = plugin.get_diffusion_config(_runtime_config())
    assert plugin.runtime_strategy == "diffusion_wan2_2_ti2v"
    assert set(official) == {
        "negative_prompt",
        "num_inference_steps",
        "guidance_scale",
        "flow_shift",
        "seed",
        "video_height",
        "video_width",
        "video_num_frames",
        "frame_rate",
        "text_seq_len",
    }
    assert official["video_num_frames"] == 121
    assert official["video_height"] == 704
    assert official["video_width"] == 1280
    assert official["num_inference_steps"] == 50
    assert official["guidance_scale"] == 5.0
    assert official["text_seq_len"] == 512

    l0 = plugin.get_diffusion_config(
        _runtime_config(
            video_width=672,
            video_height=384,
            video_num_frames=5,
            num_inference_steps=15,
        )
    )
    assert (l0["video_width"], l0["video_height"], l0["video_num_frames"]) == (
        672,
        384,
        5,
    )

    with pytest.raises(ValueError, match="exact qualified generation profile"):
        plugin.get_diffusion_config(_runtime_config(video_num_frames=81))
    for seed in (-1, 2_147_483_648):
        with pytest.raises(ValueError, match="bundle seed"):
            plugin.get_diffusion_config(_runtime_config(seed=seed))
    for seed in (0, 42, 2_147_483_647):
        assert plugin.get_diffusion_config(_runtime_config(seed=seed))["seed"] == seed


def test_runtime_fp_contract_compile_option_is_model_manifest_owned() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    pipeline_source = (repo_root / "src/runtime/models/wan2_2_ti2v/pipeline.cpp").read_text()
    top_level_cmake = (repo_root / "CMakeLists.txt").read_text()

    assert '#pragma GCC optimize("fp-contract=off")' in pipeline_source
    assert "wan2_2_ti2v/pipeline.cpp" not in top_level_cmake


def test_component_builder_rejects_unsupported_precision_before_checkpoint_access() -> None:
    from tensorrt_model_connect.families.wan2_2_ti2v import trt_builder

    with pytest.raises(ValueError, match="requires BF16"):
        trt_builder.build_wan22_components(
            "unused-checkpoint",
            config=_runtime_config(),
            weights={},
            precision="fp32",
        )
