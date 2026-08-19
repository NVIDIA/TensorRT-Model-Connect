# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Wan T2V family plugin and preprocessor serialization.

Trace: ARCH-FAM-001, UD-FAM-WAN-T2V
Intent: Validate Wan T2V diffusion family plugin matching, weight serialization, and video config encoding
Preconditions: Synthetic Wan T2V model config with video dimensions and weight tensors are available
Postconditions: Plugin matches Wan aliases, serializes preprocessor weights correctly, and encodes video parameters
"""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import types

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    from tensorrt_model_connect.config import ModelConfig
    from tensorrt_model_connect.models.wan_t2v import model as wan_mod
    from tensorrt_model_connect.models.wan_t2v import diffusion_runner as py_diffusion_runner
    from tensorrt_model_connect.models.wan_t2v.tests.e2e_plugins.contracts import (
        ensure_initial_latents,
        normalize_wan_prompt,
    )
    from tensorrt_model_connect.models.wan_t2v.tests.e2e_plugins.references import hf_diffusers
    from tensorrt_model_connect.models.wan_t2v.tests.e2e_plugins.runners import diffusion
    from tests.e2e_harness.contracts import E2ECase, RunContext, StageSpec
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


def _cfg(**raw_overrides: object) -> ModelConfig:
    payload = {
        "model_type": "wan_t2v",
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "video_height": 64,
        "video_width": 80,
        "video_num_frames": 9,
    }
    payload.update(raw_overrides)
    return ModelConfig.from_json(json.dumps(payload))


def _decode_blob(blob: bytes) -> tuple[dict[str, dict], bytes]:
    idx_len = struct.unpack("<I", blob[:4])[0]
    index = json.loads(blob[4:4 + idx_len].decode("utf-8"))
    payload = blob[4 + idx_len:]
    return index, payload


def _module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def test_matches_declared_wan_model_types() -> None:
    """Intent: validate model-type routing.

    Preconditions: the family model module is imported.
    Postconditions: supported aliases match and a sibling family does not.
    """
    assert wan_mod.matches("wan_t2v")
    assert wan_mod.matches("wan")
    assert wan_mod.matches("Wan2.1")
    assert not wan_mod.matches("flux")


def test_wan_pipeline_classes_resolve_to_wan_plugin() -> None:
    """Wan owns the real Diffusers pipeline class mapping for Wan models."""
    from tensorrt_model_connect.config import ModelConfig as BuildModelConfig

    for pipeline_class in ("WanPipeline", "WanVideoToVideoPipeline"):
        config = BuildModelConfig(
            model_type="wan_t2v", raw={"_class_name": pipeline_class}
        )
        assert wan_mod.matches(config)


def test_load_weights_requires_diffusers_model_index(tmp_path) -> None:
    """Intent: cover both diffusers-detection branches in load_weights.

    Preconditions: one temp directory contains model_index.json and another does not.
    Postconditions: success path returns expected subdir keys; failure path raises ValueError.
    """
    model_dir = tmp_path / "wan"
    model_dir.mkdir()
    (model_dir / "model_index.json").write_text("{}")

    weights = wan_mod.load_weights(str(model_dir), _cfg())
    assert weights["_model_format"] == "diffusers"
    assert weights["_text_encoder_dir"].endswith("text_encoder")
    assert weights["_transformer_dir"].endswith("transformer")
    assert weights["_vae_dir"].endswith("vae")

    bad_dir = tmp_path / "wan_bad"
    bad_dir.mkdir()
    with pytest.raises(ValueError, match="Expected diffusers format"):
        wan_mod.load_weights(str(bad_dir), _cfg())


def test_load_weights_preserves_checkpoint_scheduler_config(tmp_path) -> None:
    """The bundle scheduler must inherit the checkpoint's Diffusers config."""
    model_dir = tmp_path / "wan"
    scheduler_dir = model_dir / "scheduler"
    scheduler_dir.mkdir(parents=True)
    (model_dir / "model_index.json").write_text("{}")
    (scheduler_dir / "scheduler_config.json").write_text(
        json.dumps(
            {
                "_class_name": "UniPCMultistepScheduler",
                "num_train_timesteps": 1000,
                "flow_shift": 3.0,
                "solver_order": 2,
                "solver_type": "bh2",
                "prediction_type": "flow_prediction",
                "use_flow_sigmas": True,
                "lower_order_final": True,
                "use_dynamic_shifting": False,
            }
        )
    )
    config = _cfg()

    wan_mod.load_weights(str(model_dir), config)
    diffusion = wan_mod.get_diffusion_config(config)

    assert diffusion["scheduler"] == "unipc_multistep"
    assert diffusion["flow_shift"] == pytest.approx(3.0)
    assert diffusion["unipc_lower_order_final"] == 1
    assert diffusion["use_dynamic_shifting"] == 0


def test_wan_scheduler_fallback_is_flow_match_euler() -> None:
    """A config without scheduler metadata keeps the generic Wan fallback."""
    diffusion = wan_mod.get_diffusion_config(_cfg())

    assert diffusion["scheduler"] == "flow_match_euler"
    assert diffusion["flow_shift"] == pytest.approx(1.0)


def test_wan_rejects_unsupported_unipc_variant() -> None:
    config = _cfg(
        _scheduler_config={
            "_class_name": "UniPCMultistepScheduler",
            "solver_order": 3,
        }
    )

    with pytest.raises(ValueError, match="order-2 BH2 UniPC"):
        wan_mod.get_diffusion_config(config)


def test_wan_runtime_owns_t5_special_token_framing(tmp_path) -> None:
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()

    assert wan_mod.diffusion_tokenizer_add_special_tokens(
        tmp_path,
        detect_tokenizer_add_special_tokens=lambda _path: True,
    ) is False


def test_build_components_calls_all_subbuilders(monkeypatch: pytest.MonkeyPatch) -> None:
    """Intent: verify build_components orchestration and computed num_patches.

    Preconditions: all imported builder modules are monkeypatched with deterministic stubs.
    Postconditions: each sub-builder receives expected arguments and return dict shape is correct.
    """
    calls: dict[str, object] = {}

    def load_t5_weights(path, **kwargs):
        calls["load_t5_weights"] = {"path": path, **kwargs}
        return {"t5.weight": np.array([1], dtype=np.float32)}

    def build_t5_encoder_engine(weights, **kwargs):
        calls["build_t5_encoder_engine"] = {"weights": weights, **kwargs}
        return b"t5-plan"

    def load_dit_weights(path, **kwargs):
        calls["load_dit_weights"] = {"path": path, **kwargs}
        return {"dit.weight": np.array([2], dtype=np.float32)}

    def build_standard_dit_engine(weights, **kwargs):
        calls["build_standard_dit_engine"] = {"weights": weights, **kwargs}
        return b"dit-plan"

    def load_vae_weights(path, **kwargs):
        calls["load_vae_weights"] = {"path": path, **kwargs}
        return {"vae.weight": np.array([3], dtype=np.float32)}

    def build_causal_vae_3d_engine(weights, **kwargs):
        call = {"weights": weights, **kwargs}
        calls.setdefault("build_causal_vae_3d_engine", []).append(call)
        if kwargs.get("first_frame_only"):
            return b"vae-first-frame-plan"
        return b"vae-plan"

    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.models.wan_t2v.t5_encoder_builder",
        _module(
            "tensorrt_model_connect.models.wan_t2v.t5_encoder_builder",
            load_t5_weights=load_t5_weights,
            build_t5_encoder_engine=build_t5_encoder_engine,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.models.wan_t2v.standard_dit_builder",
        _module(
            "tensorrt_model_connect.models.wan_t2v.standard_dit_builder",
            load_dit_weights=load_dit_weights,
            build_standard_dit_engine=build_standard_dit_engine,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.models.wan_t2v.causal_vae_3d_builder",
        _module(
            "tensorrt_model_connect.models.wan_t2v.causal_vae_3d_builder",
            load_vae_weights=load_vae_weights,
            build_causal_vae_3d_engine=build_causal_vae_3d_engine,
            count_vae_caches=lambda **_kwargs: 0,
        ),
    )

    monkeypatch.setattr(
        wan_mod,
        "_serialize_preprocessor_weights",
        lambda dit_weights: b"wan-preproc",
    )

    cfg = _cfg(
        video_height=64,
        video_width=80,
        video_num_frames=9,
        _fp32_layers=[24],
    )
    weights = {
        "_text_encoder_dir": "/model/text_encoder",
        "_transformer_dir": "/model/transformer",
        "_vae_dir": "/model/vae",
    }

    out = wan_mod.build_components(
        "/model", cfg, weights, precision="fp16", verbose=True)

    assert out["text_encoders"] == [("t5", b"t5-plan")]
    assert out["denoiser"] == b"dit-plan"
    assert out["vae_decoder"] == b"vae-plan"
    assert out["vae_decoder_first_frame"] == b"vae-first-frame-plan"
    assert out["preprocessor_weights"] == b"wan-preproc"

    # Preconditions ensure 64x80 and 9 frames.
    # Postcondition: num_patches = 60 using Wan's latent+patching math.
    assert calls["load_t5_weights"]["precision"] == "fp32"
    assert calls["build_t5_encoder_engine"]["precision"] == "fp32"
    assert calls["build_standard_dit_engine"]["num_patches"] == 60
    assert calls["build_standard_dit_engine"]["context_dim"] == wan_mod._DIT_DIM
    assert calls["build_standard_dit_engine"]["precision"] == "fp16"
    vae_calls = calls["build_causal_vae_3d_engine"]
    assert len(vae_calls) == 2
    assert vae_calls[0]["precision"] == "fp16"
    assert vae_calls[0].get("first_frame_only") is None
    assert vae_calls[1]["precision"] == "fp16"
    assert vae_calls[1]["first_frame_only"] is True


def test_build_components_rejects_partial_t5_fp32_selectors() -> None:
    """Wan currently supports the complete-T5 selector, not partial layers."""
    weights = {
        "_text_encoder_dir": "/model/text_encoder",
        "_transformer_dir": "/model/transformer",
        "_vae_dir": "/model/vae",
    }

    with pytest.raises(ValueError, match="supports only selector 24"):
        wan_mod.build_components(
            "/model",
            _cfg(_fp32_layers=[0, 23]),
            weights,
            precision="fp16",
        )


def test_build_components_tensor_parallel_builds_rank_denoisers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Intent: verify Wan TP packaging keeps T5/VAE single-copy and builds rank DiTs.

    Preconditions: builder modules are monkeypatched and TensorRT version is 11.0.
    Postconditions: denoiser_ranks contains one rank-local plan per requested TP rank.
    """
    from tensorrt_model_connect import trt_compat
    from tensorrt_model_connect.parallel_config import ParallelConfig

    calls: dict[str, object] = {"dit_ranks": []}

    monkeypatch.setattr(trt_compat, "tensorrt_version", lambda: "11.0.0")

    def load_t5_weights(path, **kwargs):
        calls["load_t5_weights"] = {"path": path, **kwargs}
        return {"t5.weight": np.array([1], dtype=np.float32)}

    def build_t5_encoder_engine(weights, **kwargs):
        calls["build_t5_encoder_engine"] = {"weights": weights, **kwargs}
        return b"t5-plan"

    def load_dit_weights(path, **kwargs):
        calls["load_dit_weights"] = {"path": path, **kwargs}
        return {"dit.weight": np.array([2], dtype=np.float32)}

    def build_standard_dit_engine(_weights, **_kwargs):
        raise AssertionError("single-device Wan DiT builder used for TP build")

    def build_standard_dit_tp_engine(weights, **kwargs):
        parallel = kwargs["parallel_config"]
        calls["dit_ranks"].append(parallel.rank)
        return f"dit-rank-{parallel.rank}".encode()

    def load_vae_weights(path, **kwargs):
        calls["load_vae_weights"] = {"path": path, **kwargs}
        return {"vae.weight": np.array([3], dtype=np.float32)}

    def build_causal_vae_3d_engine(weights, **kwargs):
        call = {"weights": weights, **kwargs}
        calls.setdefault("build_causal_vae_3d_engine", []).append(call)
        if kwargs.get("first_frame_only"):
            return b"vae-first-frame-plan"
        return b"vae-plan"

    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.models.wan_t2v.t5_encoder_builder",
        _module(
            "tensorrt_model_connect.models.wan_t2v.t5_encoder_builder",
            load_t5_weights=load_t5_weights,
            build_t5_encoder_engine=build_t5_encoder_engine,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.models.wan_t2v.standard_dit_builder",
        _module(
            "tensorrt_model_connect.models.wan_t2v.standard_dit_builder",
            load_dit_weights=load_dit_weights,
            build_standard_dit_engine=build_standard_dit_engine,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.models.wan_t2v.standard_dit_tp_builder",
        _module(
            "tensorrt_model_connect.models.wan_t2v.standard_dit_tp_builder",
            build_standard_dit_engine=build_standard_dit_tp_engine,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.models.wan_t2v.causal_vae_3d_builder",
        _module(
            "tensorrt_model_connect.models.wan_t2v.causal_vae_3d_builder",
            load_vae_weights=load_vae_weights,
            build_causal_vae_3d_engine=build_causal_vae_3d_engine,
            count_vae_caches=lambda **_kwargs: 0,
        ),
    )
    monkeypatch.setattr(
        wan_mod,
        "_serialize_preprocessor_weights",
        lambda dit_weights: b"wan-preproc",
    )

    weights = {
        "_text_encoder_dir": "/model/text_encoder",
        "_transformer_dir": "/model/transformer",
        "_vae_dir": "/model/vae",
    }

    out = wan_mod.build_components(
        "/model",
        _cfg(video_height=64, video_width=80, video_num_frames=9),
        weights,
        parallel_config=ParallelConfig(mode="tensor_parallel", tp_size=4),
    )

    assert out["text_encoders"] == [("t5", b"t5-plan")]
    assert out["denoiser_ranks"] == {
        0: b"dit-rank-0",
        1: b"dit-rank-1",
        2: b"dit-rank-2",
        3: b"dit-rank-3",
    }
    assert "denoiser" not in out
    assert out["vae_decoder"] == b"vae-plan"
    assert out["vae_decoder_first_frame"] == b"vae-first-frame-plan"
    assert [call.get("first_frame_only", False)
            for call in calls["build_causal_vae_3d_engine"]] == [False, True]
    assert calls["dit_ranks"] == [0, 1, 2, 3]


def test_context_parallel_bundle_packages_one_shared_denoiser() -> None:
    """Wan CP ranks load one rank-dynamic denoiser plan and shared auxiliaries."""
    from tensorrt_model_connect.parallel_config import ParallelConfig

    sections = dict(wan_mod.diffusion_bundle_sections(
        {
            "text_encoders": [("t5", b"t5-plan")],
            "denoiser": b"dit-cp-plan",
            "vae_decoder": b"vae-plan",
            "vae_decoder_first_frame": b"vae-first-frame-plan",
            "preprocessor_weights": b"wan-preproc",
        },
        parallel_config=ParallelConfig(
            mode="context_parallel", cp_size=4),
    ))

    assert sections["denoiser_plan_cp"] == b"dit-cp-plan"
    assert "denoiser_plan" not in sections
    assert not any(name.startswith("denoiser_plan_tp_rank") for name in sections)
    assert sections["text_encoder_0_plan"] == b"t5-plan"
    assert sections["vae_decoder_plan"] == b"vae-plan"


def test_get_diffusion_config_uses_count_vae_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Intent: verify diffusion config wiring and imported cache-count helper call.

    Preconditions: count_vae_caches is monkeypatched to a known constant.
    Postconditions: returned config includes custom image/video dimensions and mocked cache count.
    """
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.models.wan_t2v.causal_vae_3d_builder",
        _module(
            "tensorrt_model_connect.models.wan_t2v.causal_vae_3d_builder",
            count_vae_caches=lambda **_kwargs: 13,
        ),
    )

    cfg = _cfg(video_height=96, video_width=160, video_num_frames=13)
    dc = wan_mod.get_diffusion_config(cfg)

    assert dc["video_height"] == 96
    assert dc["video_width"] == 160
    assert dc["video_num_frames"] == 13
    assert dc["num_vae_caches"] == 13
    assert dc["diffusion_backend_type"] == "wan_3d"


def test_serialize_preprocessor_weights_transforms_patch_weight() -> None:
    """Intent: validate binary serialization, key filtering, and Conv3D flatten+transpose.

    Preconditions: dit_weights includes a Conv3D patch embedding and a subset of listed keys.
    Postconditions: output index maps stored keys with correct shapes and contiguous payload size.
    """
    dit_weights = {
        "patch_embedding.weight": np.arange(24, dtype=np.float32).reshape(2, 3, 1, 2, 2),
        "patch_embedding.bias": np.array([1.0, 2.0], dtype=np.float32),
        "condition_embedder.time_embedding.0.weight": np.arange(12, dtype=np.float32).reshape(3, 4),
        "condition_embedder.text_embedding_2.bias": np.array([9.0], dtype=np.float32),
    }

    blob = wan_mod._serialize_preprocessor_weights(dit_weights)
    index, payload = _decode_blob(blob)

    assert "patch_embedding.weight" in index
    assert index["patch_embedding.weight"]["shape"] == [12, 2]
    assert "condition_embedder.time_embedding.2.weight" not in index

    max_end = 0
    for info in index.values():
        nbytes = int(np.prod(info["shape"])) * 4
        max_end = max(max_end, info["offset"] + nbytes)
    assert len(payload) == max_end


def _parity_case(seed: int = 42) -> E2ECase:
    return E2ECase(
        name="vbench_000001",
        hf_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        family="wan_t2v",
        runtime_strategy="diffusion_wan",
        bundle="wan21-t2v-1.3b-l0.bundle",
        inputs={
            "prompt": "A red robot walks through a garden",
            "video_num_frames": 5,
            "video_height": 384,
            "video_width": 672,
            "num_inference_steps": 1,
            "seed": seed,
            "use_shared_initial_latents": True,
        },
    )


def test_hf_and_trtmc_resolve_the_same_initial_latent(tmp_path) -> None:
    case = _parity_case(seed=43)
    hf_ctx = RunContext(case=case, artifacts_dir=str(tmp_path / "hf_artifacts"))
    trt_ctx = RunContext(case=case, artifacts_dir=str(tmp_path / "bundle_artifacts"))

    hf = ensure_initial_latents(case, hf_ctx)
    trt = ensure_initial_latents(case, trt_ctx)

    assert hf.path == trt.path
    assert hf.sha256 == trt.sha256
    assert hf.shape == (1, 16, 2, 48, 84)
    assert hf.path.stat().st_size == 4 * 16 * 2 * 48 * 84


def test_wan_prompt_normalization_matches_diffusers_cleaning() -> None:
    assert normalize_wan_prompt("  A&amp;B\u00a0\n  moves  ") == "A&B moves"


def test_python_t5_diagnostic_uses_additive_attention_mask() -> None:
    input_ids = np.asarray([[289, 3735, 1, 0, 0]], dtype=np.int32)

    mask = py_diffusion_runner._build_t5_attention_mask(input_ids, np.float32)

    np.testing.assert_array_equal(
        mask,
        np.asarray([[0.0, 0.0, 0.0, -1.0e9, -1.0e9]], dtype=np.float32),
    )


def test_trtmc_runner_consumes_and_reports_shared_initial_latent(
    tmp_path, monkeypatch
) -> None:
    case = _parity_case(seed=44)
    binary = tmp_path / "trtmc"
    binary.write_text("", encoding="utf-8")
    ctx = RunContext(
        case=case,
        artifacts_dir=str(tmp_path / "bundle_artifacts"),
        binary_path=str(binary),
        engine_dir=str(tmp_path),
    )
    monkeypatch.setattr(
        diffusion.subprocess,
        "run",
        lambda cmd, **_kwargs: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
    )

    output = diffusion.DiffusionMediaRunner().run_stage(
        case, StageSpec(name="end_to_end"), ctx
    )

    command = output.metadata["command"]
    latent_path = command[command.index("--initial-latents-raw") + 1]
    assert "shared_initial_latents" in latent_path
    assert output.data["initial_latents_sha256"]


def test_trtmc_runner_normalizes_prompt_before_tokenization(
    tmp_path, monkeypatch
) -> None:
    case = _parity_case()
    case.inputs["prompt"] = "  A&amp;B\u00a0\n  moves  "
    ctx = RunContext(
        case=case,
        artifacts_dir=str(tmp_path / "bundle_artifacts"),
        binary_path=str(tmp_path / "trtmc"),
        engine_dir=str(tmp_path),
    )
    monkeypatch.setattr(
        diffusion.subprocess,
        "run",
        lambda cmd, **_kwargs: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
    )

    output = diffusion.DiffusionMediaRunner().run_stage(
        case, StageSpec(name="end_to_end"), ctx
    )

    command = output.metadata["command"]
    assert command[command.index("--prompt") + 1] == "A&B moves"
    assert output.data["prompt"] == case.inputs["prompt"]


def test_hf_reference_consumes_and_reports_shared_initial_latent(
    tmp_path, monkeypatch
) -> None:
    case = _parity_case(seed=44)
    ctx = RunContext(
        case=case,
        artifacts_dir=str(tmp_path / "hf_artifacts"),
        reference_python="/opt/venv/bin/python",
    )
    captured: dict[str, object] = {}

    def fake_run(cmd, **_kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(hf_diffusers, "_resolve_cached_model_ref", lambda _id: "/model")
    monkeypatch.setattr(hf_diffusers.subprocess, "run", fake_run)

    output = hf_diffusers.HfDiffusersReference().run_stage(
        case, StageSpec(name="end_to_end"), ctx
    )

    script = captured["cmd"][2]
    assert "latents=initial_latents" in script
    assert "shared_initial_latents" in script
    assert "max_sequence_length=226" in script
    assert "prompt='A red robot walks through a garden'" in script
    assert output.data["initial_latents_sha256"]


def test_hf_full_pipeline_leaves_prompt_cleaning_to_diffusers(
    tmp_path, monkeypatch
) -> None:
    case = _parity_case()
    case.inputs["prompt"] = "A&amp;amp;amp;B"
    ctx = RunContext(
        case=case,
        artifacts_dir=str(tmp_path / "hf_artifacts"),
        reference_python="/opt/venv/bin/python",
    )
    captured: dict[str, object] = {}

    def fake_run(cmd, **_kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(hf_diffusers, "_resolve_cached_model_ref", lambda _id: "/model")
    monkeypatch.setattr(
        hf_diffusers,
        "normalize_wan_prompt",
        lambda _prompt: (_ for _ in ()).throw(
            AssertionError("full WanPipeline reference must clean its own prompt")
        ),
    )
    monkeypatch.setattr(hf_diffusers.subprocess, "run", fake_run)

    hf_diffusers.HfDiffusersReference().run_stage(
        case, StageSpec(name="end_to_end"), ctx
    )

    script = captured["cmd"][2]
    assert "prompt='A&amp;amp;amp;B'" in script
