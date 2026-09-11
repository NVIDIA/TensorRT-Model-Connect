# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
import struct

import pytest

from tensorrt_model_connect.bundle_writer import BUNDLE_MAGIC, BundleWriter

from families.minimax_h3 import staged_build


def _read_bundle(path: Path) -> tuple[dict, dict[str, bytes]]:
    with path.open("rb") as stream:
        assert stream.read(8) == BUNDLE_MAGIC
        header_size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_size))
        payload_start = stream.tell()
        sections = {}
        for name, record in header["sections"].items():
            stream.seek(payload_start + record["offset"])
            sections[name] = stream.read(record["length"])
    return header, sections


def _model(tmp_path: Path) -> Path:
    model = tmp_path / "model"
    for name in ("transformer", "text_encoder", "vae", "audio_vae", "tokenizer"):
        (model / name).mkdir(parents=True, exist_ok=True)
    (model / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model / "audio_vae" / "config.json").write_text(
        json.dumps(
            {
                "decoder_rates": [5, 5, 2, 2, 2, 2, 2],
                "sampling_rate": 32_000,
                "latents_mean": [0.0] * 32,
                "latents_std": [1.0] * 32,
            }
        ),
        encoding="utf-8",
    )
    return model


def _writer(output: Path) -> BundleWriter:
    writer = BundleWriter(output)
    writer.set_header(family="minimax_h3", task="image_generation", backend="trt_rtx")
    return writer


def _quantized_sources(tmp_path: Path, monkeypatch) -> dict[str, Path]:
    from families.minimax_h3 import nvfp4_text_checkpoint, quantized_checkpoint
    from families.minimax_h3.delivery import QUANTIZED_SOURCES

    sources = {}
    for option, filename in QUANTIZED_SOURCES.items():
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(option.encode())
        sources[option] = path

    def validate_denoiser(path, *, workflow="fl2va"):
        identity = (
            quantized_checkpoint.QUANTIZED_REF2VA_CHECKPOINT_IDENTITY
            if workflow == "ref2va"
            else quantized_checkpoint.QUANTIZED_CHECKPOINT_IDENTITY
        )
        assert path.name == Path(identity.filename).name
        return replace(
            identity, source_file_identity=quantized_checkpoint._source_file_identity(path)
        )

    monkeypatch.setattr(
        quantized_checkpoint, "validate_quantized_transformer_checkpoint", validate_denoiser
    )
    monkeypatch.setattr(
        nvfp4_text_checkpoint,
        "validate_quantized_text_checkpoint",
        lambda path: replace(
            nvfp4_text_checkpoint.QUANTIZED_TEXT_CHECKPOINT_IDENTITY,
            source_file_identity=quantized_checkpoint._source_file_identity(path),
        ),
    )
    return sources


def test_public_quantized_build_needs_no_original_text_or_denoiser_weights(
    tmp_path: Path, monkeypatch
) -> None:
    from tensorrt_model_connect import BuildRequest
    from families.minimax_h3 import checkpoint
    import huggingface_hub

    core = importlib.import_module("tensorrt_model_connect.build")
    model_dir = tmp_path / "model"
    for name in ("transformer", "vae", "audio_vae", "tokenizer"):
        (model_dir / name).mkdir(parents=True)
    (model_dir / "model_index.json").write_text(
        json.dumps({"_class_name": "MiniMaxH3Pipeline"}), encoding="utf-8"
    )
    (model_dir / "transformer/config.json").write_text(
        json.dumps({"hidden_size": 5376, "num_layers": 50, "num_attention_heads": 56,
                    "attention_head_dim": 128, "ffn_dim": 14336}), encoding="utf-8"
    )
    (model_dir / "tokenizer/tokenizer.json").write_text("{}", encoding="utf-8")
    (model_dir / "audio_vae/config.json").write_text(
        json.dumps({"decoder_rates": [5, 5, 2, 2, 2, 2, 2], "sampling_rate": 32000,
                    "latents_mean": [0.0] * 32, "latents_std": [1.0] * 32}), encoding="utf-8"
    )
    for name in ("vae", "audio_vae"):
        (model_dir / name / "model.safetensors").write_bytes(b"auxiliary weight fixture")
    sources = _quantized_sources(model_dir, monkeypatch)
    calls = []

    def build_component(component, source_dir, plan, **options):
        assert source_dir == model_dir
        assert options["transformer_ref_path"] is None
        assert options["quantized_ref_transformer_path"] == sources["quantized_ref_transformer"]
        if component in staged_build._QUANTIZED_TRANSFORMER_COMPONENTS:
            assert options["quantized_transformer_path"] == sources["quantized_transformer"]
        if component in {"text_encoder", "vision_encoder"}:
            assert options["quantized_text_encoder_path"] == sources["quantized_text_encoder"]
        calls.append(component)
        plan.write_bytes(component.encode())

    monkeypatch.setattr(
        checkpoint, "load_selected_component_state_dict",
        lambda *_args, **_kwargs: pytest.fail("Unexpected original checkpoint fallback"),
    )
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download",
        lambda *_args, **_kwargs: pytest.fail("Local quantized fixtures must not download"),
    )
    monkeypatch.setattr(core, "_select_backend", lambda _backend: None)
    monkeypatch.setattr(staged_build, "_run_component", build_component)
    monkeypatch.setattr(staged_build.trt_compat, "tensorrt_version", lambda: "1.6.1")
    monkeypatch.setattr(staged_build.trt_compat, "tensorrt_abi", lambda _version: "1.6")
    output = tmp_path / "clean.bundle"
    core.build(BuildRequest(model_dir=model_dir, output_path=output, family="minimax_h3",
                            task="image_generation", precision="bf16", backend="trt_rtx"))

    assert not (model_dir / "text_encoder").exists()
    assert not (model_dir / "transformer_ref").exists()
    assert not list((model_dir / "transformer").glob("*.safetensors*"))
    assert len(calls) == 15
    _header, sections = _read_bundle(output)
    assert json.loads(sections["runtime.json"])["public_workflows"] == ["t2va", "fl2va", "ref2va"]
    state = json.loads(output.with_name(output.name + ".plans").joinpath("build_state.json").read_text())
    assert set(state["checkpoint"]["components"]) == {"vae", "audio_vae"}


@pytest.mark.parametrize("sr", (False, True))
def test_quantized_delivery_publishes_all_workflows_and_forwards_distinct_sources(
    tmp_path: Path, monkeypatch, sr: bool
) -> None:
    from families.minimax_h3 import nvfp4_text_checkpoint, provenance, quantized_checkpoint

    model = _model(tmp_path)
    sources = _quantized_sources(tmp_path / "comfy", monkeypatch)
    options = dict(sources)
    if sr:
        primary = tmp_path / provenance.SUPER_RESOLUTION_PRIMARY_FILENAME
        weak = tmp_path / provenance.SUPER_RESOLUTION_WEAK_FILENAME
        primary.write_bytes(b"primary")
        weak.write_bytes(b"weak")
        options.update(super_resolution_model=primary, super_resolution_weak_model=weak)
    calls = {}

    def build(component, _model, plan, **kwargs):
        calls[component] = kwargs
        plan.write_bytes(component.encode())

    monkeypatch.setattr(staged_build, "_run_component", build)
    monkeypatch.setattr(staged_build.trt_compat, "tensorrt_version", lambda: "1.6.1")
    monkeypatch.setattr(staged_build.trt_compat, "tensorrt_abi", lambda _version: "1.6")
    output = tmp_path / "all-workflows.bundle"
    writer = _writer(output)
    plans = tmp_path / "plans"
    # The same low-resolution source does not itself enable SR.
    staged_build.build_staged_bundle(
        model,
        writer,
        plans_dir=plans,
        runtime_defaults={"height": 480, "width": 864},
        **options,
    )
    writer.finish()
    _header, sections = _read_bundle(output)
    runtime = json.loads(sections["runtime.json"])
    assert runtime["public_workflows"] == ["t2va", "fl2va", "ref2va"]
    assert runtime["quantized_transformer"] == (
        quantized_checkpoint.QUANTIZED_CHECKPOINT_IDENTITY.bundle_metadata()
    )
    assert runtime["ref2va_transformer_ref"] == (
        quantized_checkpoint.QUANTIZED_REF2VA_CHECKPOINT_IDENTITY.bundle_metadata()
    )
    assert runtime["quantized_text_encoder"] == (
        nvfp4_text_checkpoint.QUANTIZED_TEXT_CHECKPOINT_IDENTITY.bundle_metadata()
    )
    assert runtime["quantized_text_encoder"]["quantization"] == "nvfp4_awq"
    assert runtime["quantized_text_encoder"]["engine_weights_dtype"] == "bfloat16"
    assert runtime["quantized_text_encoder"]["full_precision_matrix_mult"] is True
    assert runtime["denoiser_profile_count"] == 1
    assert runtime["denoiser_profile_layout"] == "public_dynamic"
    assert runtime["video_rows_min"] == 14_985
    assert runtime["packed_sequence_length_min"] == 15_400
    assert runtime["conditioning"]["text_sequence_profile"] == [1, 1144, 262144]
    assert ("video_super_resolution_plan" in sections) is sr
    assert ("super_resolution" in runtime) is sr
    for component, forwarded in calls.items():
        assert forwarded["transformer_ref_path"] is None
        assert forwarded["quantized_ref_transformer_path"] == sources["quantized_ref_transformer"]
        assert ("quantized_transformer_path" in forwarded) is (
            component in staged_build._QUANTIZED_TRANSFORMER_COMPONENTS
        )
        if "quantized_transformer_path" in forwarded:
            assert forwarded["quantized_transformer_path"] == sources["quantized_transformer"]
        assert ("quantized_text_encoder_path" in forwarded) is (
            component in {"text_encoder", "vision_encoder"}
        )
        if "quantized_text_encoder_path" in forwarded:
            assert forwarded["quantized_text_encoder_path"] == sources["quantized_text_encoder"]
    state = json.loads((plans / "build_state.json").read_text())
    assert set(state["checkpoint"]["components"]) == {"vae", "audio_vae"}
    assert state["ref2va"]["filename"] == runtime["ref2va_transformer_ref"]["filename"]
    assert set(state["ref2va"]["source_files"]) == {sources["quantized_ref_transformer"].name}
    assert "source_file_identity" in state["quantized_text_encoder"]
    assert len(state["denoiser_profiles"]) == 1
    assert str(tmp_path) not in json.dumps(runtime)


def test_staged_build_publishes_current_sections_and_runtime_json(
    tmp_path: Path, monkeypatch
) -> None:
    model = _model(tmp_path)
    output = tmp_path / "h3.bundle"
    plans = tmp_path / "plans"
    calls: list[str] = []

    def build(component: str, _model: Path, plan: Path, **_options):
        calls.append(component)
        plan.write_bytes(component.encode())
        return {"bytes": plan.stat().st_size}

    monkeypatch.setattr(staged_build, "_run_component", build)
    monkeypatch.setattr(staged_build.trt_compat, "tensorrt_version", lambda: "1.6.1")
    monkeypatch.setattr(staged_build.trt_compat, "tensorrt_abi", lambda _version: "1.6")

    writer = _writer(output)
    staged_build.build_staged_bundle(
        model,
        writer,
        plans_dir=plans,
        runtime_defaults={
            "height": 480,
            "width": 864,
            "num_frames": 345,
            "first_block_cache_threshold": 0.12,
        },
    )
    writer.finish()

    header, sections = _read_bundle(output)
    expected = {section for _component, _filename, section in staged_build._COMPONENTS}
    assert set(sections) == {*expected, "tokenizer.json", "runtime.json"}
    assert header["family"] == "minimax_h3"
    runtime = json.loads(sections["runtime.json"])
    assert runtime["public_workflows"] == ["t2va", "fl2va"]
    assert (runtime["height"], runtime["width"], runtime["num_frames"]) == (480, 864, 345)
    assert runtime["first_block_cache_threshold"] == 0.12
    assert runtime["denoiser_profile_count"] == 1
    assert runtime["denoiser_profile_layout"] == "public_dynamic"
    assert calls == [component for component, _filename, _section in staged_build._COMPONENTS]

    monkeypatch.setattr(
        staged_build,
        "_run_component",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("plan rebuilt")),
    )
    second = tmp_path / "h3-second.bundle"
    writer = _writer(second)
    staged_build.build_staged_bundle(model, writer, plans_dir=plans)
    writer.finish()
    assert second.is_file()


def test_staged_component_contract_includes_ref_and_super_resolution_sections() -> None:
    components = (
        *staged_build._COMPONENTS,
        *staged_build._REF2VA_COMPONENTS,
        staged_build._SUPER_RESOLUTION_COMPONENT,
    )
    sections = [section for _component, _filename, section in components]
    assert sections[:7] == [
        "text_encoder_plan",
        "vision_encoder_plan",
        "adaln_precompute_plan",
        "denoiser_head_plan",
        "denoiser_tail_plan",
        "denoiser_finish_plan",
        "fl2va_keyframe_vae_encoder_plan",
    ]
    assert "ref2va_denoiser_plan" in sections
    assert "ref2va_video_vae_encoder_plan" in sections
    assert sections[-1] == "video_super_resolution_plan"


@pytest.mark.parametrize(
    "stale",
    [
        "missing_profiles",
        "fixed_then_public",
        "short_then_public",
        "three_profiles",
        "short_only",
        "changed_dynamic_shape",
    ],
)
def test_resume_rejects_stale_denoiser_profiles_before_reusing_plans(
    tmp_path: Path, monkeypatch, stale: str
) -> None:
    model = _model(tmp_path)
    plans = tmp_path / "plans"

    def build(_component, _model, plan, **_options):
        plan.write_bytes(b"plan")

    monkeypatch.setattr(staged_build, "_run_component", build)
    monkeypatch.setattr(staged_build.trt_compat, "tensorrt_version", lambda: "1.6.1")
    # Profile compatibility must not rely only on source-file timestamps.
    monkeypatch.setattr(staged_build, "_builder_source_identity", lambda: "same-source")
    writer = _writer(tmp_path / "first.bundle")
    staged_build.build_staged_bundle(model, writer, plans_dir=plans)
    writer.finish()
    state_path = plans / "build_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert len(state["denoiser_profiles"]) == 1
    public = state["denoiser_profiles"][0]
    short = dict(public)
    short.update(
        opt_video_rows=38304,
        video_rows=40716,
        audio_rows=414,
        opt_text_rows=1935,
        padded_sequence_length=43771,
    )
    fixed = dict(short)
    fixed.update(
        min_video_rows=37296,
        opt_video_rows=37296,
        video_rows=37296,
        min_text_rows=537,
        opt_text_rows=537,
        text_rows=537,
        padded_sequence_length=38247,
    )
    if stale == "missing_profiles":
        del state["denoiser_profiles"]
    elif stale == "fixed_then_public":
        state["denoiser_profiles"] = [fixed, public]
    elif stale == "short_then_public":
        state["denoiser_profiles"] = [short, public]
    elif stale == "three_profiles":
        state["denoiser_profiles"] = [fixed, short, public]
    elif stale == "short_only":
        state["denoiser_profiles"] = [short]
    else:
        state["denoiser_profiles"][0]["opt_text_rows"] -= 1
    state_path.write_text(json.dumps(state), encoding="utf-8")

    def unexpected_reuse(*_args, **_kwargs):
        pytest.fail("Stale denoiser profiles reached plan construction or bundle packing")

    monkeypatch.setattr(staged_build, "_run_component", unexpected_reuse)
    monkeypatch.setattr(staged_build, "_write_plan_section", unexpected_reuse)
    with pytest.raises(ValueError, match="different build options"):
        staged_build.build_staged_bundle(
            model, _writer(tmp_path / "second.bundle"), plans_dir=plans
        )


@pytest.mark.parametrize("enabled", [True, False])
def test_ref2va_build_selects_matching_cache_plans_and_metadata(
    tmp_path: Path, monkeypatch, enabled: bool
) -> None:
    from families.minimax_h3 import ref2va_checkpoint

    model = _model(tmp_path)
    reference = model / "transformer_ref"
    reference.mkdir()
    identity = ref2va_checkpoint.TransformerRefIdentity(
        ref2va_checkpoint.MODEL_ID,
        ref2va_checkpoint.CHECKPOINT_REVISION,
        "transformer_ref",
        ref2va_checkpoint.TOTAL_TENSOR_BYTES,
        638,
        {},
    )
    monkeypatch.setattr(
        ref2va_checkpoint, "validate_transformer_ref_checkpoint", lambda _path: identity
    )
    monkeypatch.setattr(staged_build.trt_compat, "tensorrt_version", lambda: "1.6.1")
    monkeypatch.setattr(staged_build.trt_compat, "tensorrt_abi", lambda _version: "1.6")

    def build(component, _model, plan, **_options):
        plan.write_bytes(component.encode())

    monkeypatch.setattr(staged_build, "_run_component", build)
    output = tmp_path / "h3.bundle"
    writer = _writer(output)
    # Omission exercises the default-enabled public build path.
    defaults = {"ref2va_first_block_cache_threshold": 0.05}
    if not enabled:
        defaults["ref2va_first_block_cache"] = False
    staged_build.build_staged_bundle(
        model,
        writer,
        plans_dir=tmp_path / "plans",
        transformer_ref=reference,
        runtime_defaults=defaults,
    )
    writer.finish()
    _header, sections = _read_bundle(output)
    runtime = json.loads(sections["runtime.json"])
    assert runtime["ref2va_first_block_cache"] == {"enabled": enabled, "threshold": 0.05}
    assert runtime["first_block_cache_threshold"] == 0.08
    assert ("ref2va_dit_head_plan" in sections) is enabled
    assert ("ref2va_denoiser_plan" in sections) is not enabled
    assert set(runtime["ref2va_plan_sections"].values()) <= set(sections)
    if enabled:
        assert runtime["workspace_limit_bytes"]["ref2va_dit_tail.plan"] == "trt_default_max"


@pytest.mark.parametrize(
    "changed",
    [
        "weights",
        "config",
        "builder",
        "profile",
        "sr_weights",
        "sr_weak",
        "ref",
        "quant",
        "quant_ref",
        "quant_text",
    ],
)
def test_resume_rejects_changed_inputs_before_reusing_plans(
    tmp_path: Path, monkeypatch, changed: str
) -> None:
    from families.minimax_h3 import provenance, quantized_checkpoint, ref2va_checkpoint

    model = _model(tmp_path)
    weight = model / "text_encoder" / "model.safetensors"
    weight.write_bytes(b"original")
    primary = tmp_path / provenance.SUPER_RESOLUTION_PRIMARY_FILENAME
    weak = tmp_path / provenance.SUPER_RESOLUTION_WEAK_FILENAME
    primary.write_bytes(b"primary")
    weak.write_bytes(b"weak")
    options = {"super_resolution_model": primary, "super_resolution_weak_model": weak}

    if changed in {"quant_ref", "quant_text"}:
        sources = _quantized_sources(tmp_path / "comfy", monkeypatch)
        key = "quantized_ref_transformer" if changed == "quant_ref" else "quantized_text_encoder"
        weight = sources[key]
        options[key] = weight
    elif changed == "ref":
        reference = model / "transformer_ref"
        reference.mkdir()
        weight = reference / "model.safetensors"
        weight.write_bytes(b"reference")
        identity = ref2va_checkpoint.TransformerRefIdentity(
            ref2va_checkpoint.MODEL_ID,
            ref2va_checkpoint.CHECKPOINT_REVISION,
            "transformer_ref",
            ref2va_checkpoint.TOTAL_TENSOR_BYTES,
            638,
            {weight.name: {"bytes": weight.stat().st_size}},
        )
        monkeypatch.setattr(
            ref2va_checkpoint, "validate_transformer_ref_checkpoint", lambda _path: identity
        )
        options["transformer_ref"] = reference
    elif changed == "quant":
        weight = tmp_path / "quant.safetensors"
        weight.write_bytes(b"quantized")
        monkeypatch.setattr(
            quantized_checkpoint,
            "validate_quantized_transformer_checkpoint",
            lambda path: replace(
                quantized_checkpoint.QUANTIZED_CHECKPOINT_IDENTITY,
                source_file_identity=quantized_checkpoint._source_file_identity(path),
            ),
        )
        options["quantized_transformer"] = weight

    def build(_component, _model, plan, **_options):
        plan.write_bytes(b"plan")

    monkeypatch.setattr(staged_build, "_run_component", build)
    monkeypatch.setattr(staged_build.trt_compat, "tensorrt_version", lambda: "1.6.1")
    monkeypatch.setattr(staged_build, "_builder_source_identity", lambda: "original-source")
    plans = tmp_path / "plans"
    writer = _writer(tmp_path / "first.bundle")
    staged_build.build_staged_bundle(model, writer, plans_dir=plans, **options)
    writer.finish()
    state = (plans / "build_state.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in state
    assert "source_files" in state

    if changed == "sr_weak":
        del options["super_resolution_weak_model"]
    elif changed == "builder":
        monkeypatch.setattr(staged_build, "_builder_source_identity", lambda: "changed-source")
    elif changed == "profile":
        profile = staged_build._profile()
        monkeypatch.setattr(staged_build, "_profile", lambda: replace(profile, norm_eps=1.0e-4))
    else:
        if changed == "config":
            weight = model / "audio_vae" / "config.json"
        elif changed == "sr_weights":
            weight = primary
        # A same-size change still invalidates the stat receipt.
        before = weight.stat()
        payload = weight.read_bytes()
        weight.write_bytes(payload[::-1])
        os.utime(weight, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))

    monkeypatch.setattr(
        staged_build,
        "_run_component",
        lambda *_args, **_kwargs: pytest.fail("Changed sources reached plan construction"),
    )
    with pytest.raises(ValueError, match="different build options"):
        staged_build.build_staged_bundle(
            model, _writer(tmp_path / "second.bundle"), plans_dir=plans, **options
        )


def test_staged_build_rejects_ambiguous_bf16_and_int8_ref_sources(tmp_path: Path) -> None:
    model = _model(tmp_path)
    with pytest.raises(ValueError, match="cannot combine BF16 and quantized REF"):
        staged_build.build_staged_bundle(
            model,
            _writer(tmp_path / "h3.bundle"),
            plans_dir=tmp_path / "plans",
            transformer_ref=tmp_path / "legacy-ref",
            quantized_ref_transformer=tmp_path / "ref.safetensors",
        )


@pytest.mark.parametrize("stale", ("missing_text_source", "old_text_revision"))
def test_resume_rejects_stale_nvfp4_metadata_before_build_or_pack(
    tmp_path: Path, monkeypatch, stale: str
) -> None:
    model = _model(tmp_path)
    sources = _quantized_sources(tmp_path / "comfy", monkeypatch)
    plans = tmp_path / "plans"

    def build(_component, _model, plan, **_options):
        plan.write_bytes(b"plan")

    monkeypatch.setattr(staged_build, "_run_component", build)
    monkeypatch.setattr(staged_build.trt_compat, "tensorrt_version", lambda: "1.6.1")
    monkeypatch.setattr(staged_build, "_builder_source_identity", lambda: "same-source")
    writer = _writer(tmp_path / "first.bundle")
    staged_build.build_staged_bundle(model, writer, plans_dir=plans, **sources)
    writer.finish()
    state_path = plans / "build_state.json"
    state = json.loads(state_path.read_text())
    if stale == "missing_text_source":
        del state["quantized_text_encoder"]
    else:
        state["quantized_text_encoder"]["revision"] = "old-pinned-revision"
    state_path.write_text(json.dumps(state))

    def unexpected_reuse(*_args, **_kwargs):
        pytest.fail("Stale NVFP4 source metadata reached plan construction or packing")

    monkeypatch.setattr(staged_build, "_run_component", unexpected_reuse)
    monkeypatch.setattr(staged_build, "_write_plan_section", unexpected_reuse)
    with pytest.raises(ValueError, match="different build options"):
        staged_build.build_staged_bundle(
            model, _writer(tmp_path / "second.bundle"), plans_dir=plans, **sources
        )


@pytest.mark.parametrize(
    "component,build_name,key_name",
    (
        ("ref2va_denoiser", "build_ref2va_dit_engine", "checkpoint_keys"),
        ("ref2va_dit_head", "build_ref2va_dit_head_engine", "head_checkpoint_keys"),
        ("ref2va_dit_tail", "build_ref2va_dit_tail_engine", "tail_checkpoint_keys"),
        ("ref2va_dit_finish", "build_ref2va_dit_finish_engine", "finish_checkpoint_keys"),
        (
            "ref2va_adaln_precompute",
            "build_ref2va_adaln_precompute_engine",
            "adaln_checkpoint_keys",
        ),
    ),
)
def test_ref_component_loads_only_selected_quantized_ref_partition(
    tmp_path: Path, monkeypatch, component: str, build_name: str, key_name: str
) -> None:
    from families.minimax_h3 import checkpoint, quantized_checkpoint, trt_compat

    trt_compat.configure_backend(rtx=True)
    from families.minimax_h3 import ref2va_dit_builder

    path = tmp_path / "ref.safetensors"
    output = tmp_path / "ref.plan"
    loaded, observed = {"quantized-sentinel": object()}, []

    def load(source, keys, *, workflow):
        observed.append((source, tuple(keys), workflow))
        return loaded

    def build(weights, **options):
        assert weights is loaded
        assert options["consume_weights"] is True
        assert options["output_path"] == output
        output.write_bytes(b"ref-plan")
        return {"bytes": output.stat().st_size}

    monkeypatch.setattr(quantized_checkpoint, "load_selected_quantized_transformer_weights", load)
    monkeypatch.setattr(
        checkpoint,
        "load_selected_component_state_dict",
        lambda *_args: pytest.fail("Quantized REF child fell back to BF16 checkpoint loading"),
    )
    monkeypatch.setattr(ref2va_dit_builder, build_name, build)
    result = staged_build._build_component(
        component, tmp_path, output, verbose=False, quantized_ref_transformer_path=path
    )
    assert result == {"bytes": len(b"ref-plan")}
    assert observed == [(path, getattr(ref2va_dit_builder, key_name)(), "ref2va")]


@pytest.mark.parametrize("component", ("text_encoder", "vision_encoder"))
def test_shared_qwen_child_uses_nvfp4_checkpoint_for_text_and_vision(
    tmp_path: Path, monkeypatch, component: str
) -> None:
    from families.minimax_h3 import checkpoint, nvfp4_text_checkpoint, trt_compat

    trt_compat.configure_backend(rtx=True)
    from families.minimax_h3 import ref2va_qwen_builder

    path = tmp_path / "qwen.safetensors"
    output = tmp_path / "qwen.plan"
    loaded, observed = {"nvfp4-sentinel": object()}, []

    def load(source, keys):
        observed.append((source, tuple(keys)))
        return loaded

    def build(weights, **options):
        assert weights is loaded
        assert options["output_path"] == output
        output.write_bytes(b"qwen-plan")
        return {"bytes": output.stat().st_size}

    monkeypatch.setattr(nvfp4_text_checkpoint, "load_selected_quantized_text_weights", load)
    monkeypatch.setattr(
        checkpoint,
        "load_selected_component_state_dict",
        lambda *_args: pytest.fail("Qwen child fell back to the official text checkpoint"),
    )
    monkeypatch.setattr(ref2va_qwen_builder, f"build_ref2va_shared_{component}_engine", build)
    staged_build._build_component(
        component,
        tmp_path,
        output,
        verbose=False,
        quantized_ref_transformer_path=tmp_path / "ref.safetensors",
        quantized_text_encoder_path=path,
    )
    assert len(observed) == 1 and observed[0][0] == path
    assert observed[0][1]
    assert all(
        name.startswith("model.visual.") == (component == "vision_encoder")
        for name in observed[0][1]
    )


def test_staged_child_command_preserves_distinct_ref_and_qwen_sources(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "qwen.plan"
    reference, qwen = tmp_path / "ref.safetensors", tmp_path / "qwen.safetensors"
    commands = []

    def run(command, *, check):
        assert check is True
        commands.append(command)
        output.write_bytes(b"plan")

    monkeypatch.setattr(staged_build.subprocess, "run", run)
    staged_build._run_component(
        "text_encoder",
        tmp_path,
        output,
        verbose=False,
        quantized_ref_transformer_path=reference,
        quantized_text_encoder_path=qwen,
    )
    assert len(commands) == 1
    command = commands[0]
    assert command[command.index("--quantized-ref-transformer") + 1] == str(reference)
    assert command[command.index("--quantized-text-encoder") + 1] == str(qwen)
    assert "--transformer-ref" not in command
    assert "--quantized-transformer" not in command
