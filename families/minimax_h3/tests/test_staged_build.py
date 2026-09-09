# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace
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
    "changed", ["weights", "config", "builder", "profile", "sr_weights", "sr_weak", "ref", "quant"]
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

    if changed == "ref":
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
