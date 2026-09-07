# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
import struct

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
