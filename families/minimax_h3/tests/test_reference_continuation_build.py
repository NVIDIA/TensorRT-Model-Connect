# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Incremental bundle identity/streaming checks; explicit builder doubles."""

from dataclasses import replace
import json
import os
import shutil

import pytest

from tensorrt_model_connect.bundle_writer import BundleWriter
from families.minimax_h3 import reference_build as full
from families.minimax_h3 import reference_continuation_build as incremental
from . import test_reference_build as base
from .test_reference_build import request as make_request, sections

checkpoint = base.checkpoint
plans = base.plans


@pytest.fixture(params=["minimal", "original"])
def baseline(checkpoint, plans, tmp_path, request):
    preset = request.param
    root = (
        tmp_path
        / "hub/models--MiniMaxAI--MiniMax-H3/snapshots/48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc"
    )
    shutil.copytree(checkpoint, root)
    source = tmp_path / "original.bundle"
    writer = BundleWriter(source)
    original = make_request(root, source)
    writer.record_build(original)
    plans.batch, plans.text_rows, plans.condition_rows = (
        (1 if preset == "minimal" else 28),
        65536,
        (4096, 43776, 49152),
    )

    def observed(writer):
        add = writer.add_bytes

        def stage(name, value):
            if name.endswith(".plan"):
                plans.emitted.append(name)
            add(name, value)

        writer.add_bytes = stage
        return writer

    full.build(original, observed(writer), reference_capacity="six_portrait", memory_preset=preset)
    writer.finish()
    for values in [plans.built, plans.emitted, plans.loaded, plans.partitions, plans.retired]:
        values.clear()
    plans.audio_condition_rows = (0, 74, 74)
    plans.condition_rows = (4096, 50832, 56208)
    plans.decoded_audio_latents = 377
    output = tmp_path / "continuation.bundle"
    return dict(
        source=source,
        request=replace(original, video_num_frames=226, output_path=output),
        output=output,
        writer=observed,
        root=root,
        preset=preset,
    )


def test_incremental_build_loads_only_changed_partitions_and_preserves_six_plan_bytes(
    baseline, plans
):
    original = baseline["source"].read_bytes()
    writer = baseline["writer"](BundleWriter(baseline["output"]))
    incremental.build(
        baseline["request"],
        writer,
        source_bundle=baseline["source"],
        memory_preset=baseline["preset"],
    )
    writer.finish()
    assert baseline["source"].read_bytes() == original
    assert plans.built == plans.emitted == ["denoiser.plan", "audio_vae.plan"]
    assert [name for name, _ in plans.loaded] == ["transformer_ref", "audio_vae"]
    assert plans.retired == set(plans.built)
    _, before = sections(baseline["source"])
    _, after = sections(baseline["output"])
    for name in incremental.REUSED_PLANS:
        assert before[name] == after[name]
    runtime = json.loads(after["runtime.json"])
    assert runtime["schema_version"] == "minimax_h3.image_ref2va_continuation/v1"
    assert runtime["num_frames"] == 226 and runtime["total_denoiser_forwards"] == 98
    assert runtime["condition_audio_rows"] == [0, 74, 74]
    assert runtime["decoded_audio_latent_frames"] == 377
    receipt = json.loads(after["minimax_h3.reused_plans.json"])
    assert receipt["reused_sections"] == list(incremental.REUSED_PLANS)
    assert receipt["rebuilt_sections"] == ["denoiser.plan", "audio_vae.plan"]
    provenance = json.loads(after["trtmc.provenance.json"])
    assert provenance["build"]["video_num_frames"] == 226
    assert provenance["checkpoint"]["revision"] == baseline["root"].name


@pytest.mark.parametrize(
    "change", ["trailing", "profile", "checkpoint", "tokenizer", "source_output", "legacy"]
)
def test_incompatible_source_fails_before_new_weights_or_engines(baseline, plans, change):
    req = baseline["request"]
    if change == "trailing":
        with baseline["source"].open("ab") as source:
            source.write(b"\0")
    elif change == "profile":
        req = replace(req, image_height=1536)
    elif change == "checkpoint":
        root = baseline["root"].parent / ("b" * 40)
        shutil.copytree(baseline["root"], root)
        req = replace(req, model_dir=root)
    elif change == "tokenizer":
        (baseline["root"] / "tokenizer/tokenizer.json").write_bytes(b"{}")
    elif change == "source_output":
        req = replace(req, output_path=baseline["source"])
    else:
        req = replace(req, video_num_frames=124)
    writer = BundleWriter(baseline["output"])
    with pytest.raises(ValueError):
        incremental.build(
            req,
            writer,
            source_bundle=baseline["source"],
            memory_preset=baseline["preset"],
        )
    writer.abort()
    assert not plans.loaded and not plans.built and not baseline["output"].exists()


def test_streamed_copy_detects_mutation_after_source_validation(baseline):
    req = baseline["request"]
    height, width, target, reference = full._target(
        replace(req, video_num_frames=124), baseline["preset"], "six_portrait", False
    )
    configs = full._configs(baseline["root"])
    expected = full.runtime_metadata(
        height,
        width,
        target,
        reference,
        configs,
        memory_preset=baseline["preset"],
        reference_capacity="six_portrait",
        continuation=False,
    )
    from tensorrt_model_connect.provenance import checkpoint_identity

    source = incremental._ImageBundle(
        baseline["source"],
        expected,
        checkpoint_identity(baseline["root"]),
        (baseline["root"] / "tokenizer/tokenizer.json").read_bytes(),
    )
    data = bytearray(baseline["source"].read_bytes())
    data[source.base] ^= 1
    rewritten = baseline["source"].with_suffix(".rewritten")
    rewritten.write_bytes(data)
    os.replace(rewritten, baseline["source"])
    writer = BundleWriter(baseline["output"])
    writer.enable_direct_staging()
    writer.set_header(family="minimax_h3", task=full.TASK, backend="trt")
    with pytest.raises(ValueError):
        source.copy_to("vision.plan", writer)
    writer.abort()
    assert not baseline["output"].exists()


@pytest.mark.parametrize("kind", ["valid", "missing", "wrong_parent", "wrong_repository", "short_revision", "uppercase_revision", "none"])
def test_continuation_checkpoint_identity_stays_equivalent_without_core_dependency(tmp_path, kind):
    from tensorrt_model_connect.provenance import checkpoint_identity
    from families.minimax_h3.reference_continuation_build import _checkpoint_identity

    root = tmp_path / "models--MiniMaxAI--MiniMax-H3" / "snapshots" / ("a" * 40)
    if kind == "missing":
        root = tmp_path / "ordinary"
    elif kind == "wrong_parent":
        root = root.parent.parent / "revisions" / root.name
    elif kind == "wrong_repository":
        root = tmp_path / "MiniMax-H3" / "snapshots" / root.name
    elif kind == "short_revision":
        root = root.parent / ("a" * 39)
    elif kind == "uppercase_revision":
        root = root.parent / ("A" * 40)
    elif kind == "none":
        root = None
    assert _checkpoint_identity(root) == checkpoint_identity(root)
