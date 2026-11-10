# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Student partition/streaming/reuse admission with explicit engine doubles."""

from dataclasses import replace
import json
import shutil

import pytest

from tensorrt_model_connect.bundle_writer import BundleWriter
from families.minimax_h3 import reference_build as full
from families.minimax_h3 import reference_pdd_build as student
from families.minimax_h3.reference_pdd import CONTRACT, TRANSFORMER_CONFIG
from .test_reference_build import request as make_request, sections
from . import test_reference_build as original
from .test_reference_pdd_partition import fixture_index


checkpoint = original.checkpoint
plans = original.plans


@pytest.fixture(params=[(False, "minimal"), (True, "minimal"), (True, "original")])
def baseline(checkpoint, plans, tmp_path, request):
    continuation, preset = request.param
    frames = 226 if continuation else 124
    root = (
        tmp_path
        / "hub/models--MiniMaxAI--MiniMax-H3/snapshots/48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc"
    )
    shutil.copytree(checkpoint, root)
    source = tmp_path / "original.bundle"
    original_request = make_request(root, source, video_num_frames=frames)
    plans.batch, plans.text_rows = (1 if preset == "minimal" else 28), 65536
    plans.condition_rows = (4096, 50832, 56208) if continuation else (4096, 43776, 49152)
    plans.audio_condition_rows = (0, 74, 74) if continuation else (0, 0, 0)
    plans.decoded_audio_latents = 377 if continuation else 207
    writer = BundleWriter(source)
    writer.record_build(original_request)
    add = writer.add_bytes

    def stage(name, value):
        if name.endswith(".plan"):
            plans.emitted.append(name)
        add(name, value)

    writer.add_bytes = stage
    full.build(
        original_request,
        writer,
        reference_capacity="six_portrait",
        memory_preset=preset,
        continuation=continuation,
    )
    if continuation:
        writer.add_json(
            "minimax_h3.reused_plans.json", {"fixture": "explicit retained base receipt"}
        )
    writer.finish()
    model = (
        tmp_path
        / f"hub/models--FastVideo--FastVideo-FastH3-Omni-8-Step-V1/snapshots/{student.MODEL_REVISION}"
    )
    (model / "transformer_ref").mkdir(parents=True)
    for name in ("export_manifest.json", "fastvideo_inference.json"):
        (model / name).write_text(json.dumps(CONTRACT))
    (model / "transformer_ref/config.json").write_text(json.dumps(TRANSFORMER_CONFIG))
    (model / "transformer_ref/diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps(fixture_index())
    )
    bridge = tmp_path / "explicit-bridge-double.so"
    bridge.write_bytes(b"not an actual shared library; this test never builds a GPU graph")
    for value in (plans.built, plans.emitted, plans.loaded, plans.partitions, plans.retired):
        value.clear()
    output = tmp_path / "student.bundle"
    return dict(
        source=source,
        request=replace(original_request, model_dir=model, output_path=output),
        output=output,
        base=root,
        bridge=bridge,
        preset=preset,
        continuation=continuation,
    )


@pytest.fixture
def student_builders(monkeypatch, baseline):
    from families.minimax_h3 import adaln_builder, reference_pdd_dit_builder

    loaded, built = [], []
    partition = student.checkpoint_partition()

    def load(directory, keys):
        assert directory == baseline["request"].model_dir / "transformer_ref"
        group = next(group for group, names in partition.items() if tuple(keys) == names)
        loaded.append(group)
        return {"student_partition": group}

    def builder(group):
        def build(weights, target, **options):
            assert weights == {"student_partition": group}
            assert options["consume_weights"] is True
            expected_workspace = (
                baseline.get("denoiser_workspace_bytes", 8 << 30)
                if group == "denoiser"
                else 8 << 30
            )
            assert options["workspace_bytes"] == expected_workspace
            if group == "adaln":
                assert options["input_timesteps"] is True
            else:
                assert options["plugin_library"] == baseline["bridge"]
                assert options["max_vsa_tiles"] == 2048
                assert options["reference_profile"].condition_audio_rows == (
                    (0, 74, 74) if baseline["continuation"] else (0, 0, 0)
                )
            assert target.num_layers == 50 and target.hidden_size == 5376
            options["output"].write(("new-student-" + group).encode())
            built.append(group)
            weights.clear()
            return None

        return build

    monkeypatch.setattr(full, "load_selected_component_state_dict", load)
    monkeypatch.setattr(full, "numpy_state", lambda value: value)
    monkeypatch.setattr(adaln_builder, "build_adaln_precompute_engine", builder("adaln"))
    monkeypatch.setattr(reference_pdd_dit_builder, "build_engine", builder("denoiser"))
    return loaded, built


def run_build(baseline, writer, **overrides):
    options = dict(
        base_model_dir=baseline["base"],
        source_bundle=baseline["source"],
        plugin_library=baseline["bridge"],
        memory_preset=baseline["preset"],
        reference_capacity="six_portrait",
        continuation=baseline["continuation"],
    )
    options.update(overrides)
    return student.build(baseline["request"], writer, **options)


def test_student_build_replaces_both_learned_partitions_and_streams_common_plans(
    baseline,
    student_builders,
):
    writer = BundleWriter(baseline["output"])
    original_bytes = baseline["source"].read_bytes()
    run_build(baseline, writer)
    writer.finish()
    assert student_builders == (["adaln", "denoiser"], ["adaln", "denoiser"])
    assert baseline["source"].read_bytes() == original_bytes
    _, before = sections(baseline["source"])
    header, after = sections(baseline["output"])
    assert list(header["sections"])[:2] == ["adaln.plan", "denoiser.plan"]
    assert after["adaln.plan"] == b"new-student-adaln"
    assert after["denoiser.plan"] == b"new-student-denoiser"
    for name in student.REUSED_PLANS:
        assert after[name] == before[name]
    metadata = json.loads(after["runtime.json"])
    assert metadata["checkpoint"] == {"hf_id": student.MODEL_ID, "revision": student.MODEL_REVISION}
    assert metadata["inference_contract"] == CONTRACT
    assert metadata["num_inference_steps"] == metadata["denoiser_forwards"] == 8
    assert metadata["num_frames"] == (226 if baseline["continuation"] else 124)
    assert metadata.get("total_denoiser_forwards") == (16 if baseline["continuation"] else None)
    assert metadata["max_vsa_tiles"] == 2048
    assert metadata["schema_version"] == (
        "minimax_h3.omniref_pdd_continuation/v1"
        if baseline["continuation"]
        else "minimax_h3.omniref_pdd/v1"
    )
    receipt = json.loads(after["minimax_h3.reused_plans.json"])
    assert receipt["rebuilt_sections"] == ["adaln.plan", "denoiser.plan"]
    assert set(receipt["reused_sections"]) == set(student.REUSED_PLANS)
    provenance = json.loads(after["trtmc.provenance.json"])
    assert provenance["checkpoint"] == metadata["checkpoint"]


def test_explicit_denoiser_workspace_preserves_adaln_budget_and_runtime_contract(
    baseline,
    student_builders,
):
    baseline["denoiser_workspace_bytes"] = 16 << 30
    writer = BundleWriter(baseline["output"])
    run_build(baseline, writer, denoiser_workspace_bytes=16 << 30)
    writer.finish()
    _, data = sections(baseline["output"])
    receipt = json.loads(data["minimax_h3.reused_plans.json"])
    assert receipt["rebuilt_plan_workspace_bytes"] == {
        "adaln.plan": 8 << 30,
        "denoiser.plan": 16 << 30,
    }
    assert student_builders == (["adaln", "denoiser"], ["adaln", "denoiser"])
    runtime = json.loads(data["runtime.json"])
    assert runtime["inference_contract"] == CONTRACT
    assert runtime["max_vsa_tiles"] == 2048
    assert runtime["num_inference_steps"] == 8


@pytest.mark.parametrize("workspace", [True, 0, -1, 1.5, "16 GiB"])
def test_invalid_denoiser_workspace_fails_before_weight_loading(
    baseline,
    student_builders,
    workspace,
):
    writer = BundleWriter(baseline["output"])
    with pytest.raises(ValueError, match="workspace_bytes"):
        run_build(baseline, writer, denoiser_workspace_bytes=workspace)
    assert student_builders == ([], [])
    writer.abort()


@pytest.mark.parametrize(
    "problem",
    [
        "source_layout",
        "source_profile",
        "student_contract",
        "student_index",
        "student_config",
        "student_identity",
        "base_identity",
        "tiles",
        "missing_bridge",
    ],
)
def test_invalid_source_or_student_fails_before_loading_any_weights(
    baseline,
    student_builders,
    problem,
):
    options = {}
    root = baseline["request"].model_dir
    if problem == "source_layout":
        with baseline["source"].open("ab") as source:
            source.write(b"\0")
    elif problem == "source_profile":
        options["continuation"] = not baseline["continuation"]
    elif problem == "student_contract":
        (root / "fastvideo_inference.json").write_text("{}")
    elif problem == "student_index":
        value = fixture_index()
        value["weight_map"].pop("transformer_blocks.49.attn.to_gate_compress.weight")
        (root / "transformer_ref/diffusion_pytorch_model.safetensors.index.json").write_text(
            json.dumps(value)
        )
    elif problem == "student_config":
        (root / "transformer_ref/config.json").write_text("{}")
    elif problem == "student_identity":
        baseline["request"] = replace(baseline["request"], model_dir=root.parent / ("a" * 40))
    elif problem == "base_identity":
        options["base_model_dir"] = baseline["base"].parent / ("a" * 40)
    elif problem == "tiles":
        options["max_vsa_tiles"] = 2047
    else:
        options["plugin_library"] = baseline["bridge"].parent / "missing.so"
    writer = BundleWriter(baseline["output"])
    with pytest.raises((ValueError, FileNotFoundError)):
        run_build(baseline, writer, **options)
    writer.abort()
    assert student_builders == ([], [])
    assert not baseline["output"].exists()
