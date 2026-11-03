# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU build-wiring/lifetime proofs, not trained-model or engine oracles."""

from dataclasses import replace
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest

from tensorrt_model_connect import BuildRequest
from tensorrt_model_connect.bundle_writer import BundleWriter
from families.minimax_h3 import model, reference_build as owner


@pytest.fixture
def checkpoint(tmp_path):
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/original-ref2va-configs.json").read_text()
    )
    assert fixture["source_revision"] == "48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc"
    root = tmp_path / "checkpoint"
    for name, record in fixture["components"].items():
        (root / name).mkdir(parents=True)
        (root / name / "config.json").write_bytes(record["utf8"].encode())
    (root / "tokenizer").mkdir()
    (root / "tokenizer/tokenizer.json").write_text(
        json.dumps(
            dict(
                model=dict(type="BPE"),
                added_tokens=[
                    dict(content=token, id=index)
                    for token, index in [
                        ("<|vision_start|>", 151652),
                        ("<|vision_end|>", 151653),
                        ("<|image_pad|>", 151655),
                    ]
                ],
            )
        )
    )
    return root


@pytest.fixture
def plans(monkeypatch):
    from families.minimax_h3 import adaln_builder, audio_vae_builder, dit_builder
    from families.minimax_h3 import reference_posterior_builder, text_encoder_builder
    from families.minimax_h3 import vae_builder, vae_encoder_builder, vision_builder

    state = SimpleNamespace(
        built=[],
        emitted=[],
        retired=set(),
        loaded=[],
        partitions=[],
        text_rows=32768,
        condition_rows=(4096, 5504, 16384),
        audio_condition_rows=(0, 0, 0),
        decoded_audio_latents=207,
    )

    class Plan(bytes):
        def __new__(cls, name):
            value = super().__new__(cls, (name + "-plan-double").encode())
            value.name = name
            return value

        def __del__(self):
            state.retired.add(self.name)

    def load(directory, keys):
        assert state.retired == set(state.built), "prior serialized plan retained into next weights"
        assert state.emitted == state.built, "prior plan not staged before next weights"
        assert keys and Path(directory).name != "transformer"
        state.loaded.append((Path(directory).name, tuple(keys)))
        return {"component": Path(directory).name}

    def builder(name):
        def build(weights, *args, **kwargs):
            assert kwargs["consume_weights"] is True and kwargs["workspace_bytes"] == 8 << 30
            if name == "vision.plan":
                assert (
                    kwargs["patch_rows"] == (16384, 22016, 65536) and kwargs["precision"] == "bf16"
                )
            elif name == "text_encoder.plan":
                assert kwargs["reference"] is True and kwargs["sequence_length"] == state.text_rows
            elif name == "vae_encoder.plan":
                assert kwargs["max_frames"] == 1
            elif name == "adaln.plan":
                assert kwargs["input_timesteps"] is True and args[0].first_block_cache is False
            elif name == "denoiser.plan":
                assert (
                    kwargs["reference_profile"].condition_audio_rows == state.audio_condition_rows
                )
                assert kwargs["reference_profile"].condition_video_rows == state.condition_rows
            elif name == "vae.plan":
                assert kwargs["batch_size"] == state.batch
            elif name == "audio_vae.plan":
                assert len(args[0]) == len(args[1]) == 32
                assert kwargs.get("latent_frames", 207) == state.decoded_audio_latents
            weights.clear()
            state.built.append(name)
            return Plan(name)

        return build

    def posterior(*, latent_pixels):
        assert latent_pixels == (16384, 22016, 65536)
        assert state.retired == set(state.built)
        state.built.append("vae_reference_posterior.plan")
        return Plan("vae_reference_posterior.plan")

    def partition(directory, groups):
        assert not set(groups[0]) & set(groups[1])
        state.partitions.append((Path(directory).name, sum(map(len, groups))))

    monkeypatch.setattr(owner, "load_selected_component_state_dict", load)
    monkeypatch.setattr(owner, "numpy_state", lambda value: value)
    monkeypatch.setattr(owner, "validate_component_key_partition", partition)
    monkeypatch.setattr(reference_posterior_builder, "build_reference_posterior_engine", posterior)
    for module, function, name in [
        (vision_builder, "build_vision_encoder_engine", "vision.plan"),
        (text_encoder_builder, "build_text_encoder_engine", "text_encoder.plan"),
        (vae_encoder_builder, "build_vae_tile_encoder_engine", "vae_encoder.plan"),
        (adaln_builder, "build_adaln_precompute_engine", "adaln.plan"),
        (dit_builder, "build_dit_engine", "denoiser.plan"),
        (vae_builder, "build_vae_tile_decoder_engine", "vae.plan"),
        (audio_vae_builder, "build_audio_vae_decoder_engine", "audio_vae.plan"),
    ]:
        monkeypatch.setattr(module, function, builder(name))
    return state


def request(root, output, **overrides):
    return BuildRequest(
        model_dir=root,
        output_path=output,
        family="minimax_h3",
        task=owner.TASK,
        precision="bf16",
        **overrides,
    )


def sections(path):
    raw = path.read_bytes()
    assert raw[:8] == b"BUNDLE\x01\x00"
    length = struct.unpack("<Q", raw[8:16])[0]
    header = json.loads(raw[16 : 16 + length])
    data = raw[16 + length :]
    return header, {
        name: data[row["offset"] : row["offset"] + row["length"]]
        for name, row in header["sections"].items()
    }


@pytest.mark.parametrize("preset,batch", [("minimal", 1), ("original", 28)])
@pytest.mark.parametrize("height,width", [(1344, 768), (1536, 864)])
def test_reference_build_streams_all_eight_exact_partitions(
    checkpoint, plans, tmp_path, preset, batch, height, width
):
    output = tmp_path / "reference.bundle"
    writer = BundleWriter(output)
    plans.batch = batch
    add = writer.add_bytes

    def stage(name, value):
        if name.endswith(".plan"):
            plans.emitted.append(name)
        add(name, value)
        assert not list(tmp_path.glob(".reference.bundle.sections.*"))
        assert len(list(tmp_path.glob(".reference.bundle.*.tmp"))) == 1

    writer.add_bytes = stage
    owner.build(
        request(checkpoint, output, image_height=height, image_width=width),
        writer,
        memory_preset=preset,
    )
    writer.finish()
    assert struct.unpack("<Q", output.read_bytes()[8:16])[0] == 65536
    assert plans.built == plans.emitted == list(owner.PLAN_NAMES)
    assert plans.retired == set(owner.PLAN_NAMES)
    assert [name for name, _keys in plans.loaded] == [
        "text_encoder",
        "text_encoder",
        "vae",
        "transformer_ref",
        "transformer_ref",
        "vae",
        "audio_vae",
    ]
    assert plans.partitions == [("transformer_ref", 638), ("vae", 703)]
    header, data = sections(output)
    assert header["task"] == "references_text_to_audio_video" and header["family"] == "minimax_h3"
    assert set(data) == {*owner.PLAN_NAMES, "runtime.json", "tokenizer.json"}
    for name in owner.PLAN_NAMES:
        assert data[name] == (name + "-plan-double").encode()
    runtime = json.loads(data["runtime.json"])
    assert (
        runtime["height"],
        runtime["width"],
        runtime["latent_height"],
        runtime["latent_width"],
    ) == (height, width, height // 16, width // 16)
    assert runtime["num_frames"] == 124 and runtime["denoiser_forwards"] == 49
    assert runtime["memory_preset"] == preset and runtime["vae_tile_batch"] == batch
    assert runtime["weights_variant"] == "transformer_ref" and runtime["first_block_cache"] is False
    assert runtime["max_sequence_rows"] == 37 * (height // 32) * (width // 32) + 414 + 32768 + 16384
    assert (
        runtime["latents_mean"]
        == json.loads((checkpoint / "vae/config.json").read_text())["latents_mean"]
    )
    assert "reference_capacity" not in runtime  # Legacy default metadata stays unchanged.


@pytest.mark.parametrize("preset,batch", [("minimal", 1), ("original", 28)])
def test_six_portrait_capacity_rebuilds_language_and_dit_profiles_without_quality_changes(
    checkpoint, plans, tmp_path, preset, batch
):
    output = tmp_path / "six-portrait.bundle"
    writer = BundleWriter(output)
    plans.batch, plans.text_rows, plans.condition_rows = batch, 65536, (4096, 43776, 49152)
    add = writer.add_bytes

    def stage(name, value):
        if name.endswith(".plan"):
            plans.emitted.append(name)
        add(name, value)

    writer.add_bytes = stage
    owner.build(
        request(checkpoint, output), writer, memory_preset=preset, reference_capacity="six_portrait"
    )
    writer.finish()
    assert plans.built == plans.emitted == list(owner.PLAN_NAMES)
    assert plans.retired == set(owner.PLAN_NAMES)
    _, data = sections(output)
    runtime = json.loads(data["runtime.json"])
    assert runtime["reference_capacity"] == "six_portrait"
    assert runtime["max_text_rows"] == 65536 and runtime["text_rows"] == [1, 128, 65536]
    assert runtime["condition_video_rows"] == [4096, 43776, 49152]
    assert runtime["max_sequence_rows"] == 152398
    assert runtime["reference_pixels"] == [16384, 22016, 65536]
    assert (
        runtime["num_frames"],
        runtime["fps"],
        runtime["num_inference_steps"],
        runtime["denoiser_forwards"],
        runtime["height"],
        runtime["width"],
    ) == (124, 24, 50, 49, 1344, 768)
    assert runtime["memory_preset"] == preset and runtime["vae_tile_batch"] == batch
    assert runtime["weights_variant"] == "transformer_ref" and runtime["first_block_cache"] is False
    assert plans.partitions == [("transformer_ref", 638), ("vae", 703)]


@pytest.mark.parametrize("capacity", ["", "auto", "six_images", None, True, [65536, 49152]])
def test_unknown_reference_capacity_fails_before_weights(checkpoint, plans, tmp_path, capacity):
    output = tmp_path / "no.bundle"
    writer = BundleWriter(output)
    with pytest.raises(ValueError, match="reference_capacity"):
        owner.build(request(checkpoint, output), writer, reference_capacity=capacity)
    writer.abort()
    assert not plans.loaded and not output.exists()


@pytest.mark.parametrize("preset,batch", [("minimal", 1), ("original", 28)])
@pytest.mark.parametrize("capacity", ["standard", "six_portrait"])
@pytest.mark.parametrize("height,width", [(1344, 768), (1536, 864)])
def test_two_window_build_has_real_guide_and_cumulative_decoder_profiles(
    checkpoint, plans, tmp_path, preset, batch, capacity, height, width
):
    output = tmp_path / "two-window.bundle"
    writer = BundleWriter(output)
    plans.batch = batch
    plans.text_rows, base = owner.REFERENCE_CAPACITIES[capacity]
    guide = 7 * (height // 32) * (width // 32)
    plans.condition_rows = (base[0], base[1] + guide, base[2] + guide)
    plans.audio_condition_rows = (0, 74, 74)
    plans.decoded_audio_latents = 377
    add = writer.add_bytes

    def stage(name, value):
        if name.endswith(".plan"):
            plans.emitted.append(name)
        add(name, value)

    writer.add_bytes = stage
    owner.build(
        request(checkpoint, output, video_num_frames=226, image_height=height, image_width=width),
        writer,
        memory_preset=preset,
        reference_capacity=capacity,
        continuation=True,
    )
    writer.finish()
    _, data = sections(output)
    runtime = json.loads(data["runtime.json"])
    assert plans.built == plans.emitted == list(owner.PLAN_NAMES)
    assert plans.retired == set(owner.PLAN_NAMES)
    assert runtime["schema_version"] == "minimax_h3.image_ref2va_continuation/v1"
    assert runtime["workflow"] == "image_ref2va_continuation"
    assert runtime["condition_video_rows"] == list(plans.condition_rows)
    assert runtime["condition_audio_rows"] == [0, 74, 74]
    assert runtime["max_sequence_rows"] == (
        37 * (height // 32) * (width // 32) + 414 + plans.text_rows + base[2] + guide + 74
    )
    assert [
        runtime[k]
        for k in [
            "num_frames",
            "continuation_window_frames",
            "continuation_overlap_frames",
            "continuation_windows",
            "latent_frames",
            "audio_latent_frames",
            "decoded_latent_frames",
            "decoded_audio_latent_frames",
            "denoiser_forwards",
            "total_denoiser_forwards",
        ]
    ] == [226, 124, 22, 2, 37, 207, 67, 377, 49, 98]
    assert runtime["num_inference_steps"] == 50 and runtime["first_block_cache"] is False
    assert runtime["memory_preset"] == preset and runtime["vae_tile_batch"] == batch
    assert plans.partitions == [("transformer_ref", 638), ("vae", 703)]


@pytest.mark.parametrize(
    "continuation,frames", [(True, 124), (False, 226), (1, 226), ("true", 226), (None, 226)]
)
def test_invalid_continuation_profile_rejected_before_weights(
    checkpoint, plans, tmp_path, continuation, frames
):
    writer = BundleWriter(tmp_path / "invalid.bundle")
    with pytest.raises(ValueError, match="frames|continuation"):
        owner.build(
            request(checkpoint, tmp_path / "invalid.bundle", video_num_frames=frames),
            writer,
            continuation=continuation,
        )
    writer.abort()
    assert not plans.loaded


def test_missing_ref_partition_cannot_fall_back_to_text_only(checkpoint, plans, tmp_path):
    (checkpoint / "transformer_ref/config.json").rename(checkpoint / "text-only-config.json")
    (checkpoint / "transformer").mkdir()
    (checkpoint / "transformer/config.json").write_bytes(
        (checkpoint / "text-only-config.json").read_bytes()
    )
    output = tmp_path / "no.bundle"
    writer = BundleWriter(output)
    with pytest.raises(FileNotFoundError):
        owner.build(request(checkpoint, output), writer)
    writer.abort()
    assert not plans.loaded


@pytest.mark.parametrize(
    "component,key,value",
    [
        ("transformer_ref", "num_layers", 1),
        ("vae", "decoder_num_layers", 1),
        ("audio_vae", "sampling_rate", 24000),
        ("vae", "latents_std", [0.0] * 24),
        ("audio_vae", "latents_mean", [float("nan")] * 32),
        ("vae", "latents_mean", [1e300] * 24),
        ("vae", "latents_std", [1e-100] * 24),
    ],
)
def test_incompatible_configs_fail_before_loading_weights(
    checkpoint, plans, tmp_path, component, key, value
):
    path = checkpoint / component / "config.json"
    raw = json.loads(path.read_text())
    raw[key] = value
    path.write_text(json.dumps(raw))
    output = tmp_path / "no.bundle"
    writer = BundleWriter(output)
    with pytest.raises(ValueError):
        owner.build(request(checkpoint, output), writer)
    writer.abort()
    assert not plans.loaded and not output.exists()


def test_failed_plan_staging_preserves_published_destination(checkpoint, plans, tmp_path):
    output = tmp_path / "existing.bundle"
    output.write_bytes(b"prior-published-bundle")
    writer = BundleWriter(output)
    add = writer.add_bytes
    plans.batch = 1

    def fail(name, value):
        if name == "denoiser.plan":
            raise OSError("intentional staging failure")
        plans.emitted.append(name)
        add(name, value)

    writer.add_bytes = fail
    with pytest.raises(OSError, match="staging failure"):
        owner.build(request(checkpoint, output), writer)
    writer.abort()
    assert output.read_bytes() == b"prior-published-bundle"
    assert plans.built == list(owner.PLAN_NAMES[:6])


def test_model_dispatches_explicit_reference_task_only(monkeypatch, checkpoint, tmp_path):
    observed = []
    writer = object()
    req = request(checkpoint, tmp_path / "explicit.bundle")
    monkeypatch.setattr(owner, "build", lambda actual, target: observed.append((actual, target)))
    model.build(req, writer)
    assert observed == [(req, writer)]


@pytest.mark.parametrize(
    "overrides",
    [
        dict(precision="fp16"),
        dict(max_batch_size=2),
        dict(dynamic_kv_cache=True),
        dict(image_height=1345),
        dict(video_num_frames=16),
        dict(max_sequence_length=8192),
        dict(quantization="fp8"),
    ],
)
def test_unsupported_overrides_rejected_without_checkpoint_access(tmp_path, overrides):
    req = replace(request(tmp_path / "absent", tmp_path / "no.bundle"), **overrides)
    with pytest.raises(ValueError):
        owner._target(req, "minimal")
