# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan lifetime and exact bundle ABI tests without constructing GPU engines."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_model_connect import BuildRequest
from tensorrt_model_connect.bundle_writer import BundleWriter
from families.minimax_h3 import model
from families.minimax_h3.config import SOL_ENGINE_1344X768_124F


@pytest.fixture
def components(monkeypatch, tmp_path):
    from families.minimax_h3 import adaln_builder, dit_builder, text_encoder_builder, vae_builder

    root = tmp_path / "checkpoint"
    for name in ("transformer", "text_encoder", "vae", "audio_vae", "tokenizer"):
        (root / name).mkdir(parents=True)
    (root / "transformer/config.json").write_text(
        json.dumps(
            dict(
                hidden_size=5376,
                num_layers=50,
                num_attention_heads=56,
                attention_head_dim=128,
                ffn_dim=14336,
            )
        )
    )
    (root / "tokenizer/tokenizer.json").write_bytes(b'{"fixture":"tokenizer"}')
    state = SimpleNamespace(
        order=[], loaded=[], built=[], emitted=[], retired=set(), streaming=False
    )

    class Plan(bytes):
        def __new__(cls, name):
            instance = super().__new__(cls, (name + "-exact-plan-payload").encode())
            instance.name = name
            return instance

        def __del__(self):
            state.retired.add(self.name)

    def load(directory, selected_keys):
        name = state.order[len(state.loaded)]
        if state.streaming:
            assert state.retired == set(state.loaded), "completed plan retained into the next load"
            assert state.emitted == state.loaded, "next weights loaded before staging prior plan"
        else:
            assert not state.retired, "legacy retained-return mode changed"
        assert selected_keys
        assert Path(directory).name == (
            "text_encoder"
            if name == "text_encoder"
            else "vae"
            if name == "vae_decoder"
            else "transformer"
        )
        state.loaded.append(name)
        return {"fixture_name": name}

    def builder(name):
        def build(weights, *args, **kwargs):
            assert weights == {"fixture_name": name}
            assert kwargs["consume_weights"] is True
            assert kwargs["workspace_bytes"] > 0
            state.built.append(name)
            weights.clear()
            return Plan(name)

        return build

    monkeypatch.setattr(model, "load_selected_component_state_dict", load)
    monkeypatch.setattr(model, "numpy_state", lambda state: state)
    monkeypatch.setattr(model, "validate_component_key_partition", lambda *_args: None)
    for module, function, name in (
        (text_encoder_builder, "build_text_encoder_engine", "text_encoder"),
        (adaln_builder, "build_adaln_precompute_engine", "adaln_precompute"),
        (dit_builder, "build_dit_engine", "denoiser"),
        (dit_builder, "build_dit_head_engine", "denoiser_head"),
        (dit_builder, "build_dit_tail_engine", "denoiser_tail"),
        (dit_builder, "build_dit_finish_engine", "denoiser_finish"),
        (vae_builder, "build_vae_tile_decoder_engine", "vae_decoder"),
    ):
        monkeypatch.setattr(module, function, builder(name))
    return root, state


@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
def test_component_plans_are_staged_and_released_before_next_weights(components, cache, streaming):
    root, state = components
    denoisers = ["denoiser_head", "denoiser_tail", "denoiser_finish"] if cache else ["denoiser"]
    state.order = ["text_encoder", "adaln_precompute", *denoisers, "vae_decoder"]
    state.streaming = streaming
    config = SimpleNamespace(raw=dict(first_block_cache=cache))
    owner = model._MiniMaxH3Model()
    weights = owner.load_weights(str(root), config)
    writer = BundleWriter(root.parent / "parts.bundle")

    def emit(name, payload):
        state.emitted.append(name)
        writer.add_bytes(name + ".plan", payload)

    result = owner.build_components(str(root), config, weights, on_plan=emit if streaming else None)
    assert state.loaded == state.built == state.order
    assert result["profile"].first_block_cache == cache
    assert result["tokenizer_json"] == b'{"fixture":"tokenizer"}'
    if streaming:
        assert state.retired == set(state.order)
        assert not any(name in result for name in state.order)
    else:
        assert not state.retired
        for name in state.order:
            assert result[name] == (name + "-exact-plan-payload").encode()
    writer.abort()


def legacy_runtime():
    # Freeze the existing minimal bundle ABI independently of the new staging
    # order. No engine math is supplied by these CPU plan doubles.
    return dict(
        height=768,
        width=1344,
        num_frames=124,
        fps=24,
        num_inference_steps=50,
        seed=0,
        first_block_cache=False,
        denoiser_cache_mode="monolithic",
        first_block_cache_threshold=0.025,
        text_rows=537,
        text_rows_min=1,
        text_rows_opt=128,
        text_rows_max=537,
        audio_rows=414,
        video_rows=37296,
        padded_sequence_length=38247,
        max_timestep_count=4,
        context_parallel_size=1,
        vae_tile_batch=28,
        vae_tile_size=256,
        vae_tile_overlap=64,
    )


def test_public_build_streams_identical_bundle_bytes(components):
    root, state = components
    state.order = ["text_encoder", "adaln_precompute", "denoiser", "vae_decoder"]
    state.streaming = True
    output = root.parent / "streamed.bundle"
    writer = BundleWriter(output)
    original_add = writer.add_bytes

    def add(name, payload):
        if name.endswith(".plan"):
            component = {
                "text_encoder.plan": "text_encoder",
                "adaln.plan": "adaln_precompute",
                "denoiser.plan": "denoiser",
                "vae.plan": "vae_decoder",
            }[name]
            state.emitted.append(component)
        original_add(name, payload)

    writer.add_bytes = add
    request = BuildRequest(
        model_dir=root,
        output_path=output,
        family="minimax_h3",
        task="image_generation",
        precision="bf16",
    )
    model.build(request, writer)
    writer.finish()
    assert state.retired == set(state.order)
    assert state.loaded == state.emitted == state.order
    expected = root.parent / "legacy.bundle"
    legacy = BundleWriter(expected)
    legacy.set_header(family="minimax_h3", task="image_generation", backend="trt")
    for filename, name in (
        ("text_encoder.plan", "text_encoder"),
        ("adaln.plan", "adaln_precompute"),
        ("denoiser.plan", "denoiser"),
        ("vae.plan", "vae_decoder"),
    ):
        legacy.add_bytes(filename, (name + "-exact-plan-payload").encode())
    legacy.add_bytes("tokenizer.json", b'{"fixture":"tokenizer"}')
    legacy.add_json("runtime.json", legacy_runtime())
    legacy.finish()
    assert output.read_bytes() == expected.read_bytes()


def test_staging_failure_does_not_build_later_components_or_publish(components):
    root, state = components
    state.order = ["text_encoder", "adaln_precompute", "denoiser", "vae_decoder"]
    state.streaming = True
    output = root.parent / "existing.bundle"
    output.write_bytes(b"existing-published-bundle")
    writer = BundleWriter(output)
    add = writer.add_bytes

    def fail(name, payload):
        if name == "adaln.plan":
            raise OSError("fixture staging failure")
        state.emitted.append("text_encoder")
        add(name, payload)

    writer.add_bytes = fail
    request = BuildRequest(
        model_dir=root,
        output_path=output,
        family="minimax_h3",
        task="image_generation",
        precision="bf16",
    )
    with pytest.raises(OSError, match="fixture staging failure"):
        model.build(request, writer)
    writer.abort()
    assert state.loaded == state.built == ["text_encoder", "adaln_precompute"]
    assert output.read_bytes() == b"existing-published-bundle"
    assert not list(output.parent.glob(".existing.bundle.sections.*"))


def test_legacy_runtime_fixture_matches_fixed_profile():
    profile = SOL_ENGINE_1344X768_124F
    runtime = legacy_runtime()
    assert runtime["text_rows"] == profile.text_rows
    assert runtime["video_rows"] == profile.video_rows
    assert runtime["audio_rows"] == profile.audio_rows
    assert runtime["padded_sequence_length"] == profile.padded_sequence_length
