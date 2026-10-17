# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU graph-contract checks; these do not qualify a TensorRT engine."""

from dataclasses import replace

import numpy as np
import pytest

from families.minimax_h3 import dit_builder as dit
from families.minimax_h3.config import Ref2VAPackedProfile, SOL_ENGINE_1344X768_124F
from families.minimax_h3.packed_rows import PackedRows


class Optimization:
    def __init__(self):
        self.shapes = {}

    def set_shape(self, name, **shapes):
        self.shapes[name] = shapes


class Builder:
    def create_optimization_profile(self):
        self.optimization = Optimization()
        return self.optimization


class Config:
    def add_optimization_profile(self, optimization):
        self.optimization = optimization


def reference():
    return Ref2VAPackedProfile((8, 1024, 4096), (1, 2048, 8192), (0, 207, 414))


def test_reference_profile_covers_all_operands_without_the_text_only_cap():
    builder, config = Builder(), Config()
    target, ref = SOL_ENGINE_1344X768_124F, reference()
    dit._add_reference_profile(builder, config, target, ref)
    shapes = config.optimization.shapes
    assert set(shapes) == {
        "encoder_hidden_states",
        "video_hidden_states",
        "audio_hidden_states",
        "position_ids",
        "adaln_indices",
        "timestep_indices",
        "packed_indices",
    }
    assert shapes["encoder_hidden_states"]["max"] == (4096, 5120)
    assert shapes["video_hidden_states"]["max"] == (37296 + 8192, 96)
    assert shapes["audio_hidden_states"]["max"] == (414 + 414, 32)
    for i, point in enumerate(("min", "opt", "max")):
        total = (
            37296
            + 414
            + ref.text_rows[i]
            + ref.condition_video_rows[i]
            + ref.condition_audio_rows[i]
        )
        assert shapes["position_ids"][point] == (total, 3)
        for name in ("packed_indices", "adaln_indices", "timestep_indices"):
            assert shapes[name][point] == (total,)


def test_expanded_portrait_profile_enlarges_both_conditioning_streams(tmp_path):
    from tensorrt_model_connect import BuildRequest
    from families.minimax_h3 import reference_build

    request = BuildRequest(
        model_dir=tmp_path / "not-read",
        output_path=tmp_path / "not-built.bundle",
        family="minimax_h3",
        task=reference_build.TASK,
        precision="bf16",
    )
    _, _, target, ref = reference_build._target(request, "minimal", "six_portrait")
    builder, config = Builder(), Config()
    dit._add_reference_profile(builder, config, target, ref)
    shapes = config.optimization.shapes
    assert shapes["encoder_hidden_states"]["max"] == (65536, 5120)
    assert shapes["video_hidden_states"]["max"] == (37296 + 49152, 96)
    assert shapes["audio_hidden_states"]["max"] == (414, 32)
    assert shapes["position_ids"]["max"] == (152398, 3)
    for name in ("packed_indices", "adaln_indices", "timestep_indices"):
        assert shapes[name]["max"] == (152398,)


@pytest.mark.parametrize(
    "field,values",
    [
        ("text_rows", (0, 1, 2)),
        ("text_rows", (1, 3, 2)),
        ("text_rows", (1, 2)),
        ("text_rows", (1, 2, True)),
        ("condition_video_rows", (0, 1, 2)),
        ("condition_audio_rows", (-1, 0, 1)),
        ("condition_video_rows", (1, 2, 2**31)),
    ],
)
def test_reference_bounds_reject_invalid_profiles_before_building(field, values):
    with pytest.raises(ValueError):
        replace(reference(), **{field: values}).validate(SOL_ENGINE_1344X768_124F)


def test_reference_path_cannot_enable_approximate_first_block_cache():
    with pytest.raises(ValueError, match="uncached"):
        reference().validate(replace(SOL_ENGINE_1344X768_124F, first_block_cache=True))


def test_packed_gather_matches_original_torch_index_copy_for_arbitrary_reference_order():
    import torch

    generator = np.random.default_rng(7)
    for text_rows, audio_rows, video_rows in ((3, 2, 9), (5, 11, 4), (19, 23, 31)):
        total = text_rows + audio_rows + video_rows
        indices = generator.permutation(total)
        mapping = PackedRows(
            indices[:text_rows],
            indices[text_rows : text_rows + audio_rows],
            indices[text_rows + audio_rows :],
        )
        text = torch.from_numpy(generator.normal(size=(1, text_rows, 13)).astype(np.float32))
        audio = torch.from_numpy(generator.normal(size=(1, audio_rows, 13)).astype(np.float32))
        video = torch.from_numpy(generator.normal(size=(1, video_rows, 13)).astype(np.float32))
        original = torch.zeros(1, total, 13)
        # Upstream index_copy uses torch.long; the native TensorRT ABI uses int32.
        original = original.index_copy(
            1, torch.tensor(mapping.text_indices, dtype=torch.long), text
        )
        original = original.index_copy(
            1, torch.tensor(mapping.video_indices, dtype=torch.long), video
        )
        original = original.index_copy(
            1, torch.tensor(mapping.audio_indices, dtype=torch.long), audio
        )
        actual = torch.cat((text, audio, video), dim=1).index_select(
            1, torch.tensor(mapping.packed_to_concat)
        )
        assert torch.equal(actual, original)


def test_native_projection_packing_uses_the_supplied_permutation(monkeypatch):
    text = np.array([[10.0], [11.0]], dtype=np.float32)
    audio = np.array([[20.0]], dtype=np.float32)
    video = np.array([[30.0], [31.0]], dtype=np.float32)
    mapping = PackedRows(np.array([0, 1]), np.array([3]), np.array([2, 4]))
    monkeypatch.setattr(dit, "_refine_text", lambda *args: text)
    monkeypatch.setattr(dit.op, "linear", lambda network, value, *args, **kwargs: value)
    monkeypatch.setattr(dit.op, "cast", lambda network, value, dtype: value)
    calls = []

    def gather(network, value, indices):
        calls.append(indices)
        return value[indices]

    monkeypatch.setattr(dit.op, "gather_rows", gather)

    class Concatenation:
        def get_output(self, index):
            return np.concatenate((text, audio, video))

    class Network:
        def add_concatenation(self, operands):
            assert len(operands) == 3
            return Concatenation()

    weights = {
        name: None
        for name in ("audio_proj_in.weight", "audio_proj_in.bias", "proj_in.weight", "proj_in.bias")
    }
    output = dit._packed_hidden(
        Network(),
        video,
        audio,
        text,
        weights,
        SOL_ENGINE_1344X768_124F,
        packed_indices=mapping.packed_to_concat,
    )
    np.testing.assert_array_equal(output[:, 0], [10, 11, 30, 20, 31])
    assert len(calls) == 1


def test_reference_heads_project_only_requested_target_rows(monkeypatch):
    hidden = np.arange(20, dtype=np.float32).reshape(10, 2)
    gathers = []

    def gather(network, value, indices):
        gathers.append(indices)
        return value[indices]

    class Output:
        pass

    projected = []

    def linear(network, value, *args, **kwargs):
        projected.append(value.copy())
        return Output()

    class Network:
        def mark_output(self, value):
            pass

    monkeypatch.setattr(dit.op, "gather_rows", gather)
    monkeypatch.setattr(dit.op, "linear", linear)
    monkeypatch.setattr(
        dit.op,
        "slice_rows_from_end",
        lambda *args, **kwargs: pytest.fail("reference rows cannot use the T2VA suffix assumption"),
    )
    weights = {
        name: None
        for name in (
            "proj_out.weight",
            "proj_out.bias",
            "audio_proj_out.weight",
            "audio_proj_out.bias",
        )
    }
    dit._mark_sliced_velocity_outputs(
        Network(),
        hidden,
        weights,
        SOL_ENGINE_1344X768_124F,
        target_video_indices=np.array([6, 8]),
        target_audio_indices=np.array([7, 9]),
    )
    np.testing.assert_array_equal(projected[0], hidden[[6, 8]])
    np.testing.assert_array_equal(projected[1], hidden[[7, 9]])


def test_indexed_heads_require_both_modality_index_operands():
    with pytest.raises(ValueError, match="both audio and video"):
        dit._mark_sliced_velocity_outputs(
            None, None, {}, SOL_ENGINE_1344X768_124F, target_video_indices=np.array([1])
        )


@pytest.mark.gpu
@pytest.mark.trt
def test_tiny_reference_graph_serializes_and_retains_dynamic_reference_shapes():
    import tensorrt as trt

    from .test_trt_builders import _weights

    target = replace(
        SOL_ENGINE_1344X768_124F,
        hidden_size=8,
        num_layers=1,
        num_refiner_layers=1,
        num_heads=4,
        head_dim=8,
        ffn_dim=16,
        video_in_channels=2,
        audio_in_channels=2,
        text_dim=8,
        timestep_input_dim=4,
        timestep_hidden_size=8,
        timestep_embed_dim=4,
        rope_freq_dim=1,
        video_rows=4,
        audio_rows=2,
        min_text_rows=1,
        opt_text_rows=2,
        text_rows=4,
        padded_sequence_length=10,
    )
    reference = Ref2VAPackedProfile((1, 2, 4), (1, 2, 3), (0, 1, 2))
    plan = dit.build_dit_engine(
        _weights(target), target, reference_profile=reference, workspace_bytes=1 << 30
    )
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    context = engine.create_execution_context()
    for text, condition_video, condition_audio in ((1, 1, 0), (2, 2, 1), (4, 3, 2)):
        video, audio = target.video_rows + condition_video, target.audio_rows + condition_audio
        total = text + video + audio
        for name, shape in {
            "encoder_hidden_states": (text, target.text_dim),
            "video_hidden_states": (video, target.video_patch_dim),
            "audio_hidden_states": (audio, target.audio_in_channels),
            "position_ids": (total, 3),
            "packed_indices": (total,),
            "adaln_indices": (total,),
            "timestep_indices": (total,),
        }.items():
            assert context.set_input_shape(name, shape), name
        assert tuple(context.get_tensor_shape("video_velocity")) == (4, 8)
        assert tuple(context.get_tensor_shape("audio_velocity")) == (2, 2)
