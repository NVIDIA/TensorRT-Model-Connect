# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference conditioning ABI and exact interleaved MRoPE coordinate mapping."""

import numpy as np
import pytest

from families.minimax_h3.checkpoint import numpy_state
from families.minimax_h3 import text_encoder_builder as text


def test_reference_conditioner_shapes_do_not_use_text_only_position_layout():
    shapes = text.conditioner_input_shapes(8192, reference=True)
    assert shapes["position_ids"] == [(3, 1), (3, 128), (3, 8192)]
    assert shapes["input_ids"] == [(1,), (128,), (8192,)]
    assert shapes["visual_mask"] == [(1, 1), (128, 1), (8192, 1)]
    assert len(shapes) == 7
    for name in (
        "visual_features",
        "deepstack_features_0",
        "deepstack_features_1",
        "deepstack_features_2",
    ):
        assert shapes[name] == [(1, 5120), (128, 5120), (8192, 5120)]


def test_text_only_conditioner_retains_its_two_inputs():
    assert text.conditioner_input_shapes(37) == {
        "input_ids": [(1,), (37,), (37,)],
        "position_ids": [(1,), (37,), (37,)],
    }


@pytest.mark.parametrize(
    "length,reference", [(0, False), (-1, True), (True, True), (8.0, False), (8, "yes")]
)
def test_conditioner_rejects_invalid_bounds_before_cuda(length, reference):
    with pytest.raises(ValueError):
        text.build_text_encoder_engine({}, sequence_length=length, reference=reference)


def test_qwen_interleaved_mrope_is_not_chunked_h3_dit_rope():
    axes = text.mrope_axes()
    assert axes.dtype == np.int32 and axes.shape == (64,)
    np.testing.assert_array_equal(np.bincount(axes), [24, 20, 20])
    np.testing.assert_array_equal(axes[:60], np.tile([0, 1, 2], 20))
    np.testing.assert_array_equal(axes[60:], [0, 0, 0, 0])
    # Independent transcription of the upstream overwrite operations: each
    # coordinate deliberately differs, so a one-axis cache cannot pass.
    positions = np.arange(3 * 7, dtype=np.float32).reshape(3, 1, 7)
    inverse = np.arange(1, 65, dtype=np.float32)
    frequencies = positions[..., None] * inverse
    expected = frequencies[0].copy()
    expected[..., 1:60:3] = frequencies[1, ..., 1:60:3]
    expected[..., 2:60:3] = frequencies[2, ..., 2:60:3]
    actual = positions[axes, 0].T * inverse
    np.testing.assert_array_equal(actual, expected[0])


@pytest.mark.gpu
@pytest.mark.trt
def test_reference_language_injects_visual_features_and_deepstack_before_hidden_state_boundary(
    monkeypatch, record_property
):
    import json
    import tensorrt as trt
    import torch
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    # The original stack is deliberately one layer longer: hidden_states[4]
    # must be a pre-final-norm state, just like H3's hidden_states[50] of 64.
    values = dict(
        HIDDEN_SIZE=16,
        NUM_LAYERS=4,
        NUM_HEADS=2,
        NUM_KV_HEADS=1,
        HEAD_DIM=8,
        INTERMEDIATE_SIZE=32,
        VOCAB_SIZE=128,
        MROPE_SECTION=(2, 1, 1),
    )
    for name, value in values.items():
        monkeypatch.setattr(text, name, value)
    torch.manual_seed(491)
    config = Qwen3VLTextConfig(
        hidden_size=16,
        num_hidden_layers=5,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        intermediate_size=32,
        vocab_size=128,
        use_cache=False,
        rope_parameters=dict(
            rope_type="default",
            rope_theta=text.ROPE_THETA,
            mrope_section=[2, 1, 1],
            mrope_interleaved=True,
        ),
    )
    config._attn_implementation = "sdpa"
    original = Qwen3VLTextModel(config).eval()
    with torch.no_grad():
        for name, value in original.named_parameters():
            if name.endswith("norm.weight"):
                value.copy_(torch.linspace(0.6, 1.4, value.numel()).reshape_as(value))
    rotary_frequency = original.rotary_emb.inv_freq.clone()
    original.to(dtype=torch.bfloat16)
    original.rotary_emb.inv_freq = rotary_frequency
    state = original.state_dict()
    prefix = "model.language_model."
    weights = numpy_state({name: state[name[len(prefix) :]] for name in text.checkpoint_keys()})
    plan = text.build_text_encoder_engine(
        weights, sequence_length=16, reference=True, workspace_bytes=1 << 30
    )
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None and engine.num_io_tensors == 8
    context = engine.create_execution_context()
    assert context is not None
    original.cuda()
    stream = torch.cuda.current_stream()
    results = []
    with torch.no_grad():
        for rows in (4, 7, 11):
            ids = torch.randint(0, 128, (rows,), dtype=torch.int32, device="cuda")
            positions = torch.arange(rows, dtype=torch.int32, device="cuda").expand(3, -1).clone()
            # Nonidentical T/H/W coordinates expose a one-axis RoPE shortcut.
            positions[1] *= 3
            positions[2] *= 7
            mask = torch.zeros(rows, 1, device="cuda", dtype=torch.bool)
            mask[1:3] = True
            visual = torch.randn(rows, 16, device="cuda", dtype=torch.bfloat16)
            deepstack = [
                torch.randn(rows, 16, device="cuda", dtype=torch.bfloat16) * 0.03 for _ in range(3)
            ]
            for value in deepstack:
                value[~mask[:, 0]] = 127
            embeds = original.embed_tokens(ids.long()[None])
            embeds = torch.where(mask[None], visual[None], embeds)
            expected = original(
                inputs_embeds=embeds,
                position_ids=positions.long()[:, None],
                attention_mask=torch.ones(1, rows, device="cuda", dtype=torch.long),
                visual_pos_masks=mask[:, 0][None],
                deepstack_visual_embeds=[value[mask[:, 0]] for value in deepstack],
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            ).hidden_states[4][0]
            inputs = dict(
                input_ids=ids,
                position_ids=positions,
                visual_mask=mask,
                visual_features=visual,
                **{f"deepstack_features_{i}": value for i, value in enumerate(deepstack)},
            )
            for name, value in inputs.items():
                assert context.set_input_shape(name, tuple(value.shape))
                assert context.set_tensor_address(name, value.data_ptr())
            actual = torch.empty((rows, 16), device="cuda", dtype=torch.float32)
            assert tuple(context.get_tensor_shape("encoder_hidden_states")) == (rows, 16)
            assert context.set_tensor_address("encoder_hidden_states", actual.data_ptr())
            assert context.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
            error = (actual - expected.float()).abs()
            results.append(dict(rows=rows, max_abs_error=error.max().item()))
            torch.testing.assert_close(actual, expected.float(), rtol=0.016, atol=1e-5)
    record_property("reference_language_comparisons", json.dumps(results, allow_nan=False))
