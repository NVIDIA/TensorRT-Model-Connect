# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The atom stack's sliding window, row by row, against the transformers reference (GPU, TensorRT).

Three random atom layers of the released shape (width 128, 4 heads, window radius 64) run as one TensorRT plan built
from the family's layer code and as the reference ``EsmFold2AtomEncoder`` layers with its own window mask and 3D RoPE,
over 224 atoms (7 query blocks; the last 9 are padding). Every real atom's row must agree; float32 plans whose window
radius is one smaller or one larger must not.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from families.esmfold2.config import parse_config
from families.esmfold2.tests.test_support import released_config
from tools.variant_coverage import covers_variant

ATOMS = 224
REAL_ATOMS = 215
LAYERS = 3
# Relative L2 error of one atom's output row, measured on a B300: float32 plan against the float32 reference at most
# 8.4e-8; bf16 plan against the float32 reference on the same bf16-rounded weights at most 0.0054. A float32 plan with
# the radius one smaller or one larger moves rows by up to 6.4e-3 (more than 170 of the 215 rows above 1e-3), which
# bf16 rounding would hide, so the neighbors are checked in float32.
ROW_BOUND = {"fp32": 1.0e-5, "bf16": 0.012}


def _reference(precision: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    import torch
    from transformers.models.esmfold2.configuration_esmfold2 import EsmFold2Config
    from transformers.models.esmfold2.modeling_esmfold2 import EsmFold2AtomEncoder

    config = EsmFold2Config()
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        encoder = EsmFold2AtomEncoder(config, config.atom_encoder).cuda().float().eval()
    if precision == "bf16":
        with torch.no_grad():
            for parameter in encoder.parameters():
                parameter.copy_(parameter.to(torch.bfloat16).float())
    generator = np.random.default_rng(0)
    inputs = {
        "hidden": generator.standard_normal((1, ATOMS, 128)).astype(np.float32),
        "ref_pos": (5.0 * generator.standard_normal((1, ATOMS, 3))).astype(np.float32),
        "ref_space_uid": (np.arange(ATOMS, dtype=np.int32) // 8)[None, :],
        "atom_mask": (np.arange(ATOMS) < REAL_ATOMS).astype(np.float32)[None, :],
    }
    if precision == "bf16":
        inputs["hidden"] = torch.from_numpy(inputs["hidden"]).to(torch.bfloat16).float().numpy()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.no_grad():
            hidden = torch.from_numpy(inputs["hidden"]).cuda()
            ref_pos = torch.from_numpy(inputs["ref_pos"]).cuda()
            uid = torch.from_numpy(inputs["ref_space_uid"]).long().cuda()
            mask = torch.from_numpy(inputs["atom_mask"]).cuda() > 0
            window = encoder.build_attention_mask(mask, hidden)
            rope = encoder.rotary_emb(ref_pos, uid, dtype=torch.float32)
            state = hidden
            for layer in encoder.layers:
                state = layer(state, hidden, window, rope)
            expected = state.cpu().numpy()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    weights = {name: value.detach().float().cpu().numpy() for name, value in encoder.state_dict().items()}
    return weights, inputs, expected


def _native(weights: dict[str, np.ndarray], inputs: dict[str, np.ndarray], precision: str, radius: int) -> np.ndarray:
    import tensorrt as trt
    import torch

    from families.esmfold2 import layers
    from families.esmfold2.graph import build_plan

    stack = parse_config(released_config()).trunk_atoms
    compute = {"fp32": trt.float32, "bf16": trt.bfloat16}[precision]

    def define(net: Any) -> None:
        hidden = net.network.add_input("hidden", trt.float32, (1, -1, 128))
        ref_pos = net.network.add_input("ref_pos", trt.float32, (1, -1, 3))
        uid = net.network.add_input("ref_space_uid", trt.int32, (1, -1))
        mask = net.network.add_input("atom_mask", trt.float32, (1, -1))
        conditioning = net.cast(hidden, compute)
        rope = layers.atom_rope(net, ref_pos, uid, stack)
        bias = layers.atom_window_bias(net, mask, radius)
        state = conditioning
        for index in range(LAYERS):
            state = layers.atom_layer(net, state, conditioning, bias, rope, f"layers.{index}", stack, radius)
        net.mark_output(state, "out", trt.float32)

    shapes = {"hidden": (128,), "ref_pos": (3,), "ref_space_uid": (), "atom_mask": ()}
    profile = {name: ((1, 32, *tail), (1, ATOMS, *tail), (1, ATOMS, *tail)) for name, tail in shapes.items()}
    plan = build_plan(trt, define, compute=compute, weight=lambda name: weights[name], profiles=[profile])
    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    keep = []
    for name, value in inputs.items():
        tensor = torch.from_numpy(np.ascontiguousarray(value)).cuda()
        keep.append(tensor)
        assert context.set_input_shape(name, tuple(tensor.shape))
        context.set_tensor_address(name, tensor.data_ptr())
    output = torch.empty((1, ATOMS, 128), dtype=torch.float32, device="cuda")
    assert tuple(context.get_tensor_shape("out")) == tuple(output.shape)
    context.set_tensor_address("out", output.data_ptr())
    stream = torch.cuda.current_stream()
    assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return output.cpu().numpy()


def _row_errors(native: np.ndarray, expected: np.ndarray) -> np.ndarray:
    assert native.shape == expected.shape == (1, ATOMS, 128)
    real = slice(0, REAL_ATOMS)
    difference = np.linalg.norm((native - expected)[0, real].astype(np.float64), axis=-1)
    return difference / np.maximum(np.linalg.norm(expected[0, real].astype(np.float64), axis=-1), 1e-30)


@pytest.mark.gpu
@pytest.mark.trt
def test_float32_window_matches_row_by_row_and_rejects_neighbors() -> None:
    weights, inputs, expected = _reference("fp32")
    radius = parse_config(released_config()).atom_window_radius
    assert radius == 64
    errors = _row_errors(_native(weights, inputs, "fp32", radius), expected)
    assert np.isfinite(errors).all() and errors.max() <= ROW_BOUND["fp32"], errors.max()
    for wrong in (radius - 1, radius + 1):
        wrong_errors = _row_errors(_native(weights, inputs, "fp32", wrong), expected)
        assert wrong_errors.max() > 10 * ROW_BOUND["fp32"], (wrong, wrong_errors.max())


@pytest.mark.gpu
@pytest.mark.trt
@covers_variant(precision="bf16")
def test_bf16_window_matches_row_by_row() -> None:
    weights, inputs, expected = _reference("bf16")
    radius = parse_config(released_config()).atom_window_radius
    errors = _row_errors(_native(weights, inputs, "bf16", radius), expected)
    assert np.isfinite(errors).all() and errors.max() <= ROW_BOUND["bf16"], errors.max()
