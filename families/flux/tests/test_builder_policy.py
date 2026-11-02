# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused checks for the family-owned FLUX TensorRT build policy."""

from pathlib import Path

import pytest


FAMILY_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    ("builder", "config_count"),
    (
        ("clip_encoder_builder.py", 1),
        ("t5_encoder_builder.py", 2),
        ("mistral_encoder_builder.py", 1),
        ("flux_vae_builder.py", 1),
        ("flux_dit_builder.py", 2),
        ("flux_dit_tp_builder.py", 1),
        ("flux_dit_cp_builder.py", 1),
        ("flux2_dit_builder.py", 1),
        ("flux2_dit_tp_builder.py", 1),
    ),
)
def test_every_flux_builder_config_uses_optimization_level_one(
    builder: str, config_count: int
) -> None:
    source = (FAMILY_ROOT / builder).read_text(encoding="utf-8")

    assert source.count("config = builder.create_builder_config()") == config_count
    assert source.count("config.builder_optimization_level = 1") == config_count


@pytest.mark.parametrize("builder", ("flux2_dit_builder.py", "flux2_dit_tp_builder.py"))
def test_flux2_fp8_attention_keeps_the_linear_quantize_outside(builder: str) -> None:
    source = (FAMILY_ROOT / builder).read_text(encoding="utf-8")

    assert source.count("graph_ops.keep_quantize_out_of_attention(network, rows)") == 1
    assert (
        "    if _FP8_MODE:\n        graph_ops.keep_quantize_out_of_attention(network, rows)\n"
    ) in source
    # A build route applies to the whole DiT; "-quantize_ppg=off" dropped every FP8 GEMM.
    assert "build_route" not in source


class _FakeNetwork:
    """Stands in for INetworkDefinition; records debug marks only."""

    def __init__(self) -> None:
        self.debug_tensors: list[object] = []

    def mark_debug(self, tensor: object) -> bool:
        self.debug_tensors.append(tensor)
        return True


def test_attention_rows_are_marked_as_a_debug_tensor() -> None:
    pytest.importorskip("tensorrt")
    from families.flux import graph_ops

    network, rows = _FakeNetwork(), object()

    assert graph_ops.keep_quantize_out_of_attention(network, rows) is rows
    assert network.debug_tensors == [rows]


@pytest.mark.trt
def test_fp8_linear_after_attention_builds_and_keeps_fp8_weights() -> None:
    """The FLUX.2 DiT pattern TensorRT 11.4 fails to build (issue #1920): an
    unquantized IAttention whose output feeds an FP8 linear. It must build with
    the attention rows marked, without a build route, and keep the FP8 weights;
    the earlier ``-quantize_ppg=off`` route folded them into FP16."""
    trt = pytest.importorskip("tensorrt")
    import numpy as np

    from families.flux import graph_ops

    heads, head_dim, seq = 16, 128, 256
    width = heads * head_dim
    builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
    config = builder.create_builder_config()
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    keep_alive = []

    def scalar(value: float):
        weights = np.array(value, dtype=np.float32)
        keep_alive.append(weights)
        return network.add_constant((), trt.Weights(weights)).get_output(0)

    query, key, value = (
        network.add_input(name, trt.float16, (1, heads, seq, head_dim)) for name in ("q", "k", "v")
    )
    attention = network.add_attention(
        query, key, value, trt.AttentionNormalizationOp.SOFTMAX, False
    )
    rows = network.add_shuffle(attention.get_output(0))
    rows.first_transpose = (0, 2, 1, 3)
    rows.reshape_dims = (seq, width)
    context = graph_ops.keep_quantize_out_of_attention(network, rows.get_output(0))
    activation_scale = scalar(0.05)
    quantized = network.add_quantize(context, activation_scale, trt.DataType.FP8)
    activation = network.add_dequantize(quantized.get_output(0), activation_scale, trt.float16)
    codes = np.full((width, width), 0x38, dtype=np.uint8)  # E4M3 1.0
    keep_alive.append(codes)
    weight = network.add_constant(
        (width, width), trt.Weights(trt.DataType.FP8, codes.ctypes.data, codes.size)
    )
    weight_dq = network.add_dequantize(weight.get_output(0), scalar(0.01), trt.float16)
    output = network.add_matrix_multiply(
        activation.get_output(0),
        trt.MatrixOperation.NONE,
        weight_dq.get_output(0),
        trt.MatrixOperation.TRANSPOSE,
    )
    network.mark_output(output.get_output(0))

    plan = builder.build_serialized_network(network, config)

    assert plan is not None
    # The FP8 weight takes width * width bytes; folded into FP16 it alone would
    # take twice that.
    assert plan.nbytes < 2 * width * width


def test_preprocessor_keeps_core_weights_required() -> None:
    source = (FAMILY_ROOT / "runtime/diffusion_helpers.cpp").read_text(encoding="utf-8")

    for key in (
        "patch_embedding.weight",
        "condition_embedder.time_embedding.0.weight",
        "condition_embedder.time_embedding.2.weight",
        "context_embedder.weight",
    ):
        call = "\n".join(source.split(f'"{key}"', maxsplit=1)[0].splitlines()[-2:])
        assert "load_preprocessor_floats" in call
        assert "load_optional_preprocessor_floats" not in call
