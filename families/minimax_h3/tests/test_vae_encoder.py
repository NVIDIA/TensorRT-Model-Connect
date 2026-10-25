# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VisualVAE tile contracts and an independent upstream FP32 execution oracle.

The GPU test uses all six encoder levels and both temporal strides, with
reduced channels only. It does not qualify trained weights or Ref2VA video.
"""

from dataclasses import replace
from itertools import combinations
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from families.minimax_h3.vae_encoder_builder import (
    VideoEncoderConfig,
    _validate_weights,
    build_vae_tile_encoder_engine,
    checkpoint_keys,
    checkpoint_shapes,
)
from families.minimax_h3.tests.vae_encoder_localize import (
    compare_convolution_replays,
    division_silu,
    fixed_statistics_affine,
    self_computed_group_norm,
    upstream_hook_path,
)


TINY = replace(
    VideoEncoderConfig(),
    block_out_channels=(4, 8, 8, 12, 12, 16),
    latent_channels=2,
    norm_num_groups=2,
)


def test_diagnostic_downsample_hooks_observe_the_same_padding_boundary():
    assert (
        upstream_hook_path("encoder.down_blocks.0.downsamplers.0.conv")
        == "encoder.down_blocks.0.downsamplers.0"
    )
    for name in ("encoder.conv_in", "encoder.down_blocks.0.resnets.0.conv1", "encoder.norm_out"):
        assert upstream_hook_path(name) == name


def test_diagnostic_activation_uses_reference_division_order_without_a_sigmoid():
    import tensorrt as trt
    from families.minimax_h3 import graph_ops

    operations = []

    def tensor(value, dtype=trt.float32):
        return SimpleNamespace(value=value, shape=value.shape, dtype=dtype)

    def layer(value):
        return SimpleNamespace(get_output=lambda index: tensor(value))

    class Network:
        def add_unary(self, value, operation):
            operations.append(operation)
            assert operation in (trt.UnaryOperation.NEG, trt.UnaryOperation.EXP)
            return layer(
                -value.value if operation == trt.UnaryOperation.NEG else np.exp(value.value)
            )

        def add_constant(self, shape, value):
            operations.append(("one", shape))
            return layer(value)

        def add_elementwise(self, left, right, operation):
            operations.append(operation)
            assert operation in (trt.ElementWiseOperation.SUM, trt.ElementWiseOperation.DIV)
            return layer(
                left.value + right.value
                if operation == trt.ElementWiseOperation.SUM
                else left.value / right.value
            )

    network = Network()
    values = np.array([[-20, -1, -0.0, 0.0, 1, 20]], np.float32)
    try:
        actual = division_silu(network, tensor(values)).value
        np.testing.assert_array_equal(actual, values / (np.float32(1) + np.exp(-values)))
        assert operations == [
            trt.UnaryOperation.NEG,
            trt.UnaryOperation.EXP,
            ("one", (1, 1)),
            trt.ElementWiseOperation.SUM,
            trt.ElementWiseOperation.DIV,
        ]
        with pytest.raises(ValueError, match="FP32"):
            division_silu(network, tensor(values, trt.float16))
    finally:
        graph_ops.release_weight_buffers(network)


@pytest.mark.parametrize("mode", ["centered", "reparameterized"])
def test_oracle_assisted_affine_replay_preserves_its_declared_order(mode):
    import tensorrt as trt

    operations = []

    def tensor(value, dtype=trt.float32):
        return SimpleNamespace(value=value, dtype=dtype)

    class Network:
        def add_elementwise(self, left, right, kind):
            operations.append(kind)
            if kind == trt.ElementWiseOperation.PROD:
                value = left.value * right.value
            elif kind == trt.ElementWiseOperation.SUM:
                value = left.value + right.value
            else:
                assert kind == trt.ElementWiseOperation.SUB
                value = left.value - right.value
            return SimpleNamespace(get_output=lambda index: tensor(value))

    value = np.arange(8, dtype=np.float32).reshape(1, 1, 2, 2, 1, 2)
    mean = np.array([0.125, -0.25], np.float32).reshape(1, 1, 2, 1, 1, 1)
    rstd = np.array([0.75, 1.25], np.float32).reshape(1, 1, 2, 1, 1, 1)
    gamma = np.array([0.2, 0.3, 0.4, 0.5], np.float32).reshape(1, 1, 2, 2, 1, 1)
    beta = np.array([-0.1, 0.1, -0.2, 0.2], np.float32).reshape(1, 1, 2, 2, 1, 1)
    network = Network()
    arguments = [tensor(item) for item in (value, mean, rstd, gamma, beta)]
    actual = fixed_statistics_affine(network, *arguments, mode=mode).value
    if mode == "centered":
        expected = ((value - mean) * rstd) * gamma + beta
        order = [
            trt.ElementWiseOperation.SUB,
            trt.ElementWiseOperation.PROD,
            trt.ElementWiseOperation.PROD,
            trt.ElementWiseOperation.SUM,
        ]
    else:
        scale = rstd * gamma
        expected = value * scale + (beta - scale * mean)
        order = [
            trt.ElementWiseOperation.PROD,
            trt.ElementWiseOperation.PROD,
            trt.ElementWiseOperation.SUB,
            trt.ElementWiseOperation.PROD,
            trt.ElementWiseOperation.SUM,
        ]
    np.testing.assert_array_equal(actual, expected)
    assert operations == order
    with pytest.raises(ValueError, match="Unknown"):
        fixed_statistics_affine(network, *arguments, mode="not-a-replay")
    with pytest.raises(ValueError, match="FP32"):
        fixed_statistics_affine(network, tensor(value, trt.float16), *arguments[1:], mode=mode)


@pytest.mark.parametrize("mode", ["centered", "reparameterized"])
def test_self_computed_groupnorm_has_no_oracle_input_and_keeps_frames_isolated(mode):
    import tensorrt as trt
    from families.minimax_h3 import graph_ops

    reductions = []

    def tensor(value, dtype=trt.float32):
        return SimpleNamespace(value=value, shape=value.shape, dtype=dtype)

    def layer(value):
        return SimpleNamespace(get_output=lambda index: tensor(value))

    class Shuffle:
        def __init__(self, value):
            self.value = value.value
            self.first_transpose = self.second_transpose = self.reshape_dims = None

        def get_output(self, index):
            value = self.value
            if self.first_transpose is not None:
                # The binding stores MAX_DIMS entries; a shuffle consumes only
                # the input rank, just like the real TensorRT layer.
                value = value.transpose(tuple(self.first_transpose)[: value.ndim])
            if self.reshape_dims is not None:
                value = value.reshape(
                    tuple(
                        value.shape[index] if size == 0 else size
                        for index, size in enumerate(self.reshape_dims)
                    )
                )
            if self.second_transpose is not None:
                value = value.transpose(tuple(self.second_transpose)[: value.ndim])
            return tensor(value)

    class Network:
        def add_input(self, *arguments):
            pytest.fail("A self-computed normalization must not add reference statistics inputs")

        def add_shuffle(self, value):
            return Shuffle(value)

        def add_reduce(self, value, kind, axes, keep):
            assert kind == trt.ReduceOperation.AVG and keep
            reductions.append(axes)
            selected = tuple(index for index in range(value.value.ndim) if axes & (1 << index))
            return layer(value.value.mean(axis=selected, keepdims=True, dtype=np.float32))

        def add_unary(self, value, kind):
            assert kind in (trt.UnaryOperation.SQRT, trt.UnaryOperation.RECIP)
            return layer(
                np.sqrt(value.value)
                if kind == trt.UnaryOperation.SQRT
                else np.reciprocal(value.value)
            )

        def add_constant(self, shape, value):
            return layer(value)

        def add_elementwise(self, left, right, kind):
            functions = {
                trt.ElementWiseOperation.PROD: np.multiply,
                trt.ElementWiseOperation.SUB: np.subtract,
                trt.ElementWiseOperation.SUM: np.add,
            }
            return layer(functions[kind](left.value, right.value))

    network = Network()
    value = np.arange(48, dtype=np.float32).reshape(1, 4, 3, 2, 2)
    weights = {
        "norm.weight": np.array([0.5, 0.7, 0.9, 1.1], np.float32),
        "norm.bias": np.array([-0.2, 0.1, 0.2, -0.1], np.float32),
    }
    grouped = value.transpose(0, 2, 1, 3, 4).reshape(1, 3, 2, 2, 2, 2)
    mean = grouped.mean(axis=(3, 4, 5), keepdims=True, dtype=np.float32)
    variance = ((grouped - mean) ** 2).mean(axis=(3, 4, 5), keepdims=True, dtype=np.float32)
    rstd = np.reciprocal(np.sqrt(variance + np.float32(TINY.norm_eps)))
    gamma = weights["norm.weight"].reshape(1, 1, 2, 2, 1, 1)
    beta = weights["norm.bias"].reshape(1, 1, 2, 2, 1, 1)
    if mode == "centered":
        expected = ((grouped - mean) * rstd) * gamma + beta
    else:
        scale = rstd * gamma
        expected = grouped * scale + (beta - scale * mean)
    expected = expected.reshape(1, 3, 4, 2, 2).transpose(0, 2, 1, 3, 4)
    try:
        actual = self_computed_group_norm(
            network, tensor(value), weights, "norm", TINY, mode=mode
        ).value
        np.testing.assert_array_equal(actual, expected)
        assert reductions == [(1 << 3) | (1 << 4) | (1 << 5)] * 2
        changed = value.copy()
        changed[:, :, 1:] = changed[:, :, 1:] * 9 + 23
        altered = self_computed_group_norm(
            network, tensor(changed), weights, "norm", TINY, mode=mode
        ).value
        np.testing.assert_array_equal(actual[:, :, :1], altered[:, :, :1])
        with pytest.raises(ValueError, match="Unknown"):
            self_computed_group_norm(network, tensor(value), weights, "norm", TINY, mode="unknown")
        with pytest.raises(ValueError, match="FP32"):
            self_computed_group_norm(
                network, tensor(value, trt.float16), weights, "norm", TINY, mode=mode
            )
        with pytest.raises(ValueError, match="grouped channels"):
            self_computed_group_norm(
                network, tensor(value[:, :3]), weights, "norm", TINY, mode=mode
            )
    finally:
        graph_ops.release_weight_buffers(network)


@pytest.mark.parametrize(
    "flags",
    [
        list(selected)
        for count in (2, 3, 4)
        for selected in combinations(
            (
                "--isolate-convolution",
                "--compare-activation-order",
                "--compare-groupnorm-order",
                "--compare-computed-groupnorm",
            ),
            count,
        )
    ],
)
def test_independent_diagnostics_cannot_be_combined_or_create_outputs(
    monkeypatch, tmp_path, capsys, flags
):
    from families.minimax_h3.tests import vae_encoder_localize

    output = tmp_path / "not-created"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "diagnostic",
            "--checkpoint",
            "unused",
            "--case-directory",
            "unused",
            "--output",
            str(output),
            *flags,
        ],
    )
    with pytest.raises(SystemExit) as error:
        vae_encoder_localize.main()
    assert error.value.code == 2 and "Select one diagnostic" in capsys.readouterr().err
    assert not output.exists()


def test_same_input_replay_separates_inherited_drift_from_operator_error():
    values = {
        "native_input": np.array([1.001], np.float32),
        "upstream_input": np.array([1.0], np.float32),
        "native_output": np.array([2.002], np.float32),
        "upstream_output": np.array([2.0], np.float32),
        "native_on_native_input": np.array([2.002], np.float32),
        "native_on_upstream_input": np.array([2.0], np.float32),
        "upstream_on_native_input": np.array([2.002], np.float32),
        "upstream_on_upstream_input": np.array([2.0], np.float32),
    }
    report = compare_convolution_replays(values)
    for name in ("input_drift", "inherited_drift_through_upstream"):
        assert report[name]["mismatched_elements"] == 1
    for name in (
        "upstream_replay_control",
        "same_native_input",
        "same_upstream_input",
        "standalone_vs_instrumented_native",
    ):
        assert report[name]["mismatched_elements"] == 0
    values["native_on_upstream_input"] += np.float32(0.001)
    assert compare_convolution_replays(values)["same_upstream_input"]["mismatched_elements"] == 1


def test_same_input_replay_requires_complete_finite_fp32_matrix():
    names = (
        "native_input",
        "upstream_input",
        "native_output",
        "upstream_output",
        "native_on_native_input",
        "native_on_upstream_input",
        "upstream_on_native_input",
        "upstream_on_upstream_input",
    )
    complete = {name: np.array([1.0], np.float32) for name in names}
    for values in ({}, {**complete, "extra": np.zeros(1, np.float32)}):
        with pytest.raises(AssertionError, match="matrix"):
            compare_convolution_replays(values)
    for bad in (
        np.array([1.0], np.float64),
        np.array([float("nan")], np.float32),
        np.ones(2, np.float32),
    ):
        with pytest.raises(AssertionError):
            compare_convolution_replays({**complete, "native_output": bad})


def test_original_encoder_schema_has_exactly_118_fp32_tensors():
    shapes = checkpoint_shapes()
    assert len(shapes) == len(checkpoint_keys()) == 118
    assert shapes["encoder.conv_in.weight"] == (128, 3, 3, 3, 3)
    assert shapes["encoder.down_blocks.5.resnets.0.conv_shortcut.weight"] == (1024, 512, 1, 1, 1)
    assert shapes["encoder.conv_out.weight"] == (48, 1024, 3, 3, 3)
    assert shapes["quant_conv.weight"] == (48, 48, 1, 1, 1)
    assert not any(name.startswith(("decoder.", "post_quant_conv.")) for name in shapes)
    assert VideoEncoderConfig().spatial_ratio == 16
    assert VideoEncoderConfig().temporal_ratio == 4


@pytest.mark.parametrize(
    "change",
    [
        {"block_out_channels": ()},
        {"spatial_downsample_factors": (2,)},
        {"spatial_downsample_factors": (2, 2, 2, 2, 1, 1.0)},
        {"temporal_downsample_factors": (1, 1, 1, 1, 1, 3)},
        {"temporal_downsample_factors": (1, 2, 2, 1, 1, 2)},
        {"block_out_channels": (128, 256, 256, 512, 512, 1025)},
        {"norm_num_groups": True},
        {"latent_channels": 0},
        {"layers_per_block": 1.5},
        {"norm_eps": float("nan")},
        {"norm_eps": 0.0},
    ],
)
def test_encoder_config_rejects_unsupported_geometry(change):
    with pytest.raises(ValueError):
        checkpoint_shapes(replace(VideoEncoderConfig(), **change))


@pytest.mark.parametrize(
    "height,width,frames",
    [
        (31, 48, 1),
        (32, 47, 1),
        (16, 48, 1),
        (True, 48, 1),
        (32, 48, 0),
        (32, 48, 18),
        (32, 48, True),
    ],
)
def test_encoder_rejects_invalid_profiles_before_cuda(height, width, frames):
    with pytest.raises(ValueError, match="tile sizes|at most 17"):
        build_vae_tile_encoder_engine(
            {}, config=TINY, tile_height=height, tile_width=width, max_frames=frames
        )


def test_encoder_weights_reject_extra_missing_wrong_shape_and_downcast():
    weights = {name: np.zeros(shape, np.float32) for name, shape in checkpoint_shapes(TINY).items()}
    _validate_weights(weights, TINY)
    with pytest.raises(ValueError, match="exactly"):
        _validate_weights({**weights, "decoder.unexpected": np.zeros(1)}, TINY)
    key = "encoder.conv_in.weight"
    value = weights.pop(key)
    with pytest.raises(ValueError, match="exactly"):
        _validate_weights(weights, TINY)
    for bad in (value.astype(np.float16), value.reshape(-1)):
        with pytest.raises(ValueError, match="FP32 shape"):
            _validate_weights({**weights, key: bad}, TINY)


@pytest.mark.gpu
@pytest.mark.trt
def test_native_encoder_moments_match_upstream_fp32_and_preserve_causality(record_property):
    import tensorrt as trt
    import torch
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import MiniMaxH3VideoEncoder3d

    # Freeze the oracle policy before execution, not in response to a mismatch.
    # The tolerance is PyTorch's default FP32 assert_close tolerance.
    torch.manual_seed(203)
    with torch.backends.cudnn.flags(allow_tf32=False), torch.no_grad():
        encoder = MiniMaxH3VideoEncoder3d(
            in_channels=TINY.in_channels,
            out_channels=2 * TINY.latent_channels,
            block_out_channels=TINY.block_out_channels,
            layers_per_block=TINY.layers_per_block,
            spatial_downsample_factors=TINY.spatial_downsample_factors,
            temporal_downsample_factors=TINY.temporal_downsample_factors,
            norm_num_groups=TINY.norm_num_groups,
            norm_eps=TINY.norm_eps,
        ).eval()
        quant = torch.nn.Conv3d(2 * TINY.latent_channels, 2 * TINY.latent_channels, 1).eval()
        weights = {f"encoder.{name}": value.numpy() for name, value in encoder.state_dict().items()}
        weights.update(
            {f"quant_conv.{name}": value.numpy() for name, value in quant.state_dict().items()}
        )
        assert {name: tuple(value.shape) for name, value in weights.items()} == checkpoint_shapes(
            TINY
        )
        plan = build_vae_tile_encoder_engine(
            weights, config=TINY, tile_height=32, tile_width=48, workspace_bytes=1 << 30
        )
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(plan)
        assert engine is not None
        assert engine.num_io_tensors == 2
        assert all(engine.get_tensor_dtype(name) == trt.float32 for name in ("pixels", "moments"))
        context = engine.create_execution_context()
        assert context is not None
        encoder, quant = encoder.cuda(), quant.cuda()
        stream = torch.cuda.current_stream()

        def infer(pixels):
            assert pixels.is_contiguous() and pixels.dtype == torch.float32
            assert context.set_input_shape("pixels", tuple(pixels.shape))
            shape = tuple(context.get_tensor_shape("moments"))
            output = torch.empty(shape, dtype=torch.float32, device="cuda")
            assert context.set_tensor_address("pixels", pixels.data_ptr())
            assert context.set_tensor_address("moments", output.data_ptr())
            assert context.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
            return output

        pixels = torch.randn(1, 3, 17, 32, 48, device="cuda")
        results = []
        for frames in (1, 2, 5, 17):
            value = pixels[:, :, :frames].contiguous()
            expected = quant(encoder(value))
            actual = infer(value)
            assert actual.shape == (1, 4, (frames + 3) // 4, 2, 3)
            error = (actual - expected).abs()
            results.append(
                dict(
                    frames=frames,
                    max_abs_error=error.max().item(),
                    mean_abs_error=error.mean().item(),
                )
            )
            torch.testing.assert_close(actual, expected, rtol=1.3e-6, atol=1e-5)
        original = infer(pixels)
        changed = pixels.clone()
        changed[:, :, 1:] = changed[:, :, 1:] * 9 + 23
        altered = infer(changed)
        torch.testing.assert_close(original[:, :, :1], altered[:, :, :1], rtol=1.3e-6, atol=1e-5)
        record_property("fp32_moments_comparison", json.dumps(results, allow_nan=False))
