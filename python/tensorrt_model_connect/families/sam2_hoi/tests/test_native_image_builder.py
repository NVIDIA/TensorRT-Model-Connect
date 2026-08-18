# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tensorrt_model_connect.families.sam2_hoi import native_image_builder


def test_float32_fma_preserves_cuda_single_rounding_boundary():
    multiplier = np.float32(1.5852375)
    multiplicand = np.float32(1.5675687)
    addend = np.float32(-0.35551894)

    fused = native_image_builder._fused_multiply_add_float32(multiplier, multiplicand, addend)
    separately_rounded = np.float32(np.float32(multiplier * multiplicand) + addend)

    assert int(fused.view(np.uint32)) == 0x400848E7
    assert int(separately_rounded.view(np.uint32)) == 0x400848E8


def test_fixed_hiera_bicubic_matches_cuda_reference_bits():
    values = np.linspace(-0.2, 0.2, 49, dtype=np.float32).reshape(1, 1, 7, 7)

    resized = native_image_builder._hiera_bicubic_7x7_to_256x256(values)

    assert resized.shape == (1, 1, 256, 256)
    assert resized.dtype == np.float32
    assert hashlib.sha256(resized.tobytes()).hexdigest() == (
        "59d0acf883ae1e28b6373de6ce6af082955565c3ec30715fad00be88054517c0"
    )


def test_hiera_position_is_one_host_generated_fp32_constant(monkeypatch):
    prefix = native_image_builder._HIERA_PREFIX
    weights = {
        f"{prefix}.pos_embed": np.ones((1, 96, 7, 7), dtype=np.float32),
        f"{prefix}.pos_embed_window": np.full((1, 96, 8, 8), 2.0, dtype=np.float32),
    }
    resized = np.full((1, 96, 256, 256), 3.0, dtype=np.float32)
    output = object()
    calls = []
    monkeypatch.setattr(
        native_image_builder,
        "_hiera_bicubic_7x7_to_256x256",
        lambda actual: calls.append(("resize", actual)) or resized,
    )

    def fake_constant(network, shape, values, *, precision):
        calls.append(("constant", network, shape, values, precision))
        return output

    monkeypatch.setattr(native_image_builder.graph_ops, "add_constant", fake_constant)

    class FakeNetwork:
        def add_resize(self, _tensor):
            raise AssertionError("Hiera position must not use a runtime resize layer")

    network = FakeNetwork()
    result = native_image_builder._add_hiera_position(network, weights)

    assert result is output
    assert calls[0][0] == "resize"
    assert calls[0][1] is weights[f"{prefix}.pos_embed"]
    assert calls[1][0:3] == ("constant", network, (1, 96, 256, 256))
    np.testing.assert_array_equal(calls[1][3], np.full_like(resized, 5.0))
    assert calls[1][4] == "fp32"


def test_hiera_position_rejects_nonfixed_checkpoint_shapes(monkeypatch):
    prefix = native_image_builder._HIERA_PREFIX
    weights = {
        f"{prefix}.pos_embed": np.ones((1, 95, 7, 7), dtype=np.float32),
        f"{prefix}.pos_embed_window": np.ones((1, 96, 8, 8), dtype=np.float32),
    }
    monkeypatch.setattr(
        native_image_builder.graph_ops,
        "add_constant",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid position weights must be rejected before graph creation")
        ),
    )

    with np.testing.assert_raises_regex(ValueError, "global position must have shape"):
        native_image_builder._add_hiera_position(object(), weights)


def test_hiera_layer_norm_routes_fp32_boundary_through_exact_plugin(monkeypatch):
    calls = []
    source_tensor = SimpleNamespace(dtype="bf16")
    fp32_tensor = SimpleNamespace(dtype="fp32")
    gamma_tensor = object()
    beta_tensor = object()
    plugin_output = SimpleNamespace(dtype="fp32")
    output_tensor = SimpleNamespace(dtype="fp32")
    monkeypatch.setattr(
        native_image_builder.graph_ops,
        "_trt",
        lambda: SimpleNamespace(float32="fp32"),
    )

    def fake_cast(network, tensor, dtype):
        calls.append(("cast", network, tensor, dtype))
        return fp32_tensor if tensor is source_tensor else output_tensor

    def fake_constant(network, shape, values, *, precision):
        calls.append(("constant", network, shape, values.copy(), precision))
        return gamma_tensor if np.all(values == 1.0) else beta_tensor

    def fake_plugin(network, name, inputs, *, instance_name):
        calls.append(("plugin", network, name, tuple(inputs), instance_name))
        return plugin_output

    monkeypatch.setattr(native_image_builder.graph_ops, "cast", fake_cast)
    monkeypatch.setattr(native_image_builder.graph_ops, "add_constant", fake_constant)
    monkeypatch.setattr(native_image_builder.graph_ops, "add_plugin", fake_plugin)
    network = object()
    gamma = np.ones(96, dtype=np.float64)
    beta = np.zeros(96, dtype=np.float64)

    result = native_image_builder._add_hiera_layer_norm(
        network,
        source_tensor,
        gamma,
        beta,
        instance_name="hiera_layer_norm_block_00_norm1",
    )

    assert result is output_tensor
    assert calls[0] == ("cast", network, source_tensor, "fp32")
    assert calls[1][0:3] == ("constant", network, (96,))
    assert calls[1][3].dtype == np.float32
    assert calls[1][4] == "fp32"
    assert calls[2][0:3] == ("constant", network, (96,))
    assert calls[2][3].dtype == np.float32
    assert calls[2][4] == "fp32"
    assert calls[3] == (
        "plugin",
        network,
        "Sam2HoiHieraLayerNorm",
        (fp32_tensor, gamma_tensor, beta_tensor),
        "hiera_layer_norm_block_00_norm1",
    )
    assert calls[4] == ("cast", network, plugin_output, "fp32")


def test_hiera_layer_norm_rejects_nonfixed_parameter_contracts(monkeypatch) -> None:
    monkeypatch.setattr(
        native_image_builder.graph_ops,
        "_trt",
        lambda: SimpleNamespace(float32="fp32"),
    )
    for gamma, beta, message in (
        (np.ones(95), np.zeros(95), "unsupported Hiera LayerNorm width"),
        (np.ones((96, 1)), np.zeros((96, 1)), "invalid Hiera LayerNorm parameters"),
        (np.ones(96), np.zeros(95), "invalid Hiera LayerNorm parameters"),
    ):
        with np.testing.assert_raises_regex(ValueError, message):
            native_image_builder._add_hiera_layer_norm(
                object(),
                object(),
                gamma,
                beta,
                instance_name="invalid",
            )


def test_hiera_block_uses_fp32_norms_before_explicit_bf16_linear_casts():
    block_source = inspect.getsource(native_image_builder._add_hiera_block)
    attention_source = inspect.getsource(native_image_builder._add_attention)

    assert block_source.count("_add_hiera_layer_norm(") == 2
    assert block_source.count("_cast_to_work(network, normed, precision)") == 2
    assert "mlp = _add_hiera_gelu(" in block_source
    assert "work = _cast_to_work(network, hidden_rows, precision)" in attention_source


def test_bf16_hiera_gelu_uses_exact_plugin_for_all_four_fixed_shapes(monkeypatch):
    calls = []
    plugin_output = object()
    typed_output = object()
    monkeypatch.setattr(
        native_image_builder.graph_ops,
        "add_plugin",
        lambda network, name, inputs, *, instance_name: (
            calls.append(("plugin", network, name, tuple(inputs), instance_name)) or plugin_output
        ),
    )
    monkeypatch.setattr(
        native_image_builder.graph_ops,
        "cast",
        lambda network, tensor, dtype: (
            calls.append(("cast", network, tensor, dtype)) or typed_output
        ),
    )
    monkeypatch.setattr(
        native_image_builder.graph_ops,
        "_trt",
        lambda: SimpleNamespace(bfloat16="bf16"),
    )
    network = object()
    for block_index, shape in enumerate(
        (
            (1, 256, 256, 384),
            (1, 128, 128, 768),
            (1, 64, 64, 1536),
            (1, 32, 32, 3072),
        )
    ):
        calls.clear()
        tensor = SimpleNamespace(shape=shape)

        result = native_image_builder._add_hiera_gelu(
            network,
            tensor,
            precision="bf16",
            block_index=block_index,
        )

        assert result is typed_output
        assert calls == [
            (
                "plugin",
                network,
                "Sam2HoiHieraGeluErfBF16",
                (tensor,),
                f"hiera_gelu_erf_bf16_block_{block_index:02d}",
            ),
            ("cast", network, plugin_output, "bf16"),
        ]


def test_hiera_gelu_precision_and_shape_guards(monkeypatch):
    fp32_output = object()
    monkeypatch.setattr(
        native_image_builder.graph_ops,
        "add_activation",
        lambda network, tensor, kind: fp32_output,
    )
    monkeypatch.setattr(
        native_image_builder.graph_ops,
        "add_plugin",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("FP32 Hiera GELU must remain a TensorRT native activation")
        ),
    )

    assert (
        native_image_builder._add_hiera_gelu(
            object(),
            SimpleNamespace(shape=(1, 1, 1, 1)),
            precision="fp32",
            block_index=0,
        )
        is fp32_output
    )
    with np.testing.assert_raises_regex(ValueError, "unsupported Hiera GELU shape"):
        native_image_builder._add_hiera_gelu(
            object(),
            SimpleNamespace(shape=(1, 64, 64, 768)),
            precision="bf16",
            block_index=0,
        )


def test_hiera_attention_routes_exact_plugins_only_at_reviewed_bf16_sites():
    source = inspect.getsource(native_image_builder._add_attention)

    assert 'if precision == "bf16":' in source
    assert '"Sam2HoiHieraFlashAttention96"' in source
    assert "if head_dim != 96:" in source
    assert "query = _scale_query(network, query, head_dim=head_dim)" in source
    assert "network.add_softmax(" in source
    assert 'if precision == "bf16" and spec.index in {14, 15}:' in source
    assert "(batches, output_sequence, spec.dim_out) != (25, 49, 768)" in source
    assert '"Sam2HoiHieraBlock1415Projection"' in source
    assert "else:\n        projected = graph_ops.add_linear(" in source


def test_bf16_hiera_patch_conv_uses_only_fixed_site_family_plugin(monkeypatch):
    prefix = native_image_builder._HIERA_PREFIX
    weight_values = np.ones((96, 3, 7, 7), dtype=np.float32)
    bias_values = np.ones((96,), dtype=np.float32)
    weights = {
        f"{prefix}.patch_embed.proj.weight": weight_values,
        f"{prefix}.patch_embed.proj.bias": bias_values,
    }
    pixel_values = object()
    work = object()
    weight_tensor = object()
    bias_tensor = object()
    output = SimpleNamespace(name="")
    typed_output = object()
    cast_layer = SimpleNamespace(get_output=lambda index: typed_output if index == 0 else None)
    calls = []

    class FakeNetwork:
        def add_cast(self, tensor, dtype):
            calls.append(("explicit_cast", tensor, dtype))
            return cast_layer

    network = FakeNetwork()
    monkeypatch.setattr(
        native_image_builder,
        "_cast_to_work",
        lambda actual_network, tensor, precision: (
            calls.append(("cast", actual_network, tensor, precision)) or work
        ),
    )

    def fake_constant(actual_network, shape, values, *, precision):
        calls.append(("constant", actual_network, shape, values, precision))
        return weight_tensor if values is weight_values else bias_tensor

    monkeypatch.setattr(native_image_builder.graph_ops, "add_constant", fake_constant)
    monkeypatch.setattr(
        native_image_builder.graph_ops,
        "add_conv2d",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("BF16 Hiera patch convolution must use the fixed-site plugin")
        ),
    )
    monkeypatch.setattr(
        native_image_builder.graph_ops,
        "add_plugin",
        lambda actual_network, name, inputs, *, instance_name: (
            calls.append(("plugin", actual_network, name, tuple(inputs), instance_name)) or output
        ),
    )
    monkeypatch.setattr(
        native_image_builder.trt_compat,
        "get_trt",
        lambda: SimpleNamespace(bfloat16="bf16_dtype"),
    )

    result = native_image_builder._add_hiera_patch_conv(
        network,
        pixel_values,
        weights,
        precision="bf16",
    )

    assert result is typed_output
    assert output.name == "hiera_patch_conv.raw_output"
    assert calls[0] == ("cast", network, pixel_values, "bf16")
    assert calls[1][0:3] == ("constant", network, (96, 3, 7, 7))
    assert calls[1][3] is weight_values
    assert calls[1][4] == "bf16"
    assert calls[2][0:3] == ("constant", network, (96,))
    assert calls[2][3] is bias_values
    assert calls[2][4] == "bf16"
    assert calls[3] == (
        "plugin",
        network,
        "Sam2HoiHieraPatchConv",
        (work, weight_tensor, bias_tensor),
        "hiera_patch_conv",
    )
    assert calls[4] == ("explicit_cast", output, "bf16_dtype")


def test_hiera_patch_conv_rejects_nonfixed_checkpoint_shapes():
    prefix = native_image_builder._HIERA_PREFIX
    weights = {
        f"{prefix}.patch_embed.proj.weight": np.ones((95, 3, 7, 7), dtype=np.float32),
        f"{prefix}.patch_embed.proj.bias": np.ones((96,), dtype=np.float32),
    }

    with np.testing.assert_raises_regex(ValueError, "weight must have shape"):
        native_image_builder._add_hiera_patch_conv(
            object(),
            object(),
            weights,
            precision="bf16",
        )


def test_fp32_hiera_patch_conv_preserves_standard_tensor_rt_conv(monkeypatch):
    prefix = native_image_builder._HIERA_PREFIX
    weight_values = np.ones((96, 3, 7, 7), dtype=np.float32)
    bias_values = np.ones((96,), dtype=np.float32)
    weights = {
        f"{prefix}.patch_embed.proj.weight": weight_values,
        f"{prefix}.patch_embed.proj.bias": bias_values,
    }
    work = object()
    output = object()
    calls = []
    monkeypatch.setattr(native_image_builder, "_cast_to_work", lambda *_args: work)
    monkeypatch.setattr(
        native_image_builder.graph_ops,
        "add_plugin",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("FP32 Hiera patch convolution must not use the BF16-only plugin")
        ),
    )

    def fake_conv(network, tensor, weight, bias, **kwargs):
        calls.append((network, tensor, weight, bias, kwargs))
        return output

    monkeypatch.setattr(native_image_builder.graph_ops, "add_conv2d", fake_conv)
    network = object()

    result = native_image_builder._add_hiera_patch_conv(
        network,
        object(),
        weights,
        precision="fp32",
    )

    assert result is output
    assert calls == [
        (
            network,
            work,
            weight_values,
            bias_values,
            {"stride": (4, 4), "padding": (3, 3), "precision": "fp32"},
        )
    ]


def test_exact_bf16_1x1_uses_six_fixed_einsum_contracts_and_explicit_bias(monkeypatch):
    fixed_contracts = {
        ((1, 96, 256, 256), (256, 96, 1, 1), (256,)),
        ((1, 192, 128, 128), (256, 192, 1, 1), (256,)),
        ((1, 384, 64, 64), (256, 384, 1, 1), (256,)),
        ((1, 768, 32, 32), (256, 768, 1, 1), (256,)),
        ((1, 256, 256, 256), (32, 256, 1, 1), (32,)),
        ((1, 256, 128, 128), (64, 256, 1, 1), (64,)),
    }
    assert native_image_builder._EXACT_BF16_NCHW_1X1_CONTRACTS == fixed_contracts
    assert native_image_builder._EXACT_BF16_NCHW_1X1_EINSUM_EQUATION == "nchw,oc->nohw"

    calls = []
    product = object()
    output = object()
    weight_tensor = object()
    bias_tensor = object()

    class FakeNetwork:
        def add_einsum(self, inputs, equation):
            calls.append(("einsum", tuple(inputs), equation))
            return SimpleNamespace(get_output=lambda index: product if index == 0 else None)

        def add_elementwise(self, lhs, rhs, operation):
            calls.append(("elementwise", lhs, rhs, operation))
            return SimpleNamespace(get_output=lambda index: output if index == 0 else None)

    def fake_constant(network, shape, values, *, precision):
        calls.append(("constant", network, shape, values.shape, precision))
        return weight_tensor if len(shape) == 2 else bias_tensor

    monkeypatch.setattr(
        native_image_builder,
        "_cast_to_work",
        lambda network, tensor, precision: (
            calls.append(("cast", network, tensor, precision)) or tensor
        ),
    )
    monkeypatch.setattr(native_image_builder.graph_ops, "add_constant", fake_constant)
    monkeypatch.setattr(
        native_image_builder.graph_ops,
        "_trt",
        lambda: SimpleNamespace(ElementWiseOperation=SimpleNamespace(SUM="sum")),
    )

    network = FakeNetwork()
    for input_shape, weight_shape, bias_shape in sorted(fixed_contracts):
        calls.clear()
        tensor = SimpleNamespace(shape=input_shape)
        weight = np.ones(weight_shape, dtype=np.float32)
        bias = np.ones(bias_shape, dtype=np.float32)
        output_channels, input_channels = weight_shape[:2]

        result = native_image_builder._add_exact_bf16_nchw_1x1_projection(
            network,
            tensor,
            weight,
            bias,
            precision="bf16",
        )

        assert result is output
        assert calls[0] == ("cast", network, tensor, "bf16")
        assert calls[1][0:3] == ("constant", network, (output_channels, input_channels))
        assert calls[1][3:] == ((output_channels, input_channels), "bf16")
        assert calls[2] == (
            "einsum",
            (tensor, weight_tensor),
            "nchw,oc->nohw",
        )
        assert calls[3][0:3] == ("constant", network, (1, output_channels, 1, 1))
        assert calls[3][3:] == ((1, output_channels, 1, 1), "bf16")
        assert calls[4] == ("elementwise", product, bias_tensor, "sum")


def test_exact_bf16_1x1_rejects_every_nonallowlisted_contract_before_graph_creation(
    monkeypatch,
):
    invalid_contracts = (
        ((1, 256, 64, 64), (32, 256, 1, 1), (32,)),
        ((1, 96, 256, 256), (255, 96, 1, 1), (255,)),
        ((1, 256, 256, 256), (32, 256, 1, 1), (31,)),
        ((1, 256, 256, 256), (32, 256, 3, 3), (32,)),
    )
    for input_shape, weight_shape, bias_shape in invalid_contracts:
        tensor = SimpleNamespace(shape=input_shape)
        monkeypatch.setattr(native_image_builder, "_cast_to_work", lambda *_args: tensor)

        with np.testing.assert_raises_regex(ValueError, "six fixed contracts"):
            native_image_builder._add_exact_bf16_nchw_1x1_projection(
                object(),
                tensor,
                np.ones(weight_shape, dtype=np.float32),
                np.ones(bias_shape, dtype=np.float32),
                precision="bf16",
            )


def test_fp32_exact_1x1_helper_preserves_standard_tensor_rt_conv(monkeypatch):
    source = object()
    work = object()
    weight = np.ones((7, 5, 1, 1), dtype=np.float32)
    bias = np.ones((7,), dtype=np.float32)
    output = object()
    calls = []
    monkeypatch.setattr(native_image_builder, "_cast_to_work", lambda *_args: work)

    def fake_conv(network, tensor, actual_weight, actual_bias, **kwargs):
        calls.append((network, tensor, actual_weight, actual_bias, kwargs))
        return output

    monkeypatch.setattr(native_image_builder.graph_ops, "add_conv2d", fake_conv)
    network = object()

    result = native_image_builder._add_exact_bf16_nchw_1x1_projection(
        network,
        source,
        weight,
        bias,
        precision="fp32",
    )

    assert result is output
    assert calls == [(network, work, weight, bias, {"precision": "fp32"})]


def test_exact_bf16_1x1_helper_is_scoped_to_four_fpn_and_two_tracker_calls():
    module_source = Path(inspect.getsourcefile(native_image_builder) or "").read_text(
        encoding="utf-8"
    )
    fpn_source = inspect.getsource(native_image_builder._add_fpn)
    tracker_source = inspect.getsource(native_image_builder._add_tracker_front_outputs)
    builder_source = inspect.getsource(native_image_builder.build_image_feature_engine)

    assert module_source.count("_add_exact_bf16_nchw_1x1_projection(") == 4
    assert fpn_source.count("_add_exact_bf16_nchw_1x1_projection(") == 1
    assert tracker_source.count("_add_exact_bf16_nchw_1x1_projection(") == 2
    assert builder_source.count("_add_tracker_front_outputs(") == 1
    assert "for index in range(3, -1, -1):" in fpn_source
    assert "graph_ops.add_conv2d(" not in fpn_source
    assert '"sam_mask_decoder.conv_s0.weight"' in tracker_source
    assert '"sam_mask_decoder.conv_s1.weight"' in tracker_source


def _conv_bn_weights(prefix: str) -> dict[str, np.ndarray]:
    return {
        f"{prefix}.conv.weight": np.ones((2, 3, 1, 1), dtype=np.float32),
        f"{prefix}.bn.weight": np.asarray([1.0, 2.0], dtype=np.float32),
        f"{prefix}.bn.bias": np.asarray([3.0, 4.0], dtype=np.float32),
        f"{prefix}.bn.running_mean": np.asarray([5.0, 6.0], dtype=np.float32),
        f"{prefix}.bn.running_var": np.asarray([7.0, 8.0], dtype=np.float32),
    }


def test_pafpn_bf16_rounds_convolution_before_fp32_batch_norm(monkeypatch):
    calls = []
    input_tensor = object()
    work_tensor = object()
    convolution_tensor = object()
    batch_norm_tensor = object()
    activated_tensor = object()
    prefix = "layer"
    weights = _conv_bn_weights(prefix)

    monkeypatch.setattr(
        native_image_builder,
        "_cast_to_work",
        lambda network, tensor, precision: (
            calls.append(("work_cast", tensor, precision)) or work_tensor
        ),
    )

    def fake_conv(network, tensor, weight, bias, **kwargs):
        calls.append(("conv", tensor, weight, bias, kwargs))
        return convolution_tensor

    def fake_batch_norm(network, tensor, gamma, beta, mean, variance, **kwargs):
        calls.append(("batch_norm", tensor, gamma, beta, mean, variance, kwargs))
        return batch_norm_tensor

    monkeypatch.setattr(native_image_builder.graph_ops, "add_conv2d", fake_conv)
    monkeypatch.setattr(
        native_image_builder.graph_ops,
        "add_batch_norm2d_affine",
        fake_batch_norm,
    )
    monkeypatch.setattr(
        native_image_builder.graph_ops,
        "runtime_dtype",
        lambda precision: f"{precision}_dtype",
    )
    monkeypatch.setattr(
        native_image_builder.graph_ops,
        "fold_batch_norm",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("BF16 PAFPN must not fold BatchNorm into convolution weights")
        ),
    )
    monkeypatch.setattr(
        native_image_builder.graph_ops,
        "add_activation",
        lambda network, tensor, kind: (
            calls.append(("activation", tensor, kind)) or activated_tensor
        ),
    )

    result = native_image_builder._add_conv_bn_silu(
        object(),
        input_tensor,
        weights,
        prefix,
        precision="bf16",
    )

    assert result is activated_tensor
    assert [call[0] for call in calls] == ["work_cast", "conv", "batch_norm", "activation"]
    assert calls[1][1] is work_tensor
    np.testing.assert_array_equal(calls[1][2], weights[f"{prefix}.conv.weight"])
    assert calls[1][4]["precision"] == "bf16"
    assert calls[2][1] is convolution_tensor
    assert calls[2][6] == {"epsilon": 1.0e-5, "output_dtype": "bf16_dtype"}
    assert calls[3] == ("activation", batch_norm_tensor, "silu")


def test_pafpn_fp32_preserves_folded_batch_norm_path(monkeypatch):
    calls = []
    prefix = "layer"
    weights = _conv_bn_weights(prefix)
    folded_weight = np.full((2, 3, 1, 1), 9.0, dtype=np.float32)
    folded_bias = np.full(2, 10.0, dtype=np.float32)
    convolution_tensor = object()

    monkeypatch.setattr(native_image_builder, "_cast_to_work", lambda *_args: object())
    monkeypatch.setattr(
        native_image_builder.graph_ops,
        "fold_batch_norm",
        lambda *_args, **_kwargs: (folded_weight, folded_bias),
    )

    def fake_conv(network, tensor, weight, bias, **kwargs):
        calls.append((weight, bias, kwargs))
        return convolution_tensor

    monkeypatch.setattr(native_image_builder.graph_ops, "add_conv2d", fake_conv)
    monkeypatch.setattr(
        native_image_builder.graph_ops,
        "add_batch_norm2d_affine",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("FP32 PAFPN should retain the folded path")
        ),
    )
    monkeypatch.setattr(
        native_image_builder.graph_ops,
        "add_activation",
        lambda _network, tensor, kind: (tensor, kind),
    )

    result = native_image_builder._add_conv_bn_silu(
        object(),
        object(),
        weights,
        prefix,
        precision="fp32",
    )

    assert result == (convolution_tensor, "silu")
    assert calls == [
        (folded_weight, folded_bias, {"stride": (1, 1), "padding": (0, 0), "precision": "fp32"})
    ]


def test_hiera_small_block_contract_tracks_windows_and_q_pool_shapes():
    specs = native_image_builder._HIERA_BLOCKS
    assert len(specs) == 16
    assert [(spec.index, spec.height, spec.dim, spec.dim_out) for spec in specs] == [
        (0, 256, 96, 96),
        (1, 256, 96, 192),
        (2, 128, 192, 192),
        (3, 128, 192, 384),
        (4, 64, 384, 384),
        (5, 64, 384, 384),
        (6, 64, 384, 384),
        (7, 64, 384, 384),
        (8, 64, 384, 384),
        (9, 64, 384, 384),
        (10, 64, 384, 384),
        (11, 64, 384, 384),
        (12, 64, 384, 384),
        (13, 64, 384, 384),
        (14, 64, 384, 768),
        (15, 32, 768, 768),
    ]
    assert [spec.index for spec in specs if spec.q_pool] == [1, 3, 14]
    assert [spec.index for spec in specs if spec.window == 0] == [7, 10, 13]
    assert [spec.heads for spec in specs] == [1, 2, 2] + [4] * 11 + [8, 8]
    assert [spec.window for spec in specs] == [
        8,
        8,
        4,
        4,
        14,
        14,
        14,
        0,
        14,
        14,
        0,
        14,
        14,
        0,
        14,
        7,
    ]


def test_padded_window_indices_round_trip_nondivisible_hiera_stage():
    order, padded_h, padded_w, sentinel = native_image_builder._window_partition_indices(64, 64, 14)
    assert (padded_h, padded_w, sentinel) == (70, 70, 4096)
    assert order.shape == (4900,)
    assert np.count_nonzero(order == sentinel) == 804

    source = np.arange(4096, dtype=np.int32)
    partitioned = np.concatenate((source, np.asarray([-1], dtype=np.int32)))[order]
    inverse = native_image_builder._window_unpartition_indices(64, 64, padded_h, padded_w, 14)
    np.testing.assert_array_equal(partitioned[inverse], source)


def test_q_pool_window_unpartition_crops_35_to_32():
    inverse = native_image_builder._window_unpartition_indices(32, 32, 35, 35, 7)
    assert inverse.shape == (1024,)
    assert int(inverse.min()) == 0
    assert int(inverse.max()) < 1225
    assert len(np.unique(inverse)) == 1024


def test_tracker_position_encoding_uses_exact_native_fp32_schedule(monkeypatch):
    calls = []

    class FakeTensor:
        def __init__(self, name):
            self.name = name
            self.dtype = "fp32"

    class FakeLayer:
        def __init__(self, kind, *inputs):
            self.kind = kind
            self.inputs = inputs
            self.output = FakeTensor(f"{kind}.{len(calls)}")
            self.axis = None
            self.first_transpose = None
            self.reshape_dims = None

        def get_output(self, index):
            assert index == 0
            return self.output

    class FakeNetwork:
        def add_elementwise(self, lhs, rhs, operation):
            layer = FakeLayer("elementwise", lhs, rhs, operation)
            calls.append(layer)
            return layer

        def add_unary(self, tensor, operation):
            layer = FakeLayer("unary", tensor, operation)
            calls.append(layer)
            return layer

        def add_slice(self, tensor, start, shape, stride):
            layer = FakeLayer("slice", tensor, start, shape, stride)
            calls.append(layer)
            return layer

        def add_shuffle(self, tensor):
            layer = FakeLayer("shuffle", tensor)
            calls.append(layer)
            return layer

        def add_concatenation(self, tensors):
            layer = FakeLayer("concatenation", tuple(tensors))
            calls.append(layer)
            return layer

    trt = SimpleNamespace(
        float32="fp32",
        ElementWiseOperation=SimpleNamespace(POW="pow", DIV="div"),
        UnaryOperation=SimpleNamespace(SIN="sin", COS="cos"),
        Permutation=lambda values: tuple(values),
    )
    monkeypatch.setattr(native_image_builder.graph_ops, "_trt", lambda: trt)
    constants = []

    def fake_constant(network, shape, values, *, precision):
        tensor = FakeTensor(f"constant.{len(constants)}")
        constants.append((network, shape, np.ascontiguousarray(values), precision, tensor))
        return tensor

    monkeypatch.setattr(native_image_builder.graph_ops, "add_constant", fake_constant)
    network = FakeNetwork()

    result = native_image_builder._add_position_encoding_sine(network, 64, 64)

    assert [item[1] for item in constants] == [(64, 1), (1, 128), (1, 128)]
    assert all(item[0] is network and item[3] == "fp32" for item in constants)
    assert hashlib.sha256(constants[0][2].tobytes()).hexdigest() == (
        "b09ffa87b6d278deaeee356f34b727ad32b360860b1e9c1d910508c396a2ee64"
    )
    assert hashlib.sha256(constants[1][2].tobytes()).hexdigest() == (
        "869a7d6ed3dfb4e3c4e3fa0b4cdb624c1ea5281031a116c1674c626c7c8d43e5"
    )
    np.testing.assert_array_equal(constants[2][2], np.full((1, 128), 10000.0))

    elementwise = [call for call in calls if call.kind == "elementwise"]
    assert [call.inputs[2] for call in elementwise] == ["pow", "div"]
    assert elementwise[0].inputs[:2] == (constants[2][4], constants[1][4])
    assert elementwise[1].inputs == (
        constants[0][4],
        elementwise[0].output,
        "div",
    )

    unary = [call for call in calls if call.kind == "unary"]
    assert [call.inputs for call in unary] == [
        (elementwise[1].output, "sin"),
        (elementwise[1].output, "cos"),
    ]
    slices = [call for call in calls if call.kind == "slice"]
    assert [call.inputs[1:] for call in slices] == [
        ((0, 0), (64, 64), (1, 2)),
        ((0, 1), (64, 64), (1, 2)),
    ]

    shuffles = [call for call in calls if call.kind == "shuffle"]
    assert [call.reshape_dims for call in shuffles] == [
        (64, 64, 1),
        (64, 64, 1),
        (64, 128),
        (1, 128, 64, 1),
        (1, 128, 1, 64),
    ]
    assert [call.first_transpose for call in shuffles] == [
        None,
        None,
        None,
        (1, 0),
        (1, 0),
    ]

    concatenations = [call for call in calls if call.kind == "concatenation"]
    assert [call.axis for call in concatenations] == [2, 3, 2, 1]
    assert len(concatenations[1].inputs[0]) == 64
    assert len(concatenations[2].inputs[0]) == 64
    assert len(set(concatenations[1].inputs[0])) == 1
    assert len(set(concatenations[2].inputs[0])) == 1
    assert result is concatenations[-1].output

    source = inspect.getsource(native_image_builder._add_position_encoding_sine)
    assert "ElementWiseOperation.POW" in source
    assert "ElementWiseOperation.DIV" in source
    assert "UnaryOperation.SIN" in source
    assert "UnaryOperation.COS" in source
    assert "np.power" not in source
    assert "np.sin" not in source
    assert "np.cos" not in source


def test_tracker_position_encoding_rejects_nonfixed_contract_before_graph_creation():
    for contract in ((32, 64, 256), (64, 32, 256), (64, 64, 128)):
        with np.testing.assert_raises_regex(ValueError, "reviewed 64x64x256 contract"):
            native_image_builder._add_position_encoding_sine(object(), *contract)


def test_builder_is_family_owned_network_definition_code_only():
    source_path = Path(inspect.getsourcefile(native_image_builder) or "")
    source = source_path.read_text(encoding="utf-8")
    lowered = source.lower()
    forbidden_graph_format = "on" + "nx"
    assert forbidden_graph_format not in lowered
    assert "torch" not in lowered
    assert "families.sam3" not in lowered

    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert all(not name.startswith("torch") for name in imported)
    assert "Builder" in source
    assert "create_network" in source
    assert "strongly_typed=True" in source
    assert "add_input" in source
    assert "build_serialized_network" in source


def test_builder_publishes_exact_runtime_binding_names():
    source = inspect.getsource(native_image_builder.build_image_feature_engine)
    for name in (
        "pixel_values",
        "tracker_feature_0",
        "tracker_feature_1",
        "tracker_feature_2",
        "tracker_position_2",
    ):
        assert name in source
    assert 'f"detector_feature_{index}"' in source
    assert source.count("graph_ops.mark_output") == 5
