# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-native post graph for Fast Foundation Stereo."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from .native_graph import NativeGraph


def _full_volume_conv_bn_folding_enabled() -> bool:
    return os.environ.get("TRTMC_FAST_FOUNDATION_STEREO_FOLD_FULL_VOLUME_BN", "1") == "1"


def _post16_to_8_conv_bn_folding_enabled() -> bool:
    return os.environ.get("TRTMC_FAST_FOUNDATION_STEREO_FOLD_POST16_TO_8_BN", "1") == "1"


def _feature_att_8_conv_bn_folding_enabled() -> bool:
    return os.environ.get("TRTMC_FAST_FOUNDATION_STEREO_FOLD_FEATURE_ATT_8_BN", "1") == "1"


def _remaining_safe_conv_bn_folding_enabled() -> bool:
    return os.environ.get("TRTMC_FAST_FOUNDATION_STEREO_FOLD_REMAINING_SAFE_BN", "1") == "1"


def _post8_sum_plugin_enabled() -> bool:
    return os.environ.get("TRTMC_FAST_FOUNDATION_STEREO_POST8_SUM_PLUGIN", "1") == "1"


def _full_volume_leaky_all3_plugin_enabled() -> bool:
    return os.environ.get("TRTMC_FAST_FOUNDATION_STEREO_FULL_VOLUME_LEAKY_ALL3_PLUGIN", "1") == "1"


def _disp_head_nchw_pointwise_enabled() -> bool:
    return os.environ.get("TRTMC_FAST_FOUNDATION_STEREO_DISP_HEAD_NCHW_POINTWISE", "1") == "1"


def _disp_head_fold_gamma_enabled() -> bool:
    return os.environ.get("TRTMC_FAST_FOUNDATION_STEREO_DISP_HEAD_FOLD_GAMMA", "1") == "1"


def _disp_head_second_gelu_tanh_enabled() -> bool:
    return os.environ.get("TRTMC_FAST_FOUNDATION_STEREO_DISP_HEAD_SECOND_GELU_TANH", "1") == "1"


def _post8_sum_tile_positions() -> int:
    variable = "TRTMC_FAST_FOUNDATION_STEREO_POST8_SUM_TILE_POSITIONS"
    raw_value = os.environ.get(variable, "32")
    try:
        value = int(raw_value)
    except ValueError as error:
        raise RuntimeError(
            f"{variable} must be one of (32, 64, 128, 256), got {raw_value!r}"
        ) from error
    if value not in (32, 64, 128, 256):
        raise RuntimeError(f"{variable} must be one of (32, 64, 128, 256), got {raw_value!r}")
    return value


def _feature_attention(graph: NativeGraph, volume: Any, feature: Any, module: Any) -> Any:
    return graph.feature_attention(volume, feature, module)


_FULL_VOLUME_LEAKY_SHAPE = (1, 28, 48, 176, 176)
_FULL_VOLUME_LEAKY_SPECS = {
    "corr_feature_att.layers.0": (
        "Conv3d",
        32,
        28,
        (3, 3, 3),
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        1,
        (0, 0, 0),
    ),
    "cost_agg.conv1_up": (
        "ConvTranspose3d",
        56,
        28,
        (4, 4, 4),
        (2, 2, 2),
        (1, 1, 1),
        (1, 1, 1),
        1,
        (0, 0, 0),
    ),
    "cost_agg.post8_to_4.out.0": (
        "Conv3d",
        28,
        28,
        (3, 3, 3),
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        1,
        (0, 0, 0),
    ),
}


def _optional_tuple(instance: Any, name: str) -> tuple[int, ...] | None:
    value = getattr(instance, name, None)
    return None if value is None else tuple(int(item) for item in value)


def _require_full_volume_leaky_basic_conv(module: Any, path: str) -> None:
    convolution = getattr(module, "conv", None)
    batch_norm = getattr(module, "bn", None)
    actual = (
        convolution.__class__.__name__ if convolution is not None else "NoneType",
        getattr(convolution, "in_channels", None),
        getattr(convolution, "out_channels", None),
        _optional_tuple(convolution, "kernel_size"),
        _optional_tuple(convolution, "stride"),
        _optional_tuple(convolution, "padding"),
        _optional_tuple(convolution, "dilation"),
        getattr(convolution, "groups", None),
        _optional_tuple(convolution, "output_padding"),
    )
    expected = _FULL_VOLUME_LEAKY_SPECS[path]
    common = (
        module.__class__.__name__,
        getattr(module, "use_bn", None),
        getattr(module, "relu", None),
        batch_norm.__class__.__name__ if batch_norm is not None else "NoneType",
        getattr(batch_norm, "num_features", None),
        getattr(convolution, "bias", None) is not None,
    )
    expected_common = ("BasicConv", True, True, "SyncBatchNorm", 28, False)
    if actual != expected or common != expected_common:
        raise RuntimeError(
            "full-volume Leaky ALL3 plugin requires the distilled BasicConv topology; "
            f"{path} is {(common, actual)!r}, expected {(expected_common, expected)!r}"
        )


def _require_full_volume_leaky_all3_topology(model: Any) -> None:
    corr_helper = model.corr_feature_att
    corr_layers = tuple(corr_helper.layers)
    corr_topology = (
        corr_helper.__class__.__name__,
        tuple(child.__class__.__name__ for child in corr_layers),
    )
    expected_corr = ("ForwardHelper", ("BasicConv", "BasicConv", "FeatureAtt"))
    if corr_topology != expected_corr:
        raise RuntimeError(
            "full-volume Leaky ALL3 plugin requires the distilled corr_feature_att topology; "
            f"got {corr_topology!r}, expected {expected_corr!r}"
        )

    post8 = model.cost_agg.post8_to_4
    post8_topology = (
        post8.__class__.__name__,
        getattr(post8, "op", None),
        tuple(child.__class__.__name__ for child in post8.upsample),
        tuple(child.__class__.__name__ for child in post8.out),
    )
    expected_post8 = (
        "PostForwardHelper",
        "sum",
        ("BasicConv", "CostVolumeDisparityAttention", "Upsample"),
        ("BasicConv", "ResnetBasicBlock3D"),
    )
    if post8_topology != expected_post8:
        raise RuntimeError(
            "full-volume Leaky ALL3 plugin requires the distilled post8_to_4 topology; "
            f"got {post8_topology!r}, expected {expected_post8!r}"
        )

    targets = (
        (corr_layers[0], "corr_feature_att.layers.0"),
        (model.cost_agg.conv1_up, "cost_agg.conv1_up"),
        (tuple(post8.out)[0], "cost_agg.post8_to_4.out.0"),
    )
    for module, path in targets:
        _require_full_volume_leaky_basic_conv(module, path)


def _full_volume_leaky_plugin_output(
    graph: NativeGraph,
    tensor: Any,
    *,
    name: str,
) -> Any:
    shape = tuple(int(dimension) for dimension in tensor.shape)
    if shape != _FULL_VOLUME_LEAKY_SHAPE or tensor.dtype != graph.trt.float16:
        raise RuntimeError(
            "full-volume Leaky ALL3 plugin requires FP16 logical shape "
            f"{_FULL_VOLUME_LEAKY_SHAPE!r}; got {shape!r}/{tensor.dtype!r}"
        )

    from .native_plugin_builder import add_full_volume_leaky_plugin

    return add_full_volume_leaky_plugin(
        graph.network,
        tensor,
        trt_module=graph.trt,
        name=name,
    )


def _folded_basic_conv_full_volume_leaky(
    graph: NativeGraph,
    tensor: Any,
    module: Any,
    *,
    path: str,
    name: str,
) -> Any:
    _require_full_volume_leaky_basic_conv(module, path)
    convolution = module.conv
    output = graph._convolution_batch_norm(
        tensor,
        convolution,
        module.bn,
        dimensions=3,
        deconv="Transpose" in convolution.__class__.__name__,
    )
    return _full_volume_leaky_plugin_output(graph, output, name=name)


def _folded_corr_feature_att_with_full_volume_leaky(
    graph: NativeGraph,
    tensor: Any,
    feature: Any,
    module: Any,
) -> Any:
    layers = tuple(module.layers)
    output = _folded_basic_conv_full_volume_leaky(
        graph,
        tensor,
        layers[0],
        path="corr_feature_att.layers.0",
        name="corr_feature_att_layers_0_leaky",
    )
    output = graph.basic_conv(output, layers[1], fold_batch_norm=True)
    return graph.feature_attention(output, feature, layers[2])


def _folded_feature_att_8_forward_helper(
    graph: NativeGraph,
    tensor: Any,
    feature: Any,
    module: Any,
) -> Any:
    module_name = module.__class__.__name__
    if module_name != "ForwardHelper":
        raise RuntimeError(
            "feature_att_8 Conv-BN folding requires a ForwardHelper; "
            f"feature_att_8 is {module_name!r}"
        )
    layers = tuple(module.layers)
    actual = tuple(layer.__class__.__name__ for layer in layers)
    expected = ("BasicConv", "Conv3dNormActReduced", "FeatureAtt")
    if actual != expected:
        raise RuntimeError(
            "feature_att_8 Conv-BN folding requires the distilled topology; "
            f"layers are {actual!r}, expected {expected!r}"
        )

    basic_conv, reduced, feature_attention = layers
    output = graph.basic_conv(tensor, basic_conv, fold_batch_norm=True)
    output = graph.conv3d_reduced(output, reduced, fold_batch_norm=True)
    # Keep the FeatureAtt gate on its original path. In particular, its own
    # feature projection and normalization must not be folded by this switch.
    return graph.feature_attention(output, feature, feature_attention)


def _folded_remaining_safe_sequence(
    graph: NativeGraph,
    tensor: Any,
    module: Any,
    *,
    scope: str,
    expected: tuple[str, ...],
) -> Any:
    """Fold an audited direct BasicConv/reduced-Conv3d sequence, with no fallback."""

    children = tuple(module)
    actual = tuple(child.__class__.__name__ for child in children)
    if actual != expected:
        raise RuntimeError(
            f"remaining-safe Conv-BN folding requires the distilled {scope} topology; "
            f"layers are {actual!r}, expected {expected!r}"
        )

    for index, child in enumerate(children):
        path = f"{scope}.{index}"
        if child.__class__.__name__ == "BasicConv":
            _require_remaining_safe_basic_conv(child, path, convolution_name="Conv3d")
        else:
            _require_remaining_safe_reduced(child, path)

    output = tensor
    for child in children:
        if child.__class__.__name__ == "BasicConv":
            output = graph.basic_conv(output, child, fold_batch_norm=True)
        else:
            output = graph.conv3d_reduced(output, child, fold_batch_norm=True)
    return output


def _require_remaining_safe_basic_conv(
    module: Any,
    path: str,
    *,
    convolution_name: str,
) -> None:
    convolution = getattr(module, "conv", None)
    batch_norm = getattr(module, "bn", None)
    actual = (
        module.__class__.__name__,
        convolution.__class__.__name__ if convolution is not None else "NoneType",
        batch_norm.__class__.__name__ if batch_norm is not None else "NoneType",
        getattr(module, "use_bn", None),
        getattr(module, "relu", None),
    )
    expected = ("BasicConv", convolution_name, "SyncBatchNorm", True, True)
    if actual != expected:
        raise RuntimeError(
            "remaining-safe Conv-BN folding requires the distilled direct BasicConv topology; "
            f"{path} is {actual!r}, expected {expected!r}"
        )


def _require_remaining_safe_reduced(module: Any, path: str) -> None:
    actual = (
        module.__class__.__name__,
        tuple(child.__class__.__name__ for child in getattr(module, "conv1", ())),
        tuple(child.__class__.__name__ for child in getattr(module, "conv2", ())),
    )
    reduced = ("Conv3d", "SyncBatchNorm", "ReLU")
    expected = ("Conv3dNormActReduced", reduced, reduced)
    if actual != expected:
        raise RuntimeError(
            "remaining-safe Conv-BN folding requires the distilled Conv3dNormActReduced "
            f"topology; {path} is {actual!r}, expected {expected!r}"
        )


def _folded_remaining_safe_feature_att_16(
    graph: NativeGraph,
    tensor: Any,
    module: Any,
) -> Any:
    module_name = module.__class__.__name__
    if module_name != "ForwardHelper":
        raise RuntimeError(
            "remaining-safe Conv-BN folding requires feature_att_16 to be a ForwardHelper; "
            f"got {module_name!r}"
        )
    return _folded_remaining_safe_sequence(
        graph,
        tensor,
        module.layers,
        scope="feature_att_16",
        expected=("BasicConv", "Conv3dNormActReduced", "Conv3dNormActReduced"),
    )


def _folded_remaining_safe_conv3(graph: NativeGraph, tensor: Any, module: Any) -> Any:
    module_name = module.__class__.__name__
    if module_name != "Sequential":
        raise RuntimeError(
            f"remaining-safe Conv-BN folding requires conv3 to be a Sequential; got {module_name!r}"
        )
    return _folded_remaining_safe_sequence(
        graph,
        tensor,
        module,
        scope="conv3",
        expected=("BasicConv", "Conv3dNormActReduced"),
    )


def _folded_remaining_safe_context(graph: NativeGraph, feature: Any, module: Any) -> list[Any]:
    module_name = module.__class__.__name__
    children = tuple(module)
    actual = tuple(child.__class__.__name__ for child in children)
    expected = ("BasicConv", "BasicConv")
    if module_name != "ModuleList" or actual != expected:
        raise RuntimeError(
            "remaining-safe Conv-BN folding requires the distilled cnet.conv04 topology; "
            f"container/layers are {(module_name, actual)!r}, "
            f"expected {('ModuleList', expected)!r}"
        )
    for index, child in enumerate(children):
        _require_remaining_safe_basic_conv(
            child,
            f"cnet.conv04.{index}",
            convolution_name="Conv2d",
        )
    # These are sibling projections of the same feature tensor, not a sequence.
    return [graph.basic_conv(feature, child, fold_batch_norm=True) for child in children]


def _scaled_dot_product_attention(
    graph: NativeGraph,
    query: Any,
    key: Any,
    value: Any,
    *,
    head_dim: int,
) -> Any:
    output_dtype = query.dtype
    query = graph.cast(query, graph.trt.float32)
    key = graph.cast(key, graph.trt.float32)
    value = graph.cast(value, graph.trt.float32)
    scores = graph.matmul(
        query,
        key,
        op_rhs=graph.trt.MatrixOperation.TRANSPOSE,
    )
    scale = graph.scalar(1.0 / np.sqrt(float(head_dim)), len(tuple(scores.shape)), like=scores)
    probabilities = graph.softmax(graph.mul(scores, scale), -1)
    return graph.cast(graph.matmul(probabilities, value), output_dtype)


def _cost_attention(graph: NativeGraph, volume: Any, module: Any) -> Any:
    batch, channels, disparities, height, width = (int(dim) for dim in volume.shape)
    tokens = batch * height * width
    sequence = graph.transpose(volume, (0, 3, 4, 2, 1))
    sequence = graph.reshape(sequence, (tokens, disparities, channels))

    position = graph._array(module.pos_embed0.pe, graph._np_dtype_for(sequence))
    position = position[:, :disparities, :]
    position_tensor = graph.constant(
        position,
        tuple(position.shape),
        dtype=graph._np_dtype_for(sequence),
        target_dtype=sequence.dtype,
    )
    sequence = graph.add(sequence, position_tensor)

    for encoder in module.sa:
        attention = encoder.self_attn
        heads = int(attention.num_heads)
        head_dim = int(attention.head_dim)
        query = graph.reshape(
            graph.linear(sequence, attention.q_proj), (tokens, disparities, heads, head_dim)
        )
        key = graph.reshape(
            graph.linear(sequence, attention.k_proj), (tokens, disparities, heads, head_dim)
        )
        value = graph.reshape(
            graph.linear(sequence, attention.v_proj), (tokens, disparities, heads, head_dim)
        )
        attended = _scaled_dot_product_attention(graph, query, key, value, head_dim=head_dim)
        attended = graph.reshape(attended, (tokens, disparities, channels))
        attended = graph.linear(attended, attention.out_proj)
        sequence = graph.layer_norm_last(graph.add(sequence, attended), encoder.norm1)

        hidden = graph.linear(sequence, encoder.linear1)
        hidden = graph.gelu(hidden)
        hidden = graph.linear(hidden, encoder.linear2)
        sequence = graph.layer_norm_last(graph.add(sequence, hidden), encoder.norm2)

    output = graph.reshape(sequence, (batch, height, width, disparities, channels))
    return graph.transpose(output, (0, 4, 3, 1, 2))


def _require_post8_sum_plugin_topology(
    module: Any,
    upsample: tuple[Any, ...],
    out: tuple[Any, ...],
) -> None:
    topology = (
        module.__class__.__name__,
        getattr(module, "op", None),
        tuple(child.__class__.__name__ for child in upsample),
        tuple(child.__class__.__name__ for child in out),
    )
    expected_topology = (
        "PostForwardHelper",
        "sum",
        ("BasicConv", "CostVolumeDisparityAttention", "Upsample"),
        ("BasicConv", "ResnetBasicBlock3D"),
    )
    if topology != expected_topology:
        raise RuntimeError(
            "post8 sum plugin requires the distilled post8_to_4 topology; "
            f"container/op/upsample/out are {topology!r}, expected {expected_topology!r}"
        )

    basic_conv, attention, resize = upsample
    convolution = getattr(basic_conv, "conv", None)
    batch_norm = getattr(basic_conv, "bn", None)

    def tuple_attribute(instance: Any, name: str) -> tuple[Any, ...] | None:
        value = getattr(instance, name, None)
        return None if value is None else tuple(value)

    upsample_contract = (
        convolution.__class__.__name__ if convolution is not None else "NoneType",
        batch_norm.__class__.__name__ if batch_norm is not None else "NoneType",
        getattr(basic_conv, "use_bn", None),
        getattr(basic_conv, "relu", None),
        getattr(convolution, "in_channels", None),
        getattr(convolution, "out_channels", None),
        tuple_attribute(convolution, "kernel_size"),
        tuple_attribute(convolution, "stride"),
        tuple_attribute(convolution, "padding"),
        tuple_attribute(convolution, "dilation"),
        getattr(convolution, "groups", None),
        getattr(attention, "resize_embed", None),
        getattr(resize, "size", None),
        getattr(resize, "scale_factor", None),
        getattr(resize, "mode", None),
        getattr(resize, "align_corners", None),
        getattr(resize, "recompute_scale_factor", None),
    )
    expected_contract = (
        "Conv3d",
        "SyncBatchNorm",
        True,
        False,
        28,
        28,
        (4, 4, 4),
        (4, 4, 4),
        (0, 0, 0),
        (1, 1, 1),
        1,
        False,
        None,
        4.0,
        "trilinear",
        False,
        None,
    )
    if upsample_contract != expected_contract:
        raise RuntimeError(
            "post8 sum plugin requires the distilled post8_to_4 pre-sum contract; "
            f"got {upsample_contract!r}, expected {expected_contract!r}"
        )


def _post8_sum_plugin_output(
    graph: NativeGraph,
    linear: Any,
    skip: Any,
    *,
    tile_positions: int,
) -> Any:
    expected_shape = (1, 28, 48, 176, 176)
    linear_shape = tuple(int(dimension) for dimension in linear.shape)
    skip_shape = tuple(int(dimension) for dimension in skip.shape)
    if linear_shape != expected_shape or skip_shape != expected_shape:
        raise RuntimeError(
            "post8 sum plugin requires LINEAR and DHWC8 logical shapes "
            f"{expected_shape!r}; got {(linear_shape, skip_shape)!r}"
        )
    if linear.dtype != graph.trt.float16 or skip.dtype != graph.trt.float16:
        raise RuntimeError(
            "post8 sum plugin requires FP16 LINEAR and DHWC8 inputs; "
            f"got {(linear.dtype, skip.dtype)!r}"
        )

    from .native_plugin_builder import add_post8_sum_plugin

    return add_post8_sum_plugin(
        graph.network,
        linear,
        skip,
        trt_module=graph.trt,
        tile_positions=tile_positions,
    )


def _post_forward_helper(
    graph: NativeGraph,
    skip: Any,
    lower: Any,
    feature: Any,
    module: Any,
    *,
    fold_batch_norm: bool = False,
    fold_post16_to_8_batch_norm: bool = False,
    fold_remaining_safe_batch_norm: bool = False,
    post8_sum_plugin: bool = False,
    post8_sum_tile_positions: int = 32,
    full_volume_leaky_plugin: bool = False,
) -> Any:
    upsample = tuple(module.upsample)
    out = tuple(module.out)
    if post8_sum_plugin:
        _require_post8_sum_plugin_topology(module, upsample, out)
    if fold_post16_to_8_batch_norm:
        expected = (
            ("upsample.0", upsample, 0, "BasicConv"),
            ("out.0", out, 0, "BasicConv"),
            ("out.2", out, 2, "Conv3dNormActReduced"),
        )
        for path, children, index, class_name in expected:
            if len(children) <= index or children[index].__class__.__name__ != class_name:
                actual = None if len(children) <= index else children[index].__class__.__name__
                raise RuntimeError(
                    "post16_to_8 Conv-BN folding requires the distilled topology; "
                    f"{path} is {actual!r}, expected {class_name!r}"
                )
    if fold_remaining_safe_batch_norm:
        module_name = module.__class__.__name__
        upsample_topology = tuple(child.__class__.__name__ for child in upsample)
        out_topology = tuple(child.__class__.__name__ for child in out)
        expected_upsample = ("BasicConv",)
        expected_out = (
            "FeatureAtt",
            "BasicConv",
            "Conv3dNormActReduced",
            "BasicConv",
        )
        if (
            module_name != "PostForwardHelper"
            or getattr(module, "op", None) != "sum"
            or upsample_topology != expected_upsample
            or out_topology != expected_out
        ):
            raise RuntimeError(
                "remaining-safe Conv-BN folding requires the distilled post32_to_16 topology; "
                "container/op/upsample/out are "
                f"{(module_name, getattr(module, 'op', None), upsample_topology, out_topology)!r}, "
                f"expected {('PostForwardHelper', 'sum', expected_upsample, expected_out)!r}"
            )
        _require_remaining_safe_basic_conv(
            out[1],
            "post32_to_16.out.1",
            convolution_name="Conv3d",
        )
        _require_remaining_safe_reduced(out[2], "post32_to_16.out.2")
        _require_remaining_safe_basic_conv(
            out[3],
            "post32_to_16.out.3",
            convolution_name="Conv3d",
        )

    output = lower
    for index, child in enumerate(upsample):
        if child.__class__.__name__ == "CostVolumeDisparityAttention":
            output = _cost_attention(graph, output, child)
        elif fold_post16_to_8_batch_norm and index == 0 and child.__class__.__name__ == "BasicConv":
            output = graph.basic_conv(output, child, fold_batch_norm=True)
        else:
            output = graph.module(output, child)
    if post8_sum_plugin:
        output = _post8_sum_plugin_output(
            graph,
            output,
            skip,
            tile_positions=post8_sum_tile_positions,
        )
    elif module.op == "sum":
        output = graph.add(output, skip)
    else:
        output = graph.concat((output, skip), 1)
    for index, child in enumerate(out):
        child_name = child.__class__.__name__
        if child_name == "FeatureAtt":
            output = _feature_attention(graph, output, feature, child)
        elif full_volume_leaky_plugin and index == 0 and child_name == "BasicConv":
            output = _folded_basic_conv_full_volume_leaky(
                graph,
                output,
                child,
                path="cost_agg.post8_to_4.out.0",
                name="cost_agg_post8_to_4_out_0_leaky",
            )
        elif (fold_batch_norm or (fold_post16_to_8_batch_norm and index == 0)) and (
            child_name == "BasicConv"
        ):
            output = graph.basic_conv(output, child, fold_batch_norm=True)
        elif fold_batch_norm and child_name == "ResnetBasicBlock3D":
            output = graph.resnet(output, child, fold_batch_norm=True)
        elif fold_post16_to_8_batch_norm and index == 2 and child_name == "Conv3dNormActReduced":
            output = graph.conv3d_reduced(output, child, fold_batch_norm=True)
        elif fold_remaining_safe_batch_norm and index in (1, 3) and child_name == "BasicConv":
            output = graph.basic_conv(output, child, fold_batch_norm=True)
        elif fold_remaining_safe_batch_norm and index == 2 and child_name == "Conv3dNormActReduced":
            output = graph.conv3d_reduced(output, child, fold_batch_norm=True)
        else:
            output = graph.module(output, child)
    return output


def _cost_aggregation(
    graph: NativeGraph,
    volume: Any,
    features: tuple[Any, Any, Any, Any],
    module: Any,
    *,
    fold_full_volume_batch_norm: bool = False,
    fold_post16_to_8_batch_norm: bool = False,
    fold_feature_att_8_batch_norm: bool = False,
    fold_remaining_safe_batch_norm: bool = False,
    post8_sum_plugin: bool = False,
    post8_sum_tile_positions: int = 32,
    full_volume_leaky_all3_plugin: bool = False,
) -> Any:
    # The serialized distilled checkpoint replaces several constructor modules
    # with ForwardHelper/PostForwardHelper instances.  Follow the live module
    # objects rather than reconstructing the unpruned source topology.
    conv1 = graph.module(volume, module.conv1)
    if module.feature_att_8.__class__.__name__ == "ForwardHelper":
        if fold_feature_att_8_batch_norm:
            conv1 = _folded_feature_att_8_forward_helper(
                graph,
                conv1,
                features[1],
                module.feature_att_8,
            )
        else:
            conv1 = graph.forward_helper(conv1, features[1], module.feature_att_8)
    else:
        conv1 = _feature_attention(graph, conv1, features[1], module.feature_att_8)

    conv2 = graph.module(conv1, module.conv2)
    if fold_remaining_safe_batch_norm:
        conv2 = _folded_remaining_safe_feature_att_16(
            graph,
            conv2,
            module.feature_att_16,
        )
    elif module.feature_att_16.__class__.__name__ == "ForwardHelper":
        conv2 = graph.forward_helper(conv2, features[2], module.feature_att_16)
    else:
        conv2 = _feature_attention(graph, conv2, features[2], module.feature_att_16)

    conv3 = (
        _folded_remaining_safe_conv3(graph, conv2, module.conv3)
        if fold_remaining_safe_batch_norm
        else graph.sequential(conv2, module.conv3)
    )
    conv3 = _feature_attention(graph, conv3, features[3], module.feature_att_32)

    if module.post32_to_16 is None:
        conv3_up = graph.basic_conv(conv3, module.conv3_up)
        conv2 = graph.concat((conv3_up, conv2), 1)
        conv2 = graph.sequential(conv2, module.agg_0)
        conv2 = _feature_attention(graph, conv2, features[2], module.feature_att_up_16)
    else:
        conv2 = _post_forward_helper(
            graph,
            conv2,
            conv3,
            features[2],
            module.post32_to_16,
            fold_remaining_safe_batch_norm=fold_remaining_safe_batch_norm,
        )

    if module.post16_to_8 is None:
        conv2_up = graph.basic_conv(conv2, module.conv2_up)
        conv1 = graph.concat((conv2_up, conv1), 1)
        conv1 = graph.sequential(conv1, module.agg_1)
        conv1 = _feature_attention(graph, conv1, features[1], module.feature_att_up_8)
    else:
        conv1 = _post_forward_helper(
            graph,
            conv1,
            conv2,
            features[1],
            module.post16_to_8,
            fold_post16_to_8_batch_norm=fold_post16_to_8_batch_norm,
        )

    output = (
        _folded_basic_conv_full_volume_leaky(
            graph,
            conv1,
            module.conv1_up,
            path="cost_agg.conv1_up",
            name="cost_agg_conv1_up_leaky",
        )
        if full_volume_leaky_all3_plugin
        else graph.basic_conv(
            conv1,
            module.conv1_up,
            fold_batch_norm=fold_full_volume_batch_norm,
        )
    )
    if module.post8_to_4 is None:
        patch = graph.sequential(volume, module.conv_patch)
        patch = _cost_attention(graph, patch, module.atts["4"])
        target_shape = tuple(int(dim) for dim in output.shape)
        patch = graph.resize(patch, target_shape, mode="trilinear", align_corners=False)
        output = graph.sequential(graph.add(output, patch), module.conv_out)
    else:
        output = _post_forward_helper(
            graph,
            volume,
            output,
            features[0],
            module.post8_to_4,
            fold_batch_norm=fold_full_volume_batch_norm,
            post8_sum_plugin=post8_sum_plugin,
            post8_sum_tile_positions=post8_sum_tile_positions,
            full_volume_leaky_plugin=full_volume_leaky_all3_plugin,
        )
    return output


def _disparity_regression(graph: NativeGraph, logits: Any, disparities: int) -> Any:
    # logits: [B,1,D,H,W] -> probabilities: [B,D,H,W]
    shape = tuple(int(dim) for dim in logits.shape)
    logits = graph.reshape(logits, (shape[0], shape[2], shape[3], shape[4]))
    logits = graph.cast(logits, graph.trt.float32)
    probabilities = graph.softmax(logits, 1)
    values = np.arange(disparities, dtype=np.float32).reshape(1, disparities, 1, 1)
    value_tensor = graph.constant(values, values.shape)
    return graph.reduce_sum(graph.mul(probabilities, value_tensor), (1,), keep_dims=True)


def _channel_attention(graph: NativeGraph, tensor: Any, module: Any) -> Any:
    average = graph.reduce_avg(tensor, (2, 3), keep_dims=True)
    maximum = graph.reduce_max(tensor, (2, 3), keep_dims=True)
    average = graph.sequential(average, module.fc)
    maximum = graph.sequential(maximum, module.fc)
    return graph.activation(graph.add(average, maximum), "sigmoid")


def _spatial_attention(graph: NativeGraph, tensor: Any, module: Any) -> Any:
    input_shape = tuple(int(dim) for dim in tensor.shape)
    expected_input_shape = (1, 48, 176, 176)
    if input_shape != expected_input_shape:
        raise RuntimeError(
            f"spatial-attention input has shape {input_shape}, expected {expected_input_shape}"
        )
    from .native_plugin_builder import add_spatial_attention_reduce_plugin

    average, maximum = add_spatial_attention_reduce_plugin(
        graph.network,
        graph.cast(tensor, graph.trt.float16),
        trt_module=graph.trt,
    )
    expected_output_shape = (1, 1, 176, 176)
    for name, reduced in (("average", average), ("maximum", maximum)):
        if tuple(int(dim) for dim in reduced.shape) != expected_output_shape:
            raise RuntimeError(
                f"spatial-attention {name} output has shape {tuple(reduced.shape)}, "
                f"expected {expected_output_shape}"
            )
    attention = graph.conv2d(graph.concat((average, maximum), 1), module.samconv)
    return graph.activation(attention, "sigmoid")


def _all_pairs_correlation(graph: NativeGraph, left: Any, right: Any) -> Any:
    batch, channels, height, width = (int(dim) for dim in left.shape)
    left = graph.normalize_l2(left, 1)
    right = graph.normalize_l2(right, 1)
    left = graph.reshape(graph.transpose(left, (0, 2, 3, 1)), (batch * height, width, channels))
    right = graph.reshape(graph.transpose(right, (0, 2, 3, 1)), (batch * height, width, channels))
    correlation = graph.matmul(
        left,
        right,
        op_rhs=graph.trt.MatrixOperation.TRANSPOSE,
    )
    return graph.reshape(correlation, (batch * height * width, 1, 1, width))


def _correlation_pyramid(
    graph: NativeGraph,
    left: Any,
    right: Any,
    levels: int,
) -> list[Any]:
    correlation = _all_pairs_correlation(graph, left, right)
    correlation_pyramid = [correlation]
    for _ in range(1, levels):
        correlation = graph.pool2d(correlation, kind="avg", window=(1, 2), stride=(1, 2))
        correlation_pyramid.append(correlation)
    return correlation_pyramid


def _pack_geometry_convc1_parameters(module: Any) -> tuple[np.ndarray, np.ndarray]:
    expected_attributes = {
        "in_channels": 522,
        "out_channels": 56,
        "kernel_size": (1, 1),
        "stride": (1, 1),
        "padding": (0, 0),
        "dilation": (1, 1),
        "groups": 1,
    }
    for attribute, expected in expected_attributes.items():
        actual = getattr(module, attribute, None)
        if isinstance(expected, tuple) and actual is not None:
            actual = tuple(int(item) for item in actual)
        elif actual is not None:
            actual = int(actual)
        if actual != expected:
            raise RuntimeError(
                "direct-volume geometry-convc1 is specialized for the distilled checkpoint; "
                f"convc1.{attribute} is {actual!r}, expected {expected!r}"
            )

    weight = NativeGraph._array(module.weight, np.float16)
    bias_value = getattr(module, "bias", None)
    if tuple(weight.shape) != (56, 522, 1, 1) or bias_value is None:
        raise RuntimeError(
            "direct-volume geometry-convc1 requires weight (56, 522, 1, 1) and bias (56,)"
        )
    bias = NativeGraph._array(bias_value, np.float16)
    if tuple(bias.shape) != (56,):
        raise RuntimeError(
            f"direct-volume geometry-convc1 bias has shape {bias.shape}, expected (56,)"
        )

    # WMMA consumes B as KxN column-major. Contiguous [N,K] has that byte layout.
    packed_weight = np.zeros((64, 528), dtype=np.float16)
    packed_weight[:56, :522] = weight[:, :, 0, 0]
    packed_bias = np.zeros((64,), dtype=np.float16)
    packed_bias[:56] = bias
    return packed_weight, packed_bias


def _geometry_convc1_constants(graph: NativeGraph, module: Any) -> tuple[Any, Any]:
    if graph.work_trt_dtype != graph.trt.float16:
        raise RuntimeError("direct-volume geometry-convc1 requires an FP16 TensorRT graph")
    packed_weight, packed_bias = _pack_geometry_convc1_parameters(module)
    weight_tensor = graph.constant(
        packed_weight,
        packed_weight.shape,
        dtype=np.float16,
        target_dtype=graph.trt.float16,
    )
    bias_tensor = graph.constant(
        packed_bias,
        packed_bias.shape,
        dtype=np.float16,
        target_dtype=graph.trt.float16,
    )
    return weight_tensor, bias_tensor


def _geometry_volume_convc1_features(
    graph: NativeGraph,
    disparity: Any,
    volume: Any,
    correlation_pyramid: list[Any],
    packed_weight: Any,
    packed_bias: Any,
    *,
    radius: int,
    batch: int,
    height: int,
    width: int,
    iteration: int,
) -> Any:
    if (batch, height, width) != (1, 176, 176):
        raise RuntimeError(
            "direct-volume geometry-convc1 is specialized for batch/height/width "
            f"(1, 176, 176), got {(batch, height, width)}"
        )
    if radius != 4:
        raise RuntimeError(
            f"direct-volume geometry-convc1 is specialized for radius=4, got {radius}"
        )
    disparity_shape = tuple(int(dim) for dim in disparity.shape)
    volume_shape = tuple(int(dim) for dim in volume.shape)
    correlation_shapes = [tuple(int(dim) for dim in tensor.shape) for tensor in correlation_pyramid]
    expected_correlations = [(30976, 1, 1, 176), (30976, 1, 1, 88)]
    if disparity_shape != (1, 1, 176, 176):
        raise RuntimeError(
            f"direct-volume geometry-convc1 disparity has shape {disparity_shape}, "
            "expected (1, 1, 176, 176)"
        )
    if volume_shape != (1, 28, 48, 176, 176):
        raise RuntimeError(
            f"direct-volume geometry-convc1 volume has shape {volume_shape}, "
            "expected post-cost-aggregation shape (1, 28, 48, 176, 176)"
        )
    if correlation_shapes != expected_correlations:
        raise RuntimeError(
            "direct-volume geometry-convc1 correlations have shapes "
            f"{correlation_shapes}, expected {expected_correlations}"
        )

    from .native_plugin_builder import add_geometry_volume_convc1_plugin

    output = add_geometry_volume_convc1_plugin(
        graph.network,
        disparity,
        graph.cast(volume, graph.trt.float16),
        correlation_pyramid[0],
        correlation_pyramid[1],
        packed_weight,
        packed_bias,
        trt_module=graph.trt,
        name=f"geometry_volume_convc1_{iteration}",
    )
    if tuple(int(dim) for dim in output.shape) != (1, 56, 176, 176):
        raise RuntimeError(
            f"direct-volume geometry-convc1 output has shape {tuple(output.shape)}, "
            "expected (1, 56, 176, 176)"
        )
    return output


def _motion_encoder(graph: NativeGraph, disparity: Any, cor: Any, module: Any) -> Any:
    cor = graph.activation(graph.conv2d(cor, module.convc2), "relu")
    disp_work = graph.cast(disparity, graph.work_trt_dtype)
    disp = graph.activation(graph.conv2d(disp_work, module.convd1), "relu")
    disp = graph.activation(graph.conv2d(disp, module.convd2), "relu")
    output = graph.activation(graph.conv2d(graph.concat((cor, disp), 1), module.conv), "relu")
    return graph.concat((output, disp_work), 1)


def _require_disp_head_attributes(
    module: Any,
    path: str,
    expected_attributes: dict[str, Any],
) -> None:
    for attribute, expected in expected_attributes.items():
        actual = getattr(module, attribute, None)
        if isinstance(expected, tuple) and actual is not None:
            actual = tuple(int(item) for item in actual)
        elif isinstance(expected, int) and actual is not None:
            actual = int(actual)
        if actual != expected:
            raise RuntimeError(
                "DispHead NCHW pointwise is specialized for the distilled checkpoint; "
                f"{path}.{attribute} is {actual!r}, expected {expected!r}"
            )


def _require_disp_head_parameter(
    module: Any,
    path: str,
    name: str,
    expected_shape: tuple[int, ...],
) -> None:
    value = getattr(module, name, None)
    if value is None:
        raise RuntimeError(f"DispHead NCHW pointwise requires {path}.{name}")
    array = NativeGraph._array(value)
    if tuple(array.shape) != expected_shape:
        raise RuntimeError(
            f"DispHead NCHW pointwise requires {path}.{name} shape "
            f"{expected_shape}, got {array.shape}"
        )
    if array.dtype != np.float32:
        raise RuntimeError(
            f"DispHead NCHW pointwise requires FP32 checkpoint {path}.{name}, got {array.dtype}"
        )


def _disp_head_nchw_pointwise_layers(module: Any) -> tuple[Any, ...]:
    if module.__class__.__name__ != "Sequential":
        raise RuntimeError(
            "DispHead NCHW pointwise requires disp_head.conv to be a Sequential; "
            f"got {module.__class__.__name__!r}"
        )
    layers = tuple(module)
    expected_topology = (
        "Conv2d",
        "ReLU",
        "EdgeNextConvEncoder",
        "EdgeNextConvEncoder",
        "Conv2d",
    )
    actual_topology = tuple(layer.__class__.__name__ for layer in layers)
    if actual_topology != expected_topology:
        raise RuntimeError(
            "DispHead NCHW pointwise requires the distilled topology "
            f"{expected_topology!r}, got {actual_topology!r}"
        )

    convolution_specs = (
        (
            layers[0],
            "disp_head.conv.0",
            60,
            36,
            (3, 3),
            (1, 1),
            1,
        ),
        (
            layers[4],
            "disp_head.conv.4",
            36,
            1,
            (3, 3),
            (1, 1),
            1,
        ),
    )
    for (
        convolution,
        path,
        input_channels,
        output_channels,
        kernel_size,
        padding,
        groups,
    ) in convolution_specs:
        _require_disp_head_attributes(
            convolution,
            path,
            {
                "in_channels": input_channels,
                "out_channels": output_channels,
                "kernel_size": kernel_size,
                "stride": (1, 1),
                "padding": padding,
                "dilation": (1, 1),
                "groups": groups,
            },
        )
        _require_disp_head_parameter(
            convolution,
            path,
            "weight",
            (output_channels, input_channels // groups, *kernel_size),
        )
        _require_disp_head_parameter(convolution, path, "bias", (output_channels,))

    for block_index, (block, hidden_width) in enumerate(zip(layers[2:4], (212, 244))):
        path = f"disp_head.conv.{block_index + 2}"
        if block.norm.__class__.__name__ != "Identity":
            raise RuntimeError(
                "DispHead NCHW pointwise requires Identity normalization; "
                f"{path}.norm is {block.norm.__class__.__name__!r}"
            )
        if (
            block.act.__class__.__name__ != "GELU"
            or getattr(block.act, "approximate", None) != "none"
        ):
            raise RuntimeError(
                "DispHead NCHW pointwise requires exact GELU(approximate='none'); "
                f"{path}.act is {block.act.__class__.__name__!r}/"
                f"{getattr(block.act, 'approximate', None)!r}"
            )
        _require_disp_head_attributes(
            block.dwconv,
            f"{path}.dwconv",
            {
                "in_channels": 36,
                "out_channels": 36,
                "kernel_size": (7, 7),
                "stride": (1, 1),
                "padding": (3, 3),
                "dilation": (1, 1),
                "groups": 36,
            },
        )
        _require_disp_head_attributes(
            block.pwconv1,
            f"{path}.pwconv1",
            {"in_features": 36, "out_features": hidden_width},
        )
        _require_disp_head_attributes(
            block.pwconv2,
            f"{path}.pwconv2",
            {"in_features": hidden_width, "out_features": 36},
        )
        for child, child_path, expected_shapes in (
            (
                block.dwconv,
                f"{path}.dwconv",
                {"weight": (36, 1, 7, 7), "bias": (36,)},
            ),
            (
                block.pwconv1,
                f"{path}.pwconv1",
                {"weight": (hidden_width, 36), "bias": (hidden_width,)},
            ),
            (
                block.pwconv2,
                f"{path}.pwconv2",
                {"weight": (36, hidden_width), "bias": (36,)},
            ),
        ):
            for name, shape in expected_shapes.items():
                _require_disp_head_parameter(child, child_path, name, shape)
        _require_disp_head_parameter(block, path, "gamma", (36,))
    return layers


def _disp_head_delta(
    graph: NativeGraph,
    hidden: Any,
    module: Any,
    *,
    nchw_pointwise: bool,
    fold_gamma: bool = False,
    second_gelu_tanh: bool = False,
) -> Any:
    if fold_gamma and not nchw_pointwise:
        raise RuntimeError("DispHead gamma folding requires NCHW pointwise lowering")
    if second_gelu_tanh and not nchw_pointwise:
        raise RuntimeError("DispHead second GELU_TANH requires NCHW pointwise lowering")
    if not nchw_pointwise:
        return graph.sequential(hidden, module)

    layers = _disp_head_nchw_pointwise_layers(module)
    if graph.work_trt_dtype != graph.trt.float16:
        raise RuntimeError("DispHead NCHW pointwise requires an FP16 TensorRT graph")
    hidden_shape = tuple(int(dimension) for dimension in hidden.shape)
    expected_hidden_shape = (1, 60, 176, 176)
    if hidden_shape != expected_hidden_shape or hidden.dtype != graph.trt.float16:
        raise RuntimeError(
            "DispHead NCHW pointwise requires hidden FP16 shape "
            f"{expected_hidden_shape}, got {hidden_shape}/{hidden.dtype!r}"
        )

    output = graph.module(hidden, layers[0])
    output = graph.module(output, layers[1])
    expected_block_shape = (1, 36, 176, 176)
    if tuple(int(dimension) for dimension in output.shape) != expected_block_shape:
        raise RuntimeError(
            "DispHead NCHW pointwise stem output has shape "
            f"{tuple(output.shape)}, expected {expected_block_shape}"
        )
    if output.dtype != graph.trt.float16:
        raise RuntimeError(
            f"DispHead NCHW pointwise stem output must be FP16, got {output.dtype!r}"
        )

    for block_index, block in enumerate(layers[2:4]):
        output = graph.edge_next_encoder(
            output,
            block,
            nchw_pointwise=True,
            fold_gamma=fold_gamma,
            gelu_approximate="tanh" if second_gelu_tanh and block_index == 1 else "none",
        )
        if tuple(int(dimension) for dimension in output.shape) != expected_block_shape:
            raise RuntimeError(
                "DispHead NCHW pointwise block output has shape "
                f"{tuple(output.shape)}, expected {expected_block_shape}"
            )
        if output.dtype != graph.trt.float32:
            raise RuntimeError(
                f"DispHead NCHW pointwise block output must be FP32, got {output.dtype!r}"
            )

    output = graph.module(output, layers[4])
    expected_output_shape = (1, 1, 176, 176)
    if tuple(int(dimension) for dimension in output.shape) != expected_output_shape:
        raise RuntimeError(
            "DispHead NCHW pointwise output has shape "
            f"{tuple(output.shape)}, expected {expected_output_shape}"
        )
    if output.dtype != graph.trt.float16:
        raise RuntimeError(f"DispHead NCHW pointwise output must be FP16, got {output.dtype!r}")
    return output


def _raft_gru(graph: NativeGraph, hidden: Any, x: Any, hx: Any, module: Any) -> Any:
    batch, channels, height, width = (int(dim) for dim in hidden.shape)
    gates = graph.stacked_conv2d(hx, (module.convz, module.convr))
    gate_shape = (batch, channels, height, width)
    update = graph.activation(
        graph.slice(gates, (0, 0, 0, 0), gate_shape),
        "sigmoid",
    )
    reset = graph.activation(
        graph.slice(gates, (0, channels, 0, 0), gate_shape),
        "sigmoid",
    )
    proposal_input = graph.concat((graph.mul(reset, hidden), x), 1)
    proposal = graph.activation(graph.conv2d(proposal_input, module.convq), "tanh")
    one = graph.scalar(1.0, len(tuple(update.shape)), like=update)
    return graph.add(
        graph.mul(graph.sub(one, update), hidden),
        graph.mul(update, proposal),
    )


def _selective_gru(
    graph: NativeGraph,
    attention: Any,
    hidden: Any,
    motion: Any,
    module: Any,
) -> Any:
    x = graph.sequential(motion, module.conv0)
    hx = graph.sequential(graph.concat((x, hidden), 1), module.conv1)
    small = _raft_gru(graph, hidden, x, hx, module.small_gru)
    large = _raft_gru(graph, hidden, x, hx, module.large_gru)
    one = graph.scalar(1.0, len(tuple(attention.shape)), like=attention)
    return graph.add(
        graph.mul(small, attention),
        graph.mul(large, graph.sub(one, attention)),
    )


def _context_upsample(graph: NativeGraph, disparity: Any, weights: Any) -> Any:
    batch, _, height, width = (int(dim) for dim in disparity.shape)
    four = graph.scalar(4.0, 4, like=disparity)
    disparity = graph.mul(disparity, four)
    padding = graph.network.add_padding_nd(disparity, (1, 1), (1, 1)).get_output(0)
    neighborhoods = []
    for row in range(3):
        for column in range(3):
            neighborhoods.append(
                graph.slice(
                    padding,
                    (0, 0, row, column),
                    (batch, 1, height, width),
                )
            )
    unfolded = graph.concat(neighborhoods, 1)
    unfolded = graph.resize(
        unfolded,
        (batch, 9, height * 4, width * 4),
        mode="nearest",
    )
    output = graph.reduce_sum(graph.mul(unfolded, graph.cast(weights, unfolded.dtype)), (1,))
    return graph.reshape(output, (batch, 1, height * 4, width * 4))


def _upsample_disparity(
    graph: NativeGraph,
    disparity: Any,
    mask_feature: Any,
    stem_2x: Any,
    model: Any,
) -> Any:
    upsampled_mask = graph.basic_conv(mask_feature, model.spx_2_gru.conv1)
    upsampled_mask = graph.concat((upsampled_mask, stem_2x), 1)
    upsampled_mask = graph.basic_conv(upsampled_mask, model.spx_2_gru.conv2)
    weights = graph.deconv2d(upsampled_mask, model.spx_gru[0])
    weights = graph.softmax(graph.cast(weights, graph.trt.float32), 1)
    return graph.cast(_context_upsample(graph, disparity, weights), graph.trt.float32)


def add_post_graph(
    graph: NativeGraph,
    model: Any,
    inputs: dict[str, Any],
    *,
    max_disparity: int,
    valid_iters: int,
) -> Any:
    """Add the full distilled post network and return FP32 disparity."""
    if max_disparity != 192:
        raise ValueError("Fast Foundation Stereo native graph is specialized for max_disparity=192")
    if valid_iters != 8:
        raise ValueError("Fast Foundation Stereo native graph is specialized for valid_iters=8")
    disparities = max_disparity // 4
    fold_full_volume_batch_norm = _full_volume_conv_bn_folding_enabled()
    fold_post16_to_8_batch_norm = _post16_to_8_conv_bn_folding_enabled()
    fold_feature_att_8_batch_norm = _feature_att_8_conv_bn_folding_enabled()
    fold_remaining_safe_batch_norm = _remaining_safe_conv_bn_folding_enabled()
    post8_sum_plugin = _post8_sum_plugin_enabled()
    post8_sum_tile_positions = _post8_sum_tile_positions() if post8_sum_plugin else 32
    full_volume_leaky_all3_plugin = _full_volume_leaky_all3_plugin_enabled()
    disp_head_nchw_pointwise = _disp_head_nchw_pointwise_enabled()
    disp_head_fold_gamma = _disp_head_fold_gamma_enabled()
    disp_head_second_gelu_tanh = _disp_head_second_gelu_tanh_enabled()
    if disp_head_fold_gamma and not disp_head_nchw_pointwise:
        raise RuntimeError(
            "TRTMC_FAST_FOUNDATION_STEREO_DISP_HEAD_FOLD_GAMMA=1 requires "
            "TRTMC_FAST_FOUNDATION_STEREO_DISP_HEAD_NCHW_POINTWISE=1"
        )
    if disp_head_second_gelu_tanh and not disp_head_nchw_pointwise:
        raise RuntimeError(
            "TRTMC_FAST_FOUNDATION_STEREO_DISP_HEAD_SECOND_GELU_TANH=1 requires "
            "TRTMC_FAST_FOUNDATION_STEREO_DISP_HEAD_NCHW_POINTWISE=1"
        )
    if full_volume_leaky_all3_plugin:
        if not fold_full_volume_batch_norm:
            raise RuntimeError(
                "TRTMC_FAST_FOUNDATION_STEREO_FULL_VOLUME_LEAKY_ALL3_PLUGIN=1 requires "
                "TRTMC_FAST_FOUNDATION_STEREO_FOLD_FULL_VOLUME_BN=1"
            )
        _require_full_volume_leaky_all3_topology(model)
    features = tuple(
        graph.cast(inputs[name], graph.work_trt_dtype)
        for name in (
            "features_left_04",
            "features_left_08",
            "features_left_16",
            "features_left_32",
        )
    )
    right = graph.cast(inputs["features_right_04"], graph.work_trt_dtype)
    stem_2x = graph.cast(inputs["stem_2x"], graph.work_trt_dtype)

    left_projected = graph.conv2d(features[0], model.proj_cmb)
    right_projected = graph.conv2d(right, model.proj_cmb)

    from .native_plugin_builder import add_combined_volume_plugin

    # The fused CUDA implementation is intentionally FP16 at its tensor boundary,
    # while retaining FP32 accumulation for groupwise correlation. The family
    # builder rejects other precision modes instead of silently weakening a public
    # FP32 contract.
    gwc_reference = graph.cast(features[0], graph.trt.float16)
    gwc_target = graph.cast(right, graph.trt.float16)
    left_projected = graph.cast(left_projected, graph.trt.float16)
    right_projected = graph.cast(right_projected, graph.trt.float16)
    combined_volume = add_combined_volume_plugin(
        graph.network,
        gwc_reference,
        gwc_target,
        left_projected,
        right_projected,
        trt_module=graph.trt,
    )
    combined_volume = graph.module(combined_volume, model.corr_stem)
    if full_volume_leaky_all3_plugin:
        combined_volume = _folded_corr_feature_att_with_full_volume_leaky(
            graph,
            combined_volume,
            features[0],
            model.corr_feature_att,
        )
    elif model.corr_feature_att.__class__.__name__ == "ForwardHelper":
        combined_volume = graph.forward_helper(
            combined_volume,
            features[0],
            model.corr_feature_att,
            fold_batch_norm=fold_full_volume_batch_norm,
        )
    else:
        combined_volume = _feature_attention(
            graph, combined_volume, features[0], model.corr_feature_att
        )
    # This 28-channel post-cost-aggregation tensor, not the 32-channel GWC
    # output above, is the direct recurrent plugin's DHWC8 volume input.
    geometry_volume = _cost_aggregation(
        graph,
        combined_volume,
        features,
        model.cost_agg,
        fold_full_volume_batch_norm=fold_full_volume_batch_norm,
        fold_post16_to_8_batch_norm=fold_post16_to_8_batch_norm,
        fold_feature_att_8_batch_norm=fold_feature_att_8_batch_norm,
        fold_remaining_safe_batch_norm=fold_remaining_safe_batch_norm,
        post8_sum_plugin=post8_sum_plugin,
        post8_sum_tile_positions=post8_sum_tile_positions,
        full_volume_leaky_all3_plugin=full_volume_leaky_all3_plugin,
    )

    if model.classifier.__class__.__name__ == "ForwardHelper":
        logits = graph.forward_helper(geometry_volume, features[0], model.classifier)
    else:
        logits = graph.sequential(geometry_volume, model.classifier)
    disparity = _disparity_regression(graph, logits, disparities)

    context = (
        _folded_remaining_safe_context(graph, features[0], model.cnet.conv04)
        if fold_remaining_safe_batch_norm
        else [graph.basic_conv(features[0], layer) for layer in model.cnet.conv04]
    )
    hidden = graph.activation(context[0], "tanh")
    inp = graph.activation(context[1], "relu")
    inp = graph.mul(inp, _channel_attention(graph, inp, model.cam))
    attention = _spatial_attention(graph, inp, model.sam)

    correlation_pyramid = _correlation_pyramid(
        graph,
        features[0],
        right,
        int(model.args.corr_levels),
    )
    packed_weight, packed_bias = _geometry_convc1_constants(
        graph, model.update_block.encoder.convc1
    )
    batch, _, height, width = (int(dim) for dim in features[0].shape)
    mask_feature = None
    for iteration in range(valid_iters):
        cor = _geometry_volume_convc1_features(
            graph,
            disparity,
            geometry_volume,
            correlation_pyramid,
            packed_weight,
            packed_bias,
            radius=int(model.args.corr_radius),
            batch=batch,
            height=height,
            width=width,
            iteration=iteration,
        )
        motion = _motion_encoder(graph, disparity, cor, model.update_block.encoder)
        gru_input = graph.concat((inp, motion), 1)
        hidden = _selective_gru(
            graph,
            attention,
            hidden,
            gru_input,
            model.update_block.gru04,
        )
        delta = _disp_head_delta(
            graph,
            hidden,
            model.update_block.disp_head.conv,
            nchw_pointwise=disp_head_nchw_pointwise,
            fold_gamma=disp_head_fold_gamma,
            second_gelu_tanh=disp_head_second_gelu_tanh,
        )
        disparity = graph.add(disparity, graph.cast(delta, graph.trt.float32))
        if iteration == valid_iters - 1:
            mask_feature = graph.sequential(hidden, model.update_block.mask)
            quarter = graph.scalar(0.25, len(tuple(mask_feature.shape)), like=mask_feature)
            mask_feature = graph.mul(mask_feature, quarter)

    if mask_feature is None:
        raise AssertionError("valid_iters must produce a final mask feature")
    return _upsample_disparity(
        graph,
        disparity,
        mask_feature,
        stem_2x,
        model,
    )
