# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-native post graph for Fast Foundation Stereo."""

from __future__ import annotations

from typing import Any

import numpy as np

from .native_graph import NativeGraph


def _feature_attention(graph: NativeGraph, volume: Any, feature: Any, module: Any) -> Any:
    return graph.feature_attention(volume, feature, module)


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


def _post_forward_helper(
    graph: NativeGraph,
    skip: Any,
    lower: Any,
    feature: Any,
    module: Any,
) -> Any:
    output = lower
    for child in module.upsample:
        if child.__class__.__name__ == "CostVolumeDisparityAttention":
            output = _cost_attention(graph, output, child)
        else:
            output = graph.module(output, child)
    output = graph.add(output, skip) if module.op == "sum" else graph.concat((output, skip), 1)
    for child in module.out:
        if child.__class__.__name__ == "FeatureAtt":
            output = _feature_attention(graph, output, feature, child)
        else:
            output = graph.module(output, child)
    return output


def _cost_aggregation(
    graph: NativeGraph,
    volume: Any,
    features: tuple[Any, Any, Any, Any],
    module: Any,
) -> Any:
    # The serialized distilled checkpoint replaces several constructor modules
    # with ForwardHelper/PostForwardHelper instances.  Follow the live module
    # objects rather than reconstructing the unpruned source topology.
    conv1 = graph.module(volume, module.conv1)
    if module.feature_att_8.__class__.__name__ == "ForwardHelper":
        conv1 = graph.forward_helper(conv1, features[1], module.feature_att_8)
    else:
        conv1 = _feature_attention(graph, conv1, features[1], module.feature_att_8)

    conv2 = graph.module(conv1, module.conv2)
    if module.feature_att_16.__class__.__name__ == "ForwardHelper":
        conv2 = graph.forward_helper(conv2, features[2], module.feature_att_16)
    else:
        conv2 = _feature_attention(graph, conv2, features[2], module.feature_att_16)

    conv3 = graph.sequential(conv2, module.conv3)
    conv3 = _feature_attention(graph, conv3, features[3], module.feature_att_32)

    if module.post32_to_16 is None:
        conv3_up = graph.basic_conv(conv3, module.conv3_up)
        conv2 = graph.concat((conv3_up, conv2), 1)
        conv2 = graph.sequential(conv2, module.agg_0)
        conv2 = _feature_attention(graph, conv2, features[2], module.feature_att_up_16)
    else:
        conv2 = _post_forward_helper(graph, conv2, conv3, features[2], module.post32_to_16)

    if module.post16_to_8 is None:
        conv2_up = graph.basic_conv(conv2, module.conv2_up)
        conv1 = graph.concat((conv2_up, conv1), 1)
        conv1 = graph.sequential(conv1, module.agg_1)
        conv1 = _feature_attention(graph, conv1, features[1], module.feature_att_up_8)
    else:
        conv1 = _post_forward_helper(graph, conv1, conv2, features[1], module.post16_to_8)

    output = graph.basic_conv(conv1, module.conv1_up)
    if module.post8_to_4 is None:
        patch = graph.sequential(volume, module.conv_patch)
        patch = _cost_attention(graph, patch, module.atts["4"])
        target_shape = tuple(int(dim) for dim in output.shape)
        patch = graph.resize(patch, target_shape, mode="trilinear", align_corners=False)
        output = graph.sequential(graph.add(output, patch), module.conv_out)
    else:
        output = _post_forward_helper(graph, volume, output, features[0], module.post8_to_4)
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


def _geometry_pyramids(
    graph: NativeGraph,
    left: Any,
    right: Any,
    volume: Any,
    levels: int,
) -> tuple[list[Any], list[Any]]:
    batch, channels, disparities, height, width = (int(dim) for dim in volume.shape)
    geometry = graph.cast(volume, graph.trt.float32)
    geometry = graph.transpose(geometry, (0, 3, 4, 1, 2))
    geometry = graph.reshape(geometry, (batch * height * width, channels, 1, disparities))
    correlation = _all_pairs_correlation(graph, left, right)
    geometry_pyramid = [geometry]
    correlation_pyramid = [correlation]
    for _ in range(1, levels):
        geometry = graph.pool2d(geometry, kind="avg", window=(1, 2), stride=(1, 2))
        correlation = graph.pool2d(correlation, kind="avg", window=(1, 2), stride=(1, 2))
        geometry_pyramid.append(geometry)
        correlation_pyramid.append(correlation)
    return geometry_pyramid, correlation_pyramid


def _grid_sample_1d(graph: NativeGraph, image: Any, x_coordinates: Any) -> Any:
    width = int(image.shape[-1])
    rank = len(tuple(x_coordinates.shape))
    scale = graph.scalar(2.0 / float(width - 1), rank, like=x_coordinates)
    one = graph.scalar(1.0, rank, like=x_coordinates)
    normalized_x = graph.sub(graph.mul(x_coordinates, scale), one)
    normalized_y = graph.constant(
        np.zeros(tuple(int(dim) for dim in normalized_x.shape), dtype=np.float32),
        tuple(int(dim) for dim in normalized_x.shape),
    )
    grid = graph.concat((normalized_x, normalized_y), 3)
    layer = graph.network.add_grid_sample(image, grid)
    layer.interpolation_mode = graph.trt.InterpolationMode.LINEAR
    layer.align_corners = True
    layer.sample_mode = graph.trt.SampleMode.FILL
    return layer.get_output(0)


def _geometry_features(
    graph: NativeGraph,
    disparity: Any,
    geometry_pyramid: list[Any],
    correlation_pyramid: list[Any],
    *,
    radius: int,
    batch: int,
    height: int,
    width: int,
) -> Any:
    pixels = batch * height * width
    disparity_flat = graph.reshape(disparity, (pixels, 1, 1, 1))
    dx = np.arange(-radius, radius + 1, dtype=np.float32).reshape(1, 1, -1, 1)
    dx_tensor = graph.constant(dx, dx.shape)
    x_base = np.broadcast_to(
        np.arange(width, dtype=np.float32).reshape(1, 1, width),
        (batch, height, width),
    ).reshape(pixels, 1, 1, 1)
    coordinate_tensor = graph.constant(x_base, x_base.shape)
    outputs = []
    for level, (geometry, correlation) in enumerate(zip(geometry_pyramid, correlation_pyramid)):
        divisor = graph.scalar(float(1 << level), 4, like=disparity_flat)
        disparity_level = graph.div(disparity_flat, divisor)
        geometry_x = graph.add(disparity_level, dx_tensor)
        geometry_sample = _grid_sample_1d(graph, geometry, geometry_x)
        geometry_channels = int(geometry.shape[1]) * (2 * radius + 1)
        geometry_sample = graph.reshape(geometry_sample, (batch, height, width, geometry_channels))

        correlation_x = graph.add(
            graph.sub(graph.div(coordinate_tensor, divisor), disparity_level),
            dx_tensor,
        )
        correlation_sample = _grid_sample_1d(graph, correlation, correlation_x)
        correlation_sample = graph.reshape(
            correlation_sample, (batch, height, width, 2 * radius + 1)
        )
        outputs.extend((geometry_sample, correlation_sample))
    return graph.transpose(graph.concat(outputs, 3), (0, 3, 1, 2))


def _motion_encoder(graph: NativeGraph, disparity: Any, correlation: Any, module: Any) -> Any:
    correlation = graph.cast(correlation, graph.work_trt_dtype)
    cor = graph.activation(graph.conv2d(correlation, module.convc1), "relu")
    cor = graph.activation(graph.conv2d(cor, module.convc2), "relu")
    disp_work = graph.cast(disparity, graph.work_trt_dtype)
    disp = graph.activation(graph.conv2d(disp_work, module.convd1), "relu")
    disp = graph.activation(graph.conv2d(disp, module.convd2), "relu")
    output = graph.activation(graph.conv2d(graph.concat((cor, disp), 1), module.conv), "relu")
    return graph.concat((output, disp_work), 1)


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
    combined = add_combined_volume_plugin(
        graph.network,
        gwc_reference,
        gwc_target,
        left_projected,
        right_projected,
        trt_module=graph.trt,
    )
    combined = graph.module(combined, model.corr_stem)
    if model.corr_feature_att.__class__.__name__ == "ForwardHelper":
        combined = graph.forward_helper(combined, features[0], model.corr_feature_att)
    else:
        combined = _feature_attention(graph, combined, features[0], model.corr_feature_att)
    combined = _cost_aggregation(graph, combined, features, model.cost_agg)

    if model.classifier.__class__.__name__ == "ForwardHelper":
        logits = graph.forward_helper(combined, features[0], model.classifier)
    else:
        logits = graph.sequential(combined, model.classifier)
    disparity = _disparity_regression(graph, logits, disparities)

    context = [graph.basic_conv(features[0], layer) for layer in model.cnet.conv04]
    hidden = graph.activation(context[0], "tanh")
    inp = graph.activation(context[1], "relu")
    inp = graph.mul(inp, _channel_attention(graph, inp, model.cam))
    attention = _spatial_attention(graph, inp, model.sam)

    geometry_pyramid, correlation_pyramid = _geometry_pyramids(
        graph,
        features[0],
        right,
        combined,
        int(model.args.corr_levels),
    )
    batch, _, height, width = (int(dim) for dim in features[0].shape)
    mask_feature = None
    for iteration in range(valid_iters):
        geometry = _geometry_features(
            graph,
            disparity,
            geometry_pyramid,
            correlation_pyramid,
            radius=int(model.args.corr_radius),
            batch=batch,
            height=height,
            width=width,
        )
        motion = _motion_encoder(graph, disparity, geometry, model.update_block.encoder)
        gru_input = graph.concat((inp, motion), 1)
        hidden = _selective_gru(
            graph,
            attention,
            hidden,
            gru_input,
            model.update_block.gru04,
        )
        delta = graph.sequential(hidden, model.update_block.disp_head.conv)
        disparity = graph.add(disparity, graph.cast(delta, graph.trt.float32))
        if iteration == valid_iters - 1:
            mask_feature = graph.sequential(hidden, model.update_block.mask)
            quarter = graph.scalar(0.25, len(tuple(mask_feature.shape)), like=mask_feature)
            mask_feature = graph.mul(mask_feature, quarter)

    if mask_feature is None:
        raise AssertionError("valid_iters must produce a final mask feature")
    return _upsample_disparity(graph, disparity, mask_feature, stem_2x, model)
