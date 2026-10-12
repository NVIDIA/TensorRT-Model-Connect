# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build BiRefNet's decoder.

The checkpoint turns on three options that the class makes optional, and all
three are load-bearing at inference:

* ``cxt``: before the decoder, the three finer levels are resized down and
  concatenated onto ``x4``, taking it from 1536 to 2880 channels.
* ``dec_ipt`` with ``dec_ipt_split``: the input image is rearranged
  space-to-depth to match each decoder scale, passed through a two-convolution
  block and concatenated onto that stage's input. This is where the otherwise
  puzzling channel counts come from - 1728 = 1536 + 1536/8, and so on.
* ``out_ref``: each of the three coarser stages multiplies its output by a
  sigmoid gate derived from its own features. The surrounding gradient
  supervision is training-only, but **this multiplication is not**.

Every interpolation in this file uses ``align_corners=True``, matching the
reference.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from . import graph as g
from .aspp_builder import build_aspp_deformable

_BN_EPS = 1e-5


def _conv_bn_relu(network, x, weights, conv_prefix, bn_prefix, dtype, activate=True,
                  padding=(1, 1)):
    out = g.add_conv2d(network, x, weights[f"{conv_prefix}.weight"],
                       weights.get(f"{conv_prefix}.bias"), padding=padding, dtype=dtype)
    if bn_prefix is not None:
        gamma = np.asarray(weights[f"{bn_prefix}.weight"], dtype=np.float32)
        beta = np.asarray(weights[f"{bn_prefix}.bias"], dtype=np.float32)
        mean = np.asarray(weights[f"{bn_prefix}.running_mean"], dtype=np.float32)
        variance = np.asarray(weights[f"{bn_prefix}.running_var"], dtype=np.float32)
        scale = gamma / np.sqrt(variance + _BN_EPS)
        bias = beta - mean * scale
        channels = scale.shape[0]
        out = network.add_elementwise(
            out, g.add_constant(network, (1, channels, 1, 1), scale.reshape(1, -1, 1, 1),
                                dtype=dtype), trt.ElementWiseOperation.PROD).get_output(0)
        out = network.add_elementwise(
            out, g.add_constant(network, (1, channels, 1, 1), bias.reshape(1, -1, 1, 1),
                                dtype=dtype), trt.ElementWiseOperation.SUM).get_output(0)
    return g.add_relu(network, out) if activate else out


def _basic_dec_blk(network, x, weights, prefix, dtype):
    """conv, norm, relu, the deformable ASPP, then conv and norm with no relu."""
    out = _conv_bn_relu(network, x, weights, f"{prefix}.conv_in", f"{prefix}.bn_in", dtype)
    out = build_aspp_deformable(network, out, weights, f"{prefix}.dec_att", dtype)
    return _conv_bn_relu(network, out, weights, f"{prefix}.conv_out", f"{prefix}.bn_out",
                         dtype, activate=False)


def _image_to_patches(network, image, reference_hw):
    """'b c (hg h) (wg w) -> b (c hg wg) h w', the einops rearrangement."""
    shape = tuple(int(v) for v in image.shape)
    channels, height, width = shape[1], shape[2], shape[3]
    grid_h = height // reference_hw[0]
    grid_w = width // reference_hw[1]
    if grid_h == 1 and grid_w == 1:
        return image
    inner_h, inner_w = height // grid_h, width // grid_w
    return g.reshape_permute(
        network, image, (channels, grid_h, inner_h, grid_w, inner_w),
        (0, 1, 3, 2, 4), (1, channels * grid_h * grid_w, inner_h, inner_w))


def _input_branch(network, image, weights, prefix, reference, target_hw, dtype):
    """The dec_ipt branch: patches of the image, resized, through two convs."""
    reference_hw = tuple(int(v) for v in reference.shape)[2:]
    patches = _image_to_patches(network, image, reference_hw)
    resized = g.add_resize_bilinear(network, patches, target_hw, align_corners=True)
    # SimpleConvs is two convolutions with no norm and no activation between.
    inner = g.add_conv2d(network, resized, weights[f"{prefix}.conv1.weight"],
                         weights[f"{prefix}.conv1.bias"], padding=(1, 1), dtype=dtype)
    return g.add_conv2d(network, inner, weights[f"{prefix}.conv_out.weight"],
                        weights[f"{prefix}.conv_out.bias"], padding=(1, 1), dtype=dtype)


def _gdt_gate(network, p, weights, index, dtype):
    """out_ref: scale the stage output by a sigmoid gate built from itself."""
    hidden = _conv_bn_relu(network, p, weights, f"decoder.gdt_convs_{index}.0",
                           f"decoder.gdt_convs_{index}.1", dtype)
    attention = g.add_conv2d(network, hidden,
                             weights[f"decoder.gdt_convs_attn_{index}.0.weight"],
                             weights[f"decoder.gdt_convs_attn_{index}.0.bias"],
                             dtype=dtype)
    gate = network.add_activation(attention, trt.ActivationType.SIGMOID).get_output(0)
    return network.add_elementwise(p, gate, trt.ElementWiseOperation.PROD).get_output(0)


def build_decoder(network, image, levels, weights, dtype=np.float32):
    """Levels are x1..x4 after the dual-scale concatenation. Returns the logit map."""
    x1, x2, x3, x4 = levels
    image_hw = tuple(int(v) for v in image.shape)[2:]

    def hw(tensor):
        return tuple(int(v) for v in tensor.shape)[2:]

    # cxt: the finer levels joined onto x4 at its own resolution.
    context = [g.add_resize_bilinear(network, level, hw(x4), align_corners=True)
               for level in (x1, x2, x3)]
    x4 = g.concat(network, context + [x4], axis=1)
    x4 = _basic_dec_blk(network, x4, weights, "squeeze_module.0", dtype)

    x4 = g.concat(network, [x4, _input_branch(network, image, weights, "decoder.ipt_blk5",
                                              x4, hw(x4), dtype)], axis=1)
    p4 = _basic_dec_blk(network, x4, weights, "decoder.decoder_block4", dtype)
    p4 = _gdt_gate(network, p4, weights, 4, dtype)
    lateral = g.add_conv2d(network, x3, weights["decoder.lateral_block4.conv.weight"],
                           weights.get("decoder.lateral_block4.conv.bias"), dtype=dtype)
    p3_in = g.add_sum(network, g.add_resize_bilinear(network, p4, hw(x3), align_corners=True),
                      lateral)

    p3_in = g.concat(network, [p3_in, _input_branch(network, image, weights,
                                                    "decoder.ipt_blk4", p3_in, hw(x3), dtype)],
                     axis=1)
    p3 = _basic_dec_blk(network, p3_in, weights, "decoder.decoder_block3", dtype)
    p3 = _gdt_gate(network, p3, weights, 3, dtype)
    lateral = g.add_conv2d(network, x2, weights["decoder.lateral_block3.conv.weight"],
                           weights.get("decoder.lateral_block3.conv.bias"), dtype=dtype)
    p2_in = g.add_sum(network, g.add_resize_bilinear(network, p3, hw(x2), align_corners=True),
                      lateral)

    p2_in = g.concat(network, [p2_in, _input_branch(network, image, weights,
                                                    "decoder.ipt_blk3", p2_in, hw(x2), dtype)],
                     axis=1)
    p2 = _basic_dec_blk(network, p2_in, weights, "decoder.decoder_block2", dtype)
    p2 = _gdt_gate(network, p2, weights, 2, dtype)
    lateral = g.add_conv2d(network, x1, weights["decoder.lateral_block2.conv.weight"],
                           weights.get("decoder.lateral_block2.conv.bias"), dtype=dtype)
    p1_in = g.add_sum(network, g.add_resize_bilinear(network, p2, hw(x1), align_corners=True),
                      lateral)

    p1_in = g.concat(network, [p1_in, _input_branch(network, image, weights,
                                                    "decoder.ipt_blk2", p1_in, hw(x1), dtype)],
                     axis=1)
    p1 = _basic_dec_blk(network, p1_in, weights, "decoder.decoder_block1", dtype)
    p1 = g.add_resize_bilinear(network, p1, image_hw, align_corners=True)
    # At full resolution the patch grid is one cell, so this branch sees the
    # image itself rather than a rearrangement of it.
    p1 = g.concat(network, [p1, _input_branch(network, image, weights, "decoder.ipt_blk1",
                                              p1, image_hw, dtype)], axis=1)
    return g.add_conv2d(network, p1, weights["decoder.conv_out1.0.weight"],
                        weights["decoder.conv_out1.0.bias"], dtype=dtype)
