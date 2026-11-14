# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gemma 3n MobileNetV5-300M image encoder and soft-token embedder.

The block schedule is the published ``mobilenetv5_300m_enc`` encoder (timm),
which the checkpoint stores under ``model.vision_tower.timm_model``. Every
convolution uses TensorFlow ``SAME`` padding, every norm is a channel RMS norm
(eps 1e-6) and the activation is tanh GELU. The multi-scale fusion adapter
combines the last two stages at 48x48, pools to 16x16 and yields 256 soft
tokens that the multimodal embedder projects into the text space.
"""

from __future__ import annotations

import math

import tensorrt as trt

from .graph import Builder

IMAGE_SIZE = 768
NORM_EPS = 1e-6
_PREFIX = "model.vision_tower.timm_model"


def _uir(kernel_start, kernel_mid, stride, expansion, channels):
    return ("uir", kernel_start, kernel_mid, stride, expansion, channels)


def _mqa(heads, kv_stride, key_dim, channels):
    return ("mqa", heads, kv_stride, key_dim, channels)


# (block, ...) per stage, transcribed from the encoder's architecture strings.
STAGES = (
    (("er", 3, 2, 4, 128), ("er", 3, 1, 4, 128), ("er", 3, 1, 4, 128)),
    (_uir(3, 5, 2, 6, 256), _uir(5, 0, 1, 4, 256), _uir(3, 0, 1, 4, 256),
     _uir(5, 0, 1, 4, 256), _uir(3, 0, 1, 4, 256)),
    (_uir(5, 5, 2, 6, 640), *(_uir(5, 0, 1, 4, 640) for _ in range(7)), _uir(0, 0, 1, 1, 640),
     *(block for _ in range(14) for block in (_mqa(12, 2, 64, 640), _uir(0, 0, 1, 2, 640)))),
    (_uir(5, 5, 2, 6, 1280),
     *(block for _ in range(19) for block in (_mqa(16, 1, 96, 1280), _uir(0, 0, 1, 2, 1280)))),
)
STEM_CHANNELS = 64
OUTPUT_CHANNELS = 2048
OUTPUT_SIDE = 16


def _make_divisible(value: float, divisor: int = 8) -> int:
    # timm make_divisible(v, 8) with its 0.9 round-down guard.
    result = max(divisor, int(value + divisor / 2) // divisor * divisor)
    return result + divisor if result < 0.9 * value else result


class VisionGraph:
    """Pixels ``[1, 3, 768, 768]`` (float32, already rescaled) to soft tokens."""

    def __init__(self, builder: Builder, vision_config, text_hidden: int):
        self.b = builder
        self.vision = vision_config
        self.text_hidden = text_hidden

    def name(self, suffix: str) -> str:
        return f"{_PREFIX}.{suffix}"

    def conv(self, value, prefix: str, out_channels: int, kernel: int, stride: int = 1,
             groups: int = 1, bias: bool = False):
        b = self.b
        in_channels = value.shape[1]
        weight = b.checkpoint.get(f"{prefix}.weight")
        expected = (out_channels, in_channels // groups, kernel, kernel)
        if tuple(weight.shape) != expected:
            raise ValueError(f"Gemma 3n {prefix} weight {weight.shape} differs from {expected}")
        bias_weights = (b.weights(b.checkpoint.get_f32(f"{prefix}.bias", (out_channels,)),
                                  value.dtype) if bias else trt.Weights(value.dtype))
        layer = b.net.add_convolution_nd(value, out_channels, (kernel, kernel),
                                         b.weights(weight, value.dtype), bias_weights)
        layer.stride_nd = (stride, stride)
        layer.num_groups = groups
        # TensorFlow SAME: the odd padding pixel of a strided window goes last,
        # which is timm's dynamic Conv2dSame and its static symmetric padding.
        layer.padding_mode = trt.PaddingMode.SAME_UPPER
        return layer.get_output(0)

    def norm(self, value, prefix: str, act: bool):
        output = self.b.rms_norm(value, NORM_EPS, f"{prefix}.weight", axis=1)
        return self.b.gelu_tanh(output) if act else output

    def conv_norm(self, value, prefix: str, out_channels: int, kernel: int, stride: int = 1,
                  groups: int = 1, act: bool = True):
        value = self.conv(value, f"{prefix}.conv", out_channels, kernel, stride, groups)
        return self.norm(value, f"{prefix}.bn", act)

    def layer_scale(self, value, prefix: str):
        channels = value.shape[1]
        gamma = self.b.reshape(self.b.tensor(f"{prefix}.layer_scale.gamma", value.dtype,
                                             (channels,)), (1, channels, 1, 1))
        return self.b.mul(value, gamma)

    def edge_residual(self, value, prefix: str, kernel, stride, expansion, channels):
        in_channels = value.shape[1]
        hidden = _make_divisible(in_channels * expansion)
        x = self.conv(value, f"{prefix}.conv_exp", hidden, kernel, stride)
        x = self.norm(x, f"{prefix}.bn1", act=True)
        x = self.conv(x, f"{prefix}.conv_pwl", channels, 1)
        x = self.norm(x, f"{prefix}.bn2", act=False)
        return self.b.add(x, value) if stride == 1 and in_channels == channels else x

    def universal_inverted(self, value, prefix: str, kernel_start, kernel_mid, stride,
                           expansion, channels, layer_scale=True, skip=True):
        in_channels = value.shape[1]
        x = value
        if kernel_start:
            x = self.conv_norm(x, f"{prefix}.dw_start", in_channels, kernel_start,
                               1 if kernel_mid else stride, groups=in_channels, act=False)
        hidden = _make_divisible(in_channels * expansion)
        x = self.conv_norm(x, f"{prefix}.pw_exp", hidden, 1)
        if kernel_mid:
            x = self.conv_norm(x, f"{prefix}.dw_mid", hidden, kernel_mid, stride, groups=hidden)
        x = self.conv_norm(x, f"{prefix}.pw_proj", channels, 1, act=False)
        if layer_scale:
            x = self.layer_scale(x, prefix)
        return self.b.add(x, value) if skip and stride == 1 and in_channels == channels else x

    def mobile_attention(self, value, prefix: str, heads, kv_stride, key_dim, channels):
        b = self.b
        _, in_channels, height, width = value.shape
        x = self.norm(value, f"{prefix}.norm", act=False)
        attn = f"{prefix}.attn"
        query = self.conv(x, f"{attn}.query.proj", heads * key_dim, 1)
        # [1, heads*key, H, W] -> [1, heads, H*W, key]
        query = b.transpose(b.reshape(query, (1, heads, key_dim, height * width)), (0, 1, 3, 2))

        def key_or_value(kind):
            source = x
            if kv_stride > 1:
                source = self.conv(source, f"{attn}.{kind}.down_conv", in_channels, 3,
                                   kv_stride, groups=in_channels)
                source = self.norm(source, f"{attn}.{kind}.norm", act=False)
            source = self.conv(source, f"{attn}.{kind}.proj", key_dim, 1)
            tokens = source.shape[2] * source.shape[3]
            # One key/value head shared by every query head: [1, 1, tokens, key].
            return b.transpose(b.reshape(source, (1, 1, key_dim, tokens)), (0, 1, 3, 2))

        key, val = key_or_value("key"), key_or_value("value")
        scores = b.matmul(b.cast(query, trt.float32), b.cast(key, trt.float32), transpose_rhs=True)
        scores = b.scale(scores, key_dim ** -0.5)
        softmax = b.net.add_softmax(scores)
        softmax.axes = 1 << 3
        probabilities = b.cast(softmax.get_output(0), value.dtype)
        context = b.matmul(probabilities, val)  # [1, heads, H*W, key]
        # -> [1, H*W, heads, key] -> [1, H, W, heads*key] -> [1, heads*key, H, W]
        context = b.transpose(context, (0, 2, 1, 3))
        context = b.transpose(b.reshape(context, (1, height, width, heads * key_dim)),
                              (0, 3, 1, 2))
        output = self.conv(context, f"{attn}.output.proj", channels, 1)
        output = self.layer_scale(output, prefix)
        return b.add(output, value) if in_channels == channels else output

    def encoder(self, pixels):
        b = self.b
        x = self.conv(pixels, self.name("conv_stem.conv"), STEM_CHANNELS, 3, 2, bias=True)
        x = self.norm(x, self.name("conv_stem.bn"), act=True)
        stage_outputs = []
        for stage, blocks in enumerate(STAGES):
            for index, block in enumerate(blocks):
                prefix = self.name(f"blocks.{stage}.{index}")
                kind, *args = block
                if kind == "er":
                    x = self.edge_residual(x, prefix, *args)
                elif kind == "uir":
                    x = self.universal_inverted(x, prefix, *args)
                else:
                    x = self.mobile_attention(x, prefix, *args)
            stage_outputs.append(x)
        # Multi-scale fusion of the last two stages at the finer resolution.
        fine, coarse = stage_outputs[-2], stage_outputs[-1]
        # Resize and concatenation only move values, so running them in FP32 is
        # exact for every build precision; TensorRT 11.4 cannot build the BF16
        # form of this concatenation.
        resize = b.net.add_resize(b.cast(coarse, trt.float32))
        resize.shape = (1, coarse.shape[1], fine.shape[2], fine.shape[3])
        resize.resize_mode = trt.InterpolationMode.NEAREST
        # torch "nearest": source index floor(dst * in / out) (asymmetric, floor).
        resize.coordinate_transformation = trt.ResizeCoordinateTransformation.ASYMMETRIC
        resize.nearest_rounding = trt.ResizeRoundMode.FLOOR
        fused = b.cast(b.concat([b.cast(fine, trt.float32), resize.get_output(0)], axis=1), b.dtype)
        fused = self.universal_inverted(fused, self.name("msfa.ffn"), 0, 0, 1,
                                        2.0, OUTPUT_CHANNELS, layer_scale=False, skip=False)
        side = fused.shape[2]
        if side % OUTPUT_SIDE or fused.shape[3] != side:
            raise ValueError("Gemma 3n fusion resolution is not a multiple of the soft-token grid")
        window = side // OUTPUT_SIDE
        pool = b.net.add_pooling_nd(fused, trt.PoolingType.AVERAGE, (window, window))
        pool.stride_nd = (window, window)
        return self.norm(pool.get_output(0), self.name("msfa.norm"), act=False)

    def embed(self, features):
        """Soft tokens to text embeddings (``embed_vision`` soft-token path)."""
        b = self.b
        hidden = self.vision.hidden_size
        # [1, C, 16, 16] -> [1, 256, C], scaled by sqrt(C) as the reference does.
        tokens = b.transpose(b.reshape(features, (1, hidden, OUTPUT_SIDE * OUTPUT_SIDE)), (0, 2, 1))
        tokens = b.reshape(b.scale_fp32(tokens, math.sqrt(hidden)),
                           (OUTPUT_SIDE * OUTPUT_SIDE, hidden))
        eps = self.vision.rms_norm_eps
        tokens = b.rms_norm(tokens, eps, "model.embed_vision.soft_embedding_norm.weight")
        tokens = b.linear(tokens, "model.embed_vision.embedding_projection.weight",
                          (self.text_hidden, hidden))
        return b.rms_norm(tokens, eps)

    def build(self):
        b = self.b
        stored = {name for name in b.checkpoint.mapping if name.startswith(f"{_PREFIX}.")}
        expected = expected_tensor_names()
        if stored != expected:
            unused, missing = sorted(stored - expected), sorted(expected - stored)
            raise ValueError("Gemma 3n vision tower tensors differ from mobilenetv5_300m_enc: "
                             f"unused {unused[:4]}, missing {missing[:4]}")
        pixels = b.net.add_input("pixel_values", trt.float32, (1, 3, IMAGE_SIZE, IMAGE_SIZE))
        features = self.embed(self.encoder(b.cast(pixels, b.dtype)))
        features.name = "image_features"
        b.net.mark_output(features)


def soft_token_count() -> int:
    return OUTPUT_SIDE * OUTPUT_SIDE


def expected_tensor_names() -> set[str]:
    """Checkpoint tensors the encoder reads (validated against the index)."""
    names = {f"{_PREFIX}.conv_stem.conv.weight", f"{_PREFIX}.conv_stem.conv.bias",
             f"{_PREFIX}.conv_stem.bn.weight", f"{_PREFIX}.msfa.norm.weight"}
    for part in ("pw_exp", "pw_proj"):
        names |= {f"{_PREFIX}.msfa.ffn.{part}.conv.weight", f"{_PREFIX}.msfa.ffn.{part}.bn.weight"}
    for stage, blocks in enumerate(STAGES):
        for index, block in enumerate(blocks):
            prefix = f"{_PREFIX}.blocks.{stage}.{index}"
            kind = block[0]
            if kind == "er":
                names |= {f"{prefix}.{part}" for part in (
                    "conv_exp.weight", "bn1.weight", "conv_pwl.weight", "bn2.weight")}
            elif kind == "uir":
                _, start, mid, *_ = block
                parts = ["pw_exp", "pw_proj"] + (["dw_start"] if start else []) + (
                    ["dw_mid"] if mid else [])
                names |= {f"{prefix}.{part}.{leaf}" for part in parts for leaf in (
                    "conv.weight", "bn.weight")}
                names.add(f"{prefix}.layer_scale.gamma")
            else:
                _, _, kv_stride, _, _ = block
                names |= {f"{prefix}.norm.weight", f"{prefix}.layer_scale.gamma",
                          f"{prefix}.attn.query.proj.weight", f"{prefix}.attn.output.proj.weight"}
                for kind_name in ("key", "value"):
                    names.add(f"{prefix}.attn.{kind_name}.proj.weight")
                    if kv_stride > 1:
                        names |= {f"{prefix}.attn.{kind_name}.down_conv.weight",
                                  f"{prefix}.attn.{kind_name}.norm.weight"}
    return names


__all__ = ["IMAGE_SIZE", "VisionGraph", "expected_tensor_names", "soft_token_count"]
