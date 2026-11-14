# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Encoder-free Gemma4 unified media projections in native TensorRT math."""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from .checkpoint_mapper import _load_tensor


def media_weight_name(name, keys):
    """Resolve original serialized names and the provider's current module names."""
    if name in keys:
        return name
    if name.startswith("model.embed_vision.multimodal_embedder."):
        original = name.replace("model.embed_vision.multimodal_embedder.", "model.embed_vision.", 1)
    elif name.startswith("model.embed_vision."):
        original = name.replace("model.embed_vision.", "model.vision_embedder.", 1)
    else:
        original = name
    if original not in keys:
        raise KeyError(f"Gemma4 unified media weight is missing: {name}")
    return original


def validate_media_weights(readers, *, vision, audio):
    """Reject incomplete media checkpoints before compiling the large decoder."""
    keys = {name for reader in readers for name in reader.keys()}
    required = ["model.embed_audio.embedding_projection.weight"] if audio else []
    if vision:
        required += ["model.embed_vision." + layer + suffix
                     for layer in ("patch_ln1", "patch_dense", "patch_ln2", "pos_norm")
                     for suffix in (".weight", ".bias")]
        required += ["model.embed_vision.pos_embedding",
                     "model.embed_vision.multimodal_embedder.embedding_projection.weight"]
    for name in required:
        media_weight_name(name, keys)


class UnifiedMediaGraph:
    """Raw merged RGB patches or waveform frames, not the smaller model's towers."""

    def __init__(self, network, raw, readers, precision):
        self.net, self.raw, self.readers = network, raw, readers
        self.keys = {name for reader in readers for name in reader.keys()}
        self.dtype = {"fp16": trt.float16, "bf16": trt.bfloat16, "fp32": trt.float32}[precision]
        self.storage = np.float16 if precision == "fp16" else np.float32
        self.buffers = []

    def cast(self, value, dtype):
        return value if value.dtype == dtype else self.net.add_cast(value, dtype).get_output(0)

    def const(self, value, dtype=None):
        value = np.ascontiguousarray(value)
        self.buffers.append(value)
        tensor = self.net.add_constant(value.shape, trt.Weights(value)).get_output(0)
        return self.cast(tensor, dtype) if dtype is not None else tensor

    def weight(self, name, dtype=None):
        values = _load_tensor(self.readers, media_weight_name(name, self.keys))
        return self.const(values.astype(self.storage if dtype is None else np.float32),
                          self.dtype if dtype is None else dtype)

    def reshape(self, value, shape):
        layer = self.net.add_shuffle(value)
        layer.reshape_dims = shape
        return layer.get_output(0)

    def op(self, lhs, rhs, operation):
        return self.net.add_elementwise(lhs, rhs, operation).get_output(0)

    def scalar(self, value):
        return self.const(np.array([[value]], np.float32))

    def linear(self, value, name, *, bias=False):
        weight = self.cast(self.weight(name + ".weight"), value.dtype)
        output = self.net.add_matrix_multiply(value, trt.MatrixOperation.NONE,
                                             weight, trt.MatrixOperation.TRANSPOSE).get_output(0)
        if bias:
            output = self.op(output, self.reshape(self.cast(self.weight(name + ".bias"), value.dtype), (1, -1)),
                             trt.ElementWiseOperation.SUM)
        return output

    def norm(self, value, epsilon, name=None):
        dtype = value.dtype
        value = self.cast(value, trt.float32)
        if name is not None:
            mean = self.net.add_reduce(value, trt.ReduceOperation.AVG, 2, True).get_output(0)
            value = self.op(value, mean, trt.ElementWiseOperation.SUB)
        square = self.op(value, value, trt.ElementWiseOperation.PROD)
        variance = self.net.add_reduce(square, trt.ReduceOperation.AVG, 2, True).get_output(0)
        variance = self.op(variance, self.scalar(epsilon), trt.ElementWiseOperation.SUM)
        inverse = self.op(variance, self.scalar(-0.5), trt.ElementWiseOperation.POW)
        output = self.op(value, inverse, trt.ElementWiseOperation.PROD)
        if name is not None:
            scale = self.reshape(self.weight(name + ".weight", trt.float32), (1, -1))
            bias = self.reshape(self.weight(name + ".bias", trt.float32), (1, -1))
            output = self.op(self.op(output, scale, trt.ElementWiseOperation.PROD),
                             bias, trt.ElementWiseOperation.SUM)
        return self.cast(output, dtype)

    def patch_embeddings(self, pixels, prefix):
        value = self.norm(self.cast(pixels, self.dtype), 1e-5, prefix + ".patch_ln1")
        # Real unified checkpoints exceed FP16 range before the following norm.
        # Keep the projection and its normalization wide; never clip activations.
        value = self.linear(self.cast(value, trt.float32), prefix + ".patch_dense", bias=True)
        return self.cast(self.norm(value, 1e-5, prefix + ".patch_ln2"), self.dtype)

    def image(self, soft_tokens):
        raw, n = self.raw["vision_config"], self.net
        patch = int(raw["patch_size"]) * int(raw["pooling_kernel_size"])
        pixels = n.add_input("pixel_values", trt.float32, (soft_tokens, 3 * patch * patch))
        positions = n.add_input("pixel_position_ids", trt.int32, (soft_tokens, 2))
        prefix = "model.embed_vision"
        value = self.patch_embeddings(pixels, prefix)
        table = _load_tensor(self.readers, media_weight_name(
            prefix + ".pos_embedding", self.keys)).astype(self.storage)
        axes, masks = [], []
        for axis in (0, 1):
            position = n.add_slice(positions, (0, axis), (soft_tokens, 1), (1, 1)).get_output(0)
            valid = self.op(position, self.const(np.array([[-1]], np.int32)),
                            trt.ElementWiseOperation.GREATER)
            index = self.op(position, self.const(np.array([[0]], np.int32)),
                            trt.ElementWiseOperation.MAX)
            embedding = n.add_gather(self.const(table[:, axis, :], self.dtype),
                                     self.reshape(index, (soft_tokens,)), 0).get_output(0)
            axes.append(n.add_select(valid, embedding,
                        self.const(np.zeros((1, 1), np.float32), self.dtype)).get_output(0))
            masks.append(valid)
        position_embedding = self.op(axes[0], axes[1], trt.ElementWiseOperation.SUM)
        value = self.op(value, position_embedding, trt.ElementWiseOperation.SUM)
        value = self.norm(value, 1e-5, prefix + ".pos_norm")
        value = self.norm(value, float(raw["rms_norm_eps"]))
        value = self.linear(value, prefix + ".multimodal_embedder.embedding_projection")
        value = self.cast(value, trt.float32)
        value.name = "image_features"
        n.mark_output(value)
        mask = self.op(masks[0], masks[1], trt.ElementWiseOperation.OR)
        mask = self.reshape(self.cast(mask, trt.int32), (soft_tokens,))
        mask.name = "image_feature_mask"
        n.mark_output(mask)

    def audio(self, capacity):
        raw, n = self.raw["audio_config"], self.net
        width = int(raw["audio_embed_dim"])
        features = n.add_input("input_features", trt.float32, (capacity, width))
        mask = n.add_input("input_features_mask", trt.int32, (capacity,))
        value = self.norm(self.cast(features, self.dtype), float(raw["rms_norm_eps"]))
        value = self.linear(value, "model.embed_audio.embedding_projection")
        value = self.cast(value, trt.float32)
        value.name = "audio_features"
        n.mark_output(value)
        validity = n.add_identity(mask).get_output(0)
        validity.name = "audio_feature_mask"
        n.mark_output(validity)
