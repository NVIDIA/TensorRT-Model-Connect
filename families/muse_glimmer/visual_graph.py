# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Original Muse-Glimmer visual encoder, one temporal patch group at a time."""
from __future__ import annotations

import os

import numpy as np
import tensorrt as trt

from .native_graph import Graph
from .quantization import weight_tensor


class VisionGraph(Graph):
    """Keep patches in raster order. Spatial window membership masks local attention."""

    def affine_norm(self, x, name, eps):
        width = int(x.shape[-1])
        value = self.cast(x, trt.float32)
        mean = self.network.add_reduce(value, trt.ReduceOperation.AVG, 2, True).get_output(0)
        centered = self.ew(value, mean, trt.ElementWiseOperation.SUB)
        normalized = self.norm(centered, eps)
        scale = self.const(self.checkpoint.vector(name + ".weight", width)[None, :])
        bias = self.const(self.checkpoint.vector(name + ".bias", width)[None, :])
        return self.cast(self.ew(self.ew(normalized, scale, trt.ElementWiseOperation.PROD), bias,
                                 trt.ElementWiseOperation.SUM), x.dtype)

    def fc(self, x, name, width, *, bias=True):
        x = self.project(x, name, width)
        if bias:
            b = self.const(self.checkpoint.vector(name + ".bias", width)[None, :])
            x = self.ew(x, self.cast(b, x.dtype), trt.ElementWiseOperation.SUM)
        return x

    def gelu(self, x):
        return self.network.add_activation(x, trt.ActivationType.GELU_ERF).get_output(0)

    def attend(self, x, prefix, heads, cos, sin, mask, scale):
        width = int(x.shape[1])
        dim = width // heads
        q, k, v = [self.reshape(self.fc(x, prefix + "." + p + "_proj", width), (-1, heads, dim))
                   for p in ("q", "k", "v")]
        swapped = self.const(np.r_[dim // 2:dim, :dim // 2], np.int32)
        sign = self.const(np.repeat([-1, 1], dim // 2).reshape(1, 1, dim))
        def rotate(values):
            raw = self.cast(values, trt.float32)
            half = self.network.add_gather(raw, swapped, 2).get_output(0)
            y = self.ew(self.ew(raw, cos, trt.ElementWiseOperation.PROD),
                        self.ew(self.ew(half, sign, trt.ElementWiseOperation.PROD), sin,
                                trt.ElementWiseOperation.PROD), trt.ElementWiseOperation.SUM)
            return self.cast(y, trt.float16)
        def head_first(values):
            layer = self.network.add_shuffle(values)
            layer.first_transpose = (1, 0, 2)
            return layer.get_output(0)
        # FP32 logits/softmax, FP16 projection boundary, no plugin or platform-specific kernel.
        q = self.cast(head_first(rotate(q)), trt.float32)
        k = self.cast(head_first(rotate(k)), trt.float32)
        scores = self.network.add_matrix_multiply(q, trt.MatrixOperation.NONE, k,
                                                  trt.MatrixOperation.TRANSPOSE).get_output(0)
        scores = self.scale(scores, scale)
        if mask is not None:
            scores = self.ew(scores, mask, trt.ElementWiseOperation.SUM)
        probs = self.network.add_softmax(scores)
        probs.axes = 4
        context = self.network.add_matrix_multiply(
            probs.get_output(0), trt.MatrixOperation.NONE, self.cast(head_first(v), trt.float32),
            trt.MatrixOperation.NONE).get_output(0)
        context = self.reshape(self.cast(context, trt.float16), (-1, width), first_transpose=(1, 0, 2))
        return self.fc(context, prefix + ".proj", width)

    def populate_visual(self):
        v = self.config["vision_config"]
        c = self.config["text_config"]
        width, heads, merge = v["hidden_size"], v["num_attention_heads"], v["merge_size"]
        if (width % heads or (width // heads) % 4 or merge != 2 or v["hidden_act"] != "gelu"
                or self.config["projector_hidden_act"] != "gelu"
                or len(v["layer_types"]) != v["num_hidden_layers"]):
            raise ValueError("Invalid Muse visual geometry or activation")
        dim, lanes = width // heads, merge**2
        features = v["patch_size"]**2 * v["patch_temporal"] * v.get("num_channels", 3)
        inputs = {
            "patches": (trt.float16, (-1, features)), "position_indices": (trt.int32, (-1, 4)),
            "position_weights": (trt.float16, (-1, 4)), "rotary_cos": (trt.float32, (-1, 1, dim)),
            "rotary_sin": (trt.float32, (-1, 1, dim)), "window_mask": (trt.float32, (1, -1, -1)),
            "merge_order": (trt.int32, (-1,)),
        }
        rows = {k: self.network.add_input(k, *v) for k, v in inputs.items()}
        tower = "vision_tower"
        patch = tower + ".patch_embedder"
        x = self.fc(rows["patches"], patch + ".patch_embedding", width, bias=False)
        table = self.checkpoint.projection(patch + ".position_embedding_table",
                                           v["pos_emb_height"] * v["pos_emb_width"], width)
        table = weight_tensor(self.network, table, self.keep_alive, trt.float16)
        pos = self.network.add_gather(table, rows["position_indices"], 0).get_output(0)
        weighted = self.ew(pos, self.reshape(rows["position_weights"], (-1, 4, 1)),
                           trt.ElementWiseOperation.PROD)
        pos = self.network.add_reduce(weighted, trt.ReduceOperation.SUM, 2, False).get_output(0)
        x = self.affine_norm(self.ew(x, pos, trt.ElementWiseOperation.SUM), tower + ".ln_pre", v["layer_norm_eps"])
        for i, kind in enumerate(v["layer_types"]):
            if kind not in {"window_attention", "full_attention"}:
                raise ValueError(f"Unknown Muse visual attention type: {kind}")
            prefix = f"{tower}.layers.{i}"
            y = self.affine_norm(x, prefix + ".norm1", v["layer_norm_eps"])
            y = self.attend(y, prefix + ".attn", heads, rows["rotary_cos"], rows["rotary_sin"],
                            rows["window_mask"] if kind == "window_attention" else None, dim**-0.5)
            x = self.ew(x, y, trt.ElementWiseOperation.SUM)
            y = self.affine_norm(x, prefix + ".norm2", v["layer_norm_eps"])
            y = self.gelu(self.fc(y, prefix + ".mlp.fc1", v["intermediate_size"]))
            x = self.ew(x, self.fc(y, prefix + ".mlp.fc2", width), trt.ElementWiseOperation.SUM)
        x = self.affine_norm(x, tower + ".ln_post", v["layer_norm_eps"])
        x = self.network.add_gather(x, rows["merge_order"], 0).get_output(0)
        x = self.reshape(x, (-1, lanes, width), transpose=(0, 2, 1))
        x = self.reshape(x, (-1, width * lanes))
        for name in ("vision_adapter.fc1", "vision_adapter.fc2"):
            x = self.gelu(self.fc(x, name, self.config["projector_hidden_size"], bias=False))
        x = self.fc(x, "vision_projection", c["hidden_size"], bias=False)
        x = self.norm(x, c["rms_norm_eps"])
        x.name = "visual_embeddings"
        self.network.mark_output(x)
        return inputs


def build_visual(raw, checkpoint, max_patches, *, verbose=False):
    """Build one dynamic per-group encoder; a group's patch grid has even sides."""
    if type(max_patches) is not int or max_patches < 4 or max_patches % 4:
        raise ValueError("Muse visual profiles require a positive number of complete 2x2 blocks")
    builder = trt.Builder(trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING))
    builder.max_threads = min(8, os.cpu_count() or 1)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    graph = VisionGraph(network, checkpoint, raw, max_patches)
    inputs = graph.populate_visual()
    profile = builder.create_optimization_profile()
    for name, (_, axes) in inputs.items():
        profile.set_shape(name, *(tuple(n if a < 0 else a for a in axes)
                                  for n in (4, min(256, max_patches), max_patches)))
    settings = builder.create_builder_config()
    settings.builder_optimization_level = 1
    settings.add_optimization_profile(profile)
    result = builder.build_serialized_network(network, settings)
    if result is None:
        raise RuntimeError("TensorRT failed to build the Muse visual encoder")
    return bytes(result)
