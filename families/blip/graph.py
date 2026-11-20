# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BLIP's ViT and cross-attending BERT decoder, expressed as TensorRT math."""

import math

import numpy as np
import tensorrt as trt


class Graph:
    def __init__(self, network, weights):
        self.n = network
        self.w = weights
        self.constants = []

    def constant(self, values, shape=None):
        array = np.ascontiguousarray(values, dtype=np.float32)
        if shape is not None:
            array = array.reshape(shape)
        self.constants.append(array)
        return self.n.add_constant(array.shape, array).get_output(0)

    def op(self, a, b, kind=trt.ElementWiseOperation.SUM):
        return self.n.add_elementwise(a, b, kind).get_output(0)

    def scale(self, x, factor):
        return self.op(
            x, self.constant([factor], (1,) * len(x.shape)), trt.ElementWiseOperation.PROD
        )

    def reshape(self, x, shape, transpose=None):
        layer = self.n.add_shuffle(x)
        layer.reshape_dims = shape
        if transpose is not None:
            layer.second_transpose = transpose
        return layer.get_output(0)

    def linear(self, x, prefix):
        weight = self.w[prefix + ".weight"]
        shaped = (1,) * (len(x.shape) - 2) + weight.shape
        out = self.n.add_matrix_multiply(
            x,
            trt.MatrixOperation.NONE,
            self.constant(weight, shaped),
            trt.MatrixOperation.TRANSPOSE,
        ).get_output(0)
        bias = self.w.get(prefix + ".bias")
        if bias is not None:
            out = self.op(out, self.constant(bias, (1,) * (len(x.shape) - 1) + bias.shape))
        return out

    def norm(self, x, prefix, eps):
        axis = 1 << (len(x.shape) - 1)
        shape = (1,) * (len(x.shape) - 1) + (x.shape[-1],)
        layer = self.n.add_normalization(
            x,
            self.constant(self.w[prefix + ".weight"], shape),
            self.constant(self.w[prefix + ".bias"], shape),
            axis,
        )
        layer.epsilon = eps
        return layer.get_output(0)

    def gelu(self, x):
        # The checkpoint's gelu is the exact erf form, including in the LM head.
        erf = self.n.add_unary(self.scale(x, 1 / math.sqrt(2)), trt.UnaryOperation.ERF).get_output(
            0
        )
        gate = self.op(erf, self.constant([1], (1,) * len(x.shape)))
        return self.op(self.scale(x, 0.5), gate, trt.ElementWiseOperation.PROD)

    def attention(self, q, k, v, heads, mask=None):
        width = q.shape[-1]
        depth = width // heads
        q, k, v = [self.reshape(x, (-1, heads, depth), (1, 0, 2)) for x in (q, k, v)]
        scores = self.n.add_matrix_multiply(
            q, trt.MatrixOperation.NONE, k, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)
        scores = self.scale(scores, 1 / math.sqrt(depth))
        if mask is not None:
            scores = self.op(scores, mask)
        softmax = self.n.add_softmax(scores)
        softmax.axes = 1 << 2
        context = self.n.add_matrix_multiply(
            softmax.get_output(0), trt.MatrixOperation.NONE, v, trt.MatrixOperation.NONE
        ).get_output(0)
        transpose = self.n.add_shuffle(context)
        transpose.first_transpose = (1, 0, 2)
        transpose.reshape_dims = (-1, width)
        return transpose.get_output(0)

    def vision(self, cfg):
        width, patch, size = cfg["hidden_size"], cfg["patch_size"], cfg["image_size"]
        rows = (size // patch) ** 2 + 1
        pixels = self.n.add_input("pixel_values", trt.float32, (1, 3, size, size))
        conv = self.n.add_convolution_nd(
            pixels,
            width,
            (patch, patch),
            self.w["vision_model.embeddings.patch_embedding.weight"],
            self.w["vision_model.embeddings.patch_embedding.bias"],
        )
        conv.stride_nd = (patch, patch)
        patches = self.reshape(conv.get_output(0), (width, rows - 1), (1, 0))
        cls = self.constant(self.w["vision_model.embeddings.class_embedding"], (1, width))
        cat = self.n.add_concatenation([cls, patches])
        cat.axis = 0
        hidden = self.op(
            cat.get_output(0),
            self.constant(self.w["vision_model.embeddings.position_embedding"], (rows, width)),
        )
        for i in range(cfg["num_hidden_layers"]):
            p = f"vision_model.encoder.layers.{i}"
            normed = self.norm(hidden, p + ".layer_norm1", cfg["layer_norm_eps"])
            qkv = self.linear(normed, p + ".self_attn.qkv")
            q, k, v = [
                self.n.add_slice(qkv, (0, j * width), (rows, width), (1, 1)).get_output(0)
                for j in range(3)
            ]
            attended = self.attention(q, k, v, cfg["num_attention_heads"])
            hidden = self.op(hidden, self.linear(attended, p + ".self_attn.projection"))
            ffn = self.norm(hidden, p + ".layer_norm2", cfg["layer_norm_eps"])
            hidden = self.op(
                hidden, self.linear(self.gelu(self.linear(ffn, p + ".mlp.fc1")), p + ".mlp.fc2")
            )
        return self.norm(hidden, "vision_model.post_layernorm", cfg["layer_norm_eps"])

    def decoder(self, cfg, vision_cfg):
        width = cfg["hidden_size"]
        rows = (vision_cfg["image_size"] // vision_cfg["patch_size"]) ** 2 + 1
        image = self.n.add_input("image_embeds", trt.float32, (rows, vision_cfg["hidden_size"]))
        ids = self.n.add_input("input_ids", trt.int32, (-1,))
        positions = self.n.add_input("position_ids", trt.int32, (-1,))
        mask = self.n.add_input("causal_mask", trt.float32, (1, -1, -1))
        last = self.n.add_input("last_index", trt.int32, (1,))
        p = "text_decoder.bert.embeddings"
        words = self.n.add_gather(
            self.constant(self.w[p + ".word_embeddings.weight"]), ids, 0
        ).get_output(0)
        pos = self.n.add_gather(
            self.constant(self.w[p + ".position_embeddings.weight"]), positions, 0
        ).get_output(0)
        hidden = self.norm(self.op(words, pos), p + ".LayerNorm", cfg["layer_norm_eps"])
        for i in range(cfg["num_hidden_layers"]):
            p = f"text_decoder.bert.encoder.layer.{i}"
            for name, source, attention_mask in (
                ("attention", hidden, mask),
                ("crossattention", image, None),
            ):
                a = p + "." + name
                q = self.linear(hidden, a + ".self.query")
                k = self.linear(source, a + ".self.key")
                v = self.linear(source, a + ".self.value")
                context = self.attention(q, k, v, cfg["num_attention_heads"], attention_mask)
                hidden = self.norm(
                    self.op(hidden, self.linear(context, a + ".output.dense")),
                    a + ".output.LayerNorm",
                    cfg["layer_norm_eps"],
                )
            ffn = self.gelu(self.linear(hidden, p + ".intermediate.dense"))
            hidden = self.norm(
                self.op(hidden, self.linear(ffn, p + ".output.dense")),
                p + ".output.LayerNorm",
                cfg["layer_norm_eps"],
            )
        hidden = self.n.add_gather(hidden, last, 0).get_output(0)
        p = "text_decoder.cls.predictions"
        hidden = self.norm(
            self.gelu(self.linear(hidden, p + ".transform.dense")),
            p + ".transform.LayerNorm",
            cfg["layer_norm_eps"],
        )
        assert hidden.shape[-1] == width
        return self.linear(hidden, p + ".decoder")
