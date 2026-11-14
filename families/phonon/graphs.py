# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phonon's mathematical graphs in original checkpoint coordinates.

These modules are exported at build time; native inference executes TensorRT
plans only. RoPE rotates interleaved pairs in the original Gradium weights.
"""

import torch
from torch import nn
from torch.nn import functional as F


class Graph(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.names = {}
        for i, (name, value) in enumerate(weights.items()):
            key = f"w{i}"
            self.register_buffer(key, value.float().contiguous())
            self.names[name] = key

    def w(self, name):
        return getattr(self, self.names[name])

    def linear(self, x, name):
        bias = self.w(name + ".bias") if name + ".bias" in self.names else None
        return F.linear(x, self.w(name + ".weight"), bias)

    def norm(self, x, name, eps=1e-5):
        return F.layer_norm(
            x,
            (int(self.w(name + ".weight").numel()),),
            self.w(name + ".weight"),
            self.w(name + ".bias"),
            eps,
        )

    def transformer(self, x, prefix, layers, heads, period, context=None):
        length, width = x.shape[1], 512
        head_dim = width // heads
        positions = torch.arange(length, device=x.device)
        freqs = torch.exp(
            -torch.log(torch.tensor(float(period)))
            * torch.arange(head_dim // 2, device=x.device)
            / (head_dim // 2)
        )
        angles = positions.float()[:, None] * freqs[None, :]
        cos, sin = angles.cos()[None, None, :, :], angles.sin()[None, None, :, :]
        distance = positions[:, None] - positions[None, :]
        allowed = distance >= 0
        if context is not None:
            allowed = allowed & (distance <= context)
        mask = torch.where(allowed, 0.0, -float("inf"))[None, None, :, :]

        def rotate(t):
            pairs = t.reshape(1, heads, length, head_dim // 2, 2)
            a, b = pairs[..., 0], pairs[..., 1]
            return torch.stack((a * cos - b * sin, a * sin + b * cos), -1).flatten(-2)

        for i in range(layers):
            p = f"{prefix}.layers.{i}"
            qkv = self.linear(self.norm(x, p + ".norm1"), p + ".self_attn.in_proj")
            q, k, v = [
                t.reshape(1, length, heads, head_dim).transpose(1, 2) for t in qkv.chunk(3, dim=-1)
            ]
            q, k = rotate(q), rotate(k)
            scores = (q @ k.transpose(-1, -2)) * (head_dim**-0.5) + mask
            attention = (scores.softmax(-1) @ v).transpose(1, 2).reshape(1, length, width)
            attention = self.linear(attention, p + ".self_attn.out_proj")
            if p + ".layer_scale_1.scale" in self.names:
                attention = attention * self.w(p + ".layer_scale_1.scale")
            x = x + attention
            ff = self.linear(self.norm(x, p + ".norm2"), p + ".linear1")
            ff = self.linear(F.gelu(ff, approximate="none"), p + ".linear2")
            if p + ".layer_scale_2.scale" in self.names:
                ff = ff * self.w(p + ".layer_scale_2.scale")
            x = x + ff
        return x


class Backbone(Graph):
    def forward(self, embeddings):
        x = self.transformer(embeddings, "flow_lm.transformer", 6, 8, 10000.0)
        last = self.norm(x[:, -1:, :], "flow_lm.out_norm").squeeze(1)
        return last, self.linear(last, "flow_lm.out_eos")


class Flow(Graph):
    def forward(self, noise, condition):
        p = "flow_lm.flow_net"
        x = self.linear(noise, p + ".input_proj")
        cond = self.linear(condition, p + ".cond_embed")
        times = []
        for i, time in enumerate((0.0, 1.0)):
            t = f"{p}.time_embed.{i}"
            angle = self.w(t + ".freqs") * time
            e = torch.cat((angle.cos(), angle.sin()), -1).reshape(1, -1)
            e = self.linear(F.silu(self.linear(e, t + ".mlp.0")), t + ".mlp.2")
            # Phonon 31m uses RMS normalization, not the older Pocket variance norm.
            e = e * torch.rsqrt((e * e).mean(-1, keepdim=True) + 1e-5)
            times.append(e * self.w(t + ".mlp.3.alpha"))
        cond = cond + (times[0] + times[1]) / 2
        for i in range(6):
            r = f"{p}.res_blocks.{i}"
            shift, scale, gate = self.linear(F.silu(cond), r + ".adaLN_modulation.1").chunk(3, -1)
            modulated = self.norm(x, r + ".in_ln", 1e-6) * (1 + scale) + shift
            x = x + gate * self.linear(F.silu(self.linear(modulated, r + ".mlp.0")), r + ".mlp.2")
        r = p + ".final_layer"
        shift, scale = self.linear(F.silu(cond), r + ".adaLN_modulation.1").chunk(2, -1)
        x = F.layer_norm(x, (512,), eps=1e-6) * (1 + scale) + shift
        return noise + self.linear(x, r + ".linear")


class Codec(Graph):
    def conv(self, x, name):
        weight = self.w(name + ".weight")
        return F.conv1d(F.pad(x, (weight.shape[-1] - 1, 0)), weight, self.w(name + ".bias"))

    def transpose_conv(self, x, name, stride, groups=1):
        weight = self.w(name + ".weight")
        bias = self.w(name + ".bias") if name + ".bias" in self.names else None
        x = F.conv_transpose1d(x, weight, bias, stride=stride, groups=groups)
        return x[..., : -(weight.shape[-1] - stride)]

    def forward(self, latents):
        x = latents * self.w("flow_lm.emb_std") + self.w("flow_lm.emb_mean")
        x = F.conv1d(x.transpose(1, 2), self.w("mimi.quantizer.output_proj.weight"))
        x = self.transpose_conv(x, "mimi.upsample.convtr.convtr", 16, 512)
        x = self.transformer(
            x.transpose(1, 2), "mimi.decoder_transformer.transformer", 2, 8, 10000.0, context=250
        ).transpose(1, 2)
        p = "mimi.decoder.model"
        x = self.conv(x, p + ".0.conv")
        for i, ratio in enumerate((6, 5, 4)):
            x = self.transpose_conv(F.elu(x), f"{p}.{3 * i + 2}.convtr", ratio)
            r = f"{p}.{3 * i + 3}.block"
            y = self.conv(F.elu(x), r + ".1.conv")
            x = x + self.conv(F.elu(y), r + ".3.conv")
        return self.conv(F.elu(x), p + ".11.conv")
