# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent PyTorch equations following the pinned Flint SGLang reference.

Upstream: RecursiveSI/flint@2a247be0c0aec895ccb0ac7ca92229ec6df37288,
sglang/flint_sglang-0.1.4-py3-none-any.whl, model.py. No TRT graph helpers.
"""

import json
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors import safe_open


class Reference:
    def __init__(self, directory):
        self.root = Path(directory)
        self.config = json.loads((self.root / "config.json").read_text())
        self.files = json.loads((self.root / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        self.readers = {}

    def weight(self, name):
        filename = self.files[name]
        if filename not in self.readers:
            self.readers[filename] = safe_open(self.root / filename, framework="pt", device="cpu")
        return self.readers[filename].get_tensor(name)

    def norm(self, x, prefix):
        f = x.float()
        return (
            f
            * torch.rsqrt(f.square().mean(-1, keepdim=True) + self.config["rms_norm_eps"])
            * self.weight(prefix)
        ).to(x.dtype)

    def linear(self, x, name):
        return F.linear(x, self.weight(name).to(x.dtype))

    def swiglu(self, x, prefix):
        gate = self.linear(x, prefix + ".gate_proj.weight").float()
        up = self.linear(x, prefix + ".up_proj.weight").float()
        return self.linear((F.silu(gate) * up).to(x.dtype), prefix + ".down_proj.weight")

    def layer(self, index, x, past_k, past_v, mask, position):
        c = self.config
        prefix = f"model.layers.{index}"
        a = prefix + ".self_attn"
        normalized = self.norm(x, prefix + ".input_layernorm.weight")
        q = self.linear(normalized, a + ".q_proj.weight").reshape(
            c["num_attention_heads"], 1, c["head_dim"]
        )
        k = self.linear(normalized, a + ".k_proj.weight").reshape(
            c["num_key_value_heads"], 1, c["head_dim"]
        )
        v = self.linear(normalized, a + ".v_proj.weight").reshape(
            c["num_key_value_heads"], 1, c["head_dim"]
        )
        q, k = self.norm(q, a + ".q_norm.weight"), self.norm(k, a + ".k_norm.weight")
        if index not in c["no_rope_layer_indices"]:
            inv = c["rope_theta"] ** (
                -torch.arange(0, c["head_dim"], 2, dtype=torch.float32) / c["head_dim"]
            )
            angle = (inv * position).repeat(2)

            def rotate(t):
                left, right = t.float().chunk(2, -1)
                return (t.float() * angle.cos() + torch.cat((-right, left), -1) * angle.sin()).to(
                    t.dtype
                )

            q, k = rotate(q), rotate(k)
        k, v = torch.cat((past_k, k), 1), torch.cat((past_v, v), 1)
        repeated_k = k.repeat_interleave(c["num_attention_heads"] // c["num_key_value_heads"], 0)
        repeated_v = v.repeat_interleave(c["num_attention_heads"] // c["num_key_value_heads"], 0)
        scores = q.float() @ repeated_k.float().transpose(-1, -2) / c["head_dim"] ** 0.5 + mask
        sinks = self.weight(a + ".softmax_offset").float().reshape(-1, 1, 1)
        probabilities = torch.softmax(torch.cat((sinks, scores), -1), -1)[..., 1:]
        output = probabilities.to(x.dtype) @ repeated_v
        x = x + self.linear(output.reshape(1, -1), a + ".o_proj.weight")
        normalized = self.norm(x, prefix + ".post_attention_layernorm.weight")
        m = prefix + ".mlp"
        if index in c["mlp_only_layers"]:
            output = self.swiglu(normalized, m)
        else:
            scores = self.linear(normalized, m + ".gate.weight").float().sigmoid()
            bias = (
                self.weight(m + ".gate.expert_bias").float()
                if m + ".gate.expert_bias" in self.files
                else 0
            )
            ids = torch.topk(scores + bias, c["num_experts_per_tok"], dim=-1).indices
            probabilities = scores.gather(-1, ids)
            probabilities = probabilities / (probabilities.sum(-1, keepdim=True) + 1e-20)
            routed = torch.zeros_like(normalized, dtype=torch.float32)
            for e, probability in zip(ids[0], probabilities[0]):
                routed += self.swiglu(normalized, m + f".experts.{int(e)}").float() * probability
            output = routed.to(x.dtype) + self.swiglu(normalized, m + ".shared_expert")
        return x + output, k, v

    @torch.inference_mode()
    def logits_for_tokens(self, tokens):
        c = self.config
        caches = [
            (
                torch.empty(c["num_key_value_heads"], 0, c["head_dim"], dtype=torch.bfloat16),
                torch.empty(c["num_key_value_heads"], 0, c["head_dim"], dtype=torch.bfloat16),
            )
            for _ in range(c["num_hidden_layers"])
        ]
        embedding = self.weight("model.embed_tokens.weight")
        for position, token in enumerate(tokens):
            x = embedding[token].reshape(1, -1)
            for index, (k, v) in enumerate(caches):
                mask = torch.zeros(1, 1, k.shape[1] + 1)
                x, k, v = self.layer(index, x, k, v, mask, position)
                if c["layer_types"][index] == "sliding_attention":
                    k, v = k[:, -(c["sliding_window"] - 1) :], v[:, -(c["sliding_window"] - 1) :]
                caches[index] = k, v
            normalized = self.norm(x, "model.norm.weight").float()
            yield self.linear(normalized, "lm_head.weight")[0].float().numpy()
