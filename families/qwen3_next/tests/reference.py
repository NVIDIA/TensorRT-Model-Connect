# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact-checkpoint adapters around the official Transformers 5.2.0 model.

No native graph helpers or copied inference equations are used as the oracle.
"""

import json
from pathlib import Path
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextDecoderLayer,
    Qwen3NextDynamicCache,
    Qwen3NextRotaryEmbedding,
)


class OfficialReference:
    def __init__(self, directory):
        self.root = Path(directory)
        self.config = AutoConfig.from_pretrained(self.root, trust_remote_code=False)
        self.config._attn_implementation = "eager"
        self.config._experts_implementation = "eager"
        self.files = json.loads((self.root / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        self.readers = {}

    def weight(self, name):
        filename = self.files[name]
        if filename not in self.readers:
            self.readers[filename] = safe_open(self.root / filename, framework="pt", device="cpu")
        return self.readers[filename].get_tensor(name).to(torch.bfloat16)

    def layer(self, index):
        with torch.device("meta"):
            layer = Qwen3NextDecoderLayer(self.config, index)
        parameters = {}
        prefix = f"model.layers.{index}."
        for name, _ in layer.named_parameters():
            if name == "mlp.experts.gate_up_proj":
                parameters[name] = torch.stack(
                    [
                        torch.cat(
                            (
                                self.weight(prefix + f"mlp.experts.{e}.gate_proj.weight"),
                                self.weight(prefix + f"mlp.experts.{e}.up_proj.weight"),
                            ),
                            dim=0,
                        )
                        for e in range(self.config.num_experts)
                    ]
                )
            elif name == "mlp.experts.down_proj":
                parameters[name] = torch.stack(
                    [
                        self.weight(prefix + f"mlp.experts.{e}.down_proj.weight")
                        for e in range(self.config.num_experts)
                    ]
                )
            else:
                parameters[name] = self.weight(prefix + name)
        layer.load_state_dict(parameters, strict=True, assign=True)
        return layer.eval()

    @torch.inference_mode()
    def run_layer(self, index, hidden, inputs, position):
        layer = self.layer(index)
        cache = Qwen3NextDynamicCache(self.config)
        x = hidden.reshape(1, 1, -1)
        cache_position = torch.tensor([position], dtype=torch.int64)
        if self.config.layer_types[index] == "linear_attention":
            history = inputs["conv_state"]
            # The official update retains four entries; the oldest one is not
            # used by the next convolution. The native cache retains three.
            cache.conv_states[index] = torch.cat(
                (torch.zeros_like(history[:, :1]), history), dim=-1
            ).unsqueeze(0)
            cache.recurrent_states[index] = inputs["recurrent_state"].unsqueeze(0)
            if cache.last_linear_layer != index:
                cache.conv_states[cache.last_linear_layer] = torch.zeros(1)
            output = layer(
                x, position_embeddings=None, past_key_values=cache, cache_position=cache_position
            )
            return {
                "hidden_out": output.reshape(1, -1),
                "present_conv": cache.conv_states[index][0, :, -history.shape[-1] :],
                "present_recurrent": cache.recurrent_states[index][0],
            }
        cache.key_cache[index] = inputs["past_k"].unsqueeze(0)
        cache.value_cache[index] = inputs["past_v"].unsqueeze(0)
        rope = Qwen3NextRotaryEmbedding(self.config)
        embeddings = rope(x, cache_position.reshape(1, 1))
        output = layer(
            x,
            position_embeddings=embeddings,
            past_key_values=cache,
            cache_position=cache_position,
            attention_mask=inputs["mask"].to(torch.bfloat16).unsqueeze(0),
        )
        return {
            "hidden_out": output.reshape(1, -1),
            "present_k": cache.key_cache[index][0],
            "present_v": cache.value_cache[index][0],
        }


def load_full_model(directory):
    """Load the actual upstream model on CPU without optional GPU kernels."""
    model, loading = AutoModelForCausalLM.from_pretrained(
        directory,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=False,
        local_files_only=True,
        output_loading_info=True,
    )
    for field in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if loading.get(field):
            raise RuntimeError(
                f"incomplete official Qwen3-Next reference: {field}={loading[field]}"
            )
    return model.eval()
