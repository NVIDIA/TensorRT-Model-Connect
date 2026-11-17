# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map the original Hugging Face LLaVA safetensors without a training runtime."""

import json
from pathlib import Path
import numpy as np
from safetensors import safe_open


class WeightDict(dict):
    pass


class Checkpoint:
    def __init__(self, directory):
        directory = Path(directory)
        index = directory / "model.safetensors.index.json"
        if index.is_file():
            mapping = json.loads(index.read_text())["weight_map"]
        else:
            mapping = {key: "model.safetensors" for key in
                       safe_open(str(directory / "model.safetensors"), framework="numpy").keys()}
        self.readers = {}
        self.mapping = mapping
        for filename in set(mapping.values()):
            if Path(filename).name != filename:
                raise ValueError("checkpoint shards must be local filenames")
            self.readers[filename] = safe_open(str(directory / filename), framework="numpy")

    def tensor(self, key):
        return self.readers[self.mapping[key]].get_tensor(key)


def load_decoder(checkpoint, config):
    weights = WeightDict()
    weights["embedding"] = checkpoint.tensor("language_model.model.embed_tokens.weight")
    weights["final_norm"] = checkpoint.tensor("language_model.model.norm.weight")
    weights["w_out"] = np.ascontiguousarray(checkpoint.tensor("language_model.lm_head.weight").T)
    for i in range(config.num_hidden_layers):
        source = f"language_model.model.layers.{i}"
        target = f"layer.{i}"
        for logical, key in (("input_norm", "input_layernorm"), ("post_attn_norm", "post_attention_layernorm")):
            weights[f"{target}.{logical}"] = checkpoint.tensor(f"{source}.{key}.weight")
        for logical, key in (
            ("w_q", "self_attn.q_proj"), ("w_k", "self_attn.k_proj"),
            ("w_v", "self_attn.v_proj"), ("w_o", "self_attn.o_proj"),
            ("w_gate", "mlp.gate_proj"), ("w_up", "mlp.up_proj"), ("w_down", "mlp.down_proj"),
        ):
            weights[f"{target}.{logical}"] = np.ascontiguousarray(checkpoint.tensor(f"{source}.{key}.weight").T)
    weights["_attention_size"] = config.hidden_size
    weights["_mlp_size"] = config.intermediate_size
    # Spell out QK/softmax/V with FP32 scores. This preserves Llama semantics
    # and supports the embed-input path without a fused attention operator.
    weights["_explicit_attention"] = True
    return weights
