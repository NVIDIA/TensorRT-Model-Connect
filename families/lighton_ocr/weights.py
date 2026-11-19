# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact LightOn checkpoint inventory and FP32 views of original tensors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors import safe_open


def expected_shapes(config):
    text, vision = config.text_config, config.vision_config
    hidden, intermediate = text["hidden_size"], text["intermediate_size"]
    query = text["num_attention_heads"] * text["head_dim"]
    compact = text["num_key_value_heads"] * text["head_dim"]
    result = {"model.language_model.embed_tokens.weight": (text["vocab_size"], hidden),
              "model.language_model.norm.weight": (hidden,)}
    for index in range(text["num_hidden_layers"]):
        prefix = f"model.language_model.layers.{index}."
        shapes = {"input_layernorm.weight": (hidden,), "post_attention_layernorm.weight": (hidden,),
                  "self_attn.q_proj.weight": (query, hidden), "self_attn.k_proj.weight": (compact, hidden),
                  "self_attn.v_proj.weight": (compact, hidden), "self_attn.o_proj.weight": (hidden, query),
                  "self_attn.q_norm.weight": (text["head_dim"],), "self_attn.k_norm.weight": (text["head_dim"],),
                  "mlp.gate_proj.weight": (intermediate, hidden), "mlp.up_proj.weight": (intermediate, hidden),
                  "mlp.down_proj.weight": (hidden, intermediate)}
        result.update({prefix + name: shape for name, shape in shapes.items()})
    width, middle, patch = vision["hidden_size"], vision["intermediate_size"], vision["patch_size"]
    result.update({"model.vision_encoder.patch_conv.weight": (width, 3, patch, patch),
                   "model.vision_encoder.ln_pre.weight": (width,)})
    for index in range(vision["num_hidden_layers"]):
        prefix = f"model.vision_encoder.transformer.layers.{index}."
        shapes = {"attention_norm.weight": (width,), "ffn_norm.weight": (width,),
                  "attention.q_proj.weight": (width, width), "attention.k_proj.weight": (width, width),
                  "attention.v_proj.weight": (width, width), "attention.o_proj.weight": (width, width),
                  "feed_forward.gate_proj.weight": (middle, width), "feed_forward.up_proj.weight": (middle, width),
                  "feed_forward.down_proj.weight": (width, middle)}
        result.update({prefix + name: shape for name, shape in shapes.items()})
    result.update({"model.vision_projection.norm.weight": (width,),
                   "model.vision_projection.patch_merger.merging_layer.weight":
                       (width, width * config.spatial_merge_size ** 2),
                   "model.vision_projection.linear_1.weight": (hidden, width),
                   "model.vision_projection.linear_2.weight": (hidden, hidden)})
    return result


class Checkpoint:
    def __init__(self, directory, config):
        self.directory, self.config = Path(directory), config
        index = self.directory / "model.safetensors.index.json"
        mapping = json.loads(index.read_text()).get("weight_map", {}) if index.is_file() else None
        if mapping is not None:
            if not mapping or not all(isinstance(key, str) and isinstance(value, str) for key, value in mapping.items()):
                raise ValueError("lighton_ocr checkpoint index has no valid tensor mapping")
            filenames = sorted(set(mapping.values()))
        else:
            filenames = ["model.safetensors"]
        self.tensors, self.shapes, self.dtypes = {}, {}, {}
        for filename in filenames:
            if Path(filename).name != filename or not filename.endswith(".safetensors"):
                raise ValueError("lighton_ocr checkpoint index contains an unsafe shard filename")
            path = self.directory / filename
            with safe_open(path, framework="pt", device="cpu") as source:
                for name in source.keys():
                    if name in self.tensors:
                        raise ValueError(f"lighton_ocr checkpoint repeats tensor {name}")
                    if mapping is not None and mapping.get(name) != filename:
                        raise ValueError(f"lighton_ocr index does not own tensor {name}")
                    tensor = source.get_slice(name)
                    self.tensors[name] = path
                    self.shapes[name] = tuple(tensor.get_shape())
                    self.dtypes[name] = tensor.get_dtype()
        if mapping is not None and set(mapping) != set(self.tensors):
            raise ValueError("lighton_ocr checkpoint index names missing tensors")

    def validate(self):
        shapes = expected_shapes(self.config)
        optional = {"lm_head.weight"}
        if not set(shapes) <= set(self.tensors) or set(self.tensors) - set(shapes) - optional:
            raise ValueError("lighton_ocr checkpoint tensor inventory does not match its full model")
        if "lm_head.weight" in self.tensors:
            shapes["lm_head.weight"] = shapes["model.language_model.embed_tokens.weight"]
        for name, shape in shapes.items():
            if self.shapes[name] != shape or self.dtypes[name] not in {"BF16", "F16", "F32"}:
                raise ValueError(f"lighton_ocr tensor {name} has incompatible shape or storage dtype")
        if "lm_head.weight" in self.tensors:
            if not np.array_equal(self.tensor("lm_head.weight"), self.tensor("model.language_model.embed_tokens.weight")):
                raise ValueError("lighton_ocr requires exactly tied language embeddings and output weights")

    def tensor(self, name):
        with safe_open(self.tensors[name], framework="pt", device="cpu") as source:
            return source.get_tensor(name).float().numpy()

    def embedding(self):
        return self.tensor("model.language_model.embed_tokens.weight")

    def vision(self):
        result = {}
        for name in self.tensors:
            if name.startswith("model.vision_encoder."):
                result[name.removeprefix("model.vision_encoder.")] = self.tensor(name)
            elif name.startswith("model.vision_projection."):
                result[name.removeprefix("model.")] = self.tensor(name)
        return result

    def decoder(self, embedding=None):
        prefix = "model.language_model."
        result = {name.removeprefix(prefix): self.tensor(name) for name in self.tensors
                  if name.startswith(prefix) and name != prefix + "embed_tokens.weight"}
        result["lm_head.weight"] = self.embedding() if embedding is None else embedding
        return result
