# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The checkpoint tensors the Mask2Former graph reads, checked against the configuration before any is loaded.

Most layer widths reach the graph through the weights, so a configuration that disagrees with its weights would build
the weights' topology and ignore the configuration. Every tensor must therefore exist with the dtype and shape the
configuration implies, and the checkpoint may hold no other tensor than the training-only `criterion.empty_weight`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from safetensors import safe_open

from .config import ENCODER_POINTS, FEATURE_LEVELS, Mask2FormerConfig

# A training-only buffer that inference never reads.
_UNUSED = frozenset({"criterion.empty_weight"})
_DTYPES = {"F32": np.float32, "I64": np.int64}


def expected_tensors(config: Mask2FormerConfig) -> dict[str, tuple[tuple[int, ...], str]]:
    """Name -> (shape, safetensors dtype) of every tensor the graph reads."""

    tensors: dict[str, tuple[tuple[int, ...], str]] = {}

    def add(name: str, *shape: int, dtype: str = "F32") -> None:
        tensors[name] = (tuple(shape), dtype)

    def linear(prefix: str, out_features: int, in_features: int, *, bias: bool = True) -> None:
        add(f"{prefix}.weight", out_features, in_features)
        if bias:
            add(f"{prefix}.bias", out_features)

    def norm(prefix: str, width: int) -> None:
        add(f"{prefix}.weight", width)
        add(f"{prefix}.bias", width)

    embed, window = config.embed_dim, config.window_size
    backbone = "model.pixel_level_module.encoder"
    add(
        f"{backbone}.embeddings.patch_embeddings.projection.weight",
        embed,
        3,
        config.patch_size,
        config.patch_size,
    )
    add(f"{backbone}.embeddings.patch_embeddings.projection.bias", embed)
    norm(f"{backbone}.embeddings.norm", embed)
    for stage, (depth, heads, width) in enumerate(
        zip(config.depths, config.num_heads, config.stage_channels)
    ):
        hidden = int(config.mlp_ratio * width)
        for block in range(depth):
            prefix = f"{backbone}.encoder.layers.{stage}.blocks.{block}"
            norm(f"{prefix}.layernorm_before", width)
            for name in ("query", "key", "value"):
                linear(f"{prefix}.attention.self.{name}", width, width)
            add(
                f"{prefix}.attention.self.relative_position_bias_table",
                (2 * window - 1) ** 2,
                heads,
            )
            add(
                f"{prefix}.attention.self.relative_position_index",
                window * window,
                window * window,
                dtype="I64",
            )
            linear(f"{prefix}.attention.output.dense", width, width)
            norm(f"{prefix}.layernorm_after", width)
            linear(f"{prefix}.intermediate.dense", hidden, width)
            linear(f"{prefix}.output.dense", width, hidden)
        if stage + 1 < len(config.depths):
            norm(f"{backbone}.encoder.layers.{stage}.downsample.norm", 4 * width)
            linear(
                f"{backbone}.encoder.layers.{stage}.downsample.reduction",
                2 * width,
                4 * width,
                bias=False,
            )
        norm(f"{backbone}.hidden_states_norms.stage{stage + 1}", width)

    features, heads = config.feature_size, config.num_attention_heads
    decoder = "model.pixel_level_module.decoder"
    add(f"{decoder}.level_embed", FEATURE_LEVELS, features)
    # The transformer levels, coarsest first: stage 4, stage 3, stage 2.
    for index, channels in enumerate(reversed(config.stage_channels[-FEATURE_LEVELS:])):
        add(f"{decoder}.input_projections.{index}.0.weight", features, channels, 1, 1)
        add(f"{decoder}.input_projections.{index}.0.bias", features)
        norm(f"{decoder}.input_projections.{index}.1", features)
    for layer in range(config.encoder_layers):
        prefix = f"{decoder}.encoder.layers.{layer}"
        points = heads * FEATURE_LEVELS * ENCODER_POINTS
        linear(f"{prefix}.self_attn.sampling_offsets", 2 * points, features)
        linear(f"{prefix}.self_attn.attention_weights", points, features)
        linear(f"{prefix}.self_attn.value_proj", features, features)
        linear(f"{prefix}.self_attn.output_proj", features, features)
        norm(f"{prefix}.self_attn_layer_norm", features)
        linear(f"{prefix}.fc1", config.encoder_feedforward_dim, features)
        linear(f"{prefix}.fc2", features, config.encoder_feedforward_dim)
        norm(f"{prefix}.final_layer_norm", features)
    add(f"{decoder}.adapter_1.0.weight", features, config.stage_channels[0], 1, 1)
    norm(f"{decoder}.adapter_1.1", features)
    add(f"{decoder}.layer_1.0.weight", features, features, 3, 3)
    norm(f"{decoder}.layer_1.1", features)
    add(f"{decoder}.mask_projection.weight", config.mask_feature_size, features, 1, 1)
    add(f"{decoder}.mask_projection.bias", config.mask_feature_size)

    hidden_dim, module = config.hidden_dim, "model.transformer_module"
    add(f"{module}.queries_embedder.weight", config.num_queries, hidden_dim)
    add(f"{module}.queries_features.weight", config.num_queries, hidden_dim)
    add(f"{module}.level_embed.weight", FEATURE_LEVELS, hidden_dim)
    norm(f"{module}.decoder.layernorm", hidden_dim)
    for layer in range(config.decoder_layers - 1):
        prefix = f"{module}.decoder.layers.{layer}"
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            linear(f"{prefix}.self_attn.{name}", hidden_dim, hidden_dim)
        norm(f"{prefix}.self_attn_layer_norm", hidden_dim)
        add(f"{prefix}.cross_attn.in_proj_weight", 3 * hidden_dim, hidden_dim)
        add(f"{prefix}.cross_attn.in_proj_bias", 3 * hidden_dim)
        linear(f"{prefix}.cross_attn.out_proj", hidden_dim, hidden_dim)
        norm(f"{prefix}.cross_attn_layer_norm", hidden_dim)
        linear(f"{prefix}.fc1", config.dim_feedforward, hidden_dim)
        linear(f"{prefix}.fc2", hidden_dim, config.dim_feedforward)
        norm(f"{prefix}.final_layer_norm", hidden_dim)
    for index, out_features in enumerate((hidden_dim, hidden_dim, config.mask_feature_size)):
        linear(f"{module}.decoder.mask_predictor.mask_embedder.{index}.0", out_features, hidden_dim)
    linear("class_predictor", config.num_labels + 1, hidden_dim)
    return tensors


def _names(kind: str, names: list[str]) -> str:
    shown = ", ".join(names[:3]) + (", ..." if len(names) > 3 else "")
    return f"{len(names)} {kind} ({shown})"


def load(model_dir: Path, config: Mask2FormerConfig) -> dict[str, np.ndarray]:
    """Check every tensor's name, dtype and shape from the safetensors header, then load them."""

    path = Path(model_dir) / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"missing mask2former weights: {path}")
    expected = expected_tensors(config)
    with safe_open(str(path), framework="numpy") as reader:
        names = set(reader.keys())
        missing = sorted(set(expected) - names)
        unexpected = sorted(names - set(expected) - _UNUSED)
        if missing or unexpected:
            found = [
                _names(kind, items)
                for kind, items in (("missing", missing), ("unexpected", unexpected))
                if items
            ]
            raise ValueError(
                "mask2former checkpoint tensors do not match its configuration: " + "; ".join(found)
            )
        for name, (shape, dtype) in expected.items():
            header = reader.get_slice(name)
            found = (tuple(header.get_shape()), header.get_dtype())
            if found != (shape, dtype):
                raise ValueError(
                    f"mask2former tensor {name} is {found[1]} {list(found[0])}, "
                    f"but the configuration implies {dtype} {list(shape)}"
                )
        weights = {name: reader.get_tensor(name) for name in expected}
    for name, (_, dtype) in expected.items():
        if weights[name].dtype != _DTYPES[dtype]:
            raise ValueError(f"mask2former tensor {name} did not load as {dtype}")
    return weights
