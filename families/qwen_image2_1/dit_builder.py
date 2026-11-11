# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT engine for ``QwenImage21Transformer2DModel`` in text-to-image mode.

The model is a single-stream transformer over one joint sequence: ``text_tokens`` text slots
followed by the target image's ``latent_height * latent_width`` latent tokens.

* Attention is block-causal. Text rows attend causally to text; image rows attend to every valid
  text key and to every image key. The engine issues these as two attention calls per block: the
  causal text call needs no mask because prompts are padded on the right, and the image call takes
  an additive key mask that hides the padded text slots.
* ``causal_condition``: text rows are modulated from timestep 0 and image rows from the sampled
  timestep. One modulation projection is shared by all blocks, so the per-token scale and gate
  tensors are built once.
* The image block's rotary frame position is the prompt's token count. The engine derives it, and
  the key mask, from the ``text_length`` input.

Engine contract (FP32 I/O):
  ``latents``      [1, image_tokens, in_channels]    packed latents, row-major
  ``timestep``     [1]                               sigma in [0, 1], as the reference passes it
  ``text_hidden``  [1, text_tokens, context_dim]     text encoder states, zero or stale beyond length
  ``text_length``  [1] int32                         number of valid text slots, 1 .. text_tokens
  ``velocity``     [1, image_tokens, out_channels]   prediction for the image tokens
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import tensorrt as trt

from . import rope, trt_ops
from .trt_ops import BF16, FP32, Graph
from .weights import read_json, read_safetensors, take

_OPTIMIZATION_LEVEL = 3
_WORKSPACE_BYTES = 16 << 30
_TIME_FEATURES = 256
_TIME_FACTOR = 1000.0
_TIME_MAX_PERIOD = 10000.0


@dataclass(frozen=True)
class DenoiserConfig:
    in_channels: int
    out_channels: int
    num_layers: int
    num_heads: int
    head_dim: int
    context_dim: int
    mlp_ratio: int
    axes_dims_rope: tuple[int, int, int]
    eps: float

    @property
    def hidden(self) -> int:
        return self.num_heads * self.head_dim


def load_denoiser(transformer_dir: Path) -> tuple[DenoiserConfig, dict[str, np.ndarray]]:
    raw = read_json(transformer_dir / "config.json")
    if raw.get("_class_name") != "QwenImage21Transformer2DModel":
        raise ValueError(f"unsupported transformer class {raw.get('_class_name')!r}")
    if int(raw["patch_size"]) != 1:
        raise ValueError("Qwen-Image 2.1 consumes unpatched latents; patch_size must be 1")
    if not bool(raw["causal_condition"]):
        raise ValueError("this family implements causal_condition=true checkpoints only")
    config = DenoiserConfig(
        in_channels=int(raw["in_channels"]),
        # The reference reads a missing out_channels as in_channels.
        out_channels=int(raw.get("out_channels") or raw["in_channels"]),
        num_layers=int(raw["num_layers"]),
        num_heads=int(raw["num_attention_heads"]),
        head_dim=int(raw["attention_head_dim"]),
        context_dim=int(raw["context_in_dim"]),
        mlp_ratio=int(raw["mlp_ratio"]),
        axes_dims_rope=tuple(int(value) for value in raw["axes_dims_rope"]),
        eps=float(raw["eps"]),
    )
    if len(config.axes_dims_rope) != 3 or sum(config.axes_dims_rope) != config.head_dim:
        raise ValueError("axes_dims_rope must name three axes that sum to attention_head_dim")
    return config, read_safetensors(transformer_dir)


def _heads(graph: Graph, tensor: trt.ITensor, heads: int, head_dim: int) -> trt.ITensor:
    """``[1, S, heads * head_dim]`` -> ``[1, S, heads, head_dim]``."""
    return trt_ops.reshape(graph, tensor, (1, int(tensor.shape[1]), heads, head_dim))


def _rotate(graph: Graph, tensor: trt.ITensor, cos: trt.ITensor, sin: trt.ITensor) -> trt.ITensor:
    """Complex rotation of adjacent pairs of ``[1, S, heads, head_dim]`` in FP32.

    ``cos`` / ``sin`` are ``[1, S, 1, head_dim / 2]``. The pair ``(x[2j], x[2j + 1])`` is the complex
    number the reference multiplies by ``cos + i * sin``.
    """
    half = int(tensor.shape[-1]) // 2
    values = trt_ops.cast(graph, tensor, FP32)
    real = trt_ops.slice_axis(graph, values, 3, 0, half, stride=2)
    imaginary = trt_ops.slice_axis(graph, values, 3, 1, half, stride=2)
    rotated_real = trt_ops.sub(
        graph, trt_ops.mul(graph, real, cos), trt_ops.mul(graph, imaginary, sin)
    )
    rotated_imaginary = trt_ops.add(
        graph, trt_ops.mul(graph, real, sin), trt_ops.mul(graph, imaginary, cos)
    )
    shape = trt_ops.shape_of(tensor)
    pairs = trt_ops.concat(
        graph,
        [
            trt_ops.reshape(graph, rotated_real, (*shape[:3], half, 1)),
            trt_ops.reshape(graph, rotated_imaginary, (*shape[:3], half, 1)),
        ],
        4,
    )
    return trt_ops.cast(graph, trt_ops.reshape(graph, pairs, shape), graph.compute)


def _rotary_tables(
    graph: Graph,
    config: DenoiserConfig,
    text_length: trt.ITensor,
    *,
    text_tokens: int,
    latent_height: int,
    latent_width: int,
) -> tuple[trt.ITensor, trt.ITensor]:
    """``cos`` and ``sin`` of the joint sequence's rotary angles, each ``[1, S, 1, head_dim / 2]``."""
    image_tokens = latent_height * latent_width
    frame_pairs = config.axes_dims_rope[0] // 2
    text = rope.text_angles(text_tokens, config.axes_dims_rope)
    grid = rope.image_grid_angles(latent_height, latent_width, config.axes_dims_rope)

    # Frame angle of every image token: text_length * inverse frequency, float32 as in the reference.
    length = trt_ops.reshape(graph, trt_ops.cast(graph, text_length, FP32), (1, 1))
    inverse = rope.axis_inverse_frequencies(config.axes_dims_rope[0]).reshape(1, frame_pairs)
    frame = trt_ops.mul(graph, length, trt_ops.constant(graph, inverse))
    frame = trt_ops.mul(
        graph, frame, trt_ops.constant(graph, np.ones((image_tokens, 1), dtype=np.float32))
    )
    image = trt_ops.concat(graph, [frame, trt_ops.constant(graph, grid)], 1)
    angles = trt_ops.concat(graph, [trt_ops.constant(graph, text), image], 0)
    angles = trt_ops.reshape(graph, angles, (1, text_tokens + image_tokens, 1, config.head_dim // 2))
    cos = graph.network.add_unary(angles, trt.UnaryOperation.COS).get_output(0)
    sin = graph.network.add_unary(angles, trt.UnaryOperation.SIN).get_output(0)
    return cos, sin


def _text_key_mask(
    graph: Graph, text_length: trt.ITensor, *, text_tokens: int, image_tokens: int
) -> trt.ITensor:
    """Additive key mask ``[1, 1, 1, S]``: text slots at or beyond ``text_length`` are excluded."""
    slots = trt_ops.constant(graph, np.arange(text_tokens, dtype=np.int32))
    is_valid = graph.network.add_elementwise(
        slots, text_length, trt.ElementWiseOperation.LESS
    ).get_output(0)
    attend = trt_ops.constant(graph, np.zeros(text_tokens, dtype=np.float32))
    exclude = trt_ops.constant(
        graph, np.full(text_tokens, trt_ops.MASKED_SCORE, dtype=np.float32)
    )
    text = graph.network.add_select(is_valid, attend, exclude).get_output(0)
    image = trt_ops.constant(graph, np.zeros(image_tokens, dtype=np.float32))
    mask = trt_ops.concat(graph, [text, image], 0)
    return trt_ops.cast(
        graph, trt_ops.reshape(graph, mask, (1, 1, 1, text_tokens + image_tokens)), graph.compute
    )


def _time_embedding(
    graph: Graph, config: DenoiserConfig, weights: dict, timestep: trt.ITensor
) -> trt.ITensor:
    """Embeddings ``[2, hidden]``: row 0 for the sampled timestep, row 1 for timestep 0."""
    hidden = config.hidden
    half = _TIME_FEATURES // 2
    zero = trt_ops.constant(graph, np.zeros(1, dtype=np.float32))
    steps = trt_ops.reshape(graph, trt_ops.concat(graph, [timestep, zero], 0), (2, 1))
    scaled = trt_ops.mul_scalar(graph, steps, _TIME_FACTOR)
    frequencies = np.exp(
        -math.log(_TIME_MAX_PERIOD) * np.arange(half, dtype=np.float32) / np.float32(half)
    ).astype(np.float32)
    arguments = trt_ops.mul(graph, scaled, trt_ops.constant(graph, frequencies.reshape(1, half)))
    cos = graph.network.add_unary(arguments, trt.UnaryOperation.COS).get_output(0)
    sin = graph.network.add_unary(arguments, trt.UnaryOperation.SIN).get_output(0)
    features = trt_ops.cast(graph, trt_ops.concat(graph, [cos, sin], 1), graph.compute)
    prefix = "time_text_embed.timestep_embedder"
    first = take(weights, f"{prefix}.linear_1.weight", (hidden, _TIME_FEATURES))
    second = take(weights, f"{prefix}.linear_2.weight", (hidden, hidden))
    return trt_ops.linear(
        graph, trt_ops.silu(graph, trt_ops.linear(graph, features, first)), second
    )


def _per_token(
    graph: Graph, rows: trt.ITensor, *, text_tokens: int, image_tokens: int
) -> trt.ITensor:
    """Spread ``[2, hidden]`` over the sequence: image rows read row 0, text rows read row 1."""
    hidden = int(rows.shape[1])
    sampled = trt_ops.reshape(graph, trt_ops.slice_axis(graph, rows, 0, 0, 1), (1, 1, hidden))
    zero = trt_ops.reshape(graph, trt_ops.slice_axis(graph, rows, 0, 1, 1), (1, 1, hidden))
    text = trt_ops.mul(
        graph, zero, trt_ops.constant_compute(graph, np.ones((1, text_tokens, 1), dtype=np.float32))
    )
    image = trt_ops.mul(
        graph,
        sampled,
        trt_ops.constant_compute(graph, np.ones((1, image_tokens, 1), dtype=np.float32)),
    )
    return trt_ops.concat(graph, [text, image], 1)


def _block(
    graph: Graph,
    config: DenoiserConfig,
    weights: dict,
    index: int,
    hidden_states: trt.ITensor,
    *,
    modulation: tuple[trt.ITensor, trt.ITensor, trt.ITensor, trt.ITensor],
    cos: trt.ITensor,
    sin: trt.ITensor,
    key_mask: trt.ITensor,
    text_tokens: int,
    image_tokens: int,
) -> trt.ITensor:
    hidden = config.hidden
    heads = config.num_heads
    head_dim = config.head_dim
    sequence = text_tokens + image_tokens
    prefix = f"transformer_blocks.{index}"
    scale_attention, gate_attention, scale_mlp, gate_mlp = modulation

    def weight(name: str, shape: tuple[int, ...]) -> np.ndarray:
        return take(weights, f"{prefix}.{name}", shape)

    # Attention on the modulated, normalised sequence.
    normed = trt_ops.mul(
        graph, trt_ops.layer_norm(graph, hidden_states, config.eps), scale_attention
    )
    query = _heads(
        graph, trt_ops.linear(graph, normed, weight("attn.to_q.weight", (hidden, hidden))),
        heads, head_dim,
    )
    key = _heads(
        graph, trt_ops.linear(graph, normed, weight("attn.to_k.weight", (hidden, hidden))),
        heads, head_dim,
    )
    value = _heads(
        graph, trt_ops.linear(graph, normed, weight("attn.to_v.weight", (hidden, hidden))),
        heads, head_dim,
    )
    query = trt_ops.rms_norm(graph, query, weight("attn.norm_q.weight", (head_dim,)), config.eps)
    key = trt_ops.rms_norm(graph, key, weight("attn.norm_k.weight", (head_dim,)), config.eps)
    query = trt_ops.transpose(graph, _rotate(graph, query, cos, sin), (0, 2, 1, 3))
    key = trt_ops.transpose(graph, _rotate(graph, key, cos, sin), (0, 2, 1, 3))
    value = trt_ops.transpose(graph, value, (0, 2, 1, 3))

    text_context = trt_ops.attention(
        graph,
        trt_ops.slice_axis(graph, query, 2, 0, text_tokens),
        trt_ops.slice_axis(graph, key, 2, 0, text_tokens),
        trt_ops.slice_axis(graph, value, 2, 0, text_tokens),
        causal=True,
    )
    image_context = trt_ops.attention(
        graph,
        trt_ops.slice_axis(graph, query, 2, text_tokens, image_tokens),
        key,
        value,
        mask=key_mask,
    )
    context = trt_ops.concat(graph, [text_context, image_context], 2)
    context = trt_ops.reshape(
        graph, trt_ops.transpose(graph, context, (0, 2, 1, 3)), (1, sequence, hidden)
    )
    attended = trt_ops.linear(graph, context, weight("attn.to_out.0.weight", (hidden, hidden)))
    hidden_states = trt_ops.add(graph, hidden_states, trt_ops.mul(graph, gate_attention, attended))

    # SwiGLU feed-forward.
    inner = hidden * config.mlp_ratio
    normed = trt_ops.mul(graph, trt_ops.layer_norm(graph, hidden_states, config.eps), scale_mlp)
    gate = trt_ops.silu(
        graph, trt_ops.linear(graph, normed, weight("img_mlp.gate_layer.weight", (inner, hidden)))
    )
    projected = trt_ops.linear(graph, normed, weight("img_mlp.proj.weight", (inner, hidden)))
    feed_forward = trt_ops.linear(
        graph, trt_ops.mul(graph, gate, projected), weight("img_mlp.out.weight", (hidden, inner))
    )
    return trt_ops.add(graph, hidden_states, trt_ops.mul(graph, gate_mlp, feed_forward))


def build_denoiser_plan(
    config: DenoiserConfig,
    weights: dict[str, np.ndarray],
    output: BinaryIO,
    *,
    latent_height: int,
    latent_width: int,
    text_tokens: int,
    verbose: bool,
    compute: trt.DataType = BF16,
) -> None:
    """Build the denoiser engine for one static geometry and write its plan to ``output``.

    ``compute`` is BF16 for bundles; the parity tests pass FP32 to build the same graph without
    BF16 rounding.
    """
    if latent_height < 1 or latent_width < 1 or text_tokens < 1:
        raise ValueError("denoiser geometry must be positive")
    hidden = config.hidden
    image_tokens = latent_height * latent_width
    graph = trt_ops.create_graph(
        verbose=verbose,
        workspace_bytes=_WORKSPACE_BYTES,
        optimization_level=_OPTIMIZATION_LEVEL,
        compute=compute,
    )
    network = graph.network
    latents = network.add_input("latents", FP32, (1, image_tokens, config.in_channels))
    timestep = network.add_input("timestep", FP32, (1,))
    text_hidden = network.add_input("text_hidden", FP32, (1, text_tokens, config.context_dim))
    text_length = network.add_input("text_length", trt.int32, (1,))

    # Text projection: zero-centred RMSNorm (scale = weight + 1, FP32), two linears around GELU.
    text_scale = (
        take(weights, "txt_in.text_norm.weight", (config.context_dim,)).astype(np.float32) + 1.0
    )
    text = trt_ops.rms_norm_fp32_scale(graph, text_hidden, text_scale, config.eps)
    text = trt_ops.linear(
        graph, text, take(weights, "txt_in.in_layer.weight", (hidden, config.context_dim))
    )
    text = trt_ops.linear(
        graph, trt_ops.gelu_tanh(graph, text), take(weights, "txt_in.out_layer.weight", (hidden, hidden))
    )
    image = trt_ops.linear(
        graph, latents, take(weights, "img_in.weight", (hidden, config.in_channels))
    )
    hidden_states = trt_ops.concat(graph, [text, image], 1)

    # Shared modulation: [scale_attention, gate_attention, scale_mlp, gate_mlp] per timestep row.
    embedding = _time_embedding(graph, config, weights, timestep)
    shared = trt_ops.linear(
        graph,
        trt_ops.silu(graph, embedding),
        take(weights, "modulation.1.weight", (4 * hidden, hidden)),
    )
    chunks = [trt_ops.slice_axis(graph, shared, 1, index * hidden, hidden) for index in range(4)]
    spread = {"text_tokens": text_tokens, "image_tokens": image_tokens}
    modulation = (
        _per_token(graph, trt_ops.add_scalar(graph, chunks[0], 1.0), **spread),
        _per_token(graph, trt_ops.tanh(graph, chunks[1]), **spread),
        _per_token(graph, trt_ops.add_scalar(graph, chunks[2], 1.0), **spread),
        _per_token(graph, trt_ops.tanh(graph, chunks[3]), **spread),
    )

    cos, sin = _rotary_tables(
        graph,
        config,
        text_length,
        text_tokens=text_tokens,
        latent_height=latent_height,
        latent_width=latent_width,
    )
    key_mask = _text_key_mask(
        graph, text_length, text_tokens=text_tokens, image_tokens=image_tokens
    )
    for index in range(config.num_layers):
        hidden_states = _block(
            graph,
            config,
            weights,
            index,
            hidden_states,
            modulation=modulation,
            cos=cos,
            sin=sin,
            key_mask=key_mask,
            text_tokens=text_tokens,
            image_tokens=image_tokens,
        )

    # Output head on the image rows only: scale-only adaptive LayerNorm, then the projection.
    image_states = trt_ops.slice_axis(graph, hidden_states, 1, text_tokens, image_tokens)
    sampled = trt_ops.slice_axis(graph, embedding, 0, 0, 1)
    output_scale = trt_ops.linear(
        graph, trt_ops.silu(graph, sampled), take(weights, "norm_out.linear.weight", (hidden, hidden))
    )
    output_scale = trt_ops.reshape(graph, trt_ops.add_scalar(graph, output_scale, 1.0), (1, 1, hidden))
    normed = trt_ops.mul(graph, trt_ops.layer_norm(graph, image_states, config.eps), output_scale)
    velocity = trt_ops.linear(
        graph, normed, take(weights, "proj_out.weight", (config.out_channels, hidden))
    )
    velocity = trt_ops.cast(graph, velocity, FP32)
    velocity.name = "velocity"
    network.mark_output(velocity)

    output.write(memoryview(trt_ops.serialize(graph, "denoiser")))
