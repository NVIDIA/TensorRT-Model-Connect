# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure-TensorRT UMT5-XXL encoder for the native Wan2.2 checkpoint.

The public Wan2.2 release does not use the Hugging Face T5 state-dict layout.
It ships the encoder as ``models_t5_umt5-xxl-enc-bf16.pth`` with the names
from ``wan.modules.t5.T5Encoder``.  This module owns both that checkpoint
contract and the TensorRT graph so Wan2.2 does not depend on another model
family's implementation.

The production engine has two fixed inputs::

    input_ids      int32 [1, 512]
    attention_mask int32 [1, 512]  (one for a token, zero for padding)

It returns ``text_embeddings`` as FP32 ``[1, 512, 4096]``.  Every returned
value is first rounded to BF16 by the final source-compatible RMSNorm; the
FP32 output is only a lossless carrier for the BF16 values expected by the
Wan DiT input.

PyTorch is used only while reading the official ``.pth`` file at engine-build
time.  The serialized TensorRT engine has no PyTorch, ATen, or Torch-TensorRT
runtime dependency.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np
import tensorrt as trt


NATIVE_UMT5_CHECKPOINT = "models_t5_umt5-xxl-enc-bf16.pth"


@dataclass(frozen=True)
class Umt5EncoderConfig:
    """Shape and numerical contract for the Wan2.2 UMT5 encoder."""

    vocab_size: int = 256384
    hidden_size: int = 4096
    attention_size: int = 4096
    ffn_size: int = 10240
    num_heads: int = 64
    num_layers: int = 24
    num_buckets: int = 32
    relative_attention_max_distance: int = 128
    sequence_length: int = 512
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.attention_size % self.num_heads:
            raise ValueError("attention_size must be divisible by num_heads")
        for name in (
            "vocab_size",
            "hidden_size",
            "attention_size",
            "ffn_size",
            "num_heads",
            "num_layers",
            "num_buckets",
            "relative_attention_max_distance",
            "sequence_length",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def head_size(self) -> int:
        return self.attention_size // self.num_heads


WAN22_UMT5_XXL = Umt5EncoderConfig()


def expected_native_umt5_shapes(
    model: Umt5EncoderConfig = WAN22_UMT5_XXL,
) -> dict[str, tuple[int, ...]]:
    """Return the complete native checkpoint schema (242 tensors for XXL)."""

    shapes: dict[str, tuple[int, ...]] = {
        "token_embedding.weight": (model.vocab_size, model.hidden_size),
        "norm.weight": (model.hidden_size,),
    }
    for index in range(model.num_layers):
        prefix = f"blocks.{index}"
        shapes[f"{prefix}.norm1.weight"] = (model.hidden_size,)
        for projection in ("q", "k", "v"):
            shapes[f"{prefix}.attn.{projection}.weight"] = (
                model.attention_size,
                model.hidden_size,
            )
        shapes[f"{prefix}.attn.o.weight"] = (
            model.hidden_size,
            model.attention_size,
        )
        shapes[f"{prefix}.norm2.weight"] = (model.hidden_size,)
        shapes[f"{prefix}.ffn.gate.0.weight"] = (
            model.ffn_size,
            model.hidden_size,
        )
        shapes[f"{prefix}.ffn.fc1.weight"] = (
            model.ffn_size,
            model.hidden_size,
        )
        shapes[f"{prefix}.ffn.fc2.weight"] = (
            model.hidden_size,
            model.ffn_size,
        )
        shapes[f"{prefix}.pos_embedding.embedding.weight"] = (
            model.num_buckets,
            model.num_heads,
        )
    return shapes


def native_to_canonical_umt5_keys(
    model: Umt5EncoderConfig = WAN22_UMT5_XXL,
) -> dict[str, str]:
    """Map upstream Wan names to stable names consumed by this builder."""

    mapping = {
        "token_embedding.weight": "embedding.weight",
        "norm.weight": "final_norm.weight",
    }
    for index in range(model.num_layers):
        native = f"blocks.{index}"
        canonical = f"layers.{index}"
        mapping[f"{native}.norm1.weight"] = f"{canonical}.attention_norm.weight"
        for projection in ("q", "k", "v", "o"):
            mapping[f"{native}.attn.{projection}.weight"] = (
                f"{canonical}.attention.{projection}.weight"
            )
        mapping[f"{native}.norm2.weight"] = f"{canonical}.ffn_norm.weight"
        mapping[f"{native}.ffn.gate.0.weight"] = f"{canonical}.ffn.gate.weight"
        mapping[f"{native}.ffn.fc1.weight"] = f"{canonical}.ffn.fc1.weight"
        mapping[f"{native}.ffn.fc2.weight"] = f"{canonical}.ffn.fc2.weight"
        mapping[f"{native}.pos_embedding.embedding.weight"] = (
            f"{canonical}.relative_attention_bias.weight"
        )
    return mapping


def _native_bf16_numpy(value: Any, *, name: str) -> np.ndarray:
    """Expose native BF16 bits as NumPy without converting through FP16."""

    if str(value.dtype).removeprefix("torch.") != "bfloat16":
        raise TypeError(f"Wan2.2 UMT5 tensor {name!r} must be BF16, got {value.dtype}")

    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value, dtype=ml_dtypes.bfloat16)

    # torch.Tensor.numpy() does not support BF16 in every PyTorch release.
    # Viewing the storage as uint16 preserves every checkpoint bit and the
    # NumPy view keeps the CPU tensor storage alive for TensorRT's build.
    if not all(hasattr(value, attr) for attr in ("detach", "cpu", "view")):
        raise TypeError(f"Unsupported UMT5 tensor type for {name!r}: {type(value)!r}")
    import torch

    tensor = value.detach().cpu().contiguous()
    raw = tensor.view(torch.uint16).numpy()
    return raw.view(ml_dtypes.bfloat16).reshape(tuple(tensor.shape))


def convert_native_umt5_state_dict(
    state: Mapping[str, Any],
    *,
    model: Umt5EncoderConfig = WAN22_UMT5_XXL,
) -> dict[str, np.ndarray]:
    """Validate and map the complete official Wan UMT5 encoder state dict."""

    expected = expected_native_umt5_shapes(model)
    actual = set(state)
    missing = sorted(set(expected) - actual)
    unexpected = sorted(actual - set(expected))
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={missing[:8]}" + ("..." if len(missing) > 8 else ""))
        if unexpected:
            details.append(f"unexpected={unexpected[:8]}" + ("..." if len(unexpected) > 8 else ""))
        raise ValueError("Invalid Wan2.2 UMT5 checkpoint keys: " + ", ".join(details))

    key_map = native_to_canonical_umt5_keys(model)
    converted: dict[str, np.ndarray] = {}
    for native_name, shape in expected.items():
        value = state[native_name]
        actual_shape = tuple(value.shape)
        if actual_shape != shape:
            raise ValueError(
                f"Wan2.2 UMT5 tensor {native_name!r} has shape {actual_shape}; expected {shape}"
            )
        converted[key_map[native_name]] = _native_bf16_numpy(value, name=native_name)
    return converted


def load_native_umt5_weights(
    checkpoint_or_model_dir: str | Path,
    *,
    model: Umt5EncoderConfig = WAN22_UMT5_XXL,
) -> dict[str, np.ndarray]:
    """Load and validate the official native BF16 encoder checkpoint.

    This is an engine-build-time adapter.  The returned arrays retain the
    original BF16 bit patterns; the serialized plan itself has no Torch
    dependency.
    """

    import torch

    path = Path(checkpoint_or_model_dir)
    if path.is_dir():
        path = path / NATIVE_UMT5_CHECKPOINT
    if not path.is_file():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise TypeError(f"Expected a state dict in {path}, got {type(state)!r}")
    return convert_native_umt5_state_dict(state, model=model)


def relative_position_buckets(
    query_length: int,
    key_length: int,
    *,
    num_buckets: int = 32,
    max_distance: int = 128,
) -> np.ndarray:
    """Match upstream ``T5RelativeEmbedding`` bidirectional bucketing."""

    if num_buckets < 2 or num_buckets % 2:
        raise ValueError("bidirectional T5 num_buckets must be positive and even")
    half_buckets = num_buckets // 2
    max_exact = half_buckets // 2
    if max_exact < 1 or max_distance <= max_exact:
        raise ValueError("max_distance must be greater than num_buckets // 4")

    relative = (
        np.arange(key_length, dtype=np.int64)[None, :]
        - np.arange(query_length, dtype=np.int64)[:, None]
    )
    buckets = (relative > 0).astype(np.int64) * half_buckets
    distance = np.abs(relative)
    # Clamp before log so position zero has a defined unused large bucket.
    logarithmic = max_exact + (
        np.log(np.maximum(distance, 1).astype(np.float32) / max_exact)
        / math.log(max_distance / max_exact)
        * (half_buckets - max_exact)
    ).astype(np.int64)
    logarithmic = np.minimum(logarithmic, half_buckets - 1)
    buckets += np.where(distance < max_exact, distance, logarithmic)
    return np.ascontiguousarray(buckets, dtype=np.int32)


def _cast(network: Any, tensor: Any, dtype: Any) -> Any:
    if tensor.dtype == dtype:
        return tensor
    return network.add_cast(tensor, dtype).get_output(0)


def _bf16_constant(
    network: Any,
    value: Any,
    refs: list[np.ndarray],
    trt: Any,
) -> Any:
    array = np.ascontiguousarray(value, dtype=ml_dtypes.bfloat16)
    refs.append(array)
    weights = trt.Weights(trt.bfloat16, array.ctypes.data, array.size)
    return network.add_constant(tuple(array.shape), weights).get_output(0)


def _fp32_constant(network: Any, value: Any) -> Any:
    array = np.ascontiguousarray(value, dtype=np.float32)
    return network.add_constant(tuple(array.shape), array).get_output(0)


def _int32_constant(network: Any, value: Any) -> Any:
    array = np.ascontiguousarray(value, dtype=np.int32)
    return network.add_constant(tuple(array.shape), array).get_output(0)


def _linear_bf16(
    network: Any,
    x: Any,
    weight: np.ndarray,
    refs: list[np.ndarray],
    trt: Any,
) -> Any:
    """Bias-free source linear using native PyTorch ``[out, in]`` weights."""

    x = _cast(network, x, trt.bfloat16)
    rhs = _bf16_constant(network, weight, refs, trt)
    return network.add_matrix_multiply(
        x,
        trt.MatrixOperation.NONE,
        rhs,
        trt.MatrixOperation.TRANSPOSE,
    ).get_output(0)


def _native_bf16_barrier(network: Any, x: Any, name: str, trt: Any) -> Any:
    """Preserve a BF16 rounding boundary with TensorRT's native API.

    TensorRT's native ``mark_debug`` API prevents fusion across the marked
    tensor. Runtime emission remains disabled unless a caller opts in to tensor
    debugging, so this does not add an engine binding or a copy.
    """

    x = _cast(network, x, trt.bfloat16)
    x.name = name
    if not network.mark_debug(x):
        raise RuntimeError(f"Could not preserve native BF16 boundary {name!r}")
    return x


def _rms_norm_bf16(
    network: Any,
    x: Any,
    weight: np.ndarray,
    hidden_size: int,
    epsilon: float,
    refs: list[np.ndarray],
    trt: Any,
) -> Any:
    """Implement the precise FP32 -> BF16 -> BF16 upstream RMSNorm path."""

    x_fp32 = _cast(network, x, trt.float32)
    squared = network.add_elementwise(x_fp32, x_fp32, trt.ElementWiseOperation.PROD).get_output(0)
    mean = network.add_reduce(squared, trt.ReduceOperation.AVG, 1 << 1, True).get_output(0)
    epsilon_tensor = _fp32_constant(network, np.full((1, 1), epsilon, dtype=np.float32))
    variance = network.add_elementwise(
        mean, epsilon_tensor, trt.ElementWiseOperation.SUM
    ).get_output(0)
    root = network.add_unary(variance, trt.UnaryOperation.SQRT).get_output(0)
    inverse = network.add_unary(root, trt.UnaryOperation.RECIP).get_output(0)
    normalized = network.add_elementwise(x_fp32, inverse, trt.ElementWiseOperation.PROD).get_output(
        0
    )
    # Upstream casts normalized values to the BF16 parameter dtype before the
    # affine multiply.  Do not move this cast after the multiply.
    normalized = _cast(network, normalized, trt.bfloat16)
    gamma = _bf16_constant(network, np.asarray(weight).reshape(1, hidden_size), refs, trt)
    return network.add_elementwise(normalized, gamma, trt.ElementWiseOperation.PROD).get_output(0)


def _gelu_bf16(
    network: Any,
    x: Any,
    refs: list[np.ndarray],
    trt: Any,
    *,
    barrier_prefix: str,
) -> Any:
    """Upstream Wan GELU with native, non-fusible BF16 boundaries."""

    x = _cast(network, x, trt.bfloat16)
    x = _native_bf16_barrier(network, x, f"{barrier_prefix}_input", trt)
    square = network.add_elementwise(x, x, trt.ElementWiseOperation.PROD).get_output(0)
    square = _native_bf16_barrier(network, square, f"{barrier_prefix}_square", trt)
    cube = network.add_elementwise(square, x, trt.ElementWiseOperation.PROD).get_output(0)
    cube = _native_bf16_barrier(network, cube, f"{barrier_prefix}_cube", trt)

    cubic = network.add_elementwise(
        _cast(network, cube, trt.float32),
        _fp32_constant(network, np.full((1, 1), 0.044715, dtype=np.float32)),
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    cubic = _native_bf16_barrier(network, cubic, f"{barrier_prefix}_cubic", trt)
    polynomial = network.add_elementwise(x, cubic, trt.ElementWiseOperation.SUM).get_output(0)
    polynomial = _native_bf16_barrier(network, polynomial, f"{barrier_prefix}_polynomial", trt)

    tanh_input = network.add_elementwise(
        _cast(network, polynomial, trt.float32),
        _fp32_constant(
            network,
            np.full((1, 1), math.sqrt(2.0 / math.pi), dtype=np.float32),
        ),
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    tanh_input = _native_bf16_barrier(network, tanh_input, f"{barrier_prefix}_tanh_input", trt)
    tanh_output = network.add_activation(
        _cast(network, tanh_input, trt.float32), trt.ActivationType.TANH
    ).get_output(0)
    tanh_output = _native_bf16_barrier(network, tanh_output, f"{barrier_prefix}_tanh_output", trt)

    one = _bf16_constant(network, np.ones((1, 1), dtype=np.float32), refs, trt)
    shifted = network.add_elementwise(one, tanh_output, trt.ElementWiseOperation.SUM).get_output(0)
    shifted = _native_bf16_barrier(network, shifted, f"{barrier_prefix}_shifted", trt)
    scaled_x = network.add_elementwise(
        _cast(network, x, trt.float32),
        _fp32_constant(network, np.full((1, 1), 0.5, dtype=np.float32)),
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    scaled_x = _native_bf16_barrier(network, scaled_x, f"{barrier_prefix}_scaled_x", trt)
    result = network.add_elementwise(scaled_x, shifted, trt.ElementWiseOperation.PROD).get_output(0)
    return _native_bf16_barrier(network, result, f"{barrier_prefix}_output", trt)


def _relative_attention_bias(
    network: Any,
    weight: np.ndarray,
    bucket_indices: Any,
    model: Umt5EncoderConfig,
    refs: list[np.ndarray],
    trt: Any,
) -> Any:
    table = _bf16_constant(network, weight, refs, trt)
    gathered = network.add_gather(table, bucket_indices, 0).get_output(0)
    # [S,S,H] -> [1,H,S,S]
    shuffle = network.add_shuffle(gathered)
    shuffle.first_transpose = trt.Permutation([2, 0, 1])
    shuffle.reshape_dims = (
        1,
        model.num_heads,
        model.sequence_length,
        model.sequence_length,
    )
    return shuffle.get_output(0)


def _mask_attention_bias(
    network: Any,
    position_bias: Any,
    attention_mask: Any,
    model: Umt5EncoderConfig,
    refs: list[np.ndarray],
    trt: Any,
) -> Any:
    zero_i32 = _int32_constant(network, np.zeros((1, 1), dtype=np.int32))
    is_padding = network.add_elementwise(
        attention_mask, zero_i32, trt.ElementWiseOperation.EQUAL
    ).get_output(0)
    mask_4d = network.add_shuffle(is_padding)
    mask_4d.reshape_dims = (1, 1, 1, model.sequence_length)
    minimum = _bf16_constant(
        network,
        np.full(
            (1, 1, 1, 1),
            ml_dtypes.finfo(ml_dtypes.bfloat16).min,
        ),
        refs,
        trt,
    )
    return network.add_select(mask_4d.get_output(0), minimum, position_bias).get_output(0)


def _rows_to_heads(network: Any, x: Any, model: Umt5EncoderConfig, trt: Any) -> Any:
    shuffle = network.add_shuffle(x)
    shuffle.reshape_dims = (
        1,
        model.sequence_length,
        model.num_heads,
        model.head_size,
    )
    shuffle.second_transpose = trt.Permutation([0, 2, 1, 3])
    return shuffle.get_output(0)


def _heads_to_rows(network: Any, x: Any, model: Umt5EncoderConfig, trt: Any) -> Any:
    shuffle = network.add_shuffle(x)
    shuffle.first_transpose = trt.Permutation([0, 2, 1, 3])
    shuffle.reshape_dims = (model.sequence_length, model.attention_size)
    return shuffle.get_output(0)


def _self_attention_bf16(
    network: Any,
    x: Any,
    position_bias: Any,
    weights: Mapping[str, np.ndarray],
    prefix: str,
    model: Umt5EncoderConfig,
    refs: list[np.ndarray],
    trt: Any,
    *,
    barrier_prefix: str,
) -> Any:
    projected = []
    for projection in ("q", "k", "v"):
        value = _linear_bf16(network, x, weights[f"{prefix}.{projection}.weight"], refs, trt)
        value = _native_bf16_barrier(
            network, value, f"{barrier_prefix}_{projection}_projection", trt
        )
        projected.append(_rows_to_heads(network, value, model, trt))
    q, k, v = projected
    # UMT5 deliberately has no 1/sqrt(head_size) scaling.
    logits = network.add_matrix_multiply(
        q,
        trt.MatrixOperation.NONE,
        k,
        trt.MatrixOperation.TRANSPOSE,
    ).get_output(0)
    logits = _native_bf16_barrier(network, logits, f"{barrier_prefix}_qk_logits", trt)
    logits = network.add_elementwise(
        logits, position_bias, trt.ElementWiseOperation.SUM
    ).get_output(0)
    logits = _native_bf16_barrier(network, logits, f"{barrier_prefix}_biased_logits", trt)
    probabilities_fp32 = network.add_softmax(_cast(network, logits, trt.float32))
    probabilities_fp32.axes = 1 << 3
    probabilities = _cast(network, probabilities_fp32.get_output(0), trt.bfloat16)
    probabilities = _native_bf16_barrier(
        network, probabilities, f"{barrier_prefix}_probabilities", trt
    )
    context = network.add_matrix_multiply(
        probabilities,
        trt.MatrixOperation.NONE,
        v,
        trt.MatrixOperation.NONE,
    ).get_output(0)
    context = _native_bf16_barrier(network, context, f"{barrier_prefix}_pv_context", trt)
    context = _heads_to_rows(network, context, model, trt)
    output = _linear_bf16(network, context, weights[f"{prefix}.o.weight"], refs, trt)
    return _native_bf16_barrier(network, output, f"{barrier_prefix}_attention_output", trt)


def build_umt5_encoder_engine(
    weights: Mapping[str, np.ndarray],
    *,
    model: Umt5EncoderConfig = WAN22_UMT5_XXL,
    verbose: bool = False,
) -> bytes:
    """Build the fixed-shape, plugin-free UMT5 TensorRT plan."""

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    build_config = builder.create_builder_config()
    build_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 32 << 30)

    input_ids = network.add_input("input_ids", trt.int32, (1, model.sequence_length))
    attention_mask = network.add_input("attention_mask", trt.int32, (1, model.sequence_length))

    # Keep BF16 arrays alive until TensorRT has serialized the network.
    weight_refs: list[np.ndarray] = []
    embedding = _bf16_constant(network, weights["embedding.weight"], weight_refs, trt)
    gathered = network.add_gather(embedding, input_ids, 0).get_output(0)
    hidden_shuffle = network.add_shuffle(gathered)
    hidden_shuffle.reshape_dims = (model.sequence_length, model.hidden_size)
    hidden = hidden_shuffle.get_output(0)

    buckets = relative_position_buckets(
        model.sequence_length,
        model.sequence_length,
        num_buckets=model.num_buckets,
        max_distance=model.relative_attention_max_distance,
    )
    bucket_indices = _int32_constant(network, buckets)

    for index in range(model.num_layers):
        prefix = f"layers.{index}"
        normed = _rms_norm_bf16(
            network,
            hidden,
            weights[f"{prefix}.attention_norm.weight"],
            model.hidden_size,
            model.epsilon,
            weight_refs,
            trt,
        )
        normed = _native_bf16_barrier(
            network,
            normed,
            f"wan22_umt5_layer_{index}_attention_norm",
            trt,
        )
        position_bias = _relative_attention_bias(
            network,
            weights[f"{prefix}.relative_attention_bias.weight"],
            bucket_indices,
            model,
            weight_refs,
            trt,
        )
        position_bias = _mask_attention_bias(
            network,
            position_bias,
            attention_mask,
            model,
            weight_refs,
            trt,
        )
        attention_output = _self_attention_bf16(
            network,
            normed,
            position_bias,
            weights,
            f"{prefix}.attention",
            model,
            weight_refs,
            trt,
            barrier_prefix=f"wan22_umt5_layer_{index}",
        )
        hidden = network.add_elementwise(
            hidden, attention_output, trt.ElementWiseOperation.SUM
        ).get_output(0)
        hidden = _native_bf16_barrier(
            network,
            hidden,
            f"wan22_umt5_layer_{index}_attention_residual",
            trt,
        )

        normed = _rms_norm_bf16(
            network,
            hidden,
            weights[f"{prefix}.ffn_norm.weight"],
            model.hidden_size,
            model.epsilon,
            weight_refs,
            trt,
        )
        normed = _native_bf16_barrier(
            network,
            normed,
            f"wan22_umt5_layer_{index}_ffn_norm",
            trt,
        )
        fc1 = _linear_bf16(network, normed, weights[f"{prefix}.ffn.fc1.weight"], weight_refs, trt)
        fc1 = _native_bf16_barrier(network, fc1, f"wan22_umt5_layer_{index}_fc1", trt)
        gate = _linear_bf16(network, normed, weights[f"{prefix}.ffn.gate.weight"], weight_refs, trt)
        gate = _gelu_bf16(
            network,
            gate,
            weight_refs,
            trt,
            barrier_prefix=f"wan22_umt5_layer_{index}_source_gelu",
        )
        gated = network.add_elementwise(fc1, gate, trt.ElementWiseOperation.PROD).get_output(0)
        gated = _native_bf16_barrier(network, gated, f"wan22_umt5_layer_{index}_gated_ffn", trt)
        ffn_output = _linear_bf16(
            network,
            gated,
            weights[f"{prefix}.ffn.fc2.weight"],
            weight_refs,
            trt,
        )
        ffn_output = _native_bf16_barrier(
            network, ffn_output, f"wan22_umt5_layer_{index}_ffn_output", trt
        )
        hidden = network.add_elementwise(
            hidden, ffn_output, trt.ElementWiseOperation.SUM
        ).get_output(0)
        hidden = _native_bf16_barrier(
            network, hidden, f"wan22_umt5_layer_{index}_ffn_residual", trt
        )

    hidden = _rms_norm_bf16(
        network,
        hidden,
        weights["final_norm.weight"],
        model.hidden_size,
        model.epsilon,
        weight_refs,
        trt,
    )
    hidden = _native_bf16_barrier(network, hidden, "wan22_umt5_final_norm", trt)
    output_shape = network.add_shuffle(hidden)
    output_shape.reshape_dims = (1, model.sequence_length, model.hidden_size)
    output = _cast(network, output_shape.get_output(0), trt.float32)
    output.name = "text_embeddings"
    network.mark_output(output)

    print(
        "[wan2.2-umt5] Building pure TensorRT UMT5 encoder "
        f"(layers={model.num_layers}, seq={model.sequence_length}, "
        f"hidden={model.hidden_size}) ...",
        file=sys.stderr,
    )
    plan = builder.build_serialized_network(network, build_config)
    if plan is None:
        raise RuntimeError("TensorRT failed to serialize the Wan2.2 UMT5 encoder")
    return bytes(plan)


def build_native_umt5_encoder_engine(
    checkpoint_or_model_dir: str | Path,
    *,
    model: Umt5EncoderConfig = WAN22_UMT5_XXL,
    verbose: bool = False,
) -> bytes:
    """Load the official checkpoint and build its pure TensorRT encoder."""

    weights = load_native_umt5_weights(checkpoint_or_model_dir, model=model)
    return build_umt5_encoder_engine(
        weights,
        model=model,
        verbose=verbose,
    )
