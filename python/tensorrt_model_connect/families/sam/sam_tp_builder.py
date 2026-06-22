"""Tensor-parallel SAM image encoder builder.

SAM ViT attention and the mask decoder remain replicated. The encoder MLPs are
tensor-parallel: FC1 columns are sharded, FC2 rows are sharded, and a TensorRT
distributed ALL_REDUCE restores the full residual before the next layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import sys

import numpy as np
from tensorrt_model_connect import trt_compat

from . import graph_ops
from ...parallel_config import add_all_reduce_sum, normalize_parallel_config
from .plugin import SamPlugin, _resolve_sam_config

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict
    from .config import ModelConfig
    from ...parallel_config import ParallelConfig


def _validate_sam_encoder_tp(config: "ModelConfig", parallel: "ParallelConfig") -> None:
    parallel.validate()
    if not parallel.enabled:
        return
    if parallel.rank < 0:
        raise ValueError("SAM tensor-parallel build requires a concrete rank")
    sam_cfg = config.raw.get("_sam_config", _resolve_sam_config(config.raw))
    mlp_dim = int(sam_cfg["mlp_dim"])
    if mlp_dim % parallel.tp_size != 0:
        raise ValueError(
            "SAM tensor-parallel encoder requires mlp_dim divisible by tp_size "
            f"({mlp_dim} vs {parallel.tp_size})")


def _slice_mlp_columns(arr: np.ndarray, mlp_dim: int, parallel: "ParallelConfig") -> np.ndarray:
    local = mlp_dim // parallel.tp_size
    start = parallel.rank * local
    end = start + local
    return np.ascontiguousarray(arr[..., start:end])


def _slice_mlp_rows(arr: np.ndarray, mlp_dim: int, parallel: "ParallelConfig") -> np.ndarray:
    local = mlp_dim // parallel.tp_size
    start = parallel.rank * local
    end = start + local
    return np.ascontiguousarray(arr[start:end, ...])


def build_sam_tp_encoder_engine(
    config: "ModelConfig",
    weights: "WeightDict",
    max_cache_length: int,
    *,
    precision: str = "fp32",
    quant_ctx=None,
    verbose: bool = False,
    parallel_config=None,
) -> bytes:
    """Build one rank-local SAM image encoder with tensor-parallel MLPs."""
    del max_cache_length, precision
    if quant_ctx is not None:
        raise ValueError("SAM tensor-parallel builds do not support quantization")
    parallel = normalize_parallel_config(parallel_config)
    if not parallel.enabled:
        raise ValueError("SAM tensor-parallel builder requires an enabled parallel config")
    _validate_sam_encoder_tp(config, parallel)

    sam_cfg = config.raw.get("_sam_config", _resolve_sam_config(config.raw))
    hidden = sam_cfg["hidden_size"]
    num_layers = sam_cfg["num_hidden_layers"]
    num_heads = sam_cfg["num_attention_heads"]
    head_dim = hidden // num_heads
    mlp_dim = sam_cfg["mlp_dim"]
    local_mlp_dim = mlp_dim // parallel.tp_size
    image_size = sam_cfg["image_size"]
    patch_size = sam_cfg["patch_size"]
    window_size = sam_cfg["window_size"]
    global_attn_indexes = set(sam_cfg["global_attn_indexes"])
    decoder_hidden = sam_cfg["decoder_hidden_size"]

    grid_size = image_size // patch_size
    seq_len = grid_size * grid_size

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

    eps_t = graph_ops.add_constant(
        network, (1, 1), np.array([1e-6], dtype=np.float32))

    pixel_values = network.add_input(
        "pixel_values", trt.float32, (1, 3, image_size, image_size))

    pe_w = weights["encoder.patch_embed.weight"]
    pe_b = weights["encoder.patch_embed.bias"]
    patch_conv = network.add_convolution_nd(
        pixel_values, num_output_maps=hidden,
        kernel_shape=(patch_size, patch_size),
        kernel=trt.Weights(np.ascontiguousarray(pe_w)),
        bias=trt.Weights(np.ascontiguousarray(pe_b)))
    patch_conv.stride_nd = (patch_size, patch_size)

    to_nhwc = network.add_shuffle(patch_conv.get_output(0))
    to_nhwc.first_transpose = trt.Permutation([0, 2, 3, 1])

    pos_embed = weights["encoder.pos_embed"]
    pos_c = graph_ops.add_constant(
        network, (1, grid_size, grid_size, hidden), pos_embed)
    pos_sum = network.add_elementwise(
        to_nhwc.get_output(0), pos_c, trt.ElementWiseOperation.SUM)
    hidden_state = pos_sum.get_output(0)

    for layer_idx in range(num_layers):
        w_prefix = f"encoder.layer{layer_idx}"
        use_global_attn = layer_idx in global_attn_indexes

        norm1_w = weights[f"{w_prefix}.norm1.weight"]
        norm1_b = weights[f"{w_prefix}.norm1.bias"]

        reshape_2d = network.add_shuffle(hidden_state)
        reshape_2d.reshape_dims = (seq_len, hidden)

        normed = graph_ops.add_layer_norm(
            network, reshape_2d.get_output(0), hidden,
            norm1_w, norm1_b, eps_t)

        normed_4d = network.add_shuffle(normed)
        normed_4d.reshape_dims = (1, grid_size, grid_size, hidden)

        if use_global_attn:
            attn_out_4d = SamPlugin._build_global_attention(
                network, normed_4d.get_output(0), weights, w_prefix,
                grid_size, hidden, num_heads, head_dim, seq_len)
        else:
            attn_out_4d = SamPlugin._build_windowed_attention(
                network, normed_4d.get_output(0), weights, w_prefix,
                grid_size, hidden, num_heads, head_dim, window_size)

        res1 = network.add_elementwise(
            hidden_state, attn_out_4d, trt.ElementWiseOperation.SUM)

        norm2_w = weights[f"{w_prefix}.norm2.weight"]
        norm2_b = weights[f"{w_prefix}.norm2.bias"]

        res1_2d = network.add_shuffle(res1.get_output(0))
        res1_2d.reshape_dims = (seq_len, hidden)

        normed2 = graph_ops.add_layer_norm(
            network, res1_2d.get_output(0), hidden,
            norm2_w, norm2_b, eps_t)

        fc1_w = _slice_mlp_columns(weights[f"{w_prefix}.mlp.fc1.weight"], mlp_dim, parallel)
        fc1_b = _slice_mlp_columns(weights[f"{w_prefix}.mlp.fc1.bias"], mlp_dim, parallel)
        fc1 = graph_ops.add_matmul_rhs_constant(
            network, normed2, hidden, local_mlp_dim, fc1_w)
        fc1 = graph_ops.add_bias_sum(network, fc1, local_mlp_dim, fc1_b)
        gelu = graph_ops.add_gelu_new(network, fc1)

        fc2_w = _slice_mlp_rows(weights[f"{w_prefix}.mlp.fc2.weight"], mlp_dim, parallel)
        fc2 = graph_ops.add_matmul_rhs_constant(
            network, gelu, local_mlp_dim, hidden, fc2_w)
        fc2 = add_all_reduce_sum(network, fc2, parallel.tp_size)
        fc2 = graph_ops.add_bias_sum(
            network, fc2, hidden, weights[f"{w_prefix}.mlp.fc2.bias"])

        fc2_4d = network.add_shuffle(fc2)
        fc2_4d.reshape_dims = (1, grid_size, grid_size, hidden)

        res2 = network.add_elementwise(
            res1.get_output(0), fc2_4d.get_output(0), trt.ElementWiseOperation.SUM)
        hidden_state = res2.get_output(0)

    to_nchw = network.add_shuffle(hidden_state)
    to_nchw.first_transpose = trt.Permutation([0, 3, 1, 2])

    neck_c1_w = weights["encoder.neck.conv1.weight"]
    neck_c1_b = weights.get("encoder.neck.conv1.bias",
                            np.zeros(decoder_hidden, dtype=np.float32))
    neck_conv1 = network.add_convolution_nd(
        to_nchw.get_output(0), num_output_maps=decoder_hidden,
        kernel_shape=(1, 1),
        kernel=trt.Weights(np.ascontiguousarray(neck_c1_w)),
        bias=trt.Weights(np.ascontiguousarray(neck_c1_b)))

    to_nhwc_n1 = network.add_shuffle(neck_conv1.get_output(0))
    to_nhwc_n1.first_transpose = trt.Permutation([0, 2, 3, 1])
    flat_n1 = network.add_shuffle(to_nhwc_n1.get_output(0))
    flat_n1.reshape_dims = (seq_len, decoder_hidden)
    ln1_out = graph_ops.add_layer_norm(
        network, flat_n1.get_output(0), decoder_hidden,
        weights["encoder.neck.ln1.weight"], weights["encoder.neck.ln1.bias"], eps_t)
    unflat_n1 = network.add_shuffle(ln1_out)
    unflat_n1.reshape_dims = (1, grid_size, grid_size, decoder_hidden)
    to_nchw_n1 = network.add_shuffle(unflat_n1.get_output(0))
    to_nchw_n1.first_transpose = trt.Permutation([0, 3, 1, 2])

    neck_c2_w = weights["encoder.neck.conv2.weight"]
    neck_c2_b = weights.get("encoder.neck.conv2.bias",
                            np.zeros(decoder_hidden, dtype=np.float32))
    neck_conv2 = network.add_convolution_nd(
        to_nchw_n1.get_output(0), num_output_maps=decoder_hidden,
        kernel_shape=(3, 3),
        kernel=trt.Weights(np.ascontiguousarray(neck_c2_w)),
        bias=trt.Weights(np.ascontiguousarray(neck_c2_b)))
    neck_conv2.padding_nd = (1, 1)

    to_nhwc_n2 = network.add_shuffle(neck_conv2.get_output(0))
    to_nhwc_n2.first_transpose = trt.Permutation([0, 2, 3, 1])
    flat_n2 = network.add_shuffle(to_nhwc_n2.get_output(0))
    flat_n2.reshape_dims = (seq_len, decoder_hidden)
    ln2_out = graph_ops.add_layer_norm(
        network, flat_n2.get_output(0), decoder_hidden,
        weights["encoder.neck.ln2.weight"], weights["encoder.neck.ln2.bias"], eps_t)
    unflat_n2 = network.add_shuffle(ln2_out)
    unflat_n2.reshape_dims = (1, grid_size, grid_size, decoder_hidden)
    to_nchw_n2 = network.add_shuffle(unflat_n2.get_output(0))
    to_nchw_n2.first_transpose = trt.Permutation([0, 3, 1, 2])

    output = to_nchw_n2.get_output(0)
    output.name = "image_embeddings"
    network.mark_output(output)

    if verbose:
        print(
            f"[trtmc build] Building SAM encoder TP rank {parallel.rank}/{parallel.tp_size} "
            f"(image={image_size}x{image_size}, hidden={hidden}, layers={num_layers}) ...",
            file=sys.stderr,
        )

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed for SAM tensor-parallel encoder")
    return bytes(plan)
