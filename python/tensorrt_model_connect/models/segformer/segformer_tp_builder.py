# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel SegFormer builder.

SegFormer-B0 stage head counts are not divisible by TP=4 or TP=2, so the
attention blocks remain replicated. The Mix-FFN path is tensor-parallel:
FC1 and depthwise-conv channels are column/channel sharded, FC2 is row
sharded, and a TensorRT distributed ALL_REDUCE restores the full residual.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import sys

import numpy as np
from tensorrt_model_connect import trt_compat

from . import graph_ops
from ...parallel_config import add_all_reduce_sum, normalize_parallel_config

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict
    from .config import ModelConfig
    from ...parallel_config import ParallelConfig


def _validate_segformer_tp(config: "ModelConfig", parallel: "ParallelConfig") -> None:
    parallel.validate()
    if not parallel.enabled:
        return
    if parallel.rank < 0:
        raise ValueError("SegFormer tensor-parallel build requires a concrete rank")
    raw = config.raw
    hidden_sizes = raw.get("hidden_sizes", [32, 64, 160, 256])
    mlp_ratios = raw.get("mlp_ratios", [4, 4, 4, 4])
    for stage_idx, hidden in enumerate(hidden_sizes):
        ffn_hidden = int(hidden) * int(mlp_ratios[stage_idx])
        if ffn_hidden % parallel.tp_size != 0:
            raise ValueError(
                "SegFormer tensor-parallel Mix-FFN requires each FFN width "
                f"divisible by tp_size; stage {stage_idx} has {ffn_hidden} "
                f"vs tp_size={parallel.tp_size}")


def _slice_mlp_columns(arr: np.ndarray, ffn_hidden: int, parallel: "ParallelConfig") -> np.ndarray:
    local = ffn_hidden // parallel.tp_size
    start = parallel.rank * local
    end = start + local
    return np.ascontiguousarray(arr[..., start:end])


def _slice_mlp_rows(arr: np.ndarray, ffn_hidden: int, parallel: "ParallelConfig") -> np.ndarray:
    local = ffn_hidden // parallel.tp_size
    start = parallel.rank * local
    end = start + local
    return np.ascontiguousarray(arr[start:end, ...])


def _mark_debug_output(network, tensor, name: str, enabled: bool) -> None:
    if not enabled:
        return
    identity = network.add_identity(tensor)
    cast = network.add_cast(identity.get_output(0), trt.float32)
    out = cast.get_output(0)
    out.name = name
    network.mark_output(out)


def build_segformer_tp_engine(
    config: "ModelConfig",
    weights: "WeightDict",
    max_cache_length: int,
    *,
    precision: str = "fp32",
    quant_ctx=None,
    verbose: bool = False,
    debug_layer_outputs: bool = False,
    parallel_config=None,
) -> bytes:
    """Build one rank-local SegFormer engine with tensor-parallel Mix-FFNs."""
    del max_cache_length, precision
    if quant_ctx is not None:
        raise ValueError("SegFormer tensor-parallel builds do not support quantization")
    parallel = normalize_parallel_config(parallel_config)
    if not parallel.enabled:
        raise ValueError("SegFormer tensor-parallel builder requires an enabled parallel config")
    _validate_segformer_tp(config, parallel)

    raw = config.raw
    num_encoder_blocks = raw.get("depths", [2, 2, 2, 2])
    sr_ratios = raw.get("sr_ratios", [8, 4, 2, 1])
    hidden_sizes = raw.get("hidden_sizes", [32, 64, 160, 256])
    num_attention_heads = raw.get("num_attention_heads", [1, 2, 5, 8])
    mlp_ratios = raw.get("mlp_ratios", [4, 4, 4, 4])
    patch_sizes = raw.get("patch_sizes", [7, 3, 3, 3])
    strides = raw.get("strides", [4, 2, 2, 2])
    num_classes = raw.get("num_labels", 150)
    decoder_hidden_size = raw.get("decoder_hidden_size", 256)
    layer_norm_eps, hidden_act = graph_ops.resolve_numerical_contract(config)

    image_size = raw.get("_resolved_image_size", 512)
    H_in, W_in = image_size, image_size

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    pixel_values = network.add_input("pixel_values", trt.float32, (1, 3, H_in, W_in))

    cur_H, cur_W = H_in, W_in
    stage_outputs = []
    x = pixel_values

    for stage_idx in range(4):
        n_blocks = num_encoder_blocks[stage_idx]
        hidden = hidden_sizes[stage_idx]
        n_heads = num_attention_heads[stage_idx]
        sr = sr_ratios[stage_idx]
        mlp_ratio = mlp_ratios[stage_idx]
        ffn_hidden = hidden * mlp_ratio
        local_ffn_hidden = ffn_hidden // parallel.tp_size
        patch_size = patch_sizes[stage_idx]
        stride = strides[stage_idx]
        padding = patch_size // 2

        pe_w = weights[f"stage{stage_idx}.patch_embed.proj.weight"]
        pe_b = weights[f"stage{stage_idx}.patch_embed.proj.bias"]

        conv = network.add_convolution_nd(
            x, num_output_maps=hidden,
            kernel_shape=(patch_size, patch_size),
            kernel=trt.Weights(np.ascontiguousarray(pe_w)),
            bias=trt.Weights(np.ascontiguousarray(pe_b)))
        conv.stride_nd = (stride, stride)
        conv.padding_nd = (padding, padding)

        cur_H = (cur_H + 2 * padding - patch_size) // stride + 1
        cur_W = (cur_W + 2 * padding - patch_size) // stride + 1
        seq_len = cur_H * cur_W

        reshape_to_seq = network.add_shuffle(conv.get_output(0))
        reshape_to_seq.first_transpose = trt.Permutation([0, 2, 3, 1])
        reshape_to_seq.reshape_dims = (seq_len, hidden)

        pe_ln_w = weights[f"stage{stage_idx}.patch_embed.norm.weight"]
        pe_ln_b = weights[f"stage{stage_idx}.patch_embed.norm.bias"]
        eps_t = graph_ops.add_constant(
            network, (1, 1), np.array([layer_norm_eps], dtype=np.float32))
        hidden_state = graph_ops.add_layer_norm(
            network, reshape_to_seq.get_output(0), hidden, pe_ln_w, pe_ln_b, eps_t)

        if debug_layer_outputs:
            pe_dbg = network.add_shuffle(hidden_state)
            pe_dbg.reshape_dims = (1, cur_H, cur_W, hidden)
            pe_dbg_t = network.add_shuffle(pe_dbg.get_output(0))
            pe_dbg_t.first_transpose = trt.Permutation([0, 3, 1, 2])
            _mark_debug_output(
                network, pe_dbg_t.get_output(0), f"debug_stage{stage_idx}_patch_embed",
                debug_layer_outputs)

        for block_idx in range(n_blocks):
            w_prefix = f"stage{stage_idx}.block{block_idx}"

            norm1_w = weights[f"{w_prefix}.norm1.weight"]
            norm1_b = weights[f"{w_prefix}.norm1.bias"]
            normed = graph_ops.add_layer_norm(
                network, hidden_state, hidden, norm1_w, norm1_b, eps_t)

            if sr > 1:
                reshape_4d = network.add_shuffle(normed)
                reshape_4d.reshape_dims = (1, cur_H, cur_W, hidden)
                reshape_4d_t = network.add_shuffle(reshape_4d.get_output(0))
                reshape_4d_t.first_transpose = trt.Permutation([0, 3, 1, 2])

                sr_w = weights[f"{w_prefix}.attn.sr.weight"]
                sr_b = weights[f"{w_prefix}.attn.sr.bias"]
                sr_conv = network.add_convolution_nd(
                    reshape_4d_t.get_output(0),
                    num_output_maps=hidden,
                    kernel_shape=(sr, sr),
                    kernel=trt.Weights(np.ascontiguousarray(sr_w)),
                    bias=trt.Weights(np.ascontiguousarray(sr_b)))
                sr_conv.stride_nd = (sr, sr)

                sr_H = cur_H // sr
                sr_W = cur_W // sr
                sr_seq = sr_H * sr_W

                sr_reshape = network.add_shuffle(sr_conv.get_output(0))
                sr_reshape.first_transpose = trt.Permutation([0, 2, 3, 1])
                sr_reshape.reshape_dims = (sr_seq, hidden)

                sr_ln_w = weights[f"{w_prefix}.attn.sr_norm.weight"]
                sr_ln_b = weights[f"{w_prefix}.attn.sr_norm.bias"]
                kv_input = graph_ops.add_layer_norm(
                    network, sr_reshape.get_output(0), hidden, sr_ln_w, sr_ln_b, eps_t)
                kv_seq_len = sr_seq
            else:
                kv_input = normed
                kv_seq_len = seq_len

            head_dim = hidden // n_heads
            attn_scale = 1.0 / np.sqrt(max(head_dim, 1))

            q = graph_ops.add_matmul_rhs_constant(
                network, normed, hidden, hidden, weights[f"{w_prefix}.attn.q.weight"])
            q = graph_ops.add_bias_sum(network, q, hidden, weights[f"{w_prefix}.attn.q.bias"])

            k = graph_ops.add_matmul_rhs_constant(
                network, kv_input, hidden, hidden, weights[f"{w_prefix}.attn.k.weight"])
            k = graph_ops.add_bias_sum(network, k, hidden, weights[f"{w_prefix}.attn.k.bias"])
            v = graph_ops.add_matmul_rhs_constant(
                network, kv_input, hidden, hidden, weights[f"{w_prefix}.attn.v.weight"])
            v = graph_ops.add_bias_sum(network, v, hidden, weights[f"{w_prefix}.attn.v.bias"])

            ctx_flat = graph_ops.add_attention_from_rows(
                network, q, k, v,
                num_heads=n_heads, head_dim=head_dim,
                q_seq=seq_len, kv_seq=kv_seq_len,
                scale=attn_scale)

            attn_out = graph_ops.add_matmul_rhs_constant(
                network, ctx_flat, hidden, hidden, weights[f"{w_prefix}.attn.o.weight"])
            attn_out = graph_ops.add_bias_sum(
                network, attn_out, hidden, weights[f"{w_prefix}.attn.o.bias"])

            res1 = network.add_elementwise(
                hidden_state, attn_out, trt.ElementWiseOperation.SUM)
            hidden_state = res1.get_output(0)

            norm2_w = weights[f"{w_prefix}.norm2.weight"]
            norm2_b = weights[f"{w_prefix}.norm2.bias"]
            normed2 = graph_ops.add_layer_norm(
                network, hidden_state, hidden, norm2_w, norm2_b, eps_t)

            fc1_w = _slice_mlp_columns(
                weights[f"{w_prefix}.mlp.fc1.weight"], ffn_hidden, parallel)
            fc1_b = _slice_mlp_columns(
                weights[f"{w_prefix}.mlp.fc1.bias"], ffn_hidden, parallel)
            fc1 = graph_ops.add_matmul_rhs_constant(
                network, normed2, hidden, local_ffn_hidden, fc1_w)
            fc1 = graph_ops.add_bias_sum(network, fc1, local_ffn_hidden, fc1_b)

            dw_reshape = network.add_shuffle(fc1)
            dw_reshape.reshape_dims = (1, cur_H, cur_W, local_ffn_hidden)
            dw_t = network.add_shuffle(dw_reshape.get_output(0))
            dw_t.first_transpose = trt.Permutation([0, 3, 1, 2])

            dw_w = _slice_mlp_rows(
                weights[f"{w_prefix}.mlp.dwconv.weight"], ffn_hidden, parallel)
            dw_b = _slice_mlp_rows(
                weights[f"{w_prefix}.mlp.dwconv.bias"], ffn_hidden, parallel)
            dwconv = network.add_convolution_nd(
                dw_t.get_output(0),
                num_output_maps=local_ffn_hidden,
                kernel_shape=(3, 3),
                kernel=trt.Weights(np.ascontiguousarray(dw_w)),
                bias=trt.Weights(np.ascontiguousarray(dw_b)))
            dwconv.stride_nd = (1, 1)
            dwconv.padding_nd = (1, 1)
            dwconv.num_groups = local_ffn_hidden

            dw_back = network.add_shuffle(dwconv.get_output(0))
            dw_back.first_transpose = trt.Permutation([0, 2, 3, 1])
            dw_back.reshape_dims = (seq_len, local_ffn_hidden)

            gelu_out = graph_ops.add_activation(
                network, dw_back.get_output(0), hidden_act)

            fc2_w = _slice_mlp_rows(
                weights[f"{w_prefix}.mlp.fc2.weight"], ffn_hidden, parallel)
            fc2 = graph_ops.add_matmul_rhs_constant(
                network, gelu_out, local_ffn_hidden, hidden, fc2_w)
            fc2 = add_all_reduce_sum(network, fc2, parallel.tp_size)
            fc2 = graph_ops.add_bias_sum(
                network, fc2, hidden, weights[f"{w_prefix}.mlp.fc2.bias"])

            res2 = network.add_elementwise(
                hidden_state, fc2, trt.ElementWiseOperation.SUM)
            hidden_state = res2.get_output(0)

            if debug_layer_outputs:
                blk_dbg = network.add_shuffle(hidden_state)
                blk_dbg.reshape_dims = (1, cur_H, cur_W, hidden)
                blk_dbg_t = network.add_shuffle(blk_dbg.get_output(0))
                blk_dbg_t.first_transpose = trt.Permutation([0, 3, 1, 2])
                _mark_debug_output(
                    network, blk_dbg_t.get_output(0),
                    f"debug_stage{stage_idx}_block{block_idx}", debug_layer_outputs)

        final_ln_w = weights[f"stage{stage_idx}.final_norm.weight"]
        final_ln_b = weights[f"stage{stage_idx}.final_norm.bias"]
        hidden_state = graph_ops.add_layer_norm(
            network, hidden_state, hidden, final_ln_w, final_ln_b, eps_t)

        to_4d = network.add_shuffle(hidden_state)
        to_4d.reshape_dims = (1, cur_H, cur_W, hidden)
        to_4d_t = network.add_shuffle(to_4d.get_output(0))
        to_4d_t.first_transpose = trt.Permutation([0, 3, 1, 2])

        stage_outputs.append((to_4d_t.get_output(0), cur_H, cur_W, hidden))
        _mark_debug_output(
            network, to_4d_t.get_output(0), f"debug_stage{stage_idx}",
            debug_layer_outputs)
        x = to_4d_t.get_output(0)

    target_H = H_in // 4
    target_W = W_in // 4

    projected = []
    for i, (feat, feat_H, feat_W, feat_hidden) in enumerate(stage_outputs):
        to_2d = network.add_shuffle(feat)
        to_2d.first_transpose = trt.Permutation([0, 2, 3, 1])
        to_2d.reshape_dims = (feat_H * feat_W, feat_hidden)

        proj = graph_ops.add_matmul_rhs_constant(
            network, to_2d.get_output(0), feat_hidden, decoder_hidden_size,
            weights[f"decode_head.linear_c{i}.weight"])
        proj = graph_ops.add_bias_sum(
            network, proj, decoder_hidden_size, weights[f"decode_head.linear_c{i}.bias"])

        to_4d2 = network.add_shuffle(proj)
        to_4d2.reshape_dims = (1, feat_H, feat_W, decoder_hidden_size)
        to_4d2_t = network.add_shuffle(to_4d2.get_output(0))
        to_4d2_t.first_transpose = trt.Permutation([0, 3, 1, 2])

        if feat_H != target_H or feat_W != target_W:
            resize = network.add_resize(to_4d2_t.get_output(0))
            resize.resize_mode = trt.InterpolationMode.LINEAR
            resize.coordinate_transformation = trt.ResizeCoordinateTransformation.HALF_PIXEL
            resize.shape = (1, decoder_hidden_size, target_H, target_W)
            projected.append(resize.get_output(0))
        else:
            projected.append(to_4d2_t.get_output(0))

    concat = network.add_concatenation(projected[::-1])
    concat.axis = 1

    fuse_w = weights["decode_head.fuse.weight"]
    fuse_b = weights["decode_head.fuse.bias"]
    fuse_conv = network.add_convolution_nd(
        concat.get_output(0),
        num_output_maps=decoder_hidden_size,
        kernel_shape=(1, 1),
        kernel=trt.Weights(np.ascontiguousarray(fuse_w)),
        bias=trt.Weights(np.ascontiguousarray(fuse_b)))

    bn_w = weights["decode_head.bn.weight"]
    bn_b = weights["decode_head.bn.bias"]
    bn_mean = weights["decode_head.bn.running_mean"]
    bn_var = weights["decode_head.bn.running_var"]
    bn_scale = bn_w / np.sqrt(bn_var + 1e-5)
    bn_shift = bn_b - bn_mean * bn_scale

    bn_scale_t = graph_ops.add_constant(
        network, (1, decoder_hidden_size, 1, 1), bn_scale.reshape(1, -1, 1, 1))
    bn_shift_t = graph_ops.add_constant(
        network, (1, decoder_hidden_size, 1, 1), bn_shift.reshape(1, -1, 1, 1))

    bn_scaled = network.add_elementwise(
        fuse_conv.get_output(0), bn_scale_t, trt.ElementWiseOperation.PROD)
    bn_out = network.add_elementwise(
        bn_scaled.get_output(0), bn_shift_t, trt.ElementWiseOperation.SUM)

    relu = network.add_activation(bn_out.get_output(0), trt.ActivationType.RELU)

    cls_w = weights["decode_head.classifier.weight"]
    cls_b = weights["decode_head.classifier.bias"]
    cls_conv = network.add_convolution_nd(
        relu.get_output(0),
        num_output_maps=num_classes,
        kernel_shape=(1, 1),
        kernel=trt.Weights(np.ascontiguousarray(cls_w)),
        bias=trt.Weights(np.ascontiguousarray(cls_b)))

    logits = cls_conv.get_output(0)
    logits.name = "logits"
    network.mark_output(logits)

    if verbose:
        print(
            f"[trtmc build] Building SegFormer TP rank {parallel.rank}/{parallel.tp_size} "
            f"(image={H_in}x{W_in}, classes={num_classes}) ...",
            file=sys.stderr,
        )

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed for SegFormer tensor-parallel rank")
    return bytes(plan)
