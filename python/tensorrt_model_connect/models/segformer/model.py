# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SegFormer family model -- semantic segmentation (SegFormer-B0..B5).

SegFormer is an encoder-decoder architecture for semantic segmentation:
  - Hierarchical Transformer encoder with 4 stages
  - Lightweight All-MLP decode head
  - No positional encoding (uses overlapping patch embeddings)

Architecture per stage (encoder):
  1. Overlap Patch Embed: Conv2d with overlapping patches -> LayerNorm
  2. N transformer blocks:
     a. Efficient Self-Attention with Sequence Reduction (SR)
     b. Mix-FFN: FC1 -> DWConv3x3 -> GELU -> FC2

Decode head:
  1. Per-stage: Linear projection to decode_dim
  2. Bilinear upsample each stage to H/4 x W/4
  3. Concatenate all stages
  4. Conv2d fuse (1x1) -> BN -> ReLU -> Conv2d classifier

Weight key mapping (HF -> engine):
  HF: segformer.encoder.patch_embeddings.{i}.proj.weight/bias
  HF: segformer.encoder.patch_embeddings.{i}.layer_norm.weight/bias
  HF: segformer.encoder.block.{i}.{j}.attention.self.query/key/value/output.dense.weight/bias
  HF: segformer.encoder.block.{i}.{j}.attention.self.sr.weight/bias
  HF: segformer.encoder.block.{i}.{j}.attention.self.layer_norm.weight/bias
  HF: segformer.encoder.block.{i}.{j}.layer_norm_1/2.weight/bias
  HF: segformer.encoder.block.{i}.{j}.mlp.dense1/2.weight/bias
  HF: segformer.encoder.block.{i}.{j}.mlp.dwconv.dwconv.weight/bias
  HF: decode_head.linear_c.{i}.proj.weight/bias
  HF: decode_head.linear_fuse.weight/bias
  HF: decode_head.batch_norm.weight/bias/running_mean/running_var
  HF: decode_head.classifier.weight/bias
"""

from __future__ import annotations

import json
import re
import time

import sys
from pathlib import Path

import numpy as np
from tensorrt_model_connect import trt_compat

from .config import ModelConfig
from .checkpoint_mapper import (
    WeightDict,
    _open_safetensors,
    _load_tensor,
    _has_tensor,
    _transpose_2d,
)
from . import graph_ops
from ...parallel_config import (
    normalize_parallel_config,
    require_tensorrt_11_for_tensor_parallel,
)
from .segformer_tp_builder import build_segformer_tp_engine


trt = trt_compat.get_trt()


def _resolve_image_size(model_dir: str) -> int:
    """Read preprocessor_config.json for the actual image size."""
    pp_path = Path(model_dir) / "preprocessor_config.json"
    if pp_path.exists():
        pp = json.loads(pp_path.read_text())
        # SegFormerImageProcessor stores size as {"height": H, "width": W}
        size = pp.get("size", {})
        if isinstance(size, dict):
            h = size.get("height", 512)
            w = size.get("width", 512)
            return max(h, w)
        if isinstance(size, int):
            return size
    return 512


name = "segformer"
runtime_strategy = "segformer_segmentation"
requires_tokenizer = False


def matches(config) -> bool:
    """Return whether this module owns the resolved model config."""
    model_type = getattr(config, "model_type", config)
    model_type = str(model_type)
    return model_type.lower() in ("segformer",)


def load_weights(model_dir: str, config: ModelConfig) -> WeightDict:
    """Load SegFormer weights from safetensors."""
    model_dir_path = Path(model_dir)
    readers = _open_safetensors(model_dir_path)

    raw = config.raw
    num_encoder_blocks = raw.get("depths", [2, 2, 2, 2])
    sr_ratios = raw.get("sr_ratios", [8, 4, 2, 1])

    # Resolve actual image size from preprocessor_config.json
    image_size = _resolve_image_size(model_dir)
    config.raw["_resolved_image_size"] = image_size

    weights = WeightDict()

    # 4 encoder stages
    for stage_idx in range(4):
        n_blocks = num_encoder_blocks[stage_idx]
        sr = sr_ratios[stage_idx]

        # Overlap patch embedding
        pe_prefix = f"segformer.encoder.patch_embeddings.{stage_idx}"
        weights[f"stage{stage_idx}.patch_embed.proj.weight"] = _load_tensor(
            readers, f"{pe_prefix}.proj.weight"
        ).astype(np.float32)
        weights[f"stage{stage_idx}.patch_embed.proj.bias"] = _load_tensor(
            readers, f"{pe_prefix}.proj.bias"
        ).astype(np.float32)
        weights[f"stage{stage_idx}.patch_embed.norm.weight"] = _load_tensor(
            readers, f"{pe_prefix}.layer_norm.weight"
        ).astype(np.float32)
        weights[f"stage{stage_idx}.patch_embed.norm.bias"] = _load_tensor(
            readers, f"{pe_prefix}.layer_norm.bias"
        ).astype(np.float32)

        for block_idx in range(n_blocks):
            blk_prefix = f"segformer.encoder.block.{stage_idx}.{block_idx}"
            w_prefix = f"stage{stage_idx}.block{block_idx}"

            # Layer norms
            weights[f"{w_prefix}.norm1.weight"] = _load_tensor(
                readers, f"{blk_prefix}.layer_norm_1.weight"
            ).astype(np.float32)
            weights[f"{w_prefix}.norm1.bias"] = _load_tensor(
                readers, f"{blk_prefix}.layer_norm_1.bias"
            ).astype(np.float32)
            weights[f"{w_prefix}.norm2.weight"] = _load_tensor(
                readers, f"{blk_prefix}.layer_norm_2.weight"
            ).astype(np.float32)
            weights[f"{w_prefix}.norm2.bias"] = _load_tensor(
                readers, f"{blk_prefix}.layer_norm_2.bias"
            ).astype(np.float32)

            # Attention Q/K/V/O
            for proj in ("query", "key", "value"):
                w = _load_tensor(readers, f"{blk_prefix}.attention.{proj}.weight")
                b = _load_tensor(readers, f"{blk_prefix}.attention.{proj}.bias")
                weights[f"{w_prefix}.attn.{proj[0]}.weight"] = _transpose_2d(w, f"attn_{proj}")
                weights[f"{w_prefix}.attn.{proj[0]}.bias"] = b.astype(np.float32)

            w_o = _load_tensor(readers, f"{blk_prefix}.attention.output.dense.weight")
            b_o = _load_tensor(readers, f"{blk_prefix}.attention.output.dense.bias")
            weights[f"{w_prefix}.attn.o.weight"] = _transpose_2d(w_o, "attn_o")
            weights[f"{w_prefix}.attn.o.bias"] = b_o.astype(np.float32)

            # SR (sequence reduction) if sr_ratio > 1
            if sr > 1:
                sr_w = _load_tensor(readers, f"{blk_prefix}.attention.sr.weight")
                sr_b = _load_tensor(readers, f"{blk_prefix}.attention.sr.bias")
                weights[f"{w_prefix}.attn.sr.weight"] = sr_w.astype(np.float32)
                weights[f"{w_prefix}.attn.sr.bias"] = sr_b.astype(np.float32)

                sr_ln_w = _load_tensor(readers, f"{blk_prefix}.attention.layer_norm.weight")
                sr_ln_b = _load_tensor(readers, f"{blk_prefix}.attention.layer_norm.bias")
                weights[f"{w_prefix}.attn.sr_norm.weight"] = sr_ln_w.astype(np.float32)
                weights[f"{w_prefix}.attn.sr_norm.bias"] = sr_ln_b.astype(np.float32)

            # Mix-FFN
            w_fc1 = _load_tensor(readers, f"{blk_prefix}.mlp.dense1.weight")
            b_fc1 = _load_tensor(readers, f"{blk_prefix}.mlp.dense1.bias")
            weights[f"{w_prefix}.mlp.fc1.weight"] = _transpose_2d(w_fc1, "mlp_fc1")
            weights[f"{w_prefix}.mlp.fc1.bias"] = b_fc1.astype(np.float32)

            w_fc2 = _load_tensor(readers, f"{blk_prefix}.mlp.dense2.weight")
            b_fc2 = _load_tensor(readers, f"{blk_prefix}.mlp.dense2.bias")
            weights[f"{w_prefix}.mlp.fc2.weight"] = _transpose_2d(w_fc2, "mlp_fc2")
            weights[f"{w_prefix}.mlp.fc2.bias"] = b_fc2.astype(np.float32)

            # DWConv in Mix-FFN
            dw_w = _load_tensor(readers, f"{blk_prefix}.mlp.dwconv.dwconv.weight")
            dw_b = _load_tensor(readers, f"{blk_prefix}.mlp.dwconv.dwconv.bias")
            weights[f"{w_prefix}.mlp.dwconv.weight"] = dw_w.astype(np.float32)
            weights[f"{w_prefix}.mlp.dwconv.bias"] = dw_b.astype(np.float32)

        # Per-stage final LayerNorm
        ln_prefix = f"segformer.encoder.layer_norm.{stage_idx}"
        weights[f"stage{stage_idx}.final_norm.weight"] = _load_tensor(
            readers, f"{ln_prefix}.weight"
        ).astype(np.float32)
        weights[f"stage{stage_idx}.final_norm.bias"] = _load_tensor(
            readers, f"{ln_prefix}.bias"
        ).astype(np.float32)

    # Decode head
    for i in range(4):
        w_proj = _load_tensor(readers, f"decode_head.linear_c.{i}.proj.weight")
        b_proj = _load_tensor(readers, f"decode_head.linear_c.{i}.proj.bias")
        weights[f"decode_head.linear_c{i}.weight"] = _transpose_2d(w_proj, f"dec_proj_{i}")
        weights[f"decode_head.linear_c{i}.bias"] = b_proj.astype(np.float32)

    # Fuse conv (1x1)
    weights["decode_head.fuse.weight"] = _load_tensor(
        readers, "decode_head.linear_fuse.weight"
    ).astype(np.float32)
    if _has_tensor(readers, "decode_head.linear_fuse.bias"):
        weights["decode_head.fuse.bias"] = _load_tensor(
            readers, "decode_head.linear_fuse.bias"
        ).astype(np.float32)
    else:
        out_ch = weights["decode_head.fuse.weight"].shape[0]
        weights["decode_head.fuse.bias"] = np.zeros(out_ch, dtype=np.float32)

    # BatchNorm
    weights["decode_head.bn.weight"] = _load_tensor(
        readers, "decode_head.batch_norm.weight"
    ).astype(np.float32)
    weights["decode_head.bn.bias"] = _load_tensor(readers, "decode_head.batch_norm.bias").astype(
        np.float32
    )
    weights["decode_head.bn.running_mean"] = _load_tensor(
        readers, "decode_head.batch_norm.running_mean"
    ).astype(np.float32)
    weights["decode_head.bn.running_var"] = _load_tensor(
        readers, "decode_head.batch_norm.running_var"
    ).astype(np.float32)

    # Classifier conv
    weights["decode_head.classifier.weight"] = _load_tensor(
        readers, "decode_head.classifier.weight"
    ).astype(np.float32)
    weights["decode_head.classifier.bias"] = _load_tensor(
        readers, "decode_head.classifier.bias"
    ).astype(np.float32)

    return weights


def build_engine(
    config: ModelConfig,
    weights: WeightDict,
    max_cache_length: int,
    *,
    precision: str = "fp32",
    quant_ctx=None,
    verbose: bool = False,
    debug_layer_outputs: bool = False,
    parallel_config=None,
) -> bytes:
    """Build single TRT engine for SegFormer segmentation.

    Input:  pixel_values [1, 3, H, W]
    Output: logits [1, num_classes, H/4, W/4]
    """
    parallel = normalize_parallel_config(parallel_config)
    if parallel.enabled:
        require_tensorrt_11_for_tensor_parallel(
            parallel, feature="SegFormer tensor-parallel Mix-FFN builds"
        )
        if quant_ctx is not None:
            raise ValueError("SegFormer tensor-parallel builds do not support quantization")
        return build_segformer_tp_engine(
            config,
            weights,
            max_cache_length,
            precision=precision,
            quant_ctx=quant_ctx,
            verbose=verbose,
            debug_layer_outputs=debug_layer_outputs,
            parallel_config=parallel,
        )

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
    if precision == "fp16":
        work_np_dtype, work_trt_dtype = np.float16, trt.float16
    elif precision == "fp32":
        work_np_dtype, work_trt_dtype = np.float32, trt.float32
    else:
        raise ValueError(f"Unsupported SegFormer precision: {precision}")

    image_size = raw.get("_resolved_image_size", 512)
    H_in, W_in = image_size, image_size

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    def _mark_debug(tensor, name):
        """Mark a tensor as debug output (identity to avoid aliasing)."""
        if not debug_layer_outputs:
            return
        identity = network.add_identity(tensor)
        cast = network.add_cast(identity.get_output(0), trt.float32)
        out = cast.get_output(0)
        out.name = name
        network.mark_output(out)

    # Input: [1, 3, H, W]
    pixel_values = network.add_input("pixel_values", trt.float32, (1, 3, H_in, W_in))

    # Track spatial dims through stages
    cur_H, cur_W = H_in, W_in
    stage_outputs = []  # (tensor, H, W, hidden_size)

    # Current feature map tensor
    x = pixel_values
    if x.dtype != work_trt_dtype:
        x = network.add_cast(x, work_trt_dtype).get_output(0)

    for stage_idx in range(4):
        n_blocks = num_encoder_blocks[stage_idx]
        hidden = hidden_sizes[stage_idx]
        n_heads = num_attention_heads[stage_idx]
        sr = sr_ratios[stage_idx]
        mlp_ratio = mlp_ratios[stage_idx]
        ffn_hidden = hidden * mlp_ratio
        patch_size = patch_sizes[stage_idx]
        stride = strides[stage_idx]
        padding = patch_size // 2

        # --- Overlap Patch Embedding ---
        # Conv2d: [1, C_in, H, W] -> [1, hidden, H', W']
        pe_w = weights[f"stage{stage_idx}.patch_embed.proj.weight"]
        pe_b = weights[f"stage{stage_idx}.patch_embed.proj.bias"]

        conv = network.add_convolution_nd(
            x,
            num_output_maps=hidden,
            kernel_shape=(patch_size, patch_size),
            kernel=trt.Weights(np.ascontiguousarray(pe_w, dtype=work_np_dtype)),
            bias=trt.Weights(np.ascontiguousarray(pe_b, dtype=work_np_dtype)),
        )
        conv.stride_nd = (stride, stride)
        conv.padding_nd = (padding, padding)

        cur_H = (cur_H + 2 * padding - patch_size) // stride + 1
        cur_W = (cur_W + 2 * padding - patch_size) // stride + 1
        seq_len = cur_H * cur_W

        # Reshape [1, hidden, H', W'] -> [seq_len, hidden] for transformer
        reshape_to_seq = network.add_shuffle(conv.get_output(0))
        reshape_to_seq.first_transpose = trt.Permutation([0, 2, 3, 1])
        reshape_to_seq.reshape_dims = (seq_len, hidden)

        # LayerNorm after patch embed
        pe_ln_w = weights[f"stage{stage_idx}.patch_embed.norm.weight"]
        pe_ln_b = weights[f"stage{stage_idx}.patch_embed.norm.bias"]
        eps_t = graph_ops.add_constant(
            network, (1, 1), np.array([layer_norm_eps], dtype=work_np_dtype), dtype=work_np_dtype
        )
        hidden_state = graph_ops.add_layer_norm(
            network,
            reshape_to_seq.get_output(0),
            hidden,
            pe_ln_w,
            pe_ln_b,
            eps_t,
            dtype=work_np_dtype,
        )

        # Debug: patch embed output as NCHW [1, hidden, H', W']
        if debug_layer_outputs:
            pe_dbg = network.add_shuffle(hidden_state)
            pe_dbg.reshape_dims = (1, cur_H, cur_W, hidden)
            pe_dbg_t = network.add_shuffle(pe_dbg.get_output(0))
            pe_dbg_t.first_transpose = trt.Permutation([0, 3, 1, 2])
            _mark_debug(pe_dbg_t.get_output(0), f"debug_stage{stage_idx}_patch_embed")

        # --- Transformer blocks ---
        for block_idx in range(n_blocks):
            w_prefix = f"stage{stage_idx}.block{block_idx}"

            # -- Efficient Self-Attention --
            norm1_w = weights[f"{w_prefix}.norm1.weight"]
            norm1_b = weights[f"{w_prefix}.norm1.bias"]
            normed = graph_ops.add_layer_norm(
                network, hidden_state, hidden, norm1_w, norm1_b, eps_t, dtype=work_np_dtype
            )

            # SR: sequence reduction for K,V
            if sr > 1:
                # Reshape to [1, hidden, H', W'] for Conv2d SR
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
                    kernel=trt.Weights(np.ascontiguousarray(sr_w, dtype=work_np_dtype)),
                    bias=trt.Weights(np.ascontiguousarray(sr_b, dtype=work_np_dtype)),
                )
                sr_conv.stride_nd = (sr, sr)

                sr_H = cur_H // sr
                sr_W = cur_W // sr
                sr_seq = sr_H * sr_W

                # Reshape back to [sr_seq, hidden]
                sr_reshape = network.add_shuffle(sr_conv.get_output(0))
                sr_reshape.first_transpose = trt.Permutation([0, 2, 3, 1])
                sr_reshape.reshape_dims = (sr_seq, hidden)

                sr_ln_w = weights[f"{w_prefix}.attn.sr_norm.weight"]
                sr_ln_b = weights[f"{w_prefix}.attn.sr_norm.bias"]
                kv_input = graph_ops.add_layer_norm(
                    network,
                    sr_reshape.get_output(0),
                    hidden,
                    sr_ln_w,
                    sr_ln_b,
                    eps_t,
                    dtype=work_np_dtype,
                )
                kv_seq_len = sr_seq
            else:
                kv_input = normed
                kv_seq_len = seq_len

            head_dim = hidden // n_heads
            attn_scale = 1.0 / np.sqrt(max(head_dim, 1))

            # Q from normed [seq_len, hidden]
            q = graph_ops.add_matmul_rhs_constant(
                network,
                normed,
                hidden,
                hidden,
                weights[f"{w_prefix}.attn.q.weight"],
                dtype=work_np_dtype,
            )
            q = graph_ops.add_bias_sum(
                network, q, hidden, weights[f"{w_prefix}.attn.q.bias"], dtype=work_np_dtype
            )

            # K, V from kv_input [kv_seq_len, hidden]
            k = graph_ops.add_matmul_rhs_constant(
                network,
                kv_input,
                hidden,
                hidden,
                weights[f"{w_prefix}.attn.k.weight"],
                dtype=work_np_dtype,
            )
            k = graph_ops.add_bias_sum(
                network, k, hidden, weights[f"{w_prefix}.attn.k.bias"], dtype=work_np_dtype
            )
            v = graph_ops.add_matmul_rhs_constant(
                network,
                kv_input,
                hidden,
                hidden,
                weights[f"{w_prefix}.attn.v.weight"],
                dtype=work_np_dtype,
            )
            v = graph_ops.add_bias_sum(
                network, v, hidden, weights[f"{w_prefix}.attn.v.bias"], dtype=work_np_dtype
            )

            ctx_flat = graph_ops.add_attention_from_rows(
                network,
                q,
                k,
                v,
                num_heads=n_heads,
                head_dim=head_dim,
                q_seq=seq_len,
                kv_seq=kv_seq_len,
                scale=attn_scale,
            )

            # Output projection
            attn_out = graph_ops.add_matmul_rhs_constant(
                network,
                ctx_flat,
                hidden,
                hidden,
                weights[f"{w_prefix}.attn.o.weight"],
                dtype=work_np_dtype,
            )
            attn_out = graph_ops.add_bias_sum(
                network, attn_out, hidden, weights[f"{w_prefix}.attn.o.bias"], dtype=work_np_dtype
            )

            # Residual
            res1 = network.add_elementwise(hidden_state, attn_out, trt.ElementWiseOperation.SUM)
            hidden_state = res1.get_output(0)

            # -- Mix-FFN --
            norm2_w = weights[f"{w_prefix}.norm2.weight"]
            norm2_b = weights[f"{w_prefix}.norm2.bias"]
            normed2 = graph_ops.add_layer_norm(
                network, hidden_state, hidden, norm2_w, norm2_b, eps_t, dtype=work_np_dtype
            )

            # FC1: [seq, hidden] -> [seq, ffn_hidden]
            fc1 = graph_ops.add_matmul_rhs_constant(
                network,
                normed2,
                hidden,
                ffn_hidden,
                weights[f"{w_prefix}.mlp.fc1.weight"],
                dtype=work_np_dtype,
            )
            fc1 = graph_ops.add_bias_sum(
                network, fc1, ffn_hidden, weights[f"{w_prefix}.mlp.fc1.bias"], dtype=work_np_dtype
            )

            # DWConv3x3: reshape to 4D for depthwise conv
            dw_reshape = network.add_shuffle(fc1)
            dw_reshape.reshape_dims = (1, cur_H, cur_W, ffn_hidden)
            dw_t = network.add_shuffle(dw_reshape.get_output(0))
            dw_t.first_transpose = trt.Permutation([0, 3, 1, 2])

            dw_w = weights[f"{w_prefix}.mlp.dwconv.weight"]
            dw_b = weights[f"{w_prefix}.mlp.dwconv.bias"]
            dwconv = network.add_convolution_nd(
                dw_t.get_output(0),
                num_output_maps=ffn_hidden,
                kernel_shape=(3, 3),
                kernel=trt.Weights(np.ascontiguousarray(dw_w, dtype=work_np_dtype)),
                bias=trt.Weights(np.ascontiguousarray(dw_b, dtype=work_np_dtype)),
            )
            dwconv.stride_nd = (1, 1)
            dwconv.padding_nd = (1, 1)
            dwconv.num_groups = ffn_hidden  # depthwise

            # Reshape back to 2D BEFORE GELU (CRITICAL: GELU uses [1,1] constants)
            dw_back = network.add_shuffle(dwconv.get_output(0))
            dw_back.first_transpose = trt.Permutation([0, 2, 3, 1])
            dw_back.reshape_dims = (seq_len, ffn_hidden)

            # GELU activation
            gelu_out = graph_ops.add_activation(
                network, dw_back.get_output(0), hidden_act, dtype=work_np_dtype
            )

            # FC2: [seq, ffn_hidden] -> [seq, hidden]
            fc2 = graph_ops.add_matmul_rhs_constant(
                network,
                gelu_out,
                ffn_hidden,
                hidden,
                weights[f"{w_prefix}.mlp.fc2.weight"],
                dtype=work_np_dtype,
            )
            fc2 = graph_ops.add_bias_sum(
                network, fc2, hidden, weights[f"{w_prefix}.mlp.fc2.bias"], dtype=work_np_dtype
            )

            # Residual
            res2 = network.add_elementwise(hidden_state, fc2, trt.ElementWiseOperation.SUM)
            hidden_state = res2.get_output(0)

            # Debug: per-block output as NCHW [1, hidden, H', W']
            if debug_layer_outputs:
                blk_dbg = network.add_shuffle(hidden_state)
                blk_dbg.reshape_dims = (1, cur_H, cur_W, hidden)
                blk_dbg_t = network.add_shuffle(blk_dbg.get_output(0))
                blk_dbg_t.first_transpose = trt.Permutation([0, 3, 1, 2])
                _mark_debug(blk_dbg_t.get_output(0), f"debug_stage{stage_idx}_block{block_idx}")

        # Per-stage final LayerNorm (encoder.layer_norm[i])
        final_ln_w = weights[f"stage{stage_idx}.final_norm.weight"]
        final_ln_b = weights[f"stage{stage_idx}.final_norm.bias"]
        hidden_state = graph_ops.add_layer_norm(
            network, hidden_state, hidden, final_ln_w, final_ln_b, eps_t, dtype=work_np_dtype
        )

        # Reshape back to 4D: [seq_len, hidden] -> [1, hidden, H', W']
        to_4d = network.add_shuffle(hidden_state)
        to_4d.reshape_dims = (1, cur_H, cur_W, hidden)
        to_4d_t = network.add_shuffle(to_4d.get_output(0))
        to_4d_t.first_transpose = trt.Permutation([0, 3, 1, 2])

        stage_outputs.append((to_4d_t.get_output(0), cur_H, cur_W, hidden))
        _mark_debug(to_4d_t.get_output(0), f"debug_stage{stage_idx}")
        x = to_4d_t.get_output(0)

    # --- Decode Head ---
    target_H = H_in // 4
    target_W = W_in // 4

    projected = []
    for i, (feat, feat_H, feat_W, feat_hidden) in enumerate(stage_outputs):
        # Reshape to 2D: [1, C, H, W] -> [H*W, C]
        to_2d = network.add_shuffle(feat)
        to_2d.first_transpose = trt.Permutation([0, 2, 3, 1])
        to_2d.reshape_dims = (feat_H * feat_W, feat_hidden)

        # Linear projection
        proj = graph_ops.add_matmul_rhs_constant(
            network,
            to_2d.get_output(0),
            feat_hidden,
            decoder_hidden_size,
            weights[f"decode_head.linear_c{i}.weight"],
            dtype=work_np_dtype,
        )
        proj = graph_ops.add_bias_sum(
            network,
            proj,
            decoder_hidden_size,
            weights[f"decode_head.linear_c{i}.bias"],
            dtype=work_np_dtype,
        )

        # Reshape to 4D: [H*W, D] -> [1, D, H, W]
        to_4d2 = network.add_shuffle(proj)
        to_4d2.reshape_dims = (1, feat_H, feat_W, decoder_hidden_size)
        to_4d2_t = network.add_shuffle(to_4d2.get_output(0))
        to_4d2_t.first_transpose = trt.Permutation([0, 3, 1, 2])

        # Bilinear upsample to target_H x target_W
        # Match PyTorch F.interpolate(mode='bilinear', align_corners=False)
        if feat_H != target_H or feat_W != target_W:
            resize = network.add_resize(to_4d2_t.get_output(0))
            resize.resize_mode = trt.InterpolationMode.LINEAR
            resize.coordinate_transformation = trt.ResizeCoordinateTransformation.HALF_PIXEL
            resize.shape = (1, decoder_hidden_size, target_H, target_W)
            projected.append(resize.get_output(0))
        else:
            projected.append(to_4d2_t.get_output(0))

    # Concatenate all stages along channel dim.
    # HF reverses the order: cat(stage3, stage2, stage1, stage0).
    # The fuse conv weights are trained with this reversed layout.
    concat = network.add_concatenation(projected[::-1])
    concat.axis = 1  # [1, 4*D, target_H, target_W]

    # Fuse conv (1x1): [1, 4*D, H, W] -> [1, D, H, W]
    fuse_w = weights["decode_head.fuse.weight"]
    fuse_b = weights["decode_head.fuse.bias"]
    fuse_conv = network.add_convolution_nd(
        concat.get_output(0),
        num_output_maps=decoder_hidden_size,
        kernel_shape=(1, 1),
        kernel=trt.Weights(np.ascontiguousarray(fuse_w, dtype=work_np_dtype)),
        bias=trt.Weights(np.ascontiguousarray(fuse_b, dtype=work_np_dtype)),
    )

    # BatchNorm (fused: gamma * (x - mean) / sqrt(var + eps) + beta)
    bn_w = weights["decode_head.bn.weight"]
    bn_b = weights["decode_head.bn.bias"]
    bn_mean = weights["decode_head.bn.running_mean"]
    bn_var = weights["decode_head.bn.running_var"]
    bn_scale = bn_w / np.sqrt(bn_var + 1e-5)
    bn_shift = bn_b - bn_mean * bn_scale

    bn_scale_t = graph_ops.add_constant(
        network, (1, decoder_hidden_size, 1, 1), bn_scale.reshape(1, -1, 1, 1), dtype=work_np_dtype
    )
    bn_shift_t = graph_ops.add_constant(
        network, (1, decoder_hidden_size, 1, 1), bn_shift.reshape(1, -1, 1, 1), dtype=work_np_dtype
    )

    bn_scaled = network.add_elementwise(
        fuse_conv.get_output(0), bn_scale_t, trt.ElementWiseOperation.PROD
    )
    bn_out = network.add_elementwise(
        bn_scaled.get_output(0), bn_shift_t, trt.ElementWiseOperation.SUM
    )

    # ReLU
    relu = network.add_activation(bn_out.get_output(0), trt.ActivationType.RELU)

    # Classifier conv (1x1): [1, D, H, W] -> [1, num_classes, H, W]
    cls_w = weights["decode_head.classifier.weight"]
    cls_b = weights["decode_head.classifier.bias"]
    cls_conv = network.add_convolution_nd(
        relu.get_output(0),
        num_output_maps=num_classes,
        kernel_shape=(1, 1),
        kernel=trt.Weights(np.ascontiguousarray(cls_w, dtype=work_np_dtype)),
        bias=trt.Weights(np.ascontiguousarray(cls_b, dtype=work_np_dtype)),
    )

    # Output: [1, num_classes, H/4, W/4]
    logits = cls_conv.get_output(0)
    if logits.dtype != trt.float32:
        logits = network.add_cast(logits, trt.float32).get_output(0)
    logits.name = "logits"
    network.mark_output(logits)

    if verbose:
        print(
            f"[trtmc build] Building SegFormer engine "
            f"(image={H_in}x{W_in}, classes={num_classes}, "
            f"precision={precision}) ...",
            file=sys.stderr,
        )

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed for SegFormer")
    return bytes(plan)


def get_segmentation_config(config: ModelConfig) -> dict:
    """Return segmentation config for bundle config.json."""
    raw = config.raw
    image_size = raw.get("_resolved_image_size", 512)
    num_classes = raw.get("num_labels", 150)
    return {
        "num_classes": num_classes,
        "input_image_h": image_size,
        "input_image_w": image_size,
        "output_h": image_size // 4,
        "output_w": image_size // 4,
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
    }


requires_tokenizer = False


def _detect_tokenizer_frame(
    source: str, *, revision: str | None = None
) -> tuple[list[int], list[int]] | None:
    try:
        from transformers import AutoTokenizer

        kwargs = {"trust_remote_code": True}
        if revision:
            kwargs["revision"] = revision
        if not Path(source).is_dir():
            kwargs["local_files_only"] = True
        tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
        default_ids = list(tokenizer.encode("hello"))
        plain_ids = list(tokenizer.encode("hello", add_special_tokens=False))
    except Exception:
        return None
    if default_ids == plain_ids:
        return [], []
    if not plain_ids:
        return default_ids, []
    for start in range(len(default_ids) - len(plain_ids) + 1):
        if default_ids[start : start + len(plain_ids)] == plain_ids:
            return default_ids[:start], default_ids[start + len(plain_ids) :]
    return None


def _apply_generation_config_eos(model_dir: Path, config: dict) -> None:
    path = model_dir / "generation_config.json"
    if not path.is_file():
        return
    generation_config = json.loads(path.read_text(encoding="utf-8"))
    if "eos_token_id" in generation_config:
        config["eos_token_id"] = generation_config["eos_token_id"]


def _build_local_engine(
    config, weights, max_cache_length, precision, quant_ctx, verbose, parallel, options
):
    from tensorrt_model_connect.tvm_ffi.graph_build import engine_role, inspection_role

    role = (
        "dual_profile"
        if str(options.get("decoder_engine_layout") or "split") == "dual_profile"
        else "decode"
    )

    def build_role(selected_role: str) -> bytes:
        with engine_role(selected_role):
            return build_engine(
                config,
                weights,
                max_cache_length,
                precision=precision,
                quant_ctx=quant_ctx,
                verbose=verbose,
                parallel_config=parallel,
            )

    target_role = inspection_role()
    if target_role is not None:
        build_role(target_role)
        raise RuntimeError("graph inspection did not reach TensorRT serialization")
    return build_role(role), ("dual_profile" if role == "dual_profile" else "single")


def build(model_dir: str, output_path: str, **options) -> None:
    """Build the complete segformer bundle inside its owning family module."""
    from dataclasses import replace
    from datetime import datetime, timezone

    from tensorrt_model_connect import trt_compat as build_trt_compat
    from tensorrt_model_connect.build_timing import (
        add_build_timing,
        new_build_timing,
        write_build_timing,
    )
    from tensorrt_model_connect.bundle_writer import BundleInfo, BundleSection, write_bundle
    from tensorrt_model_connect.parallel_config import (
        normalize_parallel_config,
        rank_engine_section,
        require_tensorrt_11_for_tensor_parallel,
    )

    model_path = Path(model_dir)
    decoder_engine_layout = str(options.get("decoder_engine_layout") or "split")
    if decoder_engine_layout not in {"split", "dual_profile"}:
        raise ValueError(
            "decoder_engine_layout must be 'split' or 'dual_profile', "
            f"got {decoder_engine_layout!r}"
        )
    parallel = normalize_parallel_config(options.get("parallel_config"))
    if parallel.cp_enabled:
        raise NotImplementedError("segformer does not support context-parallel builds")
    if options.get("dynamic_kv_cache") or options.get("triattention_stats_path"):
        raise ValueError("segformer does not use a decoder KV-cache runtime")

    config = ModelConfig.from_dir(model_path)
    config.raw["_model_dir"] = str(model_path)
    config.raw["_decoder_engine_layout"] = decoder_engine_layout
    config.raw["_fp32_layers"] = sorted(set(options.get("fp32_layers") or ()))
    config.raw["_family_build_options"] = dict(options.get("family_build_options") or {})
    config.raw["_parallel_build_enabled"] = bool(parallel.enabled)
    config.raw["_rtx_build_requested"] = bool(options.get("rtx"))
    config.raw["_runtime_dynamic_kv_requested"] = False
    config.raw["_quantized_build_requested"] = bool(options.get("quantize"))
    precision = str(options.get("precision") or "fp32").lower()
    config.raw["_resolved_build_precision"] = precision
    requested_cache_length = options.get("max_cache_length")
    max_cache_length = int(256 if requested_cache_length is None else requested_cache_length)
    if max_cache_length < 1:
        raise ValueError("max_cache_length must be >= 1")

    timing = new_build_timing(options.get("build_timing_path"))
    timing["model_dir"] = str(model_path)
    timing["output_path"] = str(output_path)
    started = time.monotonic()
    write_build_timing(timing)

    weights_started = time.monotonic()
    weights = load_weights(str(model_path), config)
    add_build_timing(timing, "weights_loading_s", time.monotonic() - weights_started)
    write_build_timing(timing)

    quantize = options.get("quantize")
    quant_ctx = None
    quant_plan = None
    if quantize:
        from tensorrt_model_connect.quantization import QuantPlan, build_quant_context
        from . import graph_ops as family_graph_ops

        quant_plan = QuantPlan.from_build_args(
            precision=precision,
            quantize=str(quantize),
            quant_scales=options.get("quant_scales"),
            quant_calibration_samples=int(options.get("quant_calibration_samples") or 512),
        )
        quant_method = str(
            config.raw.get("quantization_config", {}).get("quant_method", "")
        ).lower()
        if quant_plan.scale_source == "modelopt" and quant_method in {
            "awq",
            "gptq",
            "compressed-tensors",
            "compressed_tensors",
        }:
            quant_plan = replace(quant_plan, scale_source="prequantized")
        quant_ctx = build_quant_context(
            format_name=quant_plan.quant_format,
            model_dir=str(model_path),
            config=config,
            scales_json=options.get("quant_scales"),
            num_calibration_samples=int(options.get("quant_calibration_samples") or 512),
            quant_plan=quant_plan,
            graph_ops=family_graph_ops,
        )

    if parallel.enabled:
        require_tensorrt_11_for_tensor_parallel(
            parallel, feature="segformer tensor-parallel builds"
        )
        if quant_ctx is not None:
            raise ValueError("segformer tensor-parallel builds do not support quantization")

    verbose = bool(options.get("verbose"))
    compile_started = time.monotonic()
    if parallel.enabled:
        plans = {
            rank: build_engine(
                config,
                weights,
                max_cache_length,
                precision=precision,
                quant_ctx=quant_ctx,
                verbose=verbose,
                parallel_config=parallel.for_rank(rank),
            )
            for rank in range(parallel.tp_size)
        }
        sections = [
            BundleSection(rank_engine_section(rank), plan) for rank, plan in sorted(plans.items())
        ]
        decoder_layout = "dual_profile"
    else:
        plan, decoder_layout = _build_local_engine(
            config, weights, max_cache_length, precision, quant_ctx, verbose, parallel, options
        )
        sections = [BundleSection("engine_plan", plan)]
    compile_elapsed = time.monotonic() - compile_started
    add_build_timing(timing, "trt_compile_s", compile_elapsed)
    add_build_timing(timing, "trt_compile_main_engine_s", compile_elapsed)
    write_build_timing(timing)

    tokenizer_frame = _detect_tokenizer_frame(str(model_path))
    prefix_ids, suffix_ids = tokenizer_frame or ([], [])
    add_special_tokens = bool(prefix_ids or suffix_ids)

    trt_version = build_trt_compat.tensorrt_version() or "unknown"
    version_match = re.search(r"(\d+)\.(\d+)", trt_version)
    trt_abi = f"{version_match.group(1)}.{version_match.group(2)}" if version_match else ""
    try:
        from tensorrt_model_connect.runtime_provider.target import _probe_current_target_with_device

        gpu_name = str(_probe_current_target_with_device()[0]["gpu_name"])
    except Exception:
        gpu_name = ""
    info = BundleInfo(
        model_id=model_path.name,
        model_type=config.model_type,
        family=name,
        trt_version=trt_version,
        trt_abi=trt_abi,
        gpu_name=gpu_name,
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        max_cache_length=max_cache_length,
        runtime_strategy=runtime_strategy,
        precision=precision,
        quantization=(quant_plan.quant_format if quant_plan else "none"),
        tokenizer_add_special_tokens=add_special_tokens,
    )

    source_config = model_path / "config.json"
    runtime_config = (
        json.loads(source_config.read_text(encoding="utf-8"))
        if source_config.is_file()
        else dict(config.raw)
    )
    _apply_generation_config_eos(model_path, runtime_config)
    runtime_config.update(
        {
            "runtime_strategy": runtime_strategy,
            "engine_backend": "trt_rtx" if options.get("rtx") else "trt",
            "trt_version": trt_version,
            "precision": precision,
            "tokenizer_add_special_tokens": int(add_special_tokens),
            "decoder_engine_layout": decoder_layout,
        }
    )
    if trt_abi:
        runtime_config["trt_abi"] = trt_abi
    if tokenizer_frame is not None:
        runtime_config["tokenizer_special_prefix_ids"] = prefix_ids
        runtime_config["tokenizer_special_suffix_ids"] = suffix_ids
    if options.get("fp32_layers"):
        runtime_config["fp32_layers"] = sorted(set(options["fp32_layers"]))
    if quant_plan is not None:
        runtime_config["quantization"] = quant_plan.as_config_dict()
    runtime_config.update(parallel.to_bundle_config_fields())
    segmentation = get_segmentation_config(config)
    if segmentation is not None:
        runtime_config.update(segmentation)

    from tensorrt_model_connect.tvm_ffi.graph_build import kernel_slots_section

    slot_section = kernel_slots_section()
    if slot_section is not None:
        sections.append(BundleSection("kernel_slots.json", slot_section))

    embedded_config = False
    for filename in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.model",
        "preprocessor_config.json",
        "processor_config.json",
    ):
        path = model_path / filename
        if filename == "config.json":
            sections.append(
                BundleSection(filename, json.dumps(runtime_config, indent=2).encode("utf-8"))
            )
            embedded_config = True
        elif path.is_file():
            sections.append(BundleSection(filename, path.read_bytes()))
    if not embedded_config:
        sections.append(
            BundleSection("config.json", json.dumps(runtime_config, indent=2).encode("utf-8"))
        )

    kernel_manifest = []
    for global_name, library in options.get("kernel_artifacts") or ():
        section_name = f"kernel_{global_name.replace('.', '_')}.so"
        sections.append(BundleSection(section_name, Path(library).read_bytes()))
        kernel_manifest.append(
            {"global_name": global_name, "func_name": "run", "section": section_name}
        )
    if kernel_manifest:
        sections.append(
            BundleSection(
                "kernel_manifest.json",
                json.dumps({"kernels": kernel_manifest}).encode("utf-8"),
            )
        )

    write_started = time.monotonic()
    write_bundle(output_path, info, sections)
    add_build_timing(timing, "bundle_write_s", time.monotonic() - write_started)
    timing["total_s"] = time.monotonic() - started
    write_build_timing(timing)
