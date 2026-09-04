# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SegFormer family plugin -- semantic segmentation (SegFormer-B0..B5).

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

from typing import TYPE_CHECKING

import json
import sys
from pathlib import Path

import numpy as np
import tensorrt as trt

from .config import ModelConfig
from .parallel import ParallelConfig
from .checkpoint_mapper import (
    WeightDict,
    _open_safetensors,
    _load_tensor,
    _has_tensor,
    _transpose_2d,
)
from . import graph_ops
from .parallel import normalize_parallel_config
from .segformer_tp_builder import build_segformer_tp_engine


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


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


class _SegformerModel:
    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
    ) -> WeightDict:
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
                    w = _load_tensor(readers, f"{blk_prefix}.attention.self.{proj}.weight")
                    b = _load_tensor(readers, f"{blk_prefix}.attention.self.{proj}.bias")
                    weights[f"{w_prefix}.attn.{proj[0]}.weight"] = _transpose_2d(w, f"attn_{proj}")
                    weights[f"{w_prefix}.attn.{proj[0]}.bias"] = b.astype(np.float32)

                w_o = _load_tensor(readers, f"{blk_prefix}.attention.output.dense.weight")
                b_o = _load_tensor(readers, f"{blk_prefix}.attention.output.dense.bias")
                weights[f"{w_prefix}.attn.o.weight"] = _transpose_2d(w_o, "attn_o")
                weights[f"{w_prefix}.attn.o.bias"] = b_o.astype(np.float32)

                # SR (sequence reduction) if sr_ratio > 1
                if sr > 1:
                    sr_w = _load_tensor(readers, f"{blk_prefix}.attention.self.sr.weight")
                    sr_b = _load_tensor(readers, f"{blk_prefix}.attention.self.sr.bias")
                    weights[f"{w_prefix}.attn.sr.weight"] = sr_w.astype(np.float32)
                    weights[f"{w_prefix}.attn.sr.bias"] = sr_b.astype(np.float32)

                    sr_ln_w = _load_tensor(
                        readers, f"{blk_prefix}.attention.self.layer_norm.weight"
                    )
                    sr_ln_b = _load_tensor(readers, f"{blk_prefix}.attention.self.layer_norm.bias")
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
        weights["decode_head.bn.bias"] = _load_tensor(
            readers, "decode_head.batch_norm.bias"
        ).astype(np.float32)
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
        self,
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
            parallel.validate()
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
                network,
                (1, 1),
                np.array([layer_norm_eps], dtype=work_np_dtype),
                dtype=work_np_dtype,
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
                    network,
                    attn_out,
                    hidden,
                    weights[f"{w_prefix}.attn.o.bias"],
                    dtype=work_np_dtype,
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
                    network,
                    fc1,
                    ffn_hidden,
                    weights[f"{w_prefix}.mlp.fc1.bias"],
                    dtype=work_np_dtype,
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
            # Keep the encoder in the requested precision, but perform the
            # lightweight decode head in FP32.  Small FP16 tactic differences
            # near class boundaries otherwise flip argmax labels between
            # otherwise equivalent engine builds.
            feat = graph_ops.begin_fp32_decode_head(network, feat)

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
                dtype=np.float32,
            )
            proj = graph_ops.add_bias_sum(
                network,
                proj,
                decoder_hidden_size,
                weights[f"decode_head.linear_c{i}.bias"],
                dtype=np.float32,
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
            kernel=trt.Weights(np.ascontiguousarray(fuse_w, dtype=np.float32)),
            bias=trt.Weights(np.ascontiguousarray(fuse_b, dtype=np.float32)),
        )

        # BatchNorm (fused: gamma * (x - mean) / sqrt(var + eps) + beta)
        bn_w = weights["decode_head.bn.weight"]
        bn_b = weights["decode_head.bn.bias"]
        bn_mean = weights["decode_head.bn.running_mean"]
        bn_var = weights["decode_head.bn.running_var"]
        bn_scale = bn_w / np.sqrt(bn_var + 1e-5)
        bn_shift = bn_b - bn_mean * bn_scale

        bn_scale_t = graph_ops.add_constant(
            network, (1, decoder_hidden_size, 1, 1), bn_scale.reshape(1, -1, 1, 1), dtype=np.float32
        )
        bn_shift_t = graph_ops.add_constant(
            network, (1, decoder_hidden_size, 1, 1), bn_shift.reshape(1, -1, 1, 1), dtype=np.float32
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
            kernel=trt.Weights(np.ascontiguousarray(cls_w, dtype=np.float32)),
            bias=trt.Weights(np.ascontiguousarray(cls_b, dtype=np.float32)),
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

    def get_segmentation_config(self, config: ModelConfig) -> dict:
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


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one SegFormer bundle."""
    if request.image_height is not None:
        raise NotImplementedError("segformer does not support image_height")

    if request.image_width is not None:
        raise NotImplementedError("segformer does not support image_width")

    if request.video_num_frames is not None:
        raise NotImplementedError("segformer does not support video_num_frames")

    if request.max_batch_size != 1:
        raise NotImplementedError("segformer does not support max_batch_size")

    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task != "segmentation":
        raise ValueError("segformer supports only task=segmentation")
    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    if str(config.model_type).lower() != "segformer":
        raise ValueError(f"SegFormer does not support model_type={config.model_type!r}")
    precision = str(request.precision).lower()
    max_sequence_length = _positive_int(request.max_sequence_length or 1, "max_sequence_length")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("SegFormer does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("SegFormer does not support mixed-precision layers")
    parallel = ParallelConfig(
        tp_size=_positive_int(request.tensor_parallel_size, "tensor_parallel_size")
    )
    parallel.validate()
    model = _SegformerModel()
    weights = model.load_weights(str(model_dir), config)
    writer.set_header(family="segformer", task=request.task, backend="trt")
    if parallel.enabled:
        for rank in range(parallel.tp_size):
            plan = model.build_engine(
                config,
                weights,
                max_sequence_length,
                precision=precision,
                quant_ctx=None,
                verbose=bool(request.verbose),
                parallel_config=parallel.for_rank(rank),
            )
            writer.add_bytes(f"engine.rank{rank}.plan", plan)
    else:
        plan = model.build_engine(
            config,
            weights,
            max_sequence_length,
            precision=precision,
            quant_ctx=None,
            verbose=bool(request.verbose),
            parallel_config=parallel,
        )
        writer.add_bytes("engine.plan", plan)
    runtime_source = model.get_segmentation_config(config)
    runtime = {
        key: runtime_source[key]
        for key in (
            "num_classes",
            "input_image_h",
            "input_image_w",
            "output_h",
            "output_w",
            "image_mean",
            "image_std",
        )
    }
    runtime["tensor_parallel_size"] = request.tensor_parallel_size
    writer.add_json("runtime.json", runtime)
