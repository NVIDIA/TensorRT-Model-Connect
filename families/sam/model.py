# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SAM (Segment Anything Model) family plugin -- prompted segmentation.

SAM is an image segmentation model with three components:
  1. ViT Image Encoder: Standard ViT with windowed attention
     - Input: [1, 3, 1024, 1024]
     - Output: [1, 256, 64, 64] (image embeddings)
  2. Prompt Encoder: Encodes points/boxes into embeddings (done in C++)
  3. Mask Decoder: Lightweight two-way Transformer
     - Inputs: image_embeddings, sparse_prompt, dense_prompt
     - Outputs: masks [N, 256, 256], iou_scores [N]

Weight key mapping (HF -> engine):
  HF: vision_encoder.patch_embed.projection.weight/bias
  HF: vision_encoder.layers.{i}.layer_norm1/2.weight/bias
  HF: vision_encoder.layers.{i}.attn.qkv.weight/bias
  HF: vision_encoder.layers.{i}.attn.proj.weight/bias
  HF: vision_encoder.layers.{i}.mlp.lin1/2.weight/bias
  HF: vision_encoder.layers.{i}.attn.rel_pos_h/rel_pos_w
  HF: vision_encoder.neck.conv1/conv2.weight  (Conv2d projections)
  HF: vision_encoder.neck.layer_norm1/2.weight/bias  (Neck LayerNorms)
  HF: shared_image_embedding.positional_embedding  (learned absolute pos embed)
  HF: prompt_encoder.point_embed.{i}.weight  (4 point types)
  HF: prompt_encoder.not_a_point_embed.weight
  HF: prompt_encoder.mask_embed.conv{1,2,3}.weight/bias (Conv layers)
  HF: prompt_encoder.mask_embed.layer_norm{1,2}.weight/bias (LayerNorms)
  HF: prompt_encoder.no_mask_embed.weight
  HF: mask_decoder.iou_token.weight
  HF: mask_decoder.mask_tokens.weight
  HF: mask_decoder.transformer.layers.{i}.*
  HF: mask_decoder.upscale_conv1/conv2.weight/bias, upscale_layer_norm.weight/bias
  HF: mask_decoder.output_hypernetworks_mlps.{i}.{proj_in,layers.0,proj_out}.weight/bias
  HF: mask_decoder.iou_prediction_head.{proj_in,layers.0,proj_out}.weight/bias
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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


def _precision_types(precision: str) -> tuple[np.dtype, trt.DataType]:
    if precision == "fp16":
        return np.float16, trt.float16
    if precision == "fp32":
        return np.float32, trt.float32
    raise ValueError(f"Unsupported SAM precision: {precision}")


def _resolve_sam_config(raw: dict) -> dict:
    """Extract SAM-specific config fields from raw config.json."""
    vision_config = raw.get("vision_config", {})
    mask_decoder_config = raw.get("mask_decoder_config", {})
    prompt_encoder_config = raw.get("prompt_encoder_config", {})

    return {
        # Vision encoder
        "hidden_size": vision_config.get("hidden_size", 768),
        "num_hidden_layers": vision_config.get("num_hidden_layers", 12),
        "num_attention_heads": vision_config.get("num_attention_heads", 12),
        "image_size": vision_config.get("image_size", 1024),
        "patch_size": vision_config.get("patch_size", 16),
        "mlp_dim": vision_config.get("mlp_dim", 3072),
        "window_size": vision_config.get("window_size", 14),
        "global_attn_indexes": vision_config.get("global_attn_indexes", [2, 5, 8, 11]),
        # Prompt encoder
        "prompt_hidden_size": prompt_encoder_config.get("hidden_size", 256),
        "image_embedding_size": prompt_encoder_config.get("image_embedding_size", 64),
        "mask_input_channels": prompt_encoder_config.get("mask_input_channels", 16),
        # Mask decoder
        "decoder_hidden_size": mask_decoder_config.get("hidden_size", 256),
        "num_multimask_outputs": mask_decoder_config.get("num_multimask_outputs", 3),
        "decoder_num_heads": mask_decoder_config.get("num_attention_heads", 8),
        "decoder_depth": mask_decoder_config.get(
            "num_hidden_layers", mask_decoder_config.get("depth", 2)
        ),
        "decoder_mlp_dim": mask_decoder_config.get(
            "mlp_dim", mask_decoder_config.get("hidden_size", 256) * 8
        ),
        "attention_downsample_rate": mask_decoder_config.get("attention_downsample_rate", 2),
    }


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


class _SamModel:
    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
    ) -> WeightDict:
        """Load SAM weights from safetensors."""
        model_dir_path = Path(model_dir)
        readers = _open_safetensors(model_dir_path)

        sam_cfg = _resolve_sam_config(config.raw)
        config.raw["_sam_config"] = sam_cfg

        num_layers = sam_cfg["num_hidden_layers"]
        decoder_depth = sam_cfg["decoder_depth"]
        num_multimask = sam_cfg["num_multimask_outputs"]

        weights = WeightDict()

        # --- Vision encoder ---

        # Patch embedding: Conv2d [hidden, 3, patch_size, patch_size]
        weights["encoder.patch_embed.weight"] = _load_tensor(
            readers, "vision_encoder.patch_embed.projection.weight"
        ).astype(np.float32)
        weights["encoder.patch_embed.bias"] = _load_tensor(
            readers, "vision_encoder.patch_embed.projection.bias"
        ).astype(np.float32)

        # Learned absolute position embedding
        # Shape: [1, image_size/patch_size, image_size/patch_size, hidden]
        pos_embed = _load_tensor(readers, "vision_encoder.pos_embed").astype(np.float32)
        weights["encoder.pos_embed"] = pos_embed

        # Shared image embedding for prompt encoder (sinusoidal PE coefficients)
        if _has_tensor(readers, "shared_image_embedding.positional_embedding"):
            weights["prompt.shared_image_pe"] = _load_tensor(
                readers, "shared_image_embedding.positional_embedding"
            ).astype(np.float32)

        for layer_idx in range(num_layers):
            hf_prefix = f"vision_encoder.layers.{layer_idx}"
            w_prefix = f"encoder.layer{layer_idx}"

            # Layer norms
            weights[f"{w_prefix}.norm1.weight"] = _load_tensor(
                readers, f"{hf_prefix}.layer_norm1.weight"
            ).astype(np.float32)
            weights[f"{w_prefix}.norm1.bias"] = _load_tensor(
                readers, f"{hf_prefix}.layer_norm1.bias"
            ).astype(np.float32)
            weights[f"{w_prefix}.norm2.weight"] = _load_tensor(
                readers, f"{hf_prefix}.layer_norm2.weight"
            ).astype(np.float32)
            weights[f"{w_prefix}.norm2.bias"] = _load_tensor(
                readers, f"{hf_prefix}.layer_norm2.bias"
            ).astype(np.float32)

            # Fused QKV projection [3*hidden, hidden]
            qkv_w = _load_tensor(readers, f"{hf_prefix}.attn.qkv.weight").astype(np.float32)
            qkv_b = _load_tensor(readers, f"{hf_prefix}.attn.qkv.bias").astype(np.float32)

            # Split into Q, K, V [hidden, hidden] each (transposed)
            q_w, k_w, v_w = np.split(qkv_w, 3, axis=0)
            q_b, k_b, v_b = np.split(qkv_b, 3, axis=0)

            weights[f"{w_prefix}.attn.q.weight"] = _transpose_2d(q_w, "q")
            weights[f"{w_prefix}.attn.q.bias"] = q_b.flatten().astype(np.float32)
            weights[f"{w_prefix}.attn.k.weight"] = _transpose_2d(k_w, "k")
            weights[f"{w_prefix}.attn.k.bias"] = k_b.flatten().astype(np.float32)
            weights[f"{w_prefix}.attn.v.weight"] = _transpose_2d(v_w, "v")
            weights[f"{w_prefix}.attn.v.bias"] = v_b.flatten().astype(np.float32)

            # Output projection
            o_w = _load_tensor(readers, f"{hf_prefix}.attn.proj.weight").astype(np.float32)
            o_b = _load_tensor(readers, f"{hf_prefix}.attn.proj.bias").astype(np.float32)
            weights[f"{w_prefix}.attn.o.weight"] = _transpose_2d(o_w, "o")
            weights[f"{w_prefix}.attn.o.bias"] = o_b.flatten().astype(np.float32)

            # Relative position embeddings (for windowed attention)
            if _has_tensor(readers, f"{hf_prefix}.attn.rel_pos_h"):
                weights[f"{w_prefix}.attn.rel_pos_h"] = _load_tensor(
                    readers, f"{hf_prefix}.attn.rel_pos_h"
                ).astype(np.float32)
                weights[f"{w_prefix}.attn.rel_pos_w"] = _load_tensor(
                    readers, f"{hf_prefix}.attn.rel_pos_w"
                ).astype(np.float32)

            # MLP
            fc1_w = _load_tensor(readers, f"{hf_prefix}.mlp.lin1.weight")
            fc1_b = _load_tensor(readers, f"{hf_prefix}.mlp.lin1.bias")
            fc2_w = _load_tensor(readers, f"{hf_prefix}.mlp.lin2.weight")
            fc2_b = _load_tensor(readers, f"{hf_prefix}.mlp.lin2.bias")
            weights[f"{w_prefix}.mlp.fc1.weight"] = _transpose_2d(fc1_w, "fc1")
            weights[f"{w_prefix}.mlp.fc1.bias"] = fc1_b.flatten().astype(np.float32)
            weights[f"{w_prefix}.mlp.fc2.weight"] = _transpose_2d(fc2_w, "fc2")
            weights[f"{w_prefix}.mlp.fc2.bias"] = fc2_b.flatten().astype(np.float32)

        # Neck: 2x Conv2d projections (hidden -> decoder_hidden)
        weights["encoder.neck.conv1.weight"] = _load_tensor(
            readers, "vision_encoder.neck.conv1.weight"
        ).astype(np.float32)
        if _has_tensor(readers, "vision_encoder.neck.conv1.bias"):
            weights["encoder.neck.conv1.bias"] = _load_tensor(
                readers, "vision_encoder.neck.conv1.bias"
            ).astype(np.float32)
        weights["encoder.neck.conv2.weight"] = _load_tensor(
            readers, "vision_encoder.neck.conv2.weight"
        ).astype(np.float32)
        if _has_tensor(readers, "vision_encoder.neck.conv2.bias"):
            weights["encoder.neck.conv2.bias"] = _load_tensor(
                readers, "vision_encoder.neck.conv2.bias"
            ).astype(np.float32)

        # Neck LayerNorms (applied between convs in NHWC format)
        # SAM neck: Conv2d -> LN -> Conv2d -> LN
        # HF stores them as neck.layer_norm1 and neck.layer_norm2
        weights["encoder.neck.ln1.weight"] = _load_tensor(
            readers, "vision_encoder.neck.layer_norm1.weight"
        ).astype(np.float32)
        weights["encoder.neck.ln1.bias"] = _load_tensor(
            readers, "vision_encoder.neck.layer_norm1.bias"
        ).astype(np.float32)
        weights["encoder.neck.ln2.weight"] = _load_tensor(
            readers, "vision_encoder.neck.layer_norm2.weight"
        ).astype(np.float32)
        weights["encoder.neck.ln2.bias"] = _load_tensor(
            readers, "vision_encoder.neck.layer_norm2.bias"
        ).astype(np.float32)

        # --- Prompt encoder ---
        # Point embeddings: 4 types (bg, fg, top-left, bottom-right)
        for i in range(4):
            weights[f"prompt.point_embed.{i}"] = (
                _load_tensor(readers, f"prompt_encoder.point_embed.{i}.weight")
                .flatten()
                .astype(np.float32)
            )

        weights["prompt.not_a_point_embed"] = (
            _load_tensor(readers, "prompt_encoder.not_a_point_embed.weight")
            .flatten()
            .astype(np.float32)
        )

        weights["prompt.no_mask_embed"] = (
            _load_tensor(readers, "prompt_encoder.no_mask_embed.weight")
            .flatten()
            .astype(np.float32)
        )

        # Mask embed conv layers (HF: prompt_encoder.mask_embed.conv{1,2,3})
        for i in range(3):
            hf_conv = f"prompt_encoder.mask_embed.conv{i + 1}"
            if _has_tensor(readers, f"{hf_conv}.weight"):
                weights[f"prompt.mask_down.{i}.weight"] = _load_tensor(
                    readers, f"{hf_conv}.weight"
                ).astype(np.float32)
                weights[f"prompt.mask_down.{i}.bias"] = _load_tensor(
                    readers, f"{hf_conv}.bias"
                ).astype(np.float32)
        # Mask embed layer norms (HF: prompt_encoder.mask_embed.layer_norm{1,2})
        for i in range(1, 3):
            hf_ln = f"prompt_encoder.mask_embed.layer_norm{i}"
            if _has_tensor(readers, f"{hf_ln}.weight"):
                weights[f"prompt.mask_down_ln.{i}.weight"] = _load_tensor(
                    readers, f"{hf_ln}.weight"
                ).astype(np.float32)
                weights[f"prompt.mask_down_ln.{i}.bias"] = _load_tensor(
                    readers, f"{hf_ln}.bias"
                ).astype(np.float32)

        # --- Mask decoder ---
        weights["decoder.iou_token"] = (
            _load_tensor(readers, "mask_decoder.iou_token.weight").flatten().astype(np.float32)
        )
        weights["decoder.mask_tokens"] = _load_tensor(
            readers, "mask_decoder.mask_tokens.weight"
        ).astype(np.float32)

        # Two-way transformer layers
        for layer_idx in range(decoder_depth):
            hf_prefix = f"mask_decoder.transformer.layers.{layer_idx}"
            w_prefix = f"decoder.layer{layer_idx}"

            # Self-attention on tokens
            for proj in ("q", "k", "v"):
                proj_name = {"q": "q_proj", "k": "k_proj", "v": "v_proj"}[proj]
                w = _load_tensor(readers, f"{hf_prefix}.self_attn.{proj_name}.weight")
                b = _load_tensor(readers, f"{hf_prefix}.self_attn.{proj_name}.bias")
                weights[f"{w_prefix}.self_attn.{proj}.weight"] = _transpose_2d(w, f"sa_{proj}")
                weights[f"{w_prefix}.self_attn.{proj}.bias"] = b.flatten().astype(np.float32)

            sa_o_w = _load_tensor(readers, f"{hf_prefix}.self_attn.out_proj.weight")
            sa_o_b = _load_tensor(readers, f"{hf_prefix}.self_attn.out_proj.bias")
            weights[f"{w_prefix}.self_attn.o.weight"] = _transpose_2d(sa_o_w, "sa_o")
            weights[f"{w_prefix}.self_attn.o.bias"] = sa_o_b.flatten().astype(np.float32)

            # Cross-attention (token-to-image)
            for proj in ("q", "k", "v"):
                proj_name = {"q": "q_proj", "k": "k_proj", "v": "v_proj"}[proj]
                w = _load_tensor(
                    readers, f"{hf_prefix}.cross_attn_token_to_image.{proj_name}.weight"
                )
                b = _load_tensor(readers, f"{hf_prefix}.cross_attn_token_to_image.{proj_name}.bias")
                weights[f"{w_prefix}.cross_t2i.{proj}.weight"] = _transpose_2d(w, f"t2i_{proj}")
                weights[f"{w_prefix}.cross_t2i.{proj}.bias"] = b.flatten().astype(np.float32)

            t2i_o_w = _load_tensor(
                readers, f"{hf_prefix}.cross_attn_token_to_image.out_proj.weight"
            )
            t2i_o_b = _load_tensor(readers, f"{hf_prefix}.cross_attn_token_to_image.out_proj.bias")
            weights[f"{w_prefix}.cross_t2i.o.weight"] = _transpose_2d(t2i_o_w, "t2i_o")
            weights[f"{w_prefix}.cross_t2i.o.bias"] = t2i_o_b.flatten().astype(np.float32)

            # Cross-attention (image-to-token)
            for proj in ("q", "k", "v"):
                proj_name = {"q": "q_proj", "k": "k_proj", "v": "v_proj"}[proj]
                w = _load_tensor(
                    readers, f"{hf_prefix}.cross_attn_image_to_token.{proj_name}.weight"
                )
                b = _load_tensor(readers, f"{hf_prefix}.cross_attn_image_to_token.{proj_name}.bias")
                weights[f"{w_prefix}.cross_i2t.{proj}.weight"] = _transpose_2d(w, f"i2t_{proj}")
                weights[f"{w_prefix}.cross_i2t.{proj}.bias"] = b.flatten().astype(np.float32)

            i2t_o_w = _load_tensor(
                readers, f"{hf_prefix}.cross_attn_image_to_token.out_proj.weight"
            )
            i2t_o_b = _load_tensor(readers, f"{hf_prefix}.cross_attn_image_to_token.out_proj.bias")
            weights[f"{w_prefix}.cross_i2t.o.weight"] = _transpose_2d(i2t_o_w, "i2t_o")
            weights[f"{w_prefix}.cross_i2t.o.bias"] = i2t_o_b.flatten().astype(np.float32)

            # Layer norms
            for ln_name in ("norm1", "norm2", "norm3", "norm4"):
                hf_ln = f"{hf_prefix}.layer_{ln_name}"
                if _has_tensor(readers, f"{hf_ln}.weight"):
                    weights[f"{w_prefix}.{ln_name}.weight"] = _load_tensor(
                        readers, f"{hf_ln}.weight"
                    ).astype(np.float32)
                    weights[f"{w_prefix}.{ln_name}.bias"] = _load_tensor(
                        readers, f"{hf_ln}.bias"
                    ).astype(np.float32)

            # MLP in decoder layer
            mlp_fc1_w = _load_tensor(readers, f"{hf_prefix}.mlp.lin1.weight")
            mlp_fc1_b = _load_tensor(readers, f"{hf_prefix}.mlp.lin1.bias")
            mlp_fc2_w = _load_tensor(readers, f"{hf_prefix}.mlp.lin2.weight")
            mlp_fc2_b = _load_tensor(readers, f"{hf_prefix}.mlp.lin2.bias")
            weights[f"{w_prefix}.mlp.fc1.weight"] = _transpose_2d(mlp_fc1_w, "dec_fc1")
            weights[f"{w_prefix}.mlp.fc1.bias"] = mlp_fc1_b.flatten().astype(np.float32)
            weights[f"{w_prefix}.mlp.fc2.weight"] = _transpose_2d(mlp_fc2_w, "dec_fc2")
            weights[f"{w_prefix}.mlp.fc2.bias"] = mlp_fc2_b.flatten().astype(np.float32)

        # Final attention layer in decoder (after all two-way layers)
        final_prefix = "mask_decoder.transformer.final_attn_token_to_image"
        for proj in ("q", "k", "v"):
            proj_name = {"q": "q_proj", "k": "k_proj", "v": "v_proj"}[proj]
            w = _load_tensor(readers, f"{final_prefix}.{proj_name}.weight")
            b = _load_tensor(readers, f"{final_prefix}.{proj_name}.bias")
            weights[f"decoder.final_t2i.{proj}.weight"] = _transpose_2d(w, f"final_t2i_{proj}")
            weights[f"decoder.final_t2i.{proj}.bias"] = b.flatten().astype(np.float32)

        final_o_w = _load_tensor(readers, f"{final_prefix}.out_proj.weight")
        final_o_b = _load_tensor(readers, f"{final_prefix}.out_proj.bias")
        weights["decoder.final_t2i.o.weight"] = _transpose_2d(final_o_w, "final_t2i_o")
        weights["decoder.final_t2i.o.bias"] = final_o_b.flatten().astype(np.float32)

        # Final layer norms
        weights["decoder.final_norm.weight"] = _load_tensor(
            readers, "mask_decoder.transformer.layer_norm_final_attn.weight"
        ).astype(np.float32)
        weights["decoder.final_norm.bias"] = _load_tensor(
            readers, "mask_decoder.transformer.layer_norm_final_attn.bias"
        ).astype(np.float32)

        # Output upscaling (ConvTranspose2d layers)
        weights["decoder.upscale.conv1.weight"] = _load_tensor(
            readers, "mask_decoder.upscale_conv1.weight"
        ).astype(np.float32)
        if _has_tensor(readers, "mask_decoder.upscale_conv1.bias"):
            weights["decoder.upscale.conv1.bias"] = _load_tensor(
                readers, "mask_decoder.upscale_conv1.bias"
            ).astype(np.float32)
        # LayerNorm between upsample convs
        weights["decoder.upscale.ln.weight"] = _load_tensor(
            readers, "mask_decoder.upscale_layer_norm.weight"
        ).astype(np.float32)
        weights["decoder.upscale.ln.bias"] = _load_tensor(
            readers, "mask_decoder.upscale_layer_norm.bias"
        ).astype(np.float32)
        weights["decoder.upscale.conv2.weight"] = _load_tensor(
            readers, "mask_decoder.upscale_conv2.weight"
        ).astype(np.float32)
        if _has_tensor(readers, "mask_decoder.upscale_conv2.bias"):
            weights["decoder.upscale.conv2.bias"] = _load_tensor(
                readers, "mask_decoder.upscale_conv2.bias"
            ).astype(np.float32)

        # Output hypernetworks MLPs (one per mask output)
        # HF uses: proj_in, layers.0, proj_out (3-layer MLP naming)
        num_mask_outputs = num_multimask + 1  # multimask + single mask
        _hyper_layer_map = {0: "proj_in", 1: "layers.0", 2: "proj_out"}
        for i in range(num_mask_outputs):
            for j in range(3):  # 3-layer MLPs
                hf_suffix = _hyper_layer_map[j]
                hf_key = f"mask_decoder.output_hypernetworks_mlps.{i}.{hf_suffix}"
                w = _load_tensor(readers, f"{hf_key}.weight")
                b = _load_tensor(readers, f"{hf_key}.bias")
                weights[f"decoder.hyper_mlp.{i}.{j}.weight"] = _transpose_2d(w, f"hyper_{i}_{j}")
                weights[f"decoder.hyper_mlp.{i}.{j}.bias"] = b.flatten().astype(np.float32)

        # IoU prediction head (3-layer MLP)
        # HF uses: proj_in, layers.0, proj_out
        _iou_layer_map = {0: "proj_in", 1: "layers.0", 2: "proj_out"}
        for j in range(3):
            hf_suffix = _iou_layer_map[j]
            hf_key = f"mask_decoder.iou_prediction_head.{hf_suffix}"
            w = _load_tensor(readers, f"{hf_key}.weight")
            b = _load_tensor(readers, f"{hf_key}.bias")
            weights[f"decoder.iou_head.{j}.weight"] = _transpose_2d(w, f"iou_{j}")
            weights[f"decoder.iou_head.{j}.bias"] = b.flatten().astype(np.float32)

        # Store point embeddings and shared PE in config for C++ prompt encoding
        sam_cfg["_point_embed_0"] = weights["prompt.point_embed.0"].tolist()
        sam_cfg["_point_embed_1"] = weights["prompt.point_embed.1"].tolist()
        sam_cfg["_not_a_point_embed"] = weights["prompt.not_a_point_embed"].tolist()
        if "prompt.shared_image_pe" in weights:
            sam_cfg["_shared_image_pe"] = weights["prompt.shared_image_pe"].flatten().tolist()

        # Save sam_cfg back to config so get_segmentation_config() can access embeddings
        config.raw["_sam_config"] = sam_cfg

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
        parallel_config=None,
    ) -> bytes:
        """Build TRT engine for SAM image encoder.

        Input:  pixel_values [1, 3, 1024, 1024]
        Output: image_embeddings [1, 256, 64, 64]
        """
        parallel = normalize_parallel_config(parallel_config)
        if parallel.enabled:
            parallel.validate()
            if quant_ctx is not None:
                raise ValueError("SAM tensor-parallel builds do not support quantization")
            from .sam_tp_builder import build_sam_tp_encoder_engine

            return build_sam_tp_encoder_engine(
                config,
                weights,
                max_cache_length,
                precision=precision,
                quant_ctx=quant_ctx,
                verbose=verbose,
                parallel_config=parallel,
            )

        sam_cfg = config.raw.get("_sam_config", _resolve_sam_config(config.raw))
        hidden = sam_cfg["hidden_size"]
        num_layers = sam_cfg["num_hidden_layers"]
        num_heads = sam_cfg["num_attention_heads"]
        head_dim = hidden // num_heads
        mlp_dim = sam_cfg["mlp_dim"]
        image_size = sam_cfg["image_size"]
        patch_size = sam_cfg["patch_size"]
        window_size = sam_cfg["window_size"]
        global_attn_indexes = set(sam_cfg["global_attn_indexes"])
        decoder_hidden = sam_cfg["decoder_hidden_size"]
        work_np_dtype, work_trt_dtype = _precision_types(precision)

        grid_size = image_size // patch_size  # 64 for 1024/16
        seq_len = grid_size * grid_size  # 4096

        logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        trt_config = builder.create_builder_config()
        trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

        eps_t = graph_ops.add_constant(network, (1, 1), np.array([1e-6], dtype=np.float32))

        # Input: [1, 3, image_size, image_size]
        pixel_values = network.add_input(
            "pixel_values", trt.float32, (1, 3, image_size, image_size)
        )
        pixel_values_work = pixel_values
        if work_trt_dtype != trt.float32:
            pixel_values_work = network.add_cast(pixel_values, work_trt_dtype).get_output(0)

        # Patch embedding: Conv2d [1, 3, 1024, 1024] -> [1, hidden, 64, 64]
        pe_w = weights["encoder.patch_embed.weight"]
        pe_b = weights["encoder.patch_embed.bias"]
        patch_conv = network.add_convolution_nd(
            pixel_values_work,
            num_output_maps=hidden,
            kernel_shape=(patch_size, patch_size),
            kernel=trt.Weights(np.ascontiguousarray(pe_w, dtype=work_np_dtype)),
            bias=trt.Weights(np.ascontiguousarray(pe_b, dtype=work_np_dtype)),
        )
        patch_conv.stride_nd = (patch_size, patch_size)

        # Permute to NHWC: [1, hidden, 64, 64] -> [1, 64, 64, hidden]
        to_nhwc = network.add_shuffle(patch_conv.get_output(0))
        to_nhwc.first_transpose = trt.Permutation([0, 2, 3, 1])

        # Add position embedding [1, 64, 64, hidden]
        pos_embed = weights["encoder.pos_embed"]
        pos_c = graph_ops.add_constant(
            network, (1, grid_size, grid_size, hidden), pos_embed, dtype=work_np_dtype
        )
        pos_sum = network.add_elementwise(
            to_nhwc.get_output(0), pos_c, trt.ElementWiseOperation.SUM
        )

        hidden_state = pos_sum.get_output(0)

        # Transformer layers
        for layer_idx in range(num_layers):
            w_prefix = f"encoder.layer{layer_idx}"
            use_global_attn = layer_idx in global_attn_indexes
            use_window = not use_global_attn

            # Pre-attention LayerNorm
            norm1_w = weights[f"{w_prefix}.norm1.weight"]
            norm1_b = weights[f"{w_prefix}.norm1.bias"]

            # Reshape to 2D for norm: [1, H, W, C] -> [H*W, C]
            reshape_2d = network.add_shuffle(hidden_state)
            reshape_2d.reshape_dims = (seq_len, hidden)

            normed = graph_ops.add_layer_norm(
                network,
                reshape_2d.get_output(0),
                hidden,
                norm1_w,
                norm1_b,
                eps_t,
                dtype=work_np_dtype,
            )

            # Reshape back to 4D for attention: [H*W, C] -> [1, H, W, C]
            normed_4d = network.add_shuffle(normed)
            normed_4d.reshape_dims = (1, grid_size, grid_size, hidden)

            if use_window:
                # Window partition + attention + unpartition
                attn_out_4d = self._build_windowed_attention(
                    network,
                    normed_4d.get_output(0),
                    weights,
                    w_prefix,
                    grid_size,
                    hidden,
                    num_heads,
                    head_dim,
                    window_size,
                    dtype=work_np_dtype,
                )
            else:
                # Global attention
                attn_out_4d = self._build_global_attention(
                    network,
                    normed_4d.get_output(0),
                    weights,
                    w_prefix,
                    grid_size,
                    hidden,
                    num_heads,
                    head_dim,
                    seq_len,
                    dtype=work_np_dtype,
                )

            # Residual
            res1 = network.add_elementwise(hidden_state, attn_out_4d, trt.ElementWiseOperation.SUM)

            # Post-attention MLP
            norm2_w = weights[f"{w_prefix}.norm2.weight"]
            norm2_b = weights[f"{w_prefix}.norm2.bias"]

            res1_2d = network.add_shuffle(res1.get_output(0))
            res1_2d.reshape_dims = (seq_len, hidden)

            normed2 = graph_ops.add_layer_norm(
                network, res1_2d.get_output(0), hidden, norm2_w, norm2_b, eps_t, dtype=work_np_dtype
            )

            # MLP: FC1 -> GELU -> FC2
            fc1 = graph_ops.add_matmul_rhs_constant(
                network, normed2, hidden, mlp_dim, weights[f"{w_prefix}.mlp.fc1.weight"]
            )
            fc1 = graph_ops.add_bias_sum(network, fc1, mlp_dim, weights[f"{w_prefix}.mlp.fc1.bias"])
            gelu = graph_ops.add_gelu_new(network, fc1)
            fc2 = graph_ops.add_matmul_rhs_constant(
                network, gelu, mlp_dim, hidden, weights[f"{w_prefix}.mlp.fc2.weight"]
            )
            fc2 = graph_ops.add_bias_sum(network, fc2, hidden, weights[f"{w_prefix}.mlp.fc2.bias"])

            # Reshape MLP output back to 4D
            fc2_4d = network.add_shuffle(fc2)
            fc2_4d.reshape_dims = (1, grid_size, grid_size, hidden)

            # Residual
            res2 = network.add_elementwise(
                res1.get_output(0), fc2_4d.get_output(0), trt.ElementWiseOperation.SUM
            )
            hidden_state = res2.get_output(0)

        # Neck: [1, H, W, hidden] -> [1, decoder_hidden, H, W]
        # Conv1: 1x1, hidden -> decoder_hidden
        to_nchw = network.add_shuffle(hidden_state)
        to_nchw.first_transpose = trt.Permutation([0, 3, 1, 2])

        neck_c1_w = weights["encoder.neck.conv1.weight"]
        neck_c1_b = weights.get(
            "encoder.neck.conv1.bias", np.zeros(decoder_hidden, dtype=np.float32)
        )
        neck_conv1 = network.add_convolution_nd(
            to_nchw.get_output(0),
            num_output_maps=decoder_hidden,
            kernel_shape=(1, 1),
            kernel=trt.Weights(np.ascontiguousarray(neck_c1_w, dtype=work_np_dtype)),
            bias=trt.Weights(np.ascontiguousarray(neck_c1_b, dtype=work_np_dtype)),
        )

        # LN1 (applied in NHWC domain): NCHW -> NHWC -> LN -> NCHW
        to_nhwc_n1 = network.add_shuffle(neck_conv1.get_output(0))
        to_nhwc_n1.first_transpose = trt.Permutation([0, 2, 3, 1])
        flat_n1 = network.add_shuffle(to_nhwc_n1.get_output(0))
        flat_n1.reshape_dims = (seq_len, decoder_hidden)
        ln1_out = graph_ops.add_layer_norm(
            network,
            flat_n1.get_output(0),
            decoder_hidden,
            weights["encoder.neck.ln1.weight"],
            weights["encoder.neck.ln1.bias"],
            eps_t,
            dtype=work_np_dtype,
        )
        unflat_n1 = network.add_shuffle(ln1_out)
        unflat_n1.reshape_dims = (1, grid_size, grid_size, decoder_hidden)
        to_nchw_n1 = network.add_shuffle(unflat_n1.get_output(0))
        to_nchw_n1.first_transpose = trt.Permutation([0, 3, 1, 2])

        # Conv2: 3x3, decoder_hidden -> decoder_hidden
        neck_c2_w = weights["encoder.neck.conv2.weight"]
        neck_c2_b = weights.get(
            "encoder.neck.conv2.bias", np.zeros(decoder_hidden, dtype=np.float32)
        )
        neck_conv2 = network.add_convolution_nd(
            to_nchw_n1.get_output(0),
            num_output_maps=decoder_hidden,
            kernel_shape=(3, 3),
            kernel=trt.Weights(np.ascontiguousarray(neck_c2_w, dtype=work_np_dtype)),
            bias=trt.Weights(np.ascontiguousarray(neck_c2_b, dtype=work_np_dtype)),
        )
        neck_conv2.padding_nd = (1, 1)

        # LN2
        to_nhwc_n2 = network.add_shuffle(neck_conv2.get_output(0))
        to_nhwc_n2.first_transpose = trt.Permutation([0, 2, 3, 1])
        flat_n2 = network.add_shuffle(to_nhwc_n2.get_output(0))
        flat_n2.reshape_dims = (seq_len, decoder_hidden)
        ln2_out = graph_ops.add_layer_norm(
            network,
            flat_n2.get_output(0),
            decoder_hidden,
            weights["encoder.neck.ln2.weight"],
            weights["encoder.neck.ln2.bias"],
            eps_t,
            dtype=work_np_dtype,
        )
        unflat_n2 = network.add_shuffle(ln2_out)
        unflat_n2.reshape_dims = (1, grid_size, grid_size, decoder_hidden)
        to_nchw_n2 = network.add_shuffle(unflat_n2.get_output(0))
        to_nchw_n2.first_transpose = trt.Permutation([0, 3, 1, 2])

        # Output: [1, decoder_hidden, 64, 64]
        output = to_nchw_n2.get_output(0)
        if output.dtype != trt.float32:
            output = network.add_cast(output, trt.float32).get_output(0)
        output.name = "image_embeddings"
        network.mark_output(output)

        if verbose:
            print(
                f"[trtmc build] Building SAM encoder engine "
                f"(image={image_size}x{image_size}, hidden={hidden}, "
                f"layers={num_layers}, precision={precision}) ...",
                file=sys.stderr,
            )

        plan = builder.build_serialized_network(network, trt_config)
        if plan is None:
            raise RuntimeError("TensorRT engine build failed for SAM encoder")
        return bytes(plan)

    def build_vision_engine(
        self,
        model_dir: str,
        config: ModelConfig,
        weights: WeightDict,
        *,
        precision: str = "fp32",
        verbose: bool = False,
    ) -> bytes | None:
        """Build TRT engine for SAM mask decoder.

        Matches HuggingFace SamMaskDecoder exactly:
        - Two-way transformer with post-LN residuals and PE injection
        - Image PE from SamPositionalEmbedding (shared_image_embedding)
        - Dense prompt = no_mask_embed broadcast over spatial dims
        - Layer 0 self-attention skips PE (skip_first_layer_pe=True)

        Inputs:
          image_embeddings [1, 256, 64, 64]
          sparse_prompt_embeddings [num_sparse, 256]
        Outputs:
          masks [num_masks, 256, 256]
          iou_scores [num_masks]
        """
        sam_cfg = config.raw.get("_sam_config", _resolve_sam_config(config.raw))
        decoder_hidden = sam_cfg["decoder_hidden_size"]
        num_heads = sam_cfg["decoder_num_heads"]
        head_dim = decoder_hidden // num_heads
        decoder_depth = sam_cfg["decoder_depth"]
        num_multimask = sam_cfg["num_multimask_outputs"]
        image_embedding_size = sam_cfg["image_embedding_size"]  # 64
        decoder_mlp_dim = sam_cfg["decoder_mlp_dim"]
        downsample_rate = sam_cfg.get("attention_downsample_rate", 2)
        cross_attn_dim = decoder_hidden // downsample_rate  # 128
        num_mask_outputs = num_multimask + 1  # 4
        work_np_dtype, work_trt_dtype = _precision_types(precision)

        # Sparse prompt: single point + padding point (pad=True when no box)
        num_output_tokens = 1 + num_mask_outputs  # 5
        num_sparse_fixed = 2  # point + padding (HF pads when no box)
        total_tokens = num_output_tokens + num_sparse_fixed  # 7

        img_seq = image_embedding_size * image_embedding_size  # 4096

        logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        trt_config = builder.create_builder_config()

        eps_t = graph_ops.add_constant(network, (1, 1), np.array([1e-6], dtype=np.float32))

        # --- Inputs ---
        image_embeddings_in = network.add_input(
            "image_embeddings",
            trt.float32,
            (1, decoder_hidden, image_embedding_size, image_embedding_size),
        )
        sparse_prompt = network.add_input(
            "sparse_prompt_embeddings", trt.float32, (num_sparse_fixed, decoder_hidden)
        )
        image_embeddings = image_embeddings_in
        sparse_prompt_work = sparse_prompt
        if work_trt_dtype != trt.float32:
            image_embeddings = network.add_cast(image_embeddings_in, work_trt_dtype).get_output(0)
            sparse_prompt_work = network.add_cast(sparse_prompt, work_trt_dtype).get_output(0)

        # --- Build image positional embeddings ---
        # HF: SamPositionalEmbedding generates PE over a [0,1] grid
        # Formula: coords = 2*grid - 1, B = coords @ positional_embedding,
        #          PE = cat(sin(2*pi*B), cos(2*pi*B))
        shared_pe = weights["prompt.shared_image_pe"]  # [2, num_pos_feats]
        # Build grid: [H, W, 2] with values in [0,1]
        grid = np.zeros((image_embedding_size, image_embedding_size, 2), dtype=np.float32)
        for gy in range(image_embedding_size):
            for gx in range(image_embedding_size):
                grid[gy, gx, 0] = (gx + 0.5) / image_embedding_size
                grid[gy, gx, 1] = (gy + 0.5) / image_embedding_size
        # coords = 2 * grid - 1
        coords = 2.0 * grid - 1.0  # [H, W, 2]
        coords_flat = coords.reshape(-1, 2)  # [H*W, 2]
        # B = coords @ positional_embedding  -> [H*W, num_pos_feats]
        B = coords_flat @ shared_pe  # [H*W, 128]
        B = 2.0 * np.pi * B
        image_pe_flat = np.concatenate([np.sin(B), np.cos(B)], axis=-1).astype(
            np.float32
        )  # [H*W, 256]
        image_pe_flat_c = graph_ops.add_constant(
            network, (img_seq, decoder_hidden), image_pe_flat, dtype=work_np_dtype
        )

        # --- Add dense prompt (no_mask_embed) to image embeddings ---
        # HF: image_embeddings = image_embeddings + dense_prompt_embeddings
        # dense_prompt = no_mask_embed [1, 256] broadcast to [1, 256, 64, 64]
        no_mask_embed = weights["prompt.no_mask_embed"]  # [256]
        no_mask_4d = no_mask_embed.reshape(1, decoder_hidden, 1, 1) * np.ones(
            (1, decoder_hidden, image_embedding_size, image_embedding_size), dtype=np.float32
        )
        no_mask_c = graph_ops.add_constant(
            network, no_mask_4d.shape, no_mask_4d, dtype=work_np_dtype
        )
        img_plus_dense = network.add_elementwise(
            image_embeddings, no_mask_c, trt.ElementWiseOperation.SUM
        ).get_output(0)

        # Flatten image to [H*W, C]
        img_flat_s = network.add_shuffle(img_plus_dense)
        img_flat_s.first_transpose = trt.Permutation([0, 2, 3, 1])
        img_flat_s.reshape_dims = (img_seq, decoder_hidden)

        # --- Build output tokens and concatenate with sparse prompt ---
        iou_token = weights["decoder.iou_token"]
        mask_tokens = weights["decoder.mask_tokens"]
        output_tokens_np = np.zeros((num_output_tokens, decoder_hidden), dtype=np.float32)
        output_tokens_np[0] = iou_token[:decoder_hidden]
        output_tokens_np[1 : 1 + num_mask_outputs] = mask_tokens[:num_mask_outputs]

        output_tokens = graph_ops.add_constant(
            network, (num_output_tokens, decoder_hidden), output_tokens_np, dtype=work_np_dtype
        )
        token_concat = network.add_concatenation([output_tokens, sparse_prompt_work])
        token_concat.axis = 0
        # point_embeddings = tokens (initial value, used as query_point_embedding)
        tokens_init = token_concat.get_output(0)  # [total_tokens, hidden]

        # HF two-way transformer:
        #   queries = point_embeddings (tokens)
        #   keys = image_embeddings
        # Both are updated through the layers.
        # query_point_embedding = tokens_init (constant PE added in attention)
        # key_point_embedding = image_pe (constant PE added in attention)

        queries = tokens_init  # [total_tokens, hidden]
        keys = img_flat_s.get_output(0)  # [img_seq, hidden]

        for layer_idx in range(decoder_depth):
            w_prefix = f"decoder.layer{layer_idx}"
            skip_first_layer_pe = layer_idx == 0

            # --- Self-attention on queries ---
            if skip_first_layer_pe:
                # Layer 0: Q=queries, K=queries, V=queries (no PE)
                sa_q = queries
                sa_k = queries
            else:
                # Q = queries + query_point_embedding
                sa_q = network.add_elementwise(
                    queries, tokens_init, trt.ElementWiseOperation.SUM
                ).get_output(0)
                sa_k = sa_q  # same as Q for self-attention

            sa_out = self._build_attention(
                network,
                sa_q,
                sa_k,
                total_tokens,
                total_tokens,
                decoder_hidden,
                num_heads,
                head_dim,
                weights,
                f"{w_prefix}.self_attn",
                value_input=queries,
            )

            # HF SAM: layer 0 replaces queries (no residual), other layers add residual
            if skip_first_layer_pe:
                queries = sa_out  # No residual for layer 0
            else:
                queries = network.add_elementwise(
                    queries, sa_out, trt.ElementWiseOperation.SUM
                ).get_output(0)
            # LN1 (post-norm)
            queries = graph_ops.add_layer_norm(
                network,
                queries,
                decoder_hidden,
                weights[f"{w_prefix}.norm1.weight"],
                weights[f"{w_prefix}.norm1.bias"],
                eps_t,
                dtype=work_np_dtype,
            )

            # --- Cross-attention: token-to-image ---
            # Q = queries + query_point_embedding (tokens_init)
            t2i_q = network.add_elementwise(
                queries, tokens_init, trt.ElementWiseOperation.SUM
            ).get_output(0)
            # K = keys + key_point_embedding (image_pe)
            t2i_k = network.add_elementwise(
                keys, image_pe_flat_c, trt.ElementWiseOperation.SUM
            ).get_output(0)

            t2i_out = self._build_attention(
                network,
                t2i_q,
                t2i_k,
                total_tokens,
                img_seq,
                decoder_hidden,
                num_heads,
                head_dim,
                weights,
                f"{w_prefix}.cross_t2i",
                proj_dim=cross_attn_dim,
                value_input=keys,
            )

            # Residual + LN2
            queries = network.add_elementwise(
                queries, t2i_out, trt.ElementWiseOperation.SUM
            ).get_output(0)
            queries = graph_ops.add_layer_norm(
                network,
                queries,
                decoder_hidden,
                weights[f"{w_prefix}.norm2.weight"],
                weights[f"{w_prefix}.norm2.bias"],
                eps_t,
                dtype=work_np_dtype,
            )

            # --- MLP on queries ---
            mlp_out = graph_ops.add_matmul_rhs_constant(
                network,
                queries,
                decoder_hidden,
                decoder_mlp_dim,
                weights[f"{w_prefix}.mlp.fc1.weight"],
            )
            mlp_out = graph_ops.add_bias_sum(
                network, mlp_out, decoder_mlp_dim, weights[f"{w_prefix}.mlp.fc1.bias"]
            )
            mlp_out = network.add_activation(mlp_out, trt.ActivationType.RELU).get_output(0)
            mlp_out = graph_ops.add_matmul_rhs_constant(
                network,
                mlp_out,
                decoder_mlp_dim,
                decoder_hidden,
                weights[f"{w_prefix}.mlp.fc2.weight"],
            )
            mlp_out = graph_ops.add_bias_sum(
                network, mlp_out, decoder_hidden, weights[f"{w_prefix}.mlp.fc2.bias"]
            )

            # Residual + LN3
            queries = network.add_elementwise(
                queries, mlp_out, trt.ElementWiseOperation.SUM
            ).get_output(0)
            queries = graph_ops.add_layer_norm(
                network,
                queries,
                decoder_hidden,
                weights[f"{w_prefix}.norm3.weight"],
                weights[f"{w_prefix}.norm3.bias"],
                eps_t,
                dtype=work_np_dtype,
            )

            # --- Cross-attention: image-to-token ---
            # Q = keys + key_point_embedding (image_pe)
            i2t_q = network.add_elementwise(
                keys, image_pe_flat_c, trt.ElementWiseOperation.SUM
            ).get_output(0)
            # K = queries + query_point_embedding (tokens_init)
            i2t_k = network.add_elementwise(
                queries, tokens_init, trt.ElementWiseOperation.SUM
            ).get_output(0)

            i2t_out = self._build_attention(
                network,
                i2t_q,
                i2t_k,
                img_seq,
                total_tokens,
                decoder_hidden,
                num_heads,
                head_dim,
                weights,
                f"{w_prefix}.cross_i2t",
                proj_dim=cross_attn_dim,
                value_input=queries,
            )

            # Residual on keys + LN4
            keys = network.add_elementwise(keys, i2t_out, trt.ElementWiseOperation.SUM).get_output(
                0
            )
            keys = graph_ops.add_layer_norm(
                network,
                keys,
                decoder_hidden,
                weights[f"{w_prefix}.norm4.weight"],
                weights[f"{w_prefix}.norm4.bias"],
                eps_t,
                dtype=work_np_dtype,
            )

        # --- Final cross-attention: token-to-image ---
        # Q = queries + point_embeddings (tokens_init)
        final_q = network.add_elementwise(
            queries, tokens_init, trt.ElementWiseOperation.SUM
        ).get_output(0)
        # K = keys + image_pe
        final_k = network.add_elementwise(
            keys, image_pe_flat_c, trt.ElementWiseOperation.SUM
        ).get_output(0)

        final_t2i = self._build_attention(
            network,
            final_q,
            final_k,
            total_tokens,
            img_seq,
            decoder_hidden,
            num_heads,
            head_dim,
            weights,
            "decoder.final_t2i",
            proj_dim=cross_attn_dim,
            value_input=keys,
        )

        queries = network.add_elementwise(
            queries, final_t2i, trt.ElementWiseOperation.SUM
        ).get_output(0)
        queries = graph_ops.add_layer_norm(
            network,
            queries,
            decoder_hidden,
            weights["decoder.final_norm.weight"],
            weights["decoder.final_norm.bias"],
            eps_t,
            dtype=work_np_dtype,
        )

        # --- Extract mask tokens and IoU token ---
        iou_tok_slice = network.add_slice(
            queries, start=(0, 0), shape=(1, decoder_hidden), stride=(1, 1)
        )
        iou_token_out = iou_tok_slice.get_output(0)

        mask_tok_slice = network.add_slice(
            queries, start=(1, 0), shape=(num_mask_outputs, decoder_hidden), stride=(1, 1)
        )
        mask_tokens_out = mask_tok_slice.get_output(0)

        # --- Upscale image embeddings ---
        # keys: [img_seq, C] -> [1, C, H, W]
        img_up_s = network.add_shuffle(keys)
        img_up_s.reshape_dims = (1, image_embedding_size, image_embedding_size, decoder_hidden)
        img_up_t = network.add_shuffle(img_up_s.get_output(0))
        img_up_t.first_transpose = trt.Permutation([0, 3, 1, 2])

        # ConvTranspose2d 1: [1, 256, 64, 64] -> [1, 64, 128, 128]
        up1_out_ch = decoder_hidden // 4
        up1_w = weights["decoder.upscale.conv1.weight"]
        up1_b = weights.get("decoder.upscale.conv1.bias", np.zeros(up1_out_ch, dtype=np.float32))
        up1_conv = network.add_deconvolution_nd(
            img_up_t.get_output(0),
            num_output_maps=up1_out_ch,
            kernel_shape=(2, 2),
            kernel=trt.Weights(np.ascontiguousarray(up1_w, dtype=work_np_dtype)),
            bias=trt.Weights(np.ascontiguousarray(up1_b, dtype=work_np_dtype)),
        )
        up1_conv.stride_nd = (2, 2)

        # SamLayerNorm (channels_first): permute NCHW->NHWC, LN, permute back
        up1_nhwc = network.add_shuffle(up1_conv.get_output(0))
        up1_nhwc.first_transpose = trt.Permutation([0, 2, 3, 1])
        up1_flat = network.add_shuffle(up1_nhwc.get_output(0))
        up1_flat.reshape_dims = (128 * 128, up1_out_ch)
        up1_ln = graph_ops.add_layer_norm(
            network,
            up1_flat.get_output(0),
            up1_out_ch,
            weights["decoder.upscale.ln.weight"],
            weights["decoder.upscale.ln.bias"],
            eps_t,
            dtype=work_np_dtype,
        )
        up1_unflat = network.add_shuffle(up1_ln)
        up1_unflat.reshape_dims = (1, 128, 128, up1_out_ch)
        up1_nchw = network.add_shuffle(up1_unflat.get_output(0))
        up1_nchw.first_transpose = trt.Permutation([0, 3, 1, 2])

        # GELU activation (HF: self.activation = nn.GELU())
        # GELU is element-wise, apply on NCHW then feed to conv2
        up1_act = network.add_activation(
            up1_nchw.get_output(0), trt.ActivationType.GELU_ERF
        ).get_output(0)

        # ConvTranspose2d 2: [1, 64, 128, 128] -> [1, 32, 256, 256]
        up2_out_ch = decoder_hidden // 8
        up2_w = weights["decoder.upscale.conv2.weight"]
        up2_b = weights.get("decoder.upscale.conv2.bias", np.zeros(up2_out_ch, dtype=np.float32))
        up2_conv = network.add_deconvolution_nd(
            up1_act,
            num_output_maps=up2_out_ch,
            kernel_shape=(2, 2),
            kernel=trt.Weights(np.ascontiguousarray(up2_w, dtype=work_np_dtype)),
            bias=trt.Weights(np.ascontiguousarray(up2_b, dtype=work_np_dtype)),
        )
        up2_conv.stride_nd = (2, 2)

        # GELU on conv2 output (NCHW)
        up2_act = network.add_activation(
            up2_conv.get_output(0), trt.ActivationType.GELU_ERF
        ).get_output(0)

        # Permute NCHW -> NHWC -> flatten to [H*W, C] for dot product
        # This is critical: the dot product expects [pixel, channel] layout
        up2_nhwc = network.add_shuffle(up2_act)
        up2_nhwc.first_transpose = trt.Permutation([0, 2, 3, 1])
        up2_flat = network.add_shuffle(up2_nhwc.get_output(0))
        up2_flat.reshape_dims = (256 * 256, up2_out_ch)
        up2_features = up2_flat.get_output(0)  # [65536, 32]

        # --- Generate masks via hypernetwork MLPs ---
        mask_outputs = []
        for i in range(num_mask_outputs):
            mt_slice = network.add_slice(
                mask_tokens_out, start=(i, 0), shape=(1, decoder_hidden), stride=(1, 1)
            )
            mt = mt_slice.get_output(0)

            for j in range(3):
                w_key = f"decoder.hyper_mlp.{i}.{j}.weight"
                b_key = f"decoder.hyper_mlp.{i}.{j}.bias"
                out_dim = weights[b_key].shape[0]
                in_dim = weights[w_key].shape[0]
                mt = graph_ops.add_matmul_rhs_constant(network, mt, in_dim, out_dim, weights[w_key])
                mt = graph_ops.add_bias_sum(network, mt, out_dim, weights[b_key])
                if j < 2:
                    mt = network.add_activation(mt, trt.ActivationType.RELU).get_output(0)

            # Dot product: [256*256, up2_out_ch] @ [up2_out_ch, 1]
            mt_t = network.add_shuffle(mt)
            mt_t.first_transpose = trt.Permutation([1, 0])
            mask_logits = network.add_matrix_multiply(
                up2_features, trt.MatrixOperation.NONE, mt_t.get_output(0), trt.MatrixOperation.NONE
            )
            mask_2d = network.add_shuffle(mask_logits.get_output(0))
            mask_2d.reshape_dims = (1, 256, 256)
            mask_outputs.append(mask_2d.get_output(0))

        masks_concat = network.add_concatenation(mask_outputs)
        masks_concat.axis = 0
        masks = masks_concat.get_output(0)
        cast_masks = network.add_cast(masks, trt.float32)
        masks_out = cast_masks.get_output(0)
        masks_out.name = "masks"
        network.mark_output(masks_out)

        # --- IoU prediction ---
        iou = iou_token_out
        for j in range(3):
            w_key = f"decoder.iou_head.{j}.weight"
            b_key = f"decoder.iou_head.{j}.bias"
            out_dim = weights[b_key].shape[0]
            in_dim = weights[w_key].shape[0]
            iou = graph_ops.add_matmul_rhs_constant(network, iou, in_dim, out_dim, weights[w_key])
            iou = graph_ops.add_bias_sum(network, iou, out_dim, weights[b_key])
            if j < 2:
                iou = network.add_activation(iou, trt.ActivationType.RELU).get_output(0)

        iou_out = network.add_shuffle(iou)
        iou_out.reshape_dims = (num_mask_outputs,)
        iou_scores = iou_out.get_output(0)
        cast_iou = network.add_cast(iou_scores, trt.float32)
        iou_out = cast_iou.get_output(0)
        iou_out.name = "iou_scores"
        network.mark_output(iou_out)

        if verbose:
            print(
                f"[trtmc build] Building SAM mask decoder engine "
                f"(decoder_hidden={decoder_hidden}, depth={decoder_depth}, "
                f"precision={precision}) ...",
                file=sys.stderr,
            )

        plan = builder.build_serialized_network(network, trt_config)
        if plan is None:
            raise RuntimeError("TensorRT engine build failed for SAM mask decoder")
        return bytes(plan)

    def get_segmentation_config(self, config: ModelConfig) -> dict:
        """Return SAM config for bundle config.json."""
        sam_cfg = config.raw.get("_sam_config", _resolve_sam_config(config.raw))
        result = {
            "sam_image_size": sam_cfg["image_size"],
            "sam_patch_size": sam_cfg["patch_size"],
            "sam_hidden_size": sam_cfg["hidden_size"],
            "sam_decoder_hidden_size": sam_cfg["decoder_hidden_size"],
            "sam_image_embedding_size": sam_cfg["image_embedding_size"],
            "sam_num_multimask_outputs": sam_cfg["num_multimask_outputs"],
            "sam_num_mask_outputs": sam_cfg["num_multimask_outputs"] + 1,
            "input_image_h": sam_cfg["image_size"],
            "input_image_w": sam_cfg["image_size"],
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
        }
        # Ship point embeddings and shared_image_pe for C++ prompt encoding
        pe_0 = sam_cfg.get("_point_embed_0")
        pe_1 = sam_cfg.get("_point_embed_1")
        pe_na = sam_cfg.get("_not_a_point_embed")
        shared = sam_cfg.get("_shared_image_pe")
        if pe_0 is not None:
            result["sam_point_embed_0"] = pe_0
        if pe_1 is not None:
            result["sam_point_embed_1"] = pe_1
        if pe_na is not None:
            result["sam_not_a_point_embed"] = pe_na
        if shared is not None:
            result["sam_shared_image_pe"] = shared
        return result

    # --- Internal helpers ---

    @staticmethod
    def _build_attention(
        network,
        q_input,
        kv_input,
        q_seq,
        kv_seq,
        hidden,
        num_heads,
        head_dim,
        weights,
        prefix,
        proj_dim=None,
        value_input=None,
    ):
        """Build a standard multi-head attention block.

        Args:
            proj_dim: Internal projection dimension. Defaults to hidden.
                      SAM cross-attention uses hidden // downsample_rate.
            value_input: Separate input for V projection. If None, uses kv_input.
                        HF SAM adds PE to Q/K but not V.
        """
        proj = proj_dim if proj_dim is not None else hidden
        p_head_dim = proj // num_heads
        attn_scale = 1.0 / np.sqrt(max(p_head_dim, 1))

        v_source = value_input if value_input is not None else kv_input

        q = graph_ops.add_matmul_rhs_constant(
            network, q_input, hidden, proj, weights[f"{prefix}.q.weight"]
        )
        q = graph_ops.add_bias_sum(network, q, proj, weights[f"{prefix}.q.bias"])

        k = graph_ops.add_matmul_rhs_constant(
            network, kv_input, hidden, proj, weights[f"{prefix}.k.weight"]
        )
        k = graph_ops.add_bias_sum(network, k, proj, weights[f"{prefix}.k.bias"])

        v = graph_ops.add_matmul_rhs_constant(
            network, v_source, hidden, proj, weights[f"{prefix}.v.weight"]
        )
        v = graph_ops.add_bias_sum(network, v, proj, weights[f"{prefix}.v.bias"])

        ctx_flat = graph_ops.add_attention_from_rows(
            network,
            q,
            k,
            v,
            num_heads=num_heads,
            head_dim=p_head_dim,
            q_seq=q_seq,
            kv_seq=kv_seq,
            scale=attn_scale,
        )

        out = graph_ops.add_matmul_rhs_constant(
            network, ctx_flat, proj, hidden, weights[f"{prefix}.o.weight"]
        )
        out = graph_ops.add_bias_sum(network, out, hidden, weights[f"{prefix}.o.bias"])
        return out

    @staticmethod
    def _build_windowed_attention(
        network,
        inp_4d,
        weights,
        w_prefix,
        grid_size,
        hidden,
        num_heads,
        head_dim,
        window_size,
        dtype=np.float32,
    ):
        """Partition SAM features into local windows and restore their layout."""
        padded_size = ((grid_size + window_size - 1) // window_size) * window_size
        window_count = padded_size // window_size
        pad_size = padded_size - grid_size

        padded = inp_4d
        if pad_size:
            pad_right = graph_ops.add_constant(
                network,
                (1, grid_size, pad_size, hidden),
                np.zeros((1, grid_size, pad_size, hidden), dtype=dtype),
                dtype=dtype,
            )
            concat_width = network.add_concatenation([padded, pad_right])
            concat_width.axis = 2

            pad_bottom = graph_ops.add_constant(
                network,
                (1, pad_size, padded_size, hidden),
                np.zeros((1, pad_size, padded_size, hidden), dtype=dtype),
                dtype=dtype,
            )
            concat_height = network.add_concatenation([concat_width.get_output(0), pad_bottom])
            concat_height.axis = 1
            padded = concat_height.get_output(0)

        partition = network.add_shuffle(padded)
        partition.reshape_dims = (
            1,
            window_count,
            window_size,
            window_count,
            window_size,
            hidden,
        )
        partition.second_transpose = trt.Permutation([0, 1, 3, 2, 4, 5])

        windows = network.add_shuffle(partition.get_output(0))
        windows.reshape_dims = (
            window_count * window_count,
            window_size,
            window_size,
            hidden,
        )
        attended = _SamModel._build_spatial_attention(
            network,
            windows.get_output(0),
            weights,
            w_prefix,
            batch_size=window_count * window_count,
            spatial_size=window_size,
            hidden=hidden,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
        )

        grouped = network.add_shuffle(attended)
        grouped.reshape_dims = (
            1,
            window_count,
            window_count,
            window_size,
            window_size,
            hidden,
        )
        grouped.second_transpose = trt.Permutation([0, 1, 3, 2, 4, 5])

        unpartitioned = network.add_shuffle(grouped.get_output(0))
        unpartitioned.reshape_dims = (1, padded_size, padded_size, hidden)
        if not pad_size:
            return unpartitioned.get_output(0)

        cropped = network.add_slice(
            unpartitioned.get_output(0),
            start=(0, 0, 0, 0),
            shape=(1, grid_size, grid_size, hidden),
            stride=(1, 1, 1, 1),
        )
        return cropped.get_output(0)

    @staticmethod
    def _build_spatial_attention(
        network,
        inp_4d,
        weights,
        w_prefix,
        *,
        batch_size,
        spatial_size,
        hidden,
        num_heads,
        head_dim,
        dtype=np.float32,
    ):
        """Build SAM attention over one or more equally sized spatial grids."""
        seq_len = spatial_size * spatial_size
        attn_scale = 1.0 / np.sqrt(max(head_dim, 1))

        # Flatten to rows for the projection layers.
        flat = network.add_shuffle(inp_4d)
        flat.reshape_dims = (batch_size * seq_len, hidden)

        q = graph_ops.add_matmul_rhs_constant(
            network, flat.get_output(0), hidden, hidden, weights[f"{w_prefix}.attn.q.weight"]
        )
        q = graph_ops.add_bias_sum(network, q, hidden, weights[f"{w_prefix}.attn.q.bias"])

        k = graph_ops.add_matmul_rhs_constant(
            network, flat.get_output(0), hidden, hidden, weights[f"{w_prefix}.attn.k.weight"]
        )
        k = graph_ops.add_bias_sum(network, k, hidden, weights[f"{w_prefix}.attn.k.bias"])

        v = graph_ops.add_matmul_rhs_constant(
            network, flat.get_output(0), hidden, hidden, weights[f"{w_prefix}.attn.v.weight"]
        )
        v = graph_ops.add_bias_sum(network, v, hidden, weights[f"{w_prefix}.attn.v.bias"])

        q_h = network.add_shuffle(q)
        q_h.reshape_dims = (batch_size, seq_len, num_heads, head_dim)
        q_h.second_transpose = trt.Permutation([0, 2, 1, 3])

        k_h = network.add_shuffle(k)
        k_h.reshape_dims = (batch_size, seq_len, num_heads, head_dim)
        k_h.second_transpose = trt.Permutation([0, 2, 1, 3])

        v_h = network.add_shuffle(v)
        v_h.reshape_dims = (batch_size, seq_len, num_heads, head_dim)
        v_h.second_transpose = trt.Permutation([0, 2, 1, 3])

        score = network.add_matrix_multiply(
            q_h.get_output(0),
            trt.MatrixOperation.NONE,
            k_h.get_output(0),
            trt.MatrixOperation.TRANSPOSE,
        )
        scale_c = graph_ops.add_constant(
            network, (1, 1, 1, 1), np.array([attn_scale], dtype=dtype), dtype=dtype
        )
        scaled = network.add_elementwise(
            score.get_output(0), scale_c, trt.ElementWiseOperation.PROD
        )

        # HF SAM's decomposed relative-position bias depends on each query.
        if f"{w_prefix}.attn.rel_pos_h" in weights:
            rel_pos_h = weights[f"{w_prefix}.attn.rel_pos_h"]
            rel_pos_w = weights[f"{w_prefix}.attn.rel_pos_w"]
            rp_h = _SamModel._get_rel_pos(spatial_size, spatial_size, rel_pos_h)
            rp_w = _SamModel._get_rel_pos(spatial_size, spatial_size, rel_pos_w)

            q_4d = network.add_shuffle(q_h.get_output(0))
            q_4d.reshape_dims = (batch_size, num_heads, spatial_size, spatial_size, head_dim)

            q_perm_h = network.add_shuffle(q_4d.get_output(0))
            q_perm_h.first_transpose = trt.Permutation([2, 0, 1, 3, 4])
            q_perm_h.reshape_dims = (
                spatial_size,
                batch_size * num_heads * spatial_size,
                head_dim,
            )

            rp_h_t = rp_h.transpose(0, 2, 1).astype(np.float32)
            rp_h_c = graph_ops.add_constant(network, rp_h_t.shape, rp_h_t, dtype=dtype)

            rel_h_mm = network.add_matrix_multiply(
                q_perm_h.get_output(0), trt.MatrixOperation.NONE, rp_h_c, trt.MatrixOperation.NONE
            )
            rel_h_4d = network.add_shuffle(rel_h_mm.get_output(0))
            rel_h_4d.reshape_dims = (
                spatial_size,
                batch_size,
                num_heads,
                spatial_size,
                spatial_size,
            )
            rel_h_4d.second_transpose = trt.Permutation([1, 2, 0, 3, 4])

            q_perm_w = network.add_shuffle(q_4d.get_output(0))
            q_perm_w.first_transpose = trt.Permutation([3, 0, 1, 2, 4])
            q_perm_w.reshape_dims = (
                spatial_size,
                batch_size * num_heads * spatial_size,
                head_dim,
            )

            rp_w_t = rp_w.transpose(0, 2, 1).astype(np.float32)
            rp_w_c = graph_ops.add_constant(network, rp_w_t.shape, rp_w_t, dtype=dtype)

            rel_w_mm = network.add_matrix_multiply(
                q_perm_w.get_output(0), trt.MatrixOperation.NONE, rp_w_c, trt.MatrixOperation.NONE
            )
            rel_w_4d = network.add_shuffle(rel_w_mm.get_output(0))
            rel_w_4d.reshape_dims = (
                spatial_size,
                batch_size,
                num_heads,
                spatial_size,
                spatial_size,
            )
            rel_w_4d.second_transpose = trt.Permutation([1, 2, 3, 0, 4])

            rel_h_5d = network.add_shuffle(rel_h_4d.get_output(0))
            rel_h_5d.reshape_dims = (
                batch_size,
                num_heads,
                spatial_size,
                spatial_size,
                spatial_size,
                1,
            )
            rel_w_5d = network.add_shuffle(rel_w_4d.get_output(0))
            rel_w_5d.reshape_dims = (
                batch_size,
                num_heads,
                spatial_size,
                spatial_size,
                1,
                spatial_size,
            )

            rel_bias = network.add_elementwise(
                rel_h_5d.get_output(0), rel_w_5d.get_output(0), trt.ElementWiseOperation.SUM
            )
            rel_bias_flat = network.add_shuffle(rel_bias.get_output(0))
            rel_bias_flat.reshape_dims = (batch_size, num_heads, seq_len, seq_len)

            scaled = network.add_elementwise(
                scaled.get_output(0), rel_bias_flat.get_output(0), trt.ElementWiseOperation.SUM
            )

        softmax = network.add_softmax(scaled.get_output(0))
        softmax.axes = 1 << 3

        ctx = network.add_matrix_multiply(
            softmax.get_output(0),
            trt.MatrixOperation.NONE,
            v_h.get_output(0),
            trt.MatrixOperation.NONE,
        )

        ctx_flat = network.add_shuffle(ctx.get_output(0))
        ctx_flat.first_transpose = trt.Permutation([0, 2, 1, 3])
        ctx_flat.reshape_dims = (batch_size * seq_len, hidden)

        out = graph_ops.add_matmul_rhs_constant(
            network, ctx_flat.get_output(0), hidden, hidden, weights[f"{w_prefix}.attn.o.weight"]
        )
        out = graph_ops.add_bias_sum(network, out, hidden, weights[f"{w_prefix}.attn.o.bias"])

        out_4d = network.add_shuffle(out)
        out_4d.reshape_dims = (batch_size, spatial_size, spatial_size, hidden)
        return out_4d.get_output(0)

    @staticmethod
    def _build_global_attention(
        network,
        inp_4d,
        weights,
        w_prefix,
        grid_size,
        hidden,
        num_heads,
        head_dim,
        seq_len,
        dtype=np.float32,
    ):
        """Build SAM's four full-image attention layers."""
        del seq_len
        return _SamModel._build_spatial_attention(
            network,
            inp_4d,
            weights,
            w_prefix,
            batch_size=1,
            spatial_size=grid_size,
            hidden=hidden,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
        )

    @staticmethod
    def _get_rel_pos(q_size, k_size, rel_pos):
        """Get relative positional embeddings, matching HF SAM's get_rel_pos.

        Args:
            q_size: query spatial size
            k_size: key spatial size
            rel_pos: [L, head_dim] numpy array
        Returns:
            [q_size, k_size, head_dim] numpy array
        """
        max_rel_dist = int(2 * max(q_size, k_size) - 1)
        # Interpolate rel_pos to max_rel_dist if needed
        if rel_pos.shape[0] != max_rel_dist:
            from scipy.interpolate import interp1d

            x_old = np.linspace(0, 1, rel_pos.shape[0])
            x_new = np.linspace(0, 1, max_rel_dist)
            f = interp1d(x_old, rel_pos, axis=0, kind="linear")
            rel_pos_resized = f(x_new).astype(np.float32)
        else:
            rel_pos_resized = rel_pos

        # Compute relative coordinate indices
        q_coords = np.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
        k_coords = np.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
        relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
        indices = relative_coords.astype(np.int64)
        indices = np.clip(indices, 0, max_rel_dist - 1)
        return rel_pos_resized[indices]  # [q_size, k_size, head_dim]

    @staticmethod
    def _compute_decomposed_rel_pos_bias(
        rel_pos_h,
        rel_pos_w,
        height,
        width,
        num_heads,
        head_dim,
    ):
        """Compute decomposed relative position bias as a numpy constant.

        Matches HF SAM's get_decomposed_rel_pos.

        Args:
            rel_pos_h: [2*input_h-1, head_dim]
            rel_pos_w: [2*input_w-1, head_dim]
            height, width: spatial dimensions of the attention
            num_heads, head_dim: attention config
        Returns:
            [num_heads, height*width, height*width] numpy array
        """
        seq_len = height * width
        # The query is reshaped as [num_heads, H, W, head_dim] in HF
        # We need to compute: rel_h[h_q, h_k] + rel_w[w_q, w_k] for each (q,k) pair
        # This produces a [H, W, H, W] bias that we reshape to [seq, seq]

        # For each head: bias[h_q*W+w_q, h_k*W+w_k] = rp_h[h_q, h_k, :] . q[h_q, w_q, :] + ...
        # But we don't have access to q at build time.
        #
        # Actually, looking at HF code more carefully:
        # rel_h = einsum("bhwc,hkc->bhwk", reshaped_query, rp_h)
        # This means the bias DEPENDS on the query values, so it can't be precomputed.
        #
        # For TRT, we need to implement this as graph ops. The decomposed rel pos
        # bias is query-dependent, so we need to compute it at runtime.
        #
        # Alternative: we can add the rel_pos computation directly in the attention
        # graph. This requires reshaping Q to [num_heads, H, W, head_dim], then
        # performing two einsum-like matmuls with the rel_pos tables.
        #
        # For now, return zeros and handle rel_pos in the attention builder.
        return np.zeros((num_heads, seq_len, seq_len), dtype=np.float32)


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one SAM prompted-segmentation bundle."""
    if request.max_sequence_length is not None:
        raise NotImplementedError("sam does not support max_sequence_length")

    if request.image_height is not None:
        raise NotImplementedError("sam does not support image_height")

    if request.image_width is not None:
        raise NotImplementedError("sam does not support image_width")

    if request.video_num_frames is not None:
        raise NotImplementedError("sam does not support video_num_frames")

    if request.max_batch_size != 1:
        raise NotImplementedError("sam does not support max_batch_size")

    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task != "prompted_segmentation":
        raise ValueError("sam supports only task=prompted_segmentation")
    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    if str(config.model_type).lower() != "sam":
        raise ValueError(f"SAM does not support model_type={config.model_type!r}")
    if request.quantization not in {None, "none"} or request.fp32_layers:
        raise NotImplementedError("SAM does not support quantization or mixed-precision layers")
    parallel = ParallelConfig(tp_size=int(request.tensor_parallel_size))
    parallel.validate()
    model = _SamModel()
    weights = model.load_weights(str(model_dir), config)
    writer.set_header(family="sam", task=request.task, backend="trt")
    if parallel.enabled:
        for rank in range(parallel.tp_size):
            plan = model.build_engine(
                config,
                weights,
                1,
                precision=request.precision,
                quant_ctx=None,
                verbose=request.verbose,
                parallel_config=parallel.for_rank(rank),
            )
            writer.add_bytes(f"engine.rank{rank}.plan", plan)
    else:
        writer.add_bytes(
            "engine.plan",
            model.build_engine(
                config,
                weights,
                1,
                precision=request.precision,
                quant_ctx=None,
                verbose=request.verbose,
                parallel_config=parallel,
            ),
        )
    decoder = model.build_vision_engine(
        str(model_dir), config, weights, precision=request.precision, verbose=request.verbose
    )
    if decoder is None:
        raise RuntimeError("SAM mask decoder build returned no engine")
    writer.add_bytes("decoder.plan", decoder)
    source = model.get_segmentation_config(config)
    runtime = {
        "image_size": source["sam_image_size"],
        "image_embedding_size": source["sam_image_embedding_size"],
        "decoder_hidden_size": source["sam_decoder_hidden_size"],
        "num_mask_outputs": source["sam_num_mask_outputs"],
        "num_multimask_outputs": source["sam_num_multimask_outputs"],
        "image_mean": source["image_mean"],
        "image_std": source["image_std"],
        "point_embed_bg": source["sam_point_embed_0"],
        "point_embed_fg": source["sam_point_embed_1"],
        "not_a_point_embed": source["sam_not_a_point_embed"],
        "shared_image_pe": source["sam_shared_image_pe"],
        "tensor_parallel_size": parallel.tp_size,
    }
    writer.add_json("runtime.json", runtime)
