"""Tensor-parallel timm ViT image-classification builder.

timm ViT attention remains replicated. The MLP path is tensor-parallel:
FC1 columns are sharded, FC2 rows are sharded, and a TensorRT distributed
ALL_REDUCE restores the full residual before the next layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import sys

import numpy as np
from tensorrt_model_connect import graph_ops, trt_compat

from ...checkpoint_mapper import _transpose_2d
from ...parallel_config import add_all_reduce_sum, normalize_parallel_config
from .plugin import _resolve_vit_config

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from ...checkpoint_mapper import WeightDict
    from ...config import ModelConfig
    from ...parallel_config import ParallelConfig


def _validate_timm_vit_tp(config: "ModelConfig", parallel: "ParallelConfig") -> None:
    parallel.validate()
    if not parallel.enabled:
        return
    if parallel.rank < 0:
        raise ValueError("timm_vit tensor-parallel build requires a concrete rank")
    vit_cfg = config.raw.get("_timm_vit_config") or _resolve_vit_config(config.raw)
    mlp_hidden = int(vit_cfg["mlp_hidden"])
    if mlp_hidden % parallel.tp_size != 0:
        raise ValueError(
            "timm_vit tensor-parallel MLP requires mlp_hidden divisible by tp_size "
            f"({mlp_hidden} vs {parallel.tp_size})")


def _slice_mlp_columns(
    arr: np.ndarray,
    mlp_hidden: int,
    parallel: "ParallelConfig",
) -> np.ndarray:
    local = mlp_hidden // parallel.tp_size
    start = parallel.rank * local
    end = start + local
    return np.ascontiguousarray(arr[..., start:end])


def _slice_mlp_rows(
    arr: np.ndarray,
    mlp_hidden: int,
    parallel: "ParallelConfig",
) -> np.ndarray:
    local = mlp_hidden // parallel.tp_size
    start = parallel.rank * local
    end = start + local
    return np.ascontiguousarray(arr[start:end, ...])


def build_timm_vit_tp_engine(
    config: "ModelConfig",
    weights: "WeightDict",
    max_cache_length: int,
    *,
    precision: str = "fp32",
    quant_ctx=None,
    verbose: bool = False,
    parallel_config=None,
) -> bytes:
    """Build one rank-local timm ViT engine with tensor-parallel MLPs."""
    del max_cache_length
    if quant_ctx is not None:
        raise ValueError("timm_vit tensor-parallel builds do not support quantization")
    if precision not in ("fp32",):
        raise ValueError("timm_vit tensor-parallel path currently supports precision='fp32'")

    parallel = normalize_parallel_config(parallel_config)
    if not parallel.enabled:
        raise ValueError("timm_vit tensor-parallel builder requires an enabled parallel config")
    _validate_timm_vit_tp(config, parallel)

    vit_cfg = config.raw.get("_timm_vit_config") or _resolve_vit_config(config.raw)
    image_h = vit_cfg["image_size_h"]
    image_w = vit_cfg["image_size_w"]
    patch_h = vit_cfg["patch_h"]
    patch_w = vit_cfg["patch_w"]
    hidden_size = vit_cfg["hidden"]
    depth = vit_cfg["depth"]
    num_heads = vit_cfg["heads"]
    mlp_hidden = vit_cfg["mlp_hidden"]
    local_mlp_hidden = mlp_hidden // parallel.tp_size
    num_classes = vit_cfg["num_classes"]
    eps_val = vit_cfg["eps"]

    if image_h % patch_h != 0 or image_w % patch_w != 0:
        raise ValueError(
            f"image_size {image_h}x{image_w} must be divisible by patch "
            f"{patch_h}x{patch_w}"
        )

    grid_h = image_h // patch_h
    grid_w = image_w // patch_w
    num_patches = grid_h * grid_w
    seq_len = num_patches + 1

    if verbose:
        print(
            "[trtmc build] timm_vit TP: "
            f"rank={parallel.rank}/{parallel.tp_size}, image={image_h}x{image_w}, "
            f"patch={patch_h}x{patch_w}, tokens={seq_len}, hidden={hidden_size}, "
            f"layers={depth}, heads={num_heads}, mlp_local={local_mlp_hidden}, "
            f"classes={num_classes}",
            file=sys.stderr,
        )

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        trt_compat.network_creation_flags(
            strongly_typed=True,
            explicit_batch=True,
        )
    )
    trt_config = builder.create_builder_config()
    trt_config.avg_timing_iterations = 8
    trt_config.max_aux_streams = 0
    trt_config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    pixel_values = network.add_input(
        "pixel_values", trt.float32, (1, 3, image_h, image_w))

    patch = network.add_convolution_nd(
        pixel_values,
        num_output_maps=hidden_size,
        kernel_shape=(patch_h, patch_w),
        kernel=trt.Weights(np.ascontiguousarray(
            weights["patch_embed.proj.weight"], dtype=np.float32)),
        bias=trt.Weights(np.ascontiguousarray(
            weights["patch_embed.proj.bias"], dtype=np.float32)),
    )
    patch.stride_nd = (patch_h, patch_w)

    patches_nhwc = network.add_shuffle(patch.get_output(0))
    patches_nhwc.first_transpose = (0, 2, 3, 1)
    patches_nhwc.reshape_dims = (1, num_patches, hidden_size)
    hidden = patches_nhwc.get_output(0)

    cls_token = np.ascontiguousarray(
        weights["cls_token"].reshape(1, 1, hidden_size), dtype=np.float32)
    cls_const = graph_ops.add_constant(
        network, (1, 1, hidden_size), cls_token, dtype=np.float32)
    cat = network.add_concatenation([cls_const, hidden])
    cat.axis = 1
    hidden = cat.get_output(0)

    pos_embed = np.ascontiguousarray(
        weights["pos_embed"].reshape(1, seq_len, hidden_size), dtype=np.float32)
    pos_const = graph_ops.add_constant(
        network, (1, seq_len, hidden_size), pos_embed, dtype=np.float32)
    hidden = network.add_elementwise(
        hidden, pos_const, trt.ElementWiseOperation.SUM).get_output(0)

    for layer_idx in range(depth):
        prefix = f"blocks.{layer_idx}"
        norm1 = graph_ops.add_layer_norm_native(
            network,
            hidden,
            hidden_size,
            weights[f"{prefix}.norm1.weight"].astype(np.float32),
            weights[f"{prefix}.norm1.bias"].astype(np.float32),
            eps_val,
        )

        qkv_w = weights[f"{prefix}.attn.qkv.weight"].astype(np.float32)
        q_w, k_w, v_w = np.split(qkv_w, 3, axis=1)
        qkv_b = weights.get(f"{prefix}.attn.qkv.bias")
        q_b = k_b = v_b = None
        if qkv_b is not None:
            q_b, k_b, v_b = np.split(qkv_b.astype(np.float32), 3)

        q = graph_ops.add_matmul_rhs_constant(
            network,
            norm1,
            hidden_size,
            hidden_size,
            q_w,
        )
        k = graph_ops.add_matmul_rhs_constant(
            network,
            norm1,
            hidden_size,
            hidden_size,
            k_w,
        )
        v = graph_ops.add_matmul_rhs_constant(
            network,
            norm1,
            hidden_size,
            hidden_size,
            v_w,
        )
        if q_b is not None:
            q = graph_ops.add_bias_sum(network, q, hidden_size, q_b)
            k = graph_ops.add_bias_sum(network, k, hidden_size, k_b)
            v = graph_ops.add_bias_sum(network, v, hidden_size, v_b)

        head_dim = hidden_size // num_heads

        def to_heads(x: trt.ITensor) -> trt.ITensor:
            heads = network.add_shuffle(x)
            heads.reshape_dims = (1, seq_len, num_heads, head_dim)
            heads.second_transpose = trt.Permutation([0, 2, 1, 3])
            return heads.get_output(0)

        q = to_heads(q)
        k = to_heads(k)
        v = to_heads(v)
        scale = graph_ops.add_constant(
            network,
            (1, 1, 1, 1),
            np.array([[[[1.0 / np.sqrt(head_dim)]]]], dtype=np.float32),
            dtype=np.float32,
        )
        q_scaled = network.add_elementwise(
            q, scale, trt.ElementWiseOperation.PROD).get_output(0)
        scores = network.add_matrix_multiply(
            q_scaled,
            trt.MatrixOperation.NONE,
            k,
            trt.MatrixOperation.TRANSPOSE,
        )
        probs = network.add_softmax(scores.get_output(0))
        probs.axes = 1 << 3
        context = network.add_matrix_multiply(
            probs.get_output(0),
            trt.MatrixOperation.NONE,
            v,
            trt.MatrixOperation.NONE,
        )
        attn_context = network.add_shuffle(context.get_output(0))
        attn_context.first_transpose = trt.Permutation([0, 2, 1, 3])
        attn_context.reshape_dims = (1, seq_len, hidden_size)
        attn = graph_ops.add_matmul_rhs_constant(
            network,
            attn_context.get_output(0),
            hidden_size,
            hidden_size,
            weights[f"{prefix}.attn.proj.weight"].astype(np.float32),
        )
        attn = graph_ops.add_bias_sum(
            network,
            attn,
            hidden_size,
            weights[f"{prefix}.attn.proj.bias"].astype(np.float32),
        )
        hidden = network.add_elementwise(
            hidden, attn, trt.ElementWiseOperation.SUM).get_output(0)

        norm2 = graph_ops.add_layer_norm_native(
            network,
            hidden,
            hidden_size,
            weights[f"{prefix}.norm2.weight"].astype(np.float32),
            weights[f"{prefix}.norm2.bias"].astype(np.float32),
            eps_val,
        )
        fc1_w = _slice_mlp_columns(
            weights[f"{prefix}.mlp.fc1.weight"].astype(np.float32),
            mlp_hidden,
            parallel,
        )
        fc1_b = _slice_mlp_columns(
            weights[f"{prefix}.mlp.fc1.bias"].astype(np.float32),
            mlp_hidden,
            parallel,
        )
        fc1 = graph_ops.add_matmul_rhs_constant(
            network,
            norm2,
            hidden_size,
            local_mlp_hidden,
            fc1_w,
        )
        fc1 = graph_ops.add_bias_sum(network, fc1, local_mlp_hidden, fc1_b)
        act = graph_ops.add_gelu_erf(network, fc1)
        fc2_w = _slice_mlp_rows(
            weights[f"{prefix}.mlp.fc2.weight"].astype(np.float32),
            mlp_hidden,
            parallel,
        )
        fc2 = graph_ops.add_matmul_rhs_constant(
            network,
            act,
            local_mlp_hidden,
            hidden_size,
            fc2_w,
        )
        fc2 = add_all_reduce_sum(network, fc2, parallel.tp_size)
        fc2 = graph_ops.add_bias_sum(
            network,
            fc2,
            hidden_size,
            weights[f"{prefix}.mlp.fc2.bias"].astype(np.float32),
        )
        hidden = network.add_elementwise(
            hidden, fc2, trt.ElementWiseOperation.SUM).get_output(0)

    hidden = graph_ops.add_layer_norm_native(
        network,
        hidden,
        hidden_size,
        weights["norm.weight"].astype(np.float32),
        weights["norm.bias"].astype(np.float32),
        eps_val,
    )
    cls = network.add_slice(
        hidden, start=(0, 0, 0), shape=(1, 1, hidden_size), stride=(1, 1, 1)
    ).get_output(0)

    logits = graph_ops.add_matmul_rhs_constant(
        network,
        cls,
        hidden_size,
        num_classes,
        _transpose_2d(
            weights["head.weight"].astype(np.float32),
            "head.weight",
        ),
    )
    logits = graph_ops.add_bias_sum(
        network, logits, num_classes, weights["head.bias"].astype(np.float32))
    flatten_logits = network.add_shuffle(logits)
    flatten_logits.reshape_dims = (1, num_classes)
    logits = flatten_logits.get_output(0)
    logits.name = "logits"
    network.mark_output(logits)

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT timm_vit tensor-parallel engine build failed")
    return bytes(plan)
