# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native TensorRT HSTU ranking family (recsys-examples' KuaiRand-1K model).

Three engines, one per recsys-examples module, run in order on one CUDA
stream, each reading the previous engine's output buffers in place:

- ``embedding.plan`` (``InferenceEmbedding``): the contextual, item and action
  table lookups;
- ``hstu.plan`` (``HSTUBlockInference``): the preprocessor (interleave the
  history's item and action rows, put the contextual rows in front and the
  candidates behind, add positions), the HSTU layers and the postprocessor
  (the candidates' rows, L2-normalized);
- ``ranking_head.plan`` (the ranking ``MLP``): one logit per candidate and task.

Every shape is static at the checkpoint's geometry (``config.json``
``max_history_seqlen`` and ``max_num_candidates``): the benchmark request is the
padded maximum. Network construction uses TensorRT's Python Network API
directly; there is no ONNX export or parser.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import tensorrt as trt

from .config import HstuConfig, load_config
from .graph import Graph, new_builder
from .weights import load_checkpoint, weights_record

TASK = "structured_decision"
PRECISIONS = {"bf16": trt.bfloat16, "fp32": trt.float32}
CHECKPOINT_SECTION = "hstu.checkpoint.json"
CANDIDATE_EMBEDDINGS = "candidate_embeddings"
LOGITS = "logits"
L2_FLOOR = 1e-6
#: The attention's query blocks (``HstuConfig.attention_blocks``): each block
#: attends only to its key prefix, skipping the keys the causal mask zeroes
#: for all of its rows. A tuning value for TensorRT 11.4's fused attention:
#: in engine-only screens on Jetson Thor at 8298 tokens, 3 blocks on 256-token
#: bounds beat 1, 2 and 4 blocks; other GPUs and versions are unmeasured.
QUERY_BLOCKS = 3
BLOCK_ALIGN = 256


def _serialize(builder, network, config, what: str) -> bytes:
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"TensorRT HSTU {what} engine build failed")
    return bytes(plan)


def build_embedding_engine(cfg: HstuConfig, tensors, work, *, verbose: bool) -> bytes:
    builder, network, settings = new_builder(verbose)
    g = Graph(network, work)
    contextual_ids = network.add_input("contextual_ids", trt.int32, (cfg.num_contextual,))
    history_items = network.add_input("history_item_ids", trt.int32, (cfg.history_length,))
    history_actions = network.add_input("history_action_ids", trt.int32, (cfg.history_length,))
    candidates = network.add_input("candidate_item_ids", trt.int32, (cfg.num_candidates,))
    contextual = []
    for index, feature in enumerate(cfg.contextual_features):
        one = network.add_slice(contextual_ids, (index,), (1,), (1,)).get_output(0)
        contextual.append(g.gather(g.weight(tensors[f"embedding.{feature.name}.weight"]), one))
    g.output(g.concat(contextual, 0), "contextual_embeddings")
    item_ids = g.concat([history_items, candidates], 0)
    g.output(
        g.gather(g.weight(tensors[f"embedding.{cfg.item_feature.name}.weight"]), item_ids),
        "item_embeddings",
    )
    g.output(
        g.gather(g.weight(tensors[f"embedding.{cfg.action_feature.name}.weight"]), history_actions),
        "action_embeddings",
    )
    return _serialize(builder, network, settings, "embedding")


def _valid_attention(g: Graph, cfg: HstuConfig):
    """recsys-examples' ``_get_valid_attn_mask`` (causal, one target group per
    candidate, no contextual prefix) over the unpadded sequence, as [1, N, N]
    of 0 and 1 in the work dtype: a token sees itself and every earlier token,
    except that a candidate is seen only by itself.

    The attention multiplies it in: with a select on the bool mask between
    the two GEMMs, TensorRT 11.4's Myelin takes a much slower fused-attention
    kernel. The mask stays a graph expression of index compares: the same 0/1
    mask as a build-time constant produced logits outside the BF16 tolerance
    on TensorRT 11.4."""
    n = cfg.tokens
    rows = g.i32(np.arange(n), (n, 1))
    cols = g.i32(np.arange(n), (1, n))
    causal = g.logical_not(g.compare(cols, rows, trt.ElementWiseOperation.GREATER))
    before_candidates = g.compare(
        cols, g.i32([cfg.candidate_start], (1, 1)), trt.ElementWiseOperation.LESS
    )
    itself = g.compare(cols, rows, trt.ElementWiseOperation.EQUAL)
    visible = g.compare(before_candidates, itself, trt.ElementWiseOperation.OR)
    valid = g.compare(causal, visible, trt.ElementWiseOperation.AND)
    return g.cast(g.reshape(valid, (1, n, n)), g.work)


def _hstu_layer(g: Graph, cfg: HstuConfig, tensors, x, valid, index: int):
    """``PagedHSTUInferLayer.forward_naive`` without a KV cache."""
    prefix = f"layers.{index}"
    n, width = cfg.tokens, cfg.attention_width
    normed = g.to_work(
        g.layer_norm_f32(
            x,
            tensors[f"{prefix}.input_layernorm.weight"],
            tensors[f"{prefix}.input_layernorm.bias"],
            cfg.layernorm_epsilon,
        )
    )
    uvqk = g.add(
        g.to_f32(g.linear(normed, tensors[f"{prefix}.uvqk.weight"])),
        g.weight_f32(tensors[f"{prefix}.uvqk.bias"], (1, cfg.uvqk_width)),
    )
    uvqk = g.to_work(g.silu(uvqk))
    user, value, query, key = (g.columns(uvqk, part * width, width) for part in range(4))
    heads = (cfg.num_attention_heads, cfg.head_dim)
    query, key, value = (g.heads(t, n, *heads) for t in (query, key, value))
    # pytorch_hstu_mha's SiLU(alpha QK^T) under the mask, in the work dtype: an
    # FP32 pointwise between the two GEMMs (the CUTLASS kernel's arithmetic)
    # makes TensorRT 11.4's Myelin FMHA codegen fail ("FMHA IR verification
    # failed": operands of mixed type). The 1/scaling_seqlen factor is linear,
    # so it scales P.V in FP32 instead, where 8398 is exact (BF16 rounds it to
    # 8384).
    alpha = g.work_scalar(1.0 / math.sqrt(cfg.head_dim), 3)
    blocks = []
    for start, end in cfg.attention_blocks(QUERY_BLOCKS, BLOCK_ALIGN):
        # Each query block attends to keys [0, end) only: the mask is zero
        # right of the block's last row, so the skipped products were zeros.
        rows = end - start
        scores = g.mul(
            g.matmul(
                g.window(query, start, rows, cfg.head_dim),
                g.window(key, 0, end, cfg.head_dim),
                transpose_b=True,
            ),
            alpha,
        )
        probs = g.mul(g.silu(scores), g.window(valid, start, rows, end))  # x * 1 = x, x * 0 = +-0
        blocks.append(g.matmul(probs, g.window(value, 0, end, cfg.head_dim)))
    attended = blocks[0] if len(blocks) == 1 else g.concat(blocks, 1)
    attended = g.div(
        g.to_f32(g.merge_heads(attended, n, width)), g.scalar(float(cfg.scaling_seqlen), 2)
    )
    normed_attention = g.layer_norm_f32(
        attended,
        tensors[f"{prefix}.output_layernorm.weight"],
        tensors[f"{prefix}.output_layernorm.bias"],
        cfg.layernorm_epsilon,
    )
    parallel = g.to_work(g.mul(g.to_f32(user), normed_attention))
    projected = g.linear(parallel, tensors[f"{prefix}.proj.weight"])
    return g.to_work(g.add(g.to_f32(projected), g.to_f32(x)))


def build_hstu_engine(cfg: HstuConfig, tensors, work, *, verbose: bool) -> bytes:
    builder, network, settings = new_builder(verbose)
    g = Graph(network, work)
    d = cfg.hidden_size
    contextual = network.add_input("contextual_embeddings", work, (cfg.num_contextual, d))
    items = network.add_input("item_embeddings", work, (cfg.history_length + cfg.num_candidates, d))
    actions = network.add_input("action_embeddings", work, (cfg.history_length, d))
    # hstu_preprocess_embeddings (inference): [contextual | (item, action) x H | candidates]
    history = g.reshape(g.rows(items, 0, cfg.history_length), (cfg.history_length, 1, d))
    pairs = g.concat([history, g.reshape(actions, (cfg.history_length, 1, d))], 1)
    interleaved = g.reshape(pairs, (2 * cfg.history_length, d))
    candidates = g.rows(items, cfg.history_length, cfg.num_candidates)
    x = g.concat([contextual, interleaved, candidates], 0)
    # HSTUPositionalEncoder (inference): x * sqrt(d) + position[min(n, min(N - C, K - 1))]
    position = g.gather(
        g.weight(tensors["position.weight"]), g.i32(cfg.position_ids(), (cfg.tokens,))
    )
    x = g.to_work(g.add(g.mul(g.to_f32(x), g.scalar(math.sqrt(d), 2)), g.to_f32(position)))
    valid = _valid_attention(g, cfg)
    for index in range(cfg.num_layers):
        x = _hstu_layer(g, cfg, tensors, x, valid, index)
    # HSTUBlockPostprocessor: the candidates' rows, L2-normalized
    rows = g.rows(x, cfg.candidate_start, cfg.num_candidates)
    g.output(g.to_work(g.l2_normalize_f32(rows, L2_FLOOR)), CANDIDATE_EMBEDDINGS)
    return _serialize(builder, network, settings, "HSTU block")


def build_ranking_head_engine(cfg: HstuConfig, tensors, work, *, verbose: bool) -> bytes:
    builder, network, settings = new_builder(verbose)
    g = Graph(network, work)
    x = network.add_input(CANDIDATE_EMBEDDINGS, work, (cfg.num_candidates, cfg.hidden_size))
    layers = len(cfg.prediction_head_arch)
    for index, width in enumerate(cfg.prediction_head_arch):
        x = g.add(
            g.to_f32(g.linear(x, tensors[f"head.{index}.weight"])),
            g.weight_f32(tensors[f"head.{index}.bias"], (1, width)),
        )
        if index < layers - 1:
            x = g.to_work(g.relu(x))
    g.output(x, LOGITS)  # FP32 logits [candidates, tasks]
    return _serialize(builder, network, settings, "ranking head")


def runtime_config(cfg: HstuConfig, precision: str) -> dict:
    """What the native runtime reads: the request geometry and every table's
    vocabulary (ids outside a table are refused before any enqueue)."""
    feature = lambda f: {"name": f.name, "vocab_size": f.vocab_size}  # noqa: E731
    return {
        "precision": precision,
        "history_length": cfg.history_length,
        "num_candidates": cfg.num_candidates,
        "tokens": cfg.tokens,
        "num_tasks": cfg.num_tasks,
        "contextual_features": [feature(f) for f in cfg.contextual_features],
        "item_feature": feature(cfg.item_feature),
        "action_feature": feature(cfg.action_feature),
    }


def checkpoint_record(cfg: HstuConfig) -> dict:
    """What names the weights a bundle holds: no trained checkpoint is public,
    so a bundle states the synthetic checkpoint's name, generator and seed
    (the generator writes the same bytes for the same seed)."""
    weights = weights_record(cfg)
    name = cfg.raw.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("HSTU config.json must name its checkpoint (name)")
    return {
        "name": name,
        "origin": weights["origin"],
        "generator": weights["generator"],
        "seed": weights["seed"],
        "reference": cfg.raw.get("reference"),
        "geometry": {
            "history_length": cfg.history_length,
            "num_candidates": cfg.num_candidates,
            "tokens": cfg.tokens,
        },
    }


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one HSTU ranking bundle."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("hstu does not support dynamic_kv_cache")
    if request.task != TASK:
        raise ValueError(f"hstu supports only task={TASK}")
    if request.backend != "trt":
        raise ValueError("hstu supports only the trt backend")
    # The geometry is the checkpoint's config.json (max_history_seqlen,
    # max_num_candidates): every engine shape is static.
    if request.max_sequence_length is not None:
        raise NotImplementedError(
            "hstu does not support max_sequence_length: its geometry is "
            "the checkpoint's config.json"
        )
    if request.image_height is not None:
        raise NotImplementedError("hstu does not support image_height")
    if request.image_width is not None:
        raise NotImplementedError("hstu does not support image_width")
    if request.video_num_frames is not None:
        raise NotImplementedError("hstu does not support video_num_frames")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("hstu does not support MTP speculative decoding")
    if request.max_batch_size != 1:
        raise NotImplementedError("hstu serves batch 1")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("hstu does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("hstu does not support context parallelism")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("hstu does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("hstu does not support mixed-precision layers")
    if request.graph_transform is not None:
        raise NotImplementedError("hstu builds its network directly: no graph transform")
    precision = str(request.precision).lower()
    if precision not in PRECISIONS:
        raise ValueError("hstu builds bf16 (the reference's) or fp32 (structure validation)")
    work = PRECISIONS[precision]
    model_dir = Path(request.model_dir)
    cfg = load_config(model_dir)
    tensors = load_checkpoint(model_dir, cfg)
    writer.set_header(family="hstu", task=request.task, backend=request.backend)
    writer.add_json("runtime.json", runtime_config(cfg, precision))
    writer.add_json(CHECKPOINT_SECTION, checkpoint_record(cfg))
    verbose = bool(request.verbose)
    writer.add_bytes("embedding.plan", build_embedding_engine(cfg, tensors, work, verbose=verbose))
    writer.add_bytes("hstu.plan", build_hstu_engine(cfg, tensors, work, verbose=verbose))
    writer.add_bytes(
        "ranking_head.plan", build_ranking_head_engine(cfg, tensors, work, verbose=verbose)
    )


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter
