# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OneFormer semantic-segmentation family builder.

Builds ``OneFormerForUniversalSegmentation`` checkpoints with a Swin backbone,
for example ``shi-labs/oneformer_cityscapes_swin_large``, as one TensorRT
engine for the ``image_to_semantic_segmentation`` Task: batch 1, dynamic image
height and width, fp32 with TF32 disabled. The engine is conditioned on the
semantic task prompt and returns the stride-4 semantic score map; the runtime
preprocesses the image and resizes the scores to it. An fp16 engine is not
offered: it loses Swin-L stage-3 accuracy (see the family README).
"""

from __future__ import annotations

import sys
from pathlib import Path

import tensorrt as trt

from .config import OneFormerConfig, load_config
from .graph import Graph
from .pixel_decoder import build_pixel_decoder
from .position import sine_dim_t
from .swin import build_swin
from .transformer import build_transformer, semantic_scores
from .weights import _open_torch_checkpoint

TASK = "image_to_semantic_segmentation"
TASK_PROMPT = "the task is semantic"
INPUT = "pixel_values"
OUTPUT = "semantic_scores"
MIN_IMAGE_SIDE = 32
# Tensors the inference graph does not read: the training criterion only.
UNUSED_TENSORS = frozenset({"criterion.empty_weight", "criterion.logit_scale"})


def task_token_ids(model_dir: Path, length: int) -> list[int]:
    """The processor's task input: CLIP ids of the prompt, padded to ``length``, times the attention mask."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    tokens = tokenizer([TASK_PROMPT], padding="max_length", max_length=length, truncation=True)
    ids = [
        int(token) * int(mask)
        for token, mask in zip(tokens["input_ids"][0], tokens["attention_mask"][0])
    ]
    if len(ids) != length or not any(ids):
        raise ValueError("oneformer task prompt did not tokenize to the task sequence length")
    return ids


def build_network(network, config: OneFormerConfig, weights, task_ids, debug=()):
    """Populate ``network``; ``debug`` names extra outputs used by the family's parity checks."""
    g = Graph(network)
    pixels = network.add_input(INPUT, trt.float32, (1, 3, -1, -1))
    dim_t = sine_dim_t(config.hidden_dim // 2)
    features = build_swin(g, pixels, weights, config.swin)
    levels, mask_features = build_pixel_decoder(g, features, weights, config, dim_t)
    logits, masks = build_transformer(g, levels, mask_features, task_ids, weights, config, dim_t)
    outputs = {OUTPUT: semantic_scores(g, logits, masks, config)}
    available = {
        "class_queries_logits": logits,
        "masks_queries_logits": masks,
        "mask_features": mask_features,
    }
    available.update(
        {f"backbone_stage{index + 1}": feature for index, feature in enumerate(features)}
    )
    available.update({f"pixel_decoder_level{index}": level for index, level in enumerate(levels)})
    for name in debug:
        outputs[name] = available[name]
    for name, tensor in outputs.items():
        tensor.name = name
        network.mark_output(tensor)
    unused = set(weights.unused()) - UNUSED_TENSORS
    if unused:
        raise ValueError(f"oneformer graph left checkpoint tensors unused: {sorted(unused)[:8]}")
    return g


def build_engine(
    config: OneFormerConfig,
    weights,
    task_ids,
    *,
    max_height: int,
    max_width: int,
    verbose: bool = False,
    debug=(),
) -> bytes:
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    graph = build_network(network, config, weights, task_ids, debug)
    trt_config = builder.create_builder_config()
    trt_config.builder_optimization_level = 3
    trt_config.clear_flag(trt.BuilderFlag.TF32)
    profile = builder.create_optimization_profile()
    opt = (
        1,
        3,
        min(config.preprocess.shortest_edge, max_height),
        min(config.preprocess.longest_edge, max_width),
    )
    profile.set_shape(
        INPUT, (1, 3, MIN_IMAGE_SIDE, MIN_IMAGE_SIDE), opt, (1, 3, max_height, max_width)
    )
    trt_config.add_optimization_profile(profile)
    if verbose:
        print(
            f"[trtmc build] oneformer: fp32, image {MIN_IMAGE_SIDE}..{max_height} x "
            f"{MIN_IMAGE_SIDE}..{max_width}, opt {opt[2]}x{opt[3]}",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, trt_config)
    del graph
    if plan is None:
        raise RuntimeError("TensorRT oneformer engine build failed")
    return bytes(plan)


def runtime_config(config: OneFormerConfig) -> dict:
    preprocess = config.preprocess
    return {
        "shortest_edge": preprocess.shortest_edge,
        "longest_edge": preprocess.longest_edge,
        "image_mean": list(preprocess.image_mean),
        "image_std": list(preprocess.image_std),
        "rescale_factor": preprocess.rescale_factor,
        "class_names": list(config.class_names),
    }


def build(request, writer) -> None:
    """Build one OneFormer semantic-segmentation bundle."""
    if request.task != TASK:
        raise ValueError(f"oneformer supports only task={TASK}")
    if request.backend != "trt":
        raise ValueError("oneformer supports only backend=trt")
    if str(request.precision).lower() != "fp32":
        raise ValueError(f"oneformer supports only precision fp32, got {request.precision!r}")
    if request.max_sequence_length not in {None, 1}:
        raise NotImplementedError("oneformer supports only max_sequence_length=1")
    if request.max_batch_size != 1:
        raise NotImplementedError("oneformer supports only max_batch_size=1")
    if request.tensor_parallel_size != 1 or request.context_parallel_size != 1:
        raise NotImplementedError("oneformer does not support tensor or context parallelism")
    if request.video_num_frames is not None:
        raise NotImplementedError("oneformer does not support video_num_frames")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("oneformer does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("oneformer does not support fp32_layers")
    if request.dynamic_kv_cache:
        raise NotImplementedError("oneformer does not support dynamic_kv_cache")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("oneformer does not support MTP speculative decoding")

    model_dir = Path(request.model_dir)
    config = load_config(model_dir)
    # Image capacity of the engine profile; the processor never exceeds longest_edge on either side.
    max_height = int(request.image_height or config.preprocess.longest_edge)
    max_width = int(request.image_width or config.preprocess.longest_edge)
    if min(max_height, max_width) < MIN_IMAGE_SIDE:
        raise ValueError(
            f"oneformer image capacity must be at least {MIN_IMAGE_SIDE} pixels per side"
        )
    task_ids = task_token_ids(model_dir, config.task_seq_len)
    weights = _open_torch_checkpoint(model_dir)
    plan = build_engine(
        config,
        weights,
        task_ids,
        max_height=max_height,
        max_width=max_width,
        verbose=bool(request.verbose),
    )
    writer.set_header(family="oneformer", task=request.task, backend=request.backend)
    writer.add_bytes("engine.plan", plan)
    writer.add_json("runtime.json", runtime_config(config))
