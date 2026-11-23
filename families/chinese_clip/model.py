# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a CLIP bundle.

The bundle carries **two** engines, because CLIP is two towers that meet only
at a dot product. Fusing them into one engine would force every call to run
both towers, which is the wrong shape for the three tasks this family serves:
embedding an image, embedding a text, and scoring one against the other.

Neither engine normalises its output. The projected embedding is what the
checkpoint computes - its norm is around 8.8 for an image and 10.5 for a text,
not 1 - and the runtime is where L2 normalisation and the ``logit_scale``
belong, so that what the engine returns stays the thing the reference returns.

This family requires torch in the build environment: the published
``OFA-Sys/chinese-clip-vit-base-patch16`` checkpoint supplies
``pytorch_model.bin`` weights and a ``vocab.txt`` WordPiece vocabulary.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import tensorrt as trt

from . import config as config_module, vision_builder, graph as g

_SUPPORTED_TASKS = ("image_to_embedding", "text_to_embedding", "text_image_to_relevance")


def _open_torch_checkpoint(model_dir: Path) -> dict:
    """Open the family's required pytorch_model.bin checkpoint."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("This family requires torch in the build environment") from error

    path = Path(model_dir) / "pytorch_model.bin"
    if not path.is_file():
        raise FileNotFoundError(f"Required PyTorch checkpoint is missing: {path}")
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    return {key: value.numpy() for key, value in state.items()}


def _builder(precision, verbose, quant_ctx=None):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 3
    if precision == "fp32":
        # TF32 caps a conv-and-matmul graph near 0.99998 cosine by default,
        # which hides real graph error behind a plausible-looking number.
        config.clear_flag(trt.BuilderFlag.TF32)
    if quant_ctx is not None:
        # Keep layer names so the plan inspector shows which kernels run.
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    return builder, network, config


def _serialize(builder, network, config, what):
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"chinese_clip {what} engine build failed")
    return bytes(plan)


def build_vision_engine(
    weights, cfg, *, image_size, precision, verbose=False, quant_ctx=None
) -> bytes:
    builder, network, config = _builder(precision, verbose, quant_ctx)
    work_np, work_trt = (
        (np.float16, trt.float16) if precision == "fp16" else (np.float32, trt.float32)
    )
    pixels = network.add_input("pixel_values", trt.float32, (1, 3, image_size, image_size))
    source = (
        pixels if pixels.dtype == work_trt else network.add_cast(pixels, work_trt).get_output(0)
    )
    embeds = vision_builder.build_vision_tower(
        network,
        source,
        weights,
        cfg["vision"],
        weights["visual_projection.weight"],
        dtype=work_np,
        quant_ctx=quant_ctx,
    )
    out = (
        embeds
        if embeds.dtype == trt.float32
        else network.add_cast(embeds, trt.float32).get_output(0)
    )
    out.name = "image_embeds"
    network.mark_output(out)
    return _serialize(builder, network, config, "vision")


def build_text_engine(weights, cfg, *, max_text_len, verbose=False):
    builder, network, options = _builder("fp32", verbose)
    text = cfg["text"]
    width = text["hidden_size"]

    def const(key):
        value = weights[key]
        return g.add_constant(network, value.shape, value)

    def linear(x, key):
        return g.add_linear(network, x, weights[key + ".weight"], weights[key + ".bias"])

    def norm(x, key):
        return g.add_layer_norm(
            network, x, weights[key + ".weight"], weights[key + ".bias"], text["eps"]
        )

    ids = network.add_input("input_ids", trt.int32, (max_text_len,))
    valid = network.add_input("attention_mask", trt.int32, (max_text_len,))
    hidden = network.add_gather(
        const("text_model.embeddings.word_embeddings.weight"), ids, 0
    ).get_output(0)
    positions = weights["text_model.embeddings.position_embeddings.weight"][:max_text_len]
    types = weights["text_model.embeddings.token_type_embeddings.weight"][:1]
    hidden = g.add_sum(network, hidden, g.add_constant(network, positions.shape, positions))
    hidden = norm(
        g.add_sum(network, hidden, g.add_constant(network, types.shape, types)),
        "text_model.embeddings.LayerNorm",
    )
    valid = network.add_cast(valid, trt.float32).get_output(0)
    penalty = network.add_elementwise(
        g.add_constant(network, (1,), np.ones(1)), valid, trt.ElementWiseOperation.SUB
    ).get_output(0)
    penalty = network.add_elementwise(
        penalty, g.add_constant(network, (1,), np.array([-1e9])), trt.ElementWiseOperation.PROD
    ).get_output(0)
    mask = network.add_shuffle(penalty)
    mask.reshape_dims = (1, 1, 1, max_text_len)
    for index in range(text["num_layers"]):
        prefix = f"text_model.encoder.layer.{index}"
        projections = [
            linear(hidden, prefix + ".attention.self." + name) for name in ("query", "key", "value")
        ]
        attention = g.add_attention(
            network,
            *projections,
            num_heads=text["num_heads"],
            head_dim=text["head_dim"],
            mask=mask.get_output(0),
        )
        hidden = norm(
            g.add_sum(network, hidden, linear(attention, prefix + ".attention.output.dense")),
            prefix + ".attention.output.LayerNorm",
        )
        inner = g.add_gelu(network, linear(hidden, prefix + ".intermediate.dense"))
        hidden = norm(
            g.add_sum(network, hidden, linear(inner, prefix + ".output.dense")),
            prefix + ".output.LayerNorm",
        )
    cls = network.add_slice(hidden, (0, 0), (1, width), (1, 1)).get_output(0)
    output = g.add_linear(network, cls, weights["text_projection.weight"])
    output.name = "text_embeds"
    network.mark_output(output)
    return _serialize(builder, network, options, "text")


def build(request, writer):
    if (
        request.task not in ("image_to_embedding", "text_to_embedding", "text_image_to_relevance")
        or request.backend != "trt"
    ):
        raise ValueError(
            "Chinese-CLIP requires a supported embedding/relevance task and backend=trt"
        )
    if request.dynamic_kv_cache:
        raise NotImplementedError("chinese_clip does not support dynamic_kv_cache")
    if (
        request.precision != "fp32"
        or request.tensor_parallel_size != 1
        or request.context_parallel_size != 1
        or request.max_batch_size != 1
        or request.dynamic_kv_cache
        or request.fp32_layers
        or request.quantization not in (None, "none")
        or request.video_num_frames is not None
        or request.mtp_seq_len is not None
    ):
        raise NotImplementedError("Chinese-CLIP supports unquantized fp32 single-example inference")
    directory = Path(request.model_dir)
    cfg = config_module.resolve(directory)
    size = cfg["vision"]["image_size"]
    if any(
        value is not None and value != size for value in (request.image_height, request.image_width)
    ):
        raise ValueError("Chinese-CLIP image size must match the vision position table")
    length = request.max_sequence_length or cfg["text"]["max_position_embeddings"]
    if length < 2 or length > cfg["text"]["max_position_embeddings"]:
        raise ValueError("Chinese-CLIP text profile exceeds its position table")
    weights = _open_torch_checkpoint(directory)
    for name in ("visual_projection.weight", "text_projection.weight"):
        if weights[name].shape[0] != cfg["projection_dim"]:
            raise ValueError(f"Chinese-CLIP {name} rows must match projection_dim")
    from tokenizers import BertWordPieceTokenizer

    tokenizer = BertWordPieceTokenizer(str(directory / "vocab.txt"), lowercase=True)
    writer.set_header(family="chinese_clip", task=request.task, backend=request.backend)
    writer.add_bytes(
        "vision.plan",
        build_vision_engine(
            weights, cfg, image_size=size, precision="fp32", verbose=request.verbose
        ),
    )
    writer.add_bytes(
        "text.plan", build_text_engine(weights, cfg, max_text_len=length, verbose=request.verbose)
    )
    writer.add_bytes("tokenizer.json", tokenizer.to_str().encode())
    writer.add_json(
        "runtime.json",
        {
            "input_image_h": size,
            "input_image_w": size,
            "image_mean": cfg["image_mean"],
            "image_std": cfg["image_std"],
            "max_text_len": length,
            "projection_dim": cfg["projection_dim"],
            "logit_scale": math.exp(float(weights["logit_scale"])),
        },
    )
