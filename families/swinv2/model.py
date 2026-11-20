# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a Swin Transformer V2 image-classification bundle.

One engine, one output: the class logits of ``Swinv2ForImageClassification``.
The head averages the final layer norm over every position of the last stage
and applies the classifier; nothing here softmaxes the result, because the Task
reports a ScoreKind and the honest kind for this head is a logit.

SwinV2 is not Swin with new weights. Four of its rules build and classify when
they are broken, so each is spelled out where it is applied:

* **res-post-norm**: a block normalises each branch's *output* and then adds it
  to the residual - ``x + norm(attention(x))`` - where Swin normalises the
  branch input. The checkpoint still calls the two norms ``layernorm_before``
  and ``layernorm_after``; the names are historical, not the order;
* **cosine attention**: the scores are the cosine similarity of query and key
  times one clamped, learned temperature per head - not a dot product scaled
  by ``1 / sqrt(head_dim)``;
* **continuous position bias** and the **shifted-window mask**, which depend
  only on the geometry and are computed once on the host (``window.py``);
* **patch merging** projects the four gathered positions first and normalises
  afterwards, at the merged width - the reverse of Swin.

The query and value projections carry a bias and the key projection does not;
the three are run as one matrix product with a zero key bias.

The class vocabulary rides in the bundle rather than being looked up at
runtime, so a bundle names its own classes or says it cannot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorrt as trt

from . import config as config_module, graph as g, window

_SUPPORTED_TASKS = ("image_to_class_scores",)
# The checkpoint stores float32, and float32 is the only precision this family
# is qualified at (tests/test_e2e.py).
_PRECISION = "fp32"
_DTYPE = np.float32
# torch.nn.functional.normalize's default epsilon, which the reference uses.
_NORMALIZE_EPSILON = 1e-12


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
    return {key: value.detach().to(torch.float32).numpy() for key, value in state.items()}


def _attention(network, hidden, weights, prefix: str, stage: dict, shift: int, *, dtype):
    """Shifted-window cosine attention over an NHWC map ``[1, side, side, dim]``."""
    side, dim, heads, size = stage["side"], stage["dim"], stage["heads"], stage["window"]
    per_side = side // size
    count, area, head_dim = per_side * per_side, size * size, dim // heads
    attention = f"{prefix}.attention.self"

    shifted = g.roll(network, hidden, (-shift, -shift), (1, 2)) if shift else hidden
    tokens = g.permute(
        network, shifted, (per_side, size, per_side, size, dim), (0, 2, 1, 3, 4), (count, area, dim))
    projection = np.concatenate(
        [weights[f"{attention}.{name}.weight"] for name in ("query", "key", "value")])
    bias = np.concatenate([
        weights[f"{attention}.query.bias"],
        np.zeros(dim, dtype=np.float32),
        weights[f"{attention}.value.bias"],
    ])
    qkv = g.linear(network, tokens, projection, bias, dtype=dtype)
    qkv = g.permute(network, qkv, (count, area, 3, heads, head_dim), (2, 0, 3, 1, 4), None)
    query, key, value = (
        g.reshape(network, g.slice_axis(network, qkv, 0, index, 1), (count, heads, area, head_dim))
        for index in range(3)
    )

    # cos(q, k) * temperature == (q / |q| * temperature) . (k / |k|), so the
    # temperature rides on the normalised query and the attention layer adds
    # the bias to the product as it is.
    query = g.l2_normalize(network, query, epsilon=_NORMALIZE_EPSILON)
    key = g.l2_normalize(network, key, epsilon=_NORMALIZE_EPSILON)
    temperature = window.logit_scale(weights[f"{attention}.logit_scale"]).reshape(1, heads, 1, 1)
    query = g.multiply(network, query, g.constant(network, temperature, dtype=dtype))
    additive = window.attention_bias(
        weights[f"{attention}.continuous_position_bias_mlp.0.weight"],
        weights[f"{attention}.continuous_position_bias_mlp.0.bias"],
        weights[f"{attention}.continuous_position_bias_mlp.2.weight"],
        height=side, width=side, window=size, shift=shift,
        pretrained_window=stage["pretrained_window"],
    )
    additive = np.broadcast_to(additive, (count, heads, area, area))
    context = g.attention(network, query, key, value, g.constant(network, additive, dtype=dtype))

    merged = g.permute(network, context, None, (0, 2, 1, 3), (count, area, dim))
    merged = g.linear(network, merged, weights[f"{prefix}.attention.output.dense.weight"],
                      weights[f"{prefix}.attention.output.dense.bias"], dtype=dtype)
    merged = g.permute(
        network, merged, (per_side, per_side, size, size, dim), (0, 2, 1, 3, 4), (1, side, side, dim))
    return g.roll(network, merged, (shift, shift), (1, 2)) if shift else merged


def _block(network, hidden, weights, prefix: str, stage: dict, shift: int, *, eps: float, dtype):
    """One res-post-norm block: ``x + norm(attention(x))``, then ``x + norm(mlp(x))``."""
    branch = _attention(network, hidden, weights, prefix, stage, shift, dtype=dtype)
    hidden = g.add(network, hidden, g.layer_norm(
        network, branch, weights[f"{prefix}.layernorm_before.weight"],
        weights[f"{prefix}.layernorm_before.bias"], epsilon=eps))
    inner = g.linear(network, hidden, weights[f"{prefix}.intermediate.dense.weight"],
                     weights[f"{prefix}.intermediate.dense.bias"], dtype=dtype)
    inner = g.gelu(network, inner, dtype=dtype)
    inner = g.linear(network, inner, weights[f"{prefix}.output.dense.weight"],
                     weights[f"{prefix}.output.dense.bias"], dtype=dtype)
    return g.add(network, hidden, g.layer_norm(
        network, inner, weights[f"{prefix}.layernorm_after.weight"],
        weights[f"{prefix}.layernorm_after.bias"], epsilon=eps))


def _merge(network, hidden, weights, prefix: str, stage: dict, *, eps: float, dtype):
    """Gather each 2 x 2 patch into the channels, project it, then normalise it.

    The reference concatenates (even row, even column), (odd row, even column),
    (even row, odd column), (odd row, odd column): the column parity is the
    outer index. Swapping the two still builds.
    """
    side, dim = stage["side"], stage["dim"]
    half = side // 2
    folded = g.permute(
        network, hidden, (half, 2, half, 2, dim), (0, 2, 3, 1, 4), (1, half, half, 4 * dim))
    reduced = g.linear(network, folded, weights[f"{prefix}.reduction.weight"], None, dtype=dtype)
    return g.layer_norm(network, reduced, weights[f"{prefix}.norm.weight"],
                        weights[f"{prefix}.norm.bias"], epsilon=eps)


def build_engine(weights: dict, cfg: dict, *, verbose: bool = False) -> bytes:
    """Serialize the TensorRT engine: ``pixel_values [1, 3, S, S]`` -> ``logits [1, classes]``."""
    work = _DTYPE
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 3
    # TF32 would cap a float32 build near 1e-3 of the reference and hide real
    # graph error behind a plausible-looking number.
    config.clear_flag(trt.BuilderFlag.TF32)

    size, patch = cfg["image_size"], cfg["patch_size"]
    pixels = network.add_input("pixel_values", trt.float32, (1, 3, size, size))
    projection = network.add_convolution_nd(
        pixels, cfg["embed_dim"], (patch, patch),
        trt.Weights(np.ascontiguousarray(
            weights["swinv2.embeddings.patch_embeddings.projection.weight"], dtype=work)),
        trt.Weights(np.ascontiguousarray(
            weights["swinv2.embeddings.patch_embeddings.projection.bias"], dtype=work)))
    if projection is None:
        raise RuntimeError("TensorRT rejected the SwinV2 patch embedding")
    projection.stride_nd = (patch, patch)
    # NCHW -> NHWC: windows, shifts and merges index positions spatially.
    hidden = g.permute(network, projection.get_output(0), None, (0, 2, 3, 1), None)
    hidden = g.layer_norm(network, hidden, weights["swinv2.embeddings.norm.weight"],
                          weights["swinv2.embeddings.norm.bias"], epsilon=cfg["norm_eps"])

    for index, stage in enumerate(cfg["stages"]):
        prefix = f"swinv2.encoder.layers.{index}"
        for block, shift in enumerate(stage["shifts"]):
            hidden = _block(network, hidden, weights, f"{prefix}.blocks.{block}", stage, shift,
                            eps=cfg["eps"], dtype=work)
        if f"{prefix}.downsample.reduction.weight" in weights:
            hidden = _merge(network, hidden, weights, f"{prefix}.downsample", stage,
                            eps=cfg["norm_eps"], dtype=work)

    hidden = g.layer_norm(network, hidden, weights["swinv2.layernorm.weight"],
                          weights["swinv2.layernorm.bias"], epsilon=cfg["eps"])
    pooled = g.mean_spatial(network, hidden)
    logits = g.linear(network, pooled, weights["classifier.weight"], weights["classifier.bias"],
                      dtype=work)
    logits.name = "logits"
    network.mark_output(logits)

    if verbose:
        windows = [(stage["side"], stage["window"], stage["shifts"]) for stage in cfg["stages"]]
        print(f"[trtmc build] swinv2: image={size}, patch={patch}, (side, window, shifts)={windows}, "
              f"classes={cfg['num_classes']}", file=sys.stderr)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("swinv2 engine build failed")
    return bytes(plan)


def build(request, writer) -> None:
    """Build one Swin Transformer V2 image-classification bundle."""
    if request.task not in _SUPPORTED_TASKS:
        raise ValueError("swinv2 supports only task=" + ", ".join(_SUPPORTED_TASKS))
    if request.backend != "trt":
        raise ValueError("swinv2 supports only backend=trt")
    if request.dynamic_kv_cache:
        raise NotImplementedError("swinv2 does not support dynamic_kv_cache")
    if request.max_batch_size != 1:
        raise NotImplementedError("swinv2 does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("swinv2 does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("swinv2 does not support context parallelism")
    if request.video_num_frames is not None:
        raise NotImplementedError("swinv2 does not support video_num_frames")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError(f"swinv2 does not support quantization {request.quantization!r}")
    if request.fp32_layers:
        raise NotImplementedError("swinv2 does not support mixed-precision layers")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("swinv2 does not support MTP speculative decoding")
    if request.max_sequence_length is not None and int(request.max_sequence_length) != 1:
        # The token grid follows from the image size; it is not a free parameter.
        raise NotImplementedError("swinv2 derives its token grid from the image size")
    precision = str(request.precision).lower()
    if precision != _PRECISION:
        raise NotImplementedError(
            f"swinv2 builds fp32, the checkpoint's precision; {precision} is not qualified")

    model_dir = Path(request.model_dir)
    weights = _open_torch_checkpoint(model_dir)
    cfg = config_module.resolve(model_dir, weights)
    for name, value in (("image_height", request.image_height), ("image_width", request.image_width)):
        if value is not None and int(value) != cfg["image_size"]:
            # Every stage's window and shift follow from the input size.
            raise NotImplementedError(
                f"swinv2 {name} is fixed at the checkpoint's {cfg['image_size']}")

    writer.set_header(family="swinv2", task=request.task, backend=request.backend)
    writer.add_bytes("model.plan", build_engine(weights, cfg, verbose=bool(request.verbose)))
    writer.add_json(
        "runtime.json",
        {
            "input_image_h": cfg["image_size"],
            "input_image_w": cfg["image_size"],
            "image_mean": cfg["image_mean"],
            "image_std": cfg["image_std"],
            "resample": cfg["resample"],
            "num_classes": cfg["num_classes"],
            "labels": cfg["labels"],
            "vocabulary_id": cfg["vocabulary_id"],
        },
    )
