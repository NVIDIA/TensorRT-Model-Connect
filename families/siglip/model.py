# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a SigLIP bundle.

The bundle carries **two** engines, because SigLIP is two towers that meet
only at a dot product. Fusing them would make every call run both, which is
the wrong shape for the three tasks here: embedding an image, embedding a
text, and scoring one against the other.

Neither engine normalises its output, and neither applies the scale or the
bias. The engine returns what the checkpoint computes; the runtime owns the
L2 normalisation, ``exp(logit_scale)`` and ``logit_bias``, so that what comes
out of the engine stays comparable with what the reference returns.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import tensorrt as trt

from . import config as config_module, text_builder, vision_builder

_SUPPORTED_TASKS = ("image_to_embedding", "text_to_embedding", "text_image_to_relevance")


def _open_checkpoint(model_dir: Path) -> dict:
    """Load the safetensors checkpoint this family publishes."""
    try:
        from safetensors.numpy import load_file
    except ImportError as error:
        raise RuntimeError("This family requires safetensors in the build environment") from error

    path = Path(model_dir) / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"Required safetensors checkpoint is missing: {path}")
    return dict(load_file(str(path)))


def _require_weights(weights: dict) -> None:
    required = (
        "logit_scale",
        "logit_bias",
        "text_model.embeddings.token_embedding.weight",
        "text_model.final_layer_norm.weight",
        "text_model.head.weight",
        "vision_model.embeddings.patch_embedding.bias",
        "vision_model.post_layernorm.weight",
        # The pooling head is what makes this family not a CLIP copy.
        "vision_model.head.probe",
        "vision_model.head.attention.in_proj_weight",
    )
    missing = [key for key in required if key not in weights]
    if missing:
        raise KeyError("siglip checkpoint is missing expected tensors: " + ", ".join(missing))


def _pad_token_id(model_dir: Path) -> int:
    """The id the reference pads the window with, read from the checkpoint.

    It matters which id this is even though the padded rows are never pooled
    directly: the text tower is bidirectional and unmasked, so every padding
    row takes part in the attention that produces the row that *is* pooled.
    """
    tokenizer = json.loads((model_dir / "tokenizer.json").read_text())
    if tokenizer.get("model", {}).get("type") != "Unigram":
        # SigLIP 2 reuses this model_type but ships a Gemma BPE tokenizer and
        # a 256k vocabulary. The towers would build; the text would not
        # tokenise. Refused by name rather than mis-tokenised.
        raise NotImplementedError(
            "siglip is qualified for the unigram tokenizer; this checkpoint ships "
            f"{tokenizer.get('model', {}).get('type')!r} (SigLIP 2 is not yet supported)")

    config_path = model_dir / "tokenizer_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing siglip tokenizer config: {config_path}")
    pad_token = json.loads(config_path.read_text()).get("pad_token")
    if isinstance(pad_token, dict):
        pad_token = pad_token.get("content")
    if not pad_token:
        raise ValueError("siglip tokenizer declares no pad token")
    for entry in tokenizer.get("added_tokens", []):
        if entry.get("content") == pad_token:
            return int(entry["id"])
    for index, entry in enumerate(tokenizer["model"]["vocab"]):
        if entry[0] == pad_token:
            return index
    raise ValueError(f"siglip pad token {pad_token!r} is not in the vocabulary")


def _builder(verbose):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 3
    # TF32 caps a conv-and-matmul graph near 0.99998 cosine by default,
    # which hides real graph error behind a plausible-looking number.
    config.clear_flag(trt.BuilderFlag.TF32)
    return builder, network, config


def _serialize(builder, network, config, what):
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"siglip {what} engine build failed")
    return bytes(plan)


def build_vision_engine(weights, cfg, *, verbose=False) -> bytes:
    builder, network, config = _builder(verbose)
    # fp32 throughout. build() refuses every other precision and records the
    # measurement behind that refusal.
    work_np = np.float32
    size = cfg["vision"]["image_size"]
    source = network.add_input("pixel_values", trt.float32, (1, 3, size, size))
    embeds = vision_builder.build_vision_tower(network, source, weights, cfg["vision"],
                                               dtype=work_np)
    embeds.name = "image_embeds"
    network.mark_output(embeds)
    return _serialize(builder, network, config, "vision")


def build_text_engine(weights, cfg, *, verbose=False) -> bytes:
    builder, network, config = _builder(verbose)
    work_np = np.float32
    max_text_len = cfg["text"]["max_text_len"]
    input_ids = network.add_input("input_ids", trt.int32, (max_text_len,))
    embeds = text_builder.build_text_tower(network, input_ids, weights, cfg["text"],
                                           dtype=work_np)
    embeds.name = "text_embeds"
    network.mark_output(embeds)
    return _serialize(builder, network, config, "text")


def build(request, writer) -> None:
    """Build one SigLIP bundle."""
    if request.task not in _SUPPORTED_TASKS:
        raise ValueError("siglip supports only task=" + ", ".join(_SUPPORTED_TASKS))
    if request.backend not in {"trt", "trt_rtx"}:
        raise ValueError("siglip supports only backend=trt")
    if request.dynamic_kv_cache:
        raise NotImplementedError("siglip does not support dynamic_kv_cache")
    if request.max_batch_size != 1:
        raise NotImplementedError("siglip does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("siglip does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("siglip does not support context parallelism")
    if request.video_num_frames is not None:
        raise NotImplementedError("siglip does not support video_num_frames")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("siglip does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("siglip does not support mixed-precision layers")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("siglip does not support MTP speculative decoding")

    # fp32 only. Measured on this checkpoint against the reference, a
    # correct fp32 engine is 2.1e-07 off on the embeddings and an all-fp16
    # one is 3.6e-04, against the 1e-04 the E2E gate allows and the 8.4e-04
    # the weakest defect it catches - the exact GELU - is worth. The
    # relevance logit moves the same way, 3.6e-06 to 9.9e-03 against a
    # 2e-03 tolerance. Under 3x separates fp16's own noise from the break,
    # so fp16 is refused rather than admitted under a loosened gate.
    precision = str(request.precision).lower()
    if precision != "fp32":
        raise NotImplementedError(
            f"siglip builds fp32 only; {precision} is not qualified for this family")

    model_dir = Path(request.model_dir)
    weights = _open_checkpoint(model_dir)
    _require_weights(weights)
    cfg = config_module.resolve(model_dir, weights)
    vision, text = cfg["vision"], cfg["text"]

    image_size = int(request.image_height or 0) or vision["image_size"]
    if request.image_width is not None and int(request.image_width) != image_size:
        raise NotImplementedError("siglip needs a square input size")
    if image_size != vision["image_size"]:
        # One position row per patch of the trained grid. A different input
        # size needs the table interpolated, not just resized.
        raise NotImplementedError(
            f"siglip input size is fixed at the checkpoint's {vision['image_size']}")

    if request.max_sequence_length is not None:
        if int(request.max_sequence_length) != text["max_text_len"]:
            # Not a capacity: the pooled row is the last row of this window.
            raise NotImplementedError(
                "siglip prompt length is fixed at the checkpoint's window of "
                f"{text['max_text_len']}")

    tokenizer = model_dir / "tokenizer.json"
    if not tokenizer.is_file():
        raise FileNotFoundError(f"missing siglip tokenizer: {tokenizer}")
    pad_token_id = _pad_token_id(model_dir)

    # Both are stored the way the reference applies them: the scale is kept in
    # log space and every use is exp() of it, and the bias is added after.
    logit_scale = float(math.exp(float(np.asarray(weights["logit_scale"]).reshape(()))))
    logit_bias = float(np.asarray(weights["logit_bias"]).reshape(()))

    writer.set_header(family="siglip", task=request.task, backend=request.backend)
    writer.add_bytes("vision.plan", build_vision_engine(
        weights, cfg, verbose=bool(request.verbose)))
    writer.add_bytes("text.plan", build_text_engine(
        weights, cfg, verbose=bool(request.verbose)))
    writer.add_bytes("tokenizer.json", tokenizer.read_bytes())
    writer.add_json(
        "runtime.json",
        {
            "input_image_h": image_size,
            "input_image_w": image_size,
            "image_mean": cfg["image_mean"],
            "image_std": cfg["image_std"],
            "resample": cfg["resample"],
            "max_text_len": text["max_text_len"],
            "pad_token_id": pad_token_id,
            "embedding_dim": text["projection_size"],
            "logit_scale": logit_scale,
            "logit_bias": logit_bias,
        },
    )
