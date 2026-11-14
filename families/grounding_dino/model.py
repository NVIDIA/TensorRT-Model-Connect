# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a Grounding DINO bundle.

The engine takes the image and the **tokenised prompt**, because the prompt
is what this model detects against. Shapes are fixed:

* the image is one square ``image_size`` - shifted-window attention needs a
  static partition, so the runtime resizes to this size rather than
  preserving the aspect ratio the way a dynamic detector would;
* the prompt is padded to ``max_text_len`` and the padding is masked. Fixing
  the length keeps every builder on static shapes; masking it is what makes
  the padded run agree with an unpadded reference.

The host side of the text inputs - the phrase block mask, the per-phrase
position ids and the one-dimensional sine embedding - is built by the runtime
(``runtime/text_inputs.cpp``). ``text_builder`` holds the numpy twin of that
derivation, which is what the parity harness scores.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorrt as trt
from safetensors import safe_open

from . import config as config_module, decoder_builder, encoder_builder
from . import graph as g, position_builder, projection_builder
from . import query_selection_builder, swin_builder, text_builder

_DEFAULT_IMAGE_SIZE = 800
# The reference demo keeps boxes above 0.3 and phrases above 0.25.
_DEFAULT_BOX_THRESHOLD = 0.3
_DEFAULT_TEXT_THRESHOLD = 0.25


def _load_weights(model_dir: Path) -> dict:
    """Read the tensors by the names the checkpoint file actually stores.

    No rename table here, unlike ``deformable_detr``: the names in
    ``model.safetensors`` are the names the builders use, verified key by key
    against the file rather than against a loaded model's ``state_dict()``.
    """
    path = Path(model_dir) / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"missing grounding_dino weights: {path}")
    weights: dict = {}
    with safe_open(str(path), framework="numpy") as reader:
        for key in reader.keys():
            weights[key] = reader.get_tensor(key)
    required = (
        "model.backbone.conv_encoder.model.embeddings.patch_embeddings.projection.weight",
        "model.text_backbone.embeddings.word_embeddings.weight",
        "model.text_projection.weight",
        "model.level_embed",
        "model.query_position_embeddings.weight",
        "bbox_embed.0.layers.0.weight",
    )
    missing = [key for key in required if key not in weights]
    if missing:
        raise KeyError(
            "grounding_dino checkpoint is missing expected tensors: " + ", ".join(missing))
    return weights


def build_detector_engine(weights, cfg, *, precision, verbose=False) -> bytes:
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    work_np, work_trt = ((np.float16, trt.float16) if precision == "fp16"
                         else (np.float32, trt.float32))

    size = cfg["image_size"]
    length = cfg["max_text_len"]
    shapes = cfg["shapes"]

    pixels = network.add_input("pixel_values", trt.float32, (1, 3, size, size))
    input_ids = network.add_input("input_ids", trt.int32, (length,))
    token_type_ids = network.add_input("token_type_ids", trt.int32, (length,))
    position_ids = network.add_input("position_ids", trt.int32, (length,))
    self_mask = network.add_input(
        "text_self_attention_mask", trt.float32, (1, length, length))
    key_mask = network.add_input("text_key_mask", trt.float32, (1, 1, length))
    text_position = network.add_input(
        "text_position_embedding", trt.float32, (1, length, cfg["d_model"]))

    def work(tensor):
        return tensor if tensor.dtype == work_trt else network.add_cast(
            tensor, work_trt).get_output(0)

    features = swin_builder.build_swin(network, work(pixels), weights, cfg["swin"],
                                       dtype=work_np)
    levels = projection_builder.build_projections(network, features, weights, dtype=work_np)
    vision = g.concat(
        network, [g.spatial_to_tokens(network, level) for level in levels], axis=1)
    position = g.add_constant(
        network, (1, sum(h * w for h, w in shapes), cfg["d_model"]),
        position_builder.build_level_embeddings(shapes, weights["model.level_embed"], work_np),
        work_np)

    text = text_builder.build_text_encoder(
        network, weights, cfg["text"], input_ids, token_type_ids, position_ids,
        work(self_mask), dtype=work_np)
    text = text_builder.project_text(network, text, weights, dtype=work_np)

    vision, text = encoder_builder.build_encoder(
        network, vision, text, position, weights, shapes, cfg, dtype=work_np,
        text_mask=work(self_mask), text_key_mask=work(key_mask),
        text_position=work(text_position))

    target, reference = query_selection_builder.build_query_selection(
        network, vision, text, weights, cfg, shapes, dtype=work_np,
        text_mask=work(key_mask))
    hidden, _boxes, entering = decoder_builder.build_decoder(
        network, target, reference, vision, text, weights, shapes, cfg,
        dtype=work_np, text_mask=work(key_mask))
    logits, boxes = decoder_builder.build_heads(
        network, hidden, entering, text, weights, cfg, dtype=work_np,
        text_mask=work(key_mask))

    for tensor, name in ((logits, "logits"), (boxes, "pred_boxes")):
        out = tensor if tensor.dtype == trt.float32 else network.add_cast(
            tensor, trt.float32).get_output(0)
        out.name = name
        network.mark_output(out)

    config = builder.create_builder_config()
    config.builder_optimization_level = 3
    if precision == "fp32":
        # Conv-heavy fp32 parity caps near 0.99998 with TF32 left on, which
        # hides real graph error behind a plausible-looking number.
        config.clear_flag(trt.BuilderFlag.TF32)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("grounding_dino engine build failed")
    return bytes(plan)


def build(request, writer) -> None:
    """Build one Grounding DINO bundle."""
    if request.task != "image_text_to_boxes":
        raise ValueError("grounding_dino supports only task=image_text_to_boxes")
    if request.backend not in {"trt", "trt_rtx"}:
        raise ValueError("grounding_dino supports only backend=trt")
    if request.dynamic_kv_cache:
        raise NotImplementedError("grounding_dino does not support dynamic_kv_cache")
    if request.max_batch_size != 1:
        raise NotImplementedError("grounding_dino does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("grounding_dino does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("grounding_dino does not support context parallelism")
    if request.video_num_frames is not None:
        raise NotImplementedError("grounding_dino does not support video_num_frames")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("grounding_dino does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("grounding_dino does not support mixed-precision layers")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("grounding_dino does not support MTP speculative decoding")

    precision = str(request.precision).lower()
    if precision not in {"fp16", "fp32"}:
        raise ValueError(f"Unsupported grounding_dino precision: {precision}")

    height = int(request.image_height or 0) or _DEFAULT_IMAGE_SIZE
    width = int(request.image_width or 0) or _DEFAULT_IMAGE_SIZE
    if height != width:
        raise NotImplementedError("grounding_dino needs a square input size")
    if height <= 0 or height % 32:
        raise ValueError("grounding_dino input size must be a positive multiple of 32")

    model_dir = Path(request.model_dir)
    cfg = config_module.resolve(model_dir, image_size=height)
    if request.max_sequence_length is not None:
        if int(request.max_sequence_length) != cfg["max_text_len"]:
            raise NotImplementedError(
                "grounding_dino prompt length is fixed at the checkpoint's max_text_len")
    weights = _load_weights(model_dir)

    tokenizer = model_dir / "tokenizer.json"
    if not tokenizer.is_file():
        raise FileNotFoundError(f"missing grounding_dino tokenizer: {tokenizer}")

    writer.set_header(family="grounding_dino", task=request.task, backend=request.backend)
    writer.add_bytes("detector.plan", build_detector_engine(
        weights, cfg, precision=precision, verbose=bool(request.verbose)))
    writer.add_bytes("tokenizer.json", tokenizer.read_bytes())
    writer.add_json(
        "runtime.json",
        {
            "input_image_h": height,
            "input_image_w": width,
            "max_text_len": cfg["max_text_len"],
            "num_queries": cfg["num_queries"],
            "d_model": cfg["d_model"],
            "box_threshold": _DEFAULT_BOX_THRESHOLD,
            "text_threshold": _DEFAULT_TEXT_THRESHOLD,
            "image_mean": cfg["image_mean"],
            "image_std": cfg["image_std"],
        },
    )
