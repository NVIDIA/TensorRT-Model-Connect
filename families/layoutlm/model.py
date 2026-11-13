# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a LayoutLM document question answering bundle.

The bundle answers a question about a document image (``images_text_to_text``),
the way transformers' DocumentQuestionAnsweringPipeline does for LayoutLM: the
runtime reads the words and boxes with Tesseract OCR, tokenizes question and
words into overlapping windows of ``max_sequence_length`` tokens, runs this
engine on every window and decodes the best start/end span into answer text.

Sections:

* ``model.plan`` - the encoder and question-answering head, one dynamic-length
  window per call;
* ``tokenizer.json`` - the checkpoint's byte-level BPE tokenizer with the
  ``add_prefix_space`` setting the reference applies;
* ``runtime.json`` - window, stride, answer-length and OCR settings, which are
  the pipeline's defaults (top 1 answer, at most 15 tokens, English OCR,
  automatic page segmentation).
"""

from __future__ import annotations

from pathlib import Path

import tensorrt as trt

from . import checkpoint, graph

_TASK = "images_text_to_text"
_PRECISIONS = ("fp32", "fp16")
# transformers' postprocess default for extractive answers.
_MAX_ANSWER_LENGTH = 15
# Tesseract's automatic page segmentation (PSM_AUTO): the command-line default
# the reference runs; the C API would otherwise start in single-block mode.
_PAGE_SEGMENTATION_MODE = 3
_OCR_LANGUAGE = "eng"


def build_engine(weights, config, *, max_sequence_length: int, precision: str = "fp32",
                 verbose: bool = False) -> bytes:
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    builder_config = builder.create_builder_config()
    builder_config.builder_optimization_level = 3
    if precision == "fp32":
        # TF32 would cap the float32 build's agreement with the reference near 1e-3.
        builder_config.clear_flag(trt.BuilderFlag.TF32)
    else:
        # The fp16 graph multiplies fp16-valued float32 operands, which TF32 represents
        # exactly, to accumulate in float32 (graph._Graph.matmul).
        builder_config.set_flag(trt.BuilderFlag.TF32)
    network_graph = graph.build_network(network, weights, config, precision)
    profile = builder.create_optimization_profile()
    window = max_sequence_length
    profile.set_shape("input_ids", (1,), (window,), (window,))
    profile.set_shape("bbox", (1, 4), (window, 4), (window, 4))
    builder_config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, builder_config)
    del network_graph
    if plan is None:
        raise RuntimeError("layoutlm engine build failed")
    return bytes(plan)


def build(request, writer) -> None:
    """Build one LayoutLM document question answering bundle."""
    if request.task != _TASK:
        raise ValueError(f"layoutlm supports only task={_TASK}")
    if request.backend != "trt":
        raise ValueError("layoutlm supports only backend=trt")
    if request.quantization not in (None, "none"):
        raise NotImplementedError("layoutlm builds the checkpoint's own weights; no quantization")
    if request.max_batch_size != 1:
        raise NotImplementedError("layoutlm answers one document per call; max_batch_size is 1")
    if request.tensor_parallel_size != 1 or request.context_parallel_size != 1:
        raise NotImplementedError("layoutlm runs on one GPU")
    if request.dynamic_kv_cache:
        raise NotImplementedError("layoutlm does not support dynamic_kv_cache")
    if request.image_height is not None or request.image_width is not None:
        raise NotImplementedError(
            "layoutlm reads the document at its own size through OCR; it takes no image dimensions")
    if request.video_num_frames is not None:
        raise NotImplementedError("layoutlm does not take video")
    if request.fp32_layers:
        raise NotImplementedError("layoutlm does not support mixed-precision layers")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("layoutlm does not support MTP speculative decoding")
    precision = str(request.precision).lower()
    if precision not in _PRECISIONS:
        raise NotImplementedError(f"layoutlm builds fp32 (the checkpoint's own) or fp16, not {precision}")

    model_dir = Path(request.model_dir)
    config = checkpoint.read_config(model_dir)
    tokenizer = checkpoint.read_tokenizer(model_dir)
    if tokenizer["vocab_size"] > config["vocab_size"]:
        raise ValueError("layoutlm tokenizer ids exceed the word embedding table")
    # The reference's default window is the tokenizer's model_max_length.
    default_window = tokenizer["model_max_length"] or config["max_position_embeddings"]
    window = int(request.max_sequence_length or default_window)
    if window < 8 or window > config["max_position_embeddings"]:
        raise ValueError(f"layoutlm windows hold 8..{config['max_position_embeddings']} tokens, "
                         f"not {window}")
    weights = checkpoint.read_weights(model_dir, config)

    writer.set_header(family="layoutlm", task=request.task, backend=request.backend)
    writer.add_bytes("model.plan", build_engine(weights, config, max_sequence_length=window,
                                                precision=precision, verbose=bool(request.verbose)))
    writer.add_bytes("tokenizer.json", tokenizer["json"])
    writer.add_json(
        "runtime.json",
        {
            "max_sequence_length": window,
            # The pipeline's default stride, min(max_seq_len // 2, 256).
            "doc_stride": min(window // 2, 256),
            "max_answer_length": _MAX_ANSWER_LENGTH,
            "max_2d_position": config["max_2d_position_embeddings"] - 1,
            "ocr_language": _OCR_LANGUAGE,
            "ocr_page_segmentation_mode": _PAGE_SEGMENTATION_MODE,
            "cls_token_id": tokenizer["cls_token_id"],
            "sep_token_id": tokenizer["sep_token_id"],
            "vocab_size": config["vocab_size"],
            "precision": precision,
        },
    )
