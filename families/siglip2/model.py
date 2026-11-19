# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a SigLIP 2 NaFlex bundle.

The bundle carries **two** engines, because SigLIP 2 is two towers that meet only at a
dot product: embedding an image runs the vision engine, embedding a text runs the text
engine, and scoring a pair runs both.

The vision engine takes one image's patches with the patch count dynamic up to the
processor's budget, plus the taps that resample the position table to the image's
grid; the runtime owns the NaFlex resize, the patch cut and those taps. Neither engine
normalises its output or applies the scale and bias: the runtime owns the L2
normalisation, ``exp(logit_scale)`` and ``logit_bias``, so that what comes out of an
engine stays comparable with what the reference returns.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import tensorrt as trt

from . import config as config_module, text_builder, vision_builder

_SUPPORTED_TASKS = ("image_to_embedding", "text_to_embedding", "text_image_to_relevance")
_SENTENCE_PIECE_SPACE = "▁"


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
        "text_model.embeddings.position_embedding.weight",
        "text_model.final_layer_norm.weight",
        "text_model.head.weight",
        "vision_model.embeddings.patch_embedding.weight",
        "vision_model.embeddings.patch_embedding.bias",
        "vision_model.embeddings.position_embedding.weight",
        "vision_model.post_layernorm.weight",
        "vision_model.head.probe",
        "vision_model.head.attention.in_proj_weight",
        "vision_model.head.attention.in_proj_bias",
    )
    missing = [key for key in required if key not in weights]
    if missing:
        raise KeyError("siglip2 checkpoint is missing expected tensors: " + ", ".join(missing))
    for key, value in weights.items():
        if np.asarray(value).dtype != np.float32:
            raise NotImplementedError(
                f"siglip2 is qualified for the float32 checkpoint; {key} is {value.dtype}")


def _token_id(tokenizer: dict, token: str) -> int:
    for entry in tokenizer.get("added_tokens", []):
        if entry.get("content") == token:
            return int(entry["id"])
    vocab = tokenizer["model"]["vocab"]
    if token in vocab:
        return int(vocab[token])
    raise ValueError(f"siglip2 token {token!r} is not in the vocabulary")


def _named_token(config: dict, name: str) -> str:
    token = config.get(name)
    if isinstance(token, dict):
        token = token.get("content")
    if not token:
        raise ValueError(f"siglip2 tokenizer declares no {name}")
    return str(token)


def _tokenizer_contract(model_dir: Path) -> int:
    """Check the tokenizer is the one the runtime reproduces; return the pad id.

    The runtime tokenizer reproduces a SentencePiece-style BPE with byte fallback whose
    only normalisation is the space marker, with no pre-tokenizer, and whose template
    appends one end token. That is what this checkpoint ships and what the reference
    tokenizer runs: it does not lowercase (``do_lower_case`` in
    ``tokenizer_config.json`` is not applied by the reference), so neither does the
    runtime. Anything else is refused here rather than tokenised differently.
    """
    path = model_dir / "tokenizer.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing siglip2 tokenizer: {path}")
    tokenizer = json.loads(path.read_text(encoding="utf-8"))
    model = tokenizer.get("model", {})
    if model.get("type") != "BPE" or not model.get("byte_fallback"):
        raise NotImplementedError(
            "siglip2 is qualified for a byte-fallback BPE tokenizer, not "
            f"{model.get('type')!r}")
    expected_normalizer = {"type": "Replace", "pattern": {"String": " "},
                           "content": _SENTENCE_PIECE_SPACE}
    if tokenizer.get("normalizer") != expected_normalizer:
        raise NotImplementedError(
            "siglip2 is qualified for a tokenizer whose only normalizer replaces spaces "
            f"with U+2581, not {tokenizer.get('normalizer')!r}")
    if tokenizer.get("pre_tokenizer") is not None:
        raise NotImplementedError("siglip2 is qualified for a tokenizer without a pre-tokenizer")
    template = (tokenizer.get("post_processor") or {}).get("single") or []
    if (tokenizer.get("post_processor") or {}).get("type") != "TemplateProcessing" or [
            next(iter(item)) for item in template] != ["Sequence", "SpecialToken"]:
        # The runtime truncates a long prompt to the window by keeping its head and
        # re-appending the end token, which is what the reference's truncation does
        # with exactly this template.
        raise NotImplementedError(
            "siglip2 is qualified for a template that appends one end token")

    config_path = model_dir / "tokenizer_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing siglip2 tokenizer config: {config_path}")
    tokenizer_config = json.loads(config_path.read_text(encoding="utf-8"))
    eos_token = _named_token(tokenizer_config, "eos_token")
    if template[1]["SpecialToken"].get("id") != eos_token:
        raise NotImplementedError("siglip2 tokenizer template must append the end token")
    _token_id(tokenizer, eos_token)
    return _token_id(tokenizer, _named_token(tokenizer_config, "pad_token"))


def _builder(verbose):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 3
    # TF32 caps a matmul graph near 1e-4 relative error by default, which would hide
    # real graph error behind a plausible number.
    config.clear_flag(trt.BuilderFlag.TF32)
    return builder, network, config


def _serialize(builder, network, config, what):
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"siglip2 {what} engine build failed")
    return bytes(plan)


def build_vision_engine(weights, cfg, *, verbose=False) -> bytes:
    builder, network, config = _builder(verbose)
    vision = cfg["vision"]
    budget = vision["max_num_patches"]
    patches = network.add_input("pixel_values", trt.float32, (-1, vision["patch_features"]))
    taps = network.add_input("position_weights", trt.float32, (-1, vision["num_positions"]))
    profile = builder.create_optimization_profile()
    profile.set_shape("pixel_values", (1, vision["patch_features"]),
                      (budget, vision["patch_features"]), (budget, vision["patch_features"]))
    profile.set_shape("position_weights", (1, vision["num_positions"]),
                      (budget, vision["num_positions"]), (budget, vision["num_positions"]))
    config.add_optimization_profile(profile)
    embeds = vision_builder.build_vision_tower(network, patches, taps, weights, vision,
                                               dtype=np.float32)
    embeds.name = "image_embeds"
    network.mark_output(embeds)
    return _serialize(builder, network, config, "vision")


def build_text_engine(weights, cfg, *, verbose=False) -> bytes:
    builder, network, config = _builder(verbose)
    text = cfg["text"]
    input_ids = network.add_input("input_ids", trt.int32, (text["max_text_len"],))
    embeds = text_builder.build_text_tower(network, input_ids, weights, text, dtype=np.float32)
    embeds.name = "text_embeds"
    network.mark_output(embeds)
    return _serialize(builder, network, config, "text")


def build(request, writer) -> None:
    """Build one SigLIP 2 NaFlex bundle."""
    if request.task not in _SUPPORTED_TASKS:
        raise ValueError("siglip2 supports only task=" + ", ".join(_SUPPORTED_TASKS))
    if request.backend != "trt":
        raise ValueError("siglip2 supports only backend=trt")
    if request.dynamic_kv_cache:
        raise NotImplementedError("siglip2 does not support dynamic_kv_cache")
    if request.max_batch_size != 1:
        raise NotImplementedError("siglip2 does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("siglip2 does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("siglip2 does not support context parallelism")
    if request.video_num_frames is not None:
        raise NotImplementedError("siglip2 does not support video_num_frames")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError(
            f"siglip2 does not support quantization; got {request.quantization!r}")
    if request.fp32_layers:
        raise NotImplementedError("siglip2 does not support mixed-precision layers")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("siglip2 does not support MTP speculative decoding")
    if request.image_height is not None or request.image_width is not None:
        # NaFlex resizes every image to its own aspect-preserving grid within the
        # patch budget; there is no fixed input size to build for.
        raise NotImplementedError("siglip2 sizes each image itself; image_height/width are not used")

    # fp32, the checkpoint's own precision. A reduced-precision engine is not
    # qualified for this family.
    precision = str(request.precision).lower()
    if precision != "fp32":
        raise NotImplementedError(
            f"siglip2 builds fp32 only; {precision} is not qualified for this family")

    model_dir = Path(request.model_dir)
    weights = _open_checkpoint(model_dir)
    _require_weights(weights)
    cfg = config_module.resolve(model_dir, weights)
    vision, text = cfg["vision"], cfg["text"]

    if request.max_sequence_length is not None:
        if int(request.max_sequence_length) != text["max_text_len"]:
            # Not a capacity: the pooled row is the last row of this window.
            raise NotImplementedError(
                "siglip2 prompt length is fixed at the checkpoint's window of "
                f"{text['max_text_len']}")

    pad_token_id = _tokenizer_contract(model_dir)

    # Stored the way the reference applies them: the scale in log space, used as
    # exp() of it, and the bias added after.
    logit_scale = float(math.exp(float(np.asarray(weights["logit_scale"]).reshape(()))))
    logit_bias = float(np.asarray(weights["logit_bias"]).reshape(()))

    writer.set_header(family="siglip2", task=request.task, backend=request.backend)
    writer.add_bytes("vision.plan", build_vision_engine(weights, cfg,
                                                        verbose=bool(request.verbose)))
    writer.add_bytes("text.plan", build_text_engine(weights, cfg, verbose=bool(request.verbose)))
    writer.add_bytes("tokenizer.json", (model_dir / "tokenizer.json").read_bytes())
    writer.add_json(
        "runtime.json",
        {
            "patch_size": vision["patch_size"],
            "max_num_patches": vision["max_num_patches"],
            "position_grid": vision["grid"],
            "image_mean": cfg["image_mean"],
            "image_std": cfg["image_std"],
            "resample": cfg["resample"],
            "rescale_factor": cfg["rescale_factor"],
            "max_text_len": text["max_text_len"],
            "pad_token_id": pad_token_id,
            "embedding_dim": text["projection_size"],
            "logit_scale": logit_scale,
            "logit_bias": logit_bias,
        },
    )
