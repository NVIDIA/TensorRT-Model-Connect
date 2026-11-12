# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build BLIP captioning as an image encoder and a full-prefix causal decoder."""

import json
from pathlib import Path

import numpy as np
import tensorrt as trt

from .graph import Graph


def resolve(model_dir):
    root = Path(model_dir)
    cfg = json.loads((root / "config.json").read_text())
    processor = json.loads((root / "preprocessor_config.json").read_text())
    if cfg.get("model_type") != "blip" or cfg.get("architectures") != [
        "BlipForConditionalGeneration"
    ]:
        raise ValueError("blip requires BlipForConditionalGeneration metadata")
    vision, text = cfg["vision_config"], cfg["text_config"]
    for section in (vision, text):
        if section.get("hidden_act") != "gelu":
            raise ValueError("blip supports the checkpoint's exact gelu only")
        for key in ("hidden_size", "num_attention_heads", "num_hidden_layers", "intermediate_size"):
            if not isinstance(section[key], int) or section[key] <= 0:
                raise ValueError(f"blip has invalid {key}")
        if section["hidden_size"] % section["num_attention_heads"]:
            raise ValueError("blip attention heads do not divide hidden_size")
        if not np.isfinite(section["layer_norm_eps"]) or section["layer_norm_eps"] <= 0:
            raise ValueError("blip layer_norm_eps must be positive")
    size, patch = vision["image_size"], vision["patch_size"]
    if size <= 0 or patch <= 0 or size % patch or vision.get("num_channels", 3) != 3:
        raise ValueError("blip requires a square RGB patch grid")
    if not text.get("is_decoder") or text.get("position_embedding_type", "absolute") != "absolute":
        raise ValueError("blip requires an absolute-position causal text decoder")
    processed_size = processor.get("size")
    if isinstance(processed_size, dict):
        processed_size = (
            processed_size.get("height") if processed_size.get("width") == size else None
        )
    if processed_size != size or not processor.get("do_resize", True):
        raise ValueError("blip processor size must match the vision position table")
    if processor.get("resample", 3) != 3 or not processor.get("do_normalize", True):
        raise ValueError("blip requires bicubic resize and normalization")
    if not processor.get("do_rescale", True) or processor.get("rescale_factor", 1 / 255) != 1 / 255:
        raise ValueError("blip requires standard RGB rescaling")
    for name in ("image_mean", "image_std"):
        values = processor[name]
        if len(values) != 3 or not np.isfinite(values).all():
            raise ValueError("blip image normalization needs three finite values")
    if any(v <= 0 for v in processor["image_std"]):
        raise ValueError("blip image_std must be positive")
    return cfg, processor


def load_weights(model_dir):
    root = Path(model_dir)
    import torch

    # The public checkpoint is a state dict. Never load executable pickle objects.
    result = {
        k: v.float().numpy()
        for k, v in torch.load(
            root / "pytorch_model.bin", map_location="cpu", weights_only=True
        ).items()
    }
    result.setdefault(
        "text_decoder.cls.predictions.decoder.weight",
        result["text_decoder.bert.embeddings.word_embeddings.weight"],
    )
    result.setdefault(
        "text_decoder.cls.predictions.decoder.bias", result["text_decoder.cls.predictions.bias"]
    )
    return {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in result.items()}


def build_engine(weights, cfg, stage, max_length, verbose=False):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 3
    config.clear_flag(trt.BuilderFlag.TF32)
    graph = Graph(network, weights)
    if stage == "vision":
        out = graph.vision(cfg["vision_config"])
        out.name = "image_embeds"
    else:
        out = graph.decoder(cfg["text_config"], cfg["vision_config"])
        out.name = "logits"
        profile = builder.create_optimization_profile()
        opt = min(16, max_length)
        for name in ("input_ids", "position_ids"):
            profile.set_shape(name, (1,), (opt,), (max_length,))
        profile.set_shape("causal_mask", (1, 1, 1), (1, opt, opt), (1, max_length, max_length))
        config.add_optimization_profile(profile)
    network.mark_output(out)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"blip {stage} engine build failed")
    return bytes(plan)


def build(request, writer):
    if (
        request.task != "images_text_to_text"
        or request.precision != "fp32"
        or request.backend != "trt"
    ):
        raise ValueError("blip supports images_text_to_text, precision=fp32, backend=trt")
    if request.dynamic_kv_cache:
        raise NotImplementedError("blip does not support dynamic_kv_cache")
    if (
        request.max_batch_size != 1
        or request.tensor_parallel_size != 1
        or request.context_parallel_size != 1
        or request.quantization not in (None, "none")
        or request.fp32_layers
        or request.video_num_frames is not None
        or request.mtp_seq_len is not None
    ):
        raise ValueError("blip supports single-image, single-GPU unquantized builds")
    if request.graph_transform is not None:
        raise ValueError("blip does not support graph_transform")
    cfg, processor = resolve(request.model_dir)
    size = cfg["vision_config"]["image_size"]
    if any(v not in (None, size) for v in (request.image_height, request.image_width)):
        raise ValueError("blip image dimensions must match the checkpoint")
    max_length = request.max_sequence_length or 64
    if not 2 <= max_length <= cfg["text_config"]["max_position_embeddings"]:
        raise ValueError("blip sequence capacity exceeds the text position table")
    weights = load_weights(request.model_dir)
    writer.set_header(family="blip", task=request.task, backend=request.backend)
    for stage in ("vision", "decoder"):
        writer.add_bytes(
            stage + ".plan", build_engine(weights, cfg, stage, max_length, request.verbose)
        )
    writer.add_bytes("tokenizer.json", (Path(request.model_dir) / "tokenizer.json").read_bytes())
    writer.add_json(
        "runtime.json",
        {
            "image_size": size,
            "image_tokens": (size // cfg["vision_config"]["patch_size"]) ** 2 + 1,
            "vision_width": cfg["vision_config"]["hidden_size"],
            "vocab_size": cfg["text_config"]["vocab_size"],
            "max_length": max_length,
            "bos_token_id": cfg["text_config"]["bos_token_id"],
            # BLIP generate terminates at SEP, not text_config.eos_token_id (2).
            "eos_token_id": cfg["text_config"]["sep_token_id"],
            "image_mean": processor["image_mean"],
            "image_std": processor["image_std"],
        },
    )
