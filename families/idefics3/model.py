# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Idefics3 image tiling, native vision connector and causal text decoder."""

import json
from pathlib import Path

import tensorrt as trt

from .decoder_builder import build_decoder
from .vision_builder import build_vision_tower


def build_vision(weights, cfg, verbose):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    options = builder.create_builder_config()
    options.builder_optimization_level = 3
    options.clear_flag(trt.BuilderFlag.TF32)
    vision = cfg["vision_config"]
    grid = vision["image_size"] // vision["patch_size"]
    graph_config = dict(
        hidden_size=vision["hidden_size"],
        patch_size=vision["patch_size"],
        grid=grid,
        num_positions=grid * grid,
        num_layers=vision["num_hidden_layers"],
        num_heads=vision["num_attention_heads"],
        head_dim=vision["hidden_size"] // vision["num_attention_heads"],
        eps=vision["layer_norm_eps"],
        scale_factor=cfg["scale_factor"],
    )
    pixels = network.add_input(
        "pixel_values", trt.float32, (1, 3, vision["image_size"], vision["image_size"])
    )
    output = build_vision_tower(network, pixels, weights, graph_config)
    output.name = "image_features"
    network.mark_output(output)
    plan = builder.build_serialized_network(network, options)
    if plan is None:
        raise RuntimeError("Idefics3 vision engine build failed")
    return bytes(plan)


def build(request, writer):
    if request.task != "images_text_to_text" or request.backend != "trt":
        raise ValueError("Idefics3 requires images_text_to_text with backend=trt")
    if (
        request.precision != "fp32"
        or request.tensor_parallel_size != 1
        or request.context_parallel_size != 1
        or request.max_batch_size != 1
        or request.quantization not in (None, "none")
        or request.fp32_layers
        or request.dynamic_kv_cache
        or request.image_height is not None
        or request.image_width is not None
        or request.video_num_frames is not None
        or request.mtp_seq_len is not None
    ):
        raise NotImplementedError(
            "Idefics3 supports one fp32 image/question with default checkpoint tiling"
        )
    directory = Path(request.model_dir)
    cfg = json.loads((directory / "config.json").read_text())
    text = cfg["text_config"]
    vision = cfg["vision_config"]
    preprocessing = json.loads((directory / "preprocessor_config.json").read_text())
    if (
        text["model_type"] != "llama"
        or text.get("rope_scaling") is not None
        or text.get("attention_bias", False)
        or text.get("mlp_bias", False)
        or text["hidden_act"] != "silu"
        or vision["hidden_act"] != "gelu_pytorch_tanh"
        or not preprocessing["do_image_splitting"]
        or preprocessing["resample"] != 1
        or preprocessing["image_mean"] != [0.5, 0.5, 0.5]
        or preprocessing["image_std"] != [0.5, 0.5, 0.5]
    ):
        raise ValueError("Unsupported Idefics3 decoder or image preprocessing configuration")
    length = request.max_sequence_length or 1536
    if not 2 <= length <= text["max_position_embeddings"]:
        raise ValueError("Idefics3 sequence profile exceeds the decoder context")
    from safetensors import safe_open

    with safe_open(str(directory / "model.safetensors"), framework="pt", device="cpu") as handle:
        weights = {key: handle.get_tensor(key).float().numpy() for key in handle.keys()}
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(directory, use_fast=False)
    sentinel = "TRTMC_QUESTION_PLACEHOLDER"
    template = processor.apply_chat_template(
        [dict(role="user", content=[dict(type="image"), dict(type="text", text=sentinel)])],
        add_generation_prompt=True,
    )
    if template.count("<image>") != 1 or template.count(sentinel) != 1:
        raise ValueError(
            "Idefics3 requires one image followed by one question in its chat template"
        )
    prefix, rest = template.split("<image>")
    between, suffix = rest.split(sentinel)
    generation = json.loads((directory / "generation_config.json").read_text())
    eos = generation["eos_token_id"]
    eos = eos if isinstance(eos, list) else [eos]
    image_size = vision["image_size"]
    image_sequence = (image_size // vision["patch_size"] // cfg["scale_factor"]) ** 2
    writer.set_header(family="idefics3", task=request.task, backend=request.backend)
    writer.add_bytes("vision.plan", build_vision(weights, cfg, request.verbose))
    writer.add_bytes("decoder.plan", build_decoder(weights, text, length, request.verbose))
    writer.add_bytes("tokenizer.json", processor.tokenizer.backend_tokenizer.to_str().encode())
    writer.add_json(
        "runtime.json",
        dict(
            sequence_length=length,
            hidden_size=text["hidden_size"],
            vocab_size=text["vocab_size"],
            pad_id=processor.tokenizer.pad_token_id,
            eos_ids=eos,
            image_id=cfg["image_token_id"],
            image_sequence=image_sequence,
            image_size=image_size,
            longest_edge=preprocessing["size"]["longest_edge"],
            chat_prefix=prefix,
            between_image_text=between,
            chat_suffix=suffix,
        ),
    )
