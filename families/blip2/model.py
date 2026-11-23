# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BLIP-2 OPT image/question generation with the published query-only Q-Former."""

import json
from pathlib import Path

from .builders import build_decoder, build_vision


def build(request, writer):
    if request.task != "images_text_to_text" or request.backend != "trt":
        raise ValueError("BLIP-2 requires images_text_to_text with backend=trt")
    if request.dynamic_kv_cache:
        raise NotImplementedError("blip2 does not support dynamic_kv_cache")
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
            "BLIP-2 supports one fp32 image/question at its native image resolution"
        )
    from transformers import AutoConfig, AutoProcessor
    from safetensors import safe_open

    directory = Path(request.model_dir)
    cfg = AutoConfig.from_pretrained(directory).to_dict()
    text, vision, qformer = [cfg[name + "_config"] for name in ("text", "vision", "qformer")]
    preprocessing = json.loads((directory / "preprocessor_config.json").read_text())
    if (
        text["model_type"] != "opt"
        or text["activation_function"] != "relu"
        or not text["do_layer_norm_before"]
        or text["_remove_final_layer_norm"]
        or text["word_embed_proj_dim"] != text["hidden_size"]
        or not text["tie_word_embeddings"]
        or not text["layer_norm_elementwise_affine"]
        or vision["hidden_act"] != "gelu"
        or qformer["hidden_act"] != "gelu"
        or qformer["use_qformer_text_input"]
        or not preprocessing.get("do_resize", True)
        or not preprocessing.get("do_rescale", True)
        or preprocessing.get("rescale_factor", 1 / 255) != 1 / 255
        or not preprocessing.get("do_normalize", True)
        or preprocessing["resample"] != 3
        or preprocessing["size"] != dict(height=vision["image_size"], width=vision["image_size"])
        or preprocessing["image_mean"] != [0.48145466, 0.4578275, 0.40821073]
        or preprocessing["image_std"] != [0.26862954, 0.26130258, 0.27577711]
    ):
        raise ValueError("Unsupported BLIP-2 language model, Q-Former, or image preprocessing")
    length = request.max_sequence_length or 128
    if not cfg["num_query_tokens"] + 3 <= length <= text["max_position_embeddings"]:
        raise ValueError("BLIP-2 sequence profile must fit image queries and text")
    index = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    weights = {}
    for shard in sorted(set(index.values())):
        with safe_open(str(directory / shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                weights[key] = handle.get_tensor(key).float().numpy()
    processor = AutoProcessor.from_pretrained(directory, use_fast=False)
    generation = json.loads((directory / "generation_config.json").read_text())
    eos = generation["eos_token_id"]
    eos = eos if isinstance(eos, list) else [eos]
    writer.set_header(family="blip2", task=request.task, backend=request.backend)
    writer.add_bytes("vision.plan", build_vision(weights, cfg, request.verbose))
    writer.add_bytes("decoder.plan", build_decoder(weights, cfg, length, request.verbose))
    writer.add_bytes("tokenizer.json", processor.tokenizer.backend_tokenizer.to_str().encode())
    writer.add_json(
        "runtime.json",
        dict(
            sequence_length=length,
            hidden_size=text["hidden_size"],
            vocab_size=text["vocab_size"],
            pad_id=generation["pad_token_id"],
            eos_ids=eos,
            bos_id=generation["bos_token_id"],
            image_id=cfg["image_token_index"],
            image_sequence=cfg["num_query_tokens"],
            image_size=vision["image_size"],
        ),
    )
