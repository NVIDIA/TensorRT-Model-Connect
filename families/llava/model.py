# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build LLaVA 1.5 as a vision encoder and split Llama prefill/decode engines."""

from pathlib import Path
from .checkpoint_mapper import Checkpoint, load_decoder
from .config import ModelConfig


def build(request, writer):
    if request.task != "vision_language_generation" or request.backend != "trt":
        raise ValueError("LLaVA requires vision_language_generation with the TensorRT backend")
    if request.precision != "fp16" or request.quantization not in (None, "none"):
        raise ValueError("LLaVA currently supports unquantized FP16")
    if request.tensor_parallel_size != 1 or request.context_parallel_size != 1 or request.max_batch_size != 1:
        raise ValueError("LLaVA currently supports one device and one request per call")
    if request.dynamic_kv_cache:
        raise NotImplementedError("llava does not support dynamic_kv_cache")
    if request.fp32_layers or request.mtp_seq_len is not None:
        raise ValueError("LLaVA does not support cache, layer-precision, or speculative overrides")
    if any(value is not None for value in (request.image_height, request.image_width, request.video_num_frames)):
        raise ValueError("LLaVA uses the checkpoint's fixed image shape")
    directory = Path(request.model_dir)
    config = ModelConfig.from_dir(directory)
    maximum = request.max_sequence_length or config.max_position_embeddings
    vision_cfg = config.raw["vision_config"]
    image_tokens = (vision_cfg["image_size"] // vision_cfg["patch_size"]) ** 2
    if maximum <= image_tokens or maximum > config.max_position_embeddings:
        raise ValueError("LLaVA context must fit image tokens and stay within the checkpoint limit")
    checkpoint = Checkpoint(directory)
    weights = load_decoder(checkpoint, config)
    from .default_decoder import build_standard_decoder_engine
    from .default_dual_profile_decoder import build_dual_profile_decoder_engine
    from .vision_builder import build_vision

    writer.set_header(family="llava", task=request.task, backend=request.backend)
    prefill = build_dual_profile_decoder_engine(
        config, weights, maximum, precision="fp16", embed_input=True,
        opt_prefill_length=min(image_tokens + 64, maximum),
        profile_mode="prefill", verbose=request.verbose,
    )
    writer.add_bytes("prefill.plan", prefill)
    del prefill
    config.raw["_decoder_engine_role"] = "decode"
    decode = build_standard_decoder_engine(
        config, weights, maximum, precision="fp16", embed_input=True, verbose=request.verbose,
    )
    writer.add_bytes("engine.plan", decode)
    del decode, weights
    vision = build_vision(checkpoint, config, verbose=request.verbose)
    writer.add_bytes("vision.plan", vision)
    del vision

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(directory), use_fast=True)
    if tokenizer.encode("hello") != [config.bos_token_id] + tokenizer.encode("hello", add_special_tokens=False):
        raise ValueError("LLaVA requires a tokenizer with one leading BOS token")
    image_token = config.raw.get("image_token_index", 32000)
    if tokenizer.convert_tokens_to_ids("<image>") != image_token:
        raise ValueError("LLaVA image token ID does not match its tokenizer")
    runtime = {
        "tensor_parallel_size": 1, "num_layers": config.num_hidden_layers,
        "max_cache_length": maximum, "prefill_max_length": maximum,
        "vocab_size": config.vocab_size, "id_bos": config.bos_token_id,
        "id_eos": config.eos_token_id, "eos_token_ids": [config.eos_token_id],
        "image_token_id": image_token, "vision_output_dim": config.hidden_size,
        "fixed_image_size": vision_cfg["image_size"], "num_image_pad_tokens": image_tokens,
        "vl_prompt_template": "USER: {image_pads}\n{prompt} ASSISTANT:",
        "image_token_str": "<image>", "preprocessor_type": "clip_chw",
        "tokenizer_add_special_tokens": False,
        "tokenizer_prefix_ids": [config.bos_token_id], "tokenizer_suffix_ids": [],
        "io_map": {"cache_k_pattern": "cache_k_{i}", "cache_v_pattern": "cache_v_{i}",
                   "present_k_pattern": "present_k_{i}", "present_v_pattern": "present_v_{i}"},
    }
    writer.add_json("runtime.json", runtime)
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model", "preprocessor_config.json", "chat_template.jinja"):
        path = directory / name
        if path.is_file():
            writer.add_bytes(name, path.read_bytes())
