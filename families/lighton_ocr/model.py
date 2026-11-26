# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public build entry point for single-image LightOn OCR."""

from __future__ import annotations

import gc
import json
import math
from pathlib import Path

from .config import Config
from .weights import Checkpoint


def _prompt_tokens(directory, config):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}]}],
        add_generation_prompt=True, tokenize=True, return_dict=False,
    )
    if ids.count(config.image_token_id) != 1:
        raise ValueError("lighton_ocr requires one image placeholder in the checkpoint's image-only chat template")
    image = ids.index(config.image_token_id)
    if any(type(value) is not int or not 0 <= value < config.text_config["vocab_size"] for value in ids):
        raise ValueError("lighton_ocr chat template emitted invalid token IDs")
    return ids[:image], ids[image + 1:]


def _processor_config(directory, config):
    processor_path = directory / "processor_config.json"
    if processor_path.is_file():
        source = json.loads(processor_path.read_text()).get("image_processor", {})
    else:
        source = json.loads((directory / "preprocessor_config.json").read_text())
    expected = {"do_resize": True, "do_rescale": True, "do_normalize": True, "do_convert_rgb": True}
    if any(source.get(name) is not value for name, value in expected.items()) or source.get("resample") != 3:
        raise ValueError("lighton_ocr requires the checkpoint's RGB bicubic and normalized image preprocessing")
    patch = source.get("patch_size")
    if isinstance(patch, dict):
        if patch.get("height") != patch.get("width"):
            raise ValueError("lighton_ocr requires square image patches")
        patch = patch.get("height")
    maximum = source.get("size", {}).get("longest_edge")
    if patch != config.vision_config["patch_size"] or maximum != config.vision_config["image_size"]:
        raise ValueError("lighton_ocr image processor and vision encoder dimensions disagree")
    mean, std = source.get("image_mean"), source.get("image_std")
    if not isinstance(mean, list) or not isinstance(std, list) or len(mean) != 3 or len(std) != 3:
        raise ValueError("lighton_ocr requires three image normalization channels")
    scalars = mean + std + [source.get("rescale_factor", 0)]
    if any(isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) for value in scalars):
        raise ValueError("lighton_ocr image normalization values must be finite numbers")
    if any(float(value) <= 0 for value in std) or float(source.get("rescale_factor", 0)) <= 0:
        raise ValueError("lighton_ocr requires positive image normalization scales")
    return {"patch_size": patch, "spatial_merge_size": config.spatial_merge_size,
            "max_image_size": maximum, "rescale_factor": source["rescale_factor"],
            "image_mean": mean, "image_std": std}


def runtime_config(directory, config, capacity):
    """Derive native request framing and image policy from original assets."""
    prefix, suffix = _prompt_tokens(directory, config)
    generation = json.loads((directory / "generation_config.json").read_text())
    eos = generation.get("eos_token_id", config.eos_token_id)
    eos = eos if isinstance(eos, list) else [eos]
    if not eos or any(type(value) is not int or not 0 <= value < config.text_config["vocab_size"] for value in eos):
        raise ValueError("lighton_ocr generation metadata has invalid EOS tokens")
    if capacity <= len(prefix) + len(suffix) + 1:
        raise ValueError("lighton_ocr sequence capacity must fit the image, prompt framing, and output")
    text = config.text_config
    return {"format_version": 1, "precision": "fp32", "max_sequence_length": capacity,
            "max_new_tokens": min(1024, capacity), "vocab_size": text["vocab_size"], "hidden_size": text["hidden_size"],
            "num_hidden_layers": text["num_hidden_layers"], "num_key_value_heads": text["num_key_value_heads"],
            "head_dim": text["head_dim"], "image_token_id": config.image_token_id,
            "eos_token_ids": eos, "prompt_prefix_ids": prefix, "prompt_suffix_ids": suffix,
            "image_processor": _processor_config(directory, config),
            "text_config": text, "vision_config": config.vision_config}


def build(request, writer):
    if request.family != "lighton_ocr":
        raise ValueError("lighton_ocr builds only its own model family")
    if request.task != "images_text_to_text":
        raise ValueError("lighton_ocr builds the images_text_to_text task")
    if request.backend != "trt" or request.precision != "fp32":
        raise ValueError("lighton_ocr requires the TensorRT backend and FP32 compute precision")
    parallelism = (request.max_batch_size, request.tensor_parallel_size, request.context_parallel_size)
    if parallelism != (1, 1, 1) or any(type(value) is not int for value in parallelism):
        raise ValueError("lighton_ocr supports batch 1, TP1, and CP1")
    if request.dynamic_kv_cache:
        raise NotImplementedError("lighton_ocr does not support dynamic_kv_cache")
    if request.quantization not in {None, "none"}:
        raise ValueError("lighton_ocr requires unquantized weights and its compact request-owned K/V cache")
    if request.fp32_layers:
        raise ValueError("lighton_ocr uses FP32 for the complete graph and does not accept layer overrides")
    profile_overrides = (request.image_height, request.image_width, request.video_num_frames, request.mtp_seq_len)
    if any(value is not None for value in profile_overrides):
        raise ValueError("lighton_ocr uses its checkpoint image profile and does not support video or MTP options")
    if request.graph_transform is not None:
        raise ValueError("lighton_ocr does not support external graph transformations")
    directory = Path(request.model_dir)
    config = Config.from_dir(directory)
    capacity = 4096 if request.max_sequence_length is None else request.max_sequence_length
    if type(capacity) is not int or not 1 < capacity <= config.text_config["max_position_embeddings"]:
        raise ValueError("lighton_ocr sequence capacity is outside the decoder's position range")
    if not (directory / "tokenizer.json").is_file():
        raise ValueError("lighton_ocr requires tokenizer.json in the checkpoint")
    runtime = runtime_config(directory, config, capacity)
    checkpoint = Checkpoint(directory, config)
    checkpoint.validate()
    from .graph import build_decoder_engine
    from .vision import build_vision_engine

    writer.set_header(family="lighton_ocr", task=request.task, backend=request.backend)
    writer.add_json("runtime.json", runtime)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "processor_config.json",
                 "preprocessor_config.json", "generation_config.json", "chat_template.jinja"):
        path = directory / name
        if path.is_file():
            writer.add_bytes(name, path.read_bytes())
    embedding = checkpoint.embedding()
    writer.add_bytes("embedding.bin", embedding.tobytes())
    print("lighton_ocr: building the full Pixtral vision encoder and projector", flush=True)
    weights = checkpoint.vision()
    writer.add_bytes("vision.plan", build_vision_engine(config, weights, verbose=request.verbose))
    del weights
    gc.collect()
    print("lighton_ocr: building the full Qwen3 decoder and tied output head", flush=True)
    weights = checkpoint.decoder(embedding)
    writer.add_bytes("decoder.plan", build_decoder_engine(config, weights, capacity, verbose=request.verbose))
