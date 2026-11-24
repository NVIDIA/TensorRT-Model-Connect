# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish Bolt's source-independent native TRT plans and only the assets used at runtime."""

import json
import sys
from pathlib import Path

import numpy as np

from .config import validate_inputs
from .native_language import build_plan
from .native_weights import Weights


def build_native(request, writer):
    import tensorrt as trt

    draft = getattr(request, "draft", None)
    root = request.model_dir
    cfg = validate_inputs(root, draft)
    text = cfg["llm_config"]
    capacity = request.max_sequence_length or 8192
    kinds = [block["block_type"] for block in text["block_configs"]]
    weights = Weights(root)
    logger = trt.Logger(trt.Logger.WARNING)
    writer.add_bytes("notices/gguf.txt", Path(__file__).with_name("NOTICE").read_bytes())
    writer.add_json(
        "native.json",
        {
            "version": 1,
            "model_type": text["model_type"],
            "precision": "bf16",
            "execution_variant": "lilicorr" if draft is not None else "autoregressive",
            "block_types": kinds,
            "hidden": text["hidden_size"],
            "vocab": text["vocab_size"],
            "capacity": capacity,
            "max_patches": cfg["vision_config"]["max_num_patches"],
            "min_patches": cfg["vision_config"]["min_num_patches"],
            "pruning": cfg["video_pruning_rate"],
            "image_token": cfg["img_context_token_id"],
            "video_patches": json.loads((root / "preprocessor_config.json").read_text())[
                "video_target_num_patches"
            ],
        },
    )
    generation = json.loads((root / "generation_config.json").read_text())
    if set(generation["eos_token_id"]) != {2, 11}:
        raise ValueError("Unexpected Bolt EOS tokens")
    generation["eos_token_id"] = [2, 11]
    writer.add_json("generation.json", generation)
    for name in ("tokenizer.json", "chat_template.jinja"):
        writer.add_bytes(name, (root / name).read_bytes())
    embed = weights.tensor("language_model.backbone.embeddings.weight")
    if embed.shape != (text["vocab_size"], text["hidden_size"]) or embed.dtype.name != "bfloat16":
        raise ValueError("Unexpected Bolt token embedding geometry or dtype")
    writer.add_bytes("embeddings.bf16", memoryview(embed.view(np.uint16)))
    for index in [*range(len(kinds)), "head"]:
        key = "head" if index == "head" else f"{index:02d}"
        if request.verbose:
            kind = "output" if index == "head" else kinds[index]
            print(f"Building Bolt language plan {key} ({kind})", file=sys.stderr, flush=True)
        plan = build_plan(weights, text, index, capacity, logger=logger)
        if request.verbose:
            print(f"Serialized Bolt language plan {key}", file=sys.stderr, flush=True)
        writer.add_bytes(f"language/{key}.plan", memoryview(plan))
        del plan
    if request.task == "vision_language_generation":
        from .native_vision import build_patch, build_visual

        maximum = cfg["vision_config"]["max_num_patches"]
        for video in (False, True):
            key = "video_patch" if video else "image_patch"
            writer.add_bytes(f"vision/{key}.plan", build_patch(cfg, weights, maximum, video=video))
        writer.add_bytes("vision/visual.plan", build_visual(cfg, weights, maximum))
        positions = weights.tensor("vision_model.radio_model.model.patch_generator.pos_embed")
        if positions.shape != (1, 128 * 128, 1280):
            raise ValueError("Unexpected Bolt vision positions")
        writer.add_bytes("positions.f32", memoryview(positions.astype(np.float32)))

    if draft is not None:
        from .native_draft import build_projection, build_vocabulary
        from .native_lattice import build_lattice
        from .native_proposal import build_cache, build_proposal

        companion = json.loads((draft / "config.json").read_text())
        prepared = Weights(draft)
        for name, factory, source in [
            ("project", build_projection, prepared),
            ("vocab", build_vocabulary, weights),
            ("cache", build_cache, prepared),
            ("lattice", build_lattice, prepared),
        ]:
            plan = factory(source, companion, logger=logger)
            writer.add_bytes(f"draft/{name}.plan", memoryview(plan))
            del plan
        for layer in range(companion["num_hidden_layers"]):
            plan = build_proposal(prepared, companion, layer, logger=logger)
            writer.add_bytes(f"draft/{layer}.plan", memoryview(plan))
            del plan
