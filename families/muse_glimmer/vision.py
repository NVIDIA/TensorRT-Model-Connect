# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse-owned media source and dynamic profile contract."""
import json
from pathlib import Path

MEDIA_TASKS = ("images_text_to_text", "video_text_to_text")


def media_contract(request, raw, weight_format, capacity):
    if request.task not in MEDIA_TASKS:
        return None
    if weight_format != "floating":
        raise ValueError("Muse-Glimmer image/video requires the original floating checkpoint")
    path = Path(request.model_dir)
    processor = json.loads((path / "processor_config.json").read_text(encoding="utf-8"))
    image = processor["image_processor"]
    video = processor["video_processor"]
    v = raw["vision_config"]
    if (capacity < 16 or v["patch_size"] != 14 or v["patch_temporal"] != 2 or v["merge_size"] != 2
            or v["pos_emb_height"] != v["pos_emb_width"] or v["num_hidden_layers"] != 50
            or v["hidden_size"] != 1536 or v["num_attention_heads"] != 16
            or raw["text_config"]["hidden_size"] != 6656):
        raise ValueError("Unsupported Muse-Glimmer visual geometry or capacity")
    for entry in (image, video):
        if (entry["image_mean"] != [0.5] * 3 or entry["image_std"] != [0.5] * 3
                or entry["patch_size"] != 14 or entry["merge_size"] != 2
                or entry["temporal_patch_size"] != 2):
            raise ValueError("Unsupported Muse-Glimmer pixel processing configuration")
    weights = json.loads((path / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    required = {"model.vision_adapter.fc1.weight", "model.vision_adapter.fc2.weight",
                "model.vision_projection.weight", "model.vision_tower.patch_embedder.patch_embedding.weight"}
    required.update(f"model.vision_tower.layers.{i}.attn.q_proj.weight" for i in range(v["num_hidden_layers"]))
    if not required.issubset(weights):
        raise ValueError("Muse-Glimmer checkpoint does not contain its visual encoder")
    # Save half of the sequence budget for the rest of the prompt and answer.
    per_image = min(int(image["max_image_tokens"]), capacity // 2)
    frame_limit = min(int(video["max_video_frame_tokens"]), per_image)
    fps = float(video["fps"])
    if min(per_image, frame_limit) < 4 or not 0 < fps < float("inf"):
        raise ValueError("Invalid Muse-Glimmer visual profile or frame rate")
    return {"patch_size": 14, "table_side": v["pos_emb_height"], "head_dim": v["hidden_size"] // v["num_attention_heads"],
            "text_width": raw["text_config"]["hidden_size"], "max_patches": 4 * per_image,
            "max_image_tokens": per_image, "max_video_frame_tokens": frame_limit,
            "image_token_id": raw["image_token_id"], "video_token_id": raw["video_token_id"],
            "rope_theta": v["rope_parameters"]["rope_theta"], "fps": fps}
