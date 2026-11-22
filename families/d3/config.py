# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate the five released dense D3 profiles and their decision contract."""
import json
import math
from .support import describe, profile_name, PROFILES
from tensorrt_model_connect.model_support import load_model_metadata


def read_config(request, options=None):
    if options is None:
        names = ("family", "task", "backend", "precision", "tensor_parallel_size",
                 "context_parallel_size", "quantization", "fp32_layers", "dynamic_kv_cache",
                 "video_num_frames", "mtp_seq_len", "max_batch_size", "max_sequence_length",
                 "image_height", "image_width")
        options = {name: getattr(request, name) for name in names}
    if (options["family"] != "d3" or options["task"] != "structured_decision"
            or options["backend"] != "trt" or options["precision"] != "bf16"):
        raise ValueError("D3 requires family=d3, structured_decision, trt and bf16")
    if (options["tensor_parallel_size"] != 1 or options["context_parallel_size"] != 1
            or options["quantization"] is not None
            or options["fp32_layers"] or options["dynamic_kv_cache"]
            or options["video_num_frames"] is not None or options["mtp_seq_len"] is not None):
        raise ValueError("D3 supports TP=CP=1 without quantization, video or MTP")
    if not 1 <= options["max_batch_size"] <= 8:
        raise ValueError("D3 batch capacity must be between 1 and 8")
    metadata = load_model_metadata(request.model_dir)
    if describe(metadata) is None:
        raise ValueError("not a supported D3 checkpoint")
    c = metadata.config
    if c.get("quantization_config") is not None:
        raise ValueError("D3 does not accept prequantized checkpoints")
    processor = json.loads((request.model_dir / "preprocessor_config.json").read_text())
    if (processor.get("image_mean") != [.5, .5, .5]
            or processor.get("image_std") != [.5, .5, .5]
            or processor.get("patch_size") != 16
            or processor.get("temporal_patch_size") != 2
            or processor.get("merge_size") != 2):
        raise ValueError("unsupported D3 image preprocessing")
    t = c["text_config"]
    expected = dict(PROFILES[profile_name(c)]["text"], hidden_act="silu",
                    attention_bias=False, attn_output_gate=True, rms_norm_eps=1e-6)
    if any(t.get(k) != v for k, v in expected.items()):
        raise ValueError("unsupported D3 text topology")
    if t.get("output_gate_type") is not None or t.get("mlp_only_layers") != []:
        raise ValueError("D3 requires its Qwen3.5 topology")
    vision = c["vision_config"]
    visual_expected = dict(PROFILES[profile_name(c)]["vision"], in_channels=3,
        patch_size=16, temporal_patch_size=2, spatial_merge_size=2,
        num_position_embeddings=2304, hidden_act="gelu_pytorch_tanh",
        deepstack_visual_indexes=[])
    if (any(vision.get(k) != v for k, v in visual_expected.items())
            or vision.get("rope_parameters") != {"rope_theta": 10000.0, "rope_type": "axial"}):
        raise ValueError("unsupported D3 vision topology")
    if c.get("dtype") != "bfloat16" or t.get("dtype") != "bfloat16":
        raise ValueError("D3 requires original BF16 backbone weights")
    rope = t["rope_parameters"]
    if (rope.get("rope_type") != "default" or rope.get("mrope_section") != [11, 11, 10]
            or rope.get("partial_rotary_factor") != .25 or not rope.get("mrope_interleaved")
            or rope.get("rope_theta") != 10000000):
        raise ValueError("unsupported D3 rotary configuration")
    d = json.loads((request.model_dir / "decision_config.json").read_text())
    for k, v in {"format_version": 1, "format_id": "d3-code-readout-v1", "prompt": "d3",
                 "attention_mode": "noncausal_full_attention", "pooling": "last",
                 "readout_dtype": "float32", "max_length": None}.items():
        if d.get(k) != v:
            raise ValueError(f"unsupported D3 decision field: {k}")
    if d.get("prompt_layout") is not None or d.get("temperature_by_questions") is not None:
        raise ValueError("D3 does not support alternate prompt/temperature policies")
    temperature = d.get("temperature")
    if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("D3 temperature must be finite and positive")
    if (len(d.get("codes", [])) != 255 or len(set(d["codes"])) != 255
            or len(d.get("token_ids", [])) != 255 or len(set(d["token_ids"])) != 255):
        raise ValueError("D3 requires 255 unique answer codes and token IDs")
    maximum = options["max_sequence_length"] or 2048
    if maximum > t["max_position_embeddings"]:
        raise ValueError("sequence capacity exceeds checkpoint position capacity")
    # Initial image profile accepts already-sized RGB images; never silently resize.
    height, width = options["image_height"] or 256, options["image_width"] or 256
    if height % 32 or width % 32 or not 65536 <= height * width <= 1638400:
        raise ValueError("D3 image shape must be a multiple of 32 within checkpoint pixel limits")
    return c, d, maximum, height, width
