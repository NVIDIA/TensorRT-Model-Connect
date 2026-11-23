# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build four native components of the publisher's FP8 Ideogram4 pipeline."""

import gc
import json
import tempfile
from pathlib import Path


def build(request, writer):
    if request.task != "text_to_image" or request.backend != "trt" or request.precision != "bf16":
        raise ValueError("ideogram4 supports text_to_image with backend=trt and precision=bf16")
    if (
        request.max_batch_size != 1
        or request.tensor_parallel_size != 1
        or request.context_parallel_size != 1
    ):
        raise NotImplementedError("ideogram4 supports batch one on a single GPU")
    if request.dynamic_kv_cache:
        raise NotImplementedError("ideogram4 does not support dynamic_kv_cache")
    if request.quantization not in (None, "none") or request.fp32_layers:
        raise NotImplementedError(
            "ideogram4 uses the checkpoint's weight-only FP8 format with BF16 computation"
        )
    if request.video_num_frames is not None or request.mtp_seq_len is not None:
        raise NotImplementedError("ideogram4 does not support video or speculative decoding")
    root = Path(request.model_dir)
    index = json.loads((root / "model_index.json").read_text())
    if index.get("_class_name") != "Ideogram4Pipeline":
        raise ValueError("ideogram4 requires an Ideogram4Pipeline checkpoint")
    text = json.loads((root / "text_encoder/config.json").read_text())
    if not text.get("ideogram_fp8_weight_only") or text.get("model_type") != "qwen3_vl":
        raise ValueError("ideogram4 requires the publisher's weight-only FP8 Qwen3-VL checkpoint")
    height = request.image_height if request.image_height is not None else 1024
    width = request.image_width if request.image_width is not None else 1024
    if any(type(v) is not int or v < 256 or v % 16 for v in (height, width)):
        raise ValueError("ideogram4 image dimensions must be multiples of 16, at least 256")
    length = request.max_sequence_length if request.max_sequence_length is not None else 256
    if type(length) is not int or not 1 <= length <= 2048:
        raise ValueError("ideogram4 text sequence length must be in [1, 2048]")
    for component in ("transformer", "unconditional_transformer"):
        config = json.loads((root / component / "config.json").read_text())
        if (
            config.get("_class_name") != "Ideogram4Transformer2DModel"
            or config.get("in_channels") != 128
        ):
            raise ValueError("ideogram4 requires its 128-channel asymmetric denoisers")
    import torch
    from ideogram4.sampler_configs import PRESETS
    from ideogram4.scheduler import get_schedule_for_resolution, make_step_intervals
    from . import graphs

    torch.set_num_threads(4)
    preset = PRESETS["V4_DEFAULT_20"]
    schedule = get_schedule_for_resolution((height, width), known_mean=preset.mu, std=preset.std)
    times = schedule(make_step_intervals(preset.num_steps)).tolist()
    tokenizer = (root / "tokenizer/tokenizer.json").read_bytes()
    writer.set_header(family="ideogram4", task=request.task, backend=request.backend)
    writer.add_bytes("tokenizer.json", tokenizer)
    runtime = {
        "height": height,
        "width": width,
        "text_length": length,
        "latent_channels": 128,
        "features": 53248,
        "precision": "bf16",
        "sampler": "V4_DEFAULT_20",
        "times": times,
        "guidance": list(preset.guidance_schedule),
    }
    writer.add_json("runtime.json", runtime)
    count = (height // 16) * (width // 16)
    with tempfile.TemporaryDirectory(prefix="trtmc-ideogram4-") as temporary:
        work = Path(temporary)
        # Text encoder padding is on the left; image positions never enter its causal keys.
        inputs = (
            torch.zeros((1, length), dtype=torch.int32),
            torch.ones((1, length), dtype=torch.int32),
            torch.arange(length, dtype=torch.int32).reshape(1, -1),
        )
        module = graphs.text_encoder(root)
        plan = graphs.compile_engine(
            module,
            inputs,
            ["token_ids", "token_mask", "positions"],
            "features",
            work,
            bool(request.verbose),
        )
        writer.add_bytes("text.plan", plan)
        del module, plan, inputs
        gc.collect()
    for component, prefix, size in (
        ("transformer", "positive", length + count),
        ("unconditional_transformer", "negative", count),
    ):
        with tempfile.TemporaryDirectory(prefix=f"trtmc-ideogram4-{prefix}-") as temporary:
            module = graphs.denoiser(root, component)
            inputs = (
                torch.zeros((1, size, 53248), dtype=torch.float32),
                torch.zeros((1, size, 128), dtype=torch.float32),
                torch.full((1,), 0.5, dtype=torch.float32),
                torch.zeros((1, size, 3), dtype=torch.int32),
                torch.ones((1, size), dtype=torch.int32),
                torch.full((1, size), 2, dtype=torch.int32),
            )
            plan = graphs.compile_engine(
                module,
                inputs,
                ["features", "latents", "time", "positions", "segments", "roles"],
                "velocity",
                Path(temporary),
                bool(request.verbose),
            )
            writer.add_bytes(f"{prefix}.plan", plan)
            del module, plan, inputs
            gc.collect()
    with tempfile.TemporaryDirectory(prefix="trtmc-ideogram4-vae-") as temporary:
        module = graphs.decoder(root, height // 16, width // 16)
        inputs = (torch.zeros((1, count, 128), dtype=torch.float32),)
        plan = graphs.compile_engine(
            module, inputs, ["latents"], "pixels", Path(temporary), bool(request.verbose)
        )
        writer.add_bytes("decoder.plan", plan)
