# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-to-bundle build for native request-conditioned CosyVoice3."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from types import SimpleNamespace
import gc
import shutil
import tempfile


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build all six learned engines; reference voices are supplied at runtime."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("cosyvoice3 does not support dynamic_kv_cache")
    if request.task != "audio_generation":
        raise ValueError("cosyvoice3 supports only task=audio_generation")
    if request.precision != "fp32":
        raise ValueError("cosyvoice3 supports only fp32 precision")
    if request.backend != "trt":
        raise ValueError("cosyvoice3 supports only the TensorRT backend")
    if request.image_height is not None or request.image_width is not None:
        raise NotImplementedError("cosyvoice3 does not accept image dimensions")
    if request.video_num_frames is not None:
        raise NotImplementedError("cosyvoice3 does not accept video frames")
    if request.max_batch_size != 1:
        raise NotImplementedError("cosyvoice3 supports only max_batch_size=1")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("cosyvoice3 does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("cosyvoice3 does not support context parallelism")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("cosyvoice3 does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("cosyvoice3 does not support mixed-precision layers")

    total_tokens = request.max_sequence_length or 512
    if not 16 <= total_tokens <= 2048:
        raise ValueError(
            "cosyvoice3 max_sequence_length must be 16..2048 combined reference/target speech tokens"
        )
    writer.set_header(family="cosyvoice3", task=request.task, backend=request.backend)
    _build_bundle(Path(request.model_dir), total_tokens, writer, verbose=request.verbose)


def _build_bundle(model_dir, total_tokens, writer, *, verbose=False):
    from .__main__ import build as build_flow, build_conditioner, build_speech_component
    from .config import MODEL_REVISION, read_config
    from .reference import coefficients
    from .tts import text_tokenizer
    import torch

    read_config(model_dir)
    context = total_tokens * 2 + 256
    with tempfile.TemporaryDirectory(prefix="cosyvoice3-build-") as temporary:
        root = Path(temporary)
        common = dict(model_dir=model_dir, workspace_mib=256)
        jobs = [
            (
                "llm",
                build_speech_component,
                dict(command="build-llm", max_context=context, opt_tokens=64),
            ),
            (
                "conditioning",
                build_conditioner,
                dict(min_tokens=2, opt_tokens=min(32, total_tokens), max_tokens=total_tokens),
            ),
            (
                "flow",
                build_flow,
                dict(
                    min_frames=4, opt_frames=min(64, total_tokens * 2), max_frames=total_tokens * 2
                ),
            ),
            (
                "hift",
                build_speech_component,
                dict(
                    command="build-hift",
                    min_frames=4,
                    opt_frames=min(64, total_tokens * 2),
                    max_frames=total_tokens * 2,
                ),
            ),
            (
                "campplus",
                build_speech_component,
                dict(command="build-campplus", min_frames=4, opt_frames=500, max_frames=3000),
            ),
            (
                "speech_tokenizer",
                build_speech_component,
                dict(
                    command="build-speech-tokenizer", min_frames=4, opt_frames=500, max_frames=3000
                ),
            ),
        ]
        for name, builder, options in jobs:
            directory = root / name
            if verbose:
                print(f"CosyVoice3: building {name} with {options}", flush=True)
            builder(SimpleNamespace(**common, **options, output=directory))
            manifest = (directory / "manifest.json").read_bytes()
            writer.add_bytes(name + ".manifest.json", manifest)
            with (
                (directory / (name + ".plan")).open("rb") as source,
                writer.open_section(name + ".plan") as target,
            ):
                shutil.copyfileobj(source, target)
            gc.collect()
            torch.cuda.empty_cache()
        writer.add_json("frontend.json", coefficients())
        writer.add_bytes(
            "tokenizer.json", text_tokenizer(model_dir).backend_tokenizer.to_str().encode()
        )
        writer.add_json(
            "config.json",
            dict(
                cosyvoice3_schema=2,
                precision="fp32",
                model_revision=MODEL_REVISION,
                reference_voice="per_request",
                cosyvoice3=dict(
                    instruction="You are a helpful assistant.",
                    transcript="",
                    greedy=False,
                    max_context=context,
                    max_tokens=total_tokens,
                    total_tokens=total_tokens,
                ),
            ),
        )
