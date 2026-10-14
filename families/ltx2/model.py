# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LTX-2.5 family plugin: text-to-audio-video bundles for Lightricks ``LTX2Pipeline``.

Builds a native TRTMC bundle from a diffusers LTX-2.5 checkpoint (distilled ``transformer/``):

  - ``text_encoder.plan``: Gemma 4 text tower + ``LTX2TextConnectors``
  - ``denoiser.plan``: the joint audio/video DiT, single device (``context_parallel_size=1``)
    or context parallel over the video tokens (``context_parallel_size=2``, one rank-dynamic plan)
  - ``vae.plan``: the video VAE decoder
  - ``audio.plan``: the audio VAE decoder + vocoder with bandwidth extension (48 kHz stereo)
  - ``tokenizer.json`` and ``runtime.json``

Every engine is built directly with the TensorRT network API in bf16 (the precision LTX-2.5 is
trained and released in). The runtime path is C++ + TensorRT (or TensorRT-RTX) only.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .parallel import ParallelConfig, validate_context_parallel_layout

if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter

TASK = "text_to_audio_video"
PIPELINE_CLASS = "LTX2Pipeline"

# diffusers ``pipelines/ltx2/utils.py`` DISTILLED_SIGMA_VALUES: the distilled checkpoint's
# 8-step schedule. LTX-2.5's shipped scheduler config disables dynamic shifting, so the
# pipeline uses these values unshifted (timesteps = sigma * 1000), then the terminal 0.
DISTILLED_SIGMAS = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875)

DEFAULT_HEIGHT = 544  # the LTX-2.5 model card's 960x544, 121 frames at 24 fps
DEFAULT_WIDTH = 960
DEFAULT_FRAMES = 121
FRAME_RATE = 24.0
TEXT_SEQ_LEN = 1024
SPATIAL_COMPRESSION = 32
TEMPORAL_COMPRESSION = 8

# TensorRT-RTX builds before the rel-11.4 line (1.6.x and older) ship without multi-device
# (IDistCollectiveLayer) support; 1.7.1 is the first with it and with the Myelin communicator
# handoff that context parallelism needs.
MIN_RTX_FOR_CONTEXT_PARALLEL = (1, 7, 1)


def _version_tuple(text: str) -> tuple[int, ...]:
    parts = []
    for piece in text.split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def require_rtx_context_parallel(rtx_version: str | None, cp_size: int) -> None:
    """Fail early when a TensorRT-RTX build cannot run multi-device engines."""
    if cp_size <= 1:
        return
    found = rtx_version or "not installed"
    if rtx_version is None or _version_tuple(rtx_version)[:3] < MIN_RTX_FOR_CONTEXT_PARALLEL:
        raise RuntimeError(
            "LTX-2 context parallelism on TensorRT-RTX requires TensorRT-RTX >= 1.7.1 "
            f"(found {found}); 1.6.x ships without multi-device support"
        )


def installed_rtx_version() -> str | None:
    from importlib import metadata

    for name in ("tensorrt-rtx", "tensorrt_rtx"):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return None


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"LTX-2.5 checkpoint file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _log(message: str) -> None:
    print(f"[ltx2] {message}", file=sys.stderr, flush=True)


def _write_plan(writer: "BundleWriter", name: str, plan) -> None:
    with writer.open_section(name) as section:
        section.write(memoryview(plan))


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one LTX-2.5 text-to-audio-video bundle."""
    if request.task != TASK:
        raise ValueError(f"ltx2 supports only task={TASK}")
    if request.dynamic_kv_cache:
        raise NotImplementedError("ltx2 does not support dynamic_kv_cache")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("ltx2 requires tensor_parallel_size=1 (it shards the video tokens)")
    if request.max_batch_size != 1:
        raise NotImplementedError("ltx2 requires max_batch_size=1")
    if request.quantization not in (None, "none"):
        raise NotImplementedError("ltx2 does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("ltx2 does not support fp32_layers (its fp32 islands are fixed in the graph)")
    if request.precision != "bf16":
        raise ValueError("ltx2 builds bf16 engines (precision=bf16), the precision LTX-2.5 runs in")
    parallel = ParallelConfig(cp_size=int(request.context_parallel_size))
    if parallel.cp_size not in (1, 2):
        raise ValueError("ltx2 supports context_parallel_size 1 or 2")
    if request.backend == "trt_rtx":
        require_rtx_context_parallel(installed_rtx_version(), parallel.cp_size)

    # The builders import the bound TensorRT module; validate the request before loading them.
    from .audio_builder import build_audio_decoder_engine
    from .dit_builder import DiTConfig, DiTShape, audio_latent_frames, build_dit_engine
    from .text_encoder_builder import build_text_encoder_engine
    from .vae_builder import build_vae_decoder_engine

    model_dir = Path(request.model_dir)
    index = _read_json(model_dir / "model_index.json")
    if index.get("_class_name") != PIPELINE_CLASS:
        raise ValueError(f"expected a diffusers {PIPELINE_CLASS} checkpoint, got {index.get('_class_name')!r}")
    scheduler = _read_json(model_dir / "scheduler" / "scheduler_config.json")
    if scheduler.get("use_dynamic_shifting", False) or scheduler.get("shift_terminal"):
        raise NotImplementedError("ltx2 builds the distilled schedule; this scheduler config shifts sigmas")
    if float(scheduler.get("shift", 1.0)) != 1.0:
        raise NotImplementedError("ltx2 builds the distilled schedule with shift 1.0")
    vae_cfg = _read_json(model_dir / "vae" / "config.json")
    dit_cfg = DiTConfig.from_dict(_read_json(model_dir / "transformer" / "config.json"))
    audio_cfg = _read_json(model_dir / "audio_vae" / "config.json")
    vocoder_cfg = _read_json(model_dir / "vocoder" / "config.json")

    height = int(request.image_height or DEFAULT_HEIGHT)
    width = int(request.image_width or DEFAULT_WIDTH)
    frames = int(request.video_num_frames or DEFAULT_FRAMES)
    spatial = int(vae_cfg.get("spatial_compression_ratio", SPATIAL_COMPRESSION))
    temporal = int(vae_cfg.get("temporal_compression_ratio", TEMPORAL_COMPRESSION))
    if height % spatial or width % spatial:
        raise ValueError(f"ltx2 image_height/image_width must be multiples of {spatial}")
    if (frames - 1) % temporal:
        raise ValueError(f"ltx2 video_num_frames must equal {temporal}*n+1")
    text_len = int(request.max_sequence_length or TEXT_SEQ_LEN)
    shape = DiTShape(
        batch=1,
        latent_frames=(frames - 1) // temporal + 1,
        latent_height=height // spatial,
        latent_width=width // spatial,
        audio_frames=audio_latent_frames(frames, FRAME_RATE, sampling_rate=int(audio_cfg.get("sample_rate", 16000)),
                                         hop_length=int(audio_cfg.get("mel_hop_length", 160))),
        text_len=text_len,
        fps=FRAME_RATE,
    )
    validate_context_parallel_layout(parallel, video_tokens=shape.video_tokens, video_heads=dit_cfg.heads,
                                     audio_heads=dit_cfg.audio_heads)

    writer.set_header(family="ltx2", task=request.task, backend=request.backend)
    started = time.perf_counter()
    _write_plan(writer, "text_encoder.plan", build_text_encoder_engine(model_dir, seq_len=text_len,
                                                                       verbose=request.verbose))
    _log(f"text encoder engine built in {time.perf_counter() - started:.1f} s")
    started = time.perf_counter()
    _write_plan(writer, "denoiser.plan", build_dit_engine(model_dir / "transformer", shape, cp_size=parallel.cp_size,
                                                          verbose=request.verbose))
    _log(f"DiT engine built in {time.perf_counter() - started:.1f} s (cp={parallel.cp_size}, "
         f"{shape.video_tokens} video + {shape.audio_frames} audio tokens)")
    started = time.perf_counter()
    _write_plan(writer, "vae.plan", build_vae_decoder_engine(model_dir / "vae", latent_frames=shape.latent_frames,
                                                             latent_height=shape.latent_height,
                                                             latent_width=shape.latent_width,
                                                             verbose=request.verbose))
    _log(f"video VAE engine built in {time.perf_counter() - started:.1f} s")
    started = time.perf_counter()
    _write_plan(writer, "audio.plan", build_audio_decoder_engine(model_dir, audio_frames=shape.audio_frames,
                                                                 verbose=request.verbose))
    _log(f"audio decoder engine built in {time.perf_counter() - started:.1f} s")
    writer.add_bytes("tokenizer.json", (model_dir / "tokenizer" / "tokenizer.json").read_bytes())
    writer.add_json("runtime.json", {
        "video_frames": frames,
        "video_height": height,
        "video_width": width,
        "latent_frames": shape.latent_frames,
        "latent_height": shape.latent_height,
        "latent_width": shape.latent_width,
        "latent_channels": dit_cfg.in_channels,
        "audio_frames": shape.audio_frames,
        "audio_latent_channels": dit_cfg.audio_in_channels,
        "text_seq_len": text_len,
        "frame_rate": FRAME_RATE,
        "pad_token_id": 0,
        "sigmas": [*DISTILLED_SIGMAS, 0.0],
        "dit_batch": 1,
        "audio_sample_rate": int(vocoder_cfg.get("output_sampling_rate", 48000)),
        "audio_channels": int(vocoder_cfg.get("out_channels", 2)),
        "parallel_mode": parallel.mode,
        "parallel_size": parallel.world_size,
    })
