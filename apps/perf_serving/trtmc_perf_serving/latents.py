# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Latent replay: the same initial diffusion noise on TRTMC and on the Diffusers reference.

A ``generate_image`` request with ``latent_seed`` gets standard-normal noise drawn from that seed in
the pipeline's own (unpacked) latent shape, computed from the checkpoint's model_index.json,
transformer/config.json, and vae/config.json. The noise is written as a raw float32 file and passed
as ``initial_latents_path``: the TRTMC families read that layout directly (CHW / CTHW), and the
Diffusers adapter packs it where the pipeline expects packed latents. Both sides then denoise the
same starting point, so their outputs differ only by numerics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

FILE_NAME = "initial_latents.f32"


@dataclass(frozen=True)
class Layout:
    shape: Callable[[Mapping[str, Any], Mapping[str, Any], int, int, int], tuple[int, ...]]
    # The pipeline returns supplied latents as given but packs the ones it draws (Flux.1, Qwen-Image).
    pack: bool = False
    # The pipeline keeps supplied latents' dtype instead of casting them (PixArt).
    cast_to_pipeline_dtype: bool = False


def _flux_like(transformer: Mapping[str, Any], vae: Mapping[str, Any], height: int, width: int,
               _frames: int) -> tuple[int, ...]:
    scale = 2 ** (len(vae["block_out_channels"]) - 1)
    return (1, transformer["in_channels"] // 4, 2 * (height // (scale * 2)), 2 * (width // (scale * 2)))


def _flux2(transformer: Mapping[str, Any], vae: Mapping[str, Any], height: int, width: int,
           frames: int) -> tuple[int, ...]:
    _, channels, rows, columns = _flux_like(transformer, vae, height, width, frames)
    return (1, channels * 4, rows // 2, columns // 2)


def _pixart(transformer: Mapping[str, Any], vae: Mapping[str, Any], height: int, width: int,
            _frames: int) -> tuple[int, ...]:
    scale = 2 ** (len(vae["block_out_channels"]) - 1)
    return (1, transformer["in_channels"], height // scale, width // scale)


def _qwen_image(transformer: Mapping[str, Any], vae: Mapping[str, Any], height: int, width: int,
                _frames: int) -> tuple[int, ...]:
    scale = 2 ** len(vae["temperal_downsample"])
    return (1, 1, transformer["in_channels"] // 4, 2 * (height // (scale * 2)), 2 * (width // (scale * 2)))


def _z_image(transformer: Mapping[str, Any], vae: Mapping[str, Any], height: int, width: int,
             _frames: int) -> tuple[int, ...]:
    scale = 2 ** (len(vae["block_out_channels"]) - 1)
    return (1, transformer["in_channels"], 2 * (height // (scale * 2)), 2 * (width // (scale * 2)))


def _wan(transformer: Mapping[str, Any], vae: Mapping[str, Any], height: int, width: int,
         frames: int) -> tuple[int, ...]:
    temporal, spatial = vae.get("scale_factor_temporal", 4), vae.get("scale_factor_spatial", 8)
    return (1, transformer["in_channels"], (frames - 1) // temporal + 1, height // spatial, width // spatial)


LAYOUTS = {"FluxPipeline": Layout(_flux_like, pack=True), "Flux2Pipeline": Layout(_flux2),
           "PixArtSigmaPipeline": Layout(_pixart, cast_to_pipeline_dtype=True),
           "QwenImagePipeline": Layout(_qwen_image, pack=True), "QwenImageEditPipeline": Layout(_qwen_image, pack=True),
           "QwenImageEditPlusPipeline": Layout(_qwen_image, pack=True), "ZImagePipeline": Layout(_z_image),
           "WanPipeline": Layout(_wan)}


def canonical_shape(pipeline: str, transformer: Mapping[str, Any], vae: Mapping[str, Any],
                    request: Mapping[str, Any]) -> tuple[int, ...] | None:
    """The unpacked latent shape (batch 1) the pipeline would draw for this request; None when the
    pipeline has no known layout or the request does not state its size."""
    layout = LAYOUTS.get(pipeline)
    height, width = int(request.get("height") or 0), int(request.get("width") or 0)
    if layout is None or height <= 0 or width <= 0:
        return None
    return layout.shape(transformer, vae, height, width, int(request.get("num_frames") or 1))


def noise(shape: tuple[int, ...], seed: int) -> np.ndarray:
    return np.random.default_rng(int(seed)).standard_normal(shape, dtype=np.float32)


@dataclass(frozen=True)
class Checkpoint:
    pipeline: str
    transformer: Mapping[str, Any]
    vae: Mapping[str, Any]


def read_checkpoint(snapshot: Path) -> Checkpoint:
    def config(name: str) -> dict[str, Any]:
        return json.loads((snapshot / name).read_text())

    return Checkpoint(config("model_index.json")["_class_name"], config("transformer/config.json"),
                      config("vae/config.json"))


def snapshot(model: str, revision: str | None) -> Path:
    """The local Diffusers snapshot of ``model`` (a directory, or a cached Hugging Face repository)."""
    if Path(model).is_dir():
        return Path(model)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, revision=revision, local_files_only=True,
                                  allow_patterns=["model_index.json", "transformer/config.json", "unet/config.json",
                                                  "vae/config.json"]))


class Replay:
    """Turns ``latent_seed`` into an ``initial_latents_path`` file (per server, for one checkpoint)."""

    def __init__(self, load: Callable[[], Checkpoint] | None = None,
                 shape: Callable[[Mapping[str, Any]], tuple[int, ...]] | None = None) -> None:
        self._load = load
        self._shape = shape
        self._checkpoint: Checkpoint | None = None

    def __call__(self, request: Mapping[str, Any], directory: Path) -> tuple[dict[str, Any], bool]:
        """The request with the noise file and whether the noise is replayed (False: the seed is
        dropped because the pipeline has no known layout)."""
        rest = {key: value for key, value in request.items() if key != "latent_seed"}
        if self._shape is not None:
            shape = self._shape(request)
        else:
            if self._checkpoint is None:
                self._checkpoint = self._load()
            shape = canonical_shape(self._checkpoint.pipeline, self._checkpoint.transformer, self._checkpoint.vae,
                                    request)
        if shape is None:
            return rest, False
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / FILE_NAME
        noise(shape, int(request["latent_seed"])).tofile(path)
        return {**rest, "initial_latents_path": str(path)}, True
