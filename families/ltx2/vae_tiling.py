# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tile plan of the LTX-2.5 tiled (and tile-parallel) video VAE decode.

The decode splits the latent video into overlapping tiles, decodes every tile independently with
one static tile-shaped plan and blends the decoded tiles with linear ramps over their overlaps,
normalized by the summed weights (``out = sum_k w_k * tile_k / sum_k w_k``, the Lightricks /
TensorRT-LLM ``tiled_decode`` blend). Each tile is an independent forward, so context-parallel
ranks decode disjoint tile subsets. Rank 0 blends every tile in tile order; the result is the same
bit for bit whichever rank decoded a tile.

Geometry (all tiles share one shape, so one static plan serves every tile):

- spatial axes: the fewest (then smallest) equal tiles of at most ``T`` latents with evenly spread
  starts, every overlap at least ``O`` latents and both overlaps of a tile inside the tile (usually
  ``n = ceil((L - O) / (T - O))`` tiles of ``ceil((L + (n - 1) O) / n)`` latents). ``T`` / ``O`` come
  from the pixel tile size and overlap (512 / 64 px by default, as in diffusers ``enable_tiling``);
- time: the same split with a minimum overlap of ``O_t + 1`` latent frames (by default, clips of up
  to 257 frames decode as one temporal tile and longer clips split into 256-frame tiles overlapping
  by at least 24 frames). A tile covering latent frames ``[a, b)`` decodes ``(b - a - 1) * 8 + 1``
  frames placed at frame ``a * 8`` (its first latent frame decodes to a single frame, ``ltx-core``
  ``map_temporal_interval_to_frame``);
- ramps: a spatial overlap of ``r`` pixels fades in as ``k / (r + 1)`` (``k = 1..r``) and out as
  ``1 - k / (r + 1)``; a temporal overlap fades the later tile in from 0 (``k / r``, ``k = 0..r-1``)
  because its first frame stands in for a whole latent group, and the earlier tile out over the
  remaining ``r - 1`` frames.

The runtime (``runtime/vae_tiling.h``) consumes the plan written into ``runtime.json`` and owns
the ramp evaluation and blend; :func:`blend_tiles` is the NumPy reference of the same arithmetic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SPATIAL_SCALE = 32
TEMPORAL_SCALE = 8


@dataclass(frozen=True)
class TileConfig:
    """Tile size and minimum overlap in output pixels / frames (0 disables tiling on that axis)."""

    tile_pixels: int = 512
    overlap_pixels: int = 64
    tile_frames: int = 256
    overlap_frames: int = 24

    def validate(self) -> None:
        if self.tile_pixels < 0 or self.tile_frames < 0:
            raise ValueError("VAE tile sizes must be non-negative")
        # Overlaps below half a tile keep every ramp inside its tile and let only neighbouring
        # tiles overlap.
        if self.tile_pixels and (self.tile_pixels % SPATIAL_SCALE or self.overlap_pixels % SPATIAL_SCALE
                                 or not 0 < 2 * self.overlap_pixels < self.tile_pixels):
            raise ValueError(f"VAE spatial tiles need multiples of {SPATIAL_SCALE} px with "
                             "0 < overlap < tile / 2")
        if self.tile_frames and (self.tile_frames % TEMPORAL_SCALE or self.overlap_frames % TEMPORAL_SCALE
                                 or not 0 < self.overlap_frames
                                 or 2 * (self.overlap_frames + TEMPORAL_SCALE) >= self.tile_frames):
            raise ValueError(f"VAE temporal tiles need multiples of {TEMPORAL_SCALE} frames with "
                             f"0 < overlap < tile / 2 - {TEMPORAL_SCALE}")

    @property
    def enabled(self) -> bool:
        return self.tile_pixels > 0 or self.tile_frames > 0


def split_axis(length: int, tile: int, overlap: int) -> tuple[int, list[int]]:
    """``(size, starts)`` of equal tiles covering ``[0, length)`` with overlaps of at least ``overlap``."""
    if tile <= 0 or length <= tile:
        return length, [0]
    if not 0 < overlap < tile:
        raise ValueError("tile overlap must be positive and smaller than the tile")
    # Fewest tiles first, then the smallest size: every overlap >= `overlap`, and each tile's two
    # overlaps fit in the tile (ramps never collide and only neighbours overlap).
    for count in range(math.ceil((length - overlap) / (tile - overlap)), length + 1):
        for size in range(math.ceil((length + (count - 1) * overlap) / count), tile + 1):
            span = length - size
            starts = [(2 * i * span + count - 1) // (2 * (count - 1)) for i in range(count)]
            overlaps = [0] + [starts[i] + size - starts[i + 1] for i in range(count - 1)] + [0]
            if min(overlaps[1:-1]) >= overlap and all(overlaps[i] + overlaps[i + 1] <= size
                                                       for i in range(count)):
                return size, starts
    raise ValueError(f"tiles of at most {tile} cannot cover {length} with overlaps of {overlap}")


def _spatial_ramps(starts: list[int], size: int) -> list[tuple[int, int]]:
    """Per tile (left, right) ramps in pixels: both sides of an overlap ramp over all of it."""
    ramps = [[0, 0] for _ in starts]
    for i in range(len(starts) - 1):
        overlap = (starts[i] + size - starts[i + 1]) * SPATIAL_SCALE
        ramps[i][1] = overlap
        ramps[i + 1][0] = overlap
    return [tuple(r) for r in ramps]


def _temporal_ramps(starts: list[int], size: int) -> list[tuple[int, int]]:
    """Per tile (left, right) ramps in frames.

    Latent overlap ``ov`` shares ``(ov - 1) * 8 + 1`` frames: the later tile ramps in from 0 over all
    of them, the earlier tile ramps out over the last ``(ov - 1) * 8`` (weights sum to 1 per frame).
    """
    ramps = [[0, 0] for _ in starts]
    for i in range(len(starts) - 1):
        shared = (starts[i] + size - starts[i + 1] - 1) * TEMPORAL_SCALE + 1
        ramps[i][1] = shared - 1
        ramps[i + 1][0] = shared
    return [tuple(r) for r in ramps]


def assign_lpt(volumes: list[int], world: int) -> list[int]:
    """Longest-processing-time rank per tile (stable: ties keep tile order and the lowest rank)."""
    load = [0] * world
    ranks = [0] * len(volumes)
    for index in sorted(range(len(volumes)), key=lambda i: -volumes[i]):
        rank = min(range(world), key=lambda r: load[r])
        load[rank] += volumes[index]
        ranks[index] = rank
    return ranks


def plan_tiles(latent_frames: int, latent_height: int, latent_width: int, config: TileConfig,
               world: int = 1) -> dict:
    """Tile plan for ``runtime.json`` (``vae_tiling``)."""
    config.validate()
    if not config.enabled:
        raise ValueError("VAE tiling is disabled")
    s_tile = config.tile_pixels // SPATIAL_SCALE
    s_overlap = config.overlap_pixels // SPATIAL_SCALE
    t_tile = config.tile_frames // TEMPORAL_SCALE
    t_overlap = config.overlap_frames // TEMPORAL_SCALE + 1
    tf, f_starts = split_axis(latent_frames, t_tile, t_overlap)
    th, h_starts = split_axis(latent_height, s_tile, s_overlap)
    tw, w_starts = split_axis(latent_width, s_tile, s_overlap)
    if tf < 2 and latent_frames >= 2:
        raise ValueError("VAE temporal tiles must span at least two latent frames")
    f_ramps = _temporal_ramps(f_starts, tf)
    h_ramps = _spatial_ramps(h_starts, th)
    w_ramps = _spatial_ramps(w_starts, tw)
    tiles = []
    for fi, f0 in enumerate(f_starts):
        for hi, h0 in enumerate(h_starts):
            for wi, w0 in enumerate(w_starts):
                tiles.append({"latent_start": [f0, h0, w0],
                              "pixel_start": [f0 * TEMPORAL_SCALE, h0 * SPATIAL_SCALE, w0 * SPATIAL_SCALE],
                              "ramps": [list(f_ramps[fi]), list(h_ramps[hi]), list(w_ramps[wi])]})
    for tile, rank in zip(tiles, assign_lpt([1] * len(tiles), world)):
        tile["rank"] = rank
    return {
        "tile_latent": [tf, th, tw],
        "tile_pixels": [(tf - 1) * TEMPORAL_SCALE + 1, th * SPATIAL_SCALE, tw * SPATIAL_SCALE],
        "config": {"tile_pixels": config.tile_pixels, "overlap_pixels": config.overlap_pixels,
                   "tile_frames": config.tile_frames, "overlap_frames": config.overlap_frames},
        "world_size": world,
        "tiles": tiles,
    }


def axis_weights(length: int, left: int, right: int, *, temporal: bool) -> np.ndarray:
    """fp32 ramp weights of one tile axis (the runtime evaluates the same expressions in fp32)."""
    w = np.ones(length, dtype=np.float32)
    if left:
        if temporal:
            w[:left] = np.arange(left, dtype=np.float32) / np.float32(left)
        else:
            w[:left] = np.arange(1, left + 1, dtype=np.float32) / np.float32(left + 1)
    if right:
        denom = np.float32(right + 1)
        w[length - right:] = np.float32(1.0) - np.arange(1, right + 1, dtype=np.float32) / denom
    return w


def blend_tiles(plan: dict, decoded: list[np.ndarray], frames: int, height: int, width: int) -> np.ndarray:
    """NumPy reference blend: ``decoded[k]`` is tile k ``[T, H, W, 3]``; returns clamped ``[frames, H, W, 3]``."""
    num = np.zeros((frames, height, width, 3), dtype=np.float32)
    den = np.zeros((frames, height, width, 1), dtype=np.float32)
    tt, th, tw = plan["tile_pixels"]
    for tile, values in zip(plan["tiles"], decoded):
        t0, y0, x0 = tile["pixel_start"]
        (tl, tr), (yl, yr), (xl, xr) = tile["ramps"]
        wt = axis_weights(tt, tl, tr, temporal=True)
        wy = axis_weights(th, yl, yr, temporal=False)
        wx = axis_weights(tw, xl, xr, temporal=False)
        w = (wt[:, None, None] * wy[None, :, None]) * wx[None, None, :]
        num[t0:t0 + tt, y0:y0 + th, x0:x0 + tw] += w[..., None] * values.astype(np.float32)
        den[t0:t0 + tt, y0:y0 + th, x0:x0 + tw] += w[..., None]
    return np.clip(num / den, np.float32(0.0), np.float32(1.0))
