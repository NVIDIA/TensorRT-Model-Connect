# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-rank preparation for the tile-parallel VAE check (uses torch + diffusers).

``python -m families.ltx2.tests.vae_tile_prep OUT_DIR WORLD``: builds the tiny random VAE, its tile
plan for WORLD ranks and the tile-shaped plan, random latents, and the blend of diffusers' decode of
every tile. The multi-rank step (``dist_vae_tile_check``) is torch-free.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from families.ltx2.tests import conftest  # noqa: F401 - binds the TensorRT backend


def main() -> int:
    import numpy as np
    import torch

    from families.ltx2.tests import test_vae_parity as tv
    from families.ltx2.vae_builder import build_vae_decoder_engine
    from families.ltx2.vae_tiling import TileConfig, blend_tiles, plan_tiles

    out = Path(sys.argv[1])
    world = int(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    vae, _ = tv.tiny_vae(out / "vae")
    vae = vae.to("cuda", torch.float32)
    f, h, w = tv.TILE_GRID
    plan = plan_tiles(f, h, w, TileConfig(**tv.TILE_CONFIG), world=world)
    tf, th, tw = plan["tile_latent"]
    (out / "tile.plan").write_bytes(build_vae_decoder_engine(out / "vae", latent_frames=tf, latent_height=th,
                                                             latent_width=tw, clamp_output=False))
    packed = torch.randn(1, f * h * w, 16, generator=torch.Generator().manual_seed(4))
    tiles = [tv.diffusers_tile(vae, packed, tv.TILE_GRID, plan, tile).numpy() for tile in plan["tiles"]]
    reference = blend_tiles(plan, tiles, (f - 1) * 8 + 1, h * 32, w * 32)
    np.save(out / "latents.npy", packed.numpy())
    np.save(out / "reference.npy", reference)
    (out / "plan.json").write_text(json.dumps({"grid": [f, h, w], "plan": plan}), encoding="utf-8")
    print(f"prepared {out}: {len(plan['tiles'])} tiles of {plan['tile_latent']} latents for {world} ranks", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
