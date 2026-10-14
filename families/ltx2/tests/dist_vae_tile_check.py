# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-rank check of the tile-parallel LTX-2.5 video VAE decode (tiny random weights), torch-free.

``tools/launch_ranks.py -n WORLD --gpus ... -- python -m families.ltx2.tests.dist_vae_tile_check PREP_DIR``
after ``vae_tile_prep`` wrote the tile plan, the tile-shaped plan and the reference into ``PREP_DIR``.

Mirrors the runtime protocol: every rank decodes its tiles of the plan; the worker ranks pack their
fp16 tiles into one device buffer and send it to rank 0 (NCCL point-to-point on the communicator,
polled with a deadline that aborts the communicator); rank 0 blends every tile in tile order. Rank 0
also decodes every tile itself and checks that the tile-parallel blend equals that single-rank tiled
decode bit for bit, and that it matches the blend of diffusers' tiles. Writes ``vae_rank<r>.json``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from families.ltx2.tests import conftest  # noqa: F401 - binds the TensorRT backend


def _tile_latents(latents, grid, plan, tile):
    f, h, w = grid
    tf, th, tw = plan["tile_latent"]
    f0, h0, w0 = tile["latent_start"]
    part = latents.reshape(1, f, h, w, -1)[:, f0:f0 + tf, h0:h0 + th, w0:w0 + tw]
    return part.reshape(1, tf * th * tw, -1)


def main() -> int:
    import numpy as np
    from cuda.bindings import runtime as rt

    from families.ltx2.tests.dist_helpers import NcclComm
    from families.ltx2.tests.np_engine import NpEngine, ck, cosine
    from families.ltx2.vae_tiling import blend_tiles

    ck(rt.cudaSetDevice(int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0"))))
    ck(rt.cudaFree(0))
    prep = Path(sys.argv[1])
    comm = NcclComm()
    spec = json.loads((prep / "plan.json").read_text(encoding="utf-8"))
    grid, plan = spec["grid"], spec["plan"]
    latents = np.load(prep / "latents.npy")
    engine = NpEngine((prep / "tile.plan").read_bytes())
    out_ptr, out_shape, _ = engine.buffers["frames"]
    tile_bytes = int(np.prod(out_shape)) * 2
    stream = int(ck(rt.cudaStreamCreate()))
    tiles = plan["tiles"]
    mine = [k for k, t in enumerate(tiles) if t["rank"] == comm.rank]
    report = {"rank": comm.rank, "world": comm.world, "tiles": mine}
    ok = True
    try:
        if comm.rank != 0:
            send = int(ck(rt.cudaMalloc(max(len(mine) * tile_bytes, 1))))
            for i, k in enumerate(mine):
                engine({"latents": _tile_latents(latents, grid, plan, tiles[k])})
                ck(rt.cudaMemcpy(send + i * tile_bytes, out_ptr, tile_bytes,
                                 rt.cudaMemcpyKind.cudaMemcpyDeviceToDevice))
            if mine:
                comm.p2p([(True, send, len(mine) * tile_bytes, 0)], stream)
        else:
            decoded: list = [None] * len(tiles)
            for k in mine:
                decoded[k] = engine({"latents": _tile_latents(latents, grid, plan, tiles[k])})["frames"]
            peer_tiles = {p: [k for k, t in enumerate(tiles) if t["rank"] == p] for p in range(1, comm.world)}
            recv = {p: int(ck(rt.cudaMalloc(max(len(ks) * tile_bytes, 1)))) for p, ks in peer_tiles.items()}
            comm.p2p([(False, recv[p], len(ks) * tile_bytes, p) for p, ks in peer_tiles.items() if ks], stream)
            peer_identical = True
            for p, ks in peer_tiles.items():
                for i, k in enumerate(ks):
                    host = np.empty(out_shape, np.float16)
                    ck(rt.cudaMemcpy(host.ctypes.data, recv[p] + i * tile_bytes, tile_bytes,
                                     rt.cudaMemcpyKind.cudaMemcpyDeviceToHost))
                    decoded[k] = host
                    own = engine({"latents": _tile_latents(latents, grid, plan, tiles[k])})["frames"]
                    peer_identical &= bool(np.array_equal(own.astype(np.float16), host))
            single = [engine({"latents": _tile_latents(latents, grid, plan, t)})["frames"].astype(np.float16)
                      for t in tiles]
            f, h, w = grid
            shape = ((f - 1) * 8 + 1, h * 32, w * 32)
            parallel = blend_tiles(plan, [np.asarray(d, np.float16) for d in decoded], *shape)
            serial = blend_tiles(plan, single, *shape)
            reference = np.load(prep / "reference.npy")
            report.update(
                peer_tiles_identical_to_rank0_decode=peer_identical,
                blend_bit_identical_to_single_rank=bool(np.array_equal(parallel, serial)),
                cos_vs_diffusers_tiles=cosine(parallel - 0.5, reference - 0.5),
                finite=bool(np.isfinite(parallel).all()),
            )
            ok = (report["blend_bit_identical_to_single_rank"] and report["finite"]
                  and report["cos_vs_diffusers_tiles"] > 0.999)
    except TimeoutError as exc:
        report["error"] = str(exc)
        ok = False
    report["ok"] = ok
    print(f"[rank {comm.rank}] {json.dumps(report)}", flush=True)
    (prep / f"vae_rank{comm.rank}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    if comm.comm:
        comm.destroy()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
