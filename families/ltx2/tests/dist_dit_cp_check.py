# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-rank check of the LTX-2.5 context-parallel DiT (tiny random weights), torch-free.

``tools/launch_ranks.py -n CP --gpus ... -- python -m families.ltx2.tests.dist_dit_cp_check PREP_DIR``
after ``cp_tiny_prep`` wrote the plans and references into ``PREP_DIR``. Every rank runs the CP
plan with its NCCL communicator (a hung collective aborts the communicator instead of blocking),
runs the single-device plan, and compares the full outputs with the single-device plan and with
diffusers, shard by shard. The two-stage CP plan (full grid plus the half-resolution grid, run-time
token count) is checked the same way at both grids. Writes ``cp_rank<r>.json``; exits non-zero on
failure.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from families.ltx2.tests import conftest  # noqa: F401 - binds the TensorRT backend

INPUTS = ("video_latent", "audio_latent", "video_context", "audio_context", "timestep")


def _compare(np, cosine, world: int, got: dict, one: dict, ref_video, ref_audio) -> dict:
    gv, ga = got["video_velocity"], got["audio_velocity"]
    s = gv.shape[1] // world
    rec = {
        "video_tokens": int(gv.shape[1]),
        "video_cos_per_shard_vs_single": [cosine(gv[:, r * s:(r + 1) * s], one["video_velocity"][:, r * s:(r + 1) * s])
                                          for r in range(world)],
        "audio_cos_vs_single": cosine(ga, one["audio_velocity"]),
        "video_cos_vs_diffusers": cosine(gv, ref_video),
        "audio_cos_vs_diffusers": cosine(ga, ref_audio),
        "max_abs_vs_single": float(np.abs(gv - one["video_velocity"]).max()),
        "finite": bool(np.isfinite(gv).all() and np.isfinite(ga).all()),
    }
    rec["ok"] = bool(rec["finite"] and min(rec["video_cos_per_shard_vs_single"]) > 0.9999
                     and rec["audio_cos_vs_single"] > 0.9999 and rec["video_cos_vs_diffusers"] > 0.999
                     and rec["audio_cos_vs_diffusers"] > 0.999)
    return rec


def main() -> int:
    import numpy as np
    from cuda.bindings import runtime as rt

    from families.ltx2.tests.dist_helpers import NcclComm
    from families.ltx2.tests.np_engine import NpEngine, ck, cosine

    ck(rt.cudaSetDevice(int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0"))))
    ck(rt.cudaFree(0))
    prep = Path(sys.argv[1])
    comm = NcclComm()
    ref = np.load(prep / "reference.npz")
    cp_engine = NpEngine((prep / f"cp{comm.world}.plan").read_bytes(), comm.capsule(), on_timeout=comm.abort)
    two_grid = NpEngine((prep / f"cp{comm.world}_two_grid.plan").read_bytes(), comm.capsule(),
                        on_timeout=comm.abort)
    single = NpEngine((prep / "single.plan").read_bytes())
    single_small = NpEngine((prep / "single_small.plan").read_bytes())
    report = {"rank": comm.rank, "world": comm.world, "cases": {}}
    base = {k: ref[k] for k in INPUTS}
    plain = dict(base, stg_keep=ref["plain_stg_keep"], av_keep=ref["plain_av_keep"])
    small = dict({k: ref[f"small_{k}"] for k in INPUTS}, stg_keep=ref["plain_stg_keep"],
                 av_keep=ref["plain_av_keep"])
    cases = [(case, cp_engine, single, dict(base, stg_keep=ref[f"{case}_stg_keep"], av_keep=ref[f"{case}_av_keep"]),
              case) for case in ("plain", "stg_mixed", "isolated")]
    # The two-stage plan switches grids on one context: full, half resolution, full again.
    cases += [("two_grid_full", two_grid, single, plain, "plain"),
              ("two_grid_small", two_grid, single_small, small, "small"),
              ("two_grid_full_again", two_grid, single, plain, "plain")]
    ok = True
    for name, engine, reference_plan, feed, ref_key in cases:
        try:
            got = engine(feed, timeout_s=120)
        except TimeoutError as exc:
            print(f"[rank {comm.rank}] {name}: {exc}; communicator aborted", flush=True)
            report["cases"][name] = {"ok": False, "error": str(exc)}
            ok = False
            break
        rec = _compare(np, cosine, comm.world, got, reference_plan(feed), ref[f"{ref_key}_ref_video"],
                       ref[f"{ref_key}_ref_audio"])
        ok &= rec["ok"]
        report["cases"][name] = rec
        print(f"[rank {comm.rank}] {name}: {json.dumps(rec)}", flush=True)
    (prep / f"cp_rank{comm.rank}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    if comm.comm:
        comm.destroy()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
