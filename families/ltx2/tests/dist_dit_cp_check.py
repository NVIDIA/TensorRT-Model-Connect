# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-rank check of the LTX-2.5 context-parallel DiT (tiny random weights), torch-free.

``tools/launch_ranks.py -n CP --gpus ... -- python -m families.ltx2.tests.dist_dit_cp_check PREP_DIR``
after ``cp_tiny_prep`` wrote the plans and references into ``PREP_DIR``. Every rank runs the CP
plan with its NCCL communicator (a hung collective aborts the communicator instead of blocking),
runs the single-device plan, and compares the full outputs with the single-device plan and with
diffusers, shard by shard. Writes ``cp_rank<r>.json``; exits non-zero on failure.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from families.ltx2.tests import conftest  # noqa: F401 - binds the TensorRT backend


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
    single = NpEngine((prep / "single.plan").read_bytes())
    report = {"rank": comm.rank, "world": comm.world, "cases": {}}
    ok = True
    base = {k: ref[k] for k in ("video_latent", "audio_latent", "video_context", "audio_context", "timestep")}
    for case in ("plain", "stg_mixed", "isolated"):
        feed = dict(base, stg_keep=ref[f"{case}_stg_keep"], av_keep=ref[f"{case}_av_keep"])
        try:
            got = cp_engine(feed, timeout_s=120)
        except TimeoutError as exc:
            print(f"[rank {comm.rank}] {case}: {exc}; communicator aborted", flush=True)
            report["cases"][case] = {"ok": False, "error": str(exc)}
            ok = False
            break
        one = single(feed)
        gv, ga = got["video_velocity"], got["audio_velocity"]
        s = gv.shape[1] // comm.world
        rec = {
            "video_cos_per_shard_vs_single": [cosine(gv[:, r * s:(r + 1) * s], one["video_velocity"][:, r * s:(r + 1) * s])
                                              for r in range(comm.world)],
            "audio_cos_vs_single": cosine(ga, one["audio_velocity"]),
            "video_cos_vs_diffusers": cosine(gv, ref[f"{case}_ref_video"]),
            "audio_cos_vs_diffusers": cosine(ga, ref[f"{case}_ref_audio"]),
            "max_abs_vs_single": float(np.abs(gv - one["video_velocity"]).max()),
            "finite": bool(np.isfinite(gv).all() and np.isfinite(ga).all()),
        }
        rec["ok"] = bool(rec["finite"] and min(rec["video_cos_per_shard_vs_single"]) > 0.9999
                         and rec["audio_cos_vs_single"] > 0.9999 and rec["video_cos_vs_diffusers"] > 0.999
                         and rec["audio_cos_vs_diffusers"] > 0.999)
        ok &= rec["ok"]
        report["cases"][case] = rec
        print(f"[rank {comm.rank}] {case}: {json.dumps(rec)}", flush=True)
    (prep / f"cp_rank{comm.rank}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    if comm.comm:
        comm.destroy()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
