# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-rank preparation for the multi-rank CP check (uses torch + diffusers).

``python -m families.ltx2.tests.cp_tiny_prep OUT_DIR CP``: builds the tiny random DiT, its
single-device and CP plans, and the diffusers reference outputs for the normal, mixed-STG and
modality-isolated batches. The multi-rank step (``dist_dit_cp_check``) is torch-free.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from families.ltx2.tests import conftest  # noqa: F401 - binds the TensorRT backend

CASES = (("plain", [1.0, 1.0], [1.0, 1.0]), ("stg_mixed", [1.0, 0.0], [1.0, 1.0]),
         ("isolated", [1.0, 1.0], [0.0, 0.0]))


def main() -> int:
    import numpy as np
    import safetensors.torch as st
    import torch
    from diffusers import LTX2VideoTransformer3DModel

    from families.ltx2.dit_builder import DiTShape, audio_latent_frames, build_dit_engine
    from families.ltx2.tests import test_dit_parity as tp

    out = Path(sys.argv[1])
    cp = int(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    model = LTX2VideoTransformer3DModel(**tp.TINY_DIT).eval()
    tp._randomize(model, 5)
    folder = out / "transformer"
    folder.mkdir(exist_ok=True)
    st.save_file({k: v.to(torch.bfloat16).contiguous() for k, v in model.state_dict().items()},
                 str(folder / "diffusion_pytorch_model.safetensors"))
    (folder / "config.json").write_text(json.dumps(tp.TINY_DIT), encoding="utf-8")
    sa = audio_latent_frames((tp.FRAMES - 1) * 8 + 1, tp.FPS)
    shape = DiTShape(batch=2, latent_frames=tp.FRAMES, latent_height=tp.LH, latent_width=tp.LW, audio_frames=sa,
                     text_len=tp.TEXT, fps=tp.FPS)
    (out / "single.plan").write_bytes(build_dit_engine(folder, shape, cp_size=1, stg_blocks=(tp.STG_BLOCK,)))
    (out / f"cp{cp}.plan").write_bytes(build_dit_engine(folder, shape, cp_size=cp, stg_blocks=(tp.STG_BLOCK,)))
    inp = tp._inputs(shape)
    arrays = {k: v.float().numpy() for k, v in inp.items()}
    for case, stg, av in CASES:
        rv, ra = tp._reference(model, shape, inp, torch.float32,
                               stg_mask=torch.tensor(stg) if case == "stg_mixed" else None,
                               isolate=(case == "isolated"))
        arrays[f"{case}_ref_video"] = rv.numpy()
        arrays[f"{case}_ref_audio"] = ra.numpy()
        arrays[f"{case}_stg_keep"] = np.asarray(stg, np.float32)
        arrays[f"{case}_av_keep"] = np.asarray(av, np.float32)
    np.savez(out / "reference.npz", **arrays)
    print(f"prepared {out} (video tokens {shape.video_tokens}, audio {sa}, cp {cp})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
