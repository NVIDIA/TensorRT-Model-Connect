# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import numpy as np
import pytest

from families.stable_diffusion.reference.adapter import Adapter


def test_native_and_candidate_receive_identical_family_shaped_noise(tmp_path):
    from trtmc_perf_serving.latents import Replay

    for name, value in (("unet/config.json", {"in_channels": 4}),
                        ("vae/config.json", {"block_out_channels": [128, 256, 512, 512]})):
        path = tmp_path / "checkpoint" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    replay = Replay(shape=lambda request: Adapter.latent_shape(tmp_path / "checkpoint", request))
    request = {"height": 512, "width": 512, "latent_seed": 1000}
    native, used_native = replay(request, tmp_path / "native")
    candidate, used_candidate = replay(request, tmp_path / "candidate")
    assert used_native and used_candidate
    assert Path(native["initial_latents_path"]).read_bytes() == Path(candidate["initial_latents_path"]).read_bytes()
    assert np.fromfile(native["initial_latents_path"], dtype=np.float32).size == 4 * 64 * 64
    with pytest.raises(ValueError, match="divisible"):
        replay({**request, "width": 513}, tmp_path / "invalid")
