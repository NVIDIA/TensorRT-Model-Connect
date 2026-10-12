# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import pytest

from families.stable_diffusion.reference.adapter import Adapter


def test_latent_shape_uses_the_checkpoint_geometry(tmp_path):
    for name, value in (("unet/config.json", {"in_channels": 4}),
                        ("vae/config.json", {"block_out_channels": [128, 256, 512, 512]})):
        path = tmp_path / "checkpoint" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    request = {"height": 512, "width": 512}
    assert Adapter.latent_shape(tmp_path / "checkpoint", request) == (1, 4, 64, 64)
    with pytest.raises(ValueError, match="divisible"):
        Adapter.latent_shape(tmp_path / "checkpoint", {**request, "width": 513})
