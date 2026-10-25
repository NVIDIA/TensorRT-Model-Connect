# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual IBackend/ITrtModule composition with the production CPU RNG."""

import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from families.minimax_h3.reference_posterior_builder import build_reference_posterior_engine


@pytest.fixture(scope="module")
def native_posterior_plan(tmp_path_factory):
    root = Path(os.environ["TRTMC_RUNTIME_ROOT"])
    assert (root / "minimax_h3_reference_posterior_probe").is_file()
    assert (root / "libtrtmc_backend_trt.so").is_file()
    assert (root / "libtrtmc_model_minimax_h3.so").is_file()
    plan = tmp_path_factory.mktemp("native-posterior") / "posterior.plan"
    plan.write_bytes(build_reference_posterior_engine())
    return root, plan


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("pixels", [16384, 22016, 65536])
def test_native_backend_reference_posterior_and_cpu_rng(
    native_posterior_plan, tmp_path, pixels, record_property
):
    import torch
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

    root, plan = native_posterior_plan
    indices = torch.arange(24 * pixels).reshape(24, pixels)
    mean = ((indices % 31).float() - 15) / 16
    # Exercise a true unit-normal RNG stream without creating out-of-range
    # FP16 samples. The separate existing oracle covers the upper clamp.
    logvar = torch.tensor([-40.0, -2.0, 0.0, 2.0, 4.0])[indices % 5]
    moments = torch.cat((mean, logvar), dim=0).contiguous()
    source = tmp_path / "moments.bin"
    moments.numpy().tofile(source)
    prefix = tmp_path / "native"
    result = subprocess.run(
        [
            str(root / "minimax_h3_reference_posterior_probe"),
            str(root),
            str(plan),
            str(source),
            str(prefix),
            str(pixels),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    (tmp_path / "probe.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    expected_noise = torch.randn((1, 24, pixels), generator=torch.Generator().manual_seed(42))[0]
    noise = np.fromfile(str(prefix) + ".noise.bin", dtype=np.float32).reshape(24, pixels)
    # The existing native CPU RNG contract is exact, not a new loose tolerance.
    assert noise.tobytes() == expected_noise.numpy().tobytes()
    expected = (
        DiagonalGaussianDistribution(moments[None].cuda())
        .sample(generator=torch.Generator().manual_seed(42))
        .half()[0]
        .cpu()
    )
    cold = np.fromfile(str(prefix) + ".cold.bin", dtype=np.float32).reshape(24, pixels)
    warm = np.fromfile(str(prefix) + ".warm.bin", dtype=np.float32).reshape(24, pixels)
    assert cold.tobytes() == warm.tobytes()
    actual = torch.from_numpy(cold)
    assert torch.equal(actual, actual.half().float())
    torch.testing.assert_close(actual.half(), expected, rtol=1e-3, atol=1e-5)
    record_property(
        "native_posterior",
        json.dumps(
            dict(
                pixels=pixels,
                cpu_rng_byte_exact=True,
                cold_warm_byte_exact=True,
                max_abs_error=(actual - expected.float()).abs().max().item(),
                trained_VAE_qualified=False,
                native_Ref2VA_video_qualified=False,
            )
        ),
    )
