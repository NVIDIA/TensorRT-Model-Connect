# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared-latent contracts for unified diffusion-image task evaluation."""

from __future__ import annotations

import pytest

from tests.e2e_harness.contracts import E2ECase, RunContext


@pytest.mark.parametrize(
    ("family", "model_type", "shape"),
    [
        ("flux", "flux", (1, 16, 48, 48)),
        ("flux", "flux.2", (1, 128, 24, 24)),
        ("qwen_image", "qwen_image", (1, 16, 48, 48)),
        ("z_image", "z_image", (1, 16, 48, 48)),
    ],
)
def test_hf_and_trtmc_share_exact_family_latents(
    tmp_path, family: str, model_type: str, shape: tuple[int, int, int, int]
) -> None:
    if family == "flux":
        from tests.e2e.models.flux.e2e_plugins.parity import ensure_initial_latents
    elif family == "qwen_image":
        from tests.e2e.models.qwen_image.e2e_plugins.parity import ensure_initial_latents
    else:
        from tests.e2e.models.z_image.e2e_plugins.parity import ensure_initial_latents

    case = E2ECase(
        name=f"{family}-sample",
        family=family,
        hf_id="example/model",
        runtime_strategy=f"diffusion_{family}",
        inputs={"image_height": 384, "image_width": 384, "seed": 47},
        metadata={"model_type": model_type},
    )
    hf = ensure_initial_latents(
        case, RunContext(case=case, artifacts_dir=str(tmp_path / "hf_artifacts"))
    )
    trt = ensure_initial_latents(
        case, RunContext(case=case, artifacts_dir=str(tmp_path / "bundle_artifacts"))
    )

    assert hf.path == trt.path
    assert hf.sha256 == trt.sha256
    assert hf.shape == shape
    assert hf.path.stat().st_size == 4 * shape[0] * shape[1] * shape[2] * shape[3]
