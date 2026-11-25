# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

PROFILES = ("dinov3-vits16-pretrain-lvd1689m", "dinov3-convnext-tiny-pretrain-lvd1689m")
REPO = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("profile", PROFILES)
def test_formal_profile_uses_one_image_suite_for_accuracy_and_timing(profile, monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "apps/aiperf_qual"))
    from trtmc_aiperf_qual.config import Environment
    from trtmc_aiperf_qual.models import resolve_model

    model = resolve_model(profile, Environment({"repo": str(REPO)}))
    assert model["accuracy_source"] == "absolute"
    assert len(model["absolute"]) == 1
    item = model["absolute"][0]
    assert item["suite"] == "coco-feature-parity"
    assert item["metric"] == "family_parity"
    assert item["gate"]["profile"] == profile
    assert item["suite_definition"]["selection"] == {"method": "seeded", "count": 50}
    assert (
        item["suite_definition"]["source"]["revision"] == "cf0b22332314a937e9dc8a1957b21725430bb41d"
    )
    assert model["reference"]["precision"] == model["candidate"]["precision"] == "fp16"
    assert model["reference"]["adapter"] == "families/dinov3/reference/adapter.py"
