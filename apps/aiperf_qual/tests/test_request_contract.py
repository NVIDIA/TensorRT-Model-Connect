# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from trtmc_aiperf_qual.config import Environment
from trtmc_aiperf_qual.models import catalog_profiles, resolve_model
from trtmc_aiperf_qual.suites import unstated_defaults


def test_image_video_controls_must_be_present_but_do_not_apply_to_text_or_speech():
    assert unstated_defaults({"prompt": "cat"}, "generate_image") == ["num_steps", "guidance_scale", "height", "width"]
    image = {"num_steps": 4, "guidance_scale": 0.0, "height": 512, "width": 512}
    assert unstated_defaults(image, "generate_image") == []
    assert unstated_defaults({**image, "media_type": "video"}, "generate_image") == ["num_frames"]
    assert unstated_defaults({**image, "num_steps": -1}, "generate_image") == ["num_steps"]
    assert unstated_defaults({"prompt": "cat"}, "generate") == []
    assert unstated_defaults({"text": "hello"}, "generate_audio") == []


def test_all_configured_image_profiles_resolve_complete_catalog_requests():
    from trtmc_perf_serving.profiles import resolve_profile

    root = Path(__file__).resolve().parents[3]
    environment = Environment({"repo": str(root), "bundle_root": "/tmp/bundles"})
    profiles = [entry.name for entry in catalog_profiles(root) if entry.operation == "generate_image"]
    assert profiles
    failures = []
    for profile in profiles:
        model = resolve_model(profile, environment)
        resolved = resolve_profile(profile, manifest_root=root / "families")
        request = {**resolved.base_request, **model.get("catalog_request", {})}
        if missing := unstated_defaults(request, model["operation"]):
            failures.append((profile, missing))
    assert not failures
