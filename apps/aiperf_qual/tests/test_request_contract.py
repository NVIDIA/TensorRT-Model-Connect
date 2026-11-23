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


def test_newly_covered_task_profiles_resolve_their_benchmarks():
    """Task SDK renames covered by Task entries: each profile resolves its benchmark and its native path."""
    root = Path(__file__).resolve().parents[3]
    environment = Environment({"repo": str(root), "bundle_root": "/tmp/bundles"})
    expected = {"parakeet-tdt-0.6b-v3": "librispeech-test-clean", "wav2vec2-base-960h": "librispeech-test-clean",
                "wav2vec2-xlsr-53-japanese": "fleurs-ja-test", "grounding-dino-tiny": "coco-map-prompted",
                "owlv2-base-patch16-ensemble": "coco-map-prompted", "dinov2-base": "coco-feature-parity",
                "timesfm-2.0-500m-official": "etth1-mse", "clef": "decision-parity", "laya-router": "decision-parity"}
    for profile, suite in expected.items():
        model = resolve_model(profile, environment)
        assert [item["suite"] for item in model["absolute"]] == [suite], profile
        assert model["reference"]["backend"] == "reference", profile
    detector = resolve_model("grounding-dino-tiny", environment)["absolute"][0]["suite_definition"]["request"]
    assert detector["box_threshold"] == 0.1 and detector["prompt"].count(". ") == 79
    assert "score_threshold" not in detector  # the family declares no such config
    decision = resolve_model("clef", environment)["absolute"][0]["suite_definition"]
    assert decision["source"] == {"kind": "catalog_testcase", "all_testcases": True, "profile": "clef"}


def test_a_decision_suite_answers_every_catalog_testcase_of_its_profile(tmp_path):
    import sys

    from trtmc_aiperf_qual.suites import build_suite

    root = Path(__file__).resolve().parents[3]
    environment = Environment({"repo": str(root), "serve_python": sys.executable,
                               "hf_datasets_cache": str(tmp_path)})
    definition = {"suite": "decision-parity", "version": 1,
                  "source": {"kind": "catalog_testcase", "all_testcases": True, "profile": "clef"},
                  "selection": {"method": "first", "count": 16}}
    suite = build_suite(definition, environment)
    assert [sample["sample_id"] for sample in suite.samples] == [
        "clef-invoice", "clef-outage", "clef-paid-invoice", "clef-healthy-service", "clef-multilingual",
        "clef-receipt", "clef-video"]
    assert all(isinstance(sample["request"]["document"], str) for sample in suite.samples)
