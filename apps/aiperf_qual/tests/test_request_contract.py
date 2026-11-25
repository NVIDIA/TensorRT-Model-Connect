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


def test_configured_image_profiles_state_every_control_and_new_ones_are_left_out():
    """An image profile with a configuration of its own states every control the pipeline would otherwise choose.
    A catalog profile without one that leaves a control unstated is left out with the controls it leaves, so a
    family adding an image profile does not configure its qualification."""
    from trtmc_benchmark.types import BenchmarkError
    from trtmc_perf_serving.profiles import resolve_profile

    from trtmc_aiperf_qual.config import CONFIG_ROOT
    from trtmc_aiperf_qual.selection import EXCLUSIONS_SOURCE, MODELS_SOURCE, TASKS_SOURCE, catalog

    root = Path(__file__).resolve().parents[3]
    environment = Environment({"repo": str(root), "bundle_root": "/tmp/bundles"})
    selection = {profile.name: profile for profile in catalog(root)}
    profiles = [entry.name for entry in catalog_profiles(root) if entry.operation == "generate_image"]
    assert profiles
    for profile in profiles:
        model = resolve_model(profile, environment)
        try:
            resolved = resolve_profile(profile, manifest_root=root / "families")
        except BenchmarkError:  # no catalog request (svd-xt: no prompt): kept, its run reports what it finds
            assert selection[profile].source != MODELS_SOURCE, profile
            continue
        missing = unstated_defaults({**resolved.base_request, **model.get("catalog_request", {})}, model["operation"])
        assert not (missing and (CONFIG_ROOT / "models" / f"{profile}.yaml").is_file()), (profile, missing)
        if selection[profile].source not in (TASKS_SOURCE, EXCLUSIONS_SOURCE):  # left out earlier for another reason
            assert (selection[profile].source == MODELS_SOURCE) == bool(missing), profile
            assert not missing or ", ".join(missing) in selection[profile].excluded


def test_a_configured_image_profile_that_stops_stating_a_control_is_left_out(tmp_path):
    import shutil

    from trtmc_aiperf_qual.config import CONFIG_ROOT
    from trtmc_aiperf_qual.selection import MODELS_SOURCE, catalog

    root = Path(__file__).resolve().parents[3]
    config = tmp_path / "config"
    shutil.copytree(CONFIG_ROOT, config)
    flux = config / "models/flux-schnell.yaml"
    flux.write_text("".join(line for line in flux.read_text().splitlines(True) if not line.startswith("request:")))
    before, after = ({profile.name: profile for profile in catalog(root, path)}["flux-schnell"]
                     for path in (CONFIG_ROOT, config))
    assert before.excluded is None
    assert after.source == MODELS_SOURCE and "guidance_scale" in after.excluded


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
