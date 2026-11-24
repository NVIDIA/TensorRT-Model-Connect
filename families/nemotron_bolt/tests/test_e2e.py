# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the released checkpoint and exercise the public text, image, and ordered-video tasks."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.e2e_evidence import evidence_stage, record_evidence

_FAMILY = "nemotron_bolt"
_MANIFEST_DIR = Path(__file__).parent / "manifests"


def _cases():
    for path in sorted(_MANIFEST_DIR.glob("*.json")):
        manifest = json.loads(path.read_text())
        assert manifest["family"] == _FAMILY
        assert manifest["tensor_parallel_size"] == 1
        for case in manifest["testcases"]:
            yield pytest.param(manifest, case, id=case["name"])


def _selected(config, manifest, case):
    keys = set()
    for option in ("--e2e-testcase", "--e2e-model"):
        for value in config.getoption(option, default=[]) or []:
            keys.update(x.strip() for x in value.split(",") if x.strip())
    filename = config.getoption("--e2e-models-file", default=None)
    if filename:
        keys.update(
            x.strip()
            for x in Path(filename).read_text().splitlines()
            if x.strip() and not x.lstrip().startswith("#")
        )
    if keys:
        return bool(keys & {manifest["family"], manifest["name"], case["name"]})
    return os.environ.get("TRTMC_E2E") == "1"


def _prepare_requests(manifest, case, directory):
    from PIL import Image, ImageOps

    requests = [{k: v for k, v in r.items() if k != "expected"} for r in case["requests"]]
    if not any(r["id"] in ("species", "video-order") for r in requests):
        return requests

    roots = [Path("/opt/trtmc-ci")]
    if root := os.environ.get("TRTMC_NEMOTRON_BOLT_MODEL_DIR"):
        roots.insert(0, Path(root))
    frames = {}

    def raw(name, image):
        path = directory / (name + ".rgb")
        path.write_bytes(image.tobytes())
        return {"path": str(path), "height": image.height, "width": image.width}

    for asset in manifest["external_files"]:
        path = next(
            (root / asset["path"] for root in roots if (root / asset["path"]).is_file()), None
        )
        assert path is not None, (
            f"Stage {asset['url']} as {asset['path']} in TRTMC_NEMOTRON_BOLT_MODEL_DIR"
        )
        with Image.open(path) as source:
            rgb = source.convert("RGB")
            if path.stem == "giant_panda":
                frames["image"] = raw("image", rgb)
            frames[path.stem] = raw(
                path.stem,
                ImageOps.pad(rgb, (512, 512), method=Image.Resampling.LANCZOS, color="black"),
            )

    video = {
        "frames": [frames[k] for k in ("red_panda", "red_panda", "giant_panda", "giant_panda")],
        "timestamps": [0.0, 1.0, 2.0, 3.0],
    }
    for request in requests:
        if request["id"] == "species":
            request["image"] = frames["image"]
        elif request["id"] == "video-order":
            request["video"] = video
    return requests


def _check_answers(expected, actual):
    assert [r["id"] for r in actual] == [r["id"] for r in expected]
    misses = {}
    seeded_ids = []
    for ref, result in zip(expected, actual, strict=True):
        ids = result["token_ids"]
        assert 0 < len(ids) <= ref["max_new_tokens"] and all(type(i) is int for i in ids)
        if result["text"].strip().casefold() != ref["expected"].casefold():
            misses[ref["id"]] = {"expected": ref["expected"], "got": result["text"]}
        if "seed" in ref:
            seeded_ids.append(ids)
    if seeded_ids:
        assert len(seeded_ids) == 2 and seeded_ids[0] == seeded_ids[1], "seed did not repeat"
    assert not misses, misses


def _snapshot(hf_id, revision):
    from huggingface_hub import constants, snapshot_download

    return snapshot_download(
        repo_id=hf_id, revision=revision, local_files_only=constants.HF_HUB_OFFLINE
    )


@pytest.fixture(scope="module")
def built_bundle(tmp_path_factory):
    current_key = None
    bundle = None

    def prepare(manifest, case):
        nonlocal current_key, bundle
        draft = case.get("companion", {})
        key = (
            manifest["hf_id"],
            manifest["hf_revision"],
            draft.get("hf_id"),
            draft.get("hf_revision"),
            manifest["task"],
            manifest["precision"],
            manifest["max_sequence_length"],
        )
        if key != current_key:
            current_key = None
            if bundle is not None:
                bundle.unlink(missing_ok=True)
            with evidence_stage("checkpoint"):
                checkpoint = _snapshot(manifest["hf_id"], manifest["hf_revision"])
                pair = []
                if draft:
                    local = _snapshot(draft["hf_id"], draft["hf_revision"])
                    pair = ["--execution-variant", "lilicorr", "--companion", f"draft={local}"]
            bundle = tmp_path_factory.mktemp("bolt") / manifest["bundle"]
            with evidence_stage("build"):
                # Keep a fatal compiler error in the build command, so pytest can report it.
                command = [
                    sys.executable,
                    "-X",
                    "faulthandler",
                    "-m",
                    "tensorrt_model_connect",
                    _FAMILY,
                    "build",
                    str(checkpoint),
                    "-o",
                    str(bundle),
                    "--task",
                    manifest["task"],
                    "--precision",
                    manifest["precision"],
                    "--max-sequence-length",
                    str(manifest["max_sequence_length"]),
                    "--verbose",
                    *pair,
                ]
                if builder_python := os.environ.get("TRTMC_NEMOTRON_BOLT_EDGE_BUILDER_PYTHON"):
                    command.extend(["--edge-builder-python", builder_python])
                try:
                    subprocess.run(command, check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as exc:
                    print(exc.stderr, file=sys.stderr, end="")
                    raise
            current_key = key
        return bundle

    yield prepare
    if bundle is not None:
        bundle.unlink(missing_ok=True)


@pytest.mark.parametrize(("manifest", "case"), list(_cases()))
def test_e2e(manifest, case, pytestconfig, tmp_path, built_bundle):
    if not _selected(pytestconfig, manifest, case):
        pytest.skip("select a released Bolt E2E explicitly or set TRTMC_E2E=1")
    runtime_root = Path(os.environ["TRTMC_RUNTIME_ROOT"])
    consumer = Path(os.environ["TRTMC_NATIVE_BUILD_DIR"]) / "test_nemotron_bolt_sdk_cpp"
    assert consumer.is_file(), consumer
    requests = tmp_path / "requests.json"
    requests.write_text(json.dumps(_prepare_requests(manifest, case, tmp_path)))
    bundle = built_bundle(manifest, case)
    with evidence_stage("inspect"):
        info = json.loads(subprocess.check_output([os.environ["TRTMC_BINARY"], "inspect", bundle]))
        assert info["family"] == manifest["family"] and info["task"] == manifest["task"]
        sections = set(info["sections"])
        assert ("native.json" in sections) != ("edge_llm.json" in sections)
        native = "native.json" in sections
        route = record_evidence("execution_route", "native" if native else "edge")
        if required := os.environ.get("TRTMC_NEMOTRON_BOLT_EXPECT_ROUTE"):
            assert route == required
        pair = (
            {"draft/lattice.plan", "draft/4.plan"}
            if native
            else {"edge_llm/engine/spec_base.engine", "edge_llm/engine/spec_draft.engine"}
        )
        assert (sections & pair) == (pair if "companion" in case else set()), (
            "incorrect LiLiCorr bundle"
        )
    answers = tmp_path / "answers.json"
    with evidence_stage("inference"):
        subprocess.run([consumer, bundle, runtime_root, requests, answers], check=True)
    with evidence_stage("quality"):
        _check_answers(
            case["requests"], record_evidence("answers", json.loads(answers.read_text()))
        )
