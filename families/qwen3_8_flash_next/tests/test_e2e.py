# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the released checkpoint and exercise the public text, image, and ordered-video tasks."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from tools.e2e_evidence import evidence_stage, record_evidence

_FAMILY = "qwen3_8_flash_next"
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

    roots = [Path("/opt/trtmc-ci")]
    if root := os.environ.get("TRTMC_QWEN3_8_FLASH_NEXT_MODEL_DIR"):
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
            f"Stage {asset['url']} as {asset['path']} in TRTMC_QWEN3_8_FLASH_NEXT_MODEL_DIR"
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
    media = {"text": {}, "image": {"image": frames["image"]}, "video": {"video": video}}
    return [{"id": r["id"], "prompt": r["prompt"], **media[r["id"]]} for r in case["requests"]]


def _check_answers(expected, actual):
    assert [r["id"] for r in actual] == [r["id"] for r in expected]
    for ref, result in zip(expected, actual, strict=True):
        assert 0 < len(result["token_ids"]) <= 96, ref["id"]
        assert result["text"].strip().casefold() == ref["expected"].strip().casefold(), (
            ref["id"],
            ref["expected"],
            result["text"],
        )


@pytest.mark.parametrize(("manifest", "case"), list(_cases()))
def test_e2e(manifest, case, pytestconfig, tmp_path):
    if not _selected(pytestconfig, manifest, case):
        pytest.skip("select the released checkpoint E2E explicitly or set TRTMC_E2E=1")
    if case.get("component") is not None:
        assert case["component"] == "packed_lexical_rows"
        from families.qwen3_8_flash_next.tests.lexical_stage import check_packed_rows

        with evidence_stage("native_lexical_stage"):
            consumer = (
                Path(os.environ["TRTMC_NATIVE_BUILD_DIR"]) / "test_qwen3_8_flash_next_lexical_stage"
            )
            check_packed_rows(tmp_path, consumer)
            record_evidence(
                "scope", "native packed lexical stage only; no language or vision generation"
            )
        return
    from families.qwen3_8_flash_next.cli import build
    from huggingface_hub import constants, snapshot_download

    runtime_root = Path(os.environ["TRTMC_RUNTIME_ROOT"])
    consumer = Path(os.environ["TRTMC_NATIVE_BUILD_DIR"]) / "test_qwen3_8_flash_next_sdk_cpp"
    assert consumer.is_file(), consumer
    assert (runtime_root / f"libtrtmc_model_{manifest['family']}.so").is_file(), runtime_root
    requests = tmp_path / "requests.json"
    requests.write_text(json.dumps(_prepare_requests(manifest, case, tmp_path)))
    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("checkpoint"):
        checkpoint = snapshot_download(
            repo_id=manifest["hf_id"],
            revision=manifest["hf_revision"],
            local_files_only=constants.HF_HUB_OFFLINE,
        )
    with evidence_stage("build"):
        assert (
            build(
                model=checkpoint,
                output=bundle,
                precision=manifest["precision"],
                task=manifest["task"],
                max_sequence_length=manifest["max_sequence_length"],
                edge_builder_python=os.environ.get("TRTMC_QWEN3_8_FLASH_NEXT_EDGE_BUILDER_PYTHON"),
            )
            == 0
        )
    with evidence_stage("inspect"):
        info = json.loads(subprocess.check_output([os.environ["TRTMC_BINARY"], "inspect", bundle]))
        assert info["family"] == manifest["family"]
        assert info["task"] == manifest["task"]
        markers = {"native.json": "native", "edge_llm.json": "edge"}
        routes = [route for key, route in markers.items() if key in info["sections"]]
        assert len(routes) == 1, routes
        record_evidence("execution_route", routes[0])
        if required := os.environ.get("TRTMC_QWEN3_8_FLASH_NEXT_EXPECT_ROUTE"):
            assert required in {"native", "edge"} and routes == [required], (required, routes)
    answers = tmp_path / "answers.json"
    with evidence_stage("inference"):
        subprocess.run([consumer, bundle, runtime_root, requests, answers], check=True)
    with evidence_stage("quality"):
        actual = record_evidence("answers", json.loads(answers.read_text()))
        _check_answers(case["requests"], actual)
