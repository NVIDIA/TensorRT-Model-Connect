# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct build, upstream preprocessing, and native Task generation qualification."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import record_evidence

ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "manifests/molmo2-8b.json").read_text())


def _selected(config, case):
    models = {part for arg in config.getoption("--e2e-model") for part in arg.split(",")}
    selection_file = config.getoption("--e2e-models-file")
    if selection_file:
        models.update(Path(selection_file).read_text().splitlines())
    cases = {part for arg in config.getoption("--e2e-testcase") for part in arg.split(",")}
    return (
        bool(models or cases)
        and (not models or models.intersection({"molmo2", MANIFEST["name"], case["name"]}))
        and (not cases or case["name"] in cases)
    )


def _path(name):
    value = os.environ.get(name)
    assert value and Path(value).exists(), f"selected Molmo2 E2E requires {name}"
    # Resolving a venv's Python symlink selects the base interpreter and loses
    # the isolated reference dependencies.
    return Path(value).absolute()


@pytest.fixture(scope="session")
def artifacts(tmp_path_factory):
    root = tmp_path_factory.mktemp("molmo2")
    checkpoint = _path("TRTMC_MOLMO2_MODEL_DIR")
    runtime = _path("TRTMC_RUNTIME_ROOT")
    binary = _path("TRTMC_BINARY")
    bundle = Path(os.environ.get("TRTMC_MOLMO2_BUNDLE", root / MANIFEST["bundle"]))
    if "TRTMC_MOLMO2_BUNDLE" not in os.environ:
        build(
            BuildRequest(
                checkpoint,
                bundle,
                "molmo2",
                MANIFEST["task"],
                MANIFEST["precision"],
                max_sequence_length=MANIFEST["max_sequence_length"],
                tensor_parallel_size=MANIFEST["tensor_parallel_size"],
            )
        )
    assert bundle.is_file()
    inspection = json.loads(
        subprocess.check_output([str(binary), "inspect", str(bundle)], text=True)
    )
    assert inspection["family"] == "molmo2"
    assert inspection["task"] == MANIFEST["task"] and inspection["backend"] == "trt"
    assert inspection["provenance"]["build"]["precision"] == MANIFEST["precision"]
    reference = os.environ.get("TRTMC_MOLMO2_REFERENCE_RESULTS")
    if reference:
        reference = Path(reference)
    else:
        reference = root / "reference"
        python = _path("TRTMC_MOLMO2_REFERENCE_PYTHON")
        subprocess.run(
            [
                str(python),
                str(ROOT / "reference.py"),
                "--model-dir",
                str(checkpoint),
                "--output",
                str(reference),
            ],
            check=True,
            timeout=1800,
        )
    results = {
        case["name"]: case for case in json.loads((reference / "reference.json").read_text())
    }
    qualifier = runtime / "families/molmo2/molmo2_qualification"
    assert qualifier.is_file()
    completed = subprocess.run(
        [
            str(qualifier),
            str(bundle),
            str(runtime),
            str(reference),
            str(ROOT / "manifests/molmo2-8b.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    native = [json.loads(line) for line in completed.stdout.splitlines() if line.startswith('[{"')][
        -1
    ]
    assert len(native) == len(MANIFEST["testcases"]) + 1
    for output in native:
        expected = results[output["name"]]
        assert output["token_ids"] == expected["tokens"]
        assert output["text"] == expected["text"]
    assert native[0]["token_ids"] == native[-1]["token_ids"]
    (root / "native-results.json").write_text(json.dumps(native, indent=2) + "\n")
    record_evidence(
        "native_process",
        {"argv": completed.args, "stdout": completed.stdout, "stderr": completed.stderr},
    )
    record_evidence(
        "inputs",
        {
            "checkpoint": checkpoint,
            "manifest": ROOT / "manifests/molmo2-8b.json",
            "hf_id": MANIFEST["hf_id"],
            "hf_revision": MANIFEST["hf_revision"],
        },
    )
    record_evidence("bundle", {"path": bundle})
    return root, checkpoint, runtime, binary, bundle, reference, results


@pytest.mark.e2e
@pytest.mark.parametrize("case", MANIFEST["testcases"], ids=lambda case: case["name"])
def test_upstream_and_native(case, request):
    if not _selected(request.config, case):
        pytest.skip("Molmo2 E2E requires an explicit family or testcase selector")
    root, checkpoint, runtime, binary, bundle, reference, results = request.getfixturevalue(
        "artifacts"
    )
    name = case["name"]
    expected = results[name]
    from PIL import Image

    expected_image = (
        Image.open(ROOT / case["test_image"]).convert("RGB")
        if "test_image" in case
        else Image.new("RGB", tuple(case["size"]), tuple(case["color"]))
    )
    assert (expected["width"], expected["height"]) == expected_image.size
    np.testing.assert_array_equal(
        np.fromfile(reference / (name + ".f32"), dtype=np.float32),
        np.asarray(expected_image, dtype=np.float32).reshape(-1),
    )
    helper = runtime / "families/molmo2/test_molmo2_preprocess"
    assert helper.is_file()
    stem = root / name
    subprocess.run(
        [
            str(helper),
            str(expected["height"]),
            str(expected["width"]),
            str(reference / (name + ".f32")),
            str(stem),
            str(checkpoint / "tokenizer.json"),
            case["prompt"],
        ],
        check=True,
        timeout=60,
    )
    native = json.loads(stem.with_suffix(".json").read_text())
    assert native["ids"] == expected["input_ids"]
    actual_pooling = np.fromfile(stem.with_suffix(".pooling"), np.int32)
    expected_pooling = np.load(reference / (name + "-image_token_pooling.npy")).reshape(-1)
    np.testing.assert_array_equal(actual_pooling, expected_pooling)
    actual_pixels = np.fromfile(stem.with_suffix(".pixels"), np.float32)
    expected_pixels = np.load(reference / (name + "-pixel_values.npy")).reshape(-1)
    # CPU half-pixel bilinear interpolation may round a uint8 halfway value
    # differently. One uint8 step becomes 2/255 after SigLIP normalization.
    np.testing.assert_allclose(actual_pixels, expected_pixels, atol=2 / 255 + 1e-6, rtol=0)
    output = next(
        case
        for case in json.loads((root / "native-results.json").read_text())
        if case["name"] == name
    )
    record_evidence("reference", {"result": expected})
    assert output["token_ids"] == expected["tokens"]
    assert len(output["token_ids"]) <= case["max_new_tokens"]
    assert output["text"] == expected["text"]
    assert output["text"].strip()
