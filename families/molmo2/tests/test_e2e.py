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

from families.molmo2.tests.inputs import checkpoint_directory, native_tool
from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import record_evidence
from tools.ci.environment import prepared_python

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


def _reference_python():
    if "TRTMC_MOLMO2_REFERENCE_PYTHON" in os.environ:
        python = _path("TRTMC_MOLMO2_REFERENCE_PYTHON")
        assert python.is_file() and os.access(python, os.X_OK), (
            "TRTMC_MOLMO2_REFERENCE_PYTHON is not an executable file"
        )
        return python
    assert MANIFEST["reference_environment"] == "isolated"
    return prepared_python(ROOT.parents[2], MANIFEST["family"], dict(os.environ))


def _validate_bundle(inspection):
    assert inspection["family"] == MANIFEST["family"]
    assert inspection["task"] == MANIFEST["task"] and inspection["backend"] == "trt"
    provenance = inspection.get("provenance")
    assert isinstance(provenance, dict) and provenance.get("version") == 1, (
        "Molmo2 qualification requires recorded build provenance"
    )
    assert provenance.get("checkpoint") == {
        "hf_id": MANIFEST["hf_id"],
        "revision": MANIFEST["hf_revision"],
    }, "Molmo2 bundle checkpoint identity does not match the selected manifest"
    expected = {
        "precision": MANIFEST["precision"],
        "max_sequence_length": MANIFEST["max_sequence_length"],
        "tensor_parallel_size": MANIFEST["tensor_parallel_size"],
        "context_parallel_size": 1,
        "max_batch_size": 1,
        "image_height": None,
        "image_width": None,
        "video_num_frames": None,
        "quantization": None,
        "fp32_layers": [],
        "dynamic_kv_cache": False,
        "mtp_seq_len": None,
        "graph_transform": False,
    }
    recorded = provenance.get("build")
    assert isinstance(recorded, dict), "Molmo2 bundle has no recorded build settings"
    for name, value in expected.items():
        assert (
            name in recorded and type(recorded[name]) is type(value) and recorded[name] == value
        ), f"Molmo2 bundle build.{name} does not match the selected manifest"


@pytest.fixture(scope="session")
def artifacts(tmp_path_factory):
    root = tmp_path_factory.mktemp("molmo2")
    checkpoint = checkpoint_directory(MANIFEST["hf_id"], MANIFEST["hf_revision"])
    runtime = _path("TRTMC_RUNTIME_ROOT")
    binary = _path("TRTMC_BINARY")
    bundle = Path(os.environ.get("TRTMC_MOLMO2_BUNDLE", root / MANIFEST["bundle"]))
    reference = os.environ.get("TRTMC_MOLMO2_REFERENCE_RESULTS")
    python = None
    if "TRTMC_MOLMO2_BUNDLE" not in os.environ:
        if not reference:
            python = _reference_python()
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
    _validate_bundle(inspection)
    if reference:
        reference = Path(reference)
    else:
        reference = root / "reference"
        python = python or _reference_python()
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
    qualifier = native_tool("molmo2_qualification")
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
def test_upstream_and_native_e2e(case, request):
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
    helper = native_tool("test_molmo2_preprocess")
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
