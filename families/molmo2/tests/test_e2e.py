# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct build, upstream preprocessing, and native Task generation qualification.

Every manifest under ``manifests/`` is one bundle: the BF16 build and the FP8
build (``quantization`` in the manifest) answer the same requests and must
match the upstream BF16 reference token for token.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pytest

from families.molmo2.tests.inputs import checkpoint_directory, native_tool
from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_enabled, evidence_stage, record_evidence
from tools.ci.environment import prepared_python

ROOT = Path(__file__).resolve().parent
MANIFESTS = {
    path.name: json.loads(path.read_text()) for path in sorted((ROOT / "manifests").glob("*.json"))
}
DEFAULT_MANIFEST = "molmo2-8b.json"
MANIFEST = MANIFESTS[DEFAULT_MANIFEST]
CASES = {
    case["name"]: (file, case)
    for file, manifest in MANIFESTS.items()
    for case in manifest["testcases"]
}


def _selected(config, manifest, case):
    models = {part for arg in config.getoption("--e2e-model") for part in arg.split(",")}
    selection_file = config.getoption("--e2e-models-file")
    if selection_file:
        models.update(Path(selection_file).read_text().splitlines())
    cases = {part for arg in config.getoption("--e2e-testcase") for part in arg.split(",")}
    return (
        bool(models or cases)
        and (not models or models.intersection({"molmo2", manifest["name"], case["name"]}))
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


def _validate_bundle(inspection, manifest=MANIFEST):
    assert inspection["family"] == manifest["family"]
    assert inspection["task"] == manifest["task"] and inspection["backend"] == "trt"
    provenance = inspection.get("provenance")
    assert isinstance(provenance, dict) and provenance.get("version") == 1, (
        "Molmo2 qualification requires recorded build provenance"
    )
    assert provenance.get("checkpoint") == {
        "hf_id": manifest["hf_id"],
        "revision": manifest["hf_revision"],
    }, "Molmo2 bundle checkpoint identity does not match the selected manifest"
    expected = {
        "precision": manifest["precision"],
        "max_sequence_length": manifest["max_sequence_length"],
        "tensor_parallel_size": manifest["tensor_parallel_size"],
        "context_parallel_size": 1,
        "max_batch_size": 1,
        "image_height": None,
        "image_width": None,
        "video_num_frames": None,
        "quantization": manifest.get("quantization"),
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
    # Keep both outcomes so a failed shared build is never retried per case.
    outcomes = {}

    def prepare(manifest_file=DEFAULT_MANIFEST):
        outcome = outcomes.setdefault(manifest_file, {"stage": "preparation"})
        if "error" in outcome:
            for name in ("checkpoint", "native_process", "qualification_results"):
                if name in outcome:
                    record_evidence(name, outcome[name])
            with evidence_stage(outcome["stage"]):
                raise outcome["error"]
        if "result" not in outcome:
            try:
                outcome["result"] = _prepare_artifacts(tmp_path_factory, outcome, manifest_file)
            except Exception as error:
                outcome["error"] = error
                raise
        return outcome["result"]

    return prepare


def _prepare_artifacts(tmp_path_factory, outcome, manifest_file):
    manifest = MANIFESTS[manifest_file]
    started = time.monotonic()
    with evidence_stage("preparation"):
        root = tmp_path_factory.mktemp("molmo2")
    outcome["stage"] = "checkpoint"
    with evidence_stage("checkpoint"):
        checkpoint = checkpoint_directory(manifest["hf_id"], manifest["hf_revision"])
        overridden = "TRTMC_MOLMO2_MODEL_DIR" in os.environ
        outcome["checkpoint"] = {
            "resolved_directory": str(checkpoint.resolve()),
            "from_override": overridden,
            "resolved_revision": None if overridden else manifest["hf_revision"],
        }
        record_evidence("checkpoint", outcome["checkpoint"])
    outcome["stage"] = "preparation"
    with evidence_stage("preparation"):
        runtime = _path("TRTMC_RUNTIME_ROOT")
        binary = _path("TRTMC_BINARY")
    bundle = Path(os.environ.get("TRTMC_MOLMO2_BUNDLE", root / manifest["bundle"]))
    reference = os.environ.get("TRTMC_MOLMO2_REFERENCE_RESULTS")
    python = None
    if "TRTMC_MOLMO2_BUNDLE" not in os.environ:
        if not reference:
            outcome["stage"] = "reference"
            with evidence_stage("reference"):
                python = _reference_python()
        outcome["stage"] = "build"
        with evidence_stage("build"):
            build(
                BuildRequest(
                    checkpoint,
                    bundle,
                    "molmo2",
                    manifest["task"],
                    manifest["precision"],
                    max_sequence_length=manifest["max_sequence_length"],
                    tensor_parallel_size=manifest["tensor_parallel_size"],
                    quantization=manifest.get("quantization"),
                )
            )
    outcome["stage"] = "build"
    with evidence_stage("build"):
        assert bundle.is_file()
        inspection = json.loads(
            subprocess.check_output([str(binary), "inspect", str(bundle)], text=True)
        )
        _validate_bundle(inspection, manifest)
    outcome["stage"] = "reference"
    if reference:
        reference = Path(reference)
        reference_source = {"source": "supplied", "directory": str(reference.absolute())}
    else:
        reference = root / "reference"
        with evidence_stage("reference"):
            python = python or _reference_python()
            subprocess.run(
                [
                    str(python),
                    str(ROOT / "reference.py"),
                    "--model-dir",
                    str(checkpoint),
                    "--output",
                    str(reference),
                    "--manifest",
                    str(ROOT / "manifests" / manifest_file),
                ],
                check=True,
                timeout=1800,
            )
        # reference.py enforces this version before generating any results.
        reference_source = {
            "source": "generated",
            "mode": "hf_transformers",
            "transformers_version": "4.57.1",
            "implementation": "families/molmo2/tests/reference.py",
            "checkpoint": outcome["checkpoint"],
        }
    with evidence_stage("reference"):
        results = {
            case["name"]: case for case in json.loads((reference / "reference.json").read_text())
        }
    outcome["stage"] = "native"
    with evidence_stage("native"):
        qualifier = native_tool("molmo2_qualification")
        completed = subprocess.run(
            [
                str(qualifier),
                str(bundle),
                str(runtime),
                str(reference),
                str(ROOT / "manifests" / manifest_file),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        outcome["native_process"] = {
            "argv": completed.args,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        record_evidence("native_process", outcome["native_process"])
        native = [
            json.loads(line) for line in completed.stdout.splitlines() if line.startswith('[{"')
        ][-1]
    outcome["qualification_results"] = {
        "reference_source": reference_source,
        "pairs": [
            {
                "case_name": output.get("name"),
                "native": output,
                "reference": results.get(output.get("name")),
            }
            for output in native
        ],
    }
    record_evidence("qualification_results", outcome["qualification_results"])
    outcome["stage"] = "compare"
    with evidence_stage("compare"):
        assert len(native) == len(manifest["testcases"]) + 1
        for output in native:
            expected = results[output["name"]]
            assert output["token_ids"] == expected["tokens"]
            assert output["text"] == expected["text"]
        assert native[0]["token_ids"] == native[-1]["token_ids"]
    preparation = {
        "shared": True,
        "seconds": time.monotonic() - started,
        "cases": [case["name"] for case in manifest["testcases"]],
        "bundle_info": inspection,
        "checkpoint": outcome["checkpoint"],
        "native_process": outcome["native_process"],
        "replay": native[-1],
    }
    return checkpoint, bundle, reference, results, native, preparation, reference_source


@pytest.mark.e2e
@pytest.mark.parametrize("case_name", list(CASES))
def test_upstream_and_native_e2e(case_name, request, tmp_path):
    manifest_file, case = CASES[case_name]
    manifest = MANIFESTS[manifest_file]
    if not _selected(request.config, manifest, case):
        pytest.skip("Molmo2 E2E requires an explicit family or testcase selector")
    record_evidence("inputs", {"manifest": manifest, "case": case, "prompt": case["prompt"]})
    record_evidence(
        "checkpoint",
        {"requested_repo_id": manifest["hf_id"], "requested_revision": manifest["hf_revision"]},
    )
    record_evidence(
        "thresholds",
        {
            "token_ids": "exact equality",
            "text": "exact equality",
            "pixel_atol": 2 / 255 + 1e-6,
            "pixel_rtol": 0,
            "input_ids": "exact equality",
            "pooling": "exact equality",
        },
    )
    checkpoint, bundle, reference, results, outputs, preparation, reference_source = (
        request.getfixturevalue("artifacts")(manifest_file)
    )
    name = case["name"]
    expected = results[name]
    record_evidence("checkpoint", preparation["checkpoint"])
    record_evidence("preparation", preparation)
    record_evidence("bundle", {"path": str(bundle), "info": preparation["bundle_info"]})
    record_evidence(
        "reference",
        {
            **expected,
            **reference_source,
        },
    )
    output = next(item for item in outputs if item["name"] == name)
    record_evidence("native", output)
    from PIL import Image

    expected_image = (
        Image.open(ROOT / case["test_image"]).convert("RGB")
        if "test_image" in case
        else Image.new("RGB", tuple(case["size"]), tuple(case["color"]))
    )
    if evidence_enabled():
        try:
            image = tmp_path / "input.png"
            expected_image.save(image)
            record_evidence("inputs", {"image": image})
        except (OSError, ValueError) as error:
            record_evidence("attachment_error", {"attachment": "input.png", "error": str(error)})
    with evidence_stage("compare"):
        assert (expected["width"], expected["height"]) == expected_image.size
        np.testing.assert_array_equal(
            np.fromfile(reference / (name + ".f32"), dtype=np.float32),
            np.asarray(expected_image, dtype=np.float32).reshape(-1),
        )
    stem = tmp_path / name
    with evidence_stage("native"):
        helper = native_tool("test_molmo2_preprocess")
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
        actual_pooling = np.fromfile(stem.with_suffix(".pooling"), np.int32)
        actual_pixels = np.fromfile(stem.with_suffix(".pixels"), np.float32)
    record_evidence("native_preprocessing", native)
    with evidence_stage("reference"):
        expected_pooling = np.load(reference / (name + "-image_token_pooling.npy")).reshape(-1)
        expected_pixels = np.load(reference / (name + "-pixel_values.npy")).reshape(-1)
    record_evidence("native_pooling", actual_pooling)
    record_evidence("reference_pooling", expected_pooling)
    record_evidence("native_pixels", actual_pixels)
    record_evidence("reference_pixels", expected_pixels)
    with evidence_stage("compare"):
        assert native["ids"] == expected["input_ids"]
        np.testing.assert_array_equal(actual_pooling, expected_pooling)
        record_evidence(
            "metrics",
            {"pixel_max_absolute_error": float(np.max(np.abs(actual_pixels - expected_pixels)))},
        )
        # CPU half-pixel bilinear interpolation may round a uint8 halfway value
        # differently. One uint8 step becomes 2/255 after SigLIP normalization.
        np.testing.assert_allclose(actual_pixels, expected_pixels, atol=2 / 255 + 1e-6, rtol=0)
        assert output["token_ids"] == expected["tokens"]
        assert len(output["token_ids"]) <= case["max_new_tokens"]
        assert output["text"] == expected["text"]
        assert output["text"].strip()
