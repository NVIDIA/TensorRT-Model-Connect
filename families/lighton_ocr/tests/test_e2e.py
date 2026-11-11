# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fresh original-checkpoint OCR through the public builder and native interfaces."""

import json
import os
from pathlib import Path
import struct
import subprocess

import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_stage, record_evidence

from families.lighton_ocr.tests.oracle import compare_generation, compare_observations, validate_generation


FAMILY = "lighton_ocr"
ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "manifests/lightonocr-2-1b-fp32.json").read_text())
CASES = {case["name"]: case for case in MANIFEST["testcases"]}
assert len(CASES) == len(MANIFEST["testcases"]) == 1
THRESHOLD_NAMES = {
    "pixel_maximum_absolute_error", "image_features_relative_l2", "image_features_maximum_absolute_error",
    "logits_relative_l2", "logits_maximum_absolute_error", "logits_argmax_agreement",
    "generation_exact_token_ids", "generation_exact_text", "generation_nonempty", "generation_reaches_eos",
    "teacher_forced_steps",
}


def _selection(config):
    models = list(config.getoption("--e2e-model", default=[]) or [])
    testcases = config.getoption("--e2e-testcase", default=[]) or []
    models_file = config.getoption("--e2e-models-file", default=None)
    if models_file:
        path = Path(models_file)
        assert path.is_file(), "The selected E2E models file must exist"
        models.extend(line for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#"))
    def names(values):
        return {name.strip() for value in values for name in str(value).split(",") if name.strip()}
    return names(models), names(testcases)


def _require_selected(case_name, config):
    models, testcases = _selection(config)
    known = {FAMILY, MANIFEST["name"], MANIFEST["hf_id"], *CASES}
    local = {value for value in models if value.startswith((FAMILY, "lightonocr", "lightonai/LightOnOCR"))}
    assert not local - known, f"Unknown LightOnOCR selection: {sorted(local - known)}"
    local_cases = {value for value in testcases if value.startswith((FAMILY, "lightonocr", "lightonai/LightOnOCR"))}
    assert not local_cases - CASES.keys(), f"Unknown LightOnOCR testcase selection: {sorted(local_cases - CASES.keys())}"
    if models and not ({FAMILY, MANIFEST["name"], MANIFEST["hf_id"], case_name} & models):
        pytest.skip("This LightOnOCR model was not selected")
    if models and models <= known and testcases:
        assert not testcases - CASES.keys(), f"Unknown LightOnOCR testcase selection: {sorted(testcases - CASES.keys())}"
    if testcases and case_name not in testcases:
        pytest.skip("This LightOnOCR testcase was not selected")
    if not models and not testcases and os.environ.get("TRTMC_E2E") != "1":
        pytest.skip("The real LightOnOCR E2E requires explicit selection or TRTMC_E2E=1")


def _path(variable, *, directory=False):
    value = os.environ.get(variable)
    assert value, f"Selected LightOnOCR E2E requires {variable}"
    path = Path(value).expanduser().absolute()
    assert path.is_dir() if directory else path.is_file(), variable
    return path


def _checkpoint():
    if os.environ.get("TRTMC_LIGHTON_OCR_MODEL_DIR"):
        return _path("TRTMC_LIGHTON_OCR_MODEL_DIR", directory=True)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(MANIFEST["hf_id"], revision=MANIFEST["hf_revision"], local_files_only=True))


def _run(command, output, environment):
    error_output = output.with_suffix(output.suffix + ".stderr.log")
    try:
        with output.open("w") as stdout, error_output.open("w") as stderr:
            subprocess.run([str(value) for value in command], check=True, stdout=stdout, stderr=stderr,
                           env=environment, timeout=21600)
    finally:
        record_evidence(output.stem, {"stdout": output, "stderr": error_output})


def _runtime(bundle):
    with bundle.open("rb") as stream:
        assert stream.read(8) == b"BUNDLE\x01\x00"
        length, = struct.unpack("<Q", stream.read(8))
        assert 0 < length < 100 * 1024 * 1024
        header = json.loads(stream.read(length))
        section = header["sections"]["runtime.json"]
        stream.seek(16 + length + section["offset"])
        return json.loads(stream.read(section["length"]))


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.e2e
@pytest.mark.parametrize("case_name", sorted(CASES))
def test_lighton_ocr_native_e2e(case_name, tmp_path, request):
    _require_selected(case_name, request.config)
    case = CASES[case_name]
    limits = json.loads((ROOT / "thresholds" / (case_name + ".json")).read_text())["threshold_overrides"]
    assert set(limits) == THRESHOLD_NAMES and limits["teacher_forced_steps"] == 2
    assert case["premerge"] is True and case["prompt"] == "" and case["do_sample"] is False
    checkpoint = _checkpoint()
    binary = _path("TRTMC_BINARY")
    runtime_root = _path("TRTMC_RUNTIME_ROOT", directory=True)
    native_build = _path("TRTMC_NATIVE_BUILD_DIR", directory=True)
    reference_python = _path("TRTMC_LIGHTON_OCR_REFERENCE_PYTHON")
    image = ROOT / case["test_image"]
    assert image.is_file()
    for library in ("libtrtmc_backend_trt.so", "libtrtmc_model_lighton_ocr.so"):
        assert (runtime_root / library).is_file(), library
    consumers = {name: native_build / ("test_lighton_ocr_" + name)
                 for name in ("diagnostic", "sdk_cpp", "sdk_c")}
    assert all(path.is_file() and os.access(path, os.X_OK) for path in consumers.values())
    artifacts = tmp_path / case_name
    artifacts.mkdir(exist_ok=False)
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = str(runtime_root) + ":" + environment.get("LD_LIBRARY_PATH", "")
    record_evidence("request", {"manifest": MANIFEST, "case": case, "thresholds": limits,
                                "image": image, "working_directory": str(artifacts)})
    reference = artifacts / "reference"
    with evidence_stage("fresh_official_reference"):
        assert not reference.exists()
        _run([reference_python, ROOT / "original_reference.py", checkpoint, image, reference,
              "--max-new-tokens", str(case["max_new_tokens"])], artifacts / "official-reference.log", environment)
        expected = json.loads((reference / "metadata.json").read_text())
        assert expected["model_class"] == "LightOnOcrForConditionalGeneration"
        assert expected["processor_class"] == "LightOnOcrProcessor"
        assert expected["image_processor_class"] == "PixtralImageProcessor"
        assert expected["transformers_version"] == case["reference_transformers_version"]
        assert expected["precision"] == case["reference_precision"] == MANIFEST["precision"] == "fp32"
        assert expected["embedding_dtype"] == "float32" and expected["device"] == "cuda"
        assert expected["use_fast"] is case["reference_use_fast"] is False
        assert expected["use_cache"] is case["reference_use_cache"] is True
        assert expected["do_sample"] is case["do_sample"] is False
        assert expected["max_new_tokens"] == case["max_new_tokens"]
        assert expected["cuda_matmul_allow_tf32"] is False and expected["cudnn_allow_tf32"] is False
        assert expected["float32_matmul_precision"] == "highest"
        assert expected["checkpoint_config"] == json.loads((checkpoint / "config.json").read_text())
        assert expected["processor_config"] == json.loads((checkpoint / "processor_config.json").read_text())
        assert not any(expected["loading_info"].values())
        assert expected["teacher_forced_ids"] == expected["generation"]["token_ids"][:limits["teacher_forced_steps"] - 1]
        vocab_size = expected["checkpoint_config"]["text_config"]["vocab_size"]
        validate_generation(expected["generation"], limits, max_new_tokens=case["max_new_tokens"],
                            expected_fragments=case["expected_fragments"],
                            eos_token_ids=expected["generation"]["eos_token_ids"], vocab_size=vocab_size)
        record_evidence("official_reference", expected)
        record_evidence("official_observation_files", [reference / value["file"] for value in expected["tensors"].values()])
    with evidence_stage("public_bundle_build_and_inspection"):
        if os.environ.get("TRTMC_LIGHTON_OCR_BUNDLE"):
            bundle = _path("TRTMC_LIGHTON_OCR_BUNDLE")
        else:
            bundle = artifacts / MANIFEST["bundle"]
            build(BuildRequest(model_dir=checkpoint, output_path=bundle, family=MANIFEST["family"],
                               task=MANIFEST["task"], precision=MANIFEST["precision"],
                               max_sequence_length=MANIFEST["max_sequence_length"],
                               tensor_parallel_size=MANIFEST["tensor_parallel_size"]))
        _run([binary, "inspect", bundle], artifacts / "bundle-inspect.json", environment)
        inspection = json.loads((artifacts / "bundle-inspect.json").read_text())
        assert inspection["family"] == FAMILY and inspection["task"] == MANIFEST["task"]
        assert inspection["backend"] == "trt"
        assert {"vision.plan", "decoder.plan", "embedding.bin", "runtime.json", "tokenizer.json"} <= set(inspection["sections"])
        runtime = _runtime(bundle)
        assert runtime["precision"] == MANIFEST["precision"]
        assert runtime["max_sequence_length"] == MANIFEST["max_sequence_length"]
        assert runtime["max_new_tokens"] == case["max_new_tokens"]
        assert runtime["text_config"] == expected["checkpoint_config"]["text_config"]
        assert runtime["vision_config"] == expected["checkpoint_config"]["vision_config"]
        record_evidence("bundle_inspection", inspection)
    with evidence_stage("full_native_numerical_oracle"):
        diagnostic_request = artifacts / "diagnostic-request.json"
        diagnostic_request.write_text(json.dumps({"teacher_forced_ids": expected["teacher_forced_ids"]}))
        native = artifacts / "native"
        _run([consumers["diagnostic"], bundle, runtime_root, image, diagnostic_request, native],
             artifacts / "native-diagnostic.log", environment)
        metrics = compare_observations(native, reference, limits)
        metrics_path = artifacts / "reference-metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
        record_evidence("reference_metrics", metrics)
        native_metadata = json.loads((native / "metadata.json").read_text())
        record_evidence("native_metadata", native_metadata)
        record_evidence("native_observation_files", [native / value["file"] for value in native_metadata["tensors"].values()])
        record_evidence("numerical_metrics_file", metrics_path)
    with evidence_stage("public_native_generation"):
        generation_request = artifacts / "generation-request.json"
        generation_request.write_text(json.dumps({"max_new_tokens": case["max_new_tokens"]}))
        commands = {
            "cli": [binary, "run", bundle, "--runtime-root", runtime_root, "--image", image,
                    "--prompt", case["prompt"], "--max-new-tokens", str(case["max_new_tokens"])],
            "cpp": [consumers["sdk_cpp"], bundle, runtime_root, image, generation_request, artifacts / "cpp-output.json"],
            "c": [consumers["sdk_c"], bundle, runtime_root, image, generation_request, artifacts / "c-output.json"],
        }
        for name, command in commands.items():
            output = artifacts / (name + "-output.json")
            _run(command, output if name == "cli" else artifacts / (name + "-generation.log"), environment)
            actual = json.loads(output.read_text())
            compare_generation(actual, expected["generation"], limits, max_new_tokens=case["max_new_tokens"],
                               expected_fragments=case["expected_fragments"], vocab_size=vocab_size)
            if name in {"c", "cpp"}:
                assert actual["result_retained_after_model_release"] is True
            if name == "cpp":
                assert actual["repeat_matches"] is True and actual["rejection_recovery"] is True
            record_evidence(name + "_native_output", actual)
            record_evidence(name + "_native_output_file", output)
