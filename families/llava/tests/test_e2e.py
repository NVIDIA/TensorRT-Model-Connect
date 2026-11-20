# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Official checkpoint -> public builder -> native Task API -> HF greedy parity."""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import pytest
from PIL import Image
from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_enabled, evidence_stage, record_evidence

ROOT = Path(__file__).parent
MANIFEST = json.loads((ROOT / "manifests/llava-1.5-7b.json").read_text())
CASES = {case["name"]: case for case in MANIFEST["testcases"]}


def pytest_generate_tests(metafunc):
    if "case_name" not in metafunc.fixturenames:
        return
    models = {x for raw in metafunc.config.getoption("--e2e-model") for x in raw.split(",")}
    tests = {x for raw in metafunc.config.getoption("--e2e-testcase") for x in raw.split(",")}
    models_file = metafunc.config.getoption("--e2e-models-file")
    if models_file:
        models.update(Path(models_file).read_text().splitlines())
    enabled = bool(models or tests)
    cases = [
        case
        for case in MANIFEST["testcases"]
        if (not models or models & {"llava", MANIFEST["name"], case["name"]})
        and (not tests or case["name"] in tests)
    ]
    metafunc.parametrize(
        "case_name",
        [
            pytest.param(
                case["name"],
                id=case["name"],
                marks=[] if enabled else pytest.mark.skip(reason="select --e2e-model llava"),
            )
            for case in cases
        ],
    )


@pytest.fixture(scope="module")
def execution(tmp_path_factory):
    # Run shared preparation inside the selected test's evidence context.
    outcome = {"stage": "preparation"}

    def prepare():
        if "error" in outcome:
            for name in ("checkpoint", "bundle_inspection", "reference_process"):
                if name in outcome:
                    record_evidence(name, outcome[name])
            with evidence_stage(outcome["stage"]):
                raise outcome["error"]
        if "result" not in outcome:
            try:
                with evidence_stage("preparation"):
                    output = tmp_path_factory.mktemp("llava-e2e")
                outcome["result"] = _prepare_execution(output, outcome)
            except Exception as error:
                outcome["error"] = error
                raise
        return outcome["result"]

    return prepare


def _prepare_execution(output, outcome):
    outcome["stage"] = "checkpoint"
    with evidence_stage("checkpoint"):
        explicit_model = os.environ.get("TRTMC_LLAVA_MODEL_DIR")
        if explicit_model:
            model = Path(explicit_model)
        else:
            from huggingface_hub import snapshot_download

            model = Path(
                snapshot_download(
                    MANIFEST["hf_id"], revision=MANIFEST["hf_revision"], local_files_only=True
                )
            )
        assert (model / "config.json").is_file(), "LLaVA checkpoint requires config.json"
        outcome["checkpoint"] = {
            "resolved_directory": str(model.resolve()),
            "from_override": bool(explicit_model),
            "resolved_revision": None if explicit_model else MANIFEST["hf_revision"],
        }
        record_evidence("checkpoint", outcome["checkpoint"])
    outcome["stage"] = "preparation"
    with evidence_stage("preparation"):
        assert MANIFEST["tensor_parallel_size"] == 1
        binary = Path(os.environ["TRTMC_BINARY"])
        runtime = Path(os.environ["TRTMC_RUNTIME_ROOT"])
        reference_cases = []
        for case in MANIFEST["testcases"]:
            assert case["reference_backend"] == "hf_transformers"
            assert case["reference_precision"] == MANIFEST["precision"]
            if "solid_color" in case:
                image_path = output / (case["name"] + ".bmp")
                Image.new("RGB", tuple(case["image_size"]), case["solid_color"]).save(image_path)
            else:
                image_path = ROOT / case["test_image"]
            assert image_path.is_file()
            reference_cases.append(
                {
                    "name": case["name"],
                    "image": str(image_path),
                    "prompt": case["prompt"],
                    "max_new_tokens": case["max_new_tokens"],
                }
            )
        cases_path = output / "reference-cases.json"
        cases_path.write_text(json.dumps(reference_cases))
        bundle = output / MANIFEST["bundle"]
    started = time.monotonic()
    outcome["stage"] = "build"
    with evidence_stage("build"):
        build(
            BuildRequest(
                model_dir=model,
                output_path=bundle,
                family="llava",
                task=MANIFEST["task"],
                precision=MANIFEST["precision"],
                max_sequence_length=MANIFEST["max_sequence_length"],
            )
        )
    build_seconds = time.monotonic() - started
    bundle_info = None
    inspection = {"enabled": evidence_enabled()}
    outcome["bundle_inspection"] = inspection
    if inspection["enabled"]:
        try:
            inspected = subprocess.run(
                [str(binary), "inspect", str(bundle)],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            inspection["process"] = {
                "argv": inspected.args,
                "stdout": inspected.stdout,
                "stderr": inspected.stderr,
            }
            record_evidence("bundle_inspection_process", inspection["process"])
            bundle_info = json.loads(inspected.stdout)
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            inspection["error"] = {"type": type(error).__name__, "message": str(error)}
            if isinstance(error, (subprocess.CalledProcessError, subprocess.TimeoutExpired)):
                inspection["process"] = {
                    "argv": error.cmd,
                    "stdout": error.stdout.decode("utf-8", errors="replace")
                    if isinstance(error.stdout, bytes)
                    else error.stdout,
                    "stderr": error.stderr.decode("utf-8", errors="replace")
                    if isinstance(error.stderr, bytes)
                    else error.stderr,
                }
            record_evidence("attachment_error", {"attachment": "bundle_info", "error": str(error)})
    record_evidence("bundle_inspection", inspection)
    python = os.environ.get("TRTMC_REFERENCE_PYTHON", sys.executable)
    reference_env = os.environ.copy()
    reference_env.pop("LD_LIBRARY_PATH", None)
    started = time.monotonic()
    outcome["stage"] = "reference"
    with evidence_stage("reference"):
        process = subprocess.run(
            [
                python,
                str(ROOT / "reference.py"),
                "--model-dir",
                str(model),
                "--output",
                str(output),
                "--cases",
                str(cases_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=reference_env,
        )
        outcome["reference_process"] = {
            "argv": process.args,
            "stdout": process.stdout,
            "stderr": process.stderr,
        }
        record_evidence("reference_process", outcome["reference_process"])
        reference = json.loads((output / "reference.json").read_text())
    preparation = {
        "shared": True,
        "cases": [case["name"] for case in MANIFEST["testcases"]],
        "build_seconds": build_seconds,
        "reference_seconds": time.monotonic() - started,
        "bundle_info": bundle_info,
        "bundle_inspection": inspection,
        "checkpoint": outcome["checkpoint"],
        "reference_process": outcome["reference_process"],
        "output_directory": str(output),
    }
    return binary, runtime, bundle, reference, preparation


def test_official_checkpoint_e2e(case_name, execution, tmp_path):
    case = CASES[case_name]
    record_evidence("inputs", {"manifest": MANIFEST, "case": case, "prompt": case["prompt"]})
    record_evidence(
        "checkpoint",
        {
            "requested_repo_id": MANIFEST["hf_id"],
            "requested_revision": MANIFEST["hf_revision"],
            "precision": MANIFEST["precision"],
        },
    )
    record_evidence(
        "thresholds", {"expected_word": case["expected_word"], "token_ids": "exact equality"}
    )
    binary, runtime, bundle, reference, preparation = execution()
    record_evidence("checkpoint", preparation["checkpoint"])
    record_evidence("preparation", preparation)
    record_evidence("bundle_inspection", preparation["bundle_inspection"])
    record_evidence("reference_process", preparation["reference_process"])
    record_evidence(
        "bundle", {"path": str(bundle), "info": preparation["bundle_info"], "shared": True}
    )
    with evidence_stage("reference"):
        expected = reference[case_name]
    reference_record = {
        **expected,
        "mode": case["reference_backend"],
        "precision": case["reference_precision"],
        "implementation": "families/llava/tests/reference.py",
    }
    if evidence_enabled():
        # PNG preserves the reference's RGB pixels, including BMP color fixtures.
        # Native execution still receives the original source file below.
        try:
            image = tmp_path / "input.png"
            Image.open(expected["image"]).convert("RGB").save(image)
            record_evidence("inputs", {"image": image})
            reference_record["image"] = image
        except (OSError, ValueError) as error:
            record_evidence("attachment_error", {"attachment": "input.png", "error": str(error)})
        try:
            raw_reference = tmp_path / "reference-record.json"
            raw_reference.write_text(json.dumps(expected, indent=2))
            record_evidence("reference_record", raw_reference)
        except (OSError, TypeError, ValueError) as error:
            record_evidence(
                "attachment_error", {"attachment": "reference-record.json", "error": str(error)}
            )
        intermediate = {}
        for name in ("pixels", "features"):
            source = Path(preparation["output_directory"]) / f"{case_name}-{name}.npy"
            snapshot = tmp_path / source.name
            try:
                shutil.copyfile(source, snapshot)
                intermediate[name] = snapshot
            except OSError as error:
                record_evidence("attachment_error", {"attachment": name, "error": str(error)})
        record_evidence("reference_intermediates", intermediate)
    record_evidence("reference", reference_record)
    argv = [
        str(binary),
        "run",
        str(bundle),
        "--runtime-root",
        str(runtime),
        "--prompt",
        case["prompt"],
        "--image",
        expected["image"],
        "--max-new-tokens",
        str(case["max_new_tokens"]),
        "--temperature",
        "0",
    ]
    with evidence_stage("native"):
        process = subprocess.run(argv, check=True, capture_output=True, text=True, timeout=180)
        record_evidence(
            "native_process", {"argv": argv, "stdout": process.stdout, "stderr": process.stderr}
        )
        native = json.loads(process.stdout.strip().splitlines()[-1])
    record_evidence("native", native)
    word = case["expected_word"]
    with evidence_stage("compare"):
        assert word in re.findall(r"[a-z]+", expected["text"].lower()), expected
        assert word in re.findall(r"[a-z]+", native["text"].lower()), native
        assert native["token_ids"] == expected["token_ids"], {
            "native": native,
            "reference": expected,
        }
