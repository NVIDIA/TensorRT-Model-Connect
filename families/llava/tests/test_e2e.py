# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Official checkpoint -> public builder -> native Task API -> HF greedy parity."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import pytest
from PIL import Image
from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import record_evidence

ROOT = Path(__file__).parent
MANIFEST = json.loads((ROOT / "manifests/llava-1.5-7b.json").read_text())


def pytest_generate_tests(metafunc):
    if "case" not in metafunc.fixturenames:
        return
    models = {x for raw in metafunc.config.getoption("--e2e-model") for x in raw.split(",")}
    tests = {x for raw in metafunc.config.getoption("--e2e-testcase") for x in raw.split(",")}
    models_file = metafunc.config.getoption("--e2e-models-file")
    if models_file:
        models.update(Path(models_file).read_text().splitlines())
    enabled = bool(models or tests)
    cases = [case for case in MANIFEST["testcases"]
             if (not models or models & {"llava", MANIFEST["name"], case["name"]})
             and (not tests or case["name"] in tests)]
    metafunc.parametrize("case", [pytest.param(case, id=case["name"],
        marks=[] if enabled else pytest.mark.skip(reason="select --e2e-model llava")) for case in cases])


@pytest.fixture(scope="module")
def execution(tmp_path_factory):
    explicit_model = os.environ.get("TRTMC_LLAVA_MODEL_DIR")
    if explicit_model:
        model = Path(explicit_model)
    else:
        from huggingface_hub import snapshot_download
        model = Path(snapshot_download(MANIFEST["hf_id"], revision=MANIFEST["hf_revision"],
                                       local_files_only=True))
    assert MANIFEST["tensor_parallel_size"] == 1
    binary = Path(os.environ["TRTMC_BINARY"])
    runtime = Path(os.environ["TRTMC_RUNTIME_ROOT"])
    output = tmp_path_factory.mktemp("llava-e2e")
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
        reference_cases.append({"name": case["name"], "image": str(image_path),
                                "prompt": case["prompt"], "max_new_tokens": case["max_new_tokens"]})
    cases_path = output / "reference-cases.json"
    cases_path.write_text(json.dumps(reference_cases))
    bundle = output / MANIFEST["bundle"]
    build(BuildRequest(model_dir=model, output_path=bundle, family="llava",
                       task=MANIFEST["task"], precision=MANIFEST["precision"],
                       max_sequence_length=MANIFEST["max_sequence_length"]))
    python = os.environ.get("TRTMC_REFERENCE_PYTHON", sys.executable)
    reference_env = os.environ.copy()
    reference_env.pop("LD_LIBRARY_PATH", None)
    process = subprocess.run([python, str(ROOT / "reference.py"), "--model-dir", str(model),
                              "--output", str(output), "--cases", str(cases_path)],
                             check=True, capture_output=True, text=True,
                             env=reference_env)
    record_evidence("reference_process", {"argv": process.args, "stdout": process.stdout, "stderr": process.stderr})
    reference = json.loads((output / "reference.json").read_text())
    return binary, runtime, bundle, reference


def test_official_checkpoint_e2e(case, execution):
    binary, runtime, bundle, reference = execution
    expected = reference[case["name"]]
    argv = [str(binary), "run", str(bundle), "--runtime-root", str(runtime),
            "--prompt", case["prompt"], "--image", expected["image"],
            "--max-new-tokens", str(case["max_new_tokens"]), "--temperature", "0"]
    process = subprocess.run(argv, check=True, capture_output=True, text=True, timeout=180)
    record_evidence("native_process", {"argv": argv, "stdout": process.stdout, "stderr": process.stderr})
    native = json.loads(process.stdout.strip().splitlines()[-1])
    word = case["expected_word"]
    assert word in re.findall(r"[a-z]+", expected["text"].lower()), expected
    assert word in re.findall(r"[a-z]+", native["text"].lower()), native
    assert native["token_ids"] == expected["token_ids"], {"native": native, "reference": expected}
