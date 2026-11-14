# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A real checkpoint -> bundle -> native Task -> original-reference parity gate."""
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import pytest

ROOT = Path(__file__).parent
MANIFESTS = [json.loads(path.read_text()) for path in sorted((ROOT / "manifests").glob("d3-*.json"))]


def selected(config, manifest):
    names = set()
    for flag in ("--e2e-model", "--e2e-testcase"):
        for value in config.getoption(flag, default=[]):
            names.update(value.split(","))
    path = config.getoption("--e2e-models-file", default=None)
    if path:
        names.update(Path(path).read_text().splitlines())
    return bool(names & {"d3", manifest["name"], manifest["name"] + "-text-image"})


@pytest.mark.e2e
@pytest.mark.trt
@pytest.mark.parametrize("manifest", MANIFESTS, ids=lambda m: m["name"])
def test_edge_text_image(request, tmp_path, manifest):
    if not selected(request.config, manifest):
        pytest.skip("select --e2e-model d3 to build and execute the native D3 bundles")
    from huggingface_hub import snapshot_download
    from tensorrt_model_connect.build import BuildRequest, build
    from tensorrt_model_connect.model_support import load_model_metadata, resolve_family

    binary = os.environ["TRTMC_BINARY"]
    runtime = os.environ["TRTMC_RUNTIME_ROOT"]
    reference_python = os.environ["TRTMC_D3_REFERENCE_PYTHON"]
    checkpoint = Path(snapshot_download(manifest["hf_id"], revision=manifest["hf_revision"],
        local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1"))
    family, support = resolve_family(load_model_metadata(checkpoint))
    assert (family, support.default_task, support.default_precision) == ("d3", "structured_decision", "bf16")
    assert manifest["precision"] == "bf16" and manifest["tensor_parallel_size"] == 1
    bundle = tmp_path / manifest["bundle"]
    build(BuildRequest(checkpoint, bundle, family, support.default_task, support.default_precision,
                       max_sequence_length=manifest["max_sequence_length"], max_batch_size=manifest["max_batch_size"],
                       tensor_parallel_size=manifest["tensor_parallel_size"], image_height=256, image_width=256))
    assert bundle.is_file()
    pixels = np.zeros((256, 256, 3), np.uint8)
    pixels[..., 0] = 255
    rgb = tmp_path / "red.rgb"
    pixels.tofile(rgb)
    probe = Path(binary).parent / "d3_task_probe"
    reference_env = dict(os.environ, TRTMC_D3_REFERENCE_MODEL=manifest["name"])
    native_results = json.loads(subprocess.run([str(probe), str(bundle), runtime,
        str(ROOT / "fixtures/record.json"), str(rgb)], check=True, capture_output=True, text=True).stdout)
    (tmp_path / "native-sequence.json").write_text(json.dumps(native_results, indent=2))
    assert len(native_results) == 3
    for index, image in enumerate((False, True, False)):
        command = [binary, "d3", "decide", str(bundle), "--record", str(ROOT / "fixtures/record.json"),
                   "--runtime-root", runtime]
        reference = [reference_python, str(ROOT / "reference.py"), str(checkpoint),
                     str(ROOT / "fixtures/record.json"), str(tmp_path / f"reference-{index}.json")]
        if image:
            command.extend(["--image-rgb", str(rgb)])
            reference.append(str(rgb))
        subprocess.run(reference, check=True, env=reference_env)
        expected = json.loads((tmp_path / f"reference-{index}.json").read_text())
        actual = native_results[index]
        cli = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
        (tmp_path / f"native-cli-{index}.json").write_text(json.dumps(cli, indent=2))
        assert cli == actual
        assert actual["response"]["usage"] == expected["response"]["usage"]
        assert actual["response"]["model"] == manifest["name"]
        assert actual["response"]["answers"].keys() == expected["response"]["answers"].keys()
        for key, reference_answer in expected["response"]["answers"].items():
            native_answer = actual["response"]["answers"][key]
            assert native_answer.keys() == reference_answer.keys()
            for field in ("type", "choice", "legend"):
                if field in reference_answer:
                    assert native_answer[field] == reference_answer[field]
            if "confidence" in reference_answer:
                np.testing.assert_allclose(native_answer["confidence"], reference_answer["confidence"],
                                           atol=.004, rtol=.01)
            if "probabilities" in reference_answer:
                assert native_answer["probabilities"].keys() == reference_answer["probabilities"].keys()
                for option, probability in reference_answer["probabilities"].items():
                    np.testing.assert_allclose(native_answer["probabilities"][option], probability,
                                               atol=.002, rtol=.01)
        assert actual["response"]["answers"]["color"]["choice"] == expected["response"]["answers"]["color"]["choice"]
        for a, e in zip(actual["scores"], expected["scores"], strict=True):
            assert a["question_id"] == e["question_id"]
            assert a["option_ids"] == e["option_ids"]
            np.testing.assert_allclose(a["logits"], e["logits"], atol=.05, rtol=.01)
            np.testing.assert_allclose(a["probabilities"], e["probabilities"], atol=.002, rtol=.01)
            assert np.isfinite(a["probabilities"]).all()
            np.testing.assert_allclose(sum(a["probabilities"]), 1, atol=1e-6)
        for name, key in [("red", "noul"), ("brightness", "score")]:
            np.testing.assert_allclose(actual["response"]["answers"][name][key],
                                       expected["response"]["answers"][name][key], atol=.004, rtol=.01)
    assert native_results[0] == native_results[2]
    batch_probe = Path(binary).parent / "d3_batch_task_probe"
    batches = json.loads(subprocess.run([str(batch_probe), str(bundle), runtime,
        str(ROOT / "fixtures/record.json"), str(rgb)], check=True, capture_output=True, text=True).stdout)
    (tmp_path / "native-batches.json").write_text(json.dumps(batches, indent=2))
    validate_batch_probe(batches)
    samples = list(zip(batches["inputs"], batches["single"], strict=True))
    samples.extend(zip(batches["same_length"]["inputs"], batches["same_length"]["single"], strict=True))
    references = tmp_path / "batch-references.json"
    subprocess.run([reference_python, str(ROOT / "reference_cases.py"), str(checkpoint),
        str(tmp_path / "native-batches.json"), str(references)], check=True, env=reference_env)
    for (_, native_single), expected in zip(samples, json.loads(references.read_text()), strict=True):
        compare_batch_result(native_single, expected)


def compare_batch_result(actual, expected):
    """Use the original decision gates for both padding and regrouping parity."""
    assert actual["response"]["usage"] == expected["response"]["usage"]
    assert actual["response"]["answers"].keys() == expected["response"]["answers"].keys()
    for a, e in zip(actual["scores"], expected["scores"], strict=True):
        assert a["question_id"] == e["question_id"]
        assert a["option_ids"] == e["option_ids"]
        np.testing.assert_allclose(a["logits"], e["logits"], atol=.05, rtol=.01)
        np.testing.assert_allclose(a["probabilities"], e["probabilities"], atol=.002, rtol=.01)
    for key, answer in expected["response"]["answers"].items():
        observed = actual["response"]["answers"][key]
        for field, value in answer.items():
            if isinstance(value, (float, int)):
                np.testing.assert_allclose(observed[field], value, atol=.004, rtol=.01)
            elif field == "probabilities":
                assert observed[field].keys() == value.keys()
                for option in value:
                    np.testing.assert_allclose(observed[field][option], value[option], atol=.002, rtol=.01)
            else:
                assert observed[field] == value


def validate_batch_probe(results):
    assert len(results["single"]) == 9
    for batch in results["batches"]:
        assert len(batch["results"]) == batch["count"]
        for actual, expected in zip(batch["results"], results["single"][:batch["count"]], strict=True):
            compare_batch_result(actual, expected)
    for actual, expected in zip(results["reversed"], reversed(results["single"]), strict=True):
        compare_batch_result(actual, expected)
    assert results["after_invalid"] == results["single"][0]
    pair = results["same_length"]
    assert len(pair["batch"]) == len(pair["single"]) == 2
    assert pair["single"][0]["response"]["usage"] == pair["single"][1]["response"]["usage"]
    for actual, expected in zip(pair["batch"], pair["single"], strict=True):
        compare_batch_result(actual, expected)
