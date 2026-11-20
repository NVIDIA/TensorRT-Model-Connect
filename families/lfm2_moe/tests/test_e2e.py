# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned LFM2 MoE MoE parity through native incremental text continuation."""

from __future__ import annotations

import json
import os
import subprocess
from functools import cache
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_enabled, evidence_stage, record_evidence

FAMILY = "lfm2_moe"
TASKS = frozenset({"text_continuation"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"


def _case_index() -> dict[str, tuple[Path, dict, dict]]:
    result = {}
    for path in sorted(MANIFEST_ROOT.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == FAMILY
        assert manifest["task"] in TASKS
        for case in manifest["testcases"]:
            name = str(case["name"])
            assert name not in result
            result[name] = (path, manifest, case)
    return result


CASES = _case_index()


def _selected_cases(config) -> tuple[list[str], bool]:
    model_filters = set()
    for raw in config.getoption("--e2e-model") or []:
        model_filters.update((item.strip() for item in str(raw).split(",") if item.strip()))
    models_file = config.getoption("--e2e-models-file")
    if models_file:
        model_filters.update(
            (
                line.strip()
                for line in Path(models_file).read_text(encoding="utf-8").splitlines()
                if line.strip() and (not line.lstrip().startswith("#"))
            )
        )
    testcase_filters = set()
    for raw in config.getoption("--e2e-testcase") or []:
        testcase_filters.update((item.strip() for item in str(raw).split(",") if item.strip()))
    if not model_filters and (not testcase_filters):
        return (sorted(CASES), False)
    selected = []
    for name, (_, manifest, _) in CASES.items():
        model_match = (
            not model_filters
            or FAMILY in model_filters
            or name in model_filters
            or (manifest["name"] in model_filters)
        )
        testcase_match = not testcase_filters or name in testcase_filters
        if model_match and testcase_match:
            selected.append(name)
    return (sorted(selected), True)


def pytest_generate_tests(metafunc) -> None:
    if "case_name" in metafunc.fixturenames:
        names, enabled = _selected_cases(metafunc.config)
        parameters = names
        if not enabled:
            parameters = [
                pytest.param(
                    name,
                    marks=pytest.mark.skip(
                        reason="direct E2E requires one of the three explicit E2E selectors"
                    ),
                )
                for name in names
            ]
        metafunc.parametrize("case_name", parameters, ids=names)


def _required_path(value: str | None, label: str) -> Path:
    assert value, f"selected {FAMILY} E2E requires {label}"
    path = Path(value)
    assert path.exists(), f"selected {FAMILY} E2E {label} does not exist: {path}"
    return path


@cache
def _native_test_binary(name: str) -> Path:
    build_dir = _required_path(os.environ.get("TRTMC_NATIVE_BUILD_DIR"), "TRTMC_NATIVE_BUILD_DIR")
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--parallel", "2", "--target", name],
        check=True,
        timeout=600,
    )
    return _required_path(str(build_dir / name), name)


def _model_dir(manifest: dict) -> tuple[Path, str | None]:
    explicit = os.environ.get(f"TRTMC_{FAMILY.upper()}_MODEL_DIR")
    if explicit:
        return _required_path(explicit, f"TRTMC_{FAMILY.upper()}_MODEL_DIR"), None
    from huggingface_hub import snapshot_download

    try:
        snapshot = snapshot_download(
            repo_id=manifest["hf_id"], revision=manifest.get("hf_revision"), local_files_only=True
        )
    except Exception as error:
        raise AssertionError(
            f"selected {FAMILY} E2E requires the exact cached checkpoint {manifest['hf_id']}"
        ) from error
    return Path(snapshot), manifest["hf_revision"]


def _runtime() -> tuple[Path, Path]:
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / f"libtrtmc_model_{FAMILY}.so").is_file()
    import torch

    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    assert torch.cuda.device_count() >= 1, f"selected {FAMILY} E2E requires one GPU"
    return (binary, runtime_root)


def _build(model_dir: Path, bundle: Path, manifest: dict) -> None:
    build(
        BuildRequest(
            model_dir=model_dir,
            output_path=bundle,
            family=FAMILY,
            task=manifest["task"],
            precision=manifest["precision"],
            max_sequence_length=manifest.get("max_sequence_length"),
            image_height=manifest.get("image_height"),
            image_width=manifest.get("image_width"),
            video_num_frames=manifest.get("video_num_frames"),
            max_batch_size=int(manifest.get("max_batch_size", 1)),
            tensor_parallel_size=int(manifest["tensor_parallel_size"]),
            quantization=manifest.get("quantization"),
            fp32_layers=tuple((int(layer) for layer in manifest.get("fp32_layers", ()))),
        )
    )


def _run_json(
    binary: Path,
    runtime_root: Path,
    bundle: Path,
    case: dict,
    command: str,
    *arguments: str,
) -> dict:
    invocation = [
        str(binary),
        command,
        str(bundle),
        "--runtime-root",
        str(runtime_root),
        *arguments,
    ]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        (value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value)
    )
    completed = subprocess.run(
        invocation,
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=int(case.get("runtime_timeout_s", 3600)),
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    payloads = []
    for line in completed.stdout.splitlines():
        start = line.find("{")
        if start >= 0:
            try:
                payloads.append(json.loads(line[start:]))
            except json.JSONDecodeError:
                pass
    assert payloads, f"native {command} returned no JSON: {completed.stdout[-1000:]}"
    assert all((payload == payloads[0] for payload in payloads))
    return payloads[0]


def test_official_checkpoint_e2e(case_name, tmp_path, request):
    _, manifest, case = CASES[case_name]
    record_evidence("inputs", {"manifest": manifest, "case": case})
    if evidence_enabled():
        for key in ("image", "test_image", "audio"):
            if key in case:
                for source in (TEST_ROOT / case[key], TEST_ROOT / "data" / case[key]):
                    if source.is_file():
                        record_evidence("inputs", {key: source})
                        break
    record_evidence(
        "checkpoint",
        {
            "hf_id": manifest["hf_id"],
            "expected_hf_revision": manifest["hf_revision"],
            "hf_revision": None,
            "revision_status": "unresolved",
            "precision": manifest["precision"],
        },
    )
    record_evidence(
        "thresholds",
        {
            key: value
            for key, value in case.items()
            if any(word in key for word in ("error", "cosine", "atol", "rtol", "snr", "rmse"))
        },
    )
    import gc
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _, manifest, case = CASES[case_name]
    with evidence_stage("checkpoint"):
        directory, checkpoint_revision = _model_dir(manifest)
    record_evidence(
        "checkpoint",
        {
            "model_dir": str(directory),
            "hf_revision": checkpoint_revision,
            "revision_status": "unverified"
            if checkpoint_revision is None
            else "pinned_cache_lookup",
        },
    )
    binary, runtime_root = _runtime()
    tokenizer = AutoTokenizer.from_pretrained(directory)
    tokenizer_json = tmp_path / "tokenizer.json"
    tokenizer_json.write_text(tokenizer.backend_tokenizer.to_str())
    for prompt in [*case["prompts"], "A small test. 中文 café １２３."]:
        completed = subprocess.run(
            [str(_native_test_binary("test_lfm2_moe_tokenizer")), str(tokenizer_json), prompt],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        assert completed.returncode == 0, completed.stderr
        assert json.loads(completed.stdout) == tokenizer.encode(prompt)
    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
        _build(directory, bundle, manifest)
    record_evidence("bundle", {"path": str(bundle), "manifest": manifest})
    matmul_precision = torch.get_float32_matmul_precision()
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    request.addfinalizer(lambda: torch.set_float32_matmul_precision(matmul_precision))
    request.addfinalizer(lambda: setattr(torch.backends.cudnn, "allow_tf32", cudnn_tf32))
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    reference = (
        AutoModelForCausalLM.from_pretrained(
            directory, dtype=torch.float32, attn_implementation="eager"
        )
        .eval()
        .to("cuda")
    )
    record_evidence("thresholds", {"token_ids": "exact equality", "text": "exact equality"})
    expected = []
    for prompt in case["prompts"]:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            generated = reference.generate(
                **inputs, do_sample=False, max_new_tokens=case["max_new_tokens"]
            )
        ids = generated[0, inputs["input_ids"].shape[1] :].tolist()
        expected.append((ids, tokenizer.decode(ids, skip_special_tokens=True)))
    # Retain only the oracle outputs before loading the second full FP32 model.
    del reference, inputs, generated
    gc.collect()
    torch.cuda.empty_cache()
    for sample_index, (prompt, (expected_ids, expected_text)) in enumerate(
        zip(case["prompts"], expected)
    ):
        payload = _run_json(
            binary,
            runtime_root,
            bundle,
            case,
            "run",
            "--task",
            manifest["task"],
            "--prompt",
            prompt,
            "--max-new-tokens",
            str(case["max_new_tokens"]),
        )
        record_evidence("inputs", {"prompt": prompt})
        record_evidence("native", {"prompt": prompt, **payload})
        record_evidence(
            "reference",
            {
                "mode": "hf_transformers",
                "precision": manifest["precision"],
                "prompt": prompt,
                "token_ids": expected_ids,
                "text": expected_text,
            },
        )
        if evidence_enabled():
            # Persist the actual enforced equality checks so report readers do
            # not have to infer the native/reference roles from local names.
            pair = {}
            for role, output in (
                ("native", payload),
                ("reference", {"token_ids": expected_ids, "text": expected_text}),
            ):
                path = tmp_path / f"{role}-comparison-{sample_index}.json"
                path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
                pair[role] = path
            mismatches = abs(len(payload["token_ids"]) - len(expected_ids)) + sum(
                left != right for left, right in zip(payload["token_ids"], expected_ids)
            )
            text_mismatch = int(payload["text"] != expected_text)
            record_evidence(
                "reference_comparison",
                {
                    "label": f"Generated tokens and text for sample {sample_index}",
                    "scope": "independent_reference",
                    "enforced": True,
                    **pair,
                    "checks": [
                        {
                            "name": name,
                            "scope": "independent_reference",
                            "actual": value,
                            "operator": "==",
                            "expected": 0,
                            "passed": value == 0,
                        }
                        for name, value in (
                            ("token_mismatches", mismatches),
                            ("text_mismatch", text_mismatch),
                        )
                    ],
                },
            )

        assert payload["token_ids"] == expected_ids
        assert payload["text"] == expected_text
        assert expected_text.strip()

        print(f"LFM2 MoE generated {len(expected_ids)} exact tokens: {expected_text!r}")
