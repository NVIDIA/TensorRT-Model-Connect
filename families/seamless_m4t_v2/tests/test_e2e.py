# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned SeamlessM4T v2 native multilingual text translation parity.

The FP32 bundle must reproduce the Hugging Face FP32 greedy tokens exactly. An
FP8 bundle (``quantization`` in the manifest) must start with the reference's
first ``min_equal_prefix_tokens`` tokens, end with the end-of-sequence token
and contain the expected word of each input.
"""

from __future__ import annotations

import json
import os
import subprocess
from functools import cache
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_enabled, evidence_stage, record_evidence

FAMILY = "seamless_m4t_v2"
TASKS = frozenset({"text_translation"})
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
    """Build source-only helpers in the configured native test tree."""
    build_dir = _required_path(os.environ.get("TRTMC_NATIVE_BUILD_DIR"), "TRTMC_NATIVE_BUILD_DIR")
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--parallel", "2", "--target", name],
        check=True,
        timeout=600,
    )
    return _required_path(str(build_dir / name), name)


def _record_comparison(tmp_path, label, native, reference, checks):
    """Retain the output pair and the result of already enforced comparisons."""
    if not evidence_enabled():
        return
    import numpy as np

    paths = []
    for role, value in (("native", native), ("reference", reference)):
        if isinstance(value, np.ndarray):
            path = tmp_path / f"{label}-{role}.npz"
            np.savez_compressed(path, output=value)
        else:
            path = tmp_path / f"{label}-{role}.json"
            path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
        paths.append(path)
    record_evidence(
        "reference_comparison",
        {
            "label": label,
            "scope": "independent_reference",
            "enforced": True,
            "native": paths[0],
            "reference": paths[1],
            "checks": [dict(check, scope="independent_reference") for check in checks],
        },
    )


def _model_dir(manifest: dict) -> Path:
    explicit = os.environ.get(f"TRTMC_{FAMILY.upper()}_MODEL_DIR")
    if explicit:
        return _required_path(explicit, f"TRTMC_{FAMILY.upper()}_MODEL_DIR")
    from huggingface_hub import snapshot_download

    try:
        snapshot = snapshot_download(
            repo_id=manifest["hf_id"], revision=manifest.get("hf_revision"), local_files_only=True
        )
    except Exception as error:
        raise AssertionError(
            f"selected {FAMILY} E2E requires the exact cached checkpoint {manifest['hf_id']}"
        ) from error
    return Path(snapshot)


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
    with evidence_stage("runtime"):
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
        record_evidence(
            "native_process",
            {
                "argv": completed.args,
                "stdout": (completed.stdout or "")[-4096:],
                "stderr": (completed.stderr or "")[-16384:],
                "returncode": completed.returncode,
            },
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


def _equal_prefix(token_ids, expected_ids) -> int:
    count = 0
    for left, right in zip(token_ids, expected_ids):
        if left != right:
            break
        count += 1
    return count


def test_official_checkpoint_e2e(case_name, tmp_path):
    _, manifest, case = CASES[case_name]
    record_evidence("inputs", {"manifest": manifest, "case": case})
    record_evidence(
        "checkpoint",
        {
            "repo_id": manifest["hf_id"],
            "revision": manifest["hf_revision"],
            "precision": manifest["precision"],
        },
    )
    record_evidence(
        "thresholds",
        {
            key: value
            for key, value in case.items()
            if any(
                word in key
                for word in ("error", "cosine", "atol", "rtol", "snr", "rmse", "max_abs")
            )
        },
    )
    for asset_key in ("image", "test_image", "audio"):
        if asset_key in case:
            for asset_path in (TEST_ROOT / case[asset_key], TEST_ROOT / "data" / case[asset_key]):
                if asset_path.is_file():
                    record_evidence("inputs", {asset_key: asset_path})
                    break
    import torch
    from transformers import AutoTokenizer, SeamlessM4Tv2ForTextToText

    _, manifest, case = CASES[case_name]
    with evidence_stage("checkpoint"):
        directory = _model_dir(manifest)
    binary, runtime_root = _runtime()
    with evidence_stage("reference"):
        tokenizer = AutoTokenizer.from_pretrained(directory)
    tokenizer_json = tmp_path / "tokenizer.json"
    tokenizer_json.write_text(tokenizer.backend_tokenizer.to_str())
    (tmp_path / "config.json").write_bytes((directory / "config.json").read_bytes())
    for text in [*[item["text"] for item in case["inputs"]], "Café e\u0301 １２３ 中文\tspaces  "]:
        completed = subprocess.run(
            [str(_native_test_binary("test_seamless_m4t_v2_tokenizer")), str(tokenizer_json), text],
            capture_output=True,
            text=True,
            check=False,
        )
        with evidence_stage("comparison"):
            assert completed.returncode == 0, completed.stderr
        with evidence_stage("comparison"):
            assert json.loads(completed.stdout) == tokenizer.encode(text, add_special_tokens=False)
    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
        _build(directory, bundle, manifest)
    record_evidence("bundle", {"path": str(bundle), "manifest": manifest})
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with evidence_stage("reference"):
        reference = (
            SeamlessM4Tv2ForTextToText.from_pretrained(
                directory, dtype=torch.float32, attn_implementation="eager"
            )
            .eval()
            .to("cuda")
        )
    for item in case["inputs"]:
        tokenizer.src_lang = item["source"]
        inputs = tokenizer(item["text"], return_tensors="pt").to("cuda")
        source_id = tokenizer.convert_tokens_to_ids(f"__{item['source']}__")
        target_id = reference.generation_config.text_decoder_lang_to_code_id[item["target"]]
        with evidence_stage("comparison"):
            assert inputs.input_ids[0, 0].item() == source_id
        with evidence_stage("reference"), torch.no_grad():
            generated = reference.generate(
                **inputs,
                tgt_lang=item["target"],
                do_sample=False,
                max_new_tokens=case["max_new_tokens"],
            )
        with evidence_stage("comparison"):
            assert generated[0, :2].tolist() == [reference.config.decoder_start_token_id, target_id]
        expected_ids = generated[0, 2:].tolist()
        expected_text = tokenizer.decode(expected_ids, skip_special_tokens=True)
        payload = _run_json(
            binary,
            runtime_root,
            bundle,
            case,
            "run",
            "--task",
            manifest["task"],
            "--prompt",
            item["text"],
            "--source-language",
            item["source"],
            "--target-language",
            item["target"],
            "--max-new-tokens",
            str(case["max_new_tokens"]),
        )
        record_evidence("inputs", item)
        quantized = manifest.get("quantization") not in (None, "none")
        if quantized:
            record_evidence(
                "thresholds",
                {
                    "token_ids": f"first {case['min_equal_prefix_tokens']} tokens equal, "
                    "ends with the end-of-sequence token",
                    "text": "contains the expected word",
                },
            )
        else:
            record_evidence("thresholds", {"token_ids": "exact equality", "text": "exact equality"})
        record_evidence("native", payload)
        record_evidence(
            "reference",
            {"mode": "hf_transformers", "token_ids": expected_ids, "text": expected_text},
        )
        with evidence_stage("comparison"):
            assert expected_text.strip() and expected_ids[-1] == reference.config.eos_token_id
        if quantized:
            word = item["expected_word"]
            prefix = _equal_prefix(payload["token_ids"], expected_ids)
            ends = int(payload["token_ids"][-1:] == [reference.config.eos_token_id])
            _record_comparison(
                tmp_path,
                f"generation-{item['source']}-{item['target']}",
                payload,
                {"token_ids": expected_ids, "text": expected_text},
                [
                    {
                        "name": "equal_prefix_tokens",
                        "actual": prefix,
                        "operator": ">=",
                        "expected": case["min_equal_prefix_tokens"],
                        "passed": prefix >= case["min_equal_prefix_tokens"],
                    },
                    {
                        "name": "ends_with_eos",
                        "actual": ends,
                        "operator": "==",
                        "expected": 1,
                        "passed": ends == 1,
                    },
                    {
                        "name": "expected_word_missing",
                        "actual": int(word not in payload["text"]),
                        "operator": "==",
                        "expected": 0,
                        "passed": word in payload["text"],
                    },
                ],
            )
            # FP8 moves the logits a little: the translation must start with the reference's
            # tokens, finish, and keep the expected word; later tokens may differ after a near tie.
            with evidence_stage("comparison"):
                assert word in expected_text, expected_text
                assert prefix >= case["min_equal_prefix_tokens"], {
                    "native": payload,
                    "reference": expected_ids,
                }
                assert ends == 1, payload
                assert word in payload["text"], payload
            print(
                f"SeamlessM4T FP8 {item['source']}->{item['target']}: prefix={prefix}, "
                f"text={payload['text']!r} (reference {expected_text!r})"
            )
            continue
        with evidence_stage("comparison"):
            assert payload["token_ids"] == expected_ids
        with evidence_stage("comparison"):
            assert payload["text"] == expected_text
        token_mismatch = int(payload["token_ids"] != expected_ids)
        text_mismatch = int(payload["text"] != expected_text)
        _record_comparison(
            tmp_path,
            f"generation-{len(expected_ids)}-{len(expected_text)}",
            payload,
            {"token_ids": expected_ids, "text": expected_text},
            [
                {
                    "name": "token_mismatch",
                    "actual": token_mismatch,
                    "operator": "==",
                    "expected": 0,
                    "passed": token_mismatch == 0,
                },
                {
                    "name": "text_mismatch",
                    "actual": text_mismatch,
                    "operator": "==",
                    "expected": 0,
                    "passed": text_mismatch == 0,
                },
            ],
        )
        print(
            f"SeamlessM4T {item['source']}->{item['target']}: exact tokens={len(expected_ids)}, text={expected_text!r}"
        )
