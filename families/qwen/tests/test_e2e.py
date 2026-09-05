# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned checkpoint-to-native-runtime proof."""

from __future__ import annotations

import gc
import json
import os
import re
import shutil
import subprocess
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build


_TEST_DIR = Path(__file__).resolve().parent
_FAMILY = _TEST_DIR.parent.name
_MPI_RANK_ZERO = re.compile(r"^\[[^,]+,0\]<stdout>:(.*)$")
_LOGIT_ORACLES = frozenset({"qwen3-0.6b-fp8", "qwen3-0.6b-fp8-tp4"})
_NATIVE_KV_FIELDS = frozenset(
    {"expected_kv_cache_rows", "expected_prefill_chunks", "expected_prefill_chunk_limit"}
)


def _load_cases() -> dict[str, tuple[dict, dict]]:
    cases: dict[str, tuple[dict, dict]] = {}
    for path in sorted((_TEST_DIR / "manifests").glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == _FAMILY, path
        assert manifest["task"] == "text_generation", path
        assert isinstance(manifest["precision"], str), path
        assert isinstance(manifest["max_sequence_length"], int), path
        assert isinstance(manifest["tensor_parallel_size"], int), path
        for case in manifest["testcases"]:
            name = case["name"]
            assert name not in cases, name
            cases[name] = (manifest, case)
    assert cases, f"{_FAMILY} has no E2E cases"
    return cases


_CASES = _load_cases()


def _csv_values(values: list[str]) -> set[str]:
    return {item.strip() for value in values for item in str(value).split(",") if item.strip()}


def _selection(config) -> set[str]:
    selected = _csv_values(config.getoption("--e2e-model", default=[]) or [])
    selected |= _csv_values(config.getoption("--e2e-testcase", default=[]) or [])
    models_file = config.getoption("--e2e-models-file", default=None)
    if models_file:
        path = Path(models_file)
        assert path.is_file(), f"E2E models file does not exist: {path}"
        selected |= {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    return selected


def _require_selected(case_name: str, manifest: dict, config) -> None:
    selected = _selection(config)
    enabled = os.environ.get("TRTMC_E2E") == "1"
    if not enabled and not selected:
        pytest.skip("real family E2E requires TRTMC_E2E=1 or an explicit E2E selection")
    if selected and not ({_FAMILY, manifest["name"], case_name} & selected):
        pytest.skip(f"{case_name} was not selected")


def _required_environment(tp_size: int):
    binary_value = os.environ.get("TRTMC_BINARY")
    runtime_value = os.environ.get("TRTMC_RUNTIME_ROOT")
    assert binary_value, "selected E2E requires TRTMC_BINARY"
    assert runtime_value, "selected E2E requires TRTMC_RUNTIME_ROOT"

    binary = Path(binary_value)
    runtime_root = Path(runtime_value)
    assert binary.is_file() and os.access(binary, os.X_OK), binary
    assert runtime_root.is_dir(), runtime_root
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file(), runtime_root
    assert (runtime_root / f"libtrtmc_model_{_FAMILY}.so").is_file(), runtime_root

    import torch

    assert torch.cuda.is_available(), "selected E2E requires CUDA"
    assert torch.cuda.device_count() >= tp_size, (
        f"{_FAMILY} TP{tp_size} requires {tp_size} visible GPUs; found {torch.cuda.device_count()}"
    )
    if tp_size > 1:
        assert shutil.which("mpirun"), "selected TP E2E requires mpirun"
    return binary, runtime_root, torch


def _checkpoint(manifest: dict) -> Path:
    from huggingface_hub import snapshot_download

    path = Path(
        snapshot_download(
            repo_id=manifest["hf_id"],
            revision=manifest.get("hf_revision"),
        )
    )
    assert (path / "config.json").is_file(), path
    return path


def _prompt(case: dict) -> str:
    if "prompt" in case:
        prompt = case["prompt"]
        assert isinstance(prompt, str) and prompt, case["name"]
        return prompt
    repeated = case["prompt_repeat"]
    count = int(repeated["count"])
    assert count > 0
    return str(repeated["separator"]).join([str(repeated["text"])] * count) + str(
        repeated.get("suffix", "")
    )


def _thresholds(case_name: str) -> dict[str, float]:
    path = _TEST_DIR / "thresholds" / f"{case_name}.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    thresholds = payload["threshold_overrides"]
    assert isinstance(thresholds, dict) and thresholds, path
    return thresholds


def _build_bundle(manifest: dict, model_dir: Path, bundle: Path) -> None:
    quantization = manifest.get("quantization")
    assert quantization is None or isinstance(quantization, str)
    fp32_layers = tuple(manifest.get("fp32_layers", ()))
    build(
        BuildRequest(
            model_dir=model_dir,
            output_path=bundle,
            family=_FAMILY,
            task="text_generation",
            precision=manifest["precision"],
            max_sequence_length=manifest["max_sequence_length"],
            tensor_parallel_size=manifest["tensor_parallel_size"],
            quantization=quantization,
            fp32_layers=fp32_layers,
        )
    )
    assert bundle.is_file() and bundle.stat().st_size > 0, bundle


def _assert_rank_sections(binary: Path, bundle: Path, tp_size: int) -> None:
    inspected = subprocess.run(
        [str(binary), "inspect", str(bundle)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    payload = json.loads(inspected.stdout)
    assert payload["family"] == _FAMILY
    assert payload["task"] == "text_generation"
    if tp_size > 1:
        rank_sections = {
            name
            for name in payload["sections"]
            if name.startswith("engine.rank") and name.endswith(".plan")
        }
        assert rank_sections == {f"engine.rank{rank}.plan" for rank in range(tp_size)}


def _native_arguments(bundle: Path, runtime_root: Path, prompt: str, case: dict) -> list[str]:
    arguments = [
        "run",
        str(bundle),
        "--runtime-root",
        str(runtime_root),
        "--prompt",
        prompt,
        "--max-new-tokens",
        str(case["max_new_tokens"]),
    ]
    options = {
        "temperature": "--temperature",
        "top_k": "--top-k",
        "top_p": "--top-p",
        "min_p": "--min-p",
        "seed": "--seed",
        "repetition_penalty": "--repetition-penalty",
        "use_chat_template": "--use-chat-template",
        "enable_thinking": "--enable-thinking",
    }
    for field, option in options.items():
        if field not in case:
            continue
        value = case[field]
        if isinstance(value, bool):
            value = "true" if value else "false"
        arguments.extend([option, str(value)])
    return arguments


def _run_native(
    binary: Path,
    runtime_root: Path,
    bundle: Path,
    prompt: str,
    case: dict,
    tp_size: int,
    tmp_path: Path,
) -> dict:
    command = [str(binary), *_native_arguments(bundle, runtime_root, prompt, case)]
    environment = dict(os.environ)
    if tp_size > 1:
        environment["TRTMC_NCCL_RENDEZVOUS"] = str(tmp_path / "nccl.rendezvous")
        prefix = ["mpirun", "--tag-output", "-np", str(tp_size)]
        for name in ("LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "TRTMC_NCCL_RENDEZVOUS"):
            if name in environment:
                prefix.extend(["-x", name])
        command = [*prefix, *command]

    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
        env=environment,
    )
    if tp_size == 1:
        payload = json.loads(completed.stdout)
        payload["runtime_stderr"] = completed.stderr
        payload["runtime_command"] = command
        return payload

    rank_zero_payloads = []
    for line in completed.stdout.splitlines():
        match = _MPI_RANK_ZERO.fullmatch(line)
        if match and match.group(1).lstrip().startswith("{"):
            rank_zero_payloads.append(match.group(1))
    assert len(rank_zero_payloads) == 1, completed.stdout
    payload = json.loads(rank_zero_payloads[0])
    payload["runtime_stderr"] = completed.stderr
    payload["runtime_command"] = command
    return payload


def _raw_prompt_token_count(model_dir: Path, manifest: dict, prompt: str) -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=bool(manifest.get("trust_remote_code", False)),
    )
    return len(tokenizer.encode(prompt, add_special_tokens=False))


@cache
def _logits_trace_binary() -> tuple[Path, Path]:
    value = os.environ.get("TRTMC_NATIVE_BUILD_DIR")
    assert value, "selected Qwen logits trace requires TRTMC_NATIVE_BUILD_DIR"
    build_dir = Path(value)
    assert build_dir.is_dir(), "selected Qwen logits trace requires TRTMC_NATIVE_BUILD_DIR"
    subprocess.run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--parallel",
            "8",
            "--target",
            "qwen_logits_trace",
            "trtmc_backend_trt",
        ],
        check=True,
        timeout=600,
    )
    binary = build_dir / "families" / _FAMILY / "qwen_logits_trace"
    assert binary.is_file()
    assert (build_dir / "libtrtmc_backend_trt.so").is_file()
    return binary, build_dir


def _native_logits_trace(
    bundle: Path, prompt: str, case: dict, tp_size: int, tmp_path: Path
) -> np.ndarray:
    binary, runtime_root = _logits_trace_binary()
    prefix = tmp_path / "qwen-logits"
    command = [
        str(binary),
        str(bundle),
        str(runtime_root),
        str(prefix),
        prompt,
        str(int(case["max_new_tokens"])),
    ]
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(runtime_root), environment.get("LD_LIBRARY_PATH", "")) if value
    )
    if tp_size > 1:
        environment["TRTMC_NCCL_RENDEZVOUS"] = str(tmp_path / "trace.nccl.rendezvous")
        command = ["mpirun", "--tag-output", "-np", str(tp_size), *command]
        for name in ("LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "TRTMC_NCCL_RENDEZVOUS"):
            if name in environment:
                command[4:4] = ["-x", name]
    subprocess.run(
        command, check=True, capture_output=True, text=True, env=environment, timeout=600
    )
    shape_path = Path(f"{prefix}.rank0.shape")
    data_path = Path(f"{prefix}.rank0.f32")
    rows, columns = (int(value) for value in shape_path.read_text(encoding="utf-8").split())
    values = np.fromfile(data_path, dtype="<f4")
    assert values.size == rows * columns
    return values.reshape(rows, columns)


def _render_prompt(tokenizer, prompt: str, case: dict):
    if not case.get("use_chat_template", False):
        return tokenizer(prompt, return_tensors="pt")
    options = {"tokenize": False, "add_generation_prompt": True}
    if "enable_thinking" in case:
        options["enable_thinking"] = case["enable_thinking"]
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        **options,
    )
    return tokenizer(rendered, return_tensors="pt", add_special_tokens=False)


def _is_sampling(case: dict) -> bool:
    return bool(
        case.get("do_sample", False)
        or float(case.get("temperature", 1.0)) not in {0.0, 1.0}
        or int(case.get("top_k", 1)) > 1
        or float(case.get("top_p", 1.0)) < 1.0
        or float(case.get("min_p", 0.0)) > 0.0
    )


def _hf_reference(
    model_dir: Path,
    manifest: dict,
    case: dict,
    prompt: str,
    actual_ids: list[int],
    torch,
) -> tuple[list[int], str, float | None, str, np.ndarray | None, int]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    trust_remote_code = bool(manifest.get("trust_remote_code", False))
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )
    inputs = _render_prompt(tokenizer, prompt, case)
    prompt_ids = inputs["input_ids"][0].tolist()
    if "expected_prompt_token_ids" in case:
        assert prompt_ids == case["expected_prompt_token_ids"]
    actual_decoded = tokenizer.decode(actual_ids, skip_special_tokens=True).strip()
    if _is_sampling(case):
        return [], "", 1.0, actual_decoded, None, len(prompt_ids)

    reference_precision = case.get(
        "reference_precision",
        manifest.get("reference_precision", manifest["precision"]),
    )
    dtypes = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    assert reference_precision in dtypes, reference_precision
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
            dtype=dtypes[reference_precision],
        )
        .eval()
        .to("cuda")
    )
    inputs = inputs.to(model.device)

    reference_logits = None
    with torch.inference_mode():
        generate_options = {
            "max_new_tokens": int(case["max_new_tokens"]),
            "do_sample": False,
        }
        if "repetition_penalty" in case:
            generate_options["repetition_penalty"] = float(case["repetition_penalty"])
        generated = model.generate(**inputs, **generate_options)
        reference_ids = generated[0, inputs["input_ids"].shape[1] :].tolist()
        reference_text = tokenizer.decode(reference_ids, skip_special_tokens=True).strip()
        if case["name"] in _LOGIT_ORACLES:
            full_ids = generated
            full_attention = torch.ones_like(full_ids)
            reference_logits = (
                model(input_ids=full_ids, attention_mask=full_attention)
                .logits[0]
                .float()
                .cpu()
                .numpy()
            )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (
        reference_ids,
        reference_text,
        None,
        actual_decoded,
        reference_logits,
        len(prompt_ids),
    )


def _assert_logits_oracle(
    case_name: str, actual: np.ndarray, reference: np.ndarray, thresholds: dict
) -> float:
    assert case_name in _LOGIT_ORACLES
    actual = np.asarray(actual)
    reference = np.asarray(reference)
    assert actual.ndim == reference.ndim == 2
    rows = min(actual.shape[0], reference.shape[0])
    columns = min(actual.shape[1], reference.shape[1])
    assert rows > 0 and columns > 0
    actual = np.nan_to_num(
        actual[:rows, :columns].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    reference = np.nan_to_num(
        reference[:rows, :columns].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    actual_norm = np.linalg.norm(actual, axis=1)
    reference_norm = np.linalg.norm(reference, axis=1)
    denominator = actual_norm * reference_norm
    cosines = np.divide(
        np.sum(actual * reference, axis=1),
        denominator,
        out=np.zeros(rows, dtype=np.float64),
        where=(actual_norm >= 1e-12) & (reference_norm >= 1e-12),
    )
    difference_norm = np.linalg.norm(actual - reference, axis=1)
    relative_l2 = difference_norm / np.maximum(reference_norm, 1e-12)
    assert float(np.percentile(cosines, 5)) >= float(thresholds["logit_cosine_p5"]) or float(
        np.percentile(relative_l2, 95)
    ) <= float(thresholds["logit_rel_l2_p95"])
    partitioned = np.partition(reference, -2, axis=1)
    stable = partitioned[:, -1] - partitioned[:, -2] >= float(thresholds["stable_margin"])
    actual_top = np.argmax(actual, axis=1)
    reference_top = np.argmax(reference, axis=1)
    agreement = float(np.mean(actual_top == reference_top))
    stable_rate = (
        float(np.mean(actual_top[stable] == reference_top[stable])) if stable.any() else 1.0
    )
    unstable = ~stable
    if unstable.any():
        reference_topk = np.argsort(reference, axis=1)[:, -5:]
        unstable_rate = float(
            np.mean(
                [actual_top[index] in reference_topk[index] for index in np.flatnonzero(unstable)]
            )
        )
    else:
        unstable_rate = 1.0
    assert agreement >= float(thresholds["token_agreement_rate"]) or (
        stable_rate >= float(thresholds["stable_top1_match_rate"])
        and unstable_rate >= float(thresholds["unstable_topk_hit_rate"])
    )
    return agreement


def test_fp8_logits_oracle_keeps_the_old_stable_top1_gate() -> None:
    reference = np.asarray([[3.0, 1.0, 0.0], [0.0, 1.0, 3.0]], dtype=np.float32)
    thresholds = {
        "logit_cosine_p5": 0.2,
        "logit_rel_l2_p95": 1.5,
        "stable_margin": 0.1,
        "stable_top1_match_rate": 0.9,
        "unstable_topk_hit_rate": 0.8,
        "token_agreement_rate": 0.8,
    }
    _assert_logits_oracle("qwen3-0.6b-fp8", reference, reference, thresholds)
    with pytest.raises(AssertionError):
        _assert_logits_oracle("qwen3-0.6b-fp8", reference[:, ::-1], reference, thresholds)
    nonfinite = np.asarray([[np.nan, np.inf, -np.inf]], dtype=np.float32)
    _assert_logits_oracle(
        "qwen3-0.6b-fp8",
        np.pad(nonfinite, ((0, 0), (0, 1)), constant_values=123.0),
        nonfinite,
        thresholds,
    )


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().split()).strip()


def _normalized_edit_distance(left: str, right: str) -> float:
    left = _normalize_text(left)
    right = _normalize_text(right)
    if not left and not right:
        return 0.0
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1] / max(len(left), len(right))


def _contains_expected_answer(text: str, answer: str) -> bool:
    normalized_text = _normalize_text(text)
    normalized_answer = _normalize_text(answer).strip(" \"'`.,;:!?()[]{}")
    if not normalized_text or not normalized_answer:
        return False
    if normalized_answer.isalnum():
        return re.search(
            rf"(?<![a-z0-9]){re.escape(normalized_answer)}(?![a-z0-9])",
            normalized_text,
        ) is not None
    return normalized_answer in normalized_text


def _text_threshold(thresholds: dict[str, float]) -> float:
    return float(
        thresholds.get(
            "contract_ned_threshold", thresholds.get("normalized_text_edit_distance", 0.15)
        )
    )


def _assert_sampling_contract(payload: dict, case: dict) -> None:
    token_ids = payload["token_ids"]
    assert isinstance(token_ids, list) and token_ids
    assert all(isinstance(token_id, int) for token_id in token_ids)
    assert len(token_ids) <= int(case["max_new_tokens"])
    assert str(payload["text"]).strip()
    command = [str(value) for value in payload["runtime_command"]]
    required_flags = []
    if float(case.get("top_p", 1.0)) < 1.0:
        required_flags.append("--top-p")
    if float(case.get("temperature", 1.0)) != 1.0:
        required_flags.append("--temperature")
    if int(case.get("top_k", 1)) != 1:
        required_flags.append("--top-k")
    if int(case.get("seed", -1)) >= 0:
        required_flags.append("--seed")
    assert all(flag in command for flag in required_flags)


def test_sampling_contract_requires_native_flags_and_output() -> None:
    case = {"max_new_tokens": 4, "temperature": 0.7, "top_p": 0.9, "top_k": 50, "seed": 42}
    payload = {
        "token_ids": [1],
        "text": "sample",
        "runtime_command": [
            "trtmc",
            "--temperature",
            "0.7",
            "--top-p",
            "0.9",
            "--top-k",
            "50",
            "--seed",
            "42",
        ],
    }
    _assert_sampling_contract(payload, case)
    with pytest.raises(AssertionError):
        _assert_sampling_contract({**payload, "runtime_command": ["trtmc"]}, case)


def _assert_correctness(
    payload: dict,
    case: dict,
    thresholds: dict[str, float],
    reference_ids: list[int],
    reference_text: str,
    sampling_support: float | None,
    actual_decoded: str,
    reference_logits: np.ndarray | None,
) -> None:
    actual_ids = payload["token_ids"]
    assert isinstance(actual_ids, list) and actual_ids
    assert all(isinstance(token_id, int) for token_id in actual_ids)
    assert len(actual_ids) <= int(case["max_new_tokens"]), (
        "Task API token_ids must contain generated tokens only"
    )
    actual_text = str(payload["text"]).strip()
    if case.get("enable_thinking") is False:
        assert "<think>" not in actual_text.casefold()
    if "expected_continuation_token_ids" in case:
        assert actual_ids == case["expected_continuation_token_ids"]
    if "expected_continuation_text" in case:
        assert case["expected_continuation_text"].casefold() in actual_decoded.casefold()
    expected_answers = case.get("expected_answers", ())
    oracle_agreement = None
    if case["name"] in _LOGIT_ORACLES:
        assert reference_logits is not None
        oracle_agreement = _assert_logits_oracle(
            case["name"], payload["logits_trace"], reference_logits, thresholds
        )

    del sampling_support
    assert actual_text
    normalized_actual = _normalize_text(actual_text)
    normalized_reference = _normalize_text(reference_text)
    ned = _normalized_edit_distance(normalized_actual, normalized_reference)
    if oracle_agreement is not None and oracle_agreement >= float(
        thresholds["token_agreement_rate"]
    ):
        short, long = sorted((normalized_actual, normalized_reference), key=len)
        if len(short) >= 24 and long.startswith(short):
            ned = min(ned, _normalized_edit_distance(short, long[: len(short)]))
    expected_answer_matches = bool(expected_answers) and any(
        _contains_expected_answer(normalized_actual, answer)
        and _contains_expected_answer(normalized_reference, answer)
        for answer in expected_answers
    )
    assert ned <= _text_threshold(thresholds) or expected_answer_matches


def test_fp8_text_gate_uses_prefix_fallback_and_expected_answer_or() -> None:
    thresholds = {
        "logit_cosine_p5": 0.2,
        "logit_rel_l2_p95": 1.5,
        "normalized_text_edit_distance": 0.0,
        "stable_margin": 0.1,
        "stable_top1_match_rate": 0.9,
        "token_agreement_rate": 0.8,
        "unstable_topk_hit_rate": 0.8,
    }
    logits = np.asarray([[3.0, 1.0, 0.0]], dtype=np.float32)
    prefix = "this is a sufficiently long generated prefix"
    payload = {"token_ids": [1], "text": prefix, "logits_trace": logits}
    case = {"name": "qwen3-0.6b-fp8", "max_new_tokens": 2}
    _assert_correctness(payload, case, thresholds, [], prefix + " suffix", None, "", logits)

    answer_case = {**case, "expected_answers": ["C"]}
    _assert_correctness(
        {**payload, "text": "Answer: C"},
        answer_case,
        thresholds,
        [],
        "The answer is C indeed",
        None,
        "",
        logits,
    )
    with pytest.raises(AssertionError):
        _assert_correctness(
            {**payload, "text": "cat"},
            answer_case,
            thresholds,
            [],
            "dog",
            None,
            "",
            logits,
        )


@pytest.mark.parametrize("case_name", sorted(_CASES))
def test_e2e(case_name: str, request, tmp_path: Path) -> None:
    manifest, case = _CASES[case_name]
    _require_selected(case_name, manifest, request.config)
    tp_size = manifest["tensor_parallel_size"]
    binary, runtime_root, torch = _required_environment(tp_size)
    model_dir = _checkpoint(manifest)
    prompt = _prompt(case)
    bundle = tmp_path / manifest["bundle"]

    _build_bundle(manifest, model_dir, bundle)
    _assert_rank_sections(binary, bundle, tp_size)
    payload = _run_native(
        binary,
        runtime_root,
        bundle,
        prompt,
        case,
        tp_size,
        tmp_path,
    )
    if case_name in _LOGIT_ORACLES:
        payload["logits_trace"] = _native_logits_trace(bundle, prompt, case, tp_size, tmp_path)
    reruns = int(case.get("determinism_reruns", 0))
    for _ in range(reruns):
        repeated = _run_native(
            binary,
            runtime_root,
            bundle,
            prompt,
            case,
            tp_size,
            tmp_path,
        )
        assert repeated["token_ids"] == payload["token_ids"]
        assert repeated["text"] == payload["text"]

    if _is_sampling(case):
        _assert_sampling_contract(payload, case)
        return

    if "expected_prompt_tokens" in case:
        from families.qwen.tests.runtime_receipt import assert_native_kv_receipt

        prompt_tokens = _raw_prompt_token_count(model_dir, manifest, prompt)
        assert_native_kv_receipt(payload, case, prompt_tokens)
        return

    reference = _hf_reference(
        model_dir,
        manifest,
        case,
        prompt,
        payload["token_ids"],
        torch,
    )
    if _NATIVE_KV_FIELDS <= case.keys():
        from families.qwen.tests.runtime_receipt import assert_native_kv_receipt

        assert_native_kv_receipt(payload, case, reference[-1])
    thresholds = {} if _is_sampling(case) else _thresholds(case_name)
    _assert_correctness(payload, case, thresholds, *reference[:-1])
