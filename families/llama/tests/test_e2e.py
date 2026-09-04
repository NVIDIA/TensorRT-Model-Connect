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
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build


_TEST_DIR = Path(__file__).resolve().parent
_FAMILY = _TEST_DIR.parent.name
_MPI_RANK_ZERO = re.compile(r"^\[[^,]+,0\]<stdout>:(.*)$")
_NATIVE_KV_FIELDS = frozenset(
    {
        "expected_prompt_tokens",
        "expected_runtime_prefill_tokens",
        "expected_kv_cache_rows",
        "expected_prefill_chunks",
        "expected_prefill_chunk_limit",
    }
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
    assert path.is_file(), f"missing threshold file: {path}"
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
        return payload

    rank_zero_payloads = []
    for line in completed.stdout.splitlines():
        match = _MPI_RANK_ZERO.fullmatch(line)
        if match and match.group(1).lstrip().startswith("{"):
            rank_zero_payloads.append(match.group(1))
    assert len(rank_zero_payloads) == 1, completed.stdout
    payload = json.loads(rank_zero_payloads[0])
    payload["runtime_stderr"] = completed.stderr
    return payload


def _raw_prompt_token_count(model_dir: Path, manifest: dict, prompt: str) -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=bool(manifest.get("trust_remote_code", False)),
    )
    return len(tokenizer.encode(prompt, add_special_tokens=False))


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


def _apply_repetition_penalty(logits, token_ids: list[int], penalty: float):
    if penalty == 1.0:
        return logits
    logits = logits.clone()
    for token_id in set(token_ids):
        logits[token_id] = (
            logits[token_id] * penalty if logits[token_id] < 0 else logits[token_id] / penalty
        )
    return logits


def _allowed_tokens(torch, logits, case: dict, history: list[int]):
    penalty = float(case.get("repetition_penalty", 1.0))
    logits = _apply_repetition_penalty(logits.float(), history, penalty)
    temperature = float(case.get("temperature", 1.0))
    if temperature > 0.0:
        logits = logits / temperature
    probabilities = torch.softmax(logits, dim=-1)
    allowed = torch.ones_like(probabilities, dtype=torch.bool)

    top_k = int(case.get("top_k", 0))
    if top_k > 0 and top_k < probabilities.numel():
        top_indices = torch.topk(probabilities, top_k).indices
        top_mask = torch.zeros_like(allowed)
        top_mask[top_indices] = True
        allowed &= top_mask

    top_p = float(case.get("top_p", 1.0))
    if top_p < 1.0:
        sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
        keep = torch.cumsum(sorted_probabilities, dim=-1) - sorted_probabilities < top_p
        top_p_mask = torch.zeros_like(allowed)
        top_p_mask[sorted_indices[keep]] = True
        allowed &= top_p_mask

    min_p = float(case.get("min_p", 0.0))
    if min_p > 0.0:
        allowed &= probabilities >= probabilities.max() * min_p
    return allowed


def _hf_reference(
    model_dir: Path,
    manifest: dict,
    case: dict,
    prompt: str,
    actual_ids: list[int],
    torch,
) -> tuple[list[int], str, float | None, str]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    trust_remote_code = bool(manifest.get("trust_remote_code", False))
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )
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
    inputs = _render_prompt(tokenizer, prompt, case).to(model.device)
    prompt_ids = inputs["input_ids"][0].tolist()
    if "expected_prompt_token_ids" in case:
        assert prompt_ids == case["expected_prompt_token_ids"]

    sampling_support = None
    reference_ids: list[int] = []
    reference_text = ""
    with torch.inference_mode():
        if _is_sampling(case):
            output = model(**inputs, use_cache=True)
            past = output.past_key_values
            history = list(prompt_ids)
            attention_mask = inputs.get("attention_mask")
            accepted = 0
            for token_id in actual_ids:
                allowed = _allowed_tokens(torch, output.logits[0, -1], case, history)
                accepted += int(0 <= token_id < allowed.numel() and allowed[token_id].item())
                history.append(token_id)
                next_id = torch.tensor([[token_id]], dtype=torch.long, device=model.device)
                next_inputs = {
                    "input_ids": next_id,
                    "past_key_values": past,
                    "use_cache": True,
                }
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [
                            attention_mask,
                            torch.ones(
                                (1, 1),
                                dtype=attention_mask.dtype,
                                device=model.device,
                            ),
                        ],
                        dim=1,
                    )
                    next_inputs["attention_mask"] = attention_mask
                output = model(**next_inputs)
                past = output.past_key_values
            sampling_support = accepted / len(actual_ids)
        else:
            generate_options = {
                "max_new_tokens": int(case["max_new_tokens"]),
                "do_sample": False,
            }
            if "repetition_penalty" in case:
                generate_options["repetition_penalty"] = float(case["repetition_penalty"])
            generated = model.generate(**inputs, **generate_options)
            reference_ids = generated[0, inputs["input_ids"].shape[1] :].tolist()
            reference_text = tokenizer.decode(reference_ids, skip_special_tokens=True).strip()

    actual_decoded = tokenizer.decode(actual_ids, skip_special_tokens=True).strip()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return reference_ids, reference_text, sampling_support, actual_decoded


def _normalized_edit_distance(left: str, right: str) -> float:
    left = " ".join(left.casefold().split())
    right = " ".join(right.casefold().split())
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


def _text_threshold(thresholds: dict[str, float]) -> float:
    if "contract_ned_threshold" in thresholds:
        return thresholds["contract_ned_threshold"]
    return thresholds["normalized_text_edit_distance"]


def _assert_correctness(
    payload: dict,
    case: dict,
    thresholds: dict[str, float],
    reference_ids: list[int],
    reference_text: str,
    sampling_support: float | None,
    actual_decoded: str,
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
    assert _normalized_edit_distance(actual_text, actual_decoded) <= _text_threshold(thresholds)

    if "expected_continuation_token_ids" in case:
        assert actual_ids == case["expected_continuation_token_ids"]
    if "expected_continuation_text" in case:
        assert case["expected_continuation_text"].casefold() in actual_decoded.casefold()
    expected_answers = case.get("expected_answers", ())
    if expected_answers:
        assert any(answer.casefold() in actual_decoded.casefold() for answer in expected_answers)

    if sampling_support is not None:
        threshold = thresholds["unstable_topk_hit_rate"]
        assert sampling_support >= threshold
        return

    assert reference_ids
    common = min(len(actual_ids), len(reference_ids))
    assert common > 0
    agreement = sum(actual_ids[index] == reference_ids[index] for index in range(common)) / common
    if "token_agreement_rate" in thresholds:
        assert agreement >= thresholds["token_agreement_rate"]
    assert _normalized_edit_distance(actual_decoded, reference_text) <= _text_threshold(thresholds)


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

    if _NATIVE_KV_FIELDS <= case.keys():
        from families.llama.tests.runtime_receipt import assert_native_kv_receipt

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
    _assert_correctness(payload, case, _thresholds(case_name), *reference)
