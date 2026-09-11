# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned checkpoint-to-native-runtime proof for K2-Horizon-Uno."""

from __future__ import annotations

import gc
from importlib.metadata import version
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_stage, record_evidence


_TEST_DIR = Path(__file__).resolve().parent
_FAMILY = _TEST_DIR.parent.name
_DECODE_RECEIPT = re.compile(
    r"^\[trtmc\.k2_horizon_uno\.decode\] "
    r"mode=(ar|linear_spec_lora) block_length=([0-9]+) "
    r"noise_mode=(deterministic_uniform) forwards=([0-9]+) "
    r"committed_tokens=([0-9]+) lookaheads=([0-9]+)$"
)


def _load_cases() -> dict[str, tuple[dict, dict]]:
    cases: dict[str, tuple[dict, dict]] = {}
    for path in sorted((_TEST_DIR / "manifests").glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == _FAMILY, path
        assert manifest["task"] == "text_generation", path
        assert manifest["precision"] == "bf16", path
        assert isinstance(manifest["max_sequence_length"], int), path
        assert manifest["tensor_parallel_size"] == 1, path
        assert isinstance(manifest["hf_dependencies"], list), path
        for case in manifest["testcases"]:
            name = case["name"]
            assert name not in cases, name
            cases[name] = (manifest, case)
    assert cases, f"{_FAMILY} has no E2E cases"
    return cases


_CASES = _load_cases()


def parse_decode_receipt(stderr: str) -> dict[str, int | str]:
    matches = [
        match
        for line in stderr.splitlines()
        if (match := _DECODE_RECEIPT.fullmatch(line.strip())) is not None
    ]
    if len(matches) != 1:
        return {}
    mode, block_length, noise_mode, forwards, committed_tokens, lookaheads = matches[0].groups()
    return {
        "mode": mode,
        "block_length": int(block_length),
        "noise_mode": noise_mode,
        "forwards": int(forwards),
        "committed_tokens": int(committed_tokens),
        "lookaheads": int(lookaheads),
    }


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


def _required_environment():
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
    return binary, runtime_root, torch


def _snapshot(repo_id: str, revision: str) -> Path:
    from huggingface_hub import snapshot_download

    assert isinstance(revision, str) and len(revision) == 40
    return Path(snapshot_download(repo_id=repo_id, revision=revision))


def _checkpoints(manifest: dict) -> tuple[Path, Path]:
    adapter = _snapshot(manifest["hf_id"], manifest["hf_revision"])
    dependencies = manifest["hf_dependencies"]
    assert len(dependencies) == 1
    dependency = dependencies[0]
    assert set(dependency) == {"repo_id", "revision"}
    base = _snapshot(dependency["repo_id"], dependency["revision"])
    assert (adapter / "adapter_config.json").is_file(), adapter
    assert (adapter / "adapter_model.safetensors").is_file(), adapter
    assert (base / "config.json").is_file(), base
    return adapter, base


def _build_bundle(manifest: dict, adapter: Path, bundle: Path) -> None:
    build(
        BuildRequest(
            model_dir=adapter,
            output_path=bundle,
            family=_FAMILY,
            task=manifest["task"],
            precision=manifest["precision"],
            max_sequence_length=manifest["max_sequence_length"],
            tensor_parallel_size=manifest["tensor_parallel_size"],
        )
    )
    assert bundle.is_file() and bundle.stat().st_size > 0, bundle


def _native_arguments(bundle: Path, runtime_root: Path, case: dict) -> list[str]:
    arguments = [
        "run",
        str(bundle),
        "--runtime-root",
        str(runtime_root),
        "--prompt",
        str(case["prompt"]),
        "--max-new-tokens",
        str(case["max_new_tokens"]),
        "--temperature",
        str(case["temperature"]),
        "--top-k",
        str(case["top_k"]),
        "--generation-mode",
        str(case["generation_mode"]),
        "--block-length",
        str(case["block_length"]),
        "--use-chat-template",
        "true" if case["use_chat_template"] else "false",
        "--enable-thinking",
        "true" if case["enable_thinking"] else "false",
    ]
    return arguments


def _run_native(binary: Path, runtime_root: Path, bundle: Path, case: dict, tmp_path: Path):
    command = [str(binary), *_native_arguments(bundle, runtime_root, case)]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=900,
        )
    except subprocess.TimeoutExpired as error:
        stderr = error.stderr or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        log = tmp_path / "native-timeout.stderr.log"
        log.write_text(str(stderr), encoding="utf-8")
        record_evidence("native_process", {"argv": command, "stderr_log": log})
        raise RuntimeError(f"Uno native generation timed out; full stderr: {log}") from error
    record_evidence(
        "native_process",
        {"argv": completed.args, "stdout": completed.stdout, "stderr": completed.stderr},
    )
    payload = json.loads(completed.stdout)
    assert isinstance(payload, dict)
    return payload, completed.stderr


def _render_prompt(tokenizer, case: dict):
    if not case["use_chat_template"]:
        return tokenizer(case["prompt"], return_tensors="pt")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": case["prompt"]}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=case["enable_thinking"],
    )
    return tokenizer(rendered, return_tensors="pt", add_special_tokens=False)


def _base_reference(base: Path, manifest: dict, case: dict, torch):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    trust_remote_code = manifest["base_reference_trust_remote_code"]
    assert trust_remote_code is True
    tokenizer = AutoTokenizer.from_pretrained(
        base,
        trust_remote_code=trust_remote_code,
        local_files_only=True,
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            base,
            dtype=torch.bfloat16,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
        .eval()
        .to("cuda")
    )
    inputs = _render_prompt(tokenizer, case).to(model.device)
    prompt_ids = [int(token) for token in inputs["input_ids"][0].tolist()]
    assert prompt_ids == case["expected_prompt_token_ids"]
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=int(case["max_new_tokens"]),
            do_sample=False,
        )
    continuation = [int(token) for token in generated[0, inputs["input_ids"].shape[1] :].tolist()]
    text = tokenizer.decode(continuation, skip_special_tokens=True).strip()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return prompt_ids, continuation, text


@pytest.mark.parametrize("case_name", sorted(_CASES))
def test_e2e(case_name: str, request, tmp_path: Path) -> None:
    manifest, case = _CASES[case_name]
    _require_selected(case_name, manifest, request.config)
    record_evidence("inputs", {"manifest": manifest, "case": case, "prompt": case["prompt"]})
    record_evidence("thresholds", {"exact_token_ids": True, "exact_text": True, "case_contract": case})
    binary, runtime_root, torch = _required_environment()
    assert version("transformers") == "5.15.0"
    assert version("safetensors") == "0.8.0"
    adapter, base = _checkpoints(manifest)
    record_evidence(
        "checkpoint",
        {"adapter_dir": str(adapter), "base_dir": str(base), "hf_id": manifest["hf_id"], "hf_revision": manifest["hf_revision"], "hf_dependencies": manifest["hf_dependencies"]},
    )
    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
        _build_bundle(manifest, adapter, bundle)

    with evidence_stage("native"):
        payload, stderr = _run_native(binary, runtime_root, bundle, case, tmp_path)
    record_evidence("native", payload)
    with evidence_stage("reference"):
        prompt_ids, reference_ids, reference_text = _base_reference(base, manifest, case, torch)
    record_evidence("reference", {"prompt_token_ids": prompt_ids, "token_ids": reference_ids, "text": reference_text})
    actual_ids = payload["token_ids"]
    actual_text = str(payload["text"]).strip()

    with evidence_stage("compare"):
        assert prompt_ids == case["expected_prompt_token_ids"]
        assert actual_ids == reference_ids == case["expected_continuation_token_ids"]
        assert actual_text == reference_text == case["expected_continuation_text"]
        assert "[trtmc.k2_horizon_uno.prompt]" not in stderr
        assert case["prompt"] not in stderr

        stats = parse_decode_receipt(stderr)
        assert stats == {
            "mode": case["expected_decode_mode"],
            "block_length": case["block_length"],
            "noise_mode": "deterministic_uniform",
            "forwards": case["expected_forwards"],
            "committed_tokens": case["expected_committed_tokens"],
            "lookaheads": case["expected_lookaheads"],
        }
        if case["generation_mode"] == "linear_spec_lora":
            assert stats["committed_tokens"] > stats["forwards"]
