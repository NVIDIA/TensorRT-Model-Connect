# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for nemotron_labs_diffusion."""

from __future__ import annotations
import json
import os
import shutil
import subprocess
from pathlib import Path
import pytest
from tensorrt_model_connect import BuildRequest, build

FAMILY = "nemotron_labs_diffusion"
TASKS = frozenset({"text_generation"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
THRESHOLD_ROOT = TEST_ROOT / "thresholds"


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


def _runtime(manifest: dict) -> tuple[Path, Path]:
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / f"libtrtmc_model_{FAMILY}.so").is_file()
    import torch

    required_gpus = int(manifest["tensor_parallel_size"])
    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    assert torch.cuda.device_count() >= required_gpus, (
        f"selected {FAMILY} E2E requires {required_gpus} GPUs, found {torch.cuda.device_count()}"
    )
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
    manifest: dict,
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
    if int(manifest["tensor_parallel_size"]) > 1:
        mpirun = shutil.which("mpirun")
        assert mpirun, "selected multi-GPU E2E requires mpirun"
        invocation = [
            mpirun,
            "--tag-output",
            "-x",
            "LD_LIBRARY_PATH",
            "-np",
            str(manifest["tensor_parallel_size"]),
            *invocation,
        ]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        (value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value)
    )
    completed = subprocess.run(
        invocation,
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=int(case.get("runtime_timeout_s", 3600)),
    )
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


def _thresholds(case_name: str) -> dict:
    path = THRESHOLD_ROOT / f"{case_name}.json"
    assert path.is_file(), f"selected {FAMILY} E2E requires exact thresholds: {path}"
    return json.loads(path.read_text(encoding="utf-8"))["threshold_overrides"]


def _case_text(case: dict) -> str:
    inputs = case.get("inputs") or {}
    value = str(case.get("prompt") or case.get("test_prompt") or inputs.get("prompt") or "")
    assert value, f"selected {FAMILY} E2E requires a direct prompt"
    return value


def _edit_distance(left: str, right: str) -> float:
    a = " ".join(left.lower().split())
    b = " ".join(right.lower().split())
    previous = list(range(len(b) + 1))
    for index, char_a in enumerate(a, start=1):
        current = [index]
        for offset, char_b in enumerate(b, start=1):
            current.append(
                min(
                    current[-1] + 1, previous[offset] + 1, previous[offset - 1] + (char_a != char_b)
                )
            )
        previous = current
    return previous[-1] / max(len(a), len(b), 1)


def _canonical_terminal_tokens(
    token_ids: list[int], *, eos_token_ids: set[int], ignored_terminal_token_ids: set[int]
) -> list[int]:
    result = [int(token_id) for token_id in token_ids]
    if not result or result[-1] not in eos_token_ids:
        return result
    eos_token_id = result.pop()
    while result and result[-1] in ignored_terminal_token_ids:
        result.pop()
    result.append(eos_token_id)
    return result


def _torch_dtype(precision: str):
    import torch

    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
    }[precision]


def _native_arguments(case: dict) -> list[str]:
    inputs = case["inputs"]
    arguments = [
        "--prompt",
        _case_text(case),
        "--max-new-tokens",
        str(int(case["max_new_tokens"])),
        "--temperature",
        str(float(inputs["temperature"])),
        "--top-k",
        "1",
        "--generation-mode",
        str(inputs["generation_mode"]),
    ]
    if "block_length" in inputs:
        arguments.extend(("--block-length", str(int(inputs["block_length"]))))
    if "threshold" in inputs:
        arguments.extend(("--threshold", str(float(inputs["threshold"]))))
    return arguments


def _native(
    binary: Path,
    runtime_root: Path,
    bundle: Path,
    model_dir: Path,
    manifest: dict,
    case: dict,
    tmp_path: Path,
):
    manifest["task"]
    return _run_json(
        binary,
        runtime_root,
        bundle,
        manifest,
        case,
        "run",
        *_native_arguments(case),
    )


def _reference_generate(
    model,
    input_ids,
    *,
    mode: str,
    max_new_tokens: int,
    block_length: int,
    threshold: float,
    temperature: float,
    eos_token_id: int | None,
):
    common = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "eos_token_id": eos_token_id,
    }
    if mode == "ar":
        return model.ar_generate(input_ids, **common)
    if mode == "diffusion":
        return model.generate(
            input_ids,
            block_length=block_length,
            threshold=threshold,
            **common,
        )
    if mode in {"linear_spec", "linear_spec_lora"}:
        return model.linear_spec_generate(
            input_ids,
            block_length=block_length,
            threshold=threshold,
            **common,
        )
    raise ValueError(f"unsupported generation_mode: {mode}")


def _official_reference(model_dir: Path, manifest: dict, case: dict, tmp_path: Path):
    manifest["task"]
    import torch
    from transformers import AutoModel, AutoTokenizer

    inputs = case["inputs"]
    mode = str(inputs["generation_mode"])
    max_new_tokens = int(case["max_new_tokens"])
    block_length = int(inputs.get("block_length", 32))
    threshold = float(inputs.get("threshold", 0.9))
    temperature = float(inputs["temperature"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_dir, trust_remote_code=True, torch_dtype=_torch_dtype(case["reference_precision"])
    )
    generation_model = model
    if mode == "linear_spec_lora":
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            model_dir / "linear_spec_lora",
            adapter_name="linear_spec_lora",
        )
        generation_model = model.model
    model.to("cuda").eval()
    encoded = tokenizer(_case_text(case), return_tensors="pt")
    encoded = {key: value.to("cuda") for key, value in encoded.items()}
    with torch.no_grad():
        generated = _reference_generate(
            generation_model,
            encoded["input_ids"],
            mode=mode,
            max_new_tokens=max_new_tokens,
            block_length=block_length,
            threshold=threshold,
            temperature=temperature,
            eos_token_id=tokenizer.eos_token_id,
        )
    if isinstance(generated, tuple):
        generated = generated[0]
    ids = generated[0]
    ids = ids[encoded["input_ids"].shape[-1] :]
    ids = ids[:max_new_tokens]
    return {
        "token_ids": ids.cpu().tolist(),
        "text": tokenizer.decode(ids, skip_special_tokens=True),
    }


def _assert_parity(actual, expected, manifest: dict, case: dict, thresholds: dict) -> None:
    manifest["task"]
    limit = float(thresholds["contract_ned_threshold"])
    text_distance = _edit_distance(str(actual["text"]), str(expected["text"]))
    assert text_distance <= limit
    left = [int(token_id) for token_id in actual.get("token_ids", [])]
    right = [int(token_id) for token_id in expected.get("token_ids", [])]
    assert left and right
    assert 0 not in left
    canonical_left = _canonical_terminal_tokens(
        left, eos_token_ids={11}, ignored_terminal_token_ids={1010}
    )
    canonical_right = _canonical_terminal_tokens(
        right, eos_token_ids={11}, ignored_terminal_token_ids={1010}
    )
    assert canonical_left == canonical_right
    agreement = sum(a == b for a, b in zip(canonical_left, canonical_right)) / max(
        len(canonical_left), len(canonical_right), 1
    )
    assert agreement >= float(thresholds["canonical_token_agreement_rate"])
    return


def test_canonical_terminal_tokens_only_strip_ignored_tokens_before_eos() -> None:
    assert _canonical_terminal_tokens(
        [7, 1010, 1010, 11], eos_token_ids={11}, ignored_terminal_token_ids={1010}
    ) == [7, 11]
    assert _canonical_terminal_tokens(
        [7, 1010], eos_token_ids={11}, ignored_terminal_token_ids={1010}
    ) == [7, 1010]


def test_nemotron_labs_contract_rejects_extra_tokens_and_forbidden_token() -> None:
    manifest = {"task": "text_generation"}
    case = {}
    thresholds = {
        "contract_ned_threshold": 0.05,
        "canonical_token_agreement_rate": 1.0,
    }
    expected = {"text": "Paris", "token_ids": [7, 1010, 11]}
    _assert_parity({"text": "Paris", "token_ids": [7, 11]}, expected, manifest, case, thresholds)
    with pytest.raises(AssertionError):
        _assert_parity(
            {"text": "Paris", "token_ids": [7, 11, 8]}, expected, manifest, case, thresholds
        )
    with pytest.raises(AssertionError):
        _assert_parity(
            {"text": "Paris", "token_ids": [0, 11]}, expected, manifest, case, thresholds
        )


def test_native_arguments_preserve_generation_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    def run_json(*args):
        assert args[5] == "run"
        return list(args[6:])

    monkeypatch.setattr(f"{__name__}._run_json", run_json)
    cases = [case for _, _, case in CASES.values()]
    synthetic = {**cases[0], "inputs": {**cases[0]["inputs"], "temperature": 0.25}}
    for case in [*cases, synthetic]:
        inputs = case["inputs"]
        arguments = _native(
            Path(), Path(), Path(), Path(), {"task": "text_generation"}, case, Path()
        )
        options = dict(zip(arguments[::2], arguments[1::2], strict=True))
        expected = {
            "--prompt": _case_text(case),
            "--max-new-tokens": str(int(case["max_new_tokens"])),
            "--temperature": str(float(inputs["temperature"])),
            "--top-k": "1",
            "--generation-mode": str(inputs["generation_mode"]),
        }
        if "block_length" in inputs:
            expected["--block-length"] = str(int(inputs["block_length"]))
        if "threshold" in inputs:
            expected["--threshold"] = str(float(inputs["threshold"]))
        assert options == expected


def test_reference_generation_preserves_family_mode() -> None:
    class Reference:
        def __init__(self) -> None:
            self.calls = []

        def ar_generate(self, *args, **kwargs):
            self.calls.append(("ar", args, kwargs))
            return "ar"

        def generate(self, *args, **kwargs):
            self.calls.append(("diffusion", args, kwargs))
            return "diffusion"

        def linear_spec_generate(self, *args, **kwargs):
            self.calls.append(("linear_spec", args, kwargs))
            return "linear_spec"

    for mode, method in (
        ("ar", "ar"),
        ("diffusion", "diffusion"),
        ("linear_spec", "linear_spec"),
        ("linear_spec_lora", "linear_spec"),
    ):
        reference = Reference()
        result = _reference_generate(
            reference,
            "tokens",
            mode=mode,
            max_new_tokens=8,
            block_length=32,
            threshold=0.9,
            temperature=0.25,
            eos_token_id=11,
        )
        assert result == method
        called, args, kwargs = reference.calls[0]
        assert called == method
        assert args == ("tokens",)
        assert kwargs["max_new_tokens"] == 8
        assert kwargs["temperature"] == 0.25
        assert kwargs["eos_token_id"] == 11
        assert ("block_length" in kwargs) is (mode != "ar")
        assert ("threshold" in kwargs) is (mode != "ar")
    with pytest.raises(ValueError, match="unsupported generation_mode"):
        _reference_generate(
            Reference(),
            "tokens",
            mode="unknown",
            max_new_tokens=8,
            block_length=32,
            threshold=0.9,
            temperature=0.0,
            eos_token_id=11,
        )


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    model_dir = _model_dir(manifest)
    binary, runtime_root = _runtime(manifest)
    bundle = tmp_path / manifest["bundle"]
    _build(model_dir, bundle, manifest)
    actual = _native(binary, runtime_root, bundle, model_dir, manifest, case, tmp_path)
    expected = _official_reference(model_dir, manifest, case, tmp_path)
    _assert_parity(actual, expected, manifest, case, _thresholds(case_name))
