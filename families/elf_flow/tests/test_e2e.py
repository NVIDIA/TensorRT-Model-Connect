# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for elf_flow."""

from __future__ import annotations
import json
import os
import shutil
import subprocess
from pathlib import Path
import pytest
from tensorrt_model_connect import BuildRequest, build

FAMILY = "elf_flow"
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


def _replay(case: dict) -> tuple[dict, Path]:
    value = (case.get("inputs") or {}).get("elf_replay_artifact")
    assert value, f"selected {FAMILY} E2E requires an official replay artifact"
    path = TEST_ROOT / str(value)
    assert path.is_file(), f"selected {FAMILY} E2E replay artifact does not exist: {path}"
    return json.loads(path.read_text(encoding="utf-8")), path


def _replay_file(artifact: dict, artifact_path: Path, name: str) -> Path | None:
    value = (artifact.get("files") or {}).get(name)
    if not value:
        return None
    path = artifact_path.parent / str(value)
    assert path.is_file(), f"selected {FAMILY} E2E replay file does not exist: {path}"
    return path


def _native(
    binary: Path,
    runtime_root: Path,
    bundle: Path,
    model_dir: Path,
    manifest: dict,
    case: dict,
    tmp_path: Path,
):
    del model_dir, tmp_path
    artifact, artifact_path = _replay(case)
    arguments = [
        "--max-new-tokens",
        str(int(artifact.get("max_new_tokens", case["max_new_tokens"]))),
        "--num-steps",
        str(int(artifact["num_sampling_steps"])),
        "--guidance-scale",
        str(float(artifact["self_cond_cfg_scale"])),
        "--cfg-scale",
        str(float(artifact["cfg_scale"])),
        "--sde-gamma",
        str(float(artifact["sde_gamma"])),
        "--seed",
        str(int(artifact["seed"])),
    ]
    for option, name in (
        ("--initial-latents-raw", "initial_latents_raw"),
        ("--condition-latents-raw", "condition_latents_raw"),
        ("--condition-mask-raw", "condition_mask_raw"),
        ("--sampling-steps-raw", "sampling_steps_raw"),
        ("--sde-noise-raw", "sde_noise_raw"),
    ):
        path = _replay_file(artifact, artifact_path, name)
        if path is not None:
            arguments.extend((option, str(path)))
    if artifact.get("generation_mode") == "conditional" and not _replay_file(
        artifact, artifact_path, "condition_latents_raw"
    ):
        arguments.extend(("--prompt", _case_text(case)))
    return _run_json(
        binary,
        runtime_root,
        bundle,
        manifest,
        case,
        "run",
        *arguments,
    )


def _official_reference(model_dir: Path, manifest: dict, case: dict, tmp_path: Path):
    del model_dir, manifest, tmp_path
    artifact, artifact_path = _replay(case)
    expected_path = _replay_file(artifact, artifact_path, "expected_generated_jsonl_path")
    assert expected_path is not None
    samples = [
        json.loads(line)
        for line in expected_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(samples) == 1, f"selected {FAMILY} E2E requires exactly one replay sample"
    sample = samples[0]
    return {
        "token_ids": sample["token_ids"],
        "text": sample["generated"],
        "terminal_token_ids": artifact.get("terminal_token_ids", []),
    }


def _assert_parity(actual, expected, manifest: dict, case: dict, thresholds: dict) -> None:
    manifest["task"]
    samples = [actual]
    min_samples = int(thresholds.get("contract_min_samples", 1))
    assert len(samples) >= min_samples
    assert all(str(sample.get("text", "")).strip() for sample in samples)
    expected_samples = thresholds.get("contract_expected_samples")
    if expected_samples is not None:
        assert len(samples) == int(expected_samples)

    inputs = case.get("inputs") or {}
    if inputs.get("generation_mode") == "conditional":
        artifact, artifact_path = _replay(case)
        has_text_condition = bool(case.get("prompt") or inputs.get("source_text"))
        has_latent_condition = bool(
            _replay_file(artifact, artifact_path, "condition_latents_raw")
            and _replay_file(artifact, artifact_path, "condition_mask_raw")
        )
        assert has_text_condition or has_latent_condition

    if "normalized_text_edit_distance" in thresholds:
        limit = float(thresholds["normalized_text_edit_distance"])
    elif "contract_max_upstream_text_ned" in thresholds:
        limit = float(thresholds["contract_max_upstream_text_ned"])
    else:
        limit = float(thresholds["contract_ned_threshold"])
    assert _edit_distance(str(actual["text"]), str(expected["text"])) <= limit
    if expected.get("token_ids"):
        terminal = set(expected.get("terminal_token_ids", []))

        def strip_terminal(values):
            values = list(values)
            while values and values[-1] in terminal:
                values.pop()
            return values

        left = strip_terminal(actual.get("token_ids", []))
        right = strip_terminal(expected["token_ids"])
        agreement = sum((a == b for a, b in zip(left, right))) / max(len(left), len(right), 1)
        if "contract_min_upstream_token_agreement_rate" in thresholds:
            minimum_agreement = float(thresholds["contract_min_upstream_token_agreement_rate"])
            assert agreement >= minimum_agreement
            if minimum_agreement == 1.0:
                assert left == right
        elif "canonical_token_agreement_rate" in thresholds:
            assert agreement >= float(thresholds["canonical_token_agreement_rate"])
        elif "token_agreement_rate" in thresholds:
            assert agreement >= float(thresholds["token_agreement_rate"])
    return


def test_manifests_declare_build_dependencies() -> None:
    expected = [
        {"repo_id": "embedded-language-flows/t5_small_encoder_jax"},
        {"repo_id": "google-t5/t5-small"},
    ]
    for path in sorted(MANIFEST_ROOT.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["hf_dependencies"] == expected, path


def test_terminal_stripped_exact_token_contract() -> None:
    manifest = {"task": "text_generation"}
    case = {}
    thresholds = {
        "contract_max_upstream_text_ned": 0.0,
        "contract_min_upstream_token_agreement_rate": 1.0,
    }
    expected = {"text": "translated text", "token_ids": [4, 5, 2], "terminal_token_ids": [2]}
    _assert_parity(
        {"text": "Translated   text", "token_ids": [4, 5, 2, 2]},
        expected,
        manifest,
        case,
        thresholds,
    )
    with pytest.raises(AssertionError):
        _assert_parity(
            {"text": "translated text extra", "token_ids": [4, 5, 6, 2]},
            expected,
            manifest,
            case,
            thresholds,
        )


def test_unconditional_contract_requires_expected_sample_count() -> None:
    manifest = {"task": "text_generation"}
    case = {"inputs": {"generation_mode": "unconditional"}}
    expected = {"text": "sample", "token_ids": [7]}
    thresholds = {
        "contract_expected_samples": 1,
        "contract_max_upstream_text_ned": 0.01,
        "contract_min_upstream_token_agreement_rate": 0.99,
    }
    actual = {"text": "sample", "token_ids": [7]}
    _assert_parity(actual, expected, manifest, case, thresholds)
    with pytest.raises(AssertionError):
        _assert_parity(
            actual,
            expected,
            manifest,
            case,
            {**thresholds, "contract_expected_samples": 2},
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
