# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build and native-runtime E2E for MiniMax-Music3."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
from tensorrt_model_connect import BuildRequest, build

from tools.e2e_evidence import record_evidence

FAMILY = "minimax_music3"
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"


def _case_index() -> dict[str, tuple[dict, dict]]:
    result = {}
    for path in sorted(MANIFEST_ROOT.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == FAMILY
        for case in manifest["testcases"]:
            name = str(case["name"])
            assert name not in result
            result[name] = (manifest, case)
    return result


CASES = _case_index()

#: Debug lines of the deterministic prompt pass (printed under TRTMC_MM3_DEBUG).
_STAT_LINE = re.compile(r"^\[mm3\] (first_token_hidden|prompt_hidden) n=\d+ .*\bstd=([-0-9.e+]+)")
_TOP5_LINE = re.compile(r"^\[mm3\] prompt_top5 (.+)$")


def _selected(config, case_name: str) -> bool:
    """True when the command line selects this case (no selector runs nothing)."""

    requested = set()
    for raw in config.getoption("--e2e-model") or []:
        requested.update(item.strip() for item in str(raw).split(",") if item.strip())
    cases = set()
    for raw in config.getoption("--e2e-testcase") or []:
        cases.update(item.strip() for item in str(raw).split(",") if item.strip())
    if not requested and not cases:
        return False
    manifest, _ = CASES[case_name]
    model_match = not requested or bool({FAMILY, manifest["name"], case_name} & requested)
    return model_match and (not cases or case_name in cases)


def _required_env(name: str) -> Path:
    value = os.environ.get(name)
    assert value, f"selected {FAMILY} E2E requires {name}"
    path = Path(value)
    assert path.exists(), f"selected {FAMILY} E2E path does not exist: {path}"
    return path


def _model_dir(manifest: dict) -> Path:
    explicit = os.environ.get("TRTMC_MINIMAX_MUSIC3_MODEL_DIR")
    if explicit:
        return _required_env("TRTMC_MINIMAX_MUSIC3_MODEL_DIR")
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=manifest["hf_id"],
            revision=manifest["hf_revision"],
            local_files_only=True,
        )
    )


def _read_float_wav(path: Path) -> tuple[np.ndarray, int, int]:
    import soundfile as sf

    info = sf.info(path)
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    assert info.format == "WAV"
    assert samples.shape[1] == 2
    assert np.isfinite(samples).all()
    return samples, int(sample_rate), int(info.channels)


def _prompt_pass(stderr: str) -> dict:
    """Statistics of the deterministic prompt pass from the runtime's debug lines."""

    found: dict = {}
    for line in stderr.splitlines():
        stat = _STAT_LINE.match(line)
        if stat:
            found[f"{stat.group(1)}_std"] = float(stat.group(2))
        top5 = _TOP5_LINE.match(line)
        if top5:
            pairs = [item.split("=") for item in top5.group(1).split()]
            found["top1_token"] = int(pairs[0][0])
            found["top1_logit"] = float(pairs[0][1])
    return found


def _check_prompt_pass(stderr: str, reference: dict) -> None:
    """Compare the prompt pass with the bf16 engine's.

    The prompt pass runs before any sampling, so it is the same on every run; a
    quantized build with wrong activation scales moves it far more than FP8
    rounding does.
    """

    actual = _prompt_pass(stderr)
    record_evidence("prompt_pass", actual)
    assert set(actual) == {
        "first_token_hidden_std",
        "prompt_hidden_std",
        "top1_token",
        "top1_logit",
    }
    assert actual["top1_token"] == int(reference["top1_token"])
    assert abs(actual["top1_logit"] - float(reference["top1_logit"])) <= float(
        reference["top1_logit_abs_tolerance"]
    )
    for key in ("first_token_hidden_std", "prompt_hidden_std"):
        relative = abs(actual[key] / float(reference[key]) - 1.0)
        assert relative <= float(reference["std_relative_tolerance"]), (key, actual[key])


@pytest.mark.gpu
def test_minimax_music3_native_audio_e2e(case_name, request, tmp_path: Path) -> None:
    if not _selected(request.config, case_name):
        pytest.skip("direct E2E requires an explicit MiniMax-Music3 selector")
    manifest, case = CASES[case_name]
    binary = _required_env("TRTMC_BINARY")
    runtime_root = _required_env("TRTMC_RUNTIME_ROOT")
    assert (runtime_root / "libtrtmc_model_minimax_music3.so").is_file()
    thresholds = json.loads(
        (TEST_ROOT / "thresholds" / f"{case_name}.json").read_text(encoding="utf-8")
    )["threshold_overrides"]
    record_evidence("thresholds", thresholds)

    bundle = tmp_path / manifest["bundle"]
    build(
        BuildRequest(
            model_dir=_model_dir(manifest),
            output_path=bundle,
            family=FAMILY,
            task="audio_generation",
            precision=manifest["precision"],
            tensor_parallel_size=manifest["tensor_parallel_size"],
            max_sequence_length=manifest["max_sequence_length"],
            quantization=manifest.get("quantization"),
        )
    )
    output = tmp_path / "native.wav"
    command = [
        str(binary),
        "generate-audio",
        str(bundle),
        "--runtime-root",
        str(runtime_root),
        "--prompt",
        case["prompt"],
        "--description",
        case["description"],
        "--max-new-tokens",
        str(case["max_new_tokens"]),
        "--seed",
        str(case["seed"]),
        "--output",
        str(output),
    ]
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join(
        part for part in (str(runtime_root), environment.get("LD_LIBRARY_PATH", "")) if part
    )
    reference = thresholds.get("prompt_pass_reference")
    if reference is None:
        subprocess.run(command, check=True, env=environment, timeout=3600)
    else:
        environment["TRTMC_MM3_DEBUG"] = "1"
        completed = subprocess.run(
            command, check=False, env=environment, timeout=3600, capture_output=True, text=True
        )
        assert completed.returncode == 0, completed.stderr[-4000:]
        _check_prompt_pass(completed.stderr, reference)

    samples, sample_rate, channels = _read_float_wav(output)
    duration = samples.shape[0] / sample_rate
    rms = float(np.sqrt(np.mean(samples**2)))
    record_evidence("native", {"duration_s": duration, "rms": rms, "sample_rate": sample_rate})
    assert channels == 2
    assert sample_rate == int(thresholds["sampling_rate"])
    assert float(thresholds["contract_min_duration_s"]) <= duration
    assert duration <= float(thresholds["contract_max_duration_s"])
    assert rms >= float(thresholds["contract_min_rms"])


def pytest_generate_tests(metafunc) -> None:
    if "case_name" in metafunc.fixturenames:
        metafunc.parametrize("case_name", sorted(CASES))
