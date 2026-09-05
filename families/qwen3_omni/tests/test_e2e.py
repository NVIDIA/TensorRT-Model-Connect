# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native Task API, and official-reference E2E for Qwen3-Omni."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build


FAMILY = "qwen3_omni"
TASKS = frozenset({"audio_generation"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
THRESHOLD_ROOT = TEST_ROOT / "thresholds"
SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)
TEXT_CHAT_TEMPLATE = """{%- for message in messages %}
{{- '<|im_start|>' + message.role + '\n' }}
{%- if message.content is string %}
{{- message.content }}
{%- else %}
{%- for item in message.content %}
{%- if item.type == 'text' %}{{- item.text }}{%- endif %}
{%- endfor %}
{%- endif %}
{{- '<|im_end|>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}"""
SYNTHETIC_WAVEFORM_FALLBACK = "no Code2Wav engine, generating simple waveform"


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
        model_filters.update(item.strip() for item in str(raw).split(",") if item.strip())
    models_file = config.getoption("--e2e-models-file")
    if models_file:
        model_filters.update(
            line.strip()
            for line in Path(models_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    testcase_filters = set()
    for raw in config.getoption("--e2e-testcase") or []:
        testcase_filters.update(item.strip() for item in str(raw).split(",") if item.strip())
    if not model_filters and not testcase_filters:
        return sorted(CASES), False
    selected = []
    for name, (_, manifest, _) in CASES.items():
        model_match = (
            not model_filters
            or FAMILY in model_filters
            or name in model_filters
            or manifest["name"] in model_filters
        )
        if model_match and (not testcase_filters or name in testcase_filters):
            selected.append(name)
    return sorted(selected), True


def pytest_generate_tests(metafunc) -> None:
    if "case_name" not in metafunc.fixturenames:
        return
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


def _required_path(raw: str | None, label: str) -> Path:
    assert raw, f"selected {FAMILY} E2E requires {label}"
    path = Path(raw)
    assert path.exists(), f"selected {FAMILY} E2E {label} does not exist: {path}"
    return path


def _model_dir(manifest: dict) -> Path:
    explicit = os.environ.get("TRTMC_QWEN3_OMNI_MODEL_DIR")
    if explicit:
        return _required_path(explicit, "TRTMC_QWEN3_OMNI_MODEL_DIR")
    from huggingface_hub import snapshot_download

    try:
        return Path(
            snapshot_download(
                repo_id=manifest["hf_id"],
                revision=manifest["hf_revision"],
                local_files_only=True,
            )
        )
    except Exception as error:
        raise AssertionError(
            f"selected {FAMILY} E2E requires exact cached checkpoint {manifest['hf_id']}"
        ) from error


def _runtime_paths() -> tuple[Path, Path]:
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / "libtrtmc_model_qwen3_omni.so").is_file()
    import torch

    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    return binary, runtime_root


def _build(model_dir: Path, bundle: Path, manifest: dict) -> None:
    build(
        BuildRequest(
            model_dir=model_dir,
            output_path=bundle,
            family=FAMILY,
            task=manifest["task"],
            precision=manifest["precision"],
            max_sequence_length=int(manifest["max_sequence_length"]),
            tensor_parallel_size=int(manifest["tensor_parallel_size"]),
        )
    )


def _assert_no_synthetic_fallback(stderr: str) -> None:
    assert SYNTHETIC_WAVEFORM_FALLBACK not in stderr
    assert "synthetic fallback" not in stderr.lower()


def _native(binary: Path, runtime_root: Path, bundle: Path, case: dict, output: Path) -> dict:
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(runtime_root), environment.get("LD_LIBRARY_PATH", "")) if value
    )
    completed = subprocess.run(
        [
            str(binary),
            "generate-audio",
            str(bundle),
            "--runtime-root",
            str(runtime_root),
            "--prompt",
            str(case["prompt"]),
            "--output",
            str(output),
            "--max-new-tokens",
            str(int(case["max_new_tokens"])),
            "--talker-max-new-tokens",
            str(int(case["talker_max_new_tokens"])),
            "--seed",
            str(int(case["seed"])),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        timeout=int(case["runtime_timeout_s"]),
    )
    assert output.is_file(), completed.stderr[-2000:]
    _assert_no_synthetic_fallback(completed.stderr)
    result = json.loads(completed.stdout)
    result["_stderr"] = completed.stderr
    return result


def _native_text(binary: Path, runtime_root: Path, bundle: Path, case: dict) -> dict:
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(runtime_root), environment.get("LD_LIBRARY_PATH", "")) if value
    )
    completed = subprocess.run(
        [
            str(binary),
            "run",
            str(bundle),
            "--runtime-root",
            str(runtime_root),
            "--prompt",
            str(case["prompt"]),
            "--max-new-tokens",
            str(int(case["max_new_tokens"])),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        timeout=int(case["runtime_timeout_s"]),
    )
    return json.loads(completed.stdout)


def _official_reference(
    model_dir: Path, manifest: dict, case: dict, audio_path: Path
) -> tuple[int, str]:
    import soundfile as sf
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    processor = Qwen3OmniMoeProcessor.from_pretrained(
        model_dir, revision=manifest["hf_revision"], local_files_only=True
    )
    model = (
        Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_dir,
            revision=manifest["hf_revision"],
            local_files_only=True,
            dtype=torch.bfloat16,
            enable_audio_output=True,
        )
        .to("cuda")
        .eval()
    )
    conversation = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                }
            ],
        },
        {"role": "user", "content": [{"type": "text", "text": str(case["prompt"])}]},
    ]
    inputs = processor.apply_chat_template(
        conversation,
        chat_template=TEXT_CHAT_TEMPLATE,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to(model.device)
    torch.manual_seed(int(case["seed"]))
    torch.cuda.manual_seed_all(int(case["seed"]))
    with torch.inference_mode():
        text_ids, audio = model.generate(
            **inputs,
            thinker_max_new_tokens=int(case["max_new_tokens"]),
            talker_max_new_tokens=int(case["talker_max_new_tokens"]),
            thinker_do_sample=False,
            talker_do_sample=False,
            speaker=str(case["speaker"]),
        )
    generated = text_ids[:, inputs["input_ids"].shape[1] :]
    text = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    assert text
    samples = np.asarray(audio.detach().float().cpu()).reshape(-1)
    assert samples.size > 0 and np.isfinite(samples).all()
    sf.write(audio_path, samples, 24000, subtype="PCM_16")
    return 24000, text


def _thresholds(case_name: str) -> dict:
    path = THRESHOLD_ROOT / f"{case_name}.json"
    assert path.is_file()
    return json.loads(path.read_text(encoding="utf-8"))["threshold_overrides"]


def _assert_native_audio(
    native_path: Path, reference_path: Path, sample_rate: int, thresholds: dict
) -> int:
    import soundfile as sf

    actual_info = sf.info(native_path)
    reference_info = sf.info(reference_path)
    assert actual_info.format == "WAV"
    assert reference_info.format == "WAV"
    assert int(actual_info.channels) == 1
    assert int(reference_info.channels) == 1
    assert actual_info.subtype in {"PCM_16", "FLOAT"}
    assert reference_info.subtype in {"PCM_16", "FLOAT"}
    actual, actual_rate = sf.read(native_path, dtype="float32", always_2d=False)
    reference, reference_rate = sf.read(reference_path, dtype="float32", always_2d=False)
    actual = np.asarray(actual, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    assert actual.ndim == 1
    assert reference.ndim == 1
    assert int(actual_rate) == sample_rate
    assert int(reference_rate) == sample_rate
    return _assert_waveform_parity(actual, reference, sample_rate, thresholds)


def _assert_waveform_parity(
    actual: np.ndarray, reference: np.ndarray, sample_rate: int, thresholds: dict
) -> int:
    actual = np.asarray(actual, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    assert actual.ndim == 1
    assert reference.ndim == 1
    assert actual.size > 0 and np.isfinite(actual).all()
    assert reference.size > 0 and np.isfinite(reference).all()
    assert actual.size / sample_rate >= float(thresholds["duration_s_min"])
    assert float(np.sqrt(np.mean(actual**2))) >= float(thresholds["rms_min"])
    assert float(np.max(np.abs(actual))) >= float(thresholds["peak_min"])
    ratio = actual.size / max(reference.size, 1)
    assert ratio >= float(thresholds["duration_ratio_min"])
    assert actual.size == reference.size
    denominator = float(
        np.linalg.norm(actual.astype(np.float64)) * np.linalg.norm(reference.astype(np.float64))
    )
    assert denominator > 0.0
    cosine = float(np.dot(actual.astype(np.float64), reference.astype(np.float64)) / denominator)
    assert cosine >= float(thresholds["waveform_cosine_min"])
    return int(actual.size)


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    model_dir = _model_dir(manifest)
    binary, runtime_root = _runtime_paths()
    bundle = tmp_path / manifest["bundle"]
    _build(model_dir, bundle, manifest)
    native_audio = tmp_path / "native.wav"
    native_audio_result = _native(binary, runtime_root, bundle, case, native_audio)
    assert int(native_audio_result["sample_rate"]) == 24000
    assert int(native_audio_result["num_samples"]) > 0
    native_text = _native_text(binary, runtime_root, bundle, case)
    reference_audio = tmp_path / "official-hf.wav"
    sample_rate, reference_text = _official_reference(model_dir, manifest, case, reference_audio)
    assert native_text["text"] == reference_text
    thresholds = _thresholds(case_name)
    actual_samples = _assert_native_audio(native_audio, reference_audio, sample_rate, thresholds)
    assert int(native_audio_result["num_samples"]) == actual_samples


def test_native_audio_contract_requires_exact_reference_waveform() -> None:
    sample_rate = 24_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    samples = 0.1 * np.sin(2.0 * np.pi * 440.0 * time)
    thresholds = {
        "duration_s_min": 0.5,
        "duration_ratio_min": 0.5,
        "rms_min": 0.005,
        "peak_min": 0.02,
        "waveform_cosine_min": 0.25,
    }
    assert _assert_waveform_parity(samples, samples.copy(), sample_rate, thresholds) == samples.size
    with pytest.raises(AssertionError):
        _assert_waveform_parity(samples, samples[:-1], sample_rate, thresholds)
    with pytest.raises(AssertionError):
        _assert_waveform_parity(samples, -samples, sample_rate, thresholds)
    _assert_no_synthetic_fallback("native Code2Wav completed")
    with pytest.raises(AssertionError):
        _assert_no_synthetic_fallback(SYNTHETIC_WAVEFORM_FALLBACK)
