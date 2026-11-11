# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for WavLM."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build

FAMILY = "wavlm"
TASKS = frozenset({"speech_transcription"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
THRESHOLD_ROOT = TEST_ROOT / "thresholds"
# The released WavLM CTC checkpoints ship a single pytorch_model.bin; the E2E
# converts it next to copies of the files the builder reads and leaves the
# cache untouched.
CHECKPOINT_FILES = ("config.json", "preprocessor_config.json", "vocab.json",
                    "added_tokens.json", "tokenizer_config.json")


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
    if (model_dir / "model.safetensors").is_file():
        _build_prepared(model_dir, bundle, manifest)
        return
    from families.wavlm.convert_checkpoint import convert

    with tempfile.TemporaryDirectory(prefix="wavlm-checkpoint-") as temporary:
        prepared = Path(temporary)
        for name in CHECKPOINT_FILES:
            if (model_dir / name).is_file():
                shutil.copyfile(model_dir / name, prepared / name)
        convert(model_dir / "pytorch_model.bin", prepared / "model.safetensors")
        _build_prepared(prepared, bundle, manifest)


def _build_prepared(model_dir: Path, bundle: Path, manifest: dict) -> None:
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


def _asset(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = TEST_ROOT / path
    assert path.is_file(), f"selected {FAMILY} E2E asset does not exist: {path}"
    return path


def _native(binary: Path, runtime_root: Path, bundle: Path, case: dict) -> dict:
    return _run_json(
        binary,
        runtime_root,
        bundle,
        case,
        "transcribe",
        "--input",
        str(_asset(case["test_input_audio"])),
    )


def _official_reference(model_dir: Path, case: dict) -> tuple[str, list[int]]:
    """The reference, fed the same waveform the binary read.

    The asset is already mono at the checkpoint's own rate, so nothing
    resamples here; the feature extractor's only work is the normalisation the
    runtime performs for itself when the checkpoint asks for it.
    """
    import numpy as np
    import torch
    from transformers import AutoProcessor, WavLMForCTC

    with wave.open(str(_asset(case["test_input_audio"]))) as handle:
        assert handle.getsampwidth() == 2, "the reference path expects 16-bit PCM"
        channels = handle.getnchannels()
        raw = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    waveform = raw.astype(np.float32).reshape(-1, channels).mean(axis=1) / 32768.0

    processor = AutoProcessor.from_pretrained(str(model_dir))
    model = WavLMForCTC.from_pretrained(str(model_dir)).eval()
    values = processor(
        waveform, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt"
    ).input_values
    with torch.no_grad():
        logits = model(values).logits[0]

    predicted = logits.argmax(dim=-1).tolist()
    blank = processor.tokenizer.pad_token_id
    collapsed: list[int] = []
    previous = None
    for token_id in predicted:
        if token_id != previous and token_id != blank:
            collapsed.append(int(token_id))
        previous = token_id
    text = processor.decode(predicted)
    return (text, collapsed)


def _thresholds(case_name: str) -> dict:
    path = THRESHOLD_ROOT / f"{case_name}.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))["threshold_overrides"]


def _character_edits(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, 1):
        current = [row] + [0] * len(hypothesis)
        for column, actual in enumerate(hypothesis, 1):
            current[column] = min(previous[column] + 1, current[column - 1] + 1,
                                  previous[column - 1] + (expected != actual))
        previous = current
    return previous[-1]


def _assert_parity(native: dict, reference_text: str, reference_ids: list[int],
                   thresholds: dict) -> None:
    # Neither side is stripped here: the tokenizer strips its own transcript,
    # and the runtime must produce exactly that text, edges included.
    text = str(native["text"])
    expected = reference_text
    assert expected.strip(), "the reference produced an empty transcript, so the gate proves nothing"
    token_ids = [int(value) for value in native["token_ids"]]

    bound = thresholds.get("max_character_error_rate")
    if bound is not None:
        # An fp16 engine moves some logits far enough to flip a frame whose two
        # best classes are close, so its contract is a bounded character error
        # against the fp32 reference rather than the identical transcript.
        rate = _character_edits(expected, text) / len(expected)
        assert rate <= float(bound), (
            f"character error {rate:.4f} above {bound}: native={text!r} reference={expected!r}"
        )
        print(f"wavlm parity: character error {rate:.4f} <= {bound}, "
              f"{len(token_ids)} CTC tokens against {len(reference_ids)}")
        return

    # The fp32 gate is exact, on both the transcript and the CTC ids behind it.
    assert text == expected, f"transcript mismatch: native={text!r} reference={expected!r}"
    assert token_ids == reference_ids, (
        f"token id mismatch: native has {len(token_ids)} ids, reference {len(reference_ids)}"
    )
    print(f"wavlm parity: {len(token_ids)} CTC tokens, transcript {text!r}")


def test_e2e(tmp_path: Path, case_name: str) -> None:
    manifest_path, manifest, case = CASES[case_name]
    assert manifest_path.is_file()
    binary, runtime_root = _runtime()
    model_dir = _model_dir(manifest)
    bundle = tmp_path / str(manifest["bundle"])
    _build(model_dir, bundle, manifest)
    assert bundle.is_file(), f"{FAMILY} bundle was not produced: {bundle}"

    native = _native(binary, runtime_root, bundle, case)
    reference_text, reference_ids = _official_reference(model_dir, case)
    _assert_parity(native, reference_text, reference_ids, _thresholds(case_name))
