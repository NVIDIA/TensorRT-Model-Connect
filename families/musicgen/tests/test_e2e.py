# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt -> family bundle -> public Task CLI -> official MusicGen waveform."""

import json
import os
from pathlib import Path
import struct
import subprocess

import numpy as np
import pytest
from tensorrt_model_connect import BuildRequest, build
from tensorrt_model_connect.model_support import load_model_metadata, resolve_family
from tools.e2e_evidence import evidence_stage, record_evidence

ROOT = Path(__file__).parent
MANIFEST = json.loads((ROOT / "manifests/musicgen-medium-fp32.json").read_text())
CASE = MANIFEST["testcases"][0]
THRESHOLDS = json.loads((ROOT / "thresholds/musicgen-medium-fp32.json").read_text())


def pytest_generate_tests(metafunc):
    if "case_name" not in metafunc.fixturenames:
        return
    config = metafunc.config
    models = {value.strip() for raw in config.getoption("--e2e-model") for value in raw.split(",")}
    models_file = config.getoption("--e2e-models-file")
    if models_file:
        models.update(
            line.strip()
            for line in Path(models_file).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    cases = {
        value.strip() for raw in config.getoption("--e2e-testcase") for value in raw.split(",")
    }
    selected = (
        bool(models or cases)
        and (not models or models.intersection({"musicgen", MANIFEST["name"]}))
        and (not cases or CASE["name"] in cases)
    )
    parameter = (
        CASE["name"]
        if selected
        else pytest.param(
            CASE["name"],
            marks=pytest.mark.skip(
                reason="MusicGen E2E requires an explicit family, model, or testcase selector"
            ),
        )
    )
    metafunc.parametrize("case_name", [parameter])


def read_float_wav(path):
    data = path.read_bytes()
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    offset, rate, channels, samples = 12, None, None, None
    while offset + 8 <= len(data):
        name, size = struct.unpack_from("<4sI", data, offset)
        payload = data[offset + 8 : offset + 8 + size]
        if name == b"fmt ":
            kind, channels, rate, _, _, bits = struct.unpack_from("<HHIIHH", payload)
            assert kind == 3 and bits == 32
        elif name == b"data":
            samples = np.frombuffer(payload, dtype="<f4").copy()
        offset += 8 + size + (size % 2)
    assert samples is not None
    return samples, rate, channels


def test_musicgen_e2e(case_name, tmp_path):
    import torch
    from transformers import AutoTokenizer, MusicgenForConditionalGeneration
    from transformers.utils.logging import disable_progress_bar

    disable_progress_bar()

    model_dir = os.environ.get("TRTMC_MUSICGEN_MODEL_DIR")
    if model_dir:
        model_dir = Path(model_dir)
    else:
        from huggingface_hub import snapshot_download

        model_dir = Path(
            snapshot_download(
                MANIFEST["hf_id"], revision=MANIFEST["hf_revision"], local_files_only=True
            )
        )
    binary = Path(os.environ["TRTMC_BINARY"])
    runtime = Path(os.environ["TRTMC_RUNTIME_ROOT"])
    assert torch.cuda.is_available()
    assert resolve_family(load_model_metadata(model_dir))[0] == "musicgen"
    bundle = Path(os.environ.get("TRTMC_MUSICGEN_BUNDLE", str(tmp_path / MANIFEST["bundle"])))
    with evidence_stage("build"):
        if "TRTMC_MUSICGEN_BUNDLE" not in os.environ:
            build(
                BuildRequest(
                    model_dir=model_dir,
                    output_path=bundle,
                    family="musicgen",
                    task="text_to_audio",
                    precision="fp32",
                    max_sequence_length=54,
                )
            )
        assert bundle.is_file()
        record_evidence("manifest", MANIFEST)
        record_evidence("thresholds", THRESHOLDS)
    with evidence_stage("official_reference"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        reference = (
            MusicgenForConditionalGeneration.from_pretrained(
                model_dir, local_files_only=True, dtype=torch.float32, attn_implementation="eager"
            )
            .to("cuda")
            .eval()
        )
        expected = []
        for prompt in CASE["prompts"]:
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                audio = reference.generate(
                    **inputs,
                    do_sample=False,
                    guidance_scale=3.0,
                    max_new_tokens=MANIFEST["max_sequence_length"] - 1,
                )
            expected.append(audio[0, 0].float().cpu().numpy())
        del reference
        torch.cuda.empty_cache()
    actual = []
    with evidence_stage("native_task_runtime"):
        for index, prompt in enumerate(CASE["prompts"]):
            output = tmp_path / f"native-{index}.wav"
            command = [
                str(binary),
                "generate-audio",
                str(bundle),
                "--runtime-root",
                str(runtime),
                "--prompt",
                prompt,
                "--set",
                "do_sample=false",
                "--set",
                "guidance_scale=3.0",
                "--output",
                str(output),
            ]
            result = subprocess.run(
                command, check=True, capture_output=True, text=True, timeout=600
            )
            record_evidence(f"native_command_{index}", command)
            record_evidence(f"native_result_{index}", result.stdout)
            record_evidence(f"native_audio_{index}", output)
            samples, rate, channels = read_float_wav(output)
            assert rate == 32000 and channels == 1 and samples.shape == (32000,)
            assert np.isfinite(samples).all()
            actual.append(samples)
    with evidence_stage("waveform_oracle"):
        for index, (samples, target) in enumerate(zip(actual, expected, strict=True)):
            assert samples.shape == target.shape
            observed, expected_pcm = samples.astype(np.float64), target.astype(np.float64)
            rms = float(np.sqrt(np.mean(expected_pcm**2)))
            relative_rmse = float(np.sqrt(np.mean((observed - expected_pcm) ** 2)) / rms)
            cosine = float(
                np.dot(observed, expected_pcm)
                / (np.linalg.norm(observed) * np.linalg.norm(expected_pcm))
            )
            record_evidence(f"reference_waveform_{index}", target)
            record_evidence(
                f"waveform_metrics_{index}",
                {"rms": rms, "cosine": cosine, "relative_rmse": relative_rmse},
            )
            assert rms >= THRESHOLDS["waveform_rms_min"]
            assert cosine >= THRESHOLDS["waveform_cosine_min"]
            assert relative_rmse <= THRESHOLDS["waveform_relative_rmse_max"]
        difference = float(np.linalg.norm(actual[0] - actual[1]) / np.linalg.norm(actual[0]))
        record_evidence("prompt_relative_difference", difference)
        assert difference >= THRESHOLDS["prompt_waveform_relative_difference_min"]
