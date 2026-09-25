# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real Python bundle writer -> C++ reader/factory, with fake engine loading."""

import json
import os
from pathlib import Path
import shutil
import struct
import subprocess

import pytest

from tensorrt_model_connect.bundle_writer import BundleWriter


@pytest.fixture(scope="module")
def factory_runner(tmp_path_factory):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    cuda = Path(os.environ.get("CUDA_HOME", os.environ.get("CUDA_PATH", "/usr/local/cuda")))
    json_include = Path(os.environ.get("NLOHMANN_JSON_INCLUDE_DIR", "/usr/include"))
    if not compiler or not (cuda / "include/cuda_runtime_api.h").is_file():
        pytest.skip("factory contract requires C++17 and CUDA headers; no GPU is used")
    if not (json_include / "nlohmann/json.hpp").is_file():
        pytest.skip("factory contract requires nlohmann/json.hpp (set NLOHMANN_JSON_INCLUDE_DIR)")
    family = Path(__file__).resolve().parents[1]
    root = family.parents[1]
    output = tmp_path_factory.mktemp("factory") / "factory"
    subprocess.run([
        compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
        "-I", str(root), "-I", str(root / "core"),
        "-I", str(root / "core/runtime/include"),
        "-I", str(cuda / "include"), "-I", str(json_include),
        str(family / "tests/cpp/test_factory.cpp"),
        *(str(family / "runtime" / name) for name in (
            "plugin.cpp", "pipeline.cpp", "bpe_tokenizer.cpp", "audio_helpers.cpp", "resampler.cpp",
        )),
        str(root / "core/runtime/bundle/bundle_format.cpp"), "-o", str(output),
    ], check=True, capture_output=True, text=True, timeout=120)
    return output


@pytest.mark.parametrize(("change", "error", "calls"), [
    ("valid", "", 3),
    ("wrong-family", "identity", 0),
    ("wrong-task", "identity", 0),
    ("wrong-backend", "identity", 0),
    ("kv", "kv-cache-size", 0),
    ("dimension", "dimension", 0),
    ("fractional-dimension", "dimension", 0),
    ("policy", "duration policy", 0),
    ("mel-length", "filterbank size", 0),
    ("mel-nonfinite", "nonfinite", 0),
    ("tokenizer-type", "native BPE tokenizer", 0),
    ("tokenizer-decoder", "native BPE tokenizer", 0),
    ("tokenizer-replacement", "native BPE tokenizer", 0),
    ("tokenizer-prepend", "native BPE tokenizer", 0),
    ("missing-runtime", "missing section", 0),
    ("missing-tokenizer", "missing section", 0),
    ("missing-encoder", "missing section", 0),
    ("missing-predictor", "missing section", 1),
    ("missing-joint", "missing section", 2),
    ("fail-engine", "failed to load encoder.plan", 1),
    ("fail-predictor", "failed to load predictor.plan", 2),
    ("fail-joint", "failed to load joint.plan", 3),
])
def test_native_factory_bundle_contract(tmp_path, factory_runner, change, error, calls):
    runtime = {
        "tensor_parallel_size": 1, "mel_sampling_rate": 16000, "num_mel_bins": 128,
        "mel_n_fft": 512, "mel_win_length": 400, "mel_hop_length": 160,
        "mel_chunk_length": 30, "mel_length": 3000, "tdt_encoder_hidden_size": 1024,
        "tdt_pred_hidden_size": 640, "tdt_pred_num_layers": 2, "tdt_vocab_size": 8192,
        "tdt_blank_id": 8192, "tdt_encoder_layers": 24, "max_source_positions": 375,
        "subsampling_factor": 8, "tdt_max_symbols_per_step": 10,
        "tdt_att_context_left": -1, "tdt_att_context_right": -1,
        "tdt_duration_values": [0, 1, 2, 3, 4], "tdt_causal_downsampling": False,
        "mel_normalize": "per_feature", "mel_preemph": 0.97,
    }
    tokenizer = {
        "model": {"type": "BPE", "vocab": {"▁hello": 0}, "merges": []},
        "decoder": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always"},
    }
    if change == "dimension":
        runtime["tdt_blank_id"] = 1024
    if change == "fractional-dimension":
        runtime["tdt_blank_id"] = 8192.0
    if change == "policy":
        runtime["tdt_duration_values"] = [0, 2]
    if change == "tokenizer-type":
        tokenizer["model"]["type"] = "Unigram"
    if change == "tokenizer-decoder":
        tokenizer["decoder"]["type"] = "ByteLevel"
    if change == "tokenizer-replacement":
        tokenizer["decoder"]["replacement"] = "_"
    if change == "tokenizer-prepend":
        tokenizer["decoder"]["prepend_scheme"] = "never"
    mel = struct.pack("<ii", 257, 128) + struct.pack("<f", 1.0) * (257 * 128)
    if change == "mel-length":
        mel = mel[:-4]
    if change == "mel-nonfinite":
        mel = mel[:8] + struct.pack("<f", float("nan")) + mel[12:]
    sections = {
        "runtime.json": json.dumps(runtime).encode(), "tokenizer.json": json.dumps(tokenizer).encode(),
        "mel_filterbank": mel, "encoder.plan": b"encoder",
        "predictor.plan": b"predictor", "joint.plan": b"joint",
    }
    omitted = {"missing-runtime": "runtime.json", "missing-tokenizer": "tokenizer.json",
               "missing-encoder": "encoder.plan", "missing-predictor": "predictor.plan",
               "missing-joint": "joint.plan"}.get(change)
    destination = tmp_path / "test.bundle"
    writer = BundleWriter(destination)
    writer.set_header(
        family="other" if change == "wrong-family" else "parakeet_tdt",
        task="other" if change == "wrong-task" else "speech_transcription",
        backend="other" if change == "wrong-backend" else "trt",
    )
    for name, content in sections.items():
        if name != omitted:
            writer.add_bytes(name, content)
    writer.finish()
    result = subprocess.run([str(factory_runner), str(destination), error, str(calls), change],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
