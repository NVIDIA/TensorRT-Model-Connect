# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Failure-only same-input native source observations, never a numerical oracle."""

import json
import os
from pathlib import Path
import subprocess

import numpy as np

from .dsp_observation import _runs

ROOT = Path(__file__).resolve().parents[3]
TESTS = Path(__file__).parent
_LIMIT = 64 * 1024
_SCOPE = (
    "Observed source replay on the same inputs with full final-output bit transparency; "
    "not an observation of the unobserved binary's intermediates or instruction schedule."
)


def instrument(source):
    """Insert snapshots at unique existing locals/returns without replacing math."""
    namespace = "lfm2_audio_dsp_observation::"
    edits = [
        (
            '#include "families/lfm2_audio/runtime/audio_dsp.h"\n',
            "",
            '#include "families/lfm2_audio/tests/cpp/dsp_normalization_observer.h"\n',
        ),
        ("    return total;\n", f"    {namespace}sum(values, total);\n", ""),
        (
            "        const float mean = feature_sum(reduction) / valid_frames;\n",
            f"        {namespace}start(m, features, frames, valid_frames);\n",
            f"        {namespace}mean(m, mean);\n",
        ),
        (
            "        const float sigma = std::sqrt(feature_sum(reduction) / (valid_frames - 1)) + 1e-5F;\n",
            "",
            f"        {namespace}sigma(m, sigma);\n",
        ),
        ("    return {std::move(features), frames};\n", f"    {namespace}finish();\n", ""),
    ]
    result = source
    for anchor, before, after in edits:
        if source.count(anchor) != 1 or before and before in source or after and after in source:
            raise ValueError("native observation source anchor differs")
        result = result.replace(anchor, before + anchor + after)
    restored = result
    for _, before, after in edits:
        for insertion in (before, after):
            if insertion:
                restored = restored.replace(insertion, "", 1)
    if restored != source:
        raise ValueError("native observation changed source text")
    return result


def _admit(result, baseline, output, expected, trace):
    if (
        result.returncode != 0
        or result.stdout != baseline.stdout
        or output != expected
        or len(output) != int(baseline.stdout) * 128 * 4
    ):
        raise ValueError("observed native replay is not transparent")
    if (
        trace.get("schema_version") != 1
        or trace.get("complete") is not True
        or trace.get("frames") != int(baseline.stdout)
        or trace.get("valid_frames") != trace["frames"] - 1
        or trace.get("sum_calls") != 256
        or trace.get("mean_locals") != 128
        or trace.get("sigma_locals") != 128
        or any(
            len(trace.get(name, [])) != 128
            for name in (
                "mean_sum_words",
                "variance_sum_words",
                "mean_words",
                "sigma_after_epsilon_words",
            )
        )
    ):
        raise ValueError("native source observation is incomplete")


def replay(probe, tmp_path, baseline):
    """Run one observed CPU replay; the existing original call is never repeated."""
    runtime = TESTS.parent / "runtime/audio_dsp.cpp"
    command = json.loads(probe.with_suffix(".build.json").read_text())
    expected_command = [
        command[0],
        "-std=c++17",
        "-O3",
        "-I",
        str(ROOT),
        str(TESTS / "cpp/dsp_probe.cpp"),
        str(runtime),
        "-o",
        str(probe),
    ]
    if command != expected_command:
        raise ValueError("native probe build recipe differs")
    args = list(baseline.args)
    if (
        args
        != [
            str(probe),
            "mel",
            str(tmp_path / "input.f32"),
            str(tmp_path / "output.f32"),
            "16000",
            str(tmp_path / "filters.f32"),
        ]
        or baseline.returncode != 0
    ):
        raise ValueError("native probe invocation differs")
    paths = [Path(args[2]), Path(args[5])]
    before = [path.read_bytes() for path in paths]
    expected = Path(args[3]).read_bytes()
    directory = tmp_path / "native-normalization-observation"
    directory.mkdir()
    source = directory / "audio_dsp.observed.cpp"
    source.write_text(instrument(runtime.read_text()))
    observed_probe = directory / "probe"
    command[6], command[-1] = str(source), str(observed_probe)
    compiled = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if compiled.returncode:
        raise RuntimeError("native observation compilation failed")
    output, trace_path = directory / "output.f32", directory / "trace.json"
    args[0], args[3] = str(observed_probe), str(output)
    env = dict(os.environ, TRTMC_LFM2_AUDIO_DSP_OBSERVATION=str(trace_path))
    observed = subprocess.run(args, capture_output=True, text=True, env=env, timeout=120)
    if [path.read_bytes() for path in paths] != before:
        raise ValueError("native observation inputs changed")
    if trace_path.stat().st_size > _LIMIT:
        raise ValueError("native observation exceeds bound")
    trace = json.loads(trace_path.read_text())
    actual = output.read_bytes()
    _admit(observed, baseline, actual, expected, trace)
    words = np.frombuffer(actual, dtype=np.uint32).reshape(-1, 128)
    same = bool(np.all(words == words[:, :1]))
    runs = _runs(words[:, 0])
    trace["final_features"] = {
        "shape": list(words.shape),
        "stride": [128, 1],
        "logical_axes": ["time", "feature"],
        "all_channels_bit_identical": same,
        "first_channel_rle": runs,
        "payload": "lossless_first_channel_rle_repeated"
        if same and runs is not None
        else "bounded_unavailable",
    }
    return {
        "status": "admitted_observed_source_replay",
        "scope": _SCOPE,
        "compiler": Path(command[0]).name,
        "compiler_flags": command[1:3],
        "insertion_only_source": True,
        "input_and_filter_bytes_unchanged": True,
        "full_final_output_bits_identical": True,
        "native_replays": 1,
        "original_processor_replays": 0,
        "input": {
            "sample_count": len(before[0]) // 4,
            "sample_rate": int(args[4]),
            "filter_float_count": len(before[1]) // 4,
        },
        "intermediate_inputs_lossless": all(
            trace[name]["payload"] == "lossless_first_channel_rle_repeated"
            for name in (
                "features_before_normalization",
                "mean_reduction_input",
                "variance_reduction_input",
            )
        ),
        "observation": trace,
    }


def report_failure(probe, tmp_path, baseline):
    """Every diagnostic failure leaves the original assertion untouched."""
    try:
        try:
            data = replay(probe, tmp_path, baseline)
        except Exception as error:
            data = {"status": "unavailable", "error_type": type(error).__name__, "scope": _SCOPE}
        encoded = json.dumps(data, allow_nan=False)
        if len(encoded.encode()) > _LIMIT:
            encoded = '{"status":"unavailable","error_type":"size_bound"}'
        print("LFM2_AUDIO_NATIVE_NORMALIZATION_DIAGNOSTIC=" + encoded)
    except Exception:
        try:
            print(
                'LFM2_AUDIO_NATIVE_NORMALIZATION_DIAGNOSTIC={"status":"unavailable","error_type":"emission_error"}'
            )
        except Exception:
            pass
