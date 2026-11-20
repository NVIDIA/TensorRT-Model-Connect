# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Saved-plan parity for Qwen3 weightless builds and native in-place KV updates.

    python families/qwen/tests/test_weightless_build.py baseline.bundle candidate.bundle \
        --report /tmp/qwen-weightless-parity.json

The caller supplies final, self-contained bundles. No checkpoint, builder or
refitter is available to the executor. Every present K/V output aliases its
cache input, and each engine receives its own copy of the initial cache.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import struct

import pytest


def _section(path: Path, name: str) -> bytes:
    with path.open("rb") as stream:
        assert stream.read(8) == b"BUNDLE\x01\x00", path
        header_length = struct.unpack("<Q", stream.read(8))[0]
        assert 0 < header_length <= 100 * 1024 * 1024
        header = json.loads(stream.read(header_length))
        assert header["format"] == 1 and header["family"] == "qwen", header
        offset, length = (header["sections"][name][key] for key in ("offset", "length"))
        assert offset >= 0 and length > 0
        assert 16 + header_length + offset + length <= path.stat().st_size
        stream.seek(16 + header_length + offset)
        data = stream.read(length)
        assert len(data) == length
        return data


class _Runner:
    def __init__(self, plan: bytes, trt, torch, logger):
        self.trt, self.torch, self.logger = trt, torch, logger
        self.runtime = trt.Runtime(logger)
        self.engine = self.runtime.deserialize_cuda_engine(plan)
        assert self.engine is not None, "saved plan failed to deserialize"
        self.context = self.engine.create_execution_context()
        assert self.context is not None, "saved plan is not independently runnable"
        self.stream = torch.cuda.Stream()
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in self.names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in self.names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        self.dtypes = {trt.float32: torch.float32, trt.float16: torch.float16,
                       trt.bfloat16: torch.bfloat16, trt.int32: torch.int32, trt.int64: torch.int64}

    def signature(self):
        return {
            name: (str(self.engine.get_tensor_mode(name)), str(self.engine.get_tensor_dtype(name)),
                   str(self.engine.get_tensor_location(name)), tuple(self.engine.get_tensor_shape(name)))
            for name in self.names
        }

    def run(self, source):
        """Clone inputs before execution; never let one engine mutate the other's cache."""
        torch, trt = self.torch, self.trt
        assert set(source) == set(self.inputs)
        inputs, outputs = {}, {}
        with torch.cuda.stream(self.stream):
            for name, value in source.items():
                dtype = self.dtypes[self.engine.get_tensor_dtype(name)]
                assert value.dtype == dtype and value.is_contiguous(), name
                location = self.engine.get_tensor_location(name)
                # clone() also isolates inputs when the source is already on
                # the target device, including CPU-located shape inputs.
                inputs[name] = value.to("cpu" if location == trt.TensorLocation.HOST else "cuda").clone()
                if -1 in self.engine.get_tensor_shape(name):
                    assert self.context.set_input_shape(name, tuple(value.shape)), name
                else:
                    assert tuple(value.shape) == tuple(self.engine.get_tensor_shape(name)), name
                assert self.context.set_tensor_address(name, inputs[name].data_ptr()), name
            assert self.context.infer_shapes() == [], "unresolved execution shapes"
            for name in self.outputs:
                shape = tuple(self.context.get_tensor_shape(name))
                assert all(dim > 0 for dim in shape), (name, shape)
                if name.startswith(("present_k_", "present_v_")):
                    input_name = name.replace("present_", "cache_", 1)
                    assert shape == tuple(inputs[input_name].shape)
                    assert self.engine.get_tensor_dtype(name) == self.engine.get_tensor_dtype(input_name)
                    outputs[name] = inputs[input_name]
                else:
                    outputs[name] = torch.empty(shape, device="cuda",
                                                dtype=self.dtypes[self.engine.get_tensor_dtype(name)])
                assert self.context.set_tensor_address(name, outputs[name].data_ptr()), name
            for name in self.outputs:
                if name.startswith("present_"):
                    input_name = name.replace("present_", "cache_", 1)
                    assert self.context.get_tensor_address(name) == self.context.get_tensor_address(input_name)
            assert self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return {name: value.cpu() for name, value in outputs.items()}


def _inputs(runner, runtime, length, past, *, caches=None, seed=7301):
    torch = runner.torch
    capacity = runtime["max_cache_length"]
    assert 0 <= past and 1 <= length and past + length <= capacity
    valid = past + length
    # Match QwenKvCache::active_key_rows: the valid prefix rounded up to 32,
    # capped at the physical capacity. Inactive bucket rows remain masked.
    rows = min(capacity, ((valid + 31) // 32) * 32)
    generator = torch.Generator().manual_seed(seed + length * 31 + past)
    inputs = {
        "token_id": torch.randint(runtime["vocab_size"], (length,), generator=generator, dtype=torch.int32),
        "position_id": torch.arange(past, valid, dtype=torch.int32),
        "cache_write_indices": torch.tensor([past], dtype=torch.int32),
        "key_value_lengths": torch.tensor([valid], dtype=torch.int32),
        "key_positions": torch.arange(rows, dtype=torch.int32),
    }
    for name in runner.inputs:
        if not name.startswith(("cache_k_", "cache_v_")):
            continue
        shape = tuple(runner.engine.get_tensor_shape(name))
        expected = (1, runtime["num_key_value_heads"], capacity, runtime["head_dim"])
        assert shape == expected, (name, shape, expected)
        if caches is not None:
            inputs[name] = caches[name.replace("cache_", "present_", 1)]
        else:
            dtype = runner.dtypes[runner.engine.get_tensor_dtype(name)]
            # Finite nonzero sentinel rows also detect writes outside the
            # requested append range; only prior active rows need random data.
            values = torch.full(shape, 0.125, dtype=dtype)
            if past:
                values[:, :, :past, :] = (torch.randn((1, shape[1], past, shape[3]),
                                                     generator=generator) * 0.125).to(dtype)
            inputs[name] = values
    return inputs


def _case_result(left, right, initial_left, initial_right, *, section, length, past, torch, phase):
    result = {"section": section, "phase": phase, "sequence_length": length,
              "past_length": past, "outputs": {}, "passed": True}
    assert set(left) == set(right)
    for name in sorted(left):
        a, b = left[name], right[name]
        assert a.shape == b.shape and a.dtype == b.dtype, name
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        exact = bool(torch.equal(a, b))
        delta = (a.float() - b.float()).abs()
        item = {"shape": list(a.shape), "dtype": str(a.dtype), "finite": finite,
                "exact": exact, "different_elements": int(torch.count_nonzero(a != b)),
                "max_absolute_error": float(delta.max()), "mean_absolute_error": float(delta.mean())}
        if name.startswith("present_"):
            input_name = name.replace("present_", "cache_", 1)
            unchanged = all(
                torch.equal(output[:, :, :past, :], source[input_name][:, :, :past, :])
                and torch.equal(output[:, :, past + length:, :], source[input_name][:, :, past + length:, :])
                for output, source in ((a, initial_left), (b, initial_right))
            )
            item["outside_append_unchanged"] = unchanged
            result["passed"] = result["passed"] and unchanged
        result["outputs"][name] = item
        result["passed"] = result["passed"] and finite and exact
    result["same_argmax"] = bool(torch.equal(left["logits"].argmax(-1), right["logits"].argmax(-1)))
    return result


def compare_bundles(baseline: Path, candidate: Path):
    import tensorrt as trt
    import torch

    assert torch.cuda.is_available(), "native Qwen saved-plan parity requires CUDA"
    runtime = json.loads(_section(baseline, "runtime.json"))
    assert runtime == json.loads(_section(candidate, "runtime.json")), "runtime contract changed"
    assert runtime["native_kv_cache"] is True and runtime["native_kv_contract_version"] == 1
    assert runtime["tensor_parallel_size"] == 1 and runtime["decoder_engine_layout"] == "split"
    expected_outputs = {"logits"} | {
        f"present_{kind}_{layer}" for layer in range(runtime["num_hidden_layers"]) for kind in ("k", "v")
    }
    report = {"baseline": {"path": str(baseline.resolve()), "bytes": baseline.stat().st_size},
              "candidate": {"path": str(candidate.resolve()), "bytes": candidate.stat().st_size},
              "tensorrt": trt.__version__, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
              "runtime": runtime, "criterion": "finite and exactly equal full logits and complete KV caches",
              "cache_aliases_checked": True, "cases": [], "passed": True}
    logger = trt.Logger(trt.Logger.WARNING)
    runners = {}
    for section in ("prefill.plan", "engine.plan"):
        ordinary = _Runner(_section(baseline, section), trt, torch, logger)
        weightless = _Runner(_section(candidate, section), trt, torch, logger)
        assert ordinary.signature() == weightless.signature(), "engine I/O contract changed"
        assert set(ordinary.outputs) == expected_outputs
        assert ordinary.engine.num_optimization_profiles == weightless.engine.num_optimization_profiles == 1
        for name in ("token_id", "position_id", "key_positions"):
            assert tuple(map(tuple, ordinary.engine.get_tensor_profile_shape(name, 0))) == tuple(map(tuple, weightless.engine.get_tensor_profile_shape(name, 0)))
        runners[section] = ordinary, weightless
        if section == "prefill.plan":
            maximum = ordinary.engine.get_tensor_profile_shape("token_id", 0)[2][0]
            cases = [(length, 0) for length in sorted({1, min(7, maximum), min(31, maximum), maximum})]
            length = min(7, maximum)
            cases.append((length, min(11, runtime["max_cache_length"] - length)))
        else:
            cases = [(1, past) for past in sorted({0, min(7, runtime["max_cache_length"] - 1),
                                                  min(63, runtime["max_cache_length"] - 1),
                                                  runtime["max_cache_length"] - 1})]
        for length, past in cases:
            inputs = _inputs(ordinary, runtime, length, past)
            left, right = ordinary.run(inputs), weightless.run(inputs)
            result = _case_result(left, right, inputs, inputs, section=section,
                                  length=length, past=past, torch=torch, phase="independent")
            report["cases"].append(result)
            report["passed"] = report["passed"] and result["passed"]
            del inputs, left, right

    # Carry each engine's own prefill cache into consecutive decode steps,
    # rather than feeding a common synthetic history at every enqueue.
    maximum = runners["prefill.plan"][0].engine.get_tensor_profile_shape("token_id", 0)[2][0]
    first_length = min(7, maximum, runtime["max_cache_length"] - 2)
    if first_length > 0:
        states = (None, None)
        past = 0
        for step, length in enumerate((first_length, 1, 1)):
            section = "prefill.plan" if step == 0 else "engine.plan"
            ordinary, weightless = runners[section]
            source_left = _inputs(ordinary, runtime, length, past, caches=states[0], seed=1901)
            source_right = _inputs(weightless, runtime, length, past, caches=states[1], seed=1901)
            left, right = ordinary.run(source_left), weightless.run(source_right)
            result = _case_result(left, right, source_left, source_right, section=section,
                                  length=length, past=past, torch=torch, phase="prefill_then_decode")
            report["cases"].append(result)
            report["passed"] = report["passed"] and result["passed"]
            states = left, right
            past += length
    return report


@pytest.mark.gpu
def test_saved_weightless_bundles_preserve_native_kv_outputs():
    baseline = os.environ.get("TRTMC_QWEN_BASELINE_BUNDLE")
    candidate = os.environ.get("TRTMC_QWEN_CANDIDATE_BUNDLE")
    if not baseline and not candidate:
        pytest.skip("provide ordinary and weightless Qwen3 bundles for real saved-plan parity")
    assert baseline and candidate, "both bundle paths are required"
    report = compare_bundles(Path(baseline), Path(candidate))
    assert report["passed"], json.dumps(report, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    report = compare_bundles(args.baseline, args.candidate)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "cases": len(report["cases"]), "report": str(args.report)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
