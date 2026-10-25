# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trace the unchanged reduced BF16 language gate; exit zero is not qualification.

Run an unmodified plan first, then observe family-owned graph boundaries on
the exact retained requests. Added outputs can change fusion/tactics; preserve
both results, and never infer a production fix from the instrumented plan.
"""

import argparse
from contextlib import ExitStack
import hashlib
import importlib.metadata
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from families.minimax_h3 import text_encoder_builder as native
from families.minimax_h3.tests.reference_language_case import (
    ATOL,
    RTOL,
    ROW_COUNTS,
    SEED,
    make_reference_language_case,
    reference_language_expected,
    reference_language_inputs,
)


def compare(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert np.isfinite(actual).all() and np.isfinite(expected).all(), "nonfinite language values"
    error = np.abs(actual - expected)
    bad = error > ATOL + RTOL * np.abs(expected)
    return dict(
        shape=list(actual.shape),
        max_abs_error=float(error.max()),
        mean_abs_error=float(error.mean()),
        mismatched_elements=int(bad.sum()),
        compared_elements=error.size,
        passes_original_tolerance=not bool(bad.any()),
    )


def descriptor(tensor):
    import torch

    value = tensor.detach().contiguous().cpu()
    raw = value.view(torch.uint16).numpy() if value.dtype == torch.bfloat16 else value.numpy()
    return dict(
        dtype=str(value.dtype),
        shape=list(value.shape),
        sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
    ), raw


def execute(runtime, plan, inputs):
    import tensorrt as trt
    import torch

    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    context = engine.create_execution_context()
    assert context is not None
    for name, value in inputs.items():
        assert context.set_input_shape(name, tuple(value.shape))
        assert context.set_tensor_address(name, value.data_ptr())
    outputs = {}
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            assert engine.get_tensor_dtype(name) == trt.float32
            shape = tuple(context.get_tensor_shape(name))
            assert all(value > 0 for value in shape)
            outputs[name] = torch.empty(shape, device="cuda", dtype=torch.float32)
            assert context.set_tensor_address(name, outputs[name].data_ptr())
    stream = torch.cuda.current_stream()
    assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return {name: value.cpu().numpy() for name, value in outputs.items()}


def instrumented_plan(weights):
    import tensorrt as trt

    names = []
    norm_names = iter(
        f"layers.{index}.{suffix}"
        for index in range(4)
        for suffix in (
            "input_layernorm",
            "self_attn.q_norm",
            "self_attn.k_norm",
            "post_attention_layernorm",
        )
    )
    linear, norm, silu, rotary, cache = (
        native._linear,
        native._qwen_rms_norm,
        native.op.silu,
        native._qwen_rope,
        native._rope_cache,
    )
    activation_index = 0
    rotary_index = 0

    def observe(network, tensor, name):
        assert name not in names
        copied = network.add_cast(tensor, trt.float32).get_output(0)
        copied.name = "trace::" + name
        network.mark_output(copied)
        names.append(name)

    def wrapped_linear(network, value, weights, name):
        key = name.removeprefix("model.language_model.")
        observe(network, value, "input::" + key)
        result = linear(network, value, weights, name)
        observe(network, result, key)
        return result

    def wrapped_norm(network, value, weight, width):
        key = next(norm_names)
        observe(network, value, "input::" + key)
        result = norm(network, value, weight, width)
        observe(network, result, key)
        return result

    def wrapped_silu(network, value):
        nonlocal activation_index
        key = f"layers.{activation_index}.mlp.act_fn"
        activation_index += 1
        observe(network, value, "input::" + key)
        result = silu(network, value)
        observe(network, result, key)
        return result

    def wrapped_rotary(network, value, *args, **kwargs):
        nonlocal rotary_index
        layer, branch = divmod(rotary_index, 2)
        rotary_index += 1
        result = rotary(network, value, *args, **kwargs)
        observe(network, result, f"rope::layers.{layer}.{'k' if branch else 'q'}")
        return result

    def wrapped_cache(network, value, **kwargs):
        result = cache(network, value, **kwargs)
        for kind, tensor in zip(("cos", "sin"), result):
            observe(network, tensor, "rope." + kind)
        return result

    with (
        patch.object(native, "_linear", wrapped_linear),
        patch.object(native, "_qwen_rms_norm", wrapped_norm),
        patch.object(native.op, "silu", wrapped_silu),
        patch.object(native, "_qwen_rope", wrapped_rotary),
        patch.object(native, "_rope_cache", wrapped_cache),
    ):
        plan = native.build_text_encoder_engine(
            weights, sequence_length=16, reference=True, workspace_bytes=1 << 30
        )
    assert next(norm_names, None) is None and activation_index == 4 and rotary_index == 8
    return plan, names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    assert importlib.metadata.version("transformers") == "5.15.0"
    import tensorrt as trt
    import torch
    from transformers.models.qwen3_vl import modeling_qwen3_vl as upstream

    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    args.output.mkdir(mode=0o700)
    with pytest.MonkeyPatch.context() as monkeypatch, torch.no_grad():
        original, weights = make_reference_language_case(monkeypatch)
        weight_hashes = {
            name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in weights.items()
        }
        baseline_plan = native.build_text_encoder_engine(
            weights, sequence_length=16, reference=True, workspace_bytes=1 << 30
        )
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        original.cuda()
        requests = []
        request_arrays = {}
        request_metadata = []
        for rows in ROW_COUNTS:
            inputs = reference_language_inputs(rows)
            expected = reference_language_expected(original, inputs).float().cpu().numpy()
            actual = execute(runtime, baseline_plan, inputs)["encoder_hidden_states"]
            requests.append((rows, inputs, expected, actual))
            metadata = {}
            for name, value in inputs.items():
                metadata[name], request_arrays[f"rows{rows}_{name}"] = descriptor(value)
            request_metadata.append(dict(rows=rows, inputs=metadata))
            np.save(args.output / f"rows{rows}-upstream.npy", expected, allow_pickle=False)
            np.save(args.output / f"rows{rows}-baseline.npy", actual, allow_pickle=False)
        # Store exact BF16 bit payloads (uint16) and integer inputs, not a
        # regenerated request selected after observing a comparison outcome.
        np.savez(args.output / "requests.npz", **request_arrays)
        with (args.output / "requests.json").open("x") as file:
            json.dump(request_metadata, file, indent=2, allow_nan=False)
        traced_plan, names = instrumented_plan(weights)
        expected_boundaries = {}

        def capture(name):
            def hook(module, inputs, output):
                expected_boundaries[name] = output[0].detach().float().cpu().numpy()
                expected_boundaries["input::" + name] = inputs[0][0].detach().float().cpu().numpy()

            return hook

        def capture_cache(module, inputs, output):
            for kind, tensor in zip(("cos", "sin"), output):
                expected_boundaries["rope." + kind] = tensor[..., :4].float().cpu().numpy()

        rotary_layer = 0
        original_rotary = upstream.apply_rotary_pos_emb

        def capture_rotary(*args, **kwargs):
            nonlocal rotary_layer
            result = original_rotary(*args, **kwargs)
            index = rotary_layer
            rotary_layer += 1
            if index < 4:
                for kind, tensor in zip(("q", "k"), result):
                    expected_boundaries[f"rope::layers.{index}.{kind}"] = (
                        tensor.transpose(1, 2).reshape(tensor.shape[2], -1).float().cpu().numpy()
                    )
            return result

        modules = dict(original.named_modules())
        reports = []
        with ExitStack() as stack:
            for name in names:
                if not name.startswith(("input::", "rope")):
                    hook = modules[name].register_forward_hook(capture(name))
                    stack.callback(hook.remove)
            hook = original.rotary_emb.register_forward_hook(capture_cache)
            stack.callback(hook.remove)
            stack.enter_context(patch.object(upstream, "apply_rotary_pos_emb", capture_rotary))
            with (args.output / "boundaries.jsonl").open("x") as trace:
                for rows, inputs, expected, baseline in requests:
                    expected_boundaries.clear()
                    rotary_layer = 0
                    repeated = reference_language_expected(original, inputs).float().cpu().numpy()
                    np.testing.assert_array_equal(repeated, expected)
                    assert set(expected_boundaries) == set(names), (
                        sorted(set(names) - set(expected_boundaries)),
                        sorted(set(expected_boundaries) - set(names)),
                    )
                    outputs = execute(runtime, traced_plan, inputs)
                    boundaries = []
                    for name in names:
                        result = compare(outputs["trace::" + name], expected_boundaries[name])
                        item = dict(rows=rows, name=name, **result)
                        trace.write(json.dumps(item, allow_nan=False) + "\n")
                        trace.flush()
                        boundaries.append(item)
                    actual = outputs["encoder_hidden_states"]
                    np.save(
                        args.output / f"rows{rows}-instrumented.npy", actual, allow_pickle=False
                    )
                    reports.append(
                        dict(
                            rows=rows,
                            baseline_vs_upstream=compare(baseline, expected),
                            instrumented_vs_upstream=compare(actual, expected),
                            instrumented_vs_baseline=compare(actual, baseline),
                            first_boundary_outside_original_tolerance=next(
                                (
                                    row["name"]
                                    for row in boundaries
                                    if not row["passes_original_tolerance"]
                                ),
                                None,
                            ),
                        )
                    )
        report = dict(
            seed=SEED,
            row_counts=list(ROW_COUNTS),
            rtol=RTOL,
            atol=ATOL,
            upstream="transformers==5.15.0, BF16 SDPA",
            weights_sha256=weight_hashes,
            baseline_plan_sha256=hashlib.sha256(baseline_plan).hexdigest(),
            instrumented_plan_sha256=hashlib.sha256(traced_plan).hexdigest(),
            comparisons=reports,
            diagnostic_only=True,
            original_gate_unchanged=True,
            trained_language_qualified=False,
            native_ref2va_video_qualified=False,
        )
        with (args.output / "localization.json").open("x") as file:
            json.dump(report, file, indent=2, allow_nan=False)
        print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
