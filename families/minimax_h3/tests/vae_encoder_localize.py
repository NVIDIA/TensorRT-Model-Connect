# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Localize the unchanged trained-tile FP32 gate; never qualify on exit zero.

Compare every native convolution, frame GroupNorm and residual boundary with
the original upstream encoder on the identical saved input. Extra outputs can
change TensorRT tactics/fusion, so instrumented results are diagnostic only.
Also compare upstream CPU/CUDA execution under the same existing tolerance;
that observation does not authorize changing any criterion.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from families.minimax_h3 import vae_encoder_builder as native
from families.minimax_h3.checkpoint import load_selected_component_state_dict, numpy_state
from families.minimax_h3.tests.vae_encoder_checkpoint_oracle import (
    ATOL,
    CONFIG_SHA256,
    REVISION,
    RTOL,
)


PIXELS_SHA = "47c6cbbe02e376104c884b5d3023e1a194f3f0e900bcf02d3c55d85911b7d868"
UPSTREAM_SHA = "f90aa6f5a90276b01bfbad7971e88cd290c73dbafe86b848512fa033fc760949"
NATIVE_SHA = "8fd4cf0a036dde0c729110a351f0b11492ad2267b89ba11faf19e01bba5fc953"


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def compare(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert np.isfinite(actual).all() and np.isfinite(expected).all(), "nonfinite boundary values"
    error = np.abs(actual - expected)
    mismatches = int(np.count_nonzero(error > ATOL + RTOL * np.abs(expected)))
    return dict(
        shape=list(actual.shape),
        max_abs_error=float(error.max()),
        mean_abs_error=float(error.mean()),
        mismatched_elements=mismatches,
        compared_elements=error.size,
        passes_original_tolerance=mismatches == 0,
    )


def upstream_hook_path(name):
    # Downsample.forward pads bottom/right BEFORE calling its child conv.
    # Native _conv receives the parent's unpadded input. The parent's output
    # equals the child's output, so observing the parent aligns both ends.
    if ".downsamplers." in name and name.endswith(".conv"):
        return name[: -len(".conv")]
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--case-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    assert args.checkpoint.name == REVISION
    assert sha(args.checkpoint / "vae/config.json") == CONFIG_SHA256
    for name, digest in [
        ("pixels.npy", PIXELS_SHA),
        ("upstream-moments.npy", UPSTREAM_SHA),
        ("native-moments.npy", NATIVE_SHA),
    ]:
        assert sha(args.case_directory / name) == digest
    args.output.mkdir(mode=0o700)

    import tensorrt as trt
    import torch
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import MiniMaxH3VideoEncoder3d

    torch.set_num_threads(8)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    pixels = torch.from_numpy(
        np.load(args.case_directory / "pixels.npy", allow_pickle=False)
    ).cuda()
    old_reference = np.load(args.case_directory / "upstream-moments.npy", allow_pickle=False)
    old_native = np.load(args.case_directory / "native-moments.npy", allow_pickle=False)
    state = load_selected_component_state_dict(args.checkpoint / "vae", native.checkpoint_keys())
    with torch.device("meta"):
        original = torch.nn.Module()
        original.encoder = MiniMaxH3VideoEncoder3d()
        original.quant_conv = torch.nn.Conv3d(48, 48, 1)
    original.load_state_dict(state, strict=True, assign=True)
    original.eval().cuda()

    # Wrap existing builder operations without changing their implementation,
    # coefficients, input, output tolerance, or the failed baseline plan.
    names = []

    def instrument(function, *, convolution_input=False):
        def wrapped(network, value, weights, prefix, *positional, **keywords):
            if convolution_input and prefix != "quant_conv":
                # A separate output identity avoids renaming the producer's
                # tensor, and distinguishes SiLU drift from convolution drift.
                copied = network.add_identity(value).get_output(0)
                copied.name = "trace::input::" + prefix
                network.mark_output(copied)
                names.append("input::" + prefix)
            output = function(network, value, weights, prefix, *positional, **keywords)
            if prefix != "quant_conv":
                output.name = "trace::" + prefix
                network.mark_output(output)
                names.append(prefix)
            return output

        return wrapped

    with (
        patch.object(native, "_conv", instrument(native._conv, convolution_input=True)),
        patch.object(native, "_frame_group_norm", instrument(native._frame_group_norm)),
        patch.object(native, "_resnet", instrument(native._resnet)),
    ):
        plan = native.build_vae_tile_encoder_engine(numpy_state(state), max_frames=1)
    assert len(names) == len(set(names))
    expected_boundaries = {}

    def capture(name):
        def hook(module, inputs, output):
            expected_boundaries[name] = output.detach()
            if "input::" + name in names:
                expected_boundaries["input::" + name] = inputs[0].detach()

        return hook

    modules = dict(original.named_modules())
    hooks = [
        modules[upstream_hook_path(name)].register_forward_hook(capture(name))
        for name in names
        if not name.startswith("input::")
    ]
    with torch.no_grad(), torch.backends.cudnn.flags(allow_tf32=False):
        expected = original.quant_conv(original.encoder(pixels))
    for hook in hooks:
        hook.remove()
    assert set(expected_boundaries) == set(names)

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    context = engine.create_execution_context()
    assert context is not None and context.set_input_shape("pixels", tuple(pixels.shape))
    assert context.set_tensor_address("pixels", pixels.data_ptr())
    stream = torch.cuda.Stream()
    outputs = {}
    for name in [*("trace::" + name for name in names), "moments"]:
        shape = tuple(context.get_tensor_shape(name))
        outputs[name] = torch.empty(shape, dtype=torch.float32, device="cuda")
        assert context.set_tensor_address(name, outputs[name].data_ptr())
    stream.wait_stream(torch.cuda.current_stream())
    assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    boundaries = []
    with (args.output / "boundaries.jsonl").open("x") as trace:
        for name in names:
            actual_value = outputs["trace::" + name].cpu().numpy()
            expected_value = expected_boundaries[name].cpu().numpy()
            trace.write(
                json.dumps(
                    dict(
                        name=name,
                        actual_shape=list(actual_value.shape),
                        expected_shape=list(expected_value.shape),
                        actual_finite=bool(np.isfinite(actual_value).all()),
                        expected_finite=bool(np.isfinite(expected_value).all()),
                    )
                )
                + "\n"
            )
            trace.flush()
            result = compare(actual_value, expected_value)
            boundaries.append(dict(name=name, **result))
            trace.write(json.dumps(boundaries[-1], allow_nan=False) + "\n")
            trace.flush()
    expected_array = expected.cpu().numpy()
    instrumented = outputs["moments"].cpu().numpy()
    np.save(args.output / "instrumented-moments.npy", instrumented, allow_pickle=False)
    np.save(args.output / "upstream-cuda-moments.npy", expected_array, allow_pickle=False)
    first_failure = next(
        (row["name"] for row in boundaries if not row["passes_original_tolerance"]), None
    )
    report = dict(
        checkpoint_revision=REVISION,
        input_sha256=PIXELS_SHA,
        rtol=RTOL,
        atol=ATOL,
        fp32=True,
        tf32=False,
        instrumented_plan_bytes=len(plan),
        instrumented_plan_sha256=hashlib.sha256(plan).hexdigest(),
        upstream_cuda_reproduction=compare(expected_array, old_reference),
        failed_baseline_vs_upstream_cuda=compare(old_native, expected_array),
        instrumented_vs_upstream_cuda=compare(instrumented, expected_array),
        instrumented_vs_failed_baseline=compare(instrumented, old_native),
        first_boundary_outside_original_tolerance=first_failure,
        boundaries=boundaries,
        diagnostic_only=True,
        trained_tile_qualified=False,
        native_Ref2VA_video_qualified=False,
        full_app_qualified=False,
    )
    # Retain localization even if the independent CPU experiment fails later.
    with (args.output / "localization.json").open("x") as file:
        json.dump(report, file, indent=2, allow_nan=False)
    expected_boundaries.clear()
    del outputs, context, engine, runtime, expected, plan
    gc.collect()
    torch.cuda.empty_cache()
    original.cpu()
    with torch.no_grad():
        cpu = original.quant_conv(original.encoder(pixels.cpu())).numpy()
    np.save(args.output / "upstream-cpu-moments.npy", cpu, allow_pickle=False)
    with (args.output / "upstream-self-comparison.json").open("x") as file:
        json.dump(
            dict(
                upstream_cpu_vs_upstream_cuda=compare(cpu, expected_array),
                original_gate_unchanged=True,
                criterion_change_authorized=False,
            ),
            file,
            indent=2,
            allow_nan=False,
        )
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
