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
from contextlib import ExitStack
from functools import partial
import gc
import hashlib
import json
from pathlib import Path
import time
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
ISOLATED_CONVOLUTION = "encoder.down_blocks.2.resnets.0.conv2"


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


def division_silu(network, value):
    """Diagnostic-only CUDA-reference evaluation order, not a production edit."""
    import tensorrt as trt

    if value.dtype != trt.float32:
        raise ValueError("The encoder activation-order experiment requires FP32")
    negative = network.add_unary(value, trt.UnaryOperation.NEG).get_output(0)
    exponential = network.add_unary(negative, trt.UnaryOperation.EXP).get_output(0)
    one = native.op.constant(network, np.ones((1,) * len(tuple(value.shape)), np.float32))
    denominator = network.add_elementwise(
        one, exponential, trt.ElementWiseOperation.SUM
    ).get_output(0)
    return network.add_elementwise(value, denominator, trt.ElementWiseOperation.DIV).get_output(0)


def fixed_statistics_affine(network, value, mean, rstd, gamma, beta, *, mode):
    """Oracle-assisted diagnostic only; statistics are NOT computed by this graph."""
    import tensorrt as trt

    if mode not in {"centered", "reparameterized"}:
        raise ValueError("Unknown fixed-statistics affine diagnostic")
    if any(item.dtype != trt.float32 for item in (value, mean, rstd, gamma, beta)):
        raise ValueError("The fixed-statistics affine diagnostic requires FP32")

    def operation(left, right, kind):
        return network.add_elementwise(left, right, kind).get_output(0)

    if mode == "centered":
        centered = operation(value, mean, trt.ElementWiseOperation.SUB)
        normalized = operation(centered, rstd, trt.ElementWiseOperation.PROD)
        scaled = operation(normalized, gamma, trt.ElementWiseOperation.PROD)
        return operation(scaled, beta, trt.ElementWiseOperation.SUM)
    scale = operation(rstd, gamma, trt.ElementWiseOperation.PROD)
    shift = operation(
        beta, operation(scale, mean, trt.ElementWiseOperation.PROD), trt.ElementWiseOperation.SUB
    )
    return operation(
        operation(value, scale, trt.ElementWiseOperation.PROD), shift, trt.ElementWiseOperation.SUM
    )


def self_computed_group_norm(network, value, weights, prefix, config, *, mode):
    """Diagnostic candidate: native FP32 statistics, no reference-valued operands."""
    import tensorrt as trt

    if mode not in {"centered", "reparameterized"}:
        raise ValueError("Unknown self-computed GroupNorm candidate")
    if value.dtype != trt.float32:
        raise ValueError("Self-computed encoder GroupNorm requires FP32")
    config.validate()
    _, channels, _, height, width = tuple(value.shape)
    groups = config.norm_num_groups
    if channels <= 0 or channels % groups:
        raise ValueError("GroupNorm requires statically grouped channels")
    arrange = network.add_shuffle(value)
    arrange.first_transpose = trt.Permutation((0, 2, 1, 3, 4))
    arrange.reshape_dims = (0, 0, groups, channels // groups, height, width)
    grouped = arrange.get_output(0)
    axes = (1 << 3) | (1 << 4) | (1 << 5)  # Never reduce across frames or groups.

    def operation(left, right, kind):
        return network.add_elementwise(left, right, kind).get_output(0)

    mean = network.add_reduce(grouped, trt.ReduceOperation.AVG, axes, True).get_output(0)
    centered = operation(grouped, mean, trt.ElementWiseOperation.SUB)
    square = operation(centered, centered, trt.ElementWiseOperation.PROD)
    variance = network.add_reduce(square, trt.ReduceOperation.AVG, axes, True).get_output(0)
    epsilon = native.op.constant(network, np.full((1,) * 6, config.norm_eps, np.float32))
    variance = operation(variance, epsilon, trt.ElementWiseOperation.SUM)
    root = network.add_unary(variance, trt.UnaryOperation.SQRT).get_output(0)
    rstd = network.add_unary(root, trt.UnaryOperation.RECIP).get_output(0)
    shape = (1, 1, groups, channels // groups, 1, 1)
    gamma = native.op.weight_constant(network, weights[prefix + ".weight"].reshape(shape))
    beta = native.op.weight_constant(network, weights[prefix + ".bias"].reshape(shape))
    if mode == "centered":
        normalized = operation(centered, rstd, trt.ElementWiseOperation.PROD)
        affine = operation(
            operation(normalized, gamma, trt.ElementWiseOperation.PROD),
            beta,
            trt.ElementWiseOperation.SUM,
        )
    else:
        scale = operation(rstd, gamma, trt.ElementWiseOperation.PROD)
        bias = operation(
            beta,
            operation(scale, mean, trt.ElementWiseOperation.PROD),
            trt.ElementWiseOperation.SUB,
        )
        affine = operation(
            operation(grouped, scale, trt.ElementWiseOperation.PROD),
            bias,
            trt.ElementWiseOperation.SUM,
        )
    restore = network.add_shuffle(affine)
    restore.reshape_dims = (0, 0, channels, height, width)
    restore.second_transpose = trt.Permutation((0, 2, 1, 3, 4))
    return restore.get_output(0)


def compare_groupnorm_order(args):
    """Inspect one real GroupNorm against explicitly oracle-assisted affine replays."""
    import tensorrt as trt
    import torch
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import MiniMaxH3VideoEncoder3d

    assert torch.version.git_version == "7661cd9c6b841b62b7f411aa52ec51f05457263b"
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(8)
    prefix = "encoder.down_blocks.0.resnets.0.norm1"
    names = [
        owner + suffix for owner in ("encoder.conv_in", prefix) for suffix in (".weight", ".bias")
    ]
    state = load_selected_component_state_dict(args.checkpoint / "vae", names)
    assert set(state) == set(names) and all(
        value.dtype == torch.float32 for value in state.values()
    )
    weights = numpy_state(state)
    with torch.device("meta"):
        original = MiniMaxH3VideoEncoder3d()
    convolution = original.conv_in
    normalization = original.down_blocks[0].resnets[0].norm1
    del original
    assert normalization.num_groups == 32 and normalization.eps == 1e-6
    convolution.load_state_dict(
        {key: state["encoder.conv_in." + key] for key in ("weight", "bias")},
        strict=True,
        assign=True,
    )
    normalization.load_state_dict(
        {key: state[prefix + "." + key] for key in ("weight", "bias")}, strict=True, assign=True
    )
    convolution.eval().cuda()
    normalization.eval().cuda()
    with torch.no_grad(), torch.backends.cudnn.flags(allow_tf32=False):
        pixels = torch.from_numpy(
            np.load(args.case_directory / "pixels.npy", allow_pickle=False)
        ).cuda()
        value = convolution(pixels).contiguous()
        expected = normalization(value)
        reference, mean, rstd = torch.native_group_norm(
            value, normalization.weight, normalization.bias, 1, 128, 256 * 256, 32, 1e-6
        )
    assert torch.equal(expected.view(torch.int32), reference.view(torch.int32))
    assert tuple(value.shape) == (1, 128, 1, 256, 256)
    pristine = value.clone()
    expected_array = expected.cpu().numpy().copy()
    mean = mean.reshape(1, 1, 32, 1, 1, 1).contiguous()
    rstd = rstd.reshape(1, 1, 32, 1, 1, 1).contiguous()

    def save(name, array):
        with (args.output / name).open("xb") as handle:
            np.save(handle, array, allow_pickle=False)

    for name, array in {
        "common-input.npy": value.cpu().numpy(),
        "upstream-output.npy": expected_array,
        "upstream-mean.npy": mean.cpu().numpy(),
        "upstream-rstd.npy": rstd.cpu().numpy(),
        "gamma.npy": weights[prefix + ".weight"],
        "beta.npy": weights[prefix + ".bias"],
    }.items():
        assert array.dtype == np.float32 and np.isfinite(array).all()
        save(name, array)
    logger = trt.Logger(trt.Logger.WARNING)
    variants = ("native-normalization", "fixed-stats-centered", "fixed-stats-reparameterized")
    trials = []
    for repeat in range(1, 4):
        for variant in variants:
            assisted = variant != "native-normalization"
            print(
                json.dumps(
                    dict(
                        stage="build",
                        variant=variant,
                        repeat=repeat,
                        uses_upstream_statistics=assisted,
                    )
                ),
                flush=True,
            )
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
            )
            options = builder.create_builder_config()
            options.builder_optimization_level = 1
            native.op.configure_builder(options)
            native.op.configure_workspace(options, 1 << 30, default_bytes=1 << 30)
            options.clear_flag(trt.BuilderFlag.TF32)
            incoming = network.add_input("values", trt.float32, tuple(value.shape))
            try:
                if not assisted:
                    output = native._frame_group_norm(
                        network, incoming, weights, prefix, native.VideoEncoderConfig()
                    )
                else:
                    arrange = network.add_shuffle(incoming)
                    arrange.first_transpose = trt.Permutation((0, 2, 1, 3, 4))
                    arrange.reshape_dims = (1, 1, 32, 4, 256, 256)
                    stats_shape = (1, 1, 32, 1, 1, 1)
                    mean_input = network.add_input("upstream_mean", trt.float32, stats_shape)
                    rstd_input = network.add_input("upstream_rstd", trt.float32, stats_shape)
                    gamma = native.op.weight_constant(
                        network, weights[prefix + ".weight"].reshape(1, 1, 32, 4, 1, 1)
                    )
                    beta = native.op.weight_constant(
                        network, weights[prefix + ".bias"].reshape(1, 1, 32, 4, 1, 1)
                    )
                    affine = fixed_statistics_affine(
                        network,
                        arrange.get_output(0),
                        mean_input,
                        rstd_input,
                        gamma,
                        beta,
                        mode=variant.removeprefix("fixed-stats-"),
                    )
                    restore = network.add_shuffle(affine)
                    restore.reshape_dims = (1, 1, 128, 256, 256)
                    restore.second_transpose = trt.Permutation((0, 2, 1, 3, 4))
                    output = restore.get_output(0)
                output.name = "output"
                network.mark_output(output)
                native.op.validate_native_network(
                    network, expected_attentions=0, label="GroupNorm diagnostic"
                )
                plan = builder.build_serialized_network(network, options)
                assert plan is not None
                plan = bytes(plan)
            finally:
                native.op.release_weight_buffers(network)
            runtime = trt.Runtime(logger)
            engine = runtime.deserialize_cuda_engine(plan)
            assert engine is not None
            context = engine.create_execution_context()
            assert context is not None and tuple(context.get_tensor_shape("output")) == tuple(
                value.shape
            )
            actual = torch.empty_like(value)
            assert context.set_tensor_address("values", value.data_ptr())
            assert context.set_tensor_address("output", actual.data_ptr())
            if assisted:
                assert context.set_tensor_address("upstream_mean", mean.data_ptr())
                assert context.set_tensor_address("upstream_rstd", rstd.data_ptr())
            stream = torch.cuda.current_stream()
            assert context.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
            array = actual.cpu().numpy().copy()
            filename = variant + "-" + str(repeat) + ".npy"
            save(filename, array)
            trial = dict(
                variant=variant,
                repeat=repeat,
                output=filename,
                uses_upstream_statistics=assisted,
                is_production_candidate=False,
                **compare(array, expected_array),
                bit_exact_elements=int(
                    np.count_nonzero(array.view(np.uint32) == expected_array.view(np.uint32))
                ),
                plan_bytes=len(plan),
                plan_sha256=hashlib.sha256(plan).hexdigest(),
            )
            trials.append(trial)
            with (args.output / (variant + "-" + str(repeat) + ".json")).open("x") as handle:
                json.dump(trial, handle, indent=2, allow_nan=False)
            print(json.dumps(dict(stage="trial-complete", **trial), allow_nan=False), flush=True)
            del actual, array, context, engine, runtime, plan, network, options, builder
            gc.collect()
    assert torch.equal(value.view(torch.int32), pristine.view(torch.int32))
    report = dict(
        checkpoint_revision=REVISION,
        input_sha256=PIXELS_SHA,
        torch_git_revision=torch.version.git_version,
        normalization=prefix,
        selected_checkpoint_tensors=names,
        common_input_shape=list(value.shape),
        rtol=RTOL,
        atol=ATOL,
        precision="fp32",
        tf32=False,
        original_module_native_group_norm_control_bit_exact=True,
        common_input_unmodified=True,
        trials_per_variant=3,
        trials=trials,
        oracle_assisted_variants=list(variants[1:]),
        production_graph_changed=False,
        fixed_statistics_are_diagnostic_not_a_runtime_implementation=True,
        full_encoder_calls=0,
        full_video_calls=0,
        serialized_plans_saved=False,
        diagnostic_only=True,
        trained_tile_qualified=False,
        native_Ref2VA_video_qualified=False,
        full_app_qualified=False,
        limitation="Fixed-statistics variants consume original CUDA means/rstd as explicit "
        "inputs solely to study affine evaluation order. Their errors cannot be "
        "used to qualify a deployable normalization, encoder or full model.",
    )
    with (args.output / "report.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps(report, allow_nan=False), flush=True)


def compare_activation_order(args):
    """Repeated non-instrumented trained-tile builds; no best-run selection."""
    import tensorrt as trt
    import torch
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import MiniMaxH3VideoEncoder3d

    assert torch.version.git_version == "7661cd9c6b841b62b7f411aa52ec51f05457263b"
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(8)
    logger = trt.Logger(trt.Logger.WARNING)
    computed = args.compare_computed_groupnorm
    variants = (
        ("division", "computed-centered", "computed-reparameterized")
        if computed
        else ("baseline", "division")
    )

    def execute(plan, value):
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(plan)
        assert engine is not None
        assert engine.num_io_tensors == 2
        assert {engine.get_tensor_name(index) for index in range(2)} == {"pixels", "moments"}
        assert all(engine.get_tensor_dtype(name) == trt.float32 for name in ("pixels", "moments"))
        context = engine.create_execution_context()
        assert context is not None and context.set_input_shape("pixels", tuple(value.shape))
        result = torch.empty(
            tuple(context.get_tensor_shape("moments")), dtype=torch.float32, device="cuda"
        )
        assert context.set_tensor_address("pixels", value.data_ptr())
        assert context.set_tensor_address("moments", result.data_ptr())
        stream = torch.cuda.current_stream()
        assert context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        return result.cpu().numpy().copy()

    def pointwise_plan(function, shape):
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        options = builder.create_builder_config()
        options.builder_optimization_level = 1
        native.op.configure_builder(options)
        native.op.configure_workspace(options, 1 << 30, default_bytes=1 << 30)
        options.clear_flag(trt.BuilderFlag.TF32)
        value = network.add_input("pixels", trt.float32, shape)
        try:
            output = function(network, value)
            output.name = "moments"
            network.mark_output(output)
            plan = builder.build_serialized_network(network, options)
            assert plan is not None
            return bytes(plan)
        finally:
            native.op.release_weight_buffers(network)

    def save(name, array):
        with (args.output / name).open("xb") as handle:
            np.save(handle, array, allow_pickle=False)

    pointwise = {}
    with torch.no_grad(), torch.backends.cudnn.flags(allow_tf32=False):
        samples = torch.linspace(-20, 20, 1048577, dtype=torch.float32).reshape(1, -1).cuda()
        reference = torch.nn.functional.silu(samples).cpu().numpy()
        save("pointwise-input.npy", samples.cpu().numpy())
        save("pointwise-upstream.npy", reference)
        for name, function in (("baseline", native.op.silu), ("division", division_silu)):
            plan = pointwise_plan(function, tuple(samples.shape))
            actual = execute(plan, samples)
            save("pointwise-" + name + ".npy", actual)
            pointwise[name] = dict(
                **compare(actual, reference),
                bit_exact_elements=int(
                    np.count_nonzero(actual.view(np.uint32) == reference.view(np.uint32))
                ),
            )
            del plan
            gc.collect()
        del samples, reference
        state = load_selected_component_state_dict(
            args.checkpoint / "vae", native.checkpoint_keys()
        )
        weights = numpy_state(state)
        pixels = torch.from_numpy(
            np.load(args.case_directory / "pixels.npy", allow_pickle=False)
        ).cuda()
        with torch.device("meta"):
            original = torch.nn.Module()
            original.encoder = MiniMaxH3VideoEncoder3d()
            original.quant_conv = torch.nn.Conv3d(48, 48, 1)
        original.load_state_dict(state, strict=True, assign=True)
        original.eval().cuda()
        expected = original.quant_conv(original.encoder(pixels)).cpu().numpy().copy()
        save("upstream-moments.npy", expected)
        retained = np.load(args.case_directory / "upstream-moments.npy", allow_pickle=False)
        assert np.array_equal(expected.view(np.uint32), retained.view(np.uint32))
        del original, state
        gc.collect()
        torch.cuda.empty_cache()
        trials = []
        for repeat in range(1, 4):
            for variant in variants:
                started = time.perf_counter()
                print(json.dumps(dict(stage="build", repeat=repeat, variant=variant)), flush=True)
                with ExitStack() as patches:
                    if variant != "baseline":
                        patches.enter_context(patch.object(native.op, "silu", division_silu))
                    if variant.startswith("computed-"):
                        function = partial(
                            self_computed_group_norm, mode=variant.removeprefix("computed-")
                        )
                        patches.enter_context(patch.object(native, "_frame_group_norm", function))
                    plan = native.build_vae_tile_encoder_engine(weights, max_frames=1)
                elapsed = time.perf_counter() - started
                actual = execute(plan, pixels)
                filename = variant + "-" + str(repeat) + "-moments.npy"
                save(filename, actual)
                trial = dict(
                    variant=variant,
                    repeat=repeat,
                    output=filename,
                    **compare(actual, expected),
                    model_input_names=["pixels"],
                    reference_statistics_injected=False,
                    plan_bytes=len(plan),
                    plan_sha256=hashlib.sha256(plan).hexdigest(),
                    build_seconds=elapsed,
                )
                trials.append(trial)
                with (args.output / (variant + "-" + str(repeat) + ".json")).open("x") as handle:
                    json.dump(trial, handle, indent=2, allow_nan=False)
                print(
                    json.dumps(dict(stage="trial-complete", **trial), allow_nan=False), flush=True
                )
                del plan, actual
                gc.collect()
                torch.cuda.empty_cache()
    report = dict(
        checkpoint_revision=REVISION,
        input_sha256=PIXELS_SHA,
        torch_git_revision=torch.version.git_version,
        precision="fp32",
        tf32=False,
        rtol=RTOL,
        atol=ATOL,
        same_original_cuda_control_bit_exact=True,
        production_graph_unchanged=True,
        candidate_is_production=False,
        experiment="self-computed-groupnorm" if computed else "activation-order",
        variants=list(variants),
        trials_per_variant=3,
        trial_order=[row["variant"] for row in trials],
        pointwise=pointwise,
        trials=trials,
        division_all_three_pass_original_tile_gate=all(
            row["passes_original_tolerance"] for row in trials if row["variant"] == "division"
        ),
        all_three_pass_original_tile_gate={
            variant: all(
                row["passes_original_tolerance"] for row in trials if row["variant"] == variant
            )
            for variant in variants
        },
        reference_statistics_injected=False,
        only_input_is_original_pixels=True,
        serialized_plans_saved=False,
        diagnostic_only=True,
        full_video_calls=0,
        trained_tile_qualified=False,
        native_Ref2VA_video_qualified=False,
        full_app_qualified=False,
        limitation="Every trial is reported without best-run selection. This activation-only "
        "or self-computed-statistics experiment does not qualify a production change or replace full model gates.",
    )
    with (args.output / "report.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps(report, allow_nan=False), flush=True)


def compare_convolution_replays(values):
    """Keep inherited input drift separate from same-input operator error."""
    pairs = {
        "input_drift": ("native_input", "upstream_input"),
        "upstream_replay_control": ("upstream_on_upstream_input", "upstream_output"),
        "same_native_input": ("native_on_native_input", "upstream_on_native_input"),
        "same_upstream_input": ("native_on_upstream_input", "upstream_on_upstream_input"),
        "inherited_drift_through_upstream": (
            "upstream_on_native_input",
            "upstream_on_upstream_input",
        ),
        "standalone_vs_instrumented_native": ("native_on_native_input", "native_output"),
    }
    required = {key for pair in pairs.values() for key in pair}
    assert set(values) == required, "incomplete or ambiguous replay matrix"
    assert all(value.dtype == np.float32 for value in values.values()), "FP32 replay required"
    return {name: compare(values[left], values[right]) for name, (left, right) in pairs.items()}


def isolate_convolution(original, state, outputs, expected_boundaries, directory):
    """Cross-replay one trained convolution, without changing the encoder."""
    import tensorrt as trt
    import torch

    prefix = ISOLATED_CONVOLUTION
    convolution = dict(original.named_modules())[prefix]
    assert convolution.kernel_size == (3, 3, 3) and convolution.stride == (1, 1, 1)
    assert convolution.dilation == (1, 1, 1) and convolution.groups == 1
    assert convolution.spatial_padding == 1 and convolution.temporal_padding == 2
    assert convolution.spatial_padding_mode == "reflect"
    values = {
        "native_input": outputs["trace::input::" + prefix].detach().clone(),
        "upstream_input": expected_boundaries["input::" + prefix].detach().clone(),
        "native_output": outputs["trace::" + prefix].detach().clone(),
        "upstream_output": expected_boundaries[prefix].detach().clone(),
    }
    shape = (1, 256, 1, 64, 64)
    assert all(
        tuple(value.shape) == shape and value.dtype == torch.float32 and value.is_contiguous()
        for value in values.values()
    )
    before = {key: value.clone() for key, value in values.items()}
    with torch.no_grad(), torch.backends.cudnn.flags(allow_tf32=False):
        for source in ("native", "upstream"):
            values["upstream_on_" + source + "_input"] = convolution(values[source + "_input"])
    weights = numpy_state(
        {prefix + suffix: state[prefix + suffix] for suffix in (".weight", ".bias")}
    )
    assert all(
        torch.equal(parameter.detach().cpu(), torch.from_numpy(weights[prefix + suffix]))
        for suffix, parameter in ((".weight", convolution.weight), (".bias", convolution.bias))
    )
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    options = builder.create_builder_config()
    options.builder_optimization_level = 1
    native.op.configure_builder(options)
    native.op.configure_workspace(options, 8 << 30, default_bytes=8 << 30)
    options.clear_flag(trt.BuilderFlag.TF32)
    value = network.add_input("conv_input", trt.float32, shape)
    try:
        output = native._conv(network, value, weights, prefix, spatial_before=1, spatial_after=1)
        output.name = "conv_output"
        network.mark_output(output)
        plan = builder.build_serialized_network(network, options)
        assert plan is not None
        plan = bytes(plan)
    finally:
        native.op.release_weight_buffers(network)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    context = engine.create_execution_context()
    assert context is not None and tuple(context.get_tensor_shape("conv_output")) == shape
    stream = torch.cuda.current_stream()
    for source in ("native", "upstream"):
        output = torch.empty(shape, dtype=torch.float32, device="cuda")
        assert context.set_tensor_address("conv_input", values[source + "_input"].data_ptr())
        assert context.set_tensor_address("conv_output", output.data_ptr())
        assert context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        values["native_on_" + source + "_input"] = output
    assert all(torch.equal(values[key], value) for key, value in before.items()), "mutated capture"
    arrays = {key: value.cpu().numpy() for key, value in values.items()}
    comparisons = compare_convolution_replays(arrays)
    directory.mkdir(mode=0o700)
    files = {}
    for name, array in {
        **arrays,
        "weight": weights[prefix + ".weight"],
        "bias": weights[prefix + ".bias"],
    }.items():
        path = directory / (name + ".npy")
        with path.open("xb") as file:
            np.save(file, array, allow_pickle=False)
        files[path.name] = dict(bytes=path.stat().st_size, sha256=sha(path))
    control = arrays["upstream_on_upstream_input"]
    report = dict(
        convolution=prefix,
        input_shape=list(shape),
        rtol=RTOL,
        atol=ATOL,
        fp32=True,
        tf32=False,
        comparisons=comparisons,
        files=files,
        upstream_replay_control_bit_exact=bool(
            np.array_equal(control.view(np.uint32), arrays["upstream_output"].view(np.uint32))
        ),
        standalone_plan_bytes=len(plan),
        standalone_plan_sha256=hashlib.sha256(plan).hexdigest(),
        input_and_captured_outputs_unchanged=True,
        diagnostic_only=True,
        trained_tile_qualified=False,
        native_Ref2VA_video_qualified=False,
        full_app_qualified=False,
        limitations="Extra encoder outputs and a standalone convolution can change fusion/tactics; "
        "this is not a replacement for the original trained-tile or complete video gate.",
    )
    with (directory / "report.json").open("x") as file:
        json.dump(report, file, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--case-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--isolate-convolution",
        action="store_true",
        help="Retain and cross-replay the known first failing trained convolution",
    )
    parser.add_argument(
        "--compare-activation-order",
        action="store_true",
        help="Compare three original and three diagnostic division-SiLU tile builds",
    )
    parser.add_argument(
        "--compare-groupnorm-order",
        action="store_true",
        help="Compare the first GroupNorm with explicitly oracle-assisted affine replays",
    )
    parser.add_argument(
        "--compare-computed-groupnorm",
        action="store_true",
        help="Compare trained tile candidates that compute all statistics inside TensorRT",
    )
    args = parser.parse_args()
    if (
        sum(
            (
                args.isolate_convolution,
                args.compare_activation_order,
                args.compare_groupnorm_order,
                args.compare_computed_groupnorm,
            )
        )
        > 1
    ):
        parser.error("Select one diagnostic; never mix independent experiments")
    assert args.checkpoint.name == REVISION
    assert sha(args.checkpoint / "vae/config.json") == CONFIG_SHA256
    for name, digest in [
        ("pixels.npy", PIXELS_SHA),
        ("upstream-moments.npy", UPSTREAM_SHA),
        ("native-moments.npy", NATIVE_SHA),
    ]:
        assert sha(args.case_directory / name) == digest
    args.output.mkdir(mode=0o700)
    if args.compare_activation_order or args.compare_computed_groupnorm:
        compare_activation_order(args)
        return
    if args.compare_groupnorm_order:
        compare_groupnorm_order(args)
        return

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
    if args.isolate_convolution:
        isolate_convolution(
            original, state, outputs, expected_boundaries, args.output / "same-input-convolution"
        )
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
