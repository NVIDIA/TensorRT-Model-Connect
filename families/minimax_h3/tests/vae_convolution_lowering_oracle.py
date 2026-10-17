# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect real convolution lowering using the full failed-case operands.

Reuse hash-verified trained operands and original upstream/native results.
Additional outputs and pruning can change tactics; this is a diagnostic, not
full-prefix, full-video or Ref2VA qualification. No production math is patched.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from families.minimax_h3 import vae_builder as production
from families.minimax_h3.tests.vae_decoder_accumulation_oracle import checked_array
from families.minimax_h3.tests.vae_decoder_prefix_localize import compare, save_array, sha


OPERANDS_SHA = "bfa0c857db2655a994afff8bb5533eb8db38ff5b26b939dc544d3cd1b990339f"
RETAINED_SHA = "0f4cd8ff91ffb11e753b1540a680e0deafba5197d53758cb25794acbf6eaee74"
VARIANTS = ("production-pruned", "half-observed", "half-quantized-float-observed")
CASES = (
    ["cpu-half-convolution"]
    + [f"framework-batch{b}-repeat{r}" for b in (1, 28) for r in range(3)]
    + [
        f"inspect-{v}-batch{b}-rebuild{r}"
        for b in (1, 28)
        for v in ("baseline", "candidate")
        for r in range(3)
    ]
    + [f"{v}-batch{b}-rebuild{r}" for b in (1, 28) for v in VARIANTS for r in range(3)]
)


def voxel_rows(value):
    assert value.ndim == 5
    return value.transpose(0, 2, 3, 4, 1).reshape(value.shape[0], -1, value.shape[1])


def ideal_two_round(value, weight, bias):
    assert value.dtype == weight.dtype == bias.dtype == np.float16
    assert weight.shape == (24, 24, 1, 1, 1) and bias.shape == (24,)
    rows = voxel_rows(value)
    dot = rows.astype(np.float64) @ weight.reshape(24, 24).astype(np.float64).T
    pre = dot.astype(np.float16)
    post = (pre.astype(np.float32) + bias.astype(np.float32)).astype(np.float16)
    return pre, post


def build(variant, weights, batch, logger):
    import tensorrt as trt

    op = production.op
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 1
    config.clear_flag(trt.BuilderFlag.TF32)
    op.configure_builder(config)
    op.configure_workspace(config, 8 << 30, default_bytes=8 << 30)
    source = network.add_input("latent_tiles", trt.float32, (batch, 24, 7, 16, 16))
    try:
        if variant == "production-pruned":
            outputs = {"post_quant_conv": production._post_quant_conv(network, source, weights)}
        else:
            assert variant in VARIANTS[1:]
            compute = trt.float32 if variant == "half-quantized-float-observed" else trt.float16
            kernel = weights["post_quant_conv.weight"].astype(np.float16)
            value = op.cast(network, source, trt.float16)
            if compute == trt.float32:
                kernel = kernel.astype(np.float32)
                value = op.cast(network, value, trt.float32)
            convolution = network.add_convolution_nd(
                value, 24, (1, 1, 1), kernel, trt.Weights(compute)
            )
            assert convolution is not None
            op._WEIGHT_BUFFER_KEEPALIVE.setdefault(id(network), []).append(kernel)
            convolution.name = "observed_unbiased_convolution"
            pre = op.cast(network, convolution.get_output(0), trt.float16)
            wide = op.cast(network, pre, trt.float32)
            bias = weights["post_quant_conv.bias"].astype(np.float16).astype(np.float32)
            bias = op.constant(network, bias.reshape(1, 24, 1, 1, 1))
            post = network.add_elementwise(wide, bias, trt.ElementWiseOperation.SUM).get_output(0)
            post = op.cast(network, post, trt.float16)
            outputs = {}
            for name, value in (("pre_bias", pre), ("post_quant_conv", post)):
                reshape = network.add_shuffle(value)
                reshape.first_transpose = trt.Permutation([0, 2, 3, 4, 1])
                reshape.reshape_dims = (batch, 1792, 24)
                outputs[name] = reshape.get_output(0)
        for name, value in outputs.items():
            value.name = name
            network.mark_output(value)
        plan = builder.build_serialized_network(network, config)
        assert plan is not None
        return bytes(plan)
    finally:
        op.release_weight_buffers(network)


def inspect_engine(plan, logger):
    import tensorrt as trt

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    inspector = engine.create_engine_inspector()
    assert inspector is not None
    return json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))


def execute(plan, value, logger):
    import tensorrt as trt
    import torch

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    context = engine.create_execution_context()
    assert context is not None
    assert context.set_tensor_address("latent_tiles", value.data_ptr())
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    outputs = {}
    for name in names:
        if name == "latent_tiles":
            continue
        assert engine.get_tensor_dtype(name) == trt.float16
        outputs[name] = torch.empty(
            tuple(context.get_tensor_shape(name)), dtype=torch.float16, device="cuda"
        )
        assert context.set_tensor_address(name, outputs[name].data_ptr())
    assert set(outputs) in ({"post_quant_conv"}, {"pre_bias", "post_quant_conv"})
    torch.cuda.current_stream().synchronize()
    stream = torch.cuda.Stream()
    assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    cold = {name: tensor.cpu().numpy().copy() for name, tensor in outputs.items()}
    assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    assert all(np.array_equal(cold[name], tensor.cpu().numpy()) for name, tensor in outputs.items())
    return cold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operands", type=Path, required=True)
    parser.add_argument("--retained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert sha(args.operands / "report.json") == OPERANDS_SHA
    assert sha(args.retained / "report.json") == RETAINED_SHA
    old = json.loads((args.operands / "report.json").read_text())
    retained = json.loads((args.retained / "report.json").read_text())
    operands = old["dispatch"]["28"]["operands"]

    def load(name):
        return checked_array(args.operands, operands[name])

    value, weight, bias = load("event0-arg0"), load("event1-arg0"), load("event2-arg0")
    for original, name in ((value, "event3-arg0"), (weight, "event3-arg1"), (bias, "event3-arg2")):
        assert original.dtype == np.float32
        np.testing.assert_array_equal(original.astype(np.float16), load(name))
    assert value.shape == (28, 24, 7, 16, 16) and np.isfinite(value).all()
    expected = checked_array(
        args.retained, retained["reference_arrays"]["upstream-sequential"]["post_quant_conv"]
    )
    assert expected.dtype == np.float16 and expected.shape == (28, 1792, 24)
    assert len(retained["executions"]) == 12

    import torch
    import tensorrt as trt
    from torch.nn import functional as functional

    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    logger = trt.Logger(trt.Logger.WARNING)
    args.output.mkdir(mode=0o700)
    saved = {}
    report = dict(
        cases=[],
        framework=[],
        inspected_retained=[],
        executions=[],
        source_operands_report_sha256=OPERANDS_SHA,
        source_native_report_sha256=RETAINED_SHA,
        prefix_only=True,
        production_math_modified=False,
        extra_outputs_can_change_tactics=True,
        full_trained_decoder_qualified=False,
        full_ref2va_qualified=False,
    )
    pre_ideal, post_ideal = ideal_two_round(
        value.astype(np.float16), weight.astype(np.float16), bias.astype(np.float16)
    )
    report["ideal_double_dot"] = dict(
        post_comparison=compare(post_ideal, expected),
        pre=save_array(args.output, "ideal.pre", pre_ideal, saved),
        post=save_array(args.output, "ideal.post", post_ideal, saved),
    )
    with torch.no_grad():
        cpu = functional.conv3d(
            *(torch.from_numpy(v.astype(np.float16)) for v in (value, weight, bias))
        )
    cpu = voxel_rows(cpu.numpy().copy())
    report["cpu_half_convolution"] = dict(
        output=save_array(args.output, "cpu-half-convolution", cpu, saved),
        comparison=compare(cpu, expected),
    )
    report["cases"].append("cpu-half-convolution")
    x, w, b = (torch.from_numpy(v).cuda() for v in (value, weight, bias))
    reference_pre = {}
    with torch.no_grad(), torch.backends.cudnn.flags(allow_tf32=False):
        for batch in (1, 28):
            for repeat in range(3):
                case = f"framework-batch{batch}-repeat{repeat}"
                with torch.autocast("cuda", dtype=torch.float16):
                    post = functional.conv3d(x[:batch], w, b)
                    pre = functional.conv3d(x[:batch], w)
                pre, post = (voxel_rows(v.cpu().numpy().copy()) for v in (pre, post))
                assert np.array_equal(post, expected[:batch])
                if batch in reference_pre:
                    np.testing.assert_array_equal(pre, reference_pre[batch])
                reference_pre[batch] = pre
                row = dict(
                    case=case,
                    pre=save_array(args.output, case + ".pre", pre, saved),
                    post=save_array(args.output, case + ".post", post, saved),
                    comparison=compare(post, expected[:batch]),
                )
                report["framework"].append(row)
                report["cases"].append(case)
    for row in retained["executions"]:
        case = f"inspect-{row['variant']}-batch{row['batch']}-rebuild{row['rebuild']}"
        path = args.retained / f"{row['variant']}-batch{row['batch']}-rebuild{row['rebuild']}.plan"
        assert sha(path) == row["plan_sha256"]
        info = inspect_engine(path.read_bytes(), logger)
        with (args.output / (case + ".engine.json")).open("x") as file:
            json.dump(info, file, indent=2, allow_nan=False)
        report["inspected_retained"].append(dict(case=case, plan_sha256=row["plan_sha256"]))
        report["cases"].append(case)
    weights = {"post_quant_conv.weight": weight, "post_quant_conv.bias": bias}
    with (args.output / "boundaries.jsonl").open("x") as trace:
        for batch in (1, 28):
            for variant in VARIANTS:
                for repeat in range(3):
                    case = f"{variant}-batch{batch}-rebuild{repeat}"
                    plan = build(variant, weights, batch, logger)
                    with (args.output / (case + ".plan")).open("xb") as file:
                        file.write(plan)
                    info = inspect_engine(plan, logger)
                    with (args.output / (case + ".engine.json")).open("x") as file:
                        json.dump(info, file, indent=2, allow_nan=False)
                    outputs = execute(plan, x[:batch], logger)
                    row = dict(
                        case=case,
                        variant=variant,
                        batch=batch,
                        rebuild=repeat,
                        plan_sha256=sha(args.output / (case + ".plan")),
                        cold_warm_byte_exact=True,
                        arrays={},
                        comparisons={},
                    )
                    for name, array in outputs.items():
                        reference = reference_pre[batch] if name == "pre_bias" else expected[:batch]
                        row["arrays"][name] = save_array(
                            args.output, case + "." + name, array, saved
                        )
                        row["comparisons"][name] = compare(array, reference)
                    trace.write(json.dumps(row, allow_nan=False) + "\n")
                    trace.flush()
                    report["executions"].append(row)
                    report["cases"].append(case)
    assert report["cases"] == CASES
    with (args.output / "report.json").open("x") as file:
        json.dump(report, file, indent=2, allow_nan=False)
    with (args.output / "files.json").open("x") as file:
        json.dump(
            {
                p.name: dict(bytes=p.stat().st_size, sha256=sha(p))
                for p in sorted(args.output.iterdir())
                if p.is_file() and p.name != "files.json"
            },
            file,
            indent=2,
            allow_nan=False,
        )
    print(
        json.dumps(
            dict(cases=len(CASES), production_math_modified=False, full_ref2va_qualified=False)
        )
    )


if __name__ == "__main__":
    main()
