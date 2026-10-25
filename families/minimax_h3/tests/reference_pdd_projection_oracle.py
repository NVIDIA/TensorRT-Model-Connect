# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual TRT GPU projection and selector checks with identical BF16 head banks.

The bank inputs are synthetic, not a checkpoint or a trained PDD transformer.
CPU upstream head-composition validation is a separate required evidence tier.
"""

import argparse
import json
from pathlib import Path
import time
import sys

import numpy as np
import tensorrt as trt
import torch
import torch.nn.functional as F

from families.minimax_h3.reference_pdd_builder import build_projection_engine


HIDDEN_WIDTHS = (17, 64, 5376)
ROW_CASES = ((1, 1), (17, 5), (97, 13))


def cases():
    return [
        f"hidden-{hidden}/rows-{video}-{audio}/step-{step}"
        for hidden in HIDDEN_WIDTHS
        for video, audio in ROW_CASES
        for step in range(8)
    ]


def run(output):
    if torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one selected CUDA GPU for this component test")
    torch.set_num_threads(8)
    output.mkdir(parents=True, exist_ok=False)
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    observations = []
    enqueues = 0
    for hidden in HIDDEN_WIDTHS:
        folder = output / str(hidden)
        folder.mkdir()
        generator = torch.Generator(device="cpu").manual_seed(1900 + hidden)
        host, reference = {}, {}
        for modality, width in [("video", 96), ("audio", 32)]:
            weight = torch.randn(8, width, hidden, generator=generator).bfloat16()
            bias = torch.randn(8, width, generator=generator).bfloat16()
            host[modality] = {"weight": weight.float().numpy(), "bias": bias.float().numpy()}
            reference[modality] = (weight.cuda(), bias.cuda())
        np.savez(
            folder / "identical-banks.npz",
            **{
                name + "_" + key: value
                for name, group in host.items()
                for key, value in group.items()
            },
        )
        started = time.monotonic()
        plan = build_projection_engine(host)
        (folder / "projection.plan").write_bytes(plan)
        build_wall = time.monotonic() - started
        engine = runtime.deserialize_cuda_engine(plan)
        if engine is None or engine.num_io_tensors != 5:
            raise ValueError("Invalid serialized PDD component IO contract")
        for name, dtype in [
            ("video_hidden_states", trt.float32),
            ("audio_hidden_states", trt.float32),
            ("pdd_step", trt.int32),
            ("video_velocity", trt.float32),
            ("audio_velocity", trt.float32),
        ]:
            assert engine.get_tensor_dtype(name) == dtype
        context = engine.create_execution_context()
        assert context is not None
        for video_rows, audio_rows in ROW_CASES:
            generator.manual_seed(hidden + video_rows)
            inputs = {
                "video": torch.randn(video_rows, hidden, generator=generator).cuda().bfloat16(),
                "audio": torch.randn(audio_rows, hidden, generator=generator).cuda().bfloat16(),
            }
            np.savez(
                folder / f"inputs-{video_rows}-{audio_rows}.npz",
                **{name: value.float().cpu().numpy() for name, value in inputs.items()},
            )
            seen = {name: [] for name in inputs}
            for step in range(8):
                selector = torch.tensor([step], device="cuda", dtype=torch.int32)
                outputs = {}
                widened = {name: value.float() for name, value in inputs.items()}
                for name, value in widened.items():
                    assert context.set_input_shape(name + "_hidden_states", tuple(value.shape))
                    assert context.set_tensor_address(name + "_hidden_states", value.data_ptr())
                    shape = tuple(context.get_tensor_shape(name + "_velocity"))
                    assert shape == (value.shape[0], reference[name][1].shape[1])
                    outputs[name] = torch.empty(shape, device="cuda", dtype=torch.float32)
                    assert context.set_tensor_address(name + "_velocity", outputs[name].data_ptr())
                assert context.set_tensor_address("pdd_step", selector.data_ptr())
                assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
                torch.cuda.synchronize()
                enqueues += 1
                for name, actual in outputs.items():
                    weight, bias = reference[name]
                    expected = F.linear(inputs[name], weight[step], bias[step]).float()
                    np.savez(
                        folder / f"{name}-{video_rows}-{audio_rows}-step-{step}.npz",
                        actual=actual.cpu().numpy(),
                        expected=expected.cpu().numpy(),
                    )
                    torch.testing.assert_close(actual, expected, rtol=0.016, atol=1e-5)
                    assert torch.equal(actual, actual.bfloat16().float()), (
                        "Output BF16 rounding was lost"
                    )
                    seen[name].append(actual.cpu().numpy())
                    observations.append(
                        {
                            "hidden": hidden,
                            "video_rows": video_rows,
                            "audio_rows": audio_rows,
                            "step": step,
                            "modality": name,
                            "maximum_absolute_error": (actual - expected).abs().max().item(),
                            "engine_build_wall_seconds": build_wall,
                        }
                    )
                progress = output / "progress.pending.json"
                progress.write_text(
                    json.dumps(
                        {
                            "completed_cases": len(observations) // 2,
                            "case_count": len(cases()),
                            "TRT_projection_enqueues": enqueues,
                            "full_transformer_forwards": 0,
                        }
                    )
                )
                progress.replace(output / "progress.json")
            for name, values in seen.items():
                assert all(not np.array_equal(values[0], value) for value in values[1:]), (
                    "Selector fixture is insensitive to step changes"
                )
        del context, engine, plan, reference, host
    return {
        "passed": True,
        "evidence_level": "TRT component projection/selector using identical synthetic BF16 banks",
        "comparisons": observations,
        "comparison_count": len(observations),
        "cases": cases(),
        "TRT_projection_enqueues": enqueues,
        "rtol": 0.016,
        "atol": 1e-5,
        "all_output_BF16_boundaries_retained": True,
        "all_eight_head_choices_observed": True,
        "checkpoint_loaded": False,
        "full_transformer_forwards": 0,
        "trained_model_qualified": False,
        "VSA_128_qualified": False,
        "realtime_qualified": False,
        "GPU": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "TensorRT_version": trt.__version__,
        "torch_version": torch.__version__,
        "CUDA_version": torch.version.cuda,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        assert not torch.cuda.is_initialized(), "Preflight must not create a CUDA context"
        print(
            json.dumps(
                {
                    "cases": cases(),
                    "case_count": len(cases()),
                    "expected_comparisons": 2 * len(cases()),
                    "CUDA_initialized": False,
                    "GPU_model_calls": 0,
                    "python": sys.version,
                    "torch_version": torch.__version__,
                    "TensorRT_version": trt.__version__,
                    "numpy_version": np.__version__,
                    "trained_model_qualified": False,
                }
            )
        )
        return
    try:
        value = run(args.output)
        with (args.output / "report.json").open("x") as f:
            json.dump(value, f, indent=2)
        print(
            json.dumps(
                {
                    key: value[key]
                    for key in [
                        "passed",
                        "comparison_count",
                        "TRT_projection_enqueues",
                        "trained_model_qualified",
                    ]
                }
            )
        )
    except BaseException as error:
        if args.output.is_dir():
            with (args.output / "failure.json").open("x") as f:
                json.dump(
                    {
                        "error_type": type(error).__name__,
                        "message": str(error),
                        "trained_model_qualified": False,
                    },
                    f,
                    indent=2,
                )
        raise


if __name__ == "__main__":
    main()
