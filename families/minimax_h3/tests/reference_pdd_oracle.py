# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare CPU PDD primitives with caller-pinned, unmodified upstream methods.

No checkpoint, transformer, attention kernel or GPU is loaded. The caller owns
the upstream checkout/revision and source-integrity checks. Only the named math
definitions are extracted; package imports and model construction are omitted.
"""

import argparse
import ast
from dataclasses import dataclass
import json
from pathlib import Path
import types
from typing import Any, Mapping, Sequence
import sys

import numpy as np
import torch

from families.minimax_h3 import reference_pdd as candidate


def upstream_math(pdd_path, scheduler_path):
    module = types.ModuleType("trtmc_pinned_pdd_cpu_oracle")
    module.__dict__.update(
        torch=torch, dataclass=dataclass, Any=Any, Mapping=Mapping, Sequence=Sequence
    )
    sys.modules[module.__name__] = module
    tree = ast.parse(pdd_path.read_text())
    functions = {
        "validate_finite_nonzero",
        "validate_positive_int",
        "validate_pdd_steps",
        "shifted_noise_amount",
        "shifted_noise_delta",
        "unshifted_noise_amount",
        "pdd_fine_grid",
        "build_pdd_sampling_plan",
    }
    classes = {"PDDModalitySchedule", "PDDSamplingPlan"}
    selected = [
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        ast.Assign(
            targets=[ast.Name(id="PDD_GRID_MAX_T", ctx=ast.Store())], value=ast.Constant(0.999)
        ),
    ]
    selected += [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in functions
        or isinstance(node, ast.ClassDef)
        and node.name in classes
    ]
    linear = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PDDReplicatedLinear"
    )
    selected.append(
        ast.ClassDef(
            name="OutputHead",
            bases=[],
            keywords=[],
            decorator_list=[],
            body=[
                n
                for n in linear.body
                if isinstance(n, ast.FunctionDef) and n.name in {"_head_view", "_fused_params"}
            ],
        )
    )
    scheduler_tree = ast.parse(scheduler_path.read_text())
    scheduler = next(
        n
        for n in scheduler_tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MiniMaxH3Scheduler"
    )
    selected.append(
        ast.ClassDef(
            name="Scheduler",
            bases=[],
            keywords=[],
            decorator_list=[],
            body=[
                n
                for n in scheduler.body
                if isinstance(n, ast.FunctionDef)
                and n.name in {"set_timesteps", "index_for_timestep", "step"}
            ],
        )
    )
    extracted = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(extracted, str(pdd_path), "exec"), module.__dict__)
    return module


def run(pdd, scheduler):
    torch.set_num_threads(8)
    torch.set_num_interop_threads(2)
    reference = upstream_math(pdd, scheduler)
    schedules = {
        name: reference.PDDModalitySchedule(shift=shift) for name, shift in candidate.SHIFTS.items()
    }
    plan = reference.build_pdd_sampling_plan(candidate.STEP_INDICES, schedules, device="cpu")
    cases = []
    for modality in ("video", "audio"):
        actual = candidate.sampling_plan(modality)
        np.testing.assert_allclose(actual.fine_grid, plan.fine_grid.numpy(), rtol=1e-14, atol=1e-16)
        np.testing.assert_allclose(
            actual.integration_weights,
            plan.integration_weights[modality].numpy(),
            rtol=1e-14,
            atol=1e-16,
        )
        np.testing.assert_array_equal(actual.sigmas, plan.node_sigmas[modality].float().numpy())
        cases.append(
            {"kind": "schedule", "modality": modality, "fine_intervals": 32, "forwards": 8}
        )
        for hidden in (17, 64):
            for with_bias in (False, True):
                generator = torch.Generator(device="cpu").manual_seed(42 + hidden + int(with_bias))
                width = candidate.HEAD_FEATURES[modality]
                weight = torch.randn(32 * width, hidden, generator=generator).bfloat16()
                bias = (
                    torch.randn(32 * width, generator=generator).bfloat16() if with_bias else None
                )
                head = reference.OutputHead()
                head.weight, head.bias = weight, bias
                head.grid_size, head.channel_layout, head.patch_size = 32, "channel_major", 1
                for step in range(8):
                    start, end = plan.block(step)
                    expected = head._fused_params(
                        start, end, plan.integration_weights[modality], torch.float32
                    )
                    got = candidate.fuse_output_projection(
                        weight.float().numpy(),
                        None if bias is None else bias.float().numpy(),
                        modality,
                        step,
                    )
                    maximum = 0.0
                    for actual_parameter, expected_parameter in zip(got, expected):
                        if actual_parameter is None:
                            assert expected_parameter is None
                            continue
                        expected_array = expected_parameter.float().numpy()
                        # Retain the existing family BF16 comparison envelope.
                        np.testing.assert_allclose(
                            actual_parameter, expected_array, rtol=0.016, atol=1e-5
                        )
                        maximum = max(
                            maximum, float(np.max(np.abs(actual_parameter - expected_array)))
                        )
                    cases.append(
                        {
                            "kind": "fused_head",
                            "modality": modality,
                            "hidden": hidden,
                            "bias": with_bias,
                            "step": step,
                            "maximum_absolute_error": maximum,
                        }
                    )
        state = torch.randn(
            11, candidate.HEAD_FEATURES[modality], generator=torch.Generator().manual_seed(7)
        )
        state *= float(plan.node_sigmas[modality][0])
        actual_state = candidate.initial_target_noise(
            torch.randn(
                11, candidate.HEAD_FEATURES[modality], generator=torch.Generator().manual_seed(7)
            ).numpy(),
            modality,
        )
        np.testing.assert_array_equal(actual_state, state.numpy())
        original = reference.Scheduler()
        original.set_timesteps(sigmas=plan.node_sigmas[modality].float(), device="cpu")
        maximum = 0.0
        for step in range(8):
            velocity = (
                torch.randn(state.shape, generator=torch.Generator().manual_seed(100 + step))
                .bfloat16()
                .float()
            )
            state = original.step(velocity, original.timesteps[step], state, return_dict=False)[0]
            actual_state = candidate.scheduler_step(actual_state, velocity.numpy(), modality, step)
            np.testing.assert_allclose(actual_state, state.numpy(), rtol=1e-6, atol=1e-6)
            maximum = max(maximum, float(np.max(np.abs(actual_state - state.numpy()))))
        cases.append(
            {
                "kind": "complete_eight_step_state",
                "modality": modality,
                "maximum_absolute_error": maximum,
                "velocity_sign": "H3 x0-eps",
            }
        )
    assert not torch.cuda.is_initialized()
    return {
        "passed": True,
        "evidence_level": "reduced CPU PDD mathematical primitives only",
        "cases": cases,
        "case_count": len(cases),
        "checkpoint_loaded": False,
        "transformer_forwards": 0,
        "GPU_used": False,
        "trained_model_qualified": False,
        "VSA_128_kernel_qualified": False,
        "TRT_runtime_qualified": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdd-source", type=Path, required=True)
    parser.add_argument("--scheduler-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.pdd_source, args.scheduler_source)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(
        json.dumps(
            {
                key: result[key]
                for key in [
                    "passed",
                    "case_count",
                    "evidence_level",
                    "transformer_forwards",
                    "trained_model_qualified",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
