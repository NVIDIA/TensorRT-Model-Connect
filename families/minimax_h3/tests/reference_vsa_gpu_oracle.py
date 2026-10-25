# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real 128-token VSA binding/bridge comparisons, without a trained transformer.

Use the supplied pinned upstream pooling/routing functions and an FP32 PyTorch
attention oracle. Retain every BF16 operand/output, geometry, plan and failure.
The large Blackwell case has the actual six-portrait window's rows and heads;
it remains one synthetic attention operation, not a full-model qualification.
"""

import argparse
import ctypes
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from families.minimax_h3 import reference_vsa as owner


SMALL_CASES = (
    dict(name="single-video-partner", segments=((1, 2, 2),), heads=1, gate_scale=0.0),
    dict(
        name="interleaved-video-guide",
        segments=(129, (5, 14, 18), 17, (9, 18, 34)),
        heads=2,
        gate_scale=0.25,
    ),
    dict(
        name="segment-pure-prefix",
        segments=(1, 127, 128, 129, (7, 16, 30)),
        heads=4,
        gate_scale=0.25,
    ),
    dict(name="trained-head-count", segments=(129, 96, 414, (9, 8, 16)), heads=56, gate_scale=0.25),
)
LARGE_CASE = dict(
    name="six-portrait-first-window",
    segments=(4000, 7296, 7296, 7296, 7296, 7296, 7296, 414, (37, 84, 48)),
    heads=56,
    gate_scale=0.25,
)
RTOL, ATOL = 0.016, 1e-5


def cases(backend):
    return SMALL_CASES + ((LARGE_CASE,) if backend == "blackwell" else ())


def describe():
    return [
        f"{backend}/{case['name']}"
        for backend in ("generic", "blackwell")
        for case in cases(backend)
    ]


def write(path, value):
    with path.open("x") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")


def build_engine(torch, trt, bridge, heads, max_tiles, output):
    from families.minimax_h3.dit_builder import _native_builder

    logger, builder, network, config = _native_builder(False, 256 << 20)
    operands = {
        n: network.add_input(n, trt.bfloat16, (1, heads, -1, 128)) for n in ("q", "k", "v", "gate")
    }
    sizes = network.add_input("sizes", trt.int32, (-1,))
    regions = network.add_input("regions", trt.int32, (-1, 3))
    result = owner.add_tiled_attention(
        network, operands, sizes, regions, plugin_library=bridge, heads=heads, max_tiles=max_tiles
    )
    result.name = "attention"
    network.mark_output(result)
    profile = builder.create_optimization_profile()
    for name in operands:
        profile.set_shape(
            name,
            (1, heads, 256, 128),
            (1, heads, min(16, max_tiles) * 128, 128),
            (1, heads, max_tiles * 128, 128),
        )
    profile.set_shape("sizes", (2,), (min(16, max_tiles),), (max_tiles,))
    profile.set_shape("regions", (1, 3), (1, 3), (2, 3))
    config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("OmniRef VSA component serialization failed")
    with output.open("xb") as file:
        file.write(serialized)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError("OmniRef VSA component deserialization failed")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("OmniRef VSA component context creation failed")
    return logger, runtime, engine, context


def reference(torch, operands, layout, upstream):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    logical_rows = layout.logical_tiles * 128
    sizes = torch.tensor(layout.block_sizes[: layout.logical_tiles].copy(), device="cuda")
    live = torch.arange(logical_rows, device="cuda") % 128 < sizes.repeat_interleave(128)
    q, k, v, gate = (operands[n][:, :, :logical_rows] for n in ("q", "k", "v", "gate"))
    pooled = [upstream._pool_tiles(x.transpose(1, 2), sizes, 128) for x in (q, k, v)]
    scores = torch.matmul(pooled[0], pooled[1].transpose(-2, -1)) / (128**0.5)
    mask = upstream._build_block_mask(
        scores,
        layout.prefix_tiles,
        0.9,
        True,
        tuple(tuple(map(int, r[:2])) for r in layout.regions),
        (0.9,) * len(layout.regions),
    )
    compressed = torch.matmul(torch.softmax(scores, dim=-1), pooled[2]).to(torch.bfloat16)
    result = torch.zeros_like(operands["q"])
    # Bound oracle workspace by one query tile. Preserve the full key domain,
    # independent regional mask and BF16 fine-attention output boundary.
    k32, v32 = k.float(), v.float()
    with sdpa_kernel(SDPBackend.MATH):
        for tile in range(layout.logical_tiles):
            start, end = tile * 128, (tile + 1) * 128
            key_mask = mask[:, :, tile : tile + 1].repeat_interleave(128, dim=-1) & live
            fine = torch.nn.functional.scaled_dot_product_attention(
                q[:, :, start:end].float(),
                k32,
                v32,
                attn_mask=key_mask,
            ).to(torch.bfloat16)
            result[:, :, start:end] = torch.addcmul(
                fine,
                compressed[:, :, tile : tile + 1],
                gate[:, :, start:end],
            )
    result[:, :, :logical_rows].masked_fill_(~live.view(1, 1, -1, 1), 0)
    return result, mask, compressed


def save_tensor(torch, path, value):
    array = value.detach().cpu().contiguous().view(torch.uint16).numpy()
    with path.open("xb") as output:
        np.save(output, array, allow_pickle=False)


def compare(torch, actual, expected):
    # Count every element; retain the original numeric limits at cancellation
    # points as well. No percentile, token exclusion or minimum-error floor.
    got, ref = actual.float(), expected.float()
    difference = (got - ref).abs()
    failed = difference > ATOL + RTOL * ref.abs()
    finite = bool(torch.isfinite(got).all() and torch.isfinite(ref).all())
    count = int(failed.sum())
    record = dict(
        elements=actual.numel(),
        failed_elements=count,
        finite=finite,
        max_absolute_error=float(difference.max()),
        rtol=RTOL,
        atol=ATOL,
    )
    if count:
        index = failed.nonzero()[0].tolist()
        record["first_failure"] = dict(
            index=index, actual=float(got[tuple(index)]), expected=float(ref[tuple(index)])
        )
    return record, finite and count == 0


def run(args):
    import torch
    import tensorrt as trt
    from .reference_vsa_geometry_oracle import upstream_math

    assert os.environ["TRTMC_MINIMAX_H3_VSA_BACKEND"] == args.backend
    assert torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() == (10, 3)
    assert importlib.metadata.version("torch") == "2.12.0+cu130"
    assert trt.__version__ == "11.2.0.113"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(8)
    torch.cuda.set_device(0)
    torch.manual_seed(1917)
    torch.cuda.reset_peak_memory_stats()
    upstream = upstream_math(args.base_source, args.h3_source)
    bridge = ctypes.CDLL(str(args.bridge), mode=ctypes.RTLD_GLOBAL)
    load = bridge.trtmc_load_byok_kernel
    load.argtypes = [ctypes.c_char_p] * 3
    load.restype = ctypes.c_char_p
    error = load(os.fsencode(args.library), b"run", owner.KERNEL_NAME.encode())
    if error is not None:
        raise RuntimeError(error.decode())
    args.output.mkdir(exist_ok=False)
    selected = cases(args.backend)
    layouts = {c["name"]: owner.make_layout(c["segments"]) for c in selected}
    max_tiles = {
        h: max(len(layouts[c["name"]].block_sizes) for c in selected if c["heads"] == h)
        for h in {c["heads"] for c in selected}
    }
    engines, results = {}, []
    enqueues = 0
    started = time.monotonic()
    try:
        for case in selected:
            output = args.output / case["name"]
            output.mkdir()
            layout, heads = layouts[case["name"]], case["heads"]
            write(
                output / "case.json",
                dict(
                    case, physical_tiles=len(layout.block_sizes), logical_tiles=layout.logical_tiles
                ),
            )
            np.savez(
                output / "geometry.npz",
                gather=layout.gather_indices,
                untile=layout.untile_indices,
                sizes=layout.block_sizes,
                regions=layout.regions,
            )
            shape = (1, heads, layout.padded_rows, 128)
            valid = torch.arange(layout.padded_rows, device="cuda") % 128 < torch.tensor(
                layout.block_sizes.copy(), device="cuda"
            ).repeat_interleave(128)
            operands = {}
            for name in ("q", "k", "v", "gate"):
                value = torch.randn(shape, device="cuda", dtype=torch.float32)
                if name == "gate":
                    value *= case["gate_scale"]
                value.masked_fill_(~valid.view(1, 1, -1, 1), 0)
                operands[name] = value.to(torch.bfloat16)
                save_tensor(torch, output / f"{name}.bf16.npy", operands[name])
            del value
            operands["sizes"] = torch.tensor(
                layout.block_sizes.copy(), device="cuda", dtype=torch.int32
            )
            operands["regions"] = torch.tensor(
                layout.regions.copy(), device="cuda", dtype=torch.int32
            )
            if heads not in engines:
                engines[heads] = build_engine(
                    torch,
                    trt,
                    args.bridge,
                    heads,
                    max_tiles[heads],
                    args.output / f"heads-{heads}.plan",
                )
            context = engines[heads][-1]
            for name, value in operands.items():
                assert context.set_input_shape(name, tuple(value.shape))
            assert not context.infer_shapes()
            assert tuple(context.get_tensor_shape("attention")) == shape
            actual = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
            for name, value in dict(operands, attention=actual).items():
                assert context.set_tensor_address(name, value.data_ptr())
            tick = time.monotonic()
            assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
            enqueues += 1
            torch.cuda.synchronize()
            execution_seconds = time.monotonic() - tick
            save_tensor(torch, output / "actual.bf16.npy", actual)
            expected, mask, compressed = reference(torch, operands, layout, upstream)
            save_tensor(torch, output / "expected.bf16.npy", expected)
            save_tensor(torch, output / "compression.bf16.npy", compressed)
            np.save(output / "reference-block-mask.npy", mask.cpu().numpy(), allow_pickle=False)
            comparison, passed = compare(torch, actual, expected)
            item = dict(
                name=f"{args.backend}/{case['name']}",
                passed=passed,
                comparison=comparison,
                TRT_execution_seconds=execution_seconds,
            )
            write(output / "comparison.json", item)
            results.append(item)
            write(
                args.output / f"progress-{len(results)}.json",
                dict(completed_cases=len(results), results=results),
            )
            progress = args.output / "progress.json"
            temp = args.output / "progress.pending.json"
            temp.write_text(json.dumps(dict(completed_cases=len(results))))
            os.replace(temp, progress)
            if not passed:
                raise AssertionError("OmniRef VSA differs from its unchanged BF16 numeric gate")
            del operands, actual, expected, mask, compressed
    except Exception as exc:
        write(
            args.output / "failure.json",
            dict(
                type=type(exc).__name__,
                message=str(exc),
                completed_cases=len(results),
                enqueues=enqueues,
            ),
        )
        raise
    finally:
        write(
            args.output / "report.json",
            dict(
                passed=len(results) == len(selected) and all(r["passed"] for r in results),
                cases=[f"{args.backend}/{c['name']}" for c in selected],
                results=results,
                TRT_VSA_enqueues=enqueues,
                rtol=RTOL,
                atol=ATOL,
                TensorRT_version=trt.__version__,
                GPU=torch.cuda.get_device_name(),
                compute_capability=list(torch.cuda.get_device_capability()),
                device_allocated_peak_bytes=torch.cuda.max_memory_allocated(),
                wall_seconds=time.monotonic() - started,
                checkpoint_loaded=False,
                full_transformer_forwards=0,
                trained_model_qualified=False,
                realtime_qualified=False,
                full_App_qualified=False,
            ),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("generic", "blackwell"))
    parser.add_argument("--library", type=Path)
    parser.add_argument("--bridge", type=Path)
    parser.add_argument("--base-source", type=Path)
    parser.add_argument("--h3-source", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        print(
            json.dumps(
                dict(
                    cases=describe(),
                    case_count=len(describe()),
                    GPU_model_calls=0,
                    CUDA_initialized=False,
                    Torch_imported="torch" in sys.modules,
                    TensorRT_version=importlib.metadata.version("tensorrt"),
                    rtol=RTOL,
                    atol=ATOL,
                )
            )
        )
        return
    if any(
        getattr(args, n) is None
        for n in ("backend", "library", "bridge", "base_source", "h3_source")
    ):
        parser.error(
            "Real execution requires the explicit backend, libraries and pinned oracle sources"
        )
    run(args)


if __name__ == "__main__":
    main()
