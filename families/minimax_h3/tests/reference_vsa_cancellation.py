# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recombine saved BF16 boundaries to distinguish error propagation from FMA.

BF16 products and cancelling sums fit in FP64. The explicit FP32 rounding
before the final BF16 conversion reproduces the native fused boundary for
these operands; bitwise checks against both saved outputs verify that claim.
The separate probability experiment remains an approximate CPU simulation.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .reference_vsa_probability import as_float, bf16, emulated_exp2, observation, simulate
from .reference_vsa_gpu_oracle import ATOL, RTOL


def fused(fine, compression, gate):
    return bf16(fine.astype(np.float64) + compression.astype(np.float64) * gate.astype(np.float64))


def bitwise_equal(left, right):
    return bool(np.array_equal(left.view(np.uint32), right.view(np.uint32)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    assert manifest["case"] == "interleaved-video-guide"
    for name, row in manifest["files"].items():
        assert Path(name).name == name
        path = args.input_dir / name
        assert path.is_file() and not path.is_symlink() and path.stat().st_size == row["bytes"]
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == row["sha256"]
    assert not args.output.exists()
    arrays = {
        name: np.load(args.input_dir / (name + ".bf16.npy"), allow_pickle=False, mmap_mode="r")
        for name in ("q", "k", "v", "gate", "actual", "expected", "fine-native", "fine-reference")
    }
    assert all(x.shape == (1, 2, 4864, 128) for x in arrays.values())
    gate, actual, expected, native_fine, reference_fine = (
        as_float(arrays[n]) for n in ("gate", "actual", "expected", "fine-native", "fine-reference")
    )
    with np.load(args.input_dir / "native-taps.npz", allow_pickle=False) as taps:
        native_compressed = bf16(taps["compression"].copy())
        indices, counts = taps["route_indices"].copy(), taps["route_counts"].copy()
    reference_compressed = as_float(
        np.load(args.input_dir / "compression.bf16.npy", allow_pickle=False)
    )
    native_comp = np.repeat(native_compressed, 128, axis=2)
    reference_comp = np.repeat(reference_compressed, 128, axis=2)
    outputs = dict(
        native_recombined=fused(native_fine, native_comp, gate),
        reference_recombined=fused(reference_fine, reference_comp, gate),
        native_fine_reference_compression=fused(native_fine, reference_comp, gate),
        reference_fine_native_compression=fused(reference_fine, native_comp, gate),
    )
    failed = np.abs(actual - expected) > ATOL + RTOL * np.abs(expected)
    failed_indices = np.argwhere(failed)
    assert len(failed_indices) == manifest["candidate_failed_elements"] == 253
    query_pairs = sorted({(int(i[1]), int(i[2])) for i in failed_indices})
    with np.load(args.input_dir / "geometry.npz", allow_pickle=False) as geometry:
        sizes = geometry["sizes"].copy()
    projections, references = [], []
    for head, row in query_pairs:
        tile = row // 128
        keys = indices[head, tile, : int(counts[head, tile])]
        operands = (
            as_float(arrays["q"][0, head, row]),
            arrays["k"][0, head],
            arrays["v"][0, head],
            sizes,
            keys,
        )
        prediction = np.stack(
            [simulate(*operands, emulate_exponential=emulated) for emulated in (True, False)]
        )
        projections.append(
            fused(prediction, reference_compressed[0, head, tile], gate[0, head, row])
        )
        references.append(expected[0, head, row])
    projections, references = np.asarray(projections), np.asarray(references)
    names = ("one_bf16_limb", "two_bf16_limbs", "unrounded_probability")
    fractions = np.linspace(0, 1, 10001, dtype=np.float32)
    relative_error = np.abs(
        emulated_exp2(fractions).astype(np.float64) / np.exp2(fractions.astype(np.float64)) - 1
    )
    first = tuple(failed_indices[0])
    report = dict(
        status="CPU-counterfactual-complete",
        original_rtol=RTOL,
        original_atol=ATOL,
        original_gated_comparison=observation(actual, expected),
        native_FMA_bitwise_reproduced=bitwise_equal(outputs["native_recombined"], actual),
        reference_FMA_bitwise_reproduced=bitwise_equal(outputs["reference_recombined"], expected),
        counterfactuals={name: observation(value, expected) for name, value in outputs.items()},
        fine_output_differs_at_gated_failures=int(
            np.count_nonzero(native_fine[failed] != reference_fine[failed])
        ),
        first_failure=dict(
            index=list(map(int, first)),
            actual=float(actual[first]),
            expected=float(expected[first]),
            native_fine=float(native_fine[first]),
            reference_fine=float(reference_fine[first]),
            gate=float(gate[first]),
            native_compression=float(native_comp[first]),
        ),
        CPU_projected_failing_queries=len(query_pairs),
        CPU_projected_gated_comparisons={
            f"{exponential}/{name}": observation(projections[:, e, i], references)
            for e, exponential in enumerate(
                ("cubic_exponential", "hardware_exponential_approximation")
            )
            for i, name in enumerate(names)
        },
        cubic_exponential_maximum_relative_error=float(relative_error.max()),
        CPU_simulation_limitations="Only queries containing current failures; FP64 dot products/accumulators and approximate native exp2/FMA/reductions. This does not establish CUDA correctness or acceptance.",
        original_case_still_failed=True,
        trained_model_qualified=False,
        GPU_calls=0,
    )
    args.output.mkdir()
    np.savez(args.output / "counterfactual-outputs.npz", **outputs, failed_indices=failed_indices)
    np.savez(
        args.output / "projected-queries.npz",
        queries=np.asarray(query_pairs),
        simulated=projections,
        reference=references,
    )
    with (args.output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
