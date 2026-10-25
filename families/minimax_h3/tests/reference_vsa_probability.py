# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observe probability-rounding error on immutable saved attention operands.

This CPU experiment approximates the native blockwise softmax and uses FP64
dot products/accumulators to isolate probability storage. It is not a CUDA
instruction emulator or a replacement for the unchanged GPU oracle.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .reference_vsa_gpu_oracle import ATOL, RTOL


def as_float(array):
    assert array.dtype == np.uint16
    return (array.astype(np.uint32) << 16).view(np.float32)


def bf16(value):
    bits = np.asarray(value, dtype=np.float32).view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def emulated_exp2(value):
    """Mirror the released kernel's cubic approximation with FP32 FMA steps."""
    value = np.maximum(np.asarray(value, dtype=np.float32), np.float32(-127))
    integer = np.floor(value).astype(np.int32)
    fraction = (value - integer.astype(np.float32)).astype(np.float32)
    c3, c2, c1 = np.asarray([0x3D9DF09D, 0x3E6906A4, 0x3F31F519], dtype=np.uint32).view(np.float32)

    def fma(left, right, add):
        return (left.astype(np.float64) * right + add).astype(np.float32)

    polynomial = fma(fraction, float(c3), float(c2))
    polynomial = fma(polynomial, fraction.astype(np.float64), float(c1))
    polynomial = fma(polynomial, fraction.astype(np.float64), 1.0)
    return np.ldexp(polynomial, integer).astype(np.float32)


def simulate(q, k, v, sizes, keys, *, emulate_exponential=False, include_third=False):
    scale = np.float32(1 / math.sqrt(128))
    scale_log2 = np.float32(scale * np.float32(math.log2(math.e)))
    maximum, denominator = None, 0.0
    accumulators = np.zeros((4 if include_third else 3, 128), dtype=np.float64)
    for tile in keys:
        start = int(tile) * 128
        stop = start + int(sizes[tile])
        if stop == start:
            continue
        raw = (as_float(k[start:stop]).astype(np.float64) @ q.astype(np.float64)).astype(np.float32)
        value = as_float(v[start:stop]).astype(np.float64)
        new_maximum = np.float32(raw.max())
        alpha = np.float32(0)
        if maximum is not None:
            new_maximum = max(new_maximum, maximum)
            shift = np.float32(np.float32(maximum - new_maximum) * scale_log2)
            if shift >= -8:
                new_maximum, alpha = maximum, np.float32(1)
            else:
                alpha = np.float32(np.exp2(shift))
        # One FP32 rounding approximates the kernel's FMA before exp2. Its
        # hardware exponential approximation and GEMM reduction differ from CPU.
        exponent = (
            raw.astype(np.float64) * float(scale_log2)
            + float(np.float32(-new_maximum * scale_log2))
        ).astype(np.float32)
        probability = np.exp2(exponent).astype(np.float32)
        if emulate_exponential:
            pair = np.arange(len(probability)) // 2
            emulated = (pair % 8 >= 6) & (pair < 48)
            probability[emulated] = emulated_exp2(exponent[emulated])
        high = bf16(probability)
        low = bf16(probability - high)
        accumulators *= float(alpha)
        accumulators[0] += high.astype(np.float64) @ value
        accumulators[1] += (high.astype(np.float64) + low.astype(np.float64)) @ value
        if include_third:
            tail = bf16((probability - high) - low)
            accumulators[2] += (
                high.astype(np.float64) + low.astype(np.float64) + tail.astype(np.float64)
            ) @ value
        accumulators[-1] += probability.astype(np.float64) @ value
        denominator = denominator * float(alpha) + probability.astype(np.float64).sum()
        maximum = new_maximum
    assert denominator > 0
    return bf16(accumulators / denominator)


def observation(actual, expected):
    difference = np.abs(actual - expected)
    return dict(
        elements=int(actual.size),
        unequal_elements=int(np.count_nonzero(actual != expected)),
        elements_outside_original_tolerance=int(
            np.count_nonzero(difference > ATOL + RTOL * np.abs(expected))
        ),
        maximum_absolute_error=float(difference.max()),
        mean_absolute_error=float(difference.mean()),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    assert manifest["case"] == "interleaved-video-guide"
    for name, record in manifest["files"].items():
        assert Path(name).name == name
        path = args.input_dir / name
        assert path.is_file() and path.stat().st_size == record["bytes"]
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == record["sha256"]
    assert not args.output.exists()
    arrays = {
        name: np.load(args.input_dir / (name + ".bf16.npy"), allow_pickle=False, mmap_mode="r")
        for name in ("q", "k", "v", "fine-native", "fine-reference")
    }
    assert all(x.shape == (1, 2, 4864, 128) for x in arrays.values())
    with np.load(args.input_dir / "geometry.npz", allow_pickle=False) as geometry:
        sizes = geometry["sizes"].copy()
    with np.load(args.input_dir / "native-taps.npz", allow_pickle=False) as taps:
        indices, counts = taps["route_indices"].copy(), taps["route_counts"].copy()
    valid = np.flatnonzero(np.arange(4864) % 128 < np.repeat(sizes, 128))
    rows = sorted(set(valid[np.linspace(0, len(valid) - 1, 64).astype(int)].tolist()) | {0})
    samples, native, reference, simulated = [], [], [], []
    for head in range(2):
        for row in rows:
            tile = row // 128
            keys = indices[head, tile, : int(counts[head, tile])]
            assert bool(((keys >= 0) & (keys < len(sizes))).all())
            simulated.append(
                simulate(
                    as_float(arrays["q"][0, head, row]),
                    arrays["k"][0, head],
                    arrays["v"][0, head],
                    sizes,
                    keys,
                )
            )
            samples.append((head, row))
            native.append(as_float(arrays["fine-native"][0, head, row]))
            reference.append(as_float(arrays["fine-reference"][0, head, row]))
    native, reference, simulated = map(np.asarray, (native, reference, simulated))
    names = ("one_bf16_limb", "two_bf16_limbs", "unrounded_probability")
    report = dict(
        status="CPU-observation-complete",
        sampled_queries=len(samples),
        original_rtol=RTOL,
        original_atol=ATOL,
        native_vs_reference=observation(native, reference),
        simulation_vs_reference={
            name: observation(simulated[:, i], reference) for i, name in enumerate(names)
        },
        simulation_vs_native={
            name: observation(simulated[:, i], native) for i, name in enumerate(names)
        },
        limitations="FP64 CPU dot products/accumulators, approximate native exp2/FMA and denominator reduction; sampled operands only. No CUDA bitwise parity or full-case acceptance claim.",
        original_case_still_failed=True,
        trained_model_qualified=False,
        GPU_calls=0,
    )
    args.output.mkdir()
    np.savez(
        args.output / "sampled-outputs.npz",
        queries=np.asarray(samples),
        native=native,
        reference=reference,
        simulated=simulated,
    )
    with (args.output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
