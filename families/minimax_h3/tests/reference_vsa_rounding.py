# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observe saved fine-attention rounding without changing the GPU oracle.

The current fine output is inferred conditionally from the saved gated output
and unchanged compression. FP64 CPU attention and approximate FP32 operation
orders distinguish a rounding boundary from a transport/layout failure; they
are neither CUDA emulators nor replacement acceptance references.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .reference_vsa_gpu_oracle import ATOL, RTOL
from .reference_vsa_probability import as_float, bf16


TORCH_COMMIT = "7661cd9c6b841b62b7f411aa52ec51f05457263b"
TORCH_MATH_SOURCE = (
    "https://github.com/pytorch/pytorch/blob/" + TORCH_COMMIT
    + "/aten/src/ATen/native/transformers/attention.cpp#L821-L870"
)


def nearest_bf16(value):
    """Round a finite scalar directly, retaining FP64 evidence at FP32 ties."""
    if not math.isfinite(value) or abs(value) > float(as_float(np.uint16(0x7F7F))):
        raise ValueError("The rounding observation requires a finite BF16-range value")
    center = int(np.float32(value).view(np.uint32) >> 16)
    codes = np.arange(max(0, center - 2), min(65536, center + 3), dtype=np.uint16)
    values = as_float(codes).astype(np.float64)
    finite = np.isfinite(values)
    distance = np.abs(values - value)
    minimum = np.min(distance[finite])
    matches = np.flatnonzero(finite & (distance == minimum))
    # Choose the even significand for an exact halfway input, including negatives.
    selected = min(matches, key=lambda index: int(codes[index]) & 1)
    return float(values[selected])


def weighted_value(logits, values, dtype):
    logits = np.asarray(logits, dtype=dtype)
    probability = np.exp(logits - logits.max()).astype(dtype)
    probability = (probability / probability.sum(dtype=dtype)).astype(dtype)
    # Keep PV accumulation in FP64 to isolate scaling/softmax order here.
    return math.fsum((probability.astype(np.float64) * values).tolist())


def combine(fine, compression, gate):
    return float(bf16(np.float64(fine) + np.float64(compression) * np.float64(gate)))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def observe(directory, manifest):
    assert manifest["case"] == "interleaved-video-guide"
    for name, record in manifest["files"].items():
        path = directory / name
        assert Path(name).name == name and path.is_file() and not path.is_symlink()
        assert path.stat().st_size == record["bytes"] and digest(path) == record["sha256"]
    raw = {
        name: np.load(directory / (name + ".bf16.npy"), allow_pickle=False)
        for name in ("q", "k", "v", "gate", "actual", "expected", "fine-reference", "compression")
    }
    arrays = {name: as_float(value) for name, value in raw.items()}
    actual, expected = arrays["actual"], arrays["expected"]
    failed = np.argwhere(np.abs(actual - expected) > ATOL + RTOL * np.abs(expected))
    assert len(failed) == manifest["candidate_failed_elements"]
    with np.load(directory / "native-taps.npz", allow_pickle=False) as data:
        indices, counts = data["route_indices"].copy(), data["route_counts"].copy()
        previous_compression = bf16(data["compression"].copy())
    with np.load(directory / "geometry.npz", allow_pickle=False) as data:
        sizes = data["sizes"].copy()
    mask = np.load(directory / "reference-block-mask.npy", allow_pickle=False)
    scale64 = 1 / math.sqrt(128)
    half_scale32 = np.float32(math.sqrt(scale64))
    points = []
    for index in failed:
        key = tuple(index)
        _, head, row, dimension = map(int, index)
        tile = row // 128
        tiles = indices[head, tile, : int(counts[head, tile])]
        assert set(map(int, tiles)) == set(map(int, np.flatnonzero(mask[0, head, tile])))
        selected = np.concatenate([
            np.arange(int(t) * 128, int(t) * 128 + int(sizes[t])) for t in tiles
        ])
        q = arrays["q"][0, head, row].astype(np.float64)
        k = arrays["k"][0, head, selected].astype(np.float64)
        v = arrays["v"][0, head, selected, dimension].astype(np.float64)
        dot64 = k @ q
        fine64 = weighted_value(dot64 * scale64, v, np.float64)
        scaled_dot32 = (dot64.astype(np.float32) * np.float32(scale64)).astype(np.float32)
        prescaled_dot32 = (
            (k.astype(np.float32) * half_scale32).astype(np.float64)
            @ (q.astype(np.float32) * half_scale32).astype(np.float64)
        ).astype(np.float32)
        variants = {
            "FP64_attention": fine64,
            "postscaled_FP32_logits_FP64_softmax_PV": weighted_value(scaled_dot32, v, np.float64),
            "prescaled_FP32_inputs_logits_FP64_softmax_PV": weighted_value(prescaled_dot32, v, np.float64),
            "prescaled_FP32_inputs_logits_softmax_FP64_PV": weighted_value(prescaled_dot32, v, np.float32),
        }
        compression = float(arrays["compression"][0, head, tile, dimension])
        gate = float(arrays["gate"][key])
        assert previous_compression[0, head, tile, dimension] == compression
        reference_fine = float(arrays["fine-reference"][key])
        assert combine(reference_fine, compression, gate) == float(expected[key])
        bits = int(raw["fine-reference"][key])
        neighbors = as_float(np.arange(max(0, bits - 4), min(65536, bits + 5), dtype=np.uint16))
        inferred = [float(value) for value in neighbors if combine(value, compression, gate) == float(actual[key])]
        assert len(inferred) == 1, "Retain ambiguity rather than choosing a convenient fine value"
        native_fine = inferred[0]
        midpoint = (native_fine + reference_fine) / 2
        points.append(dict(
            index=list(map(int, index)), selected_key_rows=len(selected),
            actual=float(actual[key]), expected=float(expected[key]), gate=gate, compression=compression,
            native_fine_inferred=native_fine, reference_fine=reference_fine,
            FP64_fine=fine64, FP64_direct_BF16=nearest_bf16(fine64),
            FP64_signed_distance_to_neighbor_midpoint=fine64 - midpoint,
            native_matches_FP64_BF16=native_fine == nearest_bf16(fine64),
            reference_matches_FP64_BF16=reference_fine == nearest_bf16(fine64),
            operation_order_observations={name: dict(
                fine=value, direct_BF16=nearest_bf16(value),
                gated=combine(nearest_bf16(value), compression, gate),
            ) for name, value in variants.items()},
        ))
    return dict(
        format="minimax_h3.vsa_rounding_observation/v1", status="CPU-observation-complete",
        input_case=manifest["case"], actual_failed_elements=len(failed),
        original_rtol=RTOL, original_atol=ATOL, points=points,
        native_fine_matches_FP64=sum(p["native_matches_FP64_BF16"] for p in points),
        reference_fine_matches_FP64=sum(p["reference_matches_FP64_BF16"] for p in points),
        reference_math_source=TORCH_MATH_SOURCE,
        reference_math_source_observation="The installed math SDPA pre-scales Q and K separately before matmul; the optimized native kernel scales its QK scores.",
        limitations="Saved failing coordinates only. Native fine values are inferred conditionally from unchanged compression and the released FMA boundary, not directly measured. FP64 CPU arithmetic and approximate FP32 scaling/softmax orders do not emulate CUDA accumulation/reductions or establish GPU acceptance. No reference, operand, mask, precision boundary or passing threshold was changed.",
        original_case_still_failed=True, trained_model_qualified=False, GPU_calls=0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = observe(args.input_dir, json.loads(args.manifest.read_text()))
    report["input_manifest_sha256"] = digest(args.manifest)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({key: value for key, value in report.items() if key != "points"}))


if __name__ == "__main__":
    main()
