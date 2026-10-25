# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay saved VSA operands and observe native workspace/fine-attention boundaries.

The original gated result retains its existing reference and tolerances. A
second call with a zero gate exposes fine attention for localization only;
it cannot satisfy the original case or establish model qualification.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import numpy as np

from . import reference_vsa_gpu_oracle as oracle


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def validate(args):
    manifest = read(args.manifest)
    assert manifest["backend"] == args.backend == "blackwell"
    assert manifest["rtol"] == oracle.RTOL == 0.016
    assert manifest["atol"] == oracle.ATOL == 1e-5
    assert args.case_dir.is_relative_to("/runs/results")
    for name, record in manifest["inputs"].items():
        assert Path(name).name == name
        path = args.case_dir / name
        assert path.is_file() and not path.is_symlink()
        assert path.stat().st_size == record["bytes"] and sha(path) == record["sha256"]
    assert sha(args.library) == manifest["kernel_sha256"]
    for path in (args.base_source, args.h3_source):
        assert sha(path) == manifest["upstream_sources"][path.name]["sha256"]
    case = read(args.case_dir / "case.json")
    assert case["name"] == "interleaved-video-guide" and case["heads"] == 2
    segments = tuple(tuple(x) if isinstance(x, list) else x for x in case["segments"])
    layout = oracle.owner.make_layout(segments)
    assert len(layout.block_sizes) == case["physical_tiles"]
    assert layout.logical_tiles == case["logical_tiles"]
    assert not args.output.exists()
    return manifest, case, layout


def run(args, manifest, case, layout):
    import torch
    import tvm_ffi

    from .reference_vsa_geometry_oracle import upstream_math

    assert os.environ["TRTMC_MINIMAX_H3_VSA_BACKEND"] == args.backend
    assert importlib.metadata.version("apache-tvm-ffi") == "0.1.12"
    assert importlib.metadata.version("torch") == "2.12.0+cu130"
    assert torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() == (10, 3)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(8)
    upstream = upstream_math(args.base_source, args.h3_source)
    args.output.mkdir()

    def load_bf16(name):
        array = np.load(args.case_dir / name, allow_pickle=False)
        assert array.dtype == np.uint16
        return torch.from_numpy(array.copy()).view(torch.bfloat16).cuda()

    operands = {name: load_bf16(name + ".bf16.npy") for name in ("q", "k", "v", "gate")}
    shape = (1, case["heads"], layout.padded_rows, 128)
    assert all(tuple(value.shape) == shape for value in operands.values())
    operands["sizes"] = torch.tensor(layout.block_sizes.copy(), device="cuda", dtype=torch.int32)
    operands["regions"] = torch.tensor(layout.regions.copy(), device="cuda", dtype=torch.int32)
    workspace = torch.zeros(
        oracle.owner.workspace_bytes(case["heads"], len(layout.block_sizes)),
        device="cuda",
        dtype=torch.uint8,
    )
    module = tvm_ffi.load_module(args.library)

    def invoke(gate):
        output = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
        values = [operands[name] for name in ("q", "k", "v")]
        values += [gate, operands["sizes"], operands["regions"], workspace, output]
        wrapped = [tvm_ffi.from_dlpack(value, require_contiguous=True) for value in values]
        torch.cuda.synchronize()
        module.run(*wrapped)
        torch.cuda.synchronize()
        return output

    native = invoke(operands["gate"])
    oracle.save_tensor(torch, args.output / "native-direct.bf16.npy", native)
    stored = load_bf16("actual.bf16.npy")
    replay_equal = np.array_equal(
        native.cpu().view(torch.uint16).numpy(), stored.cpu().view(torch.uint16).numpy()
    )
    expected = load_bf16("expected.bf16.npy")
    full_comparison, full_passed = oracle.compare(torch, native, expected)
    # The kernel reuses Q-pool storage for its FP32 compressed output.
    # K-pool, V-pool, softmax scores and routing remain available in workspace.
    saved = workspace.cpu().numpy().copy()
    np.save(args.output / "workspace.uint8.npy", saved, allow_pickle=False)
    heads, tiles = case["heads"], len(layout.block_sizes)
    pools, scores = heads * tiles * 128, heads * tiles * tiles
    compressed = np.frombuffer(saved, dtype=np.float32, count=pools).reshape(1, heads, tiles, 128)
    key_pool = np.frombuffer(saved, dtype=np.float32, count=pools, offset=pools * 4)
    value_pool = np.frombuffer(saved, dtype=np.float32, count=pools, offset=pools * 8)
    softmax = np.frombuffer(saved, dtype=np.float32, count=scores, offset=pools * 12)
    indices = np.frombuffer(saved, dtype=np.int32, count=scores, offset=(3 * pools + scores) * 4)
    counts = np.frombuffer(
        saved, dtype=np.int32, count=heads * tiles, offset=(3 * pools + 2 * scores) * 4
    )
    indices = indices.reshape(heads, tiles, tiles)
    counts = counts.reshape(heads, tiles)
    logical = layout.logical_tiles
    mask = np.zeros((1, heads, logical, logical), dtype=bool)
    for head in range(heads):
        for tile in range(logical):
            count = int(counts[head, tile])
            assert 0 <= count <= tiles
            keys = indices[head, tile, :count]
            assert bool(((keys >= 0) & (keys < logical)).all())
            mask[0, head, tile, keys] = True
    reference_mask = np.load(args.case_dir / "reference-block-mask.npy", allow_pickle=False)
    assert mask.shape == reference_mask.shape
    routing_equal = bool(np.array_equal(mask, reference_mask))
    np.savez(
        args.output / "native-taps.npz",
        compression=compressed,
        key_pool=key_pool.reshape(1, heads, tiles, 128),
        value_pool=value_pool.reshape(1, heads, tiles, 128),
        softmax=softmax.reshape(1, heads, tiles, tiles),
        route_indices=indices,
        route_counts=counts,
        route_mask=mask,
    )
    compression_native = torch.from_numpy(compressed[:, :, :logical].copy()).cuda().bfloat16()
    compression_reference = load_bf16("compression.bf16.npy")
    compression_comparison, compression_passed = oracle.compare(
        torch, compression_native, compression_reference
    )
    zero_gate = torch.zeros_like(operands["gate"])
    fine_native = invoke(zero_gate)
    fine_reference, _, _ = oracle.reference(
        torch, {**operands, "gate": zero_gate}, layout, upstream
    )
    fine_comparison, fine_passed = oracle.compare(torch, fine_native, fine_reference)
    oracle.save_tensor(torch, args.output / "fine-native.bf16.npy", fine_native)
    oracle.save_tensor(torch, args.output / "fine-reference.bf16.npy", fine_reference)
    if not replay_equal:
        boundary = "TRT-versus-direct-binding-unresolved"
    elif not routing_equal:
        boundary = "routing"
    elif not compression_passed:
        boundary = "pooling-or-compression"
    elif not fine_passed:
        boundary = "fine-attention"
    elif not full_passed:
        boundary = "gated-combination"
    else:
        boundary = "not-reproduced"
    result = dict(
        status="diagnostic-complete",
        source_campaign=manifest["source_campaign"],
        original_case_still_failed=not full_passed,
        stored_TRT_output_bitwise_reproduced=replay_equal,
        routing_mask_bitwise_equal=routing_equal,
        full_original_gate_comparison=full_comparison,
        compression_comparison=compression_comparison,
        fine_attention_comparison=fine_comparison,
        first_divergent_boundary=boundary,
        zero_gate_used_for_localization_only=True,
        original_QKV_gate_and_reference_files_unchanged=all(
            sha(args.case_dir / name) == row["sha256"] for name, row in manifest["inputs"].items()
        ),
        kernel_sha256=sha(args.library),
        kernel_changed=False,
        thresholds_changed=False,
        native_FFI_calls=2,
        full_transformer_forwards=0,
        trained_model_qualified=False,
        full_App_qualified=False,
    )
    oracle.write(args.output / "report.json", result)
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--base-source", type=Path, required=True)
    parser.add_argument("--h3-source", type=Path, required=True)
    parser.add_argument("--backend", choices=("blackwell",), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    manifest, case, layout = validate(args)
    if args.preflight:
        assert "torch" not in sys.modules
        print(
            json.dumps(
                dict(
                    cases=["blackwell/interleaved-video-guide/workspace-and-fine-taps"],
                    case_count=1,
                    rtol=oracle.RTOL,
                    atol=oracle.ATOL,
                    GPU_calls=0,
                    Torch_imported=False,
                    original_case_source=manifest["source_campaign"],
                    diagnostic_only=True,
                )
            )
        )
    else:
        run(args, manifest, case, layout)


if __name__ == "__main__":
    main()
