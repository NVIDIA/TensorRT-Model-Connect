# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare OmniRef geometry with explicitly supplied, pinned upstream math."""

import argparse
import ast
import functools
import json
import math
from pathlib import Path
import types

import numpy as np
import torch

from families.minimax_h3 import reference_vsa as candidate


CASES = (
    ((1, 2, 2),),
    (1, (1, 2, 2)),
    (129, (5, 14, 18), 17, (9, 18, 34)),
    (0, 127, 128, 129, (7, 16, 30)),
    ((3, 10, 18), 17, (6, 14, 22), 257, (9, 18, 34)),
    (4000, 7296, 7296, 7296, 7296, 7296, 7296, 754, (37, 84, 48)),
    (4000, 7296, (6, 84, 48), 377, 7296, 754, (37, 84, 48)),
)


def upstream_math(base_path, h3_path):
    module = types.SimpleNamespace(
        torch=torch,
        math=math,
        functools=functools,
        VSA_TILE_SIZE=(4, 4, 4),
        VSA_H3_TILE_SIZE=(4, 8, 8),
        _TILE_ELEMS=256,
    )
    selected = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for path, names in (
        (
            base_path,
            {
                "get_tile_partition_indices",
                "construct_variable_block_sizes",
                "get_non_pad_index",
                "compute_topk",
            },
        ),
        (
            h3_path,
            {
                "_validate_h3_segment_geometry",
                "_h3_segment_tile_geometry",
                "_pool_tiles",
                "_build_block_mask",
            },
        ),
    ):
        functions = [
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name in names
        ]
        if {n.name for n in functions} != names:
            raise ValueError("Pinned upstream VSA mathematical definitions are incomplete")
        selected.extend(functions)
    extracted = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(extracted, str(h3_path), "exec"), vars(module))
    return module


def run(base_path, h3_path):
    upstream = upstream_math(base_path, h3_path)
    results = []
    for segments in CASES:
        got = candidate.make_layout(segments)
        token_segments = tuple(
            (s[0], s[1] // 2, s[2] // 2) if isinstance(s, tuple) else s
            for s in segments
            if isinstance(s, tuple) or s > 0
        )
        expected = upstream._h3_segment_tile_geometry(
            token_segments, torch.device("cpu"), candidate.TILE_SHAPE
        )
        permutation, sizes, untile, prefix, videos, spans = expected
        live = got.gather_indices[got.gather_indices != got.source_rows]
        np.testing.assert_array_equal(live, permutation.numpy())
        np.testing.assert_array_equal(got.block_sizes[: got.logical_tiles], sizes.numpy())
        np.testing.assert_array_equal(got.untile_indices, untile.numpy())
        np.testing.assert_array_equal(got.regions[:, :2], np.asarray(spans))
        np.testing.assert_array_equal(
            got.regions[:, 2],
            [upstream.compute_topk(0.9, end - start) for start, end in spans],
        )
        assert got.prefix_tiles == prefix and got.logical_tiles == prefix + videos
        assert got.block_sizes.size % 2 == 0
        assert got.block_sizes.sum() == got.source_rows
        assert got.block_sizes.size - got.logical_tiles in (0, 1)
        if got.logical_tiles % 2:
            assert got.block_sizes[-1] == 0
        rows = np.arange(got.source_rows, dtype=np.int32)
        extended = np.concatenate([rows, np.asarray([-1], dtype=np.int32)])
        np.testing.assert_array_equal(extended[got.gather_indices][got.untile_indices], rows)
        results.append(
            {
                "segments": segments,
                "source_rows": got.source_rows,
                "prefix_tiles": got.prefix_tiles,
                "logical_tiles": got.logical_tiles,
                "physical_tiles": int(got.block_sizes.size),
                "regions": got.regions.tolist(),
                "passed": True,
            }
        )
    return {
        "passed": True,
        "cases": results,
        "case_count": len(results),
        "GPU_calls": 0,
        "VSA_kernel_qualified": False,
        "trained_model_qualified": False,
        "realtime_qualified": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-source", type=Path, required=True)
    parser.add_argument("--h3-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = run(args.base_source, args.h3_source)
    with args.output.open("x") as f:
        json.dump(value, f, indent=2)
    print(json.dumps({k: v for k, v in value.items() if k != "cases"}))


if __name__ == "__main__":
    main()
