# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score, export a prefix snapshot, restore it, and repeat scoring without prefill."""

import argparse
from dataclasses import replace
from pathlib import Path
import numpy as np
from families.hstu_research.config import Config, ServingProfile
from families.hstu_research.reference import example_inputs
from families.hstu_research.triton.native_client import NativeClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, help="Optional NPZ containing application inputs")
    args = parser.parse_args()
    with NativeClient(
        args.bundle,
        args.runtime_root,
        hydration=True,
        kv_cache_size_bytes=134217728,
        max_entries=8,
        lease_seconds=0,
    ) as task:
        config = replace(
            Config.load(args.checkpoint), serving=ServingProfile(**task.metadata["serving"])
        )
        batch = task.metadata["min_batch_size"]
        if args.inputs:
            with np.load(args.inputs, allow_pickle=False) as archive:
                inputs = {name: archive[name].copy() for name in archive.files}
            batch = inputs["history_ids"].shape[0]
        else:
            inputs = example_inputs(config, batch)
        history = {s["name"]: inputs[s["name"]] for s in task.metadata["stages"][0]["inputs"]}
        request = {s["name"]: inputs[s["name"]] for s in task.metadata["stages"][1]["inputs"]}
        cold = task.score(inputs)
        snapshot = task.export_history(history, ["history-v1"] * batch)
        handles = task.restore_history(snapshot)
        cached = task.score_with_history(handles, ["history-v1"] * batch, request)
        for name in cold:
            np.testing.assert_array_equal(cold[name], cached[name])
        print("Selected candidate positions:", cached["picked_indices"].tolist())
        print("Cold and restored scores are bit identical.")
        print("Cache:", task.stats())
        task.release_history(handles)


if __name__ == "__main__":
    main()
