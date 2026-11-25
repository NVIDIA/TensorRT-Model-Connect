# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build all four public HSTU variants and prepare the Triton test repository."""

import argparse
from dataclasses import replace
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from tensorrt_model_connect import BuildRequest, build
from families.hstu.checkpoint import create_fixture
from families.hstu.config import Config
from families.hstu.reference import example_inputs
from families.hstu.triton.native_repository import generate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--runtime-root", type=Path, required=True)
    p.add_argument("--bf16-policy", choices=("strict", "fast"))
    p.add_argument("--cuda-graphs", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--batching", choices=("dynamic", "fixed"))
    p.add_argument("--max-batch-size", type=int, default=4)
    p.add_argument("--bf16-attention-epilogue", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument(
        "--checkpoint", type=Path, help="Public checkpoint; omission creates an untrained fixture"
    )
    a = p.parse_args()
    if a.max_batch_size < 1:
        p.error("--max-batch-size must be positive")
    root = a.root.resolve()
    sdk = a.runtime_root.resolve()
    if (root / "output").exists() or (root / "triton-models").exists():
        raise FileExistsError("qualification output/repository must be fresh")
    if a.checkpoint:
        Config.load(a.checkpoint)
    root.mkdir(parents=True, exist_ok=True)
    for candidates in (50, 64):
        for precision in ("fp32", "bf16"):
            name = f"hstu{candidates}-{precision}"
            out = root / "output" / name
            if a.checkpoint:
                out.mkdir(parents=True)
                for filename in ("hstu_weights.npz", "checkpoint_provenance.json"):
                    source = a.checkpoint / filename
                    if source.exists():
                        shutil.copyfile(source, out / filename)
                c = replace(Config.load(a.checkpoint), candidates=candidates)
                c.save(out)
            else:
                c = create_fixture(out, candidates)
            overrides = {}
            if a.bf16_policy is not None:
                overrides["bf16_policy"] = a.bf16_policy
            if a.cuda_graphs is not None:
                overrides["cuda_graphs"] = a.cuda_graphs
            if a.batching is not None:
                overrides["batching"] = a.batching
            if a.bf16_attention_epilogue is not None:
                overrides["bf16_attention_epilogue"] = a.bf16_attention_epilogue
            c = replace(c, **overrides)
            c.save(out)
            build(
                BuildRequest(
                    model_dir=out,
                    output_path=out / "model.bundle",
                    family="hstu",
                    task="structured_ranking",
                    precision=precision,
                    max_batch_size=a.max_batch_size,
                )
            )
            batches = (
                (a.max_batch_size,) if c.batching == "fixed" else sorted({1, a.max_batch_size})
            )
            for batch in batches:
                np.savez(out / f"inputs-b{batch}.npz", **example_inputs(c, batch))
            generate(
                out / "model.bundle", root / "triton-models", "native_" + name.replace("-", "_")
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "families.hstu.triton.generate_repository",
                    "--bundle",
                    str(out / "model.bundle"),
                    "--repository",
                    str(root / "triton-models"),
                    "--name",
                    "history_" + name.replace("-", "_"),
                    "--runtime-root",
                    str(sdk),
                    "--history-cache",
                    "--kv-cache-size-bytes",
                    "9437184",
                ],
                check=True,
            )
    print("Prepared four variants:", root, flush=True)


if __name__ == "__main__":
    main()
