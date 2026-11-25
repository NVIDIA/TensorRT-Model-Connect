# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the two checked-in serving profiles using identical complete weights."""

import argparse
from dataclasses import replace
from pathlib import Path
import shutil
import subprocess
import sys
from tensorrt_model_connect import BuildRequest, build
from families.hstu_research.config import Config, ServingProfile
from families.hstu_research.triton.native_repository import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--runtime-root", type=Path, required=True)
    args = parser.parse_args()
    if args.root.exists() and any(args.root.iterdir()):
        raise FileExistsError("preparation output must be empty")
    base = Config.load(args.checkpoint)
    profiles = {
        "default": ServingProfile(
            history_length=base.architecture.max_history_length, candidates=64
        ),
        "short": ServingProfile(history_length=64, candidates=128, batching="fixed"),
    }
    for name, profile in profiles.items():
        for precision in ("fp32", "bf16"):
            directory = args.root / "output" / f"research-{name}-{precision}"
            directory.mkdir(parents=True)
            for filename in ("hstu_research_weights.npz", "checkpoint_provenance.json"):
                shutil.copyfile(args.checkpoint / filename, directory / filename)
            replace(base, serving=profile).save(directory)
            build(
                BuildRequest(
                    model_dir=directory,
                    output_path=directory / "model.bundle",
                    family="hstu_research",
                    task="structured_ranking",
                    precision=precision,
                    max_batch_size=args.max_batch_size,
                )
            )
            generate(
                directory / "model.bundle",
                args.root / "triton-models",
                f"native_research_{name}_{precision}",
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "families.hstu_research.triton.generate_repository",
                    "--bundle",
                    str(directory / "model.bundle"),
                    "--repository",
                    str(args.root / "triton-models"),
                    "--name",
                    f"history_research_{name}_{precision}",
                    "--runtime-root",
                    str(args.runtime_root),
                    "--history-cache",
                    "--kv-cache-size-bytes",
                    "134217728",
                    "--kv-cache-max-entries",
                    "8",
                    "--kv-cache-lease-seconds",
                    "120",
                ],
                check=True,
            )


if __name__ == "__main__":
    main()
