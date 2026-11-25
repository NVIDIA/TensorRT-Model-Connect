# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare official DLRMv3 bundles and both Triton repositories outside the source tree."""

import argparse
from dataclasses import replace
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
from tensorrt_model_connect import BuildRequest, build
from ..checkpoint import DEFAULT_URL, convert
from ..config import Config, ServingProfile
from ..triton.native_repository import generate


def prepare(root, checkpoint, runtime_root, *, repositories=True):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if checkpoint is None:
        checkpoint = root / "non_sparse.ckpt"
        request = urllib.request.Request(DEFAULT_URL, headers={"User-Agent": "Mozilla/5.0"})
        with (
            urllib.request.urlopen(request, timeout=120) as source,
            checkpoint.open("wb") as target,
        ):
            shutil.copyfileobj(source, target)
    if Path(checkpoint).is_dir():
        converted = Path(checkpoint)
    else:
        converted = root / "checkpoint"
        convert(checkpoint, converted)
    for name, profile in (
        ("default", ServingProfile()),
        ("short", ServingProfile(history_length=32, candidates=8, batching="fixed")),
    ):
        for precision in ("fp32", "bf16"):
            directory = root / "output" / f"dlrmv3-{name}-{precision}"
            directory.mkdir(parents=True)
            for file in ("dlrmv3_weights.npz", "checkpoint_provenance.json"):
                shutil.copyfile(converted / file, directory / file)
            replace(Config.load(converted), serving=profile).save(directory)
            bundle = directory / "model.bundle"
            build(
                BuildRequest(
                    model_dir=directory,
                    output_path=bundle,
                    family="hstu",
                    task="structured_ranking",
                    precision=precision,
                    max_batch_size=4,
                )
            )
            suffix = f"dlrmv3_{name}_{precision}"
            if repositories:
                generate(
                    bundle, root / "triton-models", "native_" + suffix, runtime_root=runtime_root
                )
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "families.hstu.triton.generate_repository",
                        "--bundle",
                        str(bundle),
                        "--repository",
                        str(root / "triton-models"),
                        "--name",
                        "hydrated_" + suffix,
                        "--runtime-root",
                        str(runtime_root),
                        "--history-cache",
                        "--kv-cache-size-bytes",
                        "134217728",
                    ],
                    check=True,
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--runtime-root", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root, args.checkpoint, args.runtime_root)


if __name__ == "__main__":
    main()
