# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture and verify public package versions from one isolated image venv."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import platform
import re
import subprocess
import sys
import sysconfig
from pathlib import Path


def venv_packages() -> list[str]:
    root = Path(sys.prefix).resolve()
    if root == Path(sys.base_prefix).resolve():
        raise ValueError("Run this recorder with the image's isolated venv interpreter")
    paths = sorted({sysconfig.get_path("purelib"), sysconfig.get_path("platlib")})
    if any(not Path(path).resolve().is_relative_to(root) for path in paths):
        raise ValueError("The package search path must remain inside the image venv")
    packages = {}
    for dist in metadata.distributions(path=paths):
        name = re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower()
        version = dist.version
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name) or not re.fullmatch(
            r"[A-Za-z0-9.+!_-]+", version
        ):
            raise ValueError("Package metadata must contain public names and exact versions")
        if name in packages and packages[name] != version:
            raise ValueError("Conflicting distributions exist inside the same venv")
        packages[name] = version
    if not packages:
        raise ValueError("The venv package inventory is empty")
    return [f"{name}=={version}" for name, version in sorted(packages.items())]


def read_lock(path: Path) -> list[str]:
    rows = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not rows or any(
        not re.fullmatch(r"[a-z0-9][a-z0-9-]*==[A-Za-z0-9.+!_-]+", line) for line in rows
    ):
        raise ValueError("A captured lock must contain only exact public package versions")
    # The recorder orders normalized package names. Sorting the complete pin
    # instead would place "name-extra==..." before "name==..." and reject
    # valid captured environments containing both distributions.
    ordered = sorted(set(rows), key=lambda line: line.split("==", 1)[0])
    if rows != ordered or len({line.split("==", 1)[0] for line in rows}) != len(rows):
        raise ValueError("A captured lock must be sorted with one version per package")
    return rows


def environment(snapshot: str) -> dict:
    import torch
    import tensorrt

    apt = subprocess.check_output(
        ["dpkg-query", "-W", "-f=${binary:Package}==${Version}\n"], text=True
    ).splitlines()
    return {
        "schema_version": 1,
        "platform": platform.machine(),
        "python": platform.python_version(),
        "venv_packages": venv_packages(),
        "apt_snapshot": snapshot,
        "apt_packages": sorted(apt),
        "abi": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cxx11abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
            "tensorrt": tensorrt.__version__,
            "apache_tvm_ffi": metadata.version("apache-tvm-ffi"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("capture", "verify", "validate-lock"))
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--snapshot", default="20261008T000000Z")
    args = parser.parse_args()
    if args.mode == "validate-lock":
        read_lock(args.lock)
        return
    if args.receipt is None:
        parser.error("Capturing and verifying require a public environment receipt")
    observed = environment(args.snapshot)
    if args.mode == "capture":
        args.lock.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.lock.write_text("\n".join(observed["venv_packages"]) + "\n")
        args.receipt.write_text(json.dumps(observed, sort_keys=True, indent=2) + "\n")
    else:
        if (
            read_lock(args.lock) != observed["venv_packages"]
            or json.loads(args.receipt.read_text()) != observed
        ):
            raise ValueError(
                "The rebuilt package, APT or ABI environment differs from its captured receipt"
            )
    print(
        json.dumps(
            {
                "mode": args.mode,
                "venv_package_count": len(observed["venv_packages"]),
                "abi": observed["abi"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
