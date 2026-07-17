# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the Wan2.2 VAE FP32 graph-barrier plugin in isolation."""

from __future__ import annotations

import fcntl
import hashlib
import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


_PLUGIN_ENV = "TRTMC_WAN22_VAE_CUDA_PLUGIN_LIBRARY"
_BUILD_DIR_ENV = "TRTMC_WAN22_VAE_CUDA_PLUGIN_BUILD_DIR"


def _source_digest(source_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(source_dir.iterdir()):
        if path.is_file() and path.suffix in {".cu", ".h", ".txt"}:
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


@contextmanager
def _exclusive_build_lock(build_base: Path, digest: str) -> Iterator[None]:
    build_base.mkdir(parents=True, exist_ok=True)
    with (build_base / f".{digest}.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield


def ensure_vae_cuda_plugin(*, verbose: bool = False) -> Path:
    """Return the VAE CUDA/TensorRT plugin, building it when absent."""

    override = os.environ.get(_PLUGIN_ENV)
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{_PLUGIN_ENV} does not exist: {path}")
        return path

    source_dir = Path(__file__).with_name("vae_cuda_plugins")
    build_base = Path(
        os.environ.get(_BUILD_DIR_ENV, "/tmp/trtmc-wan22-vae-cuda-plugin")
    ).expanduser()
    digest = _source_digest(source_dir)
    build_dir = build_base / digest
    output = build_dir / "libtrtmc_wan22_vae_cuda_plugin.so"
    if output.is_file():
        return output

    with _exclusive_build_lock(build_base, digest):
        if output.is_file():
            return output
        build_dir.mkdir(parents=True, exist_ok=True)
        configure = [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        build = [
            "cmake",
            "--build",
            str(build_dir),
            "--target",
            "trtmc_wan22_vae_cuda_plugin",
            "-j2",
        ]
        kwargs = (
            {}
            if verbose
            else {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "text": True,
            }
        )
        try:
            subprocess.run(configure, check=True, **kwargs)
            subprocess.run(build, check=True, **kwargs)
        except subprocess.CalledProcessError as error:
            output_text = getattr(error, "stdout", "") or ""
            raise RuntimeError(f"Wan2.2 VAE CUDA plugin build failed\n{output_text}") from error
        if not output.is_file():
            raise RuntimeError(f"VAE CUDA plugin build did not produce {output}")
    return output


__all__ = ["ensure_vae_cuda_plugin"]
