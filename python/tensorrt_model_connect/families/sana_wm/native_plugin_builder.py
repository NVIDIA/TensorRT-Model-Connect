# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the SANA-WM TensorRT plugin without touching repository-wide CMake."""

from __future__ import annotations

import tensorrt_model_connect.utils.fcntl_shim as fcntl
import hashlib
import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


_PLUGIN_ENV = "TRTMC_SANA_WM_NATIVE_PLUGIN_LIBRARY"
_BUILD_DIR_ENV = "TRTMC_SANA_WM_NATIVE_PLUGIN_BUILD_DIR"


def _source_digest(source_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(source_dir.iterdir()):
        if path.is_file() and path.suffix in {".cpp", ".cu", ".h", ".txt"}:
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


@contextmanager
def _exclusive_build_lock(build_base: Path, source_digest: str) -> Iterator[None]:
    build_base.mkdir(parents=True, exist_ok=True)
    lock_path = build_base / f".{source_digest}.lock"
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield


def ensure_native_plugin(*, verbose: bool = False) -> Path:
    override = os.environ.get(_PLUGIN_ENV)
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{_PLUGIN_ENV} does not exist: {path}")
        return path

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("SANA-WM native plugin build requires the C++ libtorch package") from exc

    source_dir = Path(__file__).with_name("native_plugins")
    build_base = Path(
        os.environ.get(_BUILD_DIR_ENV, "/tmp/trtmc-sana-wm-native-plugin")
    ).expanduser()
    source_digest = _source_digest(source_dir)
    build_dir = build_base / source_digest
    output = build_dir / "libtrtmc_sana_wm_native_plugin.so"
    if output.is_file():
        return output

    with _exclusive_build_lock(build_base, source_digest):
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
            f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
        ]
        build = [
            "cmake",
            "--build",
            str(build_dir),
            "--target",
            "trtmc_sana_wm_native_plugin",
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
        except subprocess.CalledProcessError as exc:
            output_text = getattr(exc, "stdout", "") or ""
            raise RuntimeError(f"SANA-WM native plugin build failed\n{output_text}") from exc
        if not output.is_file():
            raise RuntimeError(f"SANA-WM native plugin build did not produce {output}")
    return output
