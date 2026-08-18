# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Builder-time CUDA helper for source-exact PAFPN BatchNorm constants."""

from __future__ import annotations

import ctypes
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np

from . import native_plugin_builder


_BUILD_DIR_ENV = "TRTMC_SAM2_HOI_PAFPN_BN_INVSTD_BUILD_DIR"
_HELPER_SOURCE = Path(__file__).with_name("pafpn_bn_invstd_helper.cu")
HELPER_SOURCE_SHA256 = "4d0fad825f75412c968764ed2baade5c652963dd956db385c4eed3ce932089c0"
HELPER_VERSION = "sam2-hoi-pafpn-bn-invstd-cuda133-v1"
_CUDA_ARCHITECTURES = ("89", "100")
_RECEIPT_NAME = "build-receipt.json"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configured_build_base() -> Path:
    configured = os.environ.get(_BUILD_DIR_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path(tempfile.gettempdir()) / f"trtmc-sam2-hoi-pafpn-bn-invstd-{os.geteuid()}"


def _source_identity() -> dict[str, object]:
    if not _HELPER_SOURCE.is_file() or _HELPER_SOURCE.is_symlink():
        raise RuntimeError("SAM2 HOI PAFPN invstd helper source must be a regular file")
    digest = _file_sha256(_HELPER_SOURCE)
    if digest != HELPER_SOURCE_SHA256:
        raise RuntimeError("SAM2 HOI PAFPN invstd helper source identity drift")
    return {
        "path": str(_HELPER_SOURCE.resolve()),
        "size": _HELPER_SOURCE.stat().st_size,
        "sha256": digest,
    }


def _nvcc_path() -> Path:
    command = os.environ.get("CUDACXX") or "nvcc"
    parts = shlex.split(command)
    if len(parts) != 1:
        raise RuntimeError("SAM2 HOI PAFPN invstd CUDACXX must be one direct nvcc path")
    resolved = shutil.which(parts[0])
    if resolved is None:
        raise RuntimeError(f"SAM2 HOI PAFPN invstd cannot resolve CUDA compiler {parts[0]!r}")
    path = Path(resolved).resolve(strict=True)
    if path.name != "nvcc":
        raise RuntimeError("SAM2 HOI PAFPN invstd CUDACXX must resolve directly to nvcc")
    return path


def _compile_flags() -> tuple[str, ...]:
    architectures = tuple(
        argument
        for architecture in _CUDA_ARCHITECTURES
        for argument in ("-gencode", f"arch=compute_{architecture},code=sm_{architecture}")
    )
    return (
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
        "-Xcompiler=-fPIC",
        "-shared",
        *architectures,
    )


def _build_identity() -> dict[str, object]:
    nvcc = _nvcc_path()
    compiler = native_plugin_builder._command_identity(str(nvcc), "--version")
    if compiler.get("returncode") != 0:
        raise RuntimeError("SAM2 HOI PAFPN invstd CUDA compiler identity failed")
    return {
        "schema_version": 1,
        "helper_version": HELPER_VERSION,
        "source": _source_identity(),
        "compiler": compiler,
        "cuda_architectures": list(_CUDA_ARCHITECTURES),
        "compile_flags": list(_compile_flags()),
        "kernel_expression": "rsqrtf(__fadd_rn(variance[index], epsilon))",
        "execution_scope": "builder_time_only",
        "inference_runtime_launch_added": False,
    }


def _build_helper(*, verbose: bool) -> tuple[Path, Path, dict[str, object]]:
    build_base = native_plugin_builder._secure_private_directory(_configured_build_base())
    identity = _build_identity()
    identity_digest = native_plugin_builder._identity_digest(identity)
    build_dir = native_plugin_builder._secure_private_directory(build_base / identity_digest)
    output = build_dir / "libsam2_hoi_pafpn_bn_invstd.so"
    receipt = build_dir / _RECEIPT_NAME
    if native_plugin_builder._cached_output_matches(output, receipt, identity):
        return output, receipt, identity

    with native_plugin_builder._exclusive_build_lock(build_base, identity_digest):
        build_dir = native_plugin_builder._secure_private_directory(build_dir)
        if native_plugin_builder._cached_output_matches(output, receipt, identity):
            return output, receipt, identity
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".tmp", dir=build_dir
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        temporary.unlink()
        compiler_path = Path(str(identity["compiler"]["path"])).resolve(strict=True)
        command = [
            str(compiler_path),
            *_compile_flags(),
            str(_HELPER_SOURCE),
            "-o",
            str(temporary),
        ]
        kwargs: dict[str, Any] = (
            {}
            if verbose
            else {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "text": True,
            }
        )
        try:
            subprocess.run(command, check=True, **kwargs)
            if not native_plugin_builder._harden_built_output(temporary):
                raise RuntimeError("SAM2 HOI PAFPN invstd compile produced no DSO")
            os.replace(temporary, output)
            try:
                native_plugin_builder._write_build_receipt(output, receipt, identity)
            except Exception:
                output.unlink(missing_ok=True)
                receipt.unlink(missing_ok=True)
                raise
        except subprocess.CalledProcessError as error:
            output_text = getattr(error, "stdout", "") or ""
            raise RuntimeError(
                f"SAM2 HOI PAFPN invstd helper build failed\n{output_text}"
            ) from error
        finally:
            temporary.unlink(missing_ok=True)
    return output, receipt, identity


@lru_cache(maxsize=4)
def _load_functions(path: str, digest: str):
    helper = Path(path).resolve(strict=True)
    if _file_sha256(helper) != digest:
        raise RuntimeError("SAM2 HOI PAFPN invstd helper DSO identity drift")
    library = ctypes.CDLL(str(helper), mode=ctypes.RTLD_LOCAL)
    version = library.sam2_hoi_pafpn_bn_invstd_helper_version
    version.argtypes = []
    version.restype = ctypes.c_char_p
    observed_version = version()
    if observed_version is None or observed_version.decode("utf-8") != HELPER_VERSION:
        raise RuntimeError("SAM2 HOI PAFPN invstd helper version drift")
    compute = library.sam2_hoi_pafpn_bn_invstd_f32
    compute.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
    ]
    compute.restype = ctypes.c_int
    return library, compute


def helper_build_receipt(*, verbose: bool = False) -> dict[str, object]:
    """Build or validate the helper and return its authenticated receipt."""

    output, receipt, identity = _build_helper(verbose=verbose)
    recorded = json.loads(receipt.read_text(encoding="utf-8"))
    expected = {"identity": identity, "output_sha256": _file_sha256(output)}
    if recorded != expected:
        raise RuntimeError("SAM2 HOI PAFPN invstd helper receipt drift")
    _library, compute = _load_functions(str(output.resolve()), expected["output_sha256"])
    invalid_status = compute(None, ctypes.c_int32(0), ctypes.c_float(1.0e-5), None)
    if invalid_status != -1:
        raise RuntimeError("SAM2 HOI PAFPN invstd helper invalid-argument contract drift")
    return {
        "path": str(receipt),
        "sha256": _file_sha256(receipt),
        "payload": recorded,
        "invalid_argument_contract_status": invalid_status,
    }


def compute_invstd(
    running_variance: np.ndarray,
    *,
    epsilon: float,
    verbose: bool = False,
) -> np.ndarray:
    """Compute CUDA-exact eval-mode invstd constants during engine building."""

    variance = np.ascontiguousarray(running_variance, dtype=np.float32)
    if variance.ndim != 1 or variance.size <= 0:
        raise ValueError("SAM2 HOI PAFPN running variance must be a nonempty vector")
    if not np.all(np.isfinite(variance)) or np.any(variance < 0):
        raise ValueError("SAM2 HOI PAFPN running variance must be finite and nonnegative")
    epsilon_f32 = np.float32(epsilon)
    if not np.isfinite(epsilon_f32) or epsilon_f32 <= 0:
        raise ValueError("SAM2 HOI PAFPN BatchNorm epsilon must be finite and positive")

    output_path, _receipt, _identity = _build_helper(verbose=verbose)
    digest = _file_sha256(output_path)
    _library, compute = _load_functions(str(output_path.resolve()), digest)
    invstd = np.empty_like(variance)
    status = compute(
        variance.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int32(variance.size),
        ctypes.c_float(float(epsilon_f32)),
        invstd.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    if status != 0:
        raise RuntimeError(f"SAM2 HOI PAFPN invstd helper failed with CUDA status {status}")
    if not np.all(np.isfinite(invstd)):
        raise RuntimeError("SAM2 HOI PAFPN invstd helper returned nonfinite constants")
    return np.ascontiguousarray(invstd, dtype=np.float32)


__all__ = ["HELPER_SOURCE_SHA256", "HELPER_VERSION", "compute_invstd", "helper_build_receipt"]
