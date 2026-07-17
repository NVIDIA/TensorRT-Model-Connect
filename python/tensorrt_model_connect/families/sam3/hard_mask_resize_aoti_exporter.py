# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export Meta-compatible SAM3 hard-mask resizing as fixed AOTI packages.

Meta consolidates prompt masks only after resizing every object row from the
tracker's 288 grid to the 1008 image grid with ``torch.interpolate``.  These
small B1/B2 packages are developer-only Golden oracles.  Production bundles
contain equivalent native TensorRT resize plans and never package or load the
AOTI artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import tracker_memory_aoti_exporter as _memory_exporter


HARD_MASK_RESIZE_AOTI_MANIFEST_SECTION = "sam3_hard_mask_resize_aoti_manifest.json"
_PACKAGE_SECTIONS = {
    1: "sam3_hard_mask_resize_b1.pt2",
    2: "sam3_hard_mask_resize_b2.pt2",
}
_CACHE_ROOT = Path(tempfile.gettempdir()) / "trtmc-sam3-hard-mask-resize-aoti"
_LOW_RES_MASK_SIZE = 288
_TRACKER_IMAGE_SIZE = 1008
_GLOBAL_DIGEST_CHARACTERS = 20
_SMOKE_MAXIMUM_ABSOLUTE_ERROR = 2.0e-5


def _valid_smoke_error(value: Any) -> bool:
    try:
        error = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(error) and 0.0 <= error <= _SMOKE_MAXIMUM_ABSOLUTE_ERROR


@dataclass(frozen=True)
class HardMaskResizeAotiPackage:
    """One immutable fixed-batch resize package."""

    batch_size: int
    path: Path
    section: str
    sha256: str
    package_global: str


@dataclass(frozen=True)
class HardMaskResizeAotiArtifacts:
    """Developer-only B1/B2 PyTorch Golden packages.

    ``bundle_sections`` describes the reference package layout for Golden
    validation tooling.  Production ``trtmc build sam3`` never consumes these
    sections; it emits the equivalent native TensorRT resize plans instead.
    """

    cache_directory: Path
    packages: tuple[HardMaskResizeAotiPackage, ...]
    producer_abi: _memory_exporter.MemoryAotiProducerAbi
    manifest_bytes: bytes
    bundle_sections: tuple[tuple[str, bytes], ...]

    def package(self, batch_size: int) -> HardMaskResizeAotiPackage:
        for package in self.packages:
            if package.batch_size == batch_size:
                return package
        raise ValueError("SAM3 hard-mask resize supports only B1 and B2")


def _source_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _producer(dependencies: Any, device_index: int) -> _memory_exporter.MemoryAotiProducerAbi:
    return _memory_exporter._producer_abi(dependencies, device_index)


def _global_name(batch_size: int, digest: str) -> str:
    if batch_size not in (1, 2) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("Invalid SAM3 hard-mask resize package identity")
    return (
        f"trtmc.sam3.tracker_memory.resize.b{batch_size}.fixed.{digest[:_GLOBAL_DIGEST_CHARACTERS]}"
    )


def _cache_key(
    producer: _memory_exporter.MemoryAotiProducerAbi,
    exporter_digest: str,
) -> str:
    payload = json.dumps(
        {"producer": asdict(producer), "exporter_sha256": exporter_digest},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _make_module(torch: Any, batch_size: int) -> Any:
    if batch_size not in (1, 2):
        raise ValueError("SAM3 hard-mask resize supports only B1 and B2")

    class HardMaskResize(torch.nn.Module):
        def forward(self, tracker_mask):
            return torch.nn.functional.interpolate(
                tracker_mask,
                size=(_TRACKER_IMAGE_SIZE, _TRACKER_IMAGE_SIZE),
                mode="bilinear",
                align_corners=False,
            ).contiguous()

    return HardMaskResize().eval()


def _example_input(torch: Any, batch_size: int, device: Any) -> tuple[Any, ...]:
    values = torch.linspace(
        -7.0,
        7.0,
        steps=batch_size * _LOW_RES_MASK_SIZE * _LOW_RES_MASK_SIZE,
        dtype=torch.float32,
        device=device,
    )
    return (values.reshape(batch_size, 1, _LOW_RES_MASK_SIZE, _LOW_RES_MASK_SIZE),)


def _compile_and_validate(
    torch: Any,
    *,
    batch_size: int,
    package: Path,
    device: Any,
) -> dict[str, Any]:
    module = _make_module(torch, batch_size)
    inputs = _example_input(torch, batch_size, device)
    with torch.inference_mode():
        exported = torch.export.export(module, inputs, strict=False)
    torch._inductor.aoti_compile_and_package(
        exported,
        package_path=os.fspath(package),
        inductor_configs={
            "max_autotune": True,
            "triton.cudagraphs": False,
            "aot_inductor.use_runtime_constant_folding": False,
        },
    )
    if not package.is_file() or package.stat().st_size == 0:
        raise RuntimeError(f"AOTI did not produce the SAM3 hard-mask resize package {package}")
    loaded = torch._inductor.aoti_load_package(
        os.fspath(package), device_index=int(device.index or 0)
    )
    try:
        with torch.inference_mode():
            expected = _memory_exporter._unwrap_aoti_output(module(*inputs))
            actual = _memory_exporter._unwrap_aoti_output(loaded(*inputs))
            torch.cuda.synchronize(device)
        expected_shape = (batch_size, 1, _TRACKER_IMAGE_SIZE, _TRACKER_IMAGE_SIZE)
        if tuple(actual.shape) != expected_shape or actual.dtype != torch.float32:
            raise RuntimeError(
                "SAM3 hard-mask resize AOTI output contract mismatch: "
                f"{tuple(actual.shape)}/{actual.dtype}"
            )
        maximum_absolute_error = float((actual - expected).abs().max().item())
        if not _valid_smoke_error(maximum_absolute_error):
            raise RuntimeError(
                "SAM3 hard-mask resize AOTI does not match eager interpolate: "
                f"max_abs={maximum_absolute_error}"
            )
        return {
            "batch_size": batch_size,
            "maximum_absolute_error": maximum_absolute_error,
            "passed": True,
        }
    finally:
        del loaded, module, inputs


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("utf-8")


def _validate_cache(
    cache_directory: Path,
    *,
    producer: _memory_exporter.MemoryAotiProducerAbi,
    exporter_digest: str,
) -> dict[str, Any] | None:
    manifest_path = cache_directory / HARD_MASK_RESIZE_AOTI_MANIFEST_SECTION
    if not _memory_exporter._regular_file(manifest_path):
        return None
    try:
        manifest = json.loads(manifest_path.read_bytes())
        if (
            manifest.get("schema_version") != 1
            or manifest.get("scope") != "torch_bilinear_288_to_1008_b1_b2"
            or manifest.get("artifact_format") != "torch.aot_inductor.package.pt2"
            or manifest.get("implementation")
            != {
                "library": "torch",
                "operator": "torch.nn.functional.interpolate",
                "mode": "bilinear",
                "align_corners": False,
                "source_size": 288,
                "target_size": 1008,
            }
            or manifest.get("producer") != json.loads(_canonical_json(asdict(producer)))
            or manifest.get("host_architecture") != producer.host_architecture
            or manifest.get("exporter_sha256") != exporter_digest
            or manifest.get("input_abi")
            != [{"name": "tracker_mask", "dtype": "float32", "shape": ["B", 1, 288, 288]}]
            or manifest.get("output_abi")
            != [
                {
                    "name": "resized_tracker_mask",
                    "dtype": "float32",
                    "shape": ["B", 1, 1008, 1008],
                }
            ]
        ):
            return None
        packages = manifest.get("packages")
        if not isinstance(packages, list) or len(packages) != 2:
            return None
        for record in packages:
            batch_size = int(record["batch_size"])
            digest = str(record["sha256"])
            filename = str(record["filename"])
            if (
                batch_size not in (1, 2)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                or record.get("section") != _PACKAGE_SECTIONS[batch_size]
                or record.get("package_global") != _global_name(batch_size, digest)
                or filename != f"sam3_hard_mask_resize_b{batch_size}_{digest}.pt2"
                or set(record) != {"batch_size", "filename", "section", "sha256", "package_global"}
            ):
                return None
            package = cache_directory / filename
            if (
                not _memory_exporter._regular_file(package)
                or _memory_exporter._hash_file(package) != digest
            ):
                return None
        validation = manifest.get("package_validation", {})
        cases = validation.get("cases", [])
        validation_limit = float(validation.get("maximum_absolute_error", -1.0))
        if (
            validation.get("reference") != "same torch.interpolate eager execution"
            or not math.isfinite(validation_limit)
            or validation_limit != _SMOKE_MAXIMUM_ABSOLUTE_ERROR
            or len(cases) != 2
            or {(int(case["batch_size"]), bool(case["passed"])) for case in cases}
            != {(1, True), (2, True)}
            or any(not _valid_smoke_error(case["maximum_absolute_error"]) for case in cases)
        ):
            return None
        return manifest
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _artifacts_from_cache(
    cache_directory: Path,
    *,
    producer: _memory_exporter.MemoryAotiProducerAbi,
    exporter_digest: str,
) -> HardMaskResizeAotiArtifacts:
    manifest = _validate_cache(cache_directory, producer=producer, exporter_digest=exporter_digest)
    if manifest is None:
        raise RuntimeError(f"Invalid SAM3 hard-mask resize AOTI cache {cache_directory}")
    by_batch = {int(record["batch_size"]): record for record in manifest["packages"]}
    packages = tuple(
        HardMaskResizeAotiPackage(
            batch_size=batch_size,
            path=cache_directory / by_batch[batch_size]["filename"],
            section=by_batch[batch_size]["section"],
            sha256=by_batch[batch_size]["sha256"],
            package_global=by_batch[batch_size]["package_global"],
        )
        for batch_size in (1, 2)
    )
    manifest_bytes = (cache_directory / HARD_MASK_RESIZE_AOTI_MANIFEST_SECTION).read_bytes()
    return HardMaskResizeAotiArtifacts(
        cache_directory=cache_directory,
        packages=packages,
        producer_abi=producer,
        manifest_bytes=manifest_bytes,
        bundle_sections=(
            (HARD_MASK_RESIZE_AOTI_MANIFEST_SECTION, manifest_bytes),
            *((package.section, package.path.read_bytes()) for package in packages),
        ),
    )


def export_sam3_hard_mask_resize_aoti(
    *,
    device_index: int = 0,
    cache_dir: str | Path | None = None,
) -> HardMaskResizeAotiArtifacts:
    """Export or reuse exact fixed B1/B2 PyTorch bilinear resize packages."""

    dependencies = _memory_exporter._load_dependencies()
    producer = _producer(dependencies, device_index)
    exporter_digest = _source_digest()
    key = _cache_key(producer, exporter_digest)
    cache_root = Path(cache_dir).resolve() if cache_dir is not None else _CACHE_ROOT
    cache_directory = cache_root / key
    validation = {"producer": producer, "exporter_digest": exporter_digest}
    if _validate_cache(cache_directory, **validation) is not None:
        return _artifacts_from_cache(cache_directory, **validation)

    with _memory_exporter._exclusive_cache_lock(cache_root, key):
        if _validate_cache(cache_directory, **validation) is not None:
            return _artifacts_from_cache(cache_directory, **validation)
        if cache_directory.is_symlink():
            cache_directory.unlink()
        elif cache_directory.exists():
            shutil.rmtree(cache_directory)
        staging = Path(tempfile.mkdtemp(prefix=f".{key}.build-", dir=cache_root))
        torch = dependencies.torch
        previous_device = int(torch.cuda.current_device())
        try:
            torch.cuda.set_device(device_index)
            device = torch.device(f"cuda:{device_index}")
            cases = []
            records = []
            with torch.random.fork_rng(devices=[device_index]):
                torch.manual_seed(20260717)
                for batch_size in (1, 2):
                    temporary = staging / f"resize_b{batch_size}.pt2"
                    cases.append(
                        _compile_and_validate(
                            torch,
                            batch_size=batch_size,
                            package=temporary,
                            device=device,
                        )
                    )
                    digest = _memory_exporter._hash_file(temporary)
                    destination = staging / f"sam3_hard_mask_resize_b{batch_size}_{digest}.pt2"
                    os.replace(temporary, destination)
                    records.append(
                        {
                            "batch_size": batch_size,
                            "filename": destination.name,
                            "section": _PACKAGE_SECTIONS[batch_size],
                            "sha256": digest,
                            "package_global": _global_name(batch_size, digest),
                        }
                    )
                    torch.cuda.empty_cache()
            manifest = {
                "schema_version": 1,
                "scope": "torch_bilinear_288_to_1008_b1_b2",
                "artifact_format": "torch.aot_inductor.package.pt2",
                "implementation": {
                    "library": "torch",
                    "operator": "torch.nn.functional.interpolate",
                    "mode": "bilinear",
                    "align_corners": False,
                    "source_size": 288,
                    "target_size": 1008,
                },
                "producer": asdict(producer),
                "host_architecture": platform.machine(),
                "exporter_sha256": exporter_digest,
                "input_abi": [
                    {
                        "name": "tracker_mask",
                        "dtype": "float32",
                        "shape": ["B", 1, 288, 288],
                    }
                ],
                "output_abi": [
                    {
                        "name": "resized_tracker_mask",
                        "dtype": "float32",
                        "shape": ["B", 1, 1008, 1008],
                    }
                ],
                "packages": records,
                "package_validation": {
                    "reference": "same torch.interpolate eager execution",
                    "maximum_absolute_error": _SMOKE_MAXIMUM_ABSOLUTE_ERROR,
                    "cases": cases,
                },
            }
            (staging / HARD_MASK_RESIZE_AOTI_MANIFEST_SECTION).write_bytes(
                _canonical_json(manifest)
            )
            os.replace(staging, cache_directory)
        finally:
            torch.cuda.empty_cache()
            torch.cuda.set_device(previous_device)
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
    return _artifacts_from_cache(cache_directory, **validation)


__all__ = [
    "HARD_MASK_RESIZE_AOTI_MANIFEST_SECTION",
    "HardMaskResizeAotiArtifacts",
    "HardMaskResizeAotiPackage",
    "export_sam3_hard_mask_resize_aoti",
]
