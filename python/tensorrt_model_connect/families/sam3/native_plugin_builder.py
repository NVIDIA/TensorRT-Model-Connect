# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and load SAM3's stream-aware tracker-step TensorRT plugin.

The plugin is model-owned and packaged in the SAM3 bundle.  Its build cache is
content-addressed and contains no runtime or tuning environment-variable
surface.  AOTI PT2 packages are produced separately by the tracker exporter;
this module only builds the ABI bridge that consumes them.
"""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import importlib
import json
import math
import os
import platform
import re
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .hard_mask_resize_aoti_exporter import HardMaskResizeAotiArtifacts
    from .tracker_memory_aoti_exporter import Sam3TrackerMemoryAotiArtifacts
    from .tracker_step_aoti_exporter import Sam3TrackerSplitAotiArtifacts


_PLUGIN_NAME = "libtrtmc_sam3_tracker_step_native_plugin.so"
_PLUGIN_VERSION = "2"
TRACKER_STEP_NATIVE_PLUGIN_SECTION = "sam3_tracker_step_native_plugin_so"
TRACKER_STEP_RUNTIME_MANIFEST_SECTION = "sam3_tracker_step_runtime_manifest.json"
TRACKER_STEP_RUNTIME_SCOPE = "meta_split_dynamic_encoder_static_decoder"
_BUILD_ROOT = Path(tempfile.gettempdir()) / "trtmc-sam3-tracker-step-native-plugin"
_LOADED_PLUGINS: list[ctypes.CDLL] = []


@dataclass(frozen=True)
class _BuildInputs:
    torch_root: Path
    torch_cmake_prefix: Path
    tvm_ffi_root: Path
    tensorrt_root: Path
    torch_version: str
    tvm_ffi_version: str
    tensorrt_version: str
    host_architecture: str
    torch_cxx11_abi: bool


@dataclass(frozen=True)
class TrackerStepRuntimeArtifacts:
    """Native bridge and immutable bundle payloads for split tracker AOTI."""

    plugin_library: Path
    runtime_manifest: bytes
    bundle_sections: tuple[tuple[str, bytes], ...]


def _package_root(module: Any, package_name: str) -> Path:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise RuntimeError(f"{package_name} does not expose a package path")
    return Path(module_file).resolve().parent


def _required_version(module: Any, package_name: str) -> str:
    version = str(getattr(module, "__version__", "") or "")
    if not version or version == "unknown":
        raise RuntimeError(f"{package_name} does not expose a build ABI version")
    return version


def _discover_build_inputs() -> _BuildInputs:
    try:
        torch = importlib.import_module("torch")
        tvm_ffi = importlib.import_module("tvm_ffi")
        tensorrt = importlib.import_module("tensorrt")
    except ImportError as error:
        raise RuntimeError(
            "SAM3 tracker-step bridge build requires torch, tvm_ffi, and tensorrt"
        ) from error

    torch_root = _package_root(torch, "torch")
    tvm_ffi_root = _package_root(tvm_ffi, "tvm_ffi")
    tensorrt_package = _package_root(tensorrt, "tensorrt")
    # TensorRT wheels commonly split Python bindings and native libraries into
    # sibling packages.  Passing their shared parent lets CMake inspect both,
    # while system installations remain discoverable through standard paths.
    tensorrt_root = tensorrt_package.parent
    cmake_prefix = Path(torch.utils.cmake_prefix_path).resolve()
    return _BuildInputs(
        torch_root=torch_root,
        torch_cmake_prefix=cmake_prefix,
        tvm_ffi_root=tvm_ffi_root,
        tensorrt_root=tensorrt_root,
        torch_version=_required_version(torch, "torch"),
        tvm_ffi_version=_required_version(tvm_ffi, "tvm_ffi"),
        tensorrt_version=_required_version(tensorrt, "tensorrt"),
        host_architecture=platform.machine(),
        torch_cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI),
    )


def _source_digest(source_dir: Path, inputs: _BuildInputs) -> str:
    digest = hashlib.sha256()
    for path in sorted(source_dir.iterdir()):
        if path.is_file() and path.suffix in {".cpp", ".h", ".txt"}:
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
    for value in (
        inputs.torch_root,
        inputs.torch_cmake_prefix,
        inputs.tvm_ffi_root,
        inputs.tensorrt_root,
        inputs.torch_version,
        inputs.tvm_ffi_version,
        inputs.tensorrt_version,
        inputs.host_architecture,
        str(int(inputs.torch_cxx11_abi)),
    ):
        digest.update(b"\0")
        digest.update(str(value).encode("utf-8"))
    return digest.hexdigest()[:20]


@contextmanager
def _exclusive_build_lock(build_root: Path, digest: str) -> Iterator[None]:
    build_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not stat.S_ISDIR(build_root.lstat().st_mode):
        raise RuntimeError("SAM3 tracker-step build cache is not a directory")
    build_root.chmod(0o700)
    lock_path = build_root / f".{digest}.lock"
    descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    with os.fdopen(descriptor, "a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield


def _configure_command(source_dir: Path, build_dir: Path, inputs: _BuildInputs) -> list[str]:
    return [
        "cmake",
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_PREFIX_PATH={inputs.torch_cmake_prefix}",
        f"-DSAM3_TVM_FFI_ROOT={inputs.tvm_ffi_root}",
        f"-DSAM3_TENSORRT_ROOT={inputs.tensorrt_root}",
        f"-DSAM3_TORCH_VERSION={inputs.torch_version}",
        f"-DSAM3_TVM_FFI_VERSION={inputs.tvm_ffi_version}",
        f"-DSAM3_TENSORRT_VERSION={inputs.tensorrt_version}",
        f"-DSAM3_TORCH_CXX11_ABI={int(inputs.torch_cxx11_abi)}",
    ]


def _build_command(build_dir: Path) -> list[str]:
    return [
        "cmake",
        "--build",
        str(build_dir),
        "--target",
        "trtmc_sam3_tracker_step_native_plugin",
        "-j2",
    ]


def _usable_cached_output(path: Path) -> bool:
    try:
        status = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISREG(status.st_mode) and status.st_size > 0


def _publish_output(source: Path, destination: Path) -> None:
    if not _usable_cached_output(source):
        raise RuntimeError(f"SAM3 tracker-step build did not produce {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not stat.S_ISDIR(destination.parent.lstat().st_mode):
        raise RuntimeError("SAM3 tracker-step output cache is not a directory")
    destination.parent.chmod(0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output_stream, source.open("rb") as input_stream:
            shutil.copyfileobj(input_stream, output_stream)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        temporary.chmod(0o700)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_native_plugin(*, verbose: bool = False) -> Path:
    """Return a content-addressed, ABI-matched tracker-step bridge DSO."""

    source_dir = Path(__file__).with_name("native_plugins")
    inputs = _discover_build_inputs()
    digest = _source_digest(source_dir, inputs)
    build_dir = _BUILD_ROOT / digest
    output = build_dir / _PLUGIN_NAME
    if _usable_cached_output(output):
        return output

    with _exclusive_build_lock(_BUILD_ROOT, digest):
        if _usable_cached_output(output):
            return output
        if output.exists() or output.is_symlink():
            output.unlink()
        staging_dir = Path(tempfile.mkdtemp(prefix=f".{digest}.build-", dir=_BUILD_ROOT))
        try:
            commands = (
                _configure_command(source_dir, staging_dir, inputs),
                _build_command(staging_dir),
            )
            kwargs: dict[str, Any] = {}
            if not verbose:
                kwargs = {
                    "stdout": subprocess.PIPE,
                    "stderr": subprocess.STDOUT,
                    "text": True,
                }
            for command in commands:
                subprocess.run(command, check=True, **kwargs)
            _publish_output(staging_dir / _PLUGIN_NAME, output)
        except subprocess.CalledProcessError as error:
            output_text = str(getattr(error, "stdout", "") or "")
            raise RuntimeError(
                "SAM3 tracker-step native plugin build failed\n" + output_text
            ) from error
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)
    return output


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pipeline_global(batch_size: int, encoder_sha256: str, decoder_sha256: str) -> str:
    if batch_size not in (1, 2):
        raise ValueError("SAM3 tracker-step pipelines support only B1 and B2")
    try:
        payload = (
            b"trtmc.sam3.tracker_step.split_aoti.v1\0"
            + bytes.fromhex(encoder_sha256)
            + bytes.fromhex(decoder_sha256)
        )
    except ValueError as error:
        raise RuntimeError("SAM3 tracker-step package has an invalid SHA-256") from error
    if len(encoder_sha256) != 64 or len(decoder_sha256) != 64:
        raise RuntimeError("SAM3 tracker-step package has an invalid SHA-256")
    digest = hashlib.sha256(payload).hexdigest()
    return f"trtmc.sam3.tracker_step.b{batch_size}.split_aoti.{digest[:20]}"


def _validate_split_artifacts(
    artifacts: Sam3TrackerSplitAotiArtifacts,
    inputs: _BuildInputs,
) -> dict[tuple[str, int], Any]:
    producer = artifacts.producer_abi
    if (
        producer.torch_version != inputs.torch_version
        or producer.host_architecture != inputs.host_architecture
        or producer.torch_cxx11_abi != inputs.torch_cxx11_abi
    ):
        raise RuntimeError("SAM3 split AOTI exporter/native bridge ABI mismatch")

    expected_order = (("encoder", 1), ("decoder", 1), ("encoder", 2), ("decoder", 2))
    actual_order = tuple((package.stage, package.batch_size) for package in artifacts.packages)
    if actual_order != expected_order:
        raise RuntimeError("SAM3 split AOTI artifacts do not contain the canonical four packages")
    packages = {(package.stage, package.batch_size): package for package in artifacts.packages}
    for package in artifacts.packages:
        if not package.path.is_file():
            raise FileNotFoundError(package.path)
        if _sha256_bytes(package.path.read_bytes()) != package.sha256:
            raise RuntimeError(f"SAM3 split AOTI package hash mismatch: {package.section}")
    for batch_size in (1, 2):
        expected_global = _pipeline_global(
            batch_size,
            packages[("encoder", batch_size)].sha256,
            packages[("decoder", batch_size)].sha256,
        )
        if artifacts.pipeline_global(batch_size) != expected_global:
            raise RuntimeError(f"SAM3 split AOTI B{batch_size} global does not bind both packages")
    return packages


def _memory_package_global(policy: str, batch_size: int, package_sha256: str) -> str:
    if policy not in {"soft", "hard"} or batch_size not in (1, 2):
        raise RuntimeError("SAM3 tracker-memory package has an invalid variant")
    if len(package_sha256) != 64:
        raise RuntimeError("SAM3 tracker-memory package has an invalid SHA-256")
    return f"trtmc.sam3.tracker_memory.{policy}.b{batch_size}.fixed.{package_sha256[:20]}"


def _validate_memory_artifacts(
    artifacts: Sam3TrackerMemoryAotiArtifacts,
    split_artifacts: Sam3TrackerSplitAotiArtifacts,
    inputs: _BuildInputs,
    *,
    aoti_abi_version: int,
) -> dict[tuple[str, int], Any]:
    producer = artifacts.producer_abi
    split_producer = split_artifacts.producer_abi
    shared_producer_fields = (
        "torch_version",
        "transformers_version",
        "cuda_version",
        "compute_capability",
        "host_architecture",
        "torch_cxx11_abi",
    )
    if any(
        getattr(producer, field) != getattr(split_producer, field)
        for field in shared_producer_fields
    ) or (
        producer.torch_version != inputs.torch_version
        or producer.host_architecture != inputs.host_architecture
        or producer.torch_cxx11_abi != inputs.torch_cxx11_abi
        or producer.torch_aoti_abi_version != aoti_abi_version
    ):
        raise RuntimeError("SAM3 tracker-memory/step AOTI producer ABI mismatch")

    expected_order = (("soft", 1), ("hard", 1), ("soft", 2), ("hard", 2))
    actual_order = tuple((package.policy, package.batch_size) for package in artifacts.packages)
    if actual_order != expected_order:
        raise RuntimeError(
            "SAM3 tracker-memory AOTI artifacts do not contain canonical soft/hard B1/B2 packages"
        )

    expected_abi = (
        (
            "soft",
            (
                ("tracker_feature_2", "float32", (1, 256, 72, 72)),
                ("final_mask", "float32", ("B", 1, 288, 288)),
                ("object_score_logits", "float32", ("B", 1)),
                ("suppress_area_shrinkage", "int32", ("B", 1)),
            ),
        ),
        (
            "hard",
            (
                ("tracker_feature_2", "float32", (1, 256, 72, 72)),
                ("owned_tracker_mask", "float32", ("B", 1, 1008, 1008)),
                ("object_score_logits", "float32", ("B", 1)),
                ("suppress_area_shrinkage", "int32", ("B", 1)),
            ),
        ),
    )
    actual_abi = tuple(
        (
            policy_abi.policy,
            tuple(
                (tensor.name, tensor.dtype, tuple(tensor.shape)) for tensor in policy_abi.tensors
            ),
        )
        for policy_abi in artifacts.input_abi
    )
    if actual_abi != expected_abi:
        raise RuntimeError("SAM3 tracker-memory AOTI input ABI mismatch")

    packages = {(package.policy, package.batch_size): package for package in artifacts.packages}
    for package in artifacts.packages:
        if package.hard_mask is not (package.policy == "hard"):
            raise RuntimeError("SAM3 tracker-memory package mask policy mismatch")
        if not package.path.is_file():
            raise FileNotFoundError(package.path)
        if _sha256_bytes(package.path.read_bytes()) != package.sha256:
            raise RuntimeError(f"SAM3 tracker-memory AOTI package hash mismatch: {package.section}")
        expected_section = f"sam3_tracker_memory_{package.policy}_b{package.batch_size}.pt2"
        if package.section != expected_section or package.package_global != _memory_package_global(
            package.policy, package.batch_size, package.sha256
        ):
            raise RuntimeError("SAM3 tracker-memory package content address mismatch")

    try:
        manifest = json.loads(artifacts.manifest_bytes)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("SAM3 tracker-memory AOTI manifest is invalid") from error
    if (
        manifest.get("schema_version") != 2
        or manifest.get("scope") != "fixed_memory_encoder_soft_hard_b1_b2"
        or manifest.get("artifact_format") != "torch.aot_inductor.package.pt2"
        or manifest.get("implementation")
        != {
            "library": "transformers",
            "model_class": "Sam3TrackerVideoModel",
            "module": "Sam3TrackerVideoMemoryEncoder",
            "license": "Apache-2.0",
            "source_import_policy": "transformers-only",
        }
        or manifest.get("producer") != json.loads(json.dumps(asdict(producer)))
        or manifest.get("input_abi")
        != [
            {
                "policy": policy,
                "tensors": [
                    {"name": name, "dtype": dtype, "shape": list(shape)}
                    for name, dtype, shape in tensors
                ],
            }
            for policy, tensors in expected_abi
        ]
    ):
        raise RuntimeError("SAM3 tracker-memory AOTI manifest contract mismatch")
    manifest_packages = manifest.get("packages")
    if not isinstance(manifest_packages, list) or len(manifest_packages) != 4:
        raise RuntimeError("SAM3 tracker-memory AOTI manifest requires four packages")
    for record, package in zip(manifest_packages, artifacts.packages, strict=True):
        policy_abi = dict(expected_abi)[package.policy]
        expected_inputs = [
            {
                "name": name,
                "dtype": dtype,
                "shape": [package.batch_size if value == "B" else value for value in shape],
            }
            for name, dtype, shape in policy_abi
        ]
        expected_output_shape = [2, 5184, 1, 64] if package.batch_size == 1 else [2, 2, 5184, 64]
        if (
            record.get("policy") != package.policy
            or record.get("batch_size") != package.batch_size
            or record.get("hard_mask") is not package.hard_mask
            or record.get("section") != package.section
            or record.get("sha256") != package.sha256
            or record.get("package_global") != package.package_global
            or record.get("fixed_shape") is not True
            or record.get("inputs") != expected_inputs
            or record.get("outputs")
            != [
                {
                    "name": "packed_memory_and_position",
                    "dtype": "float32",
                    "shape": expected_output_shape,
                }
            ]
        ):
            raise RuntimeError("SAM3 tracker-memory AOTI package manifest mismatch")

    section_names = [name for name, _ in artifacts.bundle_sections]
    expected_sections = [
        "sam3_tracker_memory_aoti_manifest.json",
        *(package.section for package in artifacts.packages),
    ]
    if section_names != expected_sections or len(set(section_names)) != len(section_names):
        raise RuntimeError("SAM3 tracker-memory AOTI bundle sections are incomplete")
    if artifacts.bundle_sections[0][1] != artifacts.manifest_bytes:
        raise RuntimeError("SAM3 tracker-memory AOTI manifest section mismatch")
    return packages


def _validate_resize_artifacts(
    artifacts: HardMaskResizeAotiArtifacts,
    split_artifacts: Sam3TrackerSplitAotiArtifacts,
    inputs: _BuildInputs,
    *,
    aoti_abi_version: int,
) -> None:
    producer = artifacts.producer_abi
    split_producer = split_artifacts.producer_abi
    shared_fields = (
        "torch_version",
        "transformers_version",
        "cuda_version",
        "compute_capability",
        "host_architecture",
        "torch_cxx11_abi",
    )
    if any(
        getattr(producer, field) != getattr(split_producer, field) for field in shared_fields
    ) or (
        producer.torch_version != inputs.torch_version
        or producer.host_architecture != inputs.host_architecture
        or producer.torch_cxx11_abi != inputs.torch_cxx11_abi
        or producer.torch_aoti_abi_version != aoti_abi_version
    ):
        raise RuntimeError("SAM3 hard-mask resize/step AOTI producer ABI mismatch")
    if tuple(package.batch_size for package in artifacts.packages) != (1, 2):
        raise RuntimeError("SAM3 hard-mask resize artifacts do not contain canonical B1/B2")
    for package in artifacts.packages:
        if not package.path.is_file() or _sha256_bytes(package.path.read_bytes()) != package.sha256:
            raise RuntimeError(f"SAM3 hard-mask resize package hash mismatch: {package.section}")
        expected_section = f"sam3_hard_mask_resize_b{package.batch_size}.pt2"
        expected_global = (
            f"trtmc.sam3.tracker_memory.resize.b{package.batch_size}.fixed.{package.sha256[:20]}"
        )
        if package.section != expected_section or package.package_global != expected_global:
            raise RuntimeError("SAM3 hard-mask resize content address mismatch")
    try:
        manifest = json.loads(artifacts.manifest_bytes)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("SAM3 hard-mask resize AOTI manifest is invalid") from error
    expected_input = [{"name": "tracker_mask", "dtype": "float32", "shape": ["B", 1, 288, 288]}]
    expected_output = [
        {
            "name": "resized_tracker_mask",
            "dtype": "float32",
            "shape": ["B", 1, 1008, 1008],
        }
    ]
    if (
        len(manifest) != 11
        or manifest.get("schema_version") != 1
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
        or manifest.get("producer") != json.loads(json.dumps(asdict(producer)))
        or manifest.get("host_architecture") != producer.host_architecture
        or not isinstance(manifest.get("exporter_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", manifest["exporter_sha256"]) is None
        or manifest.get("input_abi") != expected_input
        or manifest.get("output_abi") != expected_output
    ):
        raise RuntimeError("SAM3 hard-mask resize AOTI manifest contract mismatch")
    records = manifest.get("packages")
    if not isinstance(records, list) or len(records) != 2:
        raise RuntimeError("SAM3 hard-mask resize AOTI manifest requires B1/B2 packages")
    for record, package in zip(records, artifacts.packages, strict=True):
        if record != {
            "batch_size": package.batch_size,
            "filename": f"sam3_hard_mask_resize_b{package.batch_size}_{package.sha256}.pt2",
            "section": package.section,
            "sha256": package.sha256,
            "package_global": package.package_global,
        }:
            raise RuntimeError("SAM3 hard-mask resize package manifest mismatch")
    validation = manifest.get("package_validation")
    if (
        not isinstance(validation, dict)
        or validation.get("reference") != "same torch.interpolate eager execution"
        or not math.isfinite(float(validation.get("maximum_absolute_error", -1.0)))
        or float(validation.get("maximum_absolute_error", -1.0)) != 2.0e-5
        or not isinstance(validation.get("cases"), list)
        or len(validation["cases"]) != 2
        or {
            (int(case.get("batch_size", 0)), bool(case.get("passed", False)))
            for case in validation["cases"]
        }
        != {(1, True), (2, True)}
        or any(
            not math.isfinite(float(case.get("maximum_absolute_error", float("inf"))))
            or float(case.get("maximum_absolute_error", float("inf"))) > 2.0e-5
            for case in validation["cases"]
        )
    ):
        raise RuntimeError("SAM3 hard-mask resize package validation mismatch")
    section_names = [name for name, _ in artifacts.bundle_sections]
    expected_sections = [
        "sam3_hard_mask_resize_aoti_manifest.json",
        *(package.section for package in artifacts.packages),
    ]
    if (
        section_names != expected_sections
        or len(set(section_names)) != len(section_names)
        or artifacts.bundle_sections[0][1] != artifacts.manifest_bytes
        or any(
            payload != package.path.read_bytes()
            for (_, payload), package in zip(
                artifacts.bundle_sections[1:], artifacts.packages, strict=True
            )
        )
    ):
        raise RuntimeError("SAM3 hard-mask resize bundle sections are incomplete")


def _aoti_abi_version(library: ctypes.CDLL) -> int:
    symbol = library.trtmc_sam3_tracker_step_aoti_abi_version
    symbol.argtypes = []
    symbol.restype = ctypes.c_uint64
    version = int(symbol())
    if version <= 0:
        raise RuntimeError("SAM3 tracker-step plugin reported an invalid AOTI ABI")
    return version


def _register_split_pipelines(
    library: ctypes.CDLL,
    artifacts: Sam3TrackerSplitAotiArtifacts,
) -> None:
    packages = {(package.stage, package.batch_size): package for package in artifacts.packages}
    register = library.trtmc_sam3_tracker_step_register_pipeline
    register.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int32,
    ]
    register.restype = ctypes.c_int
    for batch_size in (1, 2):
        encoder = packages[("encoder", batch_size)]
        decoder = packages[("decoder", batch_size)]
        status = int(
            register(
                artifacts.pipeline_global(batch_size).encode("utf-8"),
                str(encoder.path).encode("utf-8"),
                str(decoder.path).encode("utf-8"),
                encoder.sha256.encode("ascii"),
                decoder.sha256.encode("ascii"),
                batch_size,
            )
        )
        if status != 0:
            raise RuntimeError(
                f"SAM3 split AOTI B{batch_size} build-time registration failed ({status})"
            )


def _register_memory_packages(
    library: ctypes.CDLL,
    artifacts: Sam3TrackerMemoryAotiArtifacts,
) -> None:
    register = library.trtmc_sam3_tracker_memory_register_package
    register.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int32,
    ]
    register.restype = ctypes.c_int
    for package in artifacts.packages:
        status = int(
            register(
                package.package_global.encode("utf-8"),
                str(package.path).encode("utf-8"),
                package.sha256.encode("ascii"),
                package.policy.encode("ascii"),
                package.batch_size,
            )
        )
        if status != 0:
            raise RuntimeError(
                "SAM3 tracker-memory "
                f"{package.policy} B{package.batch_size} build-time registration failed "
                f"({status})"
            )


def _register_resize_packages(
    library: ctypes.CDLL,
    artifacts: HardMaskResizeAotiArtifacts,
) -> None:
    register = library.trtmc_sam3_tracker_memory_register_package
    register.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int32,
    ]
    register.restype = ctypes.c_int
    for package in artifacts.packages:
        status = int(
            register(
                package.package_global.encode("utf-8"),
                str(package.path).encode("utf-8"),
                package.sha256.encode("ascii"),
                b"resize",
                package.batch_size,
            )
        )
        if status != 0:
            raise RuntimeError(
                f"SAM3 hard-mask resize B{package.batch_size} registration failed ({status})"
            )


def _assemble_runtime_artifacts(
    split_artifacts: Sam3TrackerSplitAotiArtifacts,
    memory_artifacts: Sam3TrackerMemoryAotiArtifacts,
    resize_artifacts: HardMaskResizeAotiArtifacts,
    plugin_library: Path,
    inputs: _BuildInputs,
    *,
    aoti_abi_version: int,
) -> TrackerStepRuntimeArtifacts:
    packages = _validate_split_artifacts(split_artifacts, inputs)
    _validate_memory_artifacts(
        memory_artifacts,
        split_artifacts,
        inputs,
        aoti_abi_version=aoti_abi_version,
    )
    _validate_resize_artifacts(
        resize_artifacts,
        split_artifacts,
        inputs,
        aoti_abi_version=aoti_abi_version,
    )
    plugin_bytes = plugin_library.read_bytes()
    if not plugin_bytes:
        raise RuntimeError("SAM3 tracker-step native plugin is empty")

    package_entries = [
        {
            "stage": package.stage,
            "package_global": package.package_global,
            "section": package.section,
            "sha256": package.sha256,
            "batch_size": package.batch_size,
        }
        for package in split_artifacts.packages
    ]
    pipeline_entries = []
    for batch_size in (1, 2):
        encoder = packages[("encoder", batch_size)]
        decoder = packages[("decoder", batch_size)]
        pipeline_entries.append(
            {
                "global_name": split_artifacts.pipeline_global(batch_size),
                "encoder_sha256": encoder.sha256,
                "decoder_sha256": decoder.sha256,
                "batch_size": batch_size,
            }
        )
    producer = split_artifacts.producer_abi
    manifest = {
        "schema_version": 1,
        "step_scope": TRACKER_STEP_RUNTIME_SCOPE,
        "plugin": {
            "section": TRACKER_STEP_NATIVE_PLUGIN_SECTION,
            "sha256": _sha256_bytes(plugin_bytes),
            "type": "Sam3TrackerStepFfi",
            "version": _PLUGIN_VERSION,
        },
        "producer": {
            "torch_version": producer.torch_version,
            "transformers_version": producer.transformers_version,
            "tvm_ffi_version": inputs.tvm_ffi_version,
            "tensorrt_version": inputs.tensorrt_version,
            "cuda_version": producer.cuda_version,
            "host_architecture": producer.host_architecture,
            "torch_cxx11_abi": producer.torch_cxx11_abi,
            "aoti_abi_version": aoti_abi_version,
            "compute_capability": list(producer.compute_capability),
        },
        "packages": package_entries,
        "pipelines": pipeline_entries,
    }
    runtime_manifest = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    existing_names = [
        name
        for name, _ in (
            *split_artifacts.bundle_sections,
            *memory_artifacts.bundle_sections,
            *resize_artifacts.bundle_sections,
        )
    ]
    if len(set(existing_names)) != len(existing_names) or {
        TRACKER_STEP_NATIVE_PLUGIN_SECTION,
        TRACKER_STEP_RUNTIME_MANIFEST_SECTION,
    } & set(existing_names):
        raise RuntimeError("SAM3 split AOTI bundle sections are not unique")
    bundle_sections = (
        *split_artifacts.bundle_sections,
        *memory_artifacts.bundle_sections,
        *resize_artifacts.bundle_sections,
        (TRACKER_STEP_NATIVE_PLUGIN_SECTION, plugin_bytes),
        (TRACKER_STEP_RUNTIME_MANIFEST_SECTION, runtime_manifest),
    )
    return TrackerStepRuntimeArtifacts(
        plugin_library=plugin_library,
        runtime_manifest=runtime_manifest,
        bundle_sections=bundle_sections,
    )


def prepare_tracker_step_runtime(
    split_artifacts: Sam3TrackerSplitAotiArtifacts,
    memory_artifacts: Sam3TrackerMemoryAotiArtifacts,
    resize_artifacts: HardMaskResizeAotiArtifacts,
    *,
    verbose: bool = False,
) -> TrackerStepRuntimeArtifacts:
    """Build the bridge and bind all step and memory AOTI packages."""

    inputs = _discover_build_inputs()
    plugin_library = ensure_native_plugin(verbose=verbose)
    library = load_native_plugin(plugin_library)
    runtime = _assemble_runtime_artifacts(
        split_artifacts,
        memory_artifacts,
        resize_artifacts,
        plugin_library,
        inputs,
        aoti_abi_version=_aoti_abi_version(library),
    )
    _register_split_pipelines(library, split_artifacts)
    _register_memory_packages(library, memory_artifacts)
    _register_resize_packages(library, resize_artifacts)
    return runtime


def load_native_plugin(path: str | Path) -> ctypes.CDLL:
    """Load the DSO globally and verify the model-owned ABI exports."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    inputs = _discover_build_inputs()
    library = ctypes.CDLL(str(resolved), mode=ctypes.RTLD_GLOBAL)
    version = library.trtmc_sam3_tracker_step_plugin_version
    version.argtypes = []
    version.restype = ctypes.c_char_p
    actual_version = version()
    if actual_version is None or actual_version.decode("utf-8") != _PLUGIN_VERSION:
        raise RuntimeError("SAM3 tracker-step plugin version mismatch")
    aoti_abi = library.trtmc_sam3_tracker_step_aoti_abi_version
    aoti_abi.argtypes = []
    aoti_abi.restype = ctypes.c_uint64
    if int(aoti_abi()) <= 0:
        raise RuntimeError("SAM3 tracker-step plugin reported an invalid AOTI ABI")
    memory_register = library.trtmc_sam3_tracker_memory_register_package
    memory_register.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int32,
    ]
    memory_register.restype = ctypes.c_int
    memory_plugin_version = library.trtmc_sam3_tracker_memory_plugin_version
    memory_plugin_version.argtypes = []
    memory_plugin_version.restype = ctypes.c_char_p
    actual_memory_plugin_version = memory_plugin_version()
    if (
        actual_memory_plugin_version is None
        or actual_memory_plugin_version.decode("utf-8") != _PLUGIN_VERSION
    ):
        raise RuntimeError("SAM3 tracker-memory plugin version mismatch")
    expected_versions = {
        "trtmc_sam3_tracker_step_torch_version": inputs.torch_version,
        "trtmc_sam3_tracker_step_tvm_ffi_version": inputs.tvm_ffi_version,
        "trtmc_sam3_tracker_step_tensorrt_version": inputs.tensorrt_version,
    }
    for symbol_name, expected in expected_versions.items():
        symbol = getattr(library, symbol_name)
        symbol.argtypes = []
        symbol.restype = ctypes.c_char_p
        actual = symbol()
        if actual is None or actual.decode("utf-8") != expected:
            raise RuntimeError(f"SAM3 tracker-step plugin {symbol_name} ABI mismatch")
    cxx11_abi = library.trtmc_sam3_tracker_step_torch_cxx11_abi
    cxx11_abi.argtypes = []
    cxx11_abi.restype = ctypes.c_int32
    if int(cxx11_abi()) != int(inputs.torch_cxx11_abi):
        raise RuntimeError("SAM3 tracker-step plugin C++ ABI mismatch")
    _LOADED_PLUGINS.append(library)
    return library


__all__ = [
    "TRACKER_STEP_NATIVE_PLUGIN_SECTION",
    "TRACKER_STEP_RUNTIME_MANIFEST_SECTION",
    "TRACKER_STEP_RUNTIME_SCOPE",
    "TrackerStepRuntimeArtifacts",
    "ensure_native_plugin",
    "load_native_plugin",
    "prepare_tracker_step_runtime",
]
