# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export SAM3's fixed memory-update graphs as target-specific AOTI packages.

The exporter uses only the Apache-licensed Transformers 5.2 SAM3 tracker
implementation and the local Hugging Face checkpoint.  It emits the four
fixed graphs used by the streaming tracker: soft/hard mask policy crossed
with object batch B1/B2.  No ONNX graph or reference-model source is involved.

The wrapper intentionally owns the soft-mask preparation around Transformers'
``Sam3TrackerVideoMemoryEncoder``.  Hard packages instead receive binary masks
whose ownership was resolved globally at Meta's 1008px tracker grid.  The
graphs reproduce the BF16 memory boundary and publish values back as lossless
FP32 carriers.
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
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


TRACKER_MEMORY_AOTI_MANIFEST_SECTION = "sam3_tracker_memory_aoti_manifest.json"

_PACKAGE_VARIANTS = (
    ("soft", 1, False),
    ("hard", 1, True),
    ("soft", 2, False),
    ("hard", 2, True),
)
_PACKAGE_SECTIONS = {
    (policy, batch_size): f"sam3_tracker_memory_{policy}_b{batch_size}.pt2"
    for policy, batch_size, _ in _PACKAGE_VARIANTS
}
_CACHE_ROOT = Path(tempfile.gettempdir()) / "trtmc-sam3-tracker-memory-aoti"
_TRANSFORMERS_MAJOR_MINOR = (5, 2)
_SUPPORTED_COMPUTE_CAPABILITIES = frozenset({(8, 9), (10, 0), (10, 3), (12, 0)})
_LOW_RES_MASK_SIZE = 288
_TRACKER_IMAGE_SIZE = 1008
_MEMORY_MASK_SIZE = 1152
_FEATURE_SIZE = 72
_FEATURE_CHANNELS = 256
_MEMORY_CHANNELS = 64
_SPATIAL_TOKENS = _FEATURE_SIZE * _FEATURE_SIZE
_SIGMOID_SCALE = 20.0
_SIGMOID_BIAS = -10.0
_GLOBAL_DIGEST_CHARACTERS = 20
_SMOKE_MINIMUM_COSINE = 0.999
_SMOKE_MAXIMUM_RELATIVE_L2 = 0.02


@dataclass(frozen=True)
class MemoryTensorAbi:
    """One tensor in the fixed SAM3 memory-update contract."""

    name: str
    dtype: str
    shape: tuple[int | str, ...]


@dataclass(frozen=True)
class MemoryPolicyInputAbi:
    """Fixed package inputs for one SAM3 memory mask policy."""

    policy: Literal["soft", "hard"]
    tensors: tuple[MemoryTensorAbi, ...]


def _policy_input_abi(policy: Literal["soft", "hard"]) -> MemoryPolicyInputAbi:
    mask = (
        MemoryTensorAbi(
            "owned_tracker_mask",
            "float32",
            ("B", 1, _TRACKER_IMAGE_SIZE, _TRACKER_IMAGE_SIZE),
        )
        if policy == "hard"
        else MemoryTensorAbi(
            "final_mask",
            "float32",
            ("B", 1, _LOW_RES_MASK_SIZE, _LOW_RES_MASK_SIZE),
        )
    )
    return MemoryPolicyInputAbi(
        policy,
        (
            MemoryTensorAbi(
                "tracker_feature_2",
                "float32",
                (1, _FEATURE_CHANNELS, _FEATURE_SIZE, _FEATURE_SIZE),
            ),
            mask,
            MemoryTensorAbi("object_score_logits", "float32", ("B", 1)),
            MemoryTensorAbi("suppress_area_shrinkage", "int32", ("B", 1)),
        ),
    )


TRACKER_MEMORY_INPUT_ABI = (
    _policy_input_abi("soft"),
    _policy_input_abi("hard"),
)


@dataclass(frozen=True)
class MemoryAotiProducerAbi:
    """Target facts that constrain AOTI package compatibility and caching."""

    torch_version: str
    transformers_version: str
    cuda_version: str
    compute_capability: tuple[int, int]
    host_architecture: str
    torch_cxx11_abi: bool
    torch_aoti_abi_version: int


@dataclass(frozen=True)
class MemoryAotiPackage:
    """One immutable, content-addressed memory-update package."""

    policy: Literal["soft", "hard"]
    batch_size: int
    hard_mask: bool
    path: Path
    section: str
    sha256: str
    package_global: str


@dataclass(frozen=True)
class Sam3TrackerMemoryAotiArtifacts:
    """Bundle-ready output of :func:`export_sam3_tracker_memory_aoti`."""

    cache_directory: Path
    packages: tuple[MemoryAotiPackage, ...]
    producer_abi: MemoryAotiProducerAbi
    input_abi: tuple[MemoryPolicyInputAbi, ...]
    manifest_bytes: bytes
    bundle_sections: tuple[tuple[str, bytes], ...]

    def package(self, *, batch_size: int, hard_mask: bool) -> MemoryAotiPackage:
        """Return the unique fixed package for ``batch_size`` and mask policy."""

        for package in self.packages:
            if package.batch_size == batch_size and package.hard_mask is hard_mask:
                return package
        raise ValueError("SAM3 memory AOTI supports only soft/hard B1/B2 packages")


@dataclass(frozen=True)
class _Dependencies:
    torch: Any
    tracker_model_class: Any
    transformers_version: str


@dataclass(frozen=True)
class _StagedPackage:
    policy: Literal["soft", "hard"]
    batch_size: int
    hard_mask: bool
    path: Path


def _version_major_minor(version: str) -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)(?:\.|$)", version)
    if match is None:
        raise RuntimeError(f"Could not parse dependency version {version!r}")
    return int(match.group(1)), int(match.group(2))


def _load_dependencies() -> _Dependencies:
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        modeling = importlib.import_module(
            "transformers.models.sam3_tracker_video.modeling_sam3_tracker_video"
        )
    except ImportError as error:
        raise RuntimeError("SAM3 memory AOTI export requires Torch and Transformers 5.2") from error

    version = str(getattr(transformers, "__version__", "") or "")
    if _version_major_minor(version) != _TRANSFORMERS_MAJOR_MINOR:
        raise RuntimeError(
            f"SAM3 memory AOTI export requires Transformers 5.2.x; found {version or 'unknown'}"
        )
    if not hasattr(torch, "export") or not hasattr(torch.export, "export"):
        raise RuntimeError("Torch does not provide torch.export.export")
    inductor = getattr(torch, "_inductor", None)
    if (
        inductor is None
        or not hasattr(inductor, "aoti_compile_and_package")
        or not hasattr(inductor, "aoti_load_package")
    ):
        raise RuntimeError("Torch does not provide AOTI package compilation and loading")
    return _Dependencies(
        torch=torch,
        tracker_model_class=modeling.Sam3TrackerVideoModel,
        transformers_version=version,
    )


def _torch_aoti_abi_version(torch: Any) -> int:
    torch_root = Path(torch.__file__).resolve().parent
    library = torch_root / "lib" / "libtorch_cpu.so"
    if not library.is_file():
        raise RuntimeError(f"Torch AOTI ABI library is missing: {library}")
    try:
        symbol = ctypes.CDLL(os.fspath(library)).aoti_torch_abi_version
    except (AttributeError, OSError) as error:
        raise RuntimeError("Torch does not expose aoti_torch_abi_version") from error
    symbol.argtypes = []
    symbol.restype = ctypes.c_uint64
    return int(symbol())


def _producer_abi(dependencies: _Dependencies, device_index: int) -> MemoryAotiProducerAbi:
    torch = dependencies.torch
    if not bool(torch.cuda.is_available()):
        raise RuntimeError("SAM3 memory AOTI export requires a CUDA GPU")
    if device_index < 0 or device_index >= int(torch.cuda.device_count()):
        raise RuntimeError(f"CUDA device {device_index} is not available")
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(device_index))
    if capability not in _SUPPORTED_COMPUTE_CAPABILITIES:
        raise RuntimeError(
            "SAM3 memory AOTI export does not support compute capability "
            f"{capability[0]}.{capability[1]}"
        )
    cuda_version = str(getattr(torch.version, "cuda", "") or "")
    if not cuda_version:
        raise RuntimeError("Torch does not report a CUDA build version")
    cxx11_abi = getattr(getattr(torch, "_C", None), "_GLIBCXX_USE_CXX11_ABI", None)
    if cxx11_abi is None:
        raise RuntimeError("Torch does not report its C++11 ABI")
    return MemoryAotiProducerAbi(
        torch_version=str(getattr(torch, "__version__", "") or "unknown"),
        transformers_version=dependencies.transformers_version,
        cuda_version=cuda_version,
        compute_capability=(capability[0], capability[1]),
        host_architecture=platform.machine(),
        torch_cxx11_abi=bool(cxx11_abi),
        torch_aoti_abi_version=_torch_aoti_abi_version(torch),
    )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_source_digest(model_dir: Path) -> str:
    config = model_dir / "config.json"
    weight_files = sorted(model_dir.rglob("*.safetensors"))
    if not config.is_file():
        raise FileNotFoundError(f"SAM3 model directory is missing {config}")
    if not weight_files:
        raise FileNotFoundError(
            f"SAM3 model directory {model_dir} does not contain safetensors weights"
        )
    digest = hashlib.sha256()
    for path in (config, *weight_files):
        if not path.is_file():
            raise RuntimeError(f"SAM3 model input is not a regular file: {path}")
        digest.update(path.relative_to(model_dir).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_hash_file(path)))
    return digest.hexdigest()


def _exporter_source_digest() -> str:
    return _hash_file(Path(__file__).resolve())


def _output_shape(batch_size: int) -> tuple[int, ...]:
    if batch_size == 1:
        return (2, _SPATIAL_TOKENS, 1, _MEMORY_CHANNELS)
    if batch_size == 2:
        return (2, 2, _SPATIAL_TOKENS, _MEMORY_CHANNELS)
    raise ValueError("SAM3 memory AOTI supports only batch sizes 1 and 2")


def _variant_contract(policy: str, batch_size: int) -> dict[str, Any]:
    if policy not in {"soft", "hard"} or batch_size not in {1, 2}:
        raise ValueError(f"Unsupported SAM3 memory package {policy!r} B{batch_size}")
    input_abi = next(value.tensors for value in TRACKER_MEMORY_INPUT_ABI if value.policy == policy)
    inputs = []
    for tensor in input_abi:
        shape = [batch_size if value == "B" else value for value in tensor.shape]
        inputs.append({"name": tensor.name, "dtype": tensor.dtype, "shape": shape})
    return {
        "policy": policy,
        "batch_size": batch_size,
        "fixed_shape": True,
        "inputs": inputs,
        "outputs": [
            {
                "name": "packed_memory_and_position",
                "dtype": "float32",
                "shape": list(_output_shape(batch_size)),
            }
        ],
    }


def _manifest_mask_policy() -> dict[str, Any]:
    return {
        "soft": ("288 bilinear 1152, clamp rejected rows to <=-10, sigmoid, scale 20, bias -10"),
        "hard": (
            "globally owned binary FP32 1008, scale 20, bias -10, "
            "antialiased bilinear 1152; suppression input ignored"
        ),
        "b1_layout": [2, _SPATIAL_TOKENS, 1, _MEMORY_CHANNELS],
        "b2_layout": [2, 2, _SPATIAL_TOKENS, _MEMORY_CHANNELS],
        "stored_precision": "bfloat16 rounded then promoted to float32 carrier",
    }


def _cache_key(
    model_digest: str,
    exporter_digest: str,
    producer: MemoryAotiProducerAbi,
) -> str:
    contract = {
        "schema_version": 2,
        "model_sha256": model_digest,
        "exporter_sha256": exporter_digest,
        "producer": asdict(producer),
        "variants": [
            _variant_contract(policy, batch_size) for policy, batch_size, _ in _PACKAGE_VARIANTS
        ],
        "mask_policy": {
            "soft_resize": _MEMORY_MASK_SIZE,
            "soft_area_shrinkage": "clamp rejected 1152-grid logits to <= -10",
            "hard_input": "globally owned binary FP32 mask",
            "hard_input_grid": _TRACKER_IMAGE_SIZE,
            "hard_area_shrinkage": "fourth input accepted and ignored",
            "memory_grid": _MEMORY_MASK_SIZE,
            "sigmoid_scale": _SIGMOID_SCALE,
            "sigmoid_bias": _SIGMOID_BIAS,
        },
    }
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@contextmanager
def _exclusive_cache_lock(cache_root: Path, key: str) -> Iterator[None]:
    cache_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not stat.S_ISDIR(cache_root.lstat().st_mode):
        raise RuntimeError("SAM3 memory AOTI cache root is not a directory")
    cache_root.chmod(0o700)
    descriptor = os.open(
        cache_root / f".{key}.lock",
        os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    with os.fdopen(descriptor, "a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield


def _load_tracker_model(dependencies: _Dependencies, model_dir: Path, device: Any) -> Any:
    torch = dependencies.torch
    loaded = dependencies.tracker_model_class.from_pretrained(
        model_dir,
        local_files_only=True,
        remove_vision_encoder=True,
        dtype=torch.float32,
        output_loading_info=True,
    )
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise RuntimeError("Transformers did not return SAM3 loading diagnostics")
    model, loading_info = loaded
    missing = tuple(str(key) for key in loading_info.get("missing_keys", ()))
    unexpected = tuple(str(key) for key in loading_info.get("unexpected_keys", ()))
    mismatched = tuple(loading_info.get("mismatched_keys", ()))
    errors = tuple(loading_info.get("error_msgs", ()))
    allowed_unexpected = tuple(
        key
        for key in unexpected
        if key.startswith("vision_encoder.") or key.startswith("tracker_model.vision_encoder.")
    )
    if missing or mismatched or errors or len(allowed_unexpected) != len(unexpected):
        raise RuntimeError(
            "SAM3 tracker memory weights did not load exactly: "
            f"missing={missing}, unexpected={unexpected}, "
            f"mismatched={mismatched}, errors={errors}"
        )

    required = ("memory_encoder", "occlusion_spatial_embedding_parameter", "config")
    absent = tuple(name for name in required if not hasattr(model, name))
    if absent:
        raise RuntimeError(f"SAM3 tracker model is missing memory modules: {absent}")
    if model.occlusion_spatial_embedding_parameter is None:
        raise RuntimeError("SAM3 tracker model has no occlusion spatial embedding")

    expected_config = {
        "memory_encoder_hidden_size": _FEATURE_CHANNELS,
        "memory_encoder_output_channels": _MEMORY_CHANNELS,
        "mask_downsampler_total_stride": 16,
        "memory_fuser_num_layers": 2,
        "sigmoid_scale_for_mem_enc": _SIGMOID_SCALE,
        "sigmoid_bias_for_mem_enc": _SIGMOID_BIAS,
    }
    mismatched_config = {
        name: (getattr(model.config, name, None), expected)
        for name, expected in expected_config.items()
        if getattr(model.config, name, None) != expected
    }
    if mismatched_config:
        raise RuntimeError(f"SAM3 tracker memory configuration is unsupported: {mismatched_config}")

    model = model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _prepare_memory_mask(
    torch: Any,
    memory_mask: Any,
    suppress_area_shrinkage: Any,
    *,
    batch_size: int,
    hard_mask: bool,
) -> Any:
    functional = torch.nn.functional
    if hard_mask:
        mask = memory_mask.float()
    else:
        resized_logits = functional.interpolate(
            memory_mask.float(),
            size=(_MEMORY_MASK_SIZE, _MEMORY_MASK_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        rejected = suppress_area_shrinkage.reshape(batch_size, 1, 1, 1) > 0
        resized_logits = torch.where(
            rejected,
            torch.clamp(resized_logits, max=-10.0),
            resized_logits,
        )
        mask = torch.sigmoid(resized_logits)

    mask = mask * _SIGMOID_SCALE + _SIGMOID_BIAS
    if hard_mask:
        # Transformers' memory encoder starts at the convolutional
        # downsampler.  Reproduce the reference SimpleMaskDownSampler's
        # internal 1008 -> 1152 antialiased interpolation explicitly.
        mask = functional.interpolate(
            mask.float(),
            size=(_MEMORY_MASK_SIZE, _MEMORY_MASK_SIZE),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    return mask


def _make_memory_module(
    torch: Any,
    model: Any,
    *,
    batch_size: int,
    hard_mask: bool,
) -> Any:
    class _FixedMemoryEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.memory_encoder = model.memory_encoder
            self.occlusion_embedding = model.occlusion_spatial_embedding_parameter

        def forward(
            self,
            vision_features,
            memory_mask,
            object_score_logits,
            suppress_area_shrinkage,
        ):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                mask = _prepare_memory_mask(
                    torch,
                    memory_mask,
                    suppress_area_shrinkage,
                    batch_size=batch_size,
                    hard_mask=hard_mask,
                )
                expanded_features = vision_features.expand(batch_size, -1, -1, -1)
                memory, position = self.memory_encoder(expanded_features, mask)
                memory = memory.clone()
                position = position.to(memory.dtype).clone()

                appearing = (object_score_logits > 0).float().reshape(batch_size, 1, 1, 1)
                memory += (1.0 - appearing) * self.occlusion_embedding[..., None, None].expand_as(
                    memory
                )

                memory = memory.to(torch.bfloat16).flatten(2).transpose(1, 2)
                position = position.to(torch.bfloat16).flatten(2).transpose(1, 2)
                if batch_size == 1:
                    memory = memory.transpose(0, 1)
                    position = position.transpose(0, 1)

                # TensorRT's recurrent state remains an FP32 public carrier;
                # these casts are lossless because both tensors were rounded
                # through the BF16 state boundary immediately above.
                return torch.stack((memory.float(), position.float()), dim=0).contiguous()

    return _FixedMemoryEncoder().eval()


def _example_inputs(
    torch: Any,
    *,
    batch_size: int,
    hard_mask: bool,
    device: Any,
) -> tuple[Any, ...]:
    feature = torch.randn(
        1,
        _FEATURE_CHANNELS,
        _FEATURE_SIZE,
        _FEATURE_SIZE,
        dtype=torch.float32,
        device=device,
    )
    if hard_mask:
        # The runtime resolves ownership once across the complete object axis.
        # Exercise the package with mutually exclusive binary tracker-grid rows.
        owners = torch.randint(
            0,
            batch_size + 1,
            (1, 1, _TRACKER_IMAGE_SIZE, _TRACKER_IMAGE_SIZE),
            dtype=torch.int64,
            device=device,
        )
        memory_mask = torch.cat(
            tuple((owners == object_index + 1).float() for object_index in range(batch_size)),
            dim=0,
        )
    else:
        memory_mask = torch.randn(
            batch_size,
            1,
            _LOW_RES_MASK_SIZE,
            _LOW_RES_MASK_SIZE,
            dtype=torch.float32,
            device=device,
        )
    if batch_size == 1:
        score_value = -1.0 if hard_mask else 1.0
        scores = torch.tensor([[score_value]], dtype=torch.float32, device=device)
    else:
        scores = torch.tensor([[1.0], [-1.0]], dtype=torch.float32, device=device)
    if batch_size == 1:
        suppress_area_shrinkage = torch.tensor(
            [[0 if hard_mask else 1]],
            dtype=torch.int32,
            device=device,
        )
    else:
        suppress_area_shrinkage = torch.tensor(
            [[1], [0]],
            dtype=torch.int32,
            device=device,
        )
    return feature, memory_mask.contiguous(), scores, suppress_area_shrinkage


def _compile_one_package(
    torch: Any,
    module: Any,
    example: tuple[Any, ...],
    output: Path,
) -> None:
    with torch.inference_mode():
        exported = torch.export.export(module, example, strict=False)
    torch._inductor.aoti_compile_and_package(
        exported,
        package_path=os.fspath(output),
        inductor_configs={
            "max_autotune": True,
            "triton.cudagraphs": False,
            "aot_inductor.use_runtime_constant_folding": False,
        },
    )
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"AOTI did not produce the SAM3 memory package {output}")


def _unwrap_aoti_output(value: Any) -> Any:
    while isinstance(value, (tuple, list)) and len(value) == 1:
        value = value[0]
    return value


def _numerical_metrics(torch: Any, actual: Any, expected: Any) -> dict[str, float]:
    if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
        raise RuntimeError("SAM3 memory AOTI smoke output is not a tensor")
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise RuntimeError(
            "SAM3 memory AOTI smoke output contract mismatch: "
            f"actual={tuple(actual.shape)}/{actual.dtype}, "
            f"expected={tuple(expected.shape)}/{expected.dtype}"
        )
    if not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(expected).all()):
        raise RuntimeError("SAM3 memory AOTI smoke output contains non-finite values")
    actual_values = actual.float().reshape(-1)
    expected_values = expected.float().reshape(-1)
    difference = actual_values - expected_values
    expected_norm = torch.linalg.vector_norm(expected_values)
    relative_l2 = torch.linalg.vector_norm(difference) / torch.clamp(expected_norm, min=1e-12)
    cosine = torch.nn.functional.cosine_similarity(
        actual_values,
        expected_values,
        dim=0,
        eps=1e-12,
    )
    return {
        "cosine": float(cosine.item()),
        "relative_l2": float(relative_l2.item()),
        "maximum_absolute_error": float(difference.abs().max().item()),
    }


def _validate_package(
    torch: Any,
    *,
    policy: Literal["soft", "hard"],
    batch_size: int,
    hard_mask: bool,
    module: Any,
    package: Path,
    inputs: tuple[Any, ...],
    device: Any,
) -> dict[str, Any]:
    loaded = torch._inductor.aoti_load_package(
        os.fspath(package),
        device_index=int(device.index or 0),
    )
    try:
        with torch.inference_mode():
            expected = _unwrap_aoti_output(module(*inputs))
            actual = _unwrap_aoti_output(loaded(*inputs))
            torch.cuda.synchronize(inputs[0].device)
        if tuple(actual.shape) != _output_shape(batch_size) or actual.dtype != torch.float32:
            raise RuntimeError(
                f"SAM3 memory {policy} B{batch_size} returned {tuple(actual.shape)}/{actual.dtype}"
            )
        metrics = _numerical_metrics(torch, actual, expected)
        plane_metrics = {
            "memory": _numerical_metrics(torch, actual[0], expected[0]),
            "position": _numerical_metrics(torch, actual[1], expected[1]),
        }
        failing = {
            name: values
            for name, values in {"packed": metrics, **plane_metrics}.items()
            if values["cosine"] < _SMOKE_MINIMUM_COSINE
            or values["relative_l2"] > _SMOKE_MAXIMUM_RELATIVE_L2
        }
        if failing:
            raise RuntimeError(
                f"SAM3 memory {policy} B{batch_size} AOTI/eager smoke failed: {failing}"
            )
        return {
            "policy": policy,
            "batch_size": batch_size,
            "hard_mask": hard_mask,
            **metrics,
            "planes": plane_metrics,
            "passed": True,
        }
    finally:
        del loaded


def _compile_fixed_packages(
    dependencies: _Dependencies,
    model: Any,
    staging: Path,
    device: Any,
) -> tuple[tuple[_StagedPackage, ...], tuple[dict[str, Any], ...]]:
    torch = dependencies.torch
    staged: list[_StagedPackage] = []
    validation: list[dict[str, Any]] = []
    for policy, batch_size, hard_mask in _PACKAGE_VARIANTS:
        module = _make_memory_module(
            torch,
            model,
            batch_size=batch_size,
            hard_mask=hard_mask,
        )
        inputs = _example_inputs(
            torch,
            batch_size=batch_size,
            hard_mask=hard_mask,
            device=device,
        )
        package = staging / f"memory_{policy}_b{batch_size}.pt2"
        _compile_one_package(torch, module, inputs, package)
        validation.append(
            _validate_package(
                torch,
                policy=policy,
                batch_size=batch_size,
                hard_mask=hard_mask,
                module=module,
                package=package,
                inputs=inputs,
                device=device,
            )
        )
        staged.append(_StagedPackage(policy, batch_size, hard_mask, package))
        del module, inputs
        torch.cuda.empty_cache()
    return tuple(staged), tuple(validation)


def _package_global(policy: str, batch_size: int, digest: str) -> str:
    if policy not in {"soft", "hard"} or batch_size not in {1, 2}:
        raise ValueError(f"Unsupported SAM3 memory package {policy!r} B{batch_size}")
    return (
        f"trtmc.sam3.tracker_memory.{policy}.b{batch_size}.fixed."
        f"{digest[:_GLOBAL_DIGEST_CHARACTERS]}"
    )


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("utf-8")


def _json_value(value: Any) -> Any:
    return json.loads(_canonical_json(value))


def _publish_staged_packages(
    staged: Sequence[_StagedPackage],
    validation: Sequence[dict[str, Any]],
    staging: Path,
    producer: MemoryAotiProducerAbi,
    model_digest: str,
    exporter_digest: str,
) -> bytes:
    expected_keys = {
        (policy, batch_size, hard_mask) for policy, batch_size, hard_mask in _PACKAGE_VARIANTS
    }
    if {(item.policy, item.batch_size, item.hard_mask) for item in staged} != expected_keys:
        raise RuntimeError("SAM3 memory AOTI export did not produce all four packages")

    records = []
    by_key = {(item.policy, item.batch_size, item.hard_mask): item for item in staged}
    for policy, batch_size, hard_mask in _PACKAGE_VARIANTS:
        source = by_key[(policy, batch_size, hard_mask)].path
        digest = _hash_file(source)
        destination = staging / f"sam3_tracker_memory_{policy}_b{batch_size}_{digest}.pt2"
        if source != destination:
            os.replace(source, destination)
        records.append(
            {
                **_variant_contract(policy, batch_size),
                "hard_mask": hard_mask,
                "filename": destination.name,
                "section": _PACKAGE_SECTIONS[(policy, batch_size)],
                "sha256": digest,
                "package_global": _package_global(policy, batch_size, digest),
            }
        )

    manifest = {
        "schema_version": 2,
        "scope": "fixed_memory_encoder_soft_hard_b1_b2",
        "artifact_format": "torch.aot_inductor.package.pt2",
        "implementation": {
            "library": "transformers",
            "model_class": "Sam3TrackerVideoModel",
            "module": "Sam3TrackerVideoMemoryEncoder",
            "license": "Apache-2.0",
            "source_import_policy": "transformers-only",
        },
        "model_sha256": model_digest,
        "exporter_sha256": exporter_digest,
        "producer": asdict(producer),
        "input_abi": [asdict(value) for value in TRACKER_MEMORY_INPUT_ABI],
        "mask_policy": _manifest_mask_policy(),
        "packages": records,
        "package_validation": {
            "reference": "same Transformers module eager execution before cache publication",
            "minimum_cosine": _SMOKE_MINIMUM_COSINE,
            "maximum_relative_l2": _SMOKE_MAXIMUM_RELATIVE_L2,
            "cases": list(validation),
        },
    }
    manifest_bytes = _canonical_json(manifest)
    (staging / TRACKER_MEMORY_AOTI_MANIFEST_SECTION).write_bytes(manifest_bytes)
    return manifest_bytes


def _regular_file(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except FileNotFoundError:
        return False


def _validate_cached_directory(
    cache_directory: Path,
    *,
    expected_model_digest: str | None = None,
    expected_exporter_digest: str | None = None,
    expected_producer: MemoryAotiProducerAbi | None = None,
) -> dict[str, Any] | None:
    manifest_path = cache_directory / TRACKER_MEMORY_AOTI_MANIFEST_SECTION
    if not _regular_file(manifest_path):
        return None
    try:
        manifest = json.loads(manifest_path.read_bytes())
        if (
            manifest["schema_version"] != 2
            or manifest["scope"] != "fixed_memory_encoder_soft_hard_b1_b2"
            or manifest["artifact_format"] != "torch.aot_inductor.package.pt2"
            or manifest["implementation"]
            != {
                "library": "transformers",
                "model_class": "Sam3TrackerVideoModel",
                "module": "Sam3TrackerVideoMemoryEncoder",
                "license": "Apache-2.0",
                "source_import_policy": "transformers-only",
            }
            or (
                expected_model_digest is not None
                and manifest["model_sha256"] != expected_model_digest
            )
            or (
                expected_exporter_digest is not None
                and manifest["exporter_sha256"] != expected_exporter_digest
            )
            or (
                expected_producer is not None
                and manifest["producer"] != _json_value(asdict(expected_producer))
            )
            or manifest["input_abi"]
            != _json_value([asdict(value) for value in TRACKER_MEMORY_INPUT_ABI])
            or manifest["mask_policy"] != _manifest_mask_policy()
        ):
            return None

        packages = manifest["packages"]
        expected_keys = {(policy, batch_size) for policy, batch_size, _ in _PACKAGE_VARIANTS}
        if (
            len(packages) != len(_PACKAGE_VARIANTS)
            or {(str(item["policy"]), int(item["batch_size"])) for item in packages}
            != expected_keys
        ):
            return None
        for item in packages:
            policy = str(item["policy"])
            batch_size = int(item["batch_size"])
            hard_mask = policy == "hard"
            digest = str(item["sha256"])
            filename = str(item["filename"])
            if (
                item
                != {
                    **_variant_contract(policy, batch_size),
                    "hard_mask": hard_mask,
                    "filename": filename,
                    "section": _PACKAGE_SECTIONS[(policy, batch_size)],
                    "sha256": digest,
                    "package_global": _package_global(policy, batch_size, digest),
                }
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or filename != f"sam3_tracker_memory_{policy}_b{batch_size}_{digest}.pt2"
            ):
                return None
            package = cache_directory / filename
            if not _regular_file(package) or _hash_file(package) != digest:
                return None

        validation = manifest["package_validation"]
        expected_cases = {
            (policy, batch_size, hard_mask) for policy, batch_size, hard_mask in _PACKAGE_VARIANTS
        }
        actual_cases = {
            (str(case["policy"]), int(case["batch_size"]), bool(case["hard_mask"]))
            for case in validation["cases"]
            if case["passed"] is True
        }
        numerical_cases_are_valid = all(
            bool(case["passed"])
            and math.isfinite(float(case["cosine"]))
            and math.isfinite(float(case["relative_l2"]))
            and math.isfinite(float(case["maximum_absolute_error"]))
            and float(case["cosine"]) >= _SMOKE_MINIMUM_COSINE
            and float(case["relative_l2"]) <= _SMOKE_MAXIMUM_RELATIVE_L2
            and set(case["planes"]) == {"memory", "position"}
            and all(
                math.isfinite(float(metrics["cosine"]))
                and math.isfinite(float(metrics["relative_l2"]))
                and math.isfinite(float(metrics["maximum_absolute_error"]))
                and float(metrics["cosine"]) >= _SMOKE_MINIMUM_COSINE
                and float(metrics["relative_l2"]) <= _SMOKE_MAXIMUM_RELATIVE_L2
                for metrics in case["planes"].values()
            )
            for case in validation["cases"]
        )
        if (
            validation["reference"]
            != "same Transformers module eager execution before cache publication"
            or float(validation["minimum_cosine"]) != _SMOKE_MINIMUM_COSINE
            or float(validation["maximum_relative_l2"]) != _SMOKE_MAXIMUM_RELATIVE_L2
            or actual_cases != expected_cases
            or len(validation["cases"]) != len(_PACKAGE_VARIANTS)
            or not numerical_cases_are_valid
        ):
            return None
        return manifest
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _artifacts_from_cache(cache_directory: Path) -> Sam3TrackerMemoryAotiArtifacts:
    manifest = _validate_cached_directory(cache_directory)
    if manifest is None:
        raise RuntimeError(f"Invalid SAM3 memory AOTI cache {cache_directory}")
    by_key = {(str(item["policy"]), int(item["batch_size"])): item for item in manifest["packages"]}
    packages = tuple(
        MemoryAotiPackage(
            policy=policy,
            batch_size=batch_size,
            hard_mask=hard_mask,
            path=cache_directory / by_key[(policy, batch_size)]["filename"],
            section=by_key[(policy, batch_size)]["section"],
            sha256=by_key[(policy, batch_size)]["sha256"],
            package_global=by_key[(policy, batch_size)]["package_global"],
        )
        for policy, batch_size, hard_mask in _PACKAGE_VARIANTS
    )
    producer = manifest["producer"]
    producer_abi = MemoryAotiProducerAbi(
        torch_version=producer["torch_version"],
        transformers_version=producer["transformers_version"],
        cuda_version=producer["cuda_version"],
        compute_capability=tuple(producer["compute_capability"]),
        host_architecture=producer["host_architecture"],
        torch_cxx11_abi=producer["torch_cxx11_abi"],
        torch_aoti_abi_version=producer["torch_aoti_abi_version"],
    )
    manifest_bytes = (cache_directory / TRACKER_MEMORY_AOTI_MANIFEST_SECTION).read_bytes()
    bundle_sections = (
        (TRACKER_MEMORY_AOTI_MANIFEST_SECTION, manifest_bytes),
        *((package.section, package.path.read_bytes()) for package in packages),
    )
    return Sam3TrackerMemoryAotiArtifacts(
        cache_directory=cache_directory,
        packages=packages,
        producer_abi=producer_abi,
        input_abi=TRACKER_MEMORY_INPUT_ABI,
        manifest_bytes=manifest_bytes,
        bundle_sections=bundle_sections,
    )


def export_sam3_tracker_memory_aoti(
    model_dir: str | Path,
    *,
    device_index: int = 0,
    cache_dir: str | Path | None = None,
) -> Sam3TrackerMemoryAotiArtifacts:
    """Export or reuse SAM3's fixed soft/hard B1/B2 memory packages.

    ``model_dir`` is the normal local Transformers SAM3 model directory.  The
    supported graph policy is deliberately fixed and has no tuning flags, so
    a default build always receives the same accuracy-oriented implementation.
    Packages are compiled on, and bound to, the selected CUDA architecture.
    """

    resolved_model_dir = Path(model_dir).resolve()
    if not resolved_model_dir.is_dir():
        raise FileNotFoundError(resolved_model_dir)
    dependencies = _load_dependencies()
    producer = _producer_abi(dependencies, device_index)
    model_digest = _model_source_digest(resolved_model_dir)
    exporter_digest = _exporter_source_digest()
    key = _cache_key(model_digest, exporter_digest, producer)
    cache_root = Path(cache_dir).resolve() if cache_dir is not None else _CACHE_ROOT
    cache_directory = cache_root / key
    validation = {
        "expected_model_digest": model_digest,
        "expected_exporter_digest": exporter_digest,
        "expected_producer": producer,
    }
    if _validate_cached_directory(cache_directory, **validation) is not None:
        return _artifacts_from_cache(cache_directory)

    with _exclusive_cache_lock(cache_root, key):
        if _validate_cached_directory(cache_directory, **validation) is not None:
            return _artifacts_from_cache(cache_directory)
        if cache_directory.is_symlink():
            cache_directory.unlink()
        elif cache_directory.exists():
            shutil.rmtree(cache_directory)
        staging = Path(tempfile.mkdtemp(prefix=f".{key}.build-", dir=cache_root))
        try:
            torch = dependencies.torch
            device = torch.device(f"cuda:{device_index}")
            previous_device = int(torch.cuda.current_device())
            model = None
            try:
                torch.cuda.set_device(device_index)
                with torch.random.fork_rng(devices=[device_index]):
                    torch.manual_seed(20260717)
                    model = _load_tracker_model(
                        dependencies,
                        resolved_model_dir,
                        device,
                    )
                    staged, package_validation = _compile_fixed_packages(
                        dependencies,
                        model,
                        staging,
                        device,
                    )
                    _publish_staged_packages(
                        staged,
                        package_validation,
                        staging,
                        producer,
                        model_digest,
                        exporter_digest,
                    )
                    os.replace(staging, cache_directory)
            finally:
                model = None
                torch.cuda.empty_cache()
                torch.cuda.set_device(previous_device)
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
    return _artifacts_from_cache(cache_directory)


__all__ = [
    "MemoryAotiPackage",
    "MemoryAotiProducerAbi",
    "MemoryPolicyInputAbi",
    "MemoryTensorAbi",
    "Sam3TrackerMemoryAotiArtifacts",
    "TRACKER_MEMORY_AOTI_MANIFEST_SECTION",
    "TRACKER_MEMORY_INPUT_ABI",
    "export_sam3_tracker_memory_aoti",
]
