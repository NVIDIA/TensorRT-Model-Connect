# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export the SAM3 recurrent tracker as split, hardware-specific AOTI packages.

The exported callable keeps the customer's ten tensor carriers as its public
contract, but mirrors the upstream compilation boundary internally:

* a dynamic memory-attention encoder for M=1..10 and P=1..19; and
* a static recurrent mask decoder for each supported object batch (B1/B2).

The encoder and decoder remain separate AOTI packages so their layout boundary
is explicit.  The runtime can register one pipeline global per batch whose
identity commits to both package hashes.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib
import json
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


TRACKER_SPLIT_AOTI_MANIFEST_SECTION = "sam3_tracker_split_aoti_manifest.json"

_PACKAGE_SECTIONS = {
    ("encoder", 1): "sam3_tracker_encoder_b1_dynamic.pt2",
    ("decoder", 1): "sam3_tracker_decoder_b1_static.pt2",
    ("encoder", 2): "sam3_tracker_encoder_b2_dynamic.pt2",
    ("decoder", 2): "sam3_tracker_decoder_b2_static.pt2",
}
_PACKAGE_ORDER = (("encoder", 1), ("decoder", 1), ("encoder", 2), ("decoder", 2))
_CACHE_ROOT = Path(tempfile.gettempdir()) / "trtmc-sam3-tracker-split-aoti"
_TRANSFORMERS_MAJOR_MINOR = (5, 2)
_SUPPORTED_COMPUTE_CAPABILITIES = frozenset({(8, 9), (10, 0), (10, 3), (12, 0)})
_MEMORY_BOUNDS = (1, 10)
_POINTER_BOUNDS = (1, 19)
_REPRESENTATIVE_MEMORY_COUNT = 4
_REPRESENTATIVE_POINTER_COUNT = 3
_SPATIAL_SIZE = 72 * 72
_MASK_SIZE = 288 * 288
_POINTER_WIDTH = 256
_PACKED_WIDTH = _MASK_SIZE + _POINTER_WIDTH + 2
_NO_OBJECT_SCORE = -1024.0
_PIPELINE_DIGEST_DOMAIN = b"trtmc.sam3.tracker_step.split_aoti.v1\0"
_GLOBAL_DIGEST_CHARACTERS = 20
_SMOKE_ENCODER_SHAPES = ((1, 1), (4, 3), (10, 19))
_SMOKE_MINIMUM_COSINE = 0.999
_SMOKE_MAXIMUM_RELATIVE_L2 = 0.02
_SMOKE_MINIMUM_BINARY_MASK_AGREEMENT = 0.995


@dataclass(frozen=True)
class TrackerTensorAbi:
    """One tensor in the external recurrent tracker carrier contract."""

    name: str
    dtype: str
    shape: tuple[int | str, ...]


TRACKER_TEN_CARRIER_ABI = (
    # The first two carriers are the outputs of mask_decoder.conv_s0/conv_s1,
    # not the raw 256-channel tracker-neck maps.  They remain FP32 at the
    # public TRT boundary and are rounded to Meta's BF16 decoder boundary in
    # _StaticRecurrentDecoder.forward.
    TrackerTensorAbi("tracker_feature_0", "float32", (1, 32, 288, 288)),
    TrackerTensorAbi("tracker_feature_1", "float32", (1, 64, 144, 144)),
    TrackerTensorAbi("tracker_feature_2", "float32", (1, 256, 72, 72)),
    TrackerTensorAbi("tracker_position_2", "float32", (1, 256, 72, 72)),
    TrackerTensorAbi("memory_features", "float32", ("B", "M", _SPATIAL_SIZE, 64)),
    TrackerTensorAbi("memory_position", "float32", ("B", "M", _SPATIAL_SIZE, 64)),
    TrackerTensorAbi("memory_temporal_offsets", "int32", ("B", "M")),
    TrackerTensorAbi("object_pointers", "float32", ("B", "P", _POINTER_WIDTH)),
    TrackerTensorAbi("object_pointer_temporal_offsets", "int32", ("B", "P")),
    TrackerTensorAbi("max_object_pointers_to_use", "int32", (1,)),
)


@dataclass(frozen=True)
class TrackerAotiProducerAbi:
    """ABI facts that affect AOTI package compatibility and cache identity."""

    torch_version: str
    transformers_version: str
    cuda_version: str
    compute_capability: tuple[int, int]
    host_architecture: str
    torch_cxx11_abi: bool


@dataclass(frozen=True)
class TrackerAotiPackage:
    """One content-addressed encoder or decoder package."""

    stage: Literal["encoder", "decoder"]
    batch_size: int
    path: Path
    section: str
    sha256: str
    package_global: str


@dataclass(frozen=True)
class Sam3TrackerSplitAotiArtifacts:
    """Immutable handoff from the exporter to SAM3's bridge and bundle builder."""

    cache_directory: Path
    packages: tuple[TrackerAotiPackage, ...]
    pipeline_global_b1: str
    pipeline_global_b2: str
    producer_abi: TrackerAotiProducerAbi
    carrier_abi: tuple[TrackerTensorAbi, ...]
    manifest_bytes: bytes
    bundle_sections: tuple[tuple[str, bytes], ...]

    def pipeline_global(self, batch_size: int) -> str:
        """Return the paired encoder+decoder pipeline global for B1 or B2."""

        if batch_size == 1:
            return self.pipeline_global_b1
        if batch_size == 2:
            return self.pipeline_global_b2
        raise ValueError("SAM3 recurrent tracker supports only batch sizes 1 and 2")


@dataclass(frozen=True)
class _Dependencies:
    torch: Any
    tracker_model_class: Any
    transformers_version: str


@dataclass(frozen=True)
class _StagedPackage:
    stage: Literal["encoder", "decoder"]
    batch_size: int
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
        raise RuntimeError(
            "SAM3 split tracker export requires Torch and Transformers 5.2"
        ) from error

    version = str(getattr(transformers, "__version__", "") or "")
    if _version_major_minor(version) != _TRANSFORMERS_MAJOR_MINOR:
        raise RuntimeError(
            f"SAM3 split tracker export requires Transformers 5.2.x; found {version or 'unknown'}"
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


def _producer_abi(dependencies: _Dependencies, device_index: int) -> TrackerAotiProducerAbi:
    torch = dependencies.torch
    if not bool(torch.cuda.is_available()):
        raise RuntimeError("SAM3 split tracker export requires a CUDA GPU")
    if device_index < 0 or device_index >= int(torch.cuda.device_count()):
        raise RuntimeError(f"CUDA device {device_index} is not available")
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(device_index))
    if capability not in _SUPPORTED_COMPUTE_CAPABILITIES:
        raise RuntimeError(
            "SAM3 split tracker AOTI export does not support compute capability "
            f"{capability[0]}.{capability[1]}"
        )
    cuda_version = str(getattr(torch.version, "cuda", "") or "")
    if not cuda_version:
        raise RuntimeError("Torch does not report a CUDA build version")
    cxx11_abi = getattr(getattr(torch, "_C", None), "_GLIBCXX_USE_CXX11_ABI", None)
    if cxx11_abi is None:
        raise RuntimeError("Torch does not report its C++11 ABI")
    return TrackerAotiProducerAbi(
        torch_version=str(getattr(torch, "__version__", "") or "unknown"),
        transformers_version=dependencies.transformers_version,
        cuda_version=cuda_version,
        compute_capability=(capability[0], capability[1]),
        host_architecture=platform.machine(),
        torch_cxx11_abi=bool(cxx11_abi),
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
        relative = path.relative_to(model_dir).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_hash_file(path)))
    return digest.hexdigest()


def _exporter_source_digest() -> str:
    return _hash_file(Path(__file__).resolve())


def _cache_key(model_digest: str, producer: TrackerAotiProducerAbi) -> str:
    contract = {
        "schema_version": 1,
        "model_sha256": model_digest,
        "exporter_sha256": _exporter_source_digest(),
        "producer": asdict(producer),
        "memory_bounds": _MEMORY_BOUNDS,
        "pointer_bounds": _POINTER_BOUNDS,
        "representative_shape": {
            "memory_count": _REPRESENTATIVE_MEMORY_COUNT,
            "pointer_count": _REPRESENTATIVE_POINTER_COUNT,
        },
    }
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@contextmanager
def _exclusive_cache_lock(cache_root: Path, key: str) -> Iterator[None]:
    cache_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not stat.S_ISDIR(cache_root.lstat().st_mode):
        raise RuntimeError("SAM3 AOTI cache root is not a directory")
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
        # Meta's tracker attention modules call
        # torch.nn.functional.scaled_dot_product_attention directly.  Select
        # the equivalent Transformers dispatch so AOTInductor sees the same
        # fused SDPA operator instead of the eager matmul/FP32-softmax
        # decomposition.
        attn_implementation="sdpa",
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
            "SAM3 tracker weights did not load exactly: "
            f"missing={missing}, unexpected={unexpected}, mismatched={mismatched}, errors={errors}"
        )
    required = (
        "memory_attention",
        "mask_decoder",
        "prompt_encoder",
        "object_pointer_proj",
        "temporal_positional_encoding_projection_layer",
        "memory_temporal_positional_encoding",
        "no_object_pointer",
        "get_image_wide_positional_embeddings",
    )
    absent = tuple(name for name in required if not hasattr(model, name))
    if absent:
        raise RuntimeError(f"SAM3 tracker model is missing required modules: {absent}")
    model = model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _make_encoder_module(torch: Any, model: Any, *, batch_size: int) -> Any:
    class _DynamicMemoryEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.batch_size = batch_size
            self.hidden_dim = int(model.hidden_dim)
            self.mem_dim = int(model.mem_dim)
            self.num_maskmem = int(model.num_maskmem)
            self.memory_attention = model.memory_attention
            self.temporal_projection = model.temporal_positional_encoding_projection_layer
            self.memory_temporal_position = model.memory_temporal_positional_encoding
            pointer_dimension = self.hidden_dim // 2
            dimensions = torch.arange(
                pointer_dimension,
                dtype=torch.float32,
                device=model.memory_temporal_positional_encoding.device,
            )
            denominator = 10000.0 ** (
                2.0 * torch.div(dimensions, 2, rounding_mode="floor") / pointer_dimension
            )
            self.register_buffer("pointer_denominator", denominator, persistent=True)

        def forward(
            self,
            tracker_feature_2,
            tracker_position_2,
            memory_features,
            memory_position,
            memory_temporal_offsets,
            object_pointers,
            object_pointer_temporal_offsets,
            max_object_pointers_to_use,
        ):
            batch = self.batch_size
            memory_count = memory_features.shape[1]
            pointer_count = object_pointers.shape[1]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Meta's tracker compile wrapper receives both current-frame
                # tensors after the BF16 vision boundary.  Keep the public TRT
                # carriers FP32, but round before flattening so attention
                # residuals and positional operations cannot promote them.
                feature = tracker_feature_2.to(torch.bfloat16).expand(batch, -1, -1, -1)
                feature_position = tracker_position_2.to(torch.bfloat16).expand(batch, -1, -1, -1)
                current = feature.flatten(2).permute(2, 0, 1).contiguous()
                current_position = feature_position.flatten(2).permute(2, 0, 1).contiguous()

                spatial_memory = (
                    memory_features.reshape(batch, memory_count, _SPATIAL_SIZE, 64)
                    .permute(1, 2, 0, 3)
                    .reshape(memory_count * _SPATIAL_SIZE, batch, 64)
                    .contiguous()
                )
                temporal_indices = torch.remainder(
                    memory_temporal_offsets.to(torch.int64) - 1,
                    self.num_maskmem,
                )
                temporal = self.memory_temporal_position.index_select(
                    0, temporal_indices.reshape(-1)
                ).reshape(batch, memory_count, 1, self.mem_dim)
                spatial_position = (
                    (memory_position.reshape(batch, memory_count, _SPATIAL_SIZE, 64) + temporal)
                    .permute(1, 2, 0, 3)
                    .reshape(memory_count * _SPATIAL_SIZE, batch, 64)
                    .contiguous()
                )

                pointer_memory = (
                    object_pointers.reshape(batch, pointer_count, 4, 64)
                    .permute(1, 2, 0, 3)
                    .reshape(pointer_count * 4, batch, 64)
                    .contiguous()
                )
                normalized_offsets = object_pointer_temporal_offsets.float() / (
                    max_object_pointers_to_use[0].float() - 1.0
                )
                angles = normalized_offsets.unsqueeze(-1) / self.pointer_denominator
                sine_position = torch.cat((angles.sin(), angles.cos()), dim=-1)
                projected_position = self.temporal_projection(sine_position)
                pointer_position = (
                    projected_position.permute(1, 0, 2).repeat_interleave(4, dim=0).contiguous()
                )

                prompt = torch.cat((spatial_memory, pointer_memory), dim=0).contiguous()
                prompt_position = torch.cat(
                    (spatial_position, pointer_position), dim=0
                ).contiguous()
                encoded = self.memory_attention(
                    current_vision_features=current,
                    memory=prompt,
                    current_vision_position_embeddings=current_position,
                    memory_posision_embeddings=prompt_position,
                    num_object_pointer_tokens=pointer_count * 4,
                )
                conditioned = encoded.squeeze(0).permute(0, 2, 1).reshape(batch, 256, 72, 72)
                return conditioned.float().contiguous().clone()

    return _DynamicMemoryEncoder().eval()


def _make_decoder_module(torch: Any, model: Any, *, batch_size: int, device: Any) -> Any:
    class _StaticRecurrentDecoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.batch_size = batch_size
            self.mask_decoder = model.mask_decoder
            self.object_pointer_proj = model.object_pointer_proj
            self.register_buffer(
                "no_object_pointer",
                model.no_object_pointer.detach().clone().contiguous(),
                persistent=True,
            )
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                points = torch.zeros(batch_size, 1, 1, 2, dtype=torch.float32, device=device)
                labels = -torch.ones(batch_size, 1, 1, dtype=torch.int32, device=device)
                sparse, dense = model.prompt_encoder(
                    input_points=points,
                    input_labels=labels,
                    input_boxes=None,
                    input_masks=None,
                )
                image_position = model.get_image_wide_positional_embeddings().repeat(
                    batch_size, 1, 1, 1
                )
            self.register_buffer(
                "sparse_prompt", sparse.detach().clone().contiguous(), persistent=True
            )
            self.register_buffer(
                "dense_prompt", dense.detach().clone().contiguous(), persistent=True
            )
            self.register_buffer(
                "image_position",
                image_position.detach().clone().contiguous(),
                persistent=True,
            )

        def forward(self, tracker_feature_0, tracker_feature_1, conditioned_features):
            batch = self.batch_size
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Meta precomputes conv_s0/conv_s1 before calling the tracker
                # decoder, then its compile wrapper materializes these maps as
                # contiguous BF16 tensors.  The external TRT carriers are
                # FP32, so make that precision boundary explicit; otherwise
                # the residual adds below are promoted back to FP32.
                high0 = tracker_feature_0.to(torch.bfloat16).expand(batch, -1, -1, -1).contiguous()
                high1 = tracker_feature_1.to(torch.bfloat16).expand(batch, -1, -1, -1).contiguous()
                conditioned = conditioned_features.float().contiguous()
                raw_masks, raw_ious, raw_tokens, raw_scores = self.mask_decoder(
                    image_embeddings=conditioned,
                    image_positional_embeddings=self.image_position.contiguous(),
                    sparse_prompt_embeddings=self.sparse_prompt.contiguous(),
                    dense_prompt_embeddings=self.dense_prompt.contiguous(),
                    multimask_output=True,
                    high_resolution_features=[high0, high1],
                )
                masks = raw_masks.squeeze(1)
                ious = raw_ious.squeeze(1)
                tokens = raw_tokens.squeeze(1)
                scores = raw_scores.reshape(batch, 1)
                appearing = scores > 0
                masks = torch.where(appearing[:, :, None, None], masks, _NO_OBJECT_SCORE)
                best = ious.argmax(dim=-1)
                rows = torch.arange(batch, device=masks.device)
                selected_masks = masks[rows, best].unsqueeze(1)
                selected_tokens = tokens[rows, best]
                pointer = self.object_pointer_proj(selected_tokens)
                visible = appearing.to(pointer.dtype)
                pointer = visible * pointer + (1.0 - visible) * self.no_object_pointer
                selected_iou = ious[rows, best]
                packed = torch.cat(
                    (
                        selected_masks.float().reshape(batch, _MASK_SIZE),
                        pointer.float().reshape(batch, _POINTER_WIDTH),
                        scores.float().reshape(batch, 1),
                        selected_iou.float().reshape(batch, 1),
                    ),
                    dim=1,
                )
                return packed.reshape(batch, _PACKED_WIDTH).contiguous().clone()

    return _StaticRecurrentDecoder().eval()


def _encoder_example(
    torch: Any,
    *,
    batch_size: int,
    device: Any,
    memory_count: int = _REPRESENTATIVE_MEMORY_COUNT,
    pointer_count: int = _REPRESENTATIVE_POINTER_COUNT,
) -> tuple[Any, ...]:
    return (
        torch.randn(1, 256, 72, 72, dtype=torch.float32, device=device),
        torch.randn(1, 256, 72, 72, dtype=torch.float32, device=device),
        torch.randn(
            batch_size,
            memory_count,
            _SPATIAL_SIZE,
            64,
            dtype=torch.float32,
            device=device,
        ),
        torch.randn(
            batch_size,
            memory_count,
            _SPATIAL_SIZE,
            64,
            dtype=torch.float32,
            device=device,
        ),
        torch.arange(1, memory_count + 1, dtype=torch.int32, device=device)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .contiguous(),
        torch.randn(
            batch_size,
            pointer_count,
            _POINTER_WIDTH,
            dtype=torch.float32,
            device=device,
        ),
        torch.arange(pointer_count, dtype=torch.int32, device=device)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .contiguous(),
        torch.tensor([16], dtype=torch.int32, device=device),
    )


def _decoder_example(torch: Any, *, batch_size: int, device: Any) -> tuple[Any, ...]:
    return (
        torch.randn(1, 32, 288, 288, dtype=torch.float32, device=device),
        torch.randn(1, 64, 144, 144, dtype=torch.float32, device=device),
        torch.randn(batch_size, 256, 72, 72, dtype=torch.float32, device=device),
    )


def _encoder_dynamic_shapes(torch: Any, *, batch_size: int) -> tuple[Any, ...]:
    memory_count = torch.export.Dim(
        f"memory_count_b{batch_size}", min=_MEMORY_BOUNDS[0], max=_MEMORY_BOUNDS[1]
    )
    pointer_count = torch.export.Dim(
        f"pointer_count_b{batch_size}", min=_POINTER_BOUNDS[0], max=_POINTER_BOUNDS[1]
    )
    return (
        None,
        None,
        {1: memory_count},
        {1: memory_count},
        {1: memory_count},
        {1: pointer_count},
        {1: pointer_count},
        None,
    )


def _compile_one_package(
    torch: Any,
    module: Any,
    example: tuple[Any, ...],
    output: Path,
    *,
    dynamic_shapes: tuple[Any, ...] | None,
) -> None:
    with torch.inference_mode():
        exported = torch.export.export(
            module,
            example,
            dynamic_shapes=dynamic_shapes,
            strict=False,
        )
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
        raise RuntimeError(f"AOTI did not produce the SAM3 package {output}")


def _unwrap_aoti_output(value: Any) -> Any:
    while isinstance(value, (tuple, list)) and len(value) == 1:
        value = value[0]
    return value


def _numerical_metrics(torch: Any, actual: Any, expected: Any) -> dict[str, float]:
    if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
        raise RuntimeError("SAM3 AOTI smoke output is not a tensor")
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise RuntimeError(
            "SAM3 AOTI smoke output contract mismatch: "
            f"actual={tuple(actual.shape)}/{actual.dtype}, "
            f"expected={tuple(expected.shape)}/{expected.dtype}"
        )
    if not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(expected).all()):
        raise RuntimeError("SAM3 AOTI smoke output contains non-finite values")
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


def _validate_package_case(
    torch: Any,
    *,
    stage: Literal["encoder", "decoder"],
    batch_size: int,
    module: Any,
    loaded: Any,
    inputs: tuple[Any, ...],
    memory_count: int | None,
    pointer_count: int | None,
) -> dict[str, Any]:
    with torch.inference_mode():
        expected = _unwrap_aoti_output(module(*inputs))
        actual = _unwrap_aoti_output(loaded(*inputs))
        torch.cuda.synchronize(inputs[0].device)
    metrics = _numerical_metrics(torch, actual, expected)
    if (
        metrics["cosine"] < _SMOKE_MINIMUM_COSINE
        or metrics["relative_l2"] > _SMOKE_MAXIMUM_RELATIVE_L2
    ):
        raise RuntimeError(f"SAM3 {stage} B{batch_size} AOTI/eager smoke failed: {metrics}")
    record: dict[str, Any] = {
        "stage": stage,
        "batch_size": batch_size,
        "memory_count": memory_count,
        "pointer_count": pointer_count,
        **metrics,
        "passed": True,
    }
    if stage == "decoder":
        expected_masks = expected[:, :_MASK_SIZE] > 0
        actual_masks = actual[:, :_MASK_SIZE] > 0
        agreement = float((expected_masks == actual_masks).float().mean().item())
        if agreement < _SMOKE_MINIMUM_BINARY_MASK_AGREEMENT:
            raise RuntimeError(f"SAM3 decoder B{batch_size} binary-mask smoke failed: {agreement}")
        record["binary_mask_agreement"] = agreement
    return record


def _validate_compiled_package(
    torch: Any,
    *,
    stage: Literal["encoder", "decoder"],
    batch_size: int,
    module: Any,
    package: Path,
    device: Any,
) -> tuple[dict[str, Any], ...]:
    loaded = torch._inductor.aoti_load_package(
        os.fspath(package),
        device_index=int(device.index or 0),
    )
    records: list[dict[str, Any]] = []
    try:
        if stage == "encoder":
            for memory_count, pointer_count in _SMOKE_ENCODER_SHAPES:
                inputs = _encoder_example(
                    torch,
                    batch_size=batch_size,
                    device=device,
                    memory_count=memory_count,
                    pointer_count=pointer_count,
                )
                records.append(
                    _validate_package_case(
                        torch,
                        stage=stage,
                        batch_size=batch_size,
                        module=module,
                        loaded=loaded,
                        inputs=inputs,
                        memory_count=memory_count,
                        pointer_count=pointer_count,
                    )
                )
        else:
            records.append(
                _validate_package_case(
                    torch,
                    stage=stage,
                    batch_size=batch_size,
                    module=module,
                    loaded=loaded,
                    inputs=_decoder_example(torch, batch_size=batch_size, device=device),
                    memory_count=None,
                    pointer_count=None,
                )
            )
    finally:
        del loaded
    return tuple(records)


def _compile_split_packages(
    dependencies: _Dependencies,
    model: Any,
    staging: Path,
    device: Any,
) -> tuple[tuple[_StagedPackage, ...], tuple[dict[str, Any], ...]]:
    torch = dependencies.torch
    staged: list[_StagedPackage] = []
    validation_records: list[dict[str, Any]] = []
    for batch_size in (1, 2):
        encoder = _make_encoder_module(torch, model, batch_size=batch_size)
        decoder = _make_decoder_module(torch, model, batch_size=batch_size, device=device)
        encoder_path = staging / f"encoder_b{batch_size}.pt2"
        decoder_path = staging / f"decoder_b{batch_size}.pt2"
        _compile_one_package(
            torch,
            encoder,
            _encoder_example(torch, batch_size=batch_size, device=device),
            encoder_path,
            dynamic_shapes=_encoder_dynamic_shapes(torch, batch_size=batch_size),
        )
        _compile_one_package(
            torch,
            decoder,
            _decoder_example(torch, batch_size=batch_size, device=device),
            decoder_path,
            dynamic_shapes=None,
        )
        validation_records.extend(
            _validate_compiled_package(
                torch,
                stage="encoder",
                batch_size=batch_size,
                module=encoder,
                package=encoder_path,
                device=device,
            )
        )
        validation_records.extend(
            _validate_compiled_package(
                torch,
                stage="decoder",
                batch_size=batch_size,
                module=decoder,
                package=decoder_path,
                device=device,
            )
        )
        staged.extend(
            (
                _StagedPackage("encoder", batch_size, encoder_path),
                _StagedPackage("decoder", batch_size, decoder_path),
            )
        )
        del encoder, decoder
        torch.cuda.empty_cache()
    return tuple(staged), tuple(validation_records)


def _package_global(stage: str, batch_size: int, digest: str) -> str:
    suffix = digest[:_GLOBAL_DIGEST_CHARACTERS]
    if stage == "encoder":
        return f"trtmc.sam3.tracker_encoder.b{batch_size}.m1_10.p1_19.{suffix}"
    if stage == "decoder":
        return f"trtmc.sam3.tracker_decoder.b{batch_size}.static.{suffix}"
    raise ValueError(f"Unsupported SAM3 AOTI stage {stage!r}")


def _pipeline_digest(encoder_sha256: str, decoder_sha256: str) -> str:
    digest = hashlib.sha256()
    digest.update(_PIPELINE_DIGEST_DOMAIN)
    digest.update(bytes.fromhex(encoder_sha256))
    digest.update(bytes.fromhex(decoder_sha256))
    return digest.hexdigest()


def _pipeline_global(batch_size: int, encoder_sha256: str, decoder_sha256: str) -> str:
    digest = _pipeline_digest(encoder_sha256, decoder_sha256)
    return f"trtmc.sam3.tracker_step.b{batch_size}.split_aoti.{digest[:_GLOBAL_DIGEST_CHARACTERS]}"


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("utf-8")


def _json_value(value: Any) -> Any:
    """Normalize tuples and dataclasses to their canonical JSON representation."""

    return json.loads(_canonical_json(value))


def _publish_staged_packages(
    staged: Sequence[_StagedPackage],
    validation_records: Sequence[dict[str, Any]],
    staging: Path,
    producer: TrackerAotiProducerAbi,
    model_digest: str,
) -> bytes:
    if {(item.stage, item.batch_size) for item in staged} != set(_PACKAGE_ORDER):
        raise RuntimeError("SAM3 split AOTI export did not produce all four packages")
    records = []
    by_key = {(item.stage, item.batch_size): item for item in staged}
    for stage, batch_size in _PACKAGE_ORDER:
        source = by_key[(stage, batch_size)].path
        digest = _hash_file(source)
        destination = staging / f"sam3_tracker_{stage}_b{batch_size}_{digest}.pt2"
        if source != destination:
            os.replace(source, destination)
        records.append(
            {
                "stage": stage,
                "batch_size": batch_size,
                "filename": destination.name,
                "section": _PACKAGE_SECTIONS[(stage, batch_size)],
                "sha256": digest,
                "package_global": _package_global(stage, batch_size, digest),
            }
        )
    pipeline_globals = {}
    for batch_size in (1, 2):
        encoder = next(
            record
            for record in records
            if record["stage"] == "encoder" and record["batch_size"] == batch_size
        )
        decoder = next(
            record
            for record in records
            if record["stage"] == "decoder" and record["batch_size"] == batch_size
        )
        pipeline_globals[f"b{batch_size}"] = _pipeline_global(
            batch_size, encoder["sha256"], decoder["sha256"]
        )
    manifest = {
        "schema_version": 1,
        "scope": "split_dynamic_encoder_static_decoder",
        "model_sha256": model_digest,
        "producer": asdict(producer),
        "carrier_abi": [asdict(value) for value in TRACKER_TEN_CARRIER_ABI],
        "dynamic_contract": {
            "memory_count": list(_MEMORY_BOUNDS),
            "pointer_count": list(_POINTER_BOUNDS),
            "representative_memory_count": _REPRESENTATIVE_MEMORY_COUNT,
            "representative_pointer_count": _REPRESENTATIVE_POINTER_COUNT,
        },
        "packages": records,
        "pipeline_globals": pipeline_globals,
        "pipeline_digest": "sha256(domain || encoder_sha256_bytes || decoder_sha256_bytes)",
        "package_validation": {
            "reference": "same-module eager execution before cache publication",
            "minimum_cosine": _SMOKE_MINIMUM_COSINE,
            "maximum_relative_l2": _SMOKE_MAXIMUM_RELATIVE_L2,
            "minimum_binary_mask_agreement": _SMOKE_MINIMUM_BINARY_MASK_AGREEMENT,
            "cases": list(validation_records),
        },
    }
    manifest_bytes = _canonical_json(manifest)
    (staging / TRACKER_SPLIT_AOTI_MANIFEST_SECTION).write_bytes(manifest_bytes)
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
    expected_producer: TrackerAotiProducerAbi | None = None,
) -> dict[str, Any] | None:
    manifest_path = cache_directory / TRACKER_SPLIT_AOTI_MANIFEST_SECTION
    if not _regular_file(manifest_path):
        return None
    try:
        manifest = json.loads(manifest_path.read_bytes())
        if (
            manifest["schema_version"] != 1
            or manifest["scope"] != "split_dynamic_encoder_static_decoder"
            or (
                expected_model_digest is not None
                and manifest["model_sha256"] != expected_model_digest
            )
            or (
                expected_producer is not None
                and manifest["producer"] != _json_value(asdict(expected_producer))
            )
            or manifest["carrier_abi"]
            != _json_value([asdict(value) for value in TRACKER_TEN_CARRIER_ABI])
            or manifest["dynamic_contract"]
            != {
                "memory_count": list(_MEMORY_BOUNDS),
                "pointer_count": list(_POINTER_BOUNDS),
                "representative_memory_count": _REPRESENTATIVE_MEMORY_COUNT,
                "representative_pointer_count": _REPRESENTATIVE_POINTER_COUNT,
            }
        ):
            return None
        packages = manifest["packages"]
        if len(packages) != len(_PACKAGE_ORDER):
            return None
        if {(str(item["stage"]), int(item["batch_size"])) for item in packages} != set(
            _PACKAGE_ORDER
        ):
            return None
        for item in packages:
            stage = str(item["stage"])
            batch_size = int(item["batch_size"])
            digest = str(item["sha256"])
            filename = str(item["filename"])
            if (
                not re.fullmatch(r"[0-9a-f]{64}", digest)
                or filename != f"sam3_tracker_{stage}_b{batch_size}_{digest}.pt2"
                or item["section"] != _PACKAGE_SECTIONS[(stage, batch_size)]
                or item["package_global"] != _package_global(stage, batch_size, digest)
            ):
                return None
            path = cache_directory / filename
            if not _regular_file(path) or _hash_file(path) != digest:
                return None
        expected = {}
        for batch_size in (1, 2):
            encoder = next(
                item
                for item in packages
                if item["stage"] == "encoder" and item["batch_size"] == batch_size
            )
            decoder = next(
                item
                for item in packages
                if item["stage"] == "decoder" and item["batch_size"] == batch_size
            )
            expected[f"b{batch_size}"] = _pipeline_global(
                batch_size, encoder["sha256"], decoder["sha256"]
            )
        if manifest["pipeline_globals"] != expected:
            return None
        validation = manifest["package_validation"]
        expected_cases = {
            ("encoder", batch_size, memory_count, pointer_count)
            for batch_size in (1, 2)
            for memory_count, pointer_count in _SMOKE_ENCODER_SHAPES
        } | {("decoder", batch_size, None, None) for batch_size in (1, 2)}
        actual_cases = {
            (
                str(case["stage"]),
                int(case["batch_size"]),
                case["memory_count"],
                case["pointer_count"],
            )
            for case in validation["cases"]
            if case["passed"] is True
        }
        if (
            validation["reference"] != "same-module eager execution before cache publication"
            or float(validation["minimum_cosine"]) != _SMOKE_MINIMUM_COSINE
            or float(validation["maximum_relative_l2"]) != _SMOKE_MAXIMUM_RELATIVE_L2
            or float(validation["minimum_binary_mask_agreement"])
            != _SMOKE_MINIMUM_BINARY_MASK_AGREEMENT
            or actual_cases != expected_cases
        ):
            return None
        return manifest
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _artifacts_from_cache(cache_directory: Path) -> Sam3TrackerSplitAotiArtifacts:
    manifest = _validate_cached_directory(cache_directory)
    if manifest is None:
        raise RuntimeError(f"Invalid SAM3 split AOTI cache {cache_directory}")
    packages_by_key = {
        (str(item["stage"]), int(item["batch_size"])): item for item in manifest["packages"]
    }
    packages = tuple(
        TrackerAotiPackage(
            stage=stage,
            batch_size=batch_size,
            path=cache_directory / packages_by_key[(stage, batch_size)]["filename"],
            section=packages_by_key[(stage, batch_size)]["section"],
            sha256=packages_by_key[(stage, batch_size)]["sha256"],
            package_global=packages_by_key[(stage, batch_size)]["package_global"],
        )
        for stage, batch_size in _PACKAGE_ORDER
    )
    producer = manifest["producer"]
    producer_abi = TrackerAotiProducerAbi(
        torch_version=producer["torch_version"],
        transformers_version=producer["transformers_version"],
        cuda_version=producer["cuda_version"],
        compute_capability=tuple(producer["compute_capability"]),
        host_architecture=producer["host_architecture"],
        torch_cxx11_abi=producer["torch_cxx11_abi"],
    )
    manifest_bytes = (cache_directory / TRACKER_SPLIT_AOTI_MANIFEST_SECTION).read_bytes()
    bundle_sections = (
        (TRACKER_SPLIT_AOTI_MANIFEST_SECTION, manifest_bytes),
        *((package.section, package.path.read_bytes()) for package in packages),
    )
    return Sam3TrackerSplitAotiArtifacts(
        cache_directory=cache_directory,
        packages=packages,
        pipeline_global_b1=manifest["pipeline_globals"]["b1"],
        pipeline_global_b2=manifest["pipeline_globals"]["b2"],
        producer_abi=producer_abi,
        carrier_abi=TRACKER_TEN_CARRIER_ABI,
        manifest_bytes=manifest_bytes,
        bundle_sections=bundle_sections,
    )


def export_sam3_tracker_split_aoti(
    model_dir: str | Path,
    *,
    device_index: int = 0,
    cache_dir: str | Path | None = None,
) -> Sam3TrackerSplitAotiArtifacts:
    """Export or reuse the fastest supported recurrent tracker packages.

    ``model_dir`` is the normal local Transformers SAM3 directory used by the
    default builder.  The API deliberately has no tuning flags: B1/B2, the
    dynamic history bounds, BF16 autocast policy, and split compilation policy
    are the supported production configuration.
    """

    resolved_model_dir = Path(model_dir).resolve()
    if not resolved_model_dir.is_dir():
        raise FileNotFoundError(resolved_model_dir)
    dependencies = _load_dependencies()
    producer = _producer_abi(dependencies, device_index)
    model_digest = _model_source_digest(resolved_model_dir)
    key = _cache_key(model_digest, producer)
    cache_root = Path(cache_dir).resolve() if cache_dir is not None else _CACHE_ROOT
    cache_directory = cache_root / key
    validation = {
        "expected_model_digest": model_digest,
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
                # Export must not perturb the caller's CPU/CUDA random state.
                # This is especially important when a build process compiles
                # more than one family in the same interpreter.
                with torch.random.fork_rng(devices=[device_index]):
                    torch.manual_seed(20260717)
                    model = _load_tracker_model(dependencies, resolved_model_dir, device)
                    staged, package_validation = _compile_split_packages(
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
    "Sam3TrackerSplitAotiArtifacts",
    "TRACKER_SPLIT_AOTI_MANIFEST_SECTION",
    "TRACKER_TEN_CARRIER_ABI",
    "TrackerAotiPackage",
    "TrackerAotiProducerAbi",
    "TrackerTensorAbi",
    "export_sam3_tracker_split_aoti",
]
