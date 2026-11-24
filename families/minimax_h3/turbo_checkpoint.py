# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Selective Comfy BF16/INT8 + Turbo-v4 loaders with unmerged LoRA weights.

The author applies ``base(x) + B(A(x))`` with BF16 rounding after each linear
and the addition. Merging ``B @ A`` into BF16 base weights is NOT equivalent.
``TurboLoraWeight`` retains those three matrices for native graph construction.
Comfy QKV rows are already grouped, not head-interleaved. Only the SwiGLU
``[gate; value]`` -> Diffusers ``[value; gate]`` row permutation is required;
apply it to both the base matrix and LoRA B, never to LoRA A.

An explicitly selected INT8 base retains the full released ConvRot graph and
its FP32 row scales. Only the base branch rotates/quantizes the input; LoRA
still consumes the original BF16 input. This is not BF16-reference parity.

T2VA/FL2VA and experimental REF2VA use separately authenticated full bases.
Matching tensor shapes permit the adapter to load; they do not certify the
author's quality on REF2VA or permit substituting FL2VA weights. Validation
reads small headers, not multi-gigabyte checksums.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import ml_dtypes
import numpy as np

from .checkpoint import numpy_state
from . import quantized_checkpoint as quantized
from .quantized_checkpoint import (
    ConvRotInt8Weight,
    _EXPECTED_CONFIG,
    _LOGICAL_MAP,
    _PHYSICAL_SPECS as _QUANTIZED_SPECS,
)

BASE_MODEL_ID = "Comfy-Org/MiniMax-H3"
BASE_REVISION = "7e75982b97cd5a41d2dcfa1904ee88d0686d6fd1"
BASE_FILENAME = "diffusion_models/minimax_h3_fl2va_bf16.safetensors"
BASE_BYTES = 66_280_487_368
REF2VA_BASE_FILENAME = "diffusion_models/minimax_h3_ref2va_bf16.safetensors"
REF2VA_BASE_BYTES = 66_280_487_368
REF2VA_BASE_ETAG = "e32c54c1a7b4f5f397f195cea267ccb18806303bb665678c4bee60953bdf3026"
LORA_MODEL_ID = "larryvrh/MiniMax-H3-Turbo-Lora"
LORA_REVISION = "43a74557ac3f6539db8e0f2a959d03feb7a81480"
LORA_FILENAME = "minimax_h3_turbo_v4_step600_ema.safetensors"
LORA_BYTES = 779_849_816
_BF16 = np.dtype(ml_dtypes.bfloat16)
_MAX_HEADER_BYTES = 8 << 20
_LORA_METADATA = {"application": "W_eff = W + lora_B @ lora_A", "base_model": "MiniMax-H3"}

# The family already owns this physical layout for its INT8 export. BF16 has
# the same tensors, without the quantization scales/markers or INT8 storage.
_BASE_SPECS = {
    name: ("BF16" if spec.dtype == "I8" else spec.dtype, spec.shape)
    for name, spec in _QUANTIZED_SPECS.items()
    if not name.endswith((".weight_scale", ".comfy_quant"))
}


def _lora_contract() -> dict[str, tuple[str, tuple[int, ...]]]:
    specs = {}
    suffixes = (
        ".attn.qkv_proj.weight",
        ".attn.out_proj.weight",
        ".mlp.fc1.weight",
        ".mlp.fc2.weight",
        ".adaln_proj.linear.weight",
    )
    for name, (_dtype, shape) in _BASE_SPECS.items():
        if not name.endswith(suffixes):
            continue
        rank = 16 if name.endswith(".adaln_proj.linear.weight") else 64
        module = name.removesuffix(".weight")
        specs[f"{module}.lora_A.weight"] = ("BF16", (rank, shape[1]))
        specs[f"{module}.lora_B.weight"] = ("BF16", (shape[0], rank))
    return specs


_LORA_SPECS = _lora_contract()


@dataclass(frozen=True, eq=False)
class TurboLoraWeight:
    """A base linear and a separate strength-one BF16 low-rank branch.

    Graph integration must round base(x) + bias, A(x), and B(A(x)) to BF16
    before the final BF16 add, matching the author's runtime LoRA. Bias is
    supplied separately by the existing builder, and belongs to the base
    branch only. An INT8 base retains its ConvRot weight, while A/B consume
    the original input, not the base branch's rotated/quantized activation.
    Parent/slice fields allow one fused QKV GEMM without copies.
    """

    base: np.ndarray | ConvRotInt8Weight
    lora_a: np.ndarray
    lora_b: np.ndarray
    packed_parent: TurboLoraWeight | None = None
    row_slice: tuple[int, int] | None = None
    is_full_fused_qkv: bool = False

    def __post_init__(self) -> None:
        values = [("lora_a", self.lora_a), ("lora_b", self.lora_b)]
        if not isinstance(self.base, ConvRotInt8Weight):
            values.append(("base", self.base))
        for name, value in values:
            if not isinstance(value, np.ndarray) or value.ndim != 2 or not value.flags.c_contiguous:
                raise ValueError(f"Turbo {name} must be a contiguous [out,in] NumPy array")
        if not isinstance(self.base, ConvRotInt8Weight) and self.base.dtype not in (
            _BF16,
            np.dtype(np.float32),
        ):
            raise TypeError("Turbo base must retain BF16 or FP32 checkpoint storage")
        if self.lora_a.dtype != _BF16 or self.lora_b.dtype != _BF16:
            raise TypeError("Turbo LoRA factors must retain BF16 storage")
        rank, in_features = self.lora_a.shape
        if rank < 1 or self.shape[1] != in_features or self.lora_b.shape != (self.shape[0], rank):
            raise ValueError("Turbo LoRA factor dimensions do not match the base linear")
        if self.is_full_fused_qkv:
            if self.packed_parent is not None or self.row_slice is not None or self.shape[0] % 3:
                raise ValueError("A full Turbo QKV parent must contain three complete row groups")
            if isinstance(self.base, ConvRotInt8Weight) and not self.base.is_full_fused_qkv:
                raise ValueError("A full Turbo INT8 QKV parent requires a full ConvRot parent")
        elif self.packed_parent is not None:
            parent = self.packed_parent
            if not parent.is_full_fused_qkv or self.row_slice is None:
                raise ValueError("Turbo QKV child requires a full parent and row slice")
            start, end = self.row_slice
            if not 0 <= start < end <= parent.shape[0] or end - start != self.shape[0]:
                raise ValueError("Turbo QKV child row slice is invalid")
            if self.lora_a is not parent.lora_a:
                raise ValueError("Turbo QKV children must share their parent's exact A factor")
            views = [(self.lora_b, parent.lora_b[start:end])]
            if isinstance(self.base, ConvRotInt8Weight):
                if (
                    not isinstance(parent.base, ConvRotInt8Weight)
                    or self.base.packed_parent is not parent.base
                    or self.base.row_slice != self.row_slice
                    or self.base.group_size != parent.base.group_size
                ):
                    raise ValueError("Turbo INT8 QKV children must retain the same ConvRot parent")
                views.extend(
                    (
                        (self.base.qweight, parent.base.qweight[start:end]),
                        (self.base.scale, parent.base.scale[start:end]),
                    )
                )
            elif isinstance(parent.base, ConvRotInt8Weight):
                raise ValueError("Turbo QKV cannot mix dense and ConvRot base storage")
            else:
                views.append((self.base, parent.base[start:end]))
            for value, expected in views:
                if (
                    value.shape != expected.shape
                    or value.dtype != expected.dtype
                    or value.strides != expected.strides
                    or value.ctypes.data != expected.ctypes.data
                ):
                    raise ValueError("Turbo QKV child must retain the exact base/B row views")
        elif self.row_slice is not None:
            raise ValueError("Turbo row slice requires a packed parent")

    @property
    def shape(self) -> tuple[int, int]:
        return (
            self.base.qweight.shape if isinstance(self.base, ConvRotInt8Weight) else self.base.shape
        )

    @property
    def dtype(self) -> np.dtype:
        return (
            self.base.qweight.dtype if isinstance(self.base, ConvRotInt8Weight) else self.base.dtype
        )


def pack_turbo_qkv(weights: Sequence[TurboLoraWeight]) -> TurboLoraWeight:
    """Return one grouped QKV base/B and shared A, without silently merging.

    Loader-provided children reuse the exact parent. Independently constructed
    Q/K/V matrices may be packed only when they share the same A object and
    have identical shapes/dtypes. Partial or reordered parent slices fail.
    """
    if len(weights) != 3 or any(not isinstance(value, TurboLoraWeight) for value in weights):
        raise ValueError("Turbo fused QKV requires three TurboLoraWeight values")
    q, k, v = weights
    if any(value.shape != q.shape or value.dtype != q.dtype for value in (k, v)):
        raise ValueError("Turbo Q/K/V base shapes and dtypes must agree")
    if any(value.lora_a is not q.lora_a for value in (k, v)):
        raise ValueError("Turbo fused QKV requires the exact shared A factor")
    if any(value.packed_parent is not None for value in weights):
        parent = q.packed_parent
        rows = q.shape[0]
        if (
            parent is None
            or any(value.packed_parent is not parent for value in weights)
            or tuple(value.row_slice for value in weights)
            != ((0, rows), (rows, rows * 2), (rows * 2, rows * 3))
            or parent.shape[0] != rows * 3
        ):
            raise ValueError("Turbo QKV children must cover the same parent in Q,K,V order")
        return parent
    if isinstance(q.base, ConvRotInt8Weight):
        bases = [value.base for value in weights]
        if any(
            not isinstance(base, ConvRotInt8Weight)
            or base.group_size != q.base.group_size
            or base.packed_parent is not None
            for base in bases
        ):
            raise ValueError("Independent Turbo INT8 QKV bases must share one ConvRot group size")
        base = ConvRotInt8Weight(
            np.concatenate([value.qweight for value in bases], axis=0),
            np.concatenate([value.scale.reshape(-1, 1) for value in bases], axis=0),
            q.base.group_size,
            is_full_fused_qkv=True,
        )
    else:
        base = np.concatenate([value.base for value in weights], axis=0)
    return TurboLoraWeight(
        base,
        q.lora_a,
        np.concatenate([value.lora_b for value in weights], axis=0),
        is_full_fused_qkv=True,
    )


def _validate_file(path: Path, specs, expected_bytes: int) -> dict[str, Any]:
    if path.stat().st_size != expected_bytes:
        raise ValueError("Turbo checkpoint size does not match the pinned public file")
    with path.open("rb") as stream:
        prefix = stream.read(8)
        length = int.from_bytes(prefix, "little")
        if len(prefix) != 8 or not 0 < length <= _MAX_HEADER_BYTES:
            raise ValueError("Turbo checkpoint header size is invalid")
        encoded = stream.read(length)
    if len(encoded) != length:
        raise ValueError("Turbo checkpoint header is incomplete")
    try:
        header = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Turbo checkpoint header is invalid JSON") from error
    if not isinstance(header, dict) or set(header) - {"__metadata__"} != set(specs):
        raise ValueError("Turbo checkpoint tensor inventory does not match the full pinned model")
    regions = []
    for name, (dtype, shape) in specs.items():
        entry = header[name]
        if not isinstance(entry, dict) or set(entry) != {"dtype", "shape", "data_offsets"}:
            raise ValueError(f"Invalid Turbo tensor header: {name}")
        if entry["dtype"] != dtype or entry["shape"] != list(shape):
            raise ValueError(f"Turbo tensor shape/dtype mismatch: {name}")
        offsets = entry["data_offsets"]
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(value) is not int for value in offsets)
        ):
            raise ValueError(f"Invalid Turbo tensor offsets: {name}")
        start, end = offsets
        if start < 0 or end - start != math.prod(shape) * {"BF16": 2, "F32": 4}[dtype]:
            raise ValueError(f"Invalid Turbo tensor byte range: {name}")
        regions.append((start, end))
    cursor = 0
    for start, end in sorted(regions):
        if start != cursor:
            raise ValueError("Turbo checkpoint tensor storage has a gap or overlap")
        cursor = end
    if 8 + length + cursor != expected_bytes:
        raise ValueError("Turbo checkpoint tensor storage does not cover its payload")
    return header.get("__metadata__", {})


def _authenticate_bf16_ref_source(path: Path) -> None:
    """Distinguish equal-shaped BF16 bases using pinned HF download provenance."""
    relative = Path(REF2VA_BASE_FILENAME)
    try:
        root = path.parents[len(relative.parts) - 1]
    except IndexError as error:
        raise ValueError("Turbo REF2VA checkpoint path is invalid") from error
    if os.path.normcase(os.path.abspath(path)) != os.path.normcase(
        os.path.abspath(root / relative)
    ):
        raise ValueError("Turbo REF2VA requires its released diffusion_models filename")
    if root.parent.name == "snapshots":
        if root.name != BASE_REVISION or root.parent.parent.name != "models--Comfy-Org--MiniMax-H3":
            raise ValueError("Turbo REF2VA checkpoint is not from the pinned HF snapshot")
        if path.is_symlink() and path.resolve().name != REF2VA_BASE_ETAG:
            raise ValueError("Turbo REF2VA snapshot points to a different checkpoint blob")
        return
    metadata_path = root / ".cache/huggingface/download" / f"{relative.as_posix()}.metadata"
    try:
        metadata = metadata_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(
            "Turbo REF2VA requires pinned HF download metadata; use hf download "
            f"{BASE_MODEL_ID} --revision {BASE_REVISION} --include "
            f'"{REF2VA_BASE_FILENAME}" --local-dir <directory>'
        ) from error
    if len(metadata) < 2 or metadata[0] != BASE_REVISION or metadata[1] != REF2VA_BASE_ETAG:
        raise ValueError("Turbo REF2VA HF metadata does not match the pinned revision and file")


def validate_turbo_transformer_checkpoint(
    checkpoint_file: str | Path,
    lora_file: str | Path,
    *,
    workflow: str = "fl2va",
    base_precision: str = "bf16",
) -> dict[str, object]:
    """Validate both small headers and report public provenance, without hashes."""
    if workflow not in ("t2va", "fl2va", "ref2va"):
        raise ValueError("Turbo workflow must be t2va, fl2va, or ref2va")
    if base_precision not in ("bf16", "int8"):
        raise ValueError("Turbo base_precision must be 'bf16' or 'int8'")
    if base_precision == "int8":
        # Preserve distinct source/header/marker checks; never accept pruning
        # or use the FL2VA model as a shape-compatible REF2VA fallback.
        identity = quantized.validate_quantized_transformer_checkpoint(
            checkpoint_file, workflow="ref2va" if workflow == "ref2va" else "fl2va"
        )
        base_metadata = {
            "base_model_id": identity.model_id,
            "base_revision": identity.revision,
            "base_filename": identity.filename,
            "base_size_bytes": identity.size_bytes,
            "base_tensor_count": identity.tensor_count,
            "base_precision": "int8",
            "base_quantization": "int8_tensorwise_convrot",
            "base_quantized_weight_count": identity.quantized_weight_count,
        }
    else:
        if workflow == "ref2va":
            _authenticate_bf16_ref_source(Path(checkpoint_file))
        base_bytes = REF2VA_BASE_BYTES if workflow == "ref2va" else BASE_BYTES
        metadata = _validate_file(Path(checkpoint_file), _BASE_SPECS, base_bytes)
        try:
            config = json.loads(metadata["config"])
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("Turbo base checkpoint config metadata is invalid") from error
        if config != _EXPECTED_CONFIG:
            raise ValueError("Turbo base must use the full released H3 architecture")
        # Keep the existing BF16 receipt contract unchanged.
        base_metadata = {
            "base_model_id": BASE_MODEL_ID,
            "base_revision": BASE_REVISION,
            "base_filename": REF2VA_BASE_FILENAME if workflow == "ref2va" else BASE_FILENAME,
            "base_size_bytes": base_bytes,
            "base_tensor_count": len(_BASE_SPECS),
        }
    if _validate_file(Path(lora_file), _LORA_SPECS, LORA_BYTES) != _LORA_METADATA:
        raise ValueError("Turbo LoRA metadata does not match the v4 strength-one adapter")
    return {
        **base_metadata,
        **(
            {
                "base_workflow": "ref2va",
                "adapter_compatibility": "experimental_not_author_certified",
            }
            if workflow == "ref2va"
            else {}
        ),
        "lora_model_id": LORA_MODEL_ID,
        "lora_revision": LORA_REVISION,
        "lora_filename": LORA_FILENAME,
        "lora_size_bytes": LORA_BYTES,
        "lora_tensor_count": len(_LORA_SPECS),
        "lora_strength": 1.0,
        "lora_merged": False,
        "runtime_framework": None,
        "qkv_layout": "comfy_grouped_q_k_v",
    }


def _swap_swiglu_rows(value: np.ndarray) -> np.ndarray:
    if value.shape[0] % 2:
        raise ValueError("Turbo SwiGLU matrix must have equal gate/value row groups")
    half = value.shape[0] // 2
    return np.concatenate((value[half:], value[:half]), axis=0)


def load_selected_turbo_transformer_weights(
    checkpoint_file: str | Path,
    lora_file: str | Path,
    logical_names: Iterable[str],
    *,
    workflow: str = "fl2va",
    base_precision: str = "bf16",
) -> dict[str, Any]:
    """Select only requested Diffusers-key matrices, retaining runtime LoRA.

    Small header validation is exhaustive. Tensor payload loading is selective
    and CPU-only; large weights are neither hashed, expanded to FP32, nor
    merged with LoRA. FP32 checkpoint-owned tensors remain FP32.
    """
    validate_turbo_transformer_checkpoint(
        checkpoint_file, lora_file, workflow=workflow, base_precision=base_precision
    )
    requested = tuple(logical_names)
    if len(requested) != len(set(requested)):
        raise ValueError("Turbo logical weight request contains duplicate names")
    if unknown := sorted(set(requested) - set(_LOGICAL_MAP)):
        raise ValueError(f"Unsupported Turbo logical tensor names: {unknown}")
    if not requested:
        return {}
    physical_names = {_LOGICAL_MAP[name][0] for name in requested}
    lora_names = {
        f"{name.removesuffix('.weight')}.{suffix}.weight"
        for name in physical_names
        for suffix in ("lora_A", "lora_B")
        if f"{name.removesuffix('.weight')}.lora_A.weight" in _LORA_SPECS
    }
    from safetensors import safe_open

    payload_names = set(physical_names)
    if base_precision == "int8":
        payload_names.update(
            f"{name.removesuffix('.weight')}.weight_scale"
            for name in physical_names
            if name.removesuffix(".weight") in quantized._QUANT_GROUPS
        )
    with safe_open(checkpoint_file, framework="pt", device="cpu") as reader:
        state = {name: reader.get_tensor(name) for name in sorted(payload_names)}
    arrays = numpy_state(state)
    del state
    with safe_open(lora_file, framework="pt", device="cpu") as reader:
        state = {name: reader.get_tensor(name) for name in sorted(lora_names)}
    adapters = numpy_state(state)
    del state
    mapped: dict[str, tuple[Any, ...]] = {}
    result: dict[str, Any] = {}
    for logical_name in requested:
        physical, transform = _LOGICAL_MAP[logical_name]
        if physical not in mapped:
            base = arrays.pop(physical)
            module = physical.removesuffix(".weight")
            a = adapters.pop(f"{module}.lora_A.weight", None)
            b = adapters.pop(f"{module}.lora_B.weight", None)
            quantized_qkv = None
            if base_precision == "int8" and module in quantized._QUANT_GROUPS:
                scale = arrays.pop(f"{module}.weight_scale")
                group_size = quantized._QUANT_GROUPS[module]
                if transform.startswith("qkv:"):
                    quantized_qkv = quantized._make_qkv_values(base, scale, group_size)
                    base = quantized_qkv[0].packed_parent
                else:
                    base = quantized._make_quantized_weight(
                        base, scale, group_size, transform=transform
                    )
            elif transform == "swap_swiglu":
                base = _swap_swiglu_rows(base)
            if transform == "swap_swiglu":
                b = _swap_swiglu_rows(b) if b is not None else None
            value = (
                TurboLoraWeight(base, a, b, is_full_fused_qkv=transform.startswith("qkv:"))
                if a is not None
                else base
            )
            if transform.startswith("qkv:"):
                rows = (base.qweight.shape[0] if quantized_qkv is not None else base.shape[0]) // 3
                values = []
                for index in range(3):
                    start, end = index * rows, (index + 1) * rows
                    child_base = (
                        quantized_qkv[index] if quantized_qkv is not None else base[start:end]
                    )
                    values.append(
                        TurboLoraWeight(
                            child_base,
                            a,
                            b[start:end],
                            packed_parent=value,
                            row_slice=(start, end),
                        )
                        if a is not None
                        else child_base
                    )
                mapped[physical] = tuple(values)
            else:
                mapped[physical] = (value,)
        index = int(transform.removeprefix("qkv:")) if transform.startswith("qkv:") else 0
        result[logical_name] = mapped[physical][index]
    return result
