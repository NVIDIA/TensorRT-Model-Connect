# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from families.minimax_h3 import quantized_checkpoint as checkpoint
from families.minimax_h3.provenance import validate_quantized_transformer_metadata
from families.minimax_h3.quantized_checkpoint import (
    CHECKPOINT_BYTES,
    CHECKPOINT_FILENAME,
    CHECKPOINT_REVISION,
    ConvRotInt8Weight,
    load_selected_quantized_transformer_weights,
    validate_quantized_transformer_checkpoint,
)


def _marker(group_size: int, **extra: object) -> np.ndarray:
    config: dict[str, object] = {
        "format": "int8_tensorwise",
        "convrot": True,
        "convrot_groupsize": group_size,
        **extra,
    }
    return np.frombuffer(json.dumps(config).encode("utf-8"), dtype=np.uint8)


def _write_hf_local_metadata(
    path: Path,
    *,
    revision: str = CHECKPOINT_REVISION,
    etag: str = "etag",
) -> Path:
    relative = Path("diffusion_models") / path.name
    source_root = path.parents[len(relative.parts) - 1]
    metadata = (
        source_root / ".cache" / "huggingface" / "download" / f"{relative.as_posix()}.metadata"
    )
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(f"{revision}\n{etag}\n1700000000.0\n", encoding="utf-8")
    return metadata


def _tiny_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    marker: np.ndarray | None = None,
    config: dict[str, object] | None = None,
    workflow: str = "fl2va",
) -> Path:
    identity_name = (
        "QUANTIZED_REF2VA_CHECKPOINT_IDENTITY"
        if workflow == "ref2va"
        else "QUANTIZED_CHECKPOINT_IDENTITY"
    )
    identity = getattr(checkpoint, identity_name)
    path = tmp_path / identity.filename
    path.parent.mkdir(parents=True)
    tensors = {
        "layer.weight": np.arange(16, dtype=np.int8).reshape(4, 4),
        "layer.weight_scale": np.arange(4, dtype=np.float32).reshape(4, 1),
        "layer.comfy_quant": _marker(4) if marker is None else marker,
    }
    expected_config = {"transformer": {"num_layers": 50}}
    save_file(tensors, path, metadata={"config": json.dumps(config or expected_config)})
    _write_hf_local_metadata(path)
    monkeypatch.setattr(checkpoint, "_EXPECTED_CONFIG", expected_config)
    monkeypatch.setattr(
        checkpoint,
        "_PHYSICAL_SPECS",
        {
            "layer.weight": checkpoint._TensorSpec("I8", (4, 4)),
            "layer.weight_scale": checkpoint._TensorSpec("F32", (4, 1)),
            "layer.comfy_quant": checkpoint._TensorSpec("U8", (len(tensors["layer.comfy_quant"]),)),
        },
    )
    monkeypatch.setattr(checkpoint, "_QUANT_GROUPS", {"layer": 4})
    monkeypatch.setattr(
        checkpoint,
        identity_name,
        replace(
            identity,
            size_bytes=path.stat().st_size,
            tensor_count=3,
            quantized_weight_count=1,
        ),
    )
    return path


def test_bundle_provenance_validates_source_identity_without_quantization_options() -> None:
    metadata = checkpoint.QUANTIZED_CHECKPOINT_IDENTITY.bundle_metadata()
    assert validate_quantized_transformer_metadata(metadata) == metadata
    with pytest.raises(ValueError, match="does not match the public model"):
        validate_quantized_transformer_metadata({**metadata, "revision": "different"})
    with pytest.raises(ValueError, match="does not match the public model"):
        validate_quantized_transformer_metadata({**metadata, "local_path": "private"})


def test_released_identity_and_full_header_contract_are_pinned() -> None:
    assert CHECKPOINT_REVISION == "4cc1d817b6184899b41293954329f576cb5ae86b"
    assert CHECKPOINT_BYTES == 34_038_892_334
    assert len(checkpoint._PHYSICAL_SPECS) == 1_035
    assert len(checkpoint._QUANT_GROUPS) == 250
    assert Counter(spec.dtype for spec in checkpoint._PHYSICAL_SPECS.values()) == {
        "BF16": 272,
        "F32": 263,
        "I8": 250,
        "U8": 250,
    }
    assert Counter(checkpoint._QUANT_GROUPS.values()) == {64: 50, 256: 200}

    metadata = checkpoint.QUANTIZED_CHECKPOINT_IDENTITY.bundle_metadata()

    assert metadata["model_id"] == "Comfy-Org/MiniMax-H3"
    assert metadata["revision"] == CHECKPOINT_REVISION
    assert metadata["filename"] == CHECKPOINT_FILENAME
    assert metadata["runtime_framework"] is None


def test_ref2va_full_identity_and_logical_partitions_match_shared_layout() -> None:
    from families.minimax_h3.ref2va_checkpoint import (
        REF2VA_ADALN_KEYS,
        REF2VA_DENOISER_KEYS,
        REF2VA_FINISH_KEYS,
        REF2VA_HEAD_KEYS,
        REF2VA_TAIL_KEYS,
    )

    identity = checkpoint.QUANTIZED_REF2VA_CHECKPOINT_IDENTITY
    assert identity.model_id == "Comfy-Org/MiniMax-H3"
    assert identity.revision == CHECKPOINT_REVISION
    assert identity.filename == "diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors"
    assert identity.size_bytes == 34_038_894_550
    assert identity.tensor_count == 1_035
    assert identity.quantized_weight_count == 250
    assert identity.filename != checkpoint.QUANTIZED_CHECKPOINT_IDENTITY.filename
    assert set(REF2VA_DENOISER_KEYS) | set(REF2VA_ADALN_KEYS) == set(checkpoint._LOGICAL_MAP)
    for keys, expected_quantized in (
        (REF2VA_HEAD_KEYS, 4),
        (REF2VA_TAIL_KEYS, 49 * 4),
        (REF2VA_ADALN_KEYS, 50),
        (REF2VA_FINISH_KEYS, 0),
    ):
        physical_quantized = {
            checkpoint._LOGICAL_MAP[name][0].removesuffix(".weight") for name in keys
        } & set(checkpoint._QUANT_GROUPS)
        assert len(physical_quantized) == expected_quantized


@pytest.mark.parametrize("workflow", ("fl2va", "ref2va"))
def test_validator_accepts_only_exact_embedded_quant_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workflow: str
) -> None:
    path = _tiny_checkpoint(tmp_path, monkeypatch, workflow=workflow)

    identity = validate_quantized_transformer_checkpoint(path, workflow=workflow)

    assert Path(identity.filename).name == path.name
    assert identity.tensor_count == 3
    assert identity.quantized_weight_count == 1
    assert identity.source_file_identity is not None
    assert identity.source_file_identity.size_bytes == path.stat().st_size


@pytest.mark.parametrize("workflow", ("fl2va", "ref2va"))
def test_validator_requires_pinned_hugging_face_source_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workflow: str
) -> None:
    path = _tiny_checkpoint(tmp_path, monkeypatch, workflow=workflow)
    metadata = _write_hf_local_metadata(path)
    metadata.unlink()

    with pytest.raises(ValueError, match="Cannot validate"):
        validate_quantized_transformer_checkpoint(path, workflow=workflow)


@pytest.mark.parametrize("workflow", ("fl2va", "ref2va"))
def test_validator_rejects_non_pinned_hugging_face_source_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    workflow: str,
) -> None:
    path = _tiny_checkpoint(tmp_path, monkeypatch, workflow=workflow)
    _write_hf_local_metadata(path, revision="b" * 40)

    with pytest.raises(ValueError, match="does not match the pinned revision"):
        validate_quantized_transformer_checkpoint(path, workflow=workflow)


@pytest.mark.parametrize("workflow", ("fl2va", "ref2va"))
def test_canonical_hugging_face_snapshot_layout_is_accepted(tmp_path: Path, workflow: str) -> None:
    repository = tmp_path / "models--Comfy-Org--MiniMax-H3"
    blob = repository / "blobs" / "model-blob"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"pinned blob")
    filename = (
        checkpoint.REF2VA_CHECKPOINT_FILENAME if workflow == "ref2va" else CHECKPOINT_FILENAME
    )
    checkpoint_path = repository / "snapshots" / CHECKPOINT_REVISION / filename
    checkpoint_path.parent.mkdir(parents=True)
    try:
        checkpoint_path.hardlink_to(blob)
    except OSError as error:  # pragma: no cover - filesystem-dependent fallback
        pytest.skip(f"hard links unavailable: {error}")

    checkpoint._authenticate_quantized_source(checkpoint_path, workflow=workflow)


@pytest.mark.parametrize("workflow,other", (("fl2va", "ref2va"), ("ref2va", "fl2va")))
def test_validator_rejects_other_workflow_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workflow: str, other: str
) -> None:
    path = _tiny_checkpoint(tmp_path, monkeypatch, workflow=other)
    with pytest.raises(ValueError, match="checkpoint filename must be"):
        validate_quantized_transformer_checkpoint(path, workflow=workflow)


def test_ref2va_validator_rejects_pruned_checkpoint_filename(tmp_path: Path) -> None:
    path = tmp_path / "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
    path.write_bytes(b"not the full checkpoint")
    with pytest.raises(ValueError, match="checkpoint filename must be"):
        validate_quantized_transformer_checkpoint(path, workflow="ref2va")


def test_validator_and_loader_reject_unknown_workflow_before_reading(tmp_path: Path) -> None:
    path = tmp_path / "missing.safetensors"
    with pytest.raises(ValueError, match="workflow must be"):
        validate_quantized_transformer_checkpoint(path, workflow="unknown")
    with pytest.raises(ValueError, match="workflow must be"):
        load_selected_quantized_transformer_weights(path, (), workflow="unknown")


@pytest.mark.parametrize(
    "marker",
    [
        _marker(4, unknown=True),
        _marker(4, convrot=False),
        _marker(4, format="other"),
        _marker(16),
    ],
)
@pytest.mark.parametrize("workflow", ("fl2va", "ref2va"))
def test_validator_rejects_unknown_or_nonmatching_quant_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, marker: np.ndarray, workflow: str
) -> None:
    path = _tiny_checkpoint(tmp_path, monkeypatch, marker=marker, workflow=workflow)

    with pytest.raises(ValueError, match="unsupported comfy_quant config"):
        validate_quantized_transformer_checkpoint(path, workflow=workflow)


@pytest.mark.parametrize("workflow", ("fl2va", "ref2va"))
def test_validator_rejects_non_full_architecture_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workflow: str
) -> None:
    path = _tiny_checkpoint(
        tmp_path,
        monkeypatch,
        config={"transformer": {"num_layers": 40}},
        workflow=workflow,
    )

    with pytest.raises(ValueError, match="full released 50-layer architecture"):
        validate_quantized_transformer_checkpoint(path, workflow=workflow)


@pytest.mark.parametrize("workflow", ("fl2va", "ref2va"))
def test_loader_maps_qkv_and_swiglu_without_dequantizing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workflow: str
) -> None:
    filename = (
        checkpoint.REF2VA_CHECKPOINT_FILENAME if workflow == "ref2va" else CHECKPOINT_FILENAME
    )
    path = tmp_path / Path(filename).name
    physical_qkv = np.arange(12 * 4, dtype=np.int8).reshape(12, 4)
    qkv_scale = np.arange(12, dtype=np.float32).reshape(12, 1)
    physical_fc1 = np.arange(8 * 4, dtype=np.int8).reshape(8, 4)
    fc1_scale = np.arange(8, dtype=np.float32).reshape(8, 1)
    save_file(
        {
            "blocks.0.attn.qkv_proj.weight": physical_qkv,
            "blocks.0.attn.qkv_proj.weight_scale": qkv_scale,
            "blocks.0.mlp.fc1.weight": physical_fc1,
            "blocks.0.mlp.fc1.weight_scale": fc1_scale,
        },
        path,
    )
    validated = []
    monkeypatch.setattr(
        checkpoint,
        "validate_quantized_transformer_checkpoint",
        lambda value, **kwargs: validated.append((value, kwargs)),
    )
    monkeypatch.setattr(checkpoint, "_NUM_HEADS", 2)
    monkeypatch.setattr(checkpoint, "_HEAD_DIM", 2)
    monkeypatch.setattr(
        checkpoint,
        "_QUANT_GROUPS",
        {
            "blocks.0.attn.qkv_proj": 4,
            "blocks.0.mlp.fc1": 4,
        },
    )
    names = (
        "transformer_blocks.0.attn.to_q.weight",
        "transformer_blocks.0.attn.to_k.weight",
        "transformer_blocks.0.attn.to_v.weight",
        "transformer_blocks.0.ff.net.0.proj.weight",
    )

    loaded = load_selected_quantized_transformer_weights(path, names, workflow=workflow)

    assert validated == [(path, {"workflow": workflow})]
    q, k, v = (loaded[name] for name in names[:3])
    assert all(isinstance(value, ConvRotInt8Weight) for value in (q, k, v))
    assert q.packed_parent is k.packed_parent is v.packed_parent
    assert q.packed_parent is not None and q.packed_parent.is_full_fused_qkv
    expected_packed = physical_qkv
    np.testing.assert_array_equal(q.packed_parent.qweight, expected_packed)
    np.testing.assert_array_equal(q.qweight, expected_packed[:4])
    np.testing.assert_array_equal(k.qweight, expected_packed[4:8])
    np.testing.assert_array_equal(v.qweight, expected_packed[8:])
    assert (q.row_slice, k.row_slice, v.row_slice) == ((0, 4), (4, 8), (8, 12))

    fc1 = loaded[names[3]]
    assert isinstance(fc1, ConvRotInt8Weight)
    np.testing.assert_array_equal(fc1.qweight, np.concatenate((physical_fc1[4:], physical_fc1[:4])))
    np.testing.assert_array_equal(fc1.scale[:, 0], np.r_[np.arange(4, 8), np.arange(4)])


def test_convrot_dataclass_rejects_non_power_of_four_group() -> None:
    with pytest.raises(ValueError, match="power of four"):
        ConvRotInt8Weight(
            qweight=np.zeros((4, 8), dtype=np.int8),
            scale=np.ones((4, 1), dtype=np.float32),
            group_size=8,
        )
