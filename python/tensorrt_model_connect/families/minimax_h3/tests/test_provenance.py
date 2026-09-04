# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from tensorrt_model_connect.bundle_writer import BundleInfo, BundleSection, write_bundle
from tensorrt_model_connect.families.minimax_h3 import provenance
from tensorrt_model_connect.families.minimax_h3.checkpoint_manifest import (
    BASE_CHECKPOINT_BYTES,
    BASE_CHECKPOINT_FILE_COUNT,
    BASE_CHECKPOINT_FILES,
    BASE_CHECKPOINT_INVENTORY_SHA256,
)
from tensorrt_model_connect.families.minimax_h3.config import (
    DEFAULT_WORKSPACE_LIMIT_BYTES,
    SOL_ENGINE_1344X768_124F,
    SOL_ENGINE_1344X768_124_TO_345F,
    default_workspace_limit_bytes,
    native_plan_filenames,
)
from tensorrt_model_connect.families.minimax_h3.provenance import (
    CHECKPOINT_REVISION,
    PLAN_FILENAMES,
    atomic_write_json,
    builder_source_sha256,
    checkpoint_snapshot_record,
    file_identity,
    file_record,
    serialized_profile,
    sha256_file,
    validate_build_receipt,
    validate_component_build_receipt,
    validate_file_identity,
    validate_git_archive_source_unchanged,
    validate_native_bundle_config,
    validated_git_archive_source_record,
    validated_git_source_record,
)

FAMILY_ROOT = Path(__file__).resolve().parents[1]
BUILD_HELPER = FAMILY_ROOT.parents[3] / "tests/e2e/models/minimax_h3/build_native_components.py"
SOURCE_REVISION = "a" * 40


def _test_archive_inventory(root: Path) -> dict[str, int | str]:
    """Reproduce the documented path/kind/size/content inventory independently."""

    digest = hashlib.sha256()
    entry_count = 0
    total_bytes = 0
    paths = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
    for path in paths:
        if path.is_dir():
            continue
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            kind = "L"
            payload = os.fsencode(os.readlink(path))
        else:
            kind = "F"
            payload = path.read_bytes()
        content_sha256 = hashlib.sha256(payload).hexdigest()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(kind.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(len(payload)).encode("ascii"))
        digest.update(b"\0")
        digest.update(content_sha256.encode("ascii"))
        digest.update(b"\n")
        entry_count += 1
        total_bytes += len(payload)
    return {
        "entry_count": entry_count,
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def _git_archive_fixture(tmp_path: Path, monkeypatch):
    storage_root = tmp_path / "reference-private"
    archive_root = storage_root / provenance.DIFFUSERS_REFERENCE_RELATIVE_PATH
    entrypoint = archive_root / provenance.DIFFUSERS_REFERENCE_ENTRYPOINT
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_bytes(b'__version__ = "test"\n')
    (archive_root / "README.md").write_bytes(b"reference archive\n")
    agents = archive_root / ".ai" / "AGENTS.md"
    agents.parent.mkdir()
    agents.write_bytes(b"archive guidance\n")
    (archive_root / "AGENTS.md").symlink_to(".ai/AGENTS.md")

    inventory = _test_archive_inventory(archive_root)
    entrypoint_record = file_record(entrypoint)
    monkeypatch.setattr(provenance, "DIFFUSERS_REFERENCE_CONTAINER_ROOT", str(storage_root))
    monkeypatch.setattr(
        provenance, "DIFFUSERS_REFERENCE_ENTRYPOINT_BYTES", entrypoint_record["bytes"]
    )
    monkeypatch.setattr(
        provenance,
        "DIFFUSERS_REFERENCE_ENTRYPOINT_SHA256",
        entrypoint_record["sha256"],
    )
    monkeypatch.setattr(provenance, "DIFFUSERS_REFERENCE_ARCHIVE_ENTRIES", inventory["entry_count"])
    monkeypatch.setattr(provenance, "DIFFUSERS_REFERENCE_ARCHIVE_BYTES", inventory["bytes"])
    monkeypatch.setattr(provenance, "DIFFUSERS_REFERENCE_ARCHIVE_SHA256", inventory["sha256"])

    evidence = {
        "schema_version": 1,
        "model": "minimax_h3",
        "isolation": "selected-pinned-private",
        "repository": provenance.DIFFUSERS_REFERENCE_REPOSITORY,
        "reference_revision": provenance.DIFFUSERS_REFERENCE_REVISION,
        "reference_tree": provenance.DIFFUSERS_REFERENCE_TREE,
        "relative_path": provenance.DIFFUSERS_REFERENCE_RELATIVE_PATH,
        "entrypoint": provenance.DIFFUSERS_REFERENCE_ENTRYPOINT,
        "container_storage_root": str(storage_root),
        "copy_method": "git-archive",
    }
    evidence_path = tmp_path / "artifacts" / "model-reference-cache.json"
    evidence_path.parent.mkdir()
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")
    return entrypoint, evidence_path, archive_root, evidence


def _git_blob_id(payload: bytes) -> str:
    try:
        digest = hashlib.sha1(usedforsecurity=False)
    except TypeError:  # pragma: no cover - for older Python implementations
        digest = hashlib.sha1()
    digest.update(f"blob {len(payload)}\0".encode())
    digest.update(payload)
    return digest.hexdigest()


def _snapshot_payloads() -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    shard_specs = (
        (
            "text_encoder/model-00001-of-00001.safetensors",
            "text_encoder/model.safetensors.index.json",
        ),
        (
            "transformer/diffusion_pytorch_model-00001-of-00001.safetensors",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
        ),
        (
            "vae/diffusion_pytorch_model-00001-of-00001.safetensors",
            "vae/diffusion_pytorch_model.safetensors.index.json",
        ),
    )
    for index, (shard, index_name) in enumerate(shard_specs, start=1):
        result[shard] = f"weight-{index}".encode()
        result[index_name] = json.dumps({"weight_map": {"weight": Path(shard).name}}).encode()
    result.update(
        {
            "audio_vae/diffusion_pytorch_model.safetensors": b"audio-vae-weight",
            "audio_vae/config.json": b"{}",
            "modular_model_index.json": b"{}",
            "scheduler/scheduler_config.json": b"{}",
            "tokenizer/tokenizer.json": b'{"model":"test"}',
        }
    )
    return result


def _snapshot_blob_id(relative: str, payload: bytes) -> str:
    return (
        hashlib.sha256(payload).hexdigest()
        if Path(relative).suffix == ".safetensors"
        else _git_blob_id(payload)
    )


def _test_checkpoint_records() -> dict[str, dict[str, int | str]]:
    return {
        relative: {
            "blob_id": _snapshot_blob_id(relative, payload),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for relative, payload in _snapshot_payloads().items()
    }


@pytest.fixture(autouse=True)
def _use_test_checkpoint_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    records = _test_checkpoint_records()
    payload = {
        "repository": provenance.CHECKPOINT_REPOSITORY,
        "revision": CHECKPOINT_REVISION,
        "files": records,
    }
    monkeypatch.setattr(provenance, "_EXPECTED_CHECKPOINT_FILE_RECORDS", records)
    monkeypatch.setattr(
        provenance,
        "BASE_CHECKPOINT_INVENTORY_SHA256",
        hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    )


def _link_blob(snapshot: Path, relative: str, payload: bytes) -> Path:
    blob_id = _snapshot_blob_id(relative, payload)
    blob = snapshot.parent.parent / "blobs" / blob_id
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(payload)
    link = snapshot / relative
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(os.path.relpath(blob, link.parent))
    return link


def _snapshot(tmp_path: Path) -> Path:
    snapshot = tmp_path / "models--MiniMaxAI--MiniMax-H3" / "snapshots" / CHECKPOINT_REVISION
    snapshot.mkdir(parents=True)
    for relative, payload in _snapshot_payloads().items():
        _link_blob(snapshot, relative, payload)
    return snapshot


def _write_local_download_entry(snapshot: Path, relative: str, payload: bytes) -> Path:
    path = snapshot / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    blob_id = _snapshot_blob_id(relative, payload)
    metadata = snapshot / ".cache" / "huggingface" / "download" / f"{relative}.metadata"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(f"{CHECKPOINT_REVISION}\n{blob_id}\n1700000000.0\n", encoding="utf-8")
    return path


def _plain_snapshot(tmp_path: Path, name: str = "authorized-local-checkpoint") -> Path:
    snapshot = tmp_path / name
    snapshot.mkdir(parents=True)
    for relative, payload in _snapshot_payloads().items():
        _write_local_download_entry(snapshot, relative, payload)
    return snapshot


def _receipt(tmp_path: Path) -> tuple[dict, Path, Path, Path]:
    snapshot = _snapshot(tmp_path)
    plans = tmp_path / "plans"
    plans.mkdir()
    components = {}
    for index, filename in enumerate(PLAN_FILENAMES, start=1):
        path = plans / filename
        path.write_bytes(bytes([index]) * index)
        components[filename] = file_record(path)
    tokenizer = snapshot / "tokenizer" / "tokenizer.json"
    receipt = {
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_snapshot": checkpoint_snapshot_record(snapshot),
        "source_revision": SOURCE_REVISION,
        "builder_source_sha256": builder_source_sha256(),
        "build_helper_sha256": sha256_file(BUILD_HELPER),
        "profile": serialized_profile(SOL_ENGINE_1344X768_124F),
        "assets": {"tokenizer.json": file_record(tokenizer)},
        "workspace_limit_bytes": dict(DEFAULT_WORKSPACE_LIMIT_BYTES),
        "components": components,
    }
    return receipt, plans, snapshot, tokenizer


def _validate(receipt: dict, plans: Path, snapshot: Path, tokenizer: Path) -> None:
    validate_build_receipt(
        receipt,
        plans_dir=plans,
        snapshot=snapshot,
        tokenizer=tokenizer,
        build_helper=BUILD_HELPER,
        source_revision=SOURCE_REVISION,
        profile=SOL_ENGINE_1344X768_124F,
        hash_files=True,
    )


def test_complete_native_build_receipt_is_accepted(tmp_path: Path) -> None:
    receipt, plans, snapshot, tokenizer = _receipt(tmp_path)
    _validate(receipt, plans, snapshot, tokenizer)


def test_first_block_cache_build_receipt_selects_exact_split_plans(tmp_path: Path) -> None:
    receipt, plans, snapshot, tokenizer = _receipt(tmp_path)
    profile = replace(SOL_ENGINE_1344X768_124F, first_block_cache=True)
    selected = native_plan_filenames(first_block_cache=True)
    components = {}
    for index, filename in enumerate(selected, start=1):
        path = plans / filename
        if not path.exists():
            path.write_bytes(bytes([index]) * index)
        components[filename] = file_record(path)
    receipt["profile"] = serialized_profile(profile)
    receipt["workspace_limit_bytes"] = default_workspace_limit_bytes(first_block_cache=True)
    receipt["components"] = components
    validate_build_receipt(
        receipt,
        plans_dir=plans,
        snapshot=snapshot,
        tokenizer=tokenizer,
        build_helper=BUILD_HELPER,
        source_revision=SOURCE_REVISION,
        profile=profile,
        hash_files=True,
    )

    receipt["components"]["denoiser.plan"] = {"bytes": 1, "sha256": "0" * 64}
    with pytest.raises(ValueError, match="selected native plans"):
        validate_build_receipt(
            receipt,
            plans_dir=plans,
            snapshot=snapshot,
            tokenizer=tokenizer,
            build_helper=BUILD_HELPER,
            source_revision=SOURCE_REVISION,
            profile=profile,
            hash_files=False,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("checkpoint_revision", "b" * 40),
        ("source_revision", "b" * 40),
        ("builder_source_sha256", "0" * 64),
        ("build_helper_sha256", "0" * 64),
        ("profile", {}),
    ],
)
def test_native_build_receipt_rejects_provenance_mismatch(
    tmp_path: Path, field: str, value
) -> None:
    receipt, plans, snapshot, tokenizer = _receipt(tmp_path)
    receipt[field] = value
    with pytest.raises(ValueError, match=field):
        _validate(receipt, plans, snapshot, tokenizer)


def test_native_build_receipt_rejects_incomplete_or_invalid_artifacts(tmp_path: Path) -> None:
    receipt, plans, snapshot, tokenizer = _receipt(tmp_path)
    incomplete = copy.deepcopy(receipt)
    incomplete["components"].pop(PLAN_FILENAMES[0])
    with pytest.raises(ValueError, match="selected native plans"):
        _validate(incomplete, plans, snapshot, tokenizer)

    malformed = copy.deepcopy(receipt)
    malformed["components"][PLAN_FILENAMES[0]]["sha256"] = "bad"
    with pytest.raises(ValueError, match="invalid SHA256"):
        _validate(malformed, plans, snapshot, tokenizer)

    wrong_size = copy.deepcopy(receipt)
    wrong_size["components"][PLAN_FILENAMES[0]]["bytes"] += 1
    with pytest.raises(ValueError, match="size"):
        _validate(wrong_size, plans, snapshot, tokenizer)

    plan = plans / PLAN_FILENAMES[-1]
    plan.write_bytes(b"x" * plan.stat().st_size)
    with pytest.raises(ValueError, match="artifact SHA256"):
        _validate(receipt, plans, snapshot, tokenizer)


def test_native_build_receipt_requires_exact_positive_workspace_limits(tmp_path: Path) -> None:
    receipt, plans, snapshot, tokenizer = _receipt(tmp_path)
    override = {filename: 8 << 30 for filename in PLAN_FILENAMES}
    receipt["workspace_limit_bytes"] = override
    _validate(receipt, plans, snapshot, tokenizer)

    malformed_receipts = []
    missing = copy.deepcopy(receipt)
    missing.pop("workspace_limit_bytes")
    malformed_receipts.append(missing)
    wrong_keys = copy.deepcopy(receipt)
    wrong_keys["workspace_limit_bytes"].pop(PLAN_FILENAMES[0])
    malformed_receipts.append(wrong_keys)
    extra_key = copy.deepcopy(receipt)
    extra_key["workspace_limit_bytes"]["extra.plan"] = 1
    malformed_receipts.append(extra_key)
    for value in (0, -1, True, 1.5, "8589934592"):
        invalid_value = copy.deepcopy(receipt)
        invalid_value["workspace_limit_bytes"][PLAN_FILENAMES[0]] = value
        malformed_receipts.append(invalid_value)

    for malformed in malformed_receipts:
        with pytest.raises(ValueError, match="workspace_limit_bytes"):
            _validate(malformed, plans, snapshot, tokenizer)


def test_snapshot_inventory_rejects_noncanonical_inputs(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    wrong_revision = snapshot.parent / ("b" * 40)
    wrong_revision.mkdir()
    with pytest.raises(ValueError, match="pinned snapshot"):
        checkpoint_snapshot_record(wrong_revision)

    shard = snapshot / "vae/diffusion_pytorch_model-00001-of-00001.safetensors"
    shard.unlink()
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"weight")
    shard.symlink_to(outside)
    with pytest.raises(ValueError, match="leaves its canonical blob cache"):
        checkpoint_snapshot_record(snapshot)


def test_checked_in_base_manifest_is_exact_and_path_free() -> None:
    files = {
        relative: {"blob_id": blob_id, "bytes": size, "sha256": sha256}
        for relative, size, blob_id, sha256 in BASE_CHECKPOINT_FILES
    }
    payload = {
        "repository": provenance.CHECKPOINT_REPOSITORY,
        "revision": CHECKPOINT_REVISION,
        "files": files,
    }
    assert len(files) == BASE_CHECKPOINT_FILE_COUNT == 55
    assert sum(record["bytes"] for record in files.values()) == BASE_CHECKPOINT_BYTES
    assert sum(len(record["blob_id"]) == 64 for record in files.values()) == 32
    assert sum(len(record["blob_id"]) == 40 for record in files.values()) == 23
    assert all(Path(relative).parts[0] != "transformer_ref" for relative in files)
    assert (
        hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        == BASE_CHECKPOINT_INVENTORY_SHA256
        == "cd0e54d3250927be68e681dde85797b9b47ef6b072ed884a6901540a1749bcf6"
    )


def test_plain_local_dir_snapshot_is_verified_and_path_independent(tmp_path: Path) -> None:
    first = _plain_snapshot(tmp_path, "first-local-dir")
    second = _plain_snapshot(tmp_path, "second-local-dir")

    first_record = checkpoint_snapshot_record(first)
    second_record = checkpoint_snapshot_record(second)

    assert first_record == second_record
    assert str(first) not in json.dumps(first_record)
    assert first_record["revision"] == CHECKPOINT_REVISION
    assert first_record["file_count"] == 11
    assert all(
        set(record) == {"blob_id", "bytes", "sha256"} for record in first_record["files"].values()
    )

    unexpected_top_level = copy.deepcopy(first_record)
    unexpected_top_level["local_path"] = str(first)
    with pytest.raises(ValueError, match="unexpected fields"):
        provenance.validate_checkpoint_snapshot_record(unexpected_top_level)

    unexpected_file_field = copy.deepcopy(first_record)
    unexpected_file_field["files"]["tokenizer/tokenizer.json"]["local_path"] = str(first)
    with pytest.raises(ValueError, match="unexpected fields"):
        provenance.validate_checkpoint_snapshot_record(unexpected_file_field)


def test_canonical_and_plain_layouts_produce_the_same_record(tmp_path: Path) -> None:
    canonical = _snapshot(tmp_path)
    local_dir = _plain_snapshot(tmp_path, "plain-layout")

    assert checkpoint_snapshot_record(canonical) == checkpoint_snapshot_record(local_dir)


def test_plain_local_dir_named_snapshots_is_not_misclassified(tmp_path: Path) -> None:
    snapshot = _plain_snapshot(tmp_path / "snapshots", "local-dir")

    assert checkpoint_snapshot_record(snapshot)["file_count"] == 11


def test_plain_local_dir_snapshot_rejects_modified_content(tmp_path: Path) -> None:
    snapshot = _plain_snapshot(tmp_path)
    shard = snapshot / "vae" / "diffusion_pytorch_model-00001-of-00001.safetensors"
    shard.write_bytes(b"changed!")
    with pytest.raises(ValueError, match="LFS SHA256 mismatch"):
        checkpoint_snapshot_record(snapshot)

    snapshot = _plain_snapshot(tmp_path, "small-file-change")
    tokenizer = snapshot / "tokenizer" / "tokenizer.json"
    tokenizer.write_bytes(b'{"model":"evil"}')
    with pytest.raises(ValueError, match="Git blob ID mismatch"):
        checkpoint_snapshot_record(snapshot)


def test_plain_local_dir_snapshot_requires_exact_download_metadata(tmp_path: Path) -> None:
    snapshot = _plain_snapshot(tmp_path)
    metadata_root = snapshot / ".cache" / "huggingface" / "download"
    tokenizer_metadata = metadata_root / "tokenizer" / "tokenizer.json.metadata"
    fields = tokenizer_metadata.read_text(encoding="utf-8").splitlines()
    tokenizer_metadata.write_text(f"{'b' * 40}\n{fields[1]}\n{fields[2]}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pinned revision"):
        checkpoint_snapshot_record(snapshot)

    snapshot = _plain_snapshot(tmp_path, "missing-metadata")
    (
        snapshot / ".cache" / "huggingface" / "download" / "tokenizer" / "tokenizer.json.metadata"
    ).unlink()
    with pytest.raises(ValueError, match="content and download metadata differ"):
        checkpoint_snapshot_record(snapshot)

    snapshot = _plain_snapshot(tmp_path, "orphan-metadata")
    orphan = snapshot / ".cache" / "huggingface" / "download" / "orphan.json.metadata"
    orphan.write_text(f"{CHECKPOINT_REVISION}\n{'c' * 40}\n1700000000.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="orphan_metadata"):
        checkpoint_snapshot_record(snapshot)


def test_plain_snapshot_rejects_jointly_forged_or_extra_content(tmp_path: Path) -> None:
    snapshot = _plain_snapshot(tmp_path)
    _write_local_download_entry(snapshot, "tokenizer/tokenizer.json", b'{"model":"evil"}')
    with pytest.raises(ValueError, match="does not match pinned file"):
        checkpoint_snapshot_record(snapshot)

    snapshot = _plain_snapshot(tmp_path, "extra-content")
    _write_local_download_entry(snapshot, "attacker/extra.json", b"self-consistent")
    with pytest.raises(ValueError, match="unexpected=.*attacker/extra.json"):
        checkpoint_snapshot_record(snapshot)


def test_plain_snapshot_excludes_optional_transformer_ref_and_cache_debris(
    tmp_path: Path,
) -> None:
    snapshot = _plain_snapshot(tmp_path)
    baseline = checkpoint_snapshot_record(snapshot)
    _write_local_download_entry(snapshot, "transformer_ref/config.json", b"partial")
    download_root = snapshot / ".cache" / "huggingface" / "download"
    incomplete = download_root / "transformer_ref" / "shard.incomplete"
    incomplete.write_bytes(b"partial shard")
    lock = download_root / "transformer_ref" / "shard.lock"
    lock.write_bytes(b"")

    current = checkpoint_snapshot_record(snapshot)

    assert current == baseline
    assert not any(Path(relative).parts[0] == "transformer_ref" for relative in current["files"])


def test_canonical_snapshot_hashes_lfs_and_excludes_transformer_ref(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    baseline = checkpoint_snapshot_record(snapshot)
    _link_blob(snapshot, "transformer_ref/config.json", b"partial")
    assert checkpoint_snapshot_record(snapshot) == baseline

    shard = snapshot / "vae" / "diffusion_pytorch_model-00001-of-00001.safetensors"
    shard.resolve(strict=True).write_bytes(b"changed!")
    with pytest.raises(ValueError, match="LFS SHA256 mismatch"):
        checkpoint_snapshot_record(snapshot)


def test_canonical_snapshot_rejects_linked_directory(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    outside = tmp_path / "outside-directory"
    outside.mkdir()
    (snapshot / "linked-directory").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="linked directory"):
        checkpoint_snapshot_record(snapshot)


def test_canonical_snapshot_binds_readlink_to_link_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _snapshot(tmp_path)
    link = snapshot / "tokenizer" / "tokenizer.json"
    replacement = snapshot.parent.parent / "blobs" / ("f" * 40)
    replacement.write_bytes(b"replacement")
    original_readlink = os.readlink
    replaced = False

    def racing_readlink(path: str | bytes | os.PathLike[str] | os.PathLike[bytes]):
        nonlocal replaced
        payload = original_readlink(path)
        if Path(path) == link and not replaced:
            replaced = True
            link.unlink()
            link.symlink_to(os.path.relpath(replacement, link.parent))
        return payload

    monkeypatch.setattr(provenance.os, "readlink", racing_readlink)

    with pytest.raises(ValueError, match="canonical snapshot link changed"):
        checkpoint_snapshot_record(snapshot)


def test_snapshot_inventory_change_invalidates_existing_receipt(tmp_path: Path) -> None:
    receipt, plans, snapshot, tokenizer = _receipt(tmp_path)
    shard = snapshot / "vae/diffusion_pytorch_model-00001-of-00001.safetensors"
    shard.unlink()
    _link_blob(snapshot, str(shard.relative_to(snapshot)), b"different-weight")
    with pytest.raises(ValueError, match="exact pinned base inventory"):
        _validate(receipt, plans, snapshot, tokenizer)


def test_component_receipt_binds_vae_plan_without_other_plan_files(tmp_path: Path) -> None:
    receipt, plans, _snapshot_path, _tokenizer = _receipt(tmp_path)
    vae = plans / "vae_tile_decoder.plan"
    validate_component_build_receipt(
        receipt,
        component="vae_tile_decoder.plan",
        artifact=vae,
        build_helper=BUILD_HELPER,
        source_revision=SOURCE_REVISION,
        profile=SOL_ENGINE_1344X768_124F,
        hash_file=True,
    )
    vae.write_bytes(b"z" * vae.stat().st_size)
    with pytest.raises(ValueError, match="artifact SHA256"):
        validate_component_build_receipt(
            receipt,
            component="vae_tile_decoder.plan",
            artifact=vae,
            build_helper=BUILD_HELPER,
            source_revision=SOURCE_REVISION,
            profile=SOL_ENGINE_1344X768_124F,
            hash_file=True,
        )


def test_atomic_receipt_write_preserves_previous_file_on_replace_failure(
    tmp_path: Path, monkeypatch
) -> None:
    destination = tmp_path / "receipt.json"
    destination.write_text('{"status":"old"}')

    def fail_replace(_source, _destination):
        raise OSError("injected replace failure")

    monkeypatch.setattr(provenance.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        atomic_write_json(destination, {"status": "new"})
    assert json.loads(destination.read_text()) == {"status": "old"}
    assert not list(tmp_path.glob(".receipt.json.tmp.*"))


def test_native_bundle_config_is_bound_to_current_family_source(tmp_path: Path) -> None:
    config = {
        "model_type": "minimax_h3",
        "runtime_strategy": "diffusion_minimax_h3",
        "checkpoint_revision": CHECKPOINT_REVISION,
        "source_revision": SOURCE_REVISION,
        "builder_source_sha256": builder_source_sha256(),
        "checkpoint_inventory_sha256": "a" * 64,
        "workspace_limit_bytes": dict(DEFAULT_WORKSPACE_LIMIT_BYTES),
        "context_parallel_size": 1,
        "padded_sequence_length": SOL_ENGINE_1344X768_124_TO_345F.padded_sequence_length,
        "packed_sequence_length_min": SOL_ENGINE_1344X768_124_TO_345F.min_sequence_length,
        "packed_sequence_length_opt": SOL_ENGINE_1344X768_124_TO_345F.opt_sequence_length,
        "packed_sequence_length_max": SOL_ENGINE_1344X768_124_TO_345F.sequence_length,
        "canvas_multiple": 32,
        "canvas_short_edge": 768,
        "canvas_max_pixels": 1032192,
        "explicit_canvas_sizes": [[544, 960], [960, 544]],
        "min_aspect_ratio": 0.25,
        "max_aspect_ratio": 4.0,
        "vae_tile_batch": 28,
        "vae_tile_batch_min": 15,
        "vae_tile_batch_opt": 28,
        "vae_tile_batch_max": 33,
        "plan_sha256": {
            filename: f"{index:064x}" for index, filename in enumerate(PLAN_FILENAMES, start=1)
        },
    }
    bundle = tmp_path / "model.bundle"
    write_bundle(
        bundle,
        BundleInfo(model_id="MiniMaxAI/MiniMax-H3"),
        [BundleSection("config.json", json.dumps(config).encode())],
    )
    validate_native_bundle_config(bundle, source_revision=SOURCE_REVISION)

    config["workspace_limit_bytes"] = {filename: 8 << 30 for filename in PLAN_FILENAMES}
    write_bundle(
        bundle,
        BundleInfo(model_id="MiniMaxAI/MiniMax-H3"),
        [BundleSection("config.json", json.dumps(config).encode())],
    )
    validated = validate_native_bundle_config(bundle, source_revision=SOURCE_REVISION)
    assert validated["workspace_limit_bytes"] == config["workspace_limit_bytes"]

    t2va_only = copy.deepcopy(config)
    t2va_only["public_workflows"] = ["t2va"]
    optional_fl2va = {"vision_encoder.plan", "fl2va_keyframe_vae_encoder.plan"}
    t2va_only["workspace_limit_bytes"] = {
        name: value
        for name, value in t2va_only["workspace_limit_bytes"].items()
        if name not in optional_fl2va
    }
    t2va_only["plan_sha256"] = {
        name: value
        for name, value in t2va_only["plan_sha256"].items()
        if name not in optional_fl2va
    }
    write_bundle(
        bundle,
        BundleInfo(model_id="MiniMaxAI/MiniMax-H3"),
        [BundleSection("config.json", json.dumps(t2va_only).encode())],
    )
    assert validate_native_bundle_config(
        bundle, source_revision=SOURCE_REVISION
    )["public_workflows"] == ["t2va"]

    invalid_workflows = copy.deepcopy(t2va_only)
    invalid_workflows["public_workflows"] = ["fl2va"]
    write_bundle(
        bundle,
        BundleInfo(model_id="MiniMaxAI/MiniMax-H3"),
        [BundleSection("config.json", json.dumps(invalid_workflows).encode())],
    )
    with pytest.raises(ValueError, match="public_workflows"):
        validate_native_bundle_config(bundle, source_revision=SOURCE_REVISION)

    config["workspace_limit_bytes"][PLAN_FILENAMES[0]] = True
    write_bundle(
        bundle,
        BundleInfo(model_id="MiniMaxAI/MiniMax-H3"),
        [BundleSection("config.json", json.dumps(config).encode())],
    )
    with pytest.raises(ValueError, match="workspace_limit_bytes"):
        validate_native_bundle_config(bundle, source_revision=SOURCE_REVISION)

    config["workspace_limit_bytes"] = {filename: 8 << 30 for filename in PLAN_FILENAMES}
    config["builder_source_sha256"] = "0" * 64
    write_bundle(
        bundle,
        BundleInfo(model_id="MiniMaxAI/MiniMax-H3"),
        [BundleSection("config.json", json.dumps(config).encode())],
    )
    with pytest.raises(ValueError, match="builder_source_sha256"):
        validate_native_bundle_config(bundle, source_revision=SOURCE_REVISION)


def test_file_identity_detects_same_size_replacement(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"old")
    identity = file_identity(artifact)
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"new")
    os.replace(replacement, artifact)
    with pytest.raises(ValueError, match="changed while it was in use"):
        validate_file_identity(artifact, identity, "artifact")


def test_git_source_record_requires_exact_clean_head(tmp_path: Path) -> None:
    checkout = tmp_path / "upstream"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "test@example.com"], check=True
    )
    subprocess.run(["git", "-C", str(checkout), "config", "user.name", "Test"], check=True)
    entrypoint = checkout / "src" / "package" / "__init__.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text('__version__ = "test"\n')
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
    subprocess.run(["git", "-C", str(checkout), "commit", "-qm", "initial"], check=True)
    revision = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    record = validated_git_source_record(
        entrypoint, expected_revision=revision, label="test source"
    )
    assert record["revision"] == revision
    with pytest.raises(ValueError, match="revision mismatch"):
        validated_git_source_record(entrypoint, expected_revision="f" * 40, label="test source")

    entrypoint.write_text('__version__ = "dirty"\n')
    with pytest.raises(ValueError, match="tracked modifications"):
        validated_git_source_record(entrypoint, expected_revision=revision, label="test source")


def test_diffusers_git_archive_contract_is_exact() -> None:
    assert (
        provenance.DIFFUSERS_REFERENCE_REPOSITORY == "https://github.com/huggingface/diffusers.git"
    )
    assert provenance.DIFFUSERS_REFERENCE_REVISION == "abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc"
    assert provenance.DIFFUSERS_REFERENCE_TREE == "a9aeec5268dd9661565a3e0af9b298744eb416b2"
    assert (
        provenance.DIFFUSERS_REFERENCE_RELATIVE_PATH
        == "minimax_h3/reference/diffusers-abc5e9bf71fd"
    )
    assert provenance.DIFFUSERS_REFERENCE_ENTRYPOINT == "src/diffusers/__init__.py"
    assert provenance.DIFFUSERS_REFERENCE_ENTRYPOINT_BYTES == 65_047
    assert (
        provenance.DIFFUSERS_REFERENCE_ENTRYPOINT_SHA256
        == "78bac2aa899c34b6d504e8dfb128d9475ad7baee179b3ad97d09ccef25999916"
    )
    assert provenance.DIFFUSERS_REFERENCE_ARCHIVE_ENTRIES == 2_772
    assert provenance.DIFFUSERS_REFERENCE_ARCHIVE_BYTES == 50_469_043
    assert (
        provenance.DIFFUSERS_REFERENCE_ARCHIVE_SHA256
        == "372c820aece801258bd4cea2458a2b85ad536e9262d7b0bbcdd450eda2d664a9"
    )
    assert provenance.DIFFUSERS_REFERENCE_CONTAINER_ROOT == "/work/reference-private"


def test_git_archive_source_record_accepts_exact_private_copy(tmp_path: Path, monkeypatch) -> None:
    entrypoint, evidence_path, _archive_root, _evidence = _git_archive_fixture(
        tmp_path, monkeypatch
    )

    record = validated_git_archive_source_record(
        entrypoint,
        evidence_path=evidence_path,
        label="Diffusers reference",
    )

    assert record["qualification"] == "selected-pinned-git-archive"
    assert record["repository"] == provenance.DIFFUSERS_REFERENCE_REPOSITORY
    assert record["revision"] == provenance.DIFFUSERS_REFERENCE_REVISION
    assert record["tree"] == provenance.DIFFUSERS_REFERENCE_TREE
    assert record["entrypoint_record"] == file_record(entrypoint)
    assert record["archive_inventory"] == _test_archive_inventory(
        Path(provenance.DIFFUSERS_REFERENCE_CONTAINER_ROOT)
        / provenance.DIFFUSERS_REFERENCE_RELATIVE_PATH
    )
    validate_git_archive_source_unchanged(
        entrypoint,
        evidence_path=evidence_path,
        expected_record=record,
        label="Diffusers reference",
    )


def test_git_archive_source_record_rejects_symlinked_evidence(tmp_path: Path, monkeypatch) -> None:
    entrypoint, evidence_path, _archive_root, _evidence = _git_archive_fixture(
        tmp_path, monkeypatch
    )
    symlink = evidence_path.with_name("linked-model-reference-cache.json")
    symlink.symlink_to(evidence_path)

    with pytest.raises(ValueError, match="evidence must not be a symlink"):
        validated_git_archive_source_record(
            entrypoint,
            evidence_path=symlink,
            label="Diffusers reference",
        )


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "model",
        "isolation",
        "repository",
        "reference_revision",
        "reference_tree",
        "relative_path",
        "entrypoint",
        "container_storage_root",
        "copy_method",
    ],
)
def test_git_archive_source_record_rejects_evidence_mismatch(
    tmp_path: Path, monkeypatch, field: str
) -> None:
    entrypoint, evidence_path, _archive_root, evidence = _git_archive_fixture(tmp_path, monkeypatch)
    evidence[field] = False if field == "schema_version" else "wrong"
    evidence_path.write_text(json.dumps(evidence))

    with pytest.raises(ValueError, match=f"evidence mismatch for {field}"):
        validated_git_archive_source_record(
            entrypoint,
            evidence_path=evidence_path,
            label="Diffusers reference",
        )


def test_git_archive_source_record_rejects_unsupported_or_duplicate_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    entrypoint, evidence_path, _archive_root, evidence = _git_archive_fixture(tmp_path, monkeypatch)
    evidence["unexpected"] = True
    evidence_path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="unsupported fields"):
        validated_git_archive_source_record(
            entrypoint,
            evidence_path=evidence_path,
            label="Diffusers reference",
        )

    evidence_path.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError, match="duplicate keys"):
        validated_git_archive_source_record(
            entrypoint,
            evidence_path=evidence_path,
            label="Diffusers reference",
        )


@pytest.mark.parametrize("mutation", ["add", "delete", "modify"])
def test_git_archive_source_revalidation_detects_inventory_mutation(
    tmp_path: Path, monkeypatch, mutation: str
) -> None:
    entrypoint, evidence_path, archive_root, _evidence = _git_archive_fixture(tmp_path, monkeypatch)
    record = validated_git_archive_source_record(
        entrypoint,
        evidence_path=evidence_path,
        label="Diffusers reference",
    )

    readme = archive_root / "README.md"
    if mutation == "add":
        (archive_root / "added.py").write_bytes(b"added\n")
    elif mutation == "delete":
        readme.unlink()
    else:
        readme.write_bytes(b"tampered archive\n")

    with pytest.raises(ValueError, match="archive inventory"):
        validate_git_archive_source_unchanged(
            entrypoint,
            evidence_path=evidence_path,
            expected_record=record,
            label="Diffusers reference",
        )


def test_git_archive_source_record_rejects_escaping_symlink(tmp_path: Path, monkeypatch) -> None:
    entrypoint, evidence_path, archive_root, _evidence = _git_archive_fixture(tmp_path, monkeypatch)
    outside = tmp_path / "outside.py"
    outside.write_bytes(b"outside\n")
    (archive_root / "escape.py").symlink_to(outside)

    with pytest.raises(ValueError, match="escaping symlink"):
        validated_git_archive_source_record(
            entrypoint,
            evidence_path=evidence_path,
            label="Diffusers reference",
        )


def test_git_archive_source_record_rejects_symlink_loop(tmp_path: Path, monkeypatch) -> None:
    entrypoint, evidence_path, archive_root, _evidence = _git_archive_fixture(tmp_path, monkeypatch)
    (archive_root / "loop-a").symlink_to("loop-b")
    (archive_root / "loop-b").symlink_to("loop-a")

    with pytest.raises(ValueError, match="broken symlink"):
        validated_git_archive_source_record(
            entrypoint,
            evidence_path=evidence_path,
            label="Diffusers reference",
        )


def test_git_archive_source_record_rejects_wrong_import_or_git_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    entrypoint, evidence_path, archive_root, _evidence = _git_archive_fixture(tmp_path, monkeypatch)
    wrong_entrypoint = tmp_path / "diffusers" / "__init__.py"
    wrong_entrypoint.parent.mkdir()
    wrong_entrypoint.write_bytes(entrypoint.read_bytes())
    with pytest.raises(ValueError, match="was not imported"):
        validated_git_archive_source_record(
            wrong_entrypoint,
            evidence_path=evidence_path,
            label="Diffusers reference",
        )

    git_config = archive_root / ".git" / "config"
    git_config.parent.mkdir()
    git_config.write_bytes(b"[core]\n")
    with pytest.raises(ValueError, match="must not contain Git metadata"):
        validated_git_archive_source_record(
            entrypoint,
            evidence_path=evidence_path,
            label="Diffusers reference",
        )
