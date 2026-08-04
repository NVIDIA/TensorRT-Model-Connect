# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path

import pytest
from tensorrt_model_connect.bundle_writer import BundleInfo, BundleSection, write_bundle
from tensorrt_model_connect.families.minimax_h3 import provenance
from tensorrt_model_connect.families.minimax_h3.config import SOL_ENGINE_1344X768_124F
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
    validate_native_bundle_config,
    validated_git_source_record,
)

FAMILY_ROOT = Path(__file__).resolve().parents[1]
BUILD_HELPER = FAMILY_ROOT.parents[3] / "tests/e2e/models/minimax_h3/build_native_components.py"
SOURCE_REVISION = "a" * 40


def _link_blob(snapshot: Path, relative: str, blob_id: str, payload: bytes) -> Path:
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
    blob_index = 1
    for shard, index in shard_specs:
        _link_blob(snapshot, shard, f"{blob_index:064x}", f"weight-{blob_index}".encode())
        blob_index += 1
        index_payload = json.dumps({"weight_map": {"weight": Path(shard).name}}).encode()
        _link_blob(snapshot, index, f"{blob_index:040x}", index_payload)
        blob_index += 1
    for relative, payload in (
        ("modular_model_index.json", b"{}"),
        ("scheduler/scheduler_config.json", b"{}"),
        ("tokenizer/tokenizer.json", b'{"model":"test"}'),
    ):
        _link_blob(snapshot, relative, f"{blob_index:040x}", payload)
        blob_index += 1
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
    with pytest.raises(ValueError, match="all four"):
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


def test_snapshot_inventory_change_invalidates_existing_receipt(tmp_path: Path) -> None:
    receipt, plans, snapshot, tokenizer = _receipt(tmp_path)
    shard = snapshot / "vae/diffusion_pytorch_model-00001-of-00001.safetensors"
    shard.unlink()
    _link_blob(snapshot, str(shard.relative_to(snapshot)), "f" * 64, b"different-weight")
    with pytest.raises(ValueError, match="checkpoint_snapshot"):
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
    receipt, _plans, _snapshot_path, _tokenizer = _receipt(tmp_path)
    config = {
        "model_type": "minimax_h3",
        "runtime_strategy": "diffusion_minimax_h3",
        "checkpoint_revision": CHECKPOINT_REVISION,
        "source_revision": SOURCE_REVISION,
        "builder_source_sha256": builder_source_sha256(),
        "checkpoint_inventory_sha256": receipt["checkpoint_snapshot"]["inventory_sha256"],
        "context_parallel_size": 4,
        "plan_sha256": {
            filename: receipt["components"][filename]["sha256"] for filename in PLAN_FILENAMES
        },
    }
    bundle = tmp_path / "model.trtfb"
    write_bundle(
        bundle,
        BundleInfo(model_id="MiniMaxAI/MiniMax-H3"),
        [BundleSection("config.json", json.dumps(config).encode())],
    )
    validate_native_bundle_config(bundle, source_revision=SOURCE_REVISION)

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
    artifact.write_bytes(b"new")
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
