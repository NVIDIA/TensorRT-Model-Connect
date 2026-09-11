# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare reusable native requests for the bounded Boltz-2 profile."""

from __future__ import annotations

import json
import os
import struct
import tempfile
from pathlib import Path
from typing import Any, Final

import numpy as np
import yaml
from tensorrt_model_connect.build import content_cache_key

from .checkpoint import validate_artifact, validate_structure_checkpoint
from .contracts import INITIAL_BF16_PROFILE, parse_request_yaml, validate_a3m
from .feature_bundle import profile_feature_shapes, serialize_features, structure_metadata_json
from .model_config import CHECKPOINT, MOLS, MOLS_ARCHIVE, resolve_package_root
from .provenance import PINNED_BOLTZ2
from .random_samples import serialize_profile_random_samples


MAGIC: Final = b"B2RQ"
VERSION: Final = 2


def _seed() -> None:
    import torch

    seed = PINNED_BOLTZ2.reference_configuration.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_profile_features(processed_dir: Path, mol_dir: Path) -> dict[str, Any]:
    """Load one processed request padded to the reusable TensorRT profile."""

    import torch
    from boltz.data.feature.featurizerv2 import Boltz2Featurizer
    from boltz.data.module.inferencev2 import (
        Boltz2InferenceDataModule,
        PredictionDataset,
        collate,
    )
    from boltz.data.types import Manifest

    profile = INITIAL_BF16_PROFILE

    class ProfileFeaturizer(Boltz2Featurizer):
        def process(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            kwargs["max_atoms"] = profile.max_padded_atoms
            kwargs["max_tokens"] = profile.max_tokens
            return super().process(*args, **kwargs)

    manifest = Manifest.load(processed_dir / "manifest.json")
    if len(manifest.records) != 1:
        raise ValueError("Boltz-2 request preparation requires exactly one processed record")
    dataset = PredictionDataset(
        manifest=manifest,
        target_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        mol_dir=mol_dir,
        constraints_dir=processed_dir / "constraints",
        template_dir=processed_dir / "templates",
        extra_mols_dir=processed_dir / "mols",
    )
    dataset.featurizer = ProfileFeaturizer()
    _seed()
    features = collate([dataset[0]])
    expected = profile_feature_shapes(
        profile.max_tokens,
        profile.max_padded_atoms,
        profile.max_msa_depth,
    )
    for name, shape in expected.items():
        actual = tuple(int(dimension) for dimension in features[name].shape)
        if actual != shape:
            raise ValueError(
                f"Boltz-2 prepared feature {name!r} has shape {actual}, expected {shape}"
            )
    data_module = Boltz2InferenceDataModule(
        manifest=manifest,
        target_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        mol_dir=mol_dir,
        num_workers=0,
        constraints_dir=processed_dir / "constraints",
        template_dir=processed_dir / "templates",
        extra_mols_dir=processed_dir / "mols",
    )
    return data_module.transfer_batch_to_device(features, torch.device("cuda"), 0)


def serialize_prepared_request(
    request: bytes,
    features: bytes,
    random_samples: bytes,
    structure_metadata: bytes,
) -> bytes:
    """Serialize the four request-owned sections consumed by the native runtime."""

    sections = (request, features, random_samples, structure_metadata)
    if any(not section for section in sections):
        raise ValueError("Boltz-2 prepared-request sections must be nonempty")
    header = MAGIC + struct.pack("<I4Q", VERSION, *(len(section) for section in sections))
    return header + b"".join(sections)


def _request_inputs(request_path: Path) -> tuple[bytes, int, tuple[tuple[Path, bytes], ...]]:
    request_bytes = request_path.read_bytes()
    try:
        request_text = request_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Boltz-2 request YAML must be UTF-8") from error
    request = parse_request_yaml(request_text)
    request_root = request_path.parent.resolve(strict=True)
    msa_inputs: list[tuple[Path, bytes]] = []
    for sequence in request.sequences:
        msa_path = (request_root / sequence.msa_path).resolve(strict=True)
        try:
            msa_path.relative_to(request_root)
        except ValueError as error:
            raise ValueError("Boltz-2 A3M path must remain inside the request root") from error
        if not msa_path.is_file():
            raise ValueError(f"Boltz-2 A3M path is not a file: {msa_path}")
        msa_bytes = msa_path.read_bytes()
        try:
            msa_text = msa_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("Boltz-2 A3M must be UTF-8") from error
        validate_a3m(msa_text, expected_query=sequence.sequence)
        msa_inputs.append((msa_path, msa_bytes))
    return request_bytes, request.token_count, tuple(msa_inputs)


def _cache_key(request: bytes, msa_inputs: tuple[tuple[Path, bytes], ...]) -> str:
    profile = INITIAL_BF16_PROFILE
    identity = json.dumps(
        {
            "format": VERSION,
            "source_revision": PINNED_BOLTZ2.source_revision,
            "checkpoint_revision": PINNED_BOLTZ2.checkpoint_revision,
            "tokens": profile.max_tokens,
            "atoms": profile.max_padded_atoms,
            "msa_depth": profile.max_msa_depth,
        },
        sort_keys=True,
    ).encode("utf-8")
    msa_payloads = tuple(payload for _, payload in msa_inputs)
    return content_cache_key("boltz2-prepared-request-v2", identity, request, *msa_payloads)


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _stage_request(
    path: Path, request: bytes, msa_inputs: tuple[tuple[Path, bytes], ...]
) -> None:
    document = yaml.safe_load(request.decode("utf-8"))
    if len(document["sequences"]) != len(msa_inputs):
        raise ValueError("Boltz-2 request and MSA inputs are inconsistent")
    for entry, (msa_path, _) in zip(document["sequences"], msa_inputs, strict=True):
        entry["protein"]["msa"] = str(msa_path)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def prepare_structure_request(
    model_dir: str | Path,
    request_path: str | Path,
    output_path: str | Path,
    *,
    cache_dir: str | Path | None = None,
) -> dict[str, object]:
    """Prepare one raw YAML/A3M request without rebuilding TensorRT plans."""

    root = resolve_package_root(model_dir)
    if root is None:
        raise ValueError(f"unsupported Boltz-2 package: {model_dir}")
    checkpoint = root / CHECKPOINT
    validate_structure_checkpoint(checkpoint)
    validate_artifact(root / MOLS_ARCHIVE, PINNED_BOLTZ2.molecular_archive)
    request_path = Path(request_path).resolve(strict=True)
    output_path = Path(output_path)
    request_bytes, request_token_count, msa_inputs = _request_inputs(request_path)
    key = _cache_key(request_bytes, msa_inputs)
    cache_root = (
        Path(cache_dir)
        if cache_dir is not None
        else Path.home() / ".cache" / "tensorrt-model-connect" / "boltz2"
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_root = cache_root.resolve(strict=True)
    entry = cache_root / key[:2] / key
    if entry.is_symlink():
        raise ValueError(f"Boltz-2 cache entry must not be a symlink: {entry}")
    cached_request = entry / "request.b2rq"
    if cached_request.is_symlink():
        raise ValueError(f"Boltz-2 cached request must not be a symlink: {cached_request}")
    cache_hit = cached_request.is_file()
    if cache_hit:
        payload = cached_request.read_bytes()
        if not payload.startswith(MAGIC + struct.pack("<I", VERSION)):
            raise ValueError(f"invalid cached Boltz-2 prepared request: {cached_request}")
    else:
        from boltz.main import process_inputs

        work = entry / "work"
        if work.is_symlink():
            raise ValueError(f"Boltz-2 cache work directory must not be a symlink: {work}")
        staged_request = work / request_path.name
        work.mkdir(parents=True, exist_ok=True)
        if staged_request.is_symlink():
            raise ValueError(f"Boltz-2 staged request must not be a symlink: {staged_request}")
        _stage_request(staged_request, request_bytes, msa_inputs)
        process_inputs(
            data=[staged_request],
            out_dir=work,
            ccd_path=root / MOLS_ARCHIVE,
            mol_dir=root / MOLS,
            msa_server_url="https://api.colabfold.com",
            msa_pairing_strategy="greedy",
            max_msa_seqs=INITIAL_BF16_PROFILE.max_msa_depth,
            use_msa_server=False,
            boltz2=True,
            preprocessing_threads=1,
        )
        processed = work / "processed"
        structure = processed / "structures" / f"{staged_request.stem}.npz"
        if not structure.is_file():
            raise RuntimeError("Boltz-2 preprocessing did not produce the requested structure")
        features = load_profile_features(processed, root / MOLS)
        active_tokens = int(features["token_pad_mask"].sum().item())
        active_atoms = int(features["atom_pad_mask"].sum().item())
        with np.load(structure, allow_pickle=False) as archive:
            structure_atom_count = int(archive["atoms"].shape[0])
        if active_tokens != request_token_count or active_atoms != structure_atom_count:
            raise ValueError(
                "Boltz-2 preprocessing cropped or changed the request outside the "
                "tokens_117_atoms_928 profile"
            )
        random_samples = serialize_profile_random_samples(
            atom_count=INITIAL_BF16_PROFILE.max_padded_atoms,
        )
        payload = serialize_prepared_request(
            request_bytes,
            serialize_features(features),
            random_samples,
            structure_metadata_json(structure),
        )
        _write_atomic(cached_request, payload)
    _write_atomic(output_path, payload)
    return {
        "family": "boltz2",
        "prepared_request": str(output_path),
        "cache_key": key,
        "cache_hit": cache_hit,
        "profile": "tokens_117_atoms_928_msa_1",
    }
