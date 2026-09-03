# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare direct Boltz-2 YAML inputs for a reusable TensorRT profile."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .checkpoint import validate_artifact
from .contracts import parse_request_yaml, validate_a3m
from .feature_bundle import (
    FEATURE_NAMES,
    profile_feature_shapes,
    structure_metadata_json,
)
from .prepared_request import (
    deserialize_prepared_request,
    serialize_prepared_request,
)
from .provenance import PINNED_BOLTZ2
from .random_samples import serialize_predict_random_samples


_PINNED_PREPROCESS_PACKAGES = {
    "boltz": "2.2.1",
    "numpy": "1.26.4",
    "pytorch-lightning": "2.5.0",
    "rdkit": "2026.3.5",
    "gemmi": "0.6.5",
    "numba": "0.61.0",
    "torch": "2.12.0+cu130",
}
_PREPROCESS_CACHE_VERSION = 3


def _ensure_pinned_profile() -> None:
    try:
        ready = all(
            version(package) == expected
            for package, expected in _PINNED_PREPROCESS_PACKAGES.items()
        )
    except PackageNotFoundError:
        ready = False
    if ready:
        return
    if os.environ.get("_TRTMC_BOLTZ2_PREPROCESS_PROFILE") == "1":
        raise RuntimeError("the pinned Boltz-2 preprocessing profile is unavailable")

    from tensorrt_model_connect.python_profiles import resolve_profile_python

    python = resolve_profile_python("boltz2_build", sys.executable)
    environment = os.environ.copy()
    environment["_TRTMC_BOLTZ2_PREPROCESS_PROFILE"] = "1"
    os.execve(
        python,
        [python, "-m", "tensorrt_model_connect.families.boltz2.runtime_preprocess", *sys.argv[1:]],
        environment,
    )


def _cache_root() -> Path:
    configured = os.environ.get("TRTMC_BOLTZ2_PREPROCESS_CACHE_DIR", "").strip()
    if configured:
        return Path(configured)
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "trtmc" / "boltz2" / "prepared-requests"


def _resolve_mols_dir(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    configured = os.environ.get("TRTMC_BOLTZ2_MOLS_DIR", "").strip()
    if configured:
        candidates.append(Path(configured))
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    cache = Path(xdg) if xdg else Path.home() / ".cache"
    candidates.append(
        cache
        / "trtmc"
        / "artifacts"
        / f"boltz-2-{PINNED_BOLTZ2.checkpoint_revision}"
        / "mols"
    )
    for candidate in candidates:
        if candidate.is_dir() and (candidate.parent / "mols.tar").is_file():
            return candidate.resolve(strict=True)
    rendered = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "the pinned Boltz-2 molecule dictionary is unavailable; set "
        f"TRTMC_BOLTZ2_MOLS_DIR or --mols-dir (checked: {rendered})"
    )


def _resolve_checkpoint(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    configured = os.environ.get("TRTMC_BOLTZ2_CHECKPOINT", "").strip()
    if configured:
        candidates.append(Path(configured))
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    cache = Path(xdg) if xdg else Path.home() / ".cache"
    candidates.append(
        cache
        / "trtmc"
        / "artifacts"
        / f"boltz-2-{PINNED_BOLTZ2.checkpoint_revision}"
        / PINNED_BOLTZ2.structure_checkpoint.filename
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve(strict=True)
    rendered = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "the pinned Boltz-2 checkpoint is unavailable; set "
        f"TRTMC_BOLTZ2_CHECKPOINT or --checkpoint (checked: {rendered})"
    )


def _validate_cached_artifact(path: Path, artifact, cache_root: Path, label: str) -> None:
    stat = path.stat()
    receipt = cache_root / f"{label}-{artifact.sha256}.json"
    identity = {
        "path": str(path.resolve(strict=True)),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "inode": stat.st_ino,
    }
    if receipt.is_file():
        try:
            if json.loads(receipt.read_text(encoding="utf-8")) == identity:
                return
        except (OSError, ValueError):
            pass
    validate_artifact(path, artifact)
    cache_root.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(identity, sort_keys=True) + "\n", encoding="utf-8")


def _semantic_cache_key(
    request_path: Path,
    *,
    token_count: int,
    atom_count: int,
    msa_depth: int,
) -> tuple[str, bytes]:
    request = request_path.read_bytes()
    parsed = parse_request_yaml(request.decode("utf-8"))
    sequence = parsed.sequences[0]
    msa_path = request_path.parent / sequence.msa_path
    msa = msa_path.read_bytes()
    validate_a3m(msa.decode("utf-8"), expected_query=sequence.sequence)
    document = json.loads(json.dumps(__import__("yaml").safe_load(request), sort_keys=True))
    digest = hashlib.sha256()
    digest.update(json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(b"\0")
    digest.update(msa)
    digest.update(
        (
            f"\0schema={_PREPROCESS_CACHE_VERSION};"
            f"boltz={PINNED_BOLTZ2.source_revision};"
            f"t={token_count};a={atom_count};m={msa_depth}"
        ).encode()
    )
    return digest.hexdigest(), request


def _seed_preprocessing() -> None:
    import numpy as np
    import torch

    seed = PINNED_BOLTZ2.reference_configuration.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _require_profile_shapes(
    features, *, token_count: int, atom_count: int, msa_depth: int
) -> None:
    expected = profile_feature_shapes(token_count, atom_count, msa_depth)
    for name in FEATURE_NAMES:
        actual = tuple(int(dimension) for dimension in features[name].shape)
        if actual != expected[name]:
            raise ValueError(
                f"Boltz-2 request feature {name!r} has shape {actual}; "
                f"the loaded bundle requires {expected[name]}"
            )


def _load_cpu_features(processed_dir: Path, mols_dir: Path):
    from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
    from boltz.data.types import Manifest

    _seed_preprocessing()
    manifest = Manifest.load(processed_dir / "manifest.json")
    if len(manifest.records) != 1:
        raise RuntimeError(
            f"Boltz-2 preprocessing produced {len(manifest.records)} records, expected one"
        )
    module = Boltz2InferenceDataModule(
        manifest=manifest,
        target_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        mol_dir=mols_dir,
        num_workers=0,
        constraints_dir=processed_dir / "constraints",
        template_dir=processed_dir / "templates",
        extra_mols_dir=processed_dir / "mols",
    )
    return module, next(iter(module.predict_dataloader()))


def _process_yaml(request_path: Path, output_root: Path, mols_dir: Path) -> None:
    from boltz.main import process_inputs

    process_path = request_path
    if request_path.suffix.lower() == ".json":
        parsed = parse_request_yaml(request_path.read_text(encoding="utf-8"))
        staged_root = output_root / "request-input"
        staged_root.mkdir(parents=True, exist_ok=True)
        process_path = staged_root / f"{request_path.stem}.yaml"
        process_path.write_bytes(request_path.read_bytes())
        for sequence in parsed.sequences:
            if sequence.msa_path is None:
                continue
            source = request_path.parent / sequence.msa_path
            destination = staged_root / sequence.msa_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    # Upstream v2.2.1 resolves MSA paths against the process working
    # directory instead of the YAML location. Scope that behavior to this
    # worker so callers can use ordinary YAML-relative paths from anywhere.
    previous = Path.cwd()
    try:
        os.chdir(process_path.parent)
        process_inputs(
            data=[process_path],
            out_dir=output_root,
            ccd_path=mols_dir / "unused.pkl",
            mol_dir=mols_dir,
            msa_server_url="https://api.colabfold.com",
            msa_pairing_strategy="greedy",
            max_msa_seqs=PINNED_BOLTZ2.reference_configuration.max_msa_sequences,
            use_msa_server=False,
            boltz2=True,
            preprocessing_threads=1,
        )
    finally:
        os.chdir(previous)


def prepare_request(
    request_path: Path,
    *,
    output_path: Path,
    mols_dir: Path | None = None,
    checkpoint_path: Path | None = None,
    token_count: int = 117,
    atom_count: int = 928,
    msa_depth: int = 1,
) -> None:
    """Prepare and cache one direct YAML request for a compiled profile."""

    request_path = request_path.resolve(strict=True)
    cache_root = _cache_root()
    key, request = _semantic_cache_key(
        request_path,
        token_count=token_count,
        atom_count=atom_count,
        msa_depth=msa_depth,
    )
    entry = cache_root / key
    cached = entry / "request.b2rq"
    entry.mkdir(parents=True, exist_ok=True)
    with (entry / ".lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if cached.is_file():
            prepared = deserialize_prepared_request(cached.read_bytes())
            payload = serialize_prepared_request(
                request,
                prepared.features,
                prepared.random_samples,
                prepared.structure_metadata,
            )
        else:
            resolved_mols = _resolve_mols_dir(mols_dir)
            _validate_cached_artifact(
                resolved_mols.parent / "mols.tar",
                PINNED_BOLTZ2.molecular_archive,
                cache_root,
                "mols",
            )
            resolved_checkpoint = _resolve_checkpoint(checkpoint_path)
            _validate_cached_artifact(
                resolved_checkpoint,
                PINNED_BOLTZ2.structure_checkpoint,
                cache_root,
                "checkpoint",
            )
            _process_yaml(request_path, entry, resolved_mols)
            processed = entry / "processed"
            module, features = _load_cpu_features(processed, resolved_mols)
            _require_profile_shapes(
                features,
                token_count=token_count,
                atom_count=atom_count,
                msa_depth=msa_depth,
            )
            import torch

            cuda_features = module.transfer_batch_to_device(
                features, torch.device("cuda"), 0
            )
            request_atom_count = int(features["ref_pos"].shape[1])
            random_samples = serialize_predict_random_samples(
                resolved_checkpoint,
                cuda_features,
                atom_count=request_atom_count,
            )
            structure = processed / "structures" / f"{request_path.stem}.npz"
            payload = serialize_prepared_request(
                request,
                features,
                random_samples,
                structure_metadata_json(structure),
            )
            temporary = cached.with_suffix(".tmp")
            temporary.write_bytes(payload)
            os.replace(temporary, cached)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output_path.parent, delete=False) as stream:
        temporary_output = Path(stream.name)
        stream.write(payload)
    os.replace(temporary_output, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mols-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--token-count", type=int, required=True)
    parser.add_argument("--atom-count", type=int, required=True)
    parser.add_argument("--msa-depth", type=int, required=True)
    args = parser.parse_args()
    _ensure_pinned_profile()
    prepare_request(
        args.input,
        output_path=args.output,
        mols_dir=args.mols_dir,
        checkpoint_path=args.checkpoint,
        token_count=args.token_count,
        atom_count=args.atom_count,
        msa_depth=args.msa_depth,
    )


if __name__ == "__main__":
    main()
