# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose the official HF checkpoint with its pinned open-source model code."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import sys
import tarfile
import tempfile
import types
import urllib.request
from pathlib import Path


_SOURCE_REVISION = "a290ba04c1b3ad1ec41a33974a157b2917b624d4"
_SOURCE_SHA256 = "c11d945db0fb765c0bdf355311f986ab914ad2619133de3a26f84a3ecbddf6c9"
_SOURCE_URL = f"https://github.com/NVlabs/Fast-FoundationStereo/archive/{_SOURCE_REVISION}.tar.gz"
_NESTED_CHECKPOINT = Path("weights/23-36-37/model_best_bp2_serialize.pth")
_FLAT_CHECKPOINT = Path("model_best_bp2_serialize.pth")


def configure_official_model_args(
    model,
    *,
    max_disparity: int,
    valid_iters: int,
) -> None:
    """Apply the complete inference contract omitted by older checkpoints."""
    model.args.max_disp = max_disparity
    model.args.valid_iters = valid_iters
    model.args.normalize = True


def install_official_io_import_shims() -> None:
    """Satisfy unused official visualization imports during model loading."""
    if "imageio" not in sys.modules and importlib.util.find_spec("imageio") is None:
        imageio = types.ModuleType("imageio")
        imageio.__spec__ = importlib.util.spec_from_loader(
            "imageio", loader=None, is_package=True
        )
        imageio.__path__ = []
        sys.modules["imageio"] = imageio
    if "cv2" not in sys.modules and importlib.util.find_spec("cv2") is None:
        cv2 = types.ModuleType("cv2")
        cv2.__spec__ = importlib.util.spec_from_loader("cv2", loader=None)
        cv2.COLORMAP_TURBO = 20
        sys.modules["cv2"] = cv2


def _complete_source(root: Path) -> bool:
    return all(
        (root / relative).is_file()
        for relative in (
            "Utils.py",
            "core/foundation_stereo.py",
            "core/submodule.py",
            "core/geometry.py",
            "core/extractor.py",
            "core/update.py",
        )
    )


def _cache_root() -> Path:
    configured = os.environ.get("TRTMC_FAST_FOUNDATION_STEREO_CACHE")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache/tensorrt_model_connect/fast_foundation_stereo"


def _verified_source(*, local_files_only: bool = False) -> Path:
    override = os.environ.get("TRTMC_FAST_FOUNDATION_STEREO_SOURCE_DIR")
    if override:
        source = Path(override).expanduser().resolve()
        if not _complete_source(source):
            raise FileNotFoundError(
                "TRTMC_FAST_FOUNDATION_STEREO_SOURCE_DIR does not contain the "
                "official Fast-FoundationStereo source tree"
            )
        return source

    destination = _cache_root() / f"source-{_SOURCE_REVISION}"
    if _complete_source(destination):
        return destination
    if local_files_only:
        raise FileNotFoundError(
            "Pinned Fast-FoundationStereo source is not present in the local cache; "
            "prepare the model once without local-files-only"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="fast-foundation-stereo-", dir=destination.parent
    ) as temporary:
        temporary_path = Path(temporary)
        archive = temporary_path / "source.tar.gz"
        urllib.request.urlretrieve(_SOURCE_URL, archive)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != _SOURCE_SHA256:
            raise RuntimeError(
                "Fast-FoundationStereo source archive checksum mismatch: "
                f"expected {_SOURCE_SHA256}, got {digest}"
            )
        extracted = temporary_path / "extracted"
        extracted.mkdir()
        prefix = f"Fast-FoundationStereo-{_SOURCE_REVISION}/"
        with tarfile.open(archive, "r:gz") as source_archive:
            for member in source_archive.getmembers():
                if not member.name.startswith(prefix):
                    continue
                relative = member.name[len(prefix) :]
                relative_path = Path(relative)
                if relative_path.is_absolute() or ".." in relative_path.parts:
                    continue
                selected = relative in {
                    "Utils.py",
                    "LICENSE.txt",
                } or relative.startswith("core/")
                if not selected or member.issym() or member.islnk():
                    continue
                destination_path = extracted / relative_path
                if member.isdir():
                    destination_path.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    continue
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                source_file = source_archive.extractfile(member)
                if source_file is None:
                    raise RuntimeError(f"Failed to extract {relative} from source archive")
                with source_file, destination_path.open("wb") as output_file:
                    shutil.copyfileobj(source_file, output_file)
        if not _complete_source(extracted):
            raise RuntimeError("Pinned Fast-FoundationStereo archive is missing model source")
        if destination.exists():
            shutil.rmtree(destination)
        extracted.rename(destination)
    return destination


def _link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(source)


def resolve_model_dir(
    model_dir: str | Path,
    *,
    local_files_only: bool = False,
) -> Path | None:
    """Stage the flat NVIDIA HF checkpoint beside pinned official source code."""
    root = Path(model_dir).resolve()
    if _complete_source(root) and (root / _NESTED_CHECKPOINT).is_file():
        return None
    checkpoint = root / _FLAT_CHECKPOINT
    if not checkpoint.is_file():
        return None

    source = _verified_source(local_files_only=local_files_only)
    identity = hashlib.sha256(str(root).encode()).hexdigest()[:16]
    staged = _cache_root() / "staged" / identity
    staged.mkdir(parents=True, exist_ok=True)
    _link(source / "core", staged / "core")
    _link(source / "Utils.py", staged / "Utils.py")
    if (source / "LICENSE.txt").is_file():
        _link(source / "LICENSE.txt", staged / "LICENSE.txt")
    _link(checkpoint, staged / _NESTED_CHECKPOINT)
    config = root / "cfg.yaml"
    if config.is_file():
        _link(config, staged / "cfg.yaml")
    return staged
