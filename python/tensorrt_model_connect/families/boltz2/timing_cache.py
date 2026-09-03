# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent, profile-scoped TensorRT timing cache for Boltz-2 builds."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Iterator

from tensorrt_model_connect import trt_compat

from .provenance import PINNED_BOLTZ2


_ENABLED_ENV = "TRTMC_BOLTZ2_TIMING_CACHE"
_CACHE_DIR_ENV = "TRTMC_BOLTZ2_TIMING_CACHE_DIR"
_GENERIC_CACHE_PATH_ENV = "TRTMC_TRT_TIMING_CACHE_PATH"
_GENERIC_CACHE_DIR_ENV = "TRTMC_TRT_TIMING_CACHE_DIR"
_GRAPH_CACHE_REVISION = 1


@dataclass(frozen=True)
class Boltz2TimingCacheSelection:
    """Resolved timing-cache policy for one static Boltz-2 profile."""

    path: Path | None
    source: str
    warm: bool


def _enabled() -> bool:
    value = os.environ.get(_ENABLED_ENV, "1").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{_ENABLED_ENV} must be a boolean value, got {value!r}")


def _cache_root() -> Path:
    configured = os.environ.get(_CACHE_DIR_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "trtmc" / "boltz2" / "timing-cache"


def _sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "unknown"


def _gpu_target() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            major, minor = torch.cuda.get_device_capability(device)
            name = _sanitize(torch.cuda.get_device_name(device))
            return f"sm{major}{minor}-{name}"
    except (ImportError, RuntimeError):
        pass
    # TensorRT validates the device encoded in a cache payload. The fallback
    # only affects the filename on installations where PyTorch cannot report
    # the GPU during builder setup.
    return "smunknown"


def timing_cache_path(
    *,
    token_count: int,
    atom_count: int,
    msa_depth: int,
    precision: str,
) -> Path:
    """Return the target- and graph-specific persistent cache path."""

    version = _sanitize(trt_compat.tensorrt_version() or "unknown")
    optimization = _sanitize(os.environ.get("TRTMC_BUILDER_OPTIMIZATION_LEVEL", "family-default"))
    timing_iterations = _sanitize(os.environ.get("TRTMC_AVG_TIMING_ITERATIONS", "8"))
    max_tactics = _sanitize(os.environ.get("TRTMC_MAX_NUM_TACTICS", "default"))
    checkpoint = PINNED_BOLTZ2.checkpoint_revision[:12]
    filename = (
        f"boltz2-trt{version}-{_gpu_target()}-{_sanitize(precision)}"
        f"-t{token_count}-a{atom_count}-m{msa_depth}"
        f"-opt{optimization}-avg{timing_iterations}-tactics{max_tactics}"
        f"-graph{_GRAPH_CACHE_REVISION}-{checkpoint}.cache"
    )
    return _cache_root() / filename


@contextmanager
def use_boltz2_timing_cache(
    *,
    token_count: int,
    atom_count: int,
    msa_depth: int,
    precision: str,
) -> Iterator[Boltz2TimingCacheSelection]:
    """Enable one shared cache for every plan in a static Boltz-2 bundle.

    Explicit repository-wide timing-cache settings take precedence. Otherwise
    Boltz-2 selects a persistent cache whose filename captures the target,
    static shape, checkpoint, graph revision, and builder-search controls.
    """

    generic_path = os.environ.get(_GENERIC_CACHE_PATH_ENV, "").strip()
    generic_dir = os.environ.get(_GENERIC_CACHE_DIR_ENV, "").strip()
    if generic_path or generic_dir:
        path = Path(generic_path).expanduser() if generic_path else None
        yield Boltz2TimingCacheSelection(
            path=path,
            source="generic",
            warm=bool(path and path.is_file() and path.stat().st_size),
        )
        return

    if not _enabled():
        yield Boltz2TimingCacheSelection(path=None, source="disabled", warm=False)
        return

    path = timing_cache_path(
        token_count=token_count,
        atom_count=atom_count,
        msa_depth=msa_depth,
        precision=precision,
    )
    previous = os.environ.get(_GENERIC_CACHE_PATH_ENV)
    os.environ[_GENERIC_CACHE_PATH_ENV] = str(path)
    try:
        yield Boltz2TimingCacheSelection(
            path=path,
            source="boltz2",
            warm=path.is_file() and path.stat().st_size > 0,
        )
    finally:
        if previous is None:
            os.environ.pop(_GENERIC_CACHE_PATH_ENV, None)
        else:
            os.environ[_GENERIC_CACHE_PATH_ENV] = previous
