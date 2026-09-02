# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tomllib

import pytest


SCRIPT = Path(__file__).with_name("native_reference.py")
SPEC = importlib.util.spec_from_file_location("minimax_h3_native_reference", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_cache_threshold_cli_args_are_model_namespaced() -> None:
    assert MODULE.cache_threshold_cli_args(None) == []
    assert MODULE.cache_threshold_cli_args(0.05) == [
        "--set",
        "minimax_h3.first_block_cache_threshold=0.05",
    ]


def test_parse_retained_frame_indices_accepts_vbench_siglip_subset() -> None:
    assert MODULE.parse_retained_frame_indices("0,18,35,53,70,88,105,123") == (
        0,
        18,
        35,
        53,
        70,
        88,
        105,
        123,
    )
    assert MODULE.parse_retained_frame_indices("") == ()


@pytest.mark.parametrize("value", ["24,0", "0,0", "-1", "124", "zero"])
def test_parse_retained_frame_indices_rejects_invalid_subsets(value: str) -> None:
    with pytest.raises(ValueError, match="retained frame indices"):
        MODULE.parse_retained_frame_indices(value)


def test_canonical_build_selects_first_block_cache() -> None:
    model_config = tomllib.loads(SCRIPT.with_name("MODEL.toml").read_text())
    assert {
        "flag": "--set",
        "value": "minimax_h3.first_block_cache=true",
    } in model_config["e2e_defaults"]["diffusion_media_generation"]["build_cli_args"]


def test_resolve_trt_backend_dso_matches_runtime_candidate_order(tmp_path: Path) -> None:
    executable = tmp_path / "trtf"
    executable.write_bytes(b"binary")
    unversioned = tmp_path / "libtrtmc_backend_trt.so"
    versioned = tmp_path / "libtrtmc_backend_trt_11_2.so"
    unversioned.write_bytes(b"unversioned")
    versioned.write_bytes(b"versioned")
    config = {"engine_backend": "trt", "trt_abi": "11.2"}

    assert MODULE.resolve_trt_backend_dso(executable, config) == versioned.resolve()
    versioned.unlink()
    assert MODULE.resolve_trt_backend_dso(executable, config) == unversioned.resolve()


def test_resolve_trt_backend_dso_fails_closed(tmp_path: Path) -> None:
    executable = tmp_path / "trtf"
    executable.write_bytes(b"binary")

    with pytest.raises(ValueError, match="engine_backend=trt"):
        MODULE.resolve_trt_backend_dso(executable, {"engine_backend": "rtx", "trt_abi": "11.2"})
    with pytest.raises(ValueError, match="invalid TensorRT ABI"):
        MODULE.resolve_trt_backend_dso(executable, {"engine_backend": "trt", "trt_abi": "latest"})
    with pytest.raises(FileNotFoundError, match="adjacent TensorRT backend"):
        MODULE.resolve_trt_backend_dso(executable, {"engine_backend": "trt", "trt_abi": "11.2"})


def test_file_page_eviction_is_portable_when_posix_fadvise_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(MODULE.os, "posix_fadvise", raising=False)

    def fail_open(*_args, **_kwargs) -> int:
        raise AssertionError("file should not be opened without posix_fadvise")

    monkeypatch.setattr(MODULE.os, "open", fail_open)

    assert MODULE.evict_file_pages(Path("model.bundle")) == {
        "supported": False,
        "attempted": False,
        "succeeded": False,
    }


def test_file_page_eviction_advises_exact_file_and_closes_descriptor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bundle = tmp_path / "model.bundle"
    bundle.write_bytes(b"bundle")
    calls: list[tuple[int, int, int, int]] = []

    def record_fadvise(descriptor: int, offset: int, length: int, advice: int) -> None:
        calls.append((descriptor, offset, length, advice))

    monkeypatch.setattr(MODULE.os, "posix_fadvise", record_fadvise, raising=False)
    monkeypatch.setattr(MODULE.os, "POSIX_FADV_DONTNEED", 4, raising=False)

    result = MODULE.evict_file_pages(bundle)

    assert result == {"supported": True, "attempted": True, "succeeded": True}
    assert len(calls) == 1
    descriptor, offset, length, advice = calls[0]
    assert (offset, length, advice) == (0, 0, 4)
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_file_page_eviction_reports_failure_and_closes_descriptor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bundle = tmp_path / "model.bundle"
    bundle.write_bytes(b"bundle")
    descriptors: list[int] = []
    real_open = MODULE.os.open

    def record_open(path: Path, flags: int) -> int:
        descriptor = real_open(path, flags)
        descriptors.append(descriptor)
        return descriptor

    def fail_fadvise(*_args) -> None:
        raise OSError("advice rejected")

    monkeypatch.setattr(MODULE.os, "open", record_open)
    monkeypatch.setattr(MODULE.os, "posix_fadvise", fail_fadvise, raising=False)
    monkeypatch.setattr(MODULE.os, "POSIX_FADV_DONTNEED", 4, raising=False)

    result = MODULE.evict_file_pages(bundle)

    assert result == {
        "supported": True,
        "attempted": True,
        "succeeded": False,
        "error": "OSError: advice rejected",
    }
    with pytest.raises(OSError):
        os.fstat(descriptors[0])
