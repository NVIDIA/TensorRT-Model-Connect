# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_model_connect.families.elf_flow import timing_cache


def _write_contract(
    tmp_path: Path,
    *,
    tensorrt_version: str = "11.1.0.106",
) -> tuple[Path, Path]:
    cache_path = tmp_path / "elf.bin"
    metadata_path = tmp_path / "elf.json"
    payload = b"model-owned-cache"
    cache_path.write_bytes(payload)
    metadata_path.write_text(
        json.dumps(
            {
                "builder_optimization_level": 1,
                "compute_capability": "10.3",
                "gpu": "NVIDIA GB300",
                "path": cache_path.name,
                "schema_version": 1,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "tensorrt_version": tensorrt_version,
            }
        ),
        encoding="utf-8",
    )
    return cache_path, metadata_path


def _configure_env(
    monkeypatch: pytest.MonkeyPatch,
    cache_path: Path,
    metadata_path: Path,
    *,
    generate: bool = False,
) -> None:
    monkeypatch.setenv("TRTMC_ELF_TIMING_CACHE_PATH", str(cache_path))
    monkeypatch.setenv("TRTMC_ELF_TIMING_CACHE_METADATA_PATH", str(metadata_path))
    monkeypatch.setenv("TRTMC_ELF_TIMING_CACHE_GENERATE", "1" if generate else "0")
    monkeypatch.setenv("TRTMC_BUILDER_OPTIMIZATION_LEVEL", "1")
    monkeypatch.delenv("TRTMC_TRT_TIMING_CACHE_PATH", raising=False)
    monkeypatch.delenv("TRTMC_TRT_TIMING_CACHE_DIR", raising=False)
    monkeypatch.setattr(timing_cache.trt_compat, "tensorrt_version", lambda: "11.1.0.106")
    monkeypatch.setattr(
        timing_cache,
        "_runtime_gpu_metadata",
        lambda: ("NVIDIA GB300", "10.3"),
    )


def test_model_timing_cache_is_attached_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_path, metadata_path = _write_contract(tmp_path)
    _configure_env(monkeypatch, cache_path, metadata_path)
    calls: list[tuple] = []

    class Config:
        def create_timing_cache(self, payload):
            calls.append(("create", bytes(payload)))
            return "cache"

        def set_timing_cache(self, cache, ignore_mismatch):
            calls.append(("set", cache, ignore_mismatch))
            return True

        def set_flag(self, flag):
            calls.append(("flag", flag))

    trt = SimpleNamespace(BuilderFlag=SimpleNamespace(ERROR_ON_TIMING_CACHE_MISS="strict"))
    state = timing_cache.attach_model_timing_cache(Config(), trt)

    assert state is not None
    assert not state.generate
    assert calls == [
        ("create", b"model-owned-cache"),
        ("set", "cache", False),
        ("flag", "strict"),
    ]
    assert cache_path.read_bytes() == b"model-owned-cache"


def test_model_timing_cache_rejects_stale_tensorrt_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_path, metadata_path = _write_contract(
        tmp_path,
        tensorrt_version="11.0.0.114",
    )
    _configure_env(monkeypatch, cache_path, metadata_path)

    with pytest.raises(RuntimeError, match="tensorrt_version"):
        timing_cache.attach_model_timing_cache(object(), object())


def test_model_timing_cache_generation_writes_binary_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_path = tmp_path / "generated.bin"
    metadata_path = tmp_path / "generated.json"
    _configure_env(monkeypatch, cache_path, metadata_path, generate=True)
    calls: list[tuple] = []

    class Cache:
        def serialize(self):
            return b"generated-cache"

    class Config:
        cache = Cache()

        def create_timing_cache(self, payload):
            calls.append(("create", bytes(payload)))
            return self.cache

        def set_timing_cache(self, cache, ignore_mismatch):
            calls.append(("set", cache, ignore_mismatch))
            return True

        def get_timing_cache(self):
            return self.cache

    config = Config()
    state = timing_cache.attach_model_timing_cache(config, object())
    timing_cache.persist_generated_model_timing_cache(config, state)

    assert calls == [("create", b""), ("set", config.cache, False)]
    assert cache_path.read_bytes() == b"generated-cache"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["tensorrt_version"] == "11.1.0.106"
    assert metadata["compute_capability"] == "10.3"
    assert metadata["sha256"] == hashlib.sha256(b"generated-cache").hexdigest()
