# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from tensorrt_model_connect.families.bark import timing_cache


class _FakeCache:
    def __init__(self, payload: bytes):
        self.payload = bytearray(payload)

    def serialize(self) -> bytes:
        return bytes(self.payload)


class _FakeConfig:
    def __init__(self):
        self.flags: list[str] = []
        self.cache: _FakeCache | None = None
        self.builder_optimization_level = -1
        self.max_num_tactics = -1
        self.avg_timing_iterations = -1
        self.created_payloads: list[bytes] = []

    def set_flag(self, flag: str) -> None:
        self.flags.append(flag)

    def create_timing_cache(self, payload: bytes) -> _FakeCache:
        self.created_payloads.append(bytes(payload))
        return _FakeCache(payload)

    def set_timing_cache(self, cache: _FakeCache, ignore_mismatch: bool) -> bool:
        assert ignore_mismatch is False
        self.cache = cache
        return True

    def get_timing_cache(self) -> _FakeCache | None:
        return self.cache


class _FakeBuilder:
    def __init__(self, appended_payload: bytes = b""):
        self.appended_payload = appended_payload
        self.calls = 0

    def build_serialized_network(self, network, config):
        self.calls += 1
        assert network == "network"
        if self.appended_payload:
            assert config.cache is not None
            config.cache.payload.extend(self.appended_payload)
        return b"plan"


@pytest.fixture
def fake_builder_flags(monkeypatch):
    flags = SimpleNamespace(
        EDITABLE_TIMING_CACHE="editable",
        DISABLE_COMPILATION_CACHE="no-compilation-cache",
        ERROR_ON_TIMING_CACHE_MISS="error-on-miss",
    )
    monkeypatch.setattr(
        timing_cache.trt_compat,
        "get_trt",
        lambda: SimpleNamespace(BuilderFlag=flags),
    )


def test_timing_cache_off_delegates_to_normal_builder(monkeypatch) -> None:
    monkeypatch.delenv("TRTMC_BARK_TIMING_CACHE_MODE", raising=False)
    builder = _FakeBuilder()

    plan = timing_cache.build_bark_serialized_network(builder, "network", _FakeConfig())

    assert plan == b"plan"
    assert builder.calls == 1


def test_record_mode_accumulates_an_editable_cache(
    monkeypatch, tmp_path, fake_builder_flags
) -> None:
    cache_path = tmp_path / "bark.cache"
    monkeypatch.setenv("TRTMC_BARK_TIMING_CACHE_MODE", "record")
    monkeypatch.setenv("TRTMC_BARK_TIMING_CACHE_PATH", str(cache_path))
    monkeypatch.setenv("TRTMC_BUILDER_OPTIMIZATION_LEVEL", "3")
    monkeypatch.setenv("TRTMC_MAX_NUM_TACTICS", "7")
    monkeypatch.setenv("TRTMC_AVG_TIMING_ITERATIONS", "5")

    first_config = _FakeConfig()
    plan = timing_cache.build_bark_serialized_network(
        _FakeBuilder(b"semantic"), "network", first_config
    )
    second_config = _FakeConfig()
    timing_cache.build_bark_serialized_network(_FakeBuilder(b"+coarse"), "network", second_config)

    assert plan == b"plan"
    assert first_config.created_payloads == [b""]
    assert second_config.created_payloads == [b"semantic"]
    assert cache_path.read_bytes() == b"semantic+coarse"
    assert first_config.flags == ["editable", "no-compilation-cache"]
    assert first_config.builder_optimization_level == 3
    assert first_config.max_num_tactics == 7
    assert first_config.avg_timing_iterations == 5


@pytest.mark.parametrize("mutate_cache", [False, True])
def test_verified_mode_is_strict_and_rejects_tactic_updates(
    monkeypatch, tmp_path, fake_builder_flags, mutate_cache
) -> None:
    cache_path = tmp_path / "bark.cache"
    payload = b"verified-tactics"
    cache_path.write_bytes(payload)
    monkeypatch.setenv("TRTMC_BARK_TIMING_CACHE_MODE", "verified")
    monkeypatch.setenv("TRTMC_BARK_TIMING_CACHE_PATH", str(cache_path))
    monkeypatch.setenv("TRTMC_BARK_TIMING_CACHE_SHA256", hashlib.sha256(payload).hexdigest())
    config = _FakeConfig()
    builder = _FakeBuilder(b"-new-tactic" if mutate_cache else b"")

    if mutate_cache:
        with pytest.raises(RuntimeError, match="timing-cache miss or tactic update"):
            timing_cache.build_bark_serialized_network(builder, "network", config)
    else:
        plan = timing_cache.build_bark_serialized_network(builder, "network", config)
        assert plan == b"plan"

    assert config.created_payloads == [payload]
    assert config.flags == [
        "editable",
        "no-compilation-cache",
        "error-on-miss",
    ]
    assert cache_path.read_bytes() == payload


def test_verified_mode_rejects_wrong_digest_before_tensorrt_attach(monkeypatch, tmp_path) -> None:
    cache_path = tmp_path / "bark.cache"
    cache_path.write_bytes(b"unexpected")
    monkeypatch.setenv("TRTMC_BARK_TIMING_CACHE_MODE", "verified")
    monkeypatch.setenv("TRTMC_BARK_TIMING_CACHE_PATH", str(cache_path))
    monkeypatch.setenv("TRTMC_BARK_TIMING_CACHE_SHA256", "0" * 64)
    config = _FakeConfig()

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        timing_cache.build_bark_serialized_network(_FakeBuilder(), "network", config)

    assert config.created_payloads == []
