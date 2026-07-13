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
        self.tactics = {"0x01": SimpleNamespace(tacticHash=101, timingMSec=1.0)}
        self.updated_keys: list[str] = []

    def serialize(self) -> bytes:
        return bytes(self.payload)

    def queryKeys(self) -> list[str]:
        return list(self.tactics)

    def query(self, key: str):
        return self.tactics[key]

    def update(self, key: str, value) -> bool:
        if key not in self.tactics:
            return False
        self.updated_keys.append(key)
        self.tactics[key] = value
        return True


class _FakeConfig:
    def __init__(self):
        self.flags: list[str] = []
        self.cache: _FakeCache | None = None
        self.builder_optimization_level = -1
        self.max_num_tactics = -1
        self.avg_timing_iterations = -1
        self.created_payloads: list[bytes] = []
        self.ignore_mismatches: list[bool] = []

    def set_flag(self, flag: str) -> None:
        self.flags.append(flag)

    def create_timing_cache(self, payload: bytes) -> _FakeCache:
        self.created_payloads.append(bytes(payload))
        return _FakeCache(payload)

    def set_timing_cache(self, cache: _FakeCache, ignore_mismatch: bool) -> bool:
        self.ignore_mismatches.append(ignore_mismatch)
        self.cache = cache
        return True

    def get_timing_cache(self) -> _FakeCache | None:
        return self.cache


class _FakeBuilder:
    def __init__(
        self,
        appended_payload: bytes = b"",
        tactic_hash: int | None = None,
        timing_msec: float | None = None,
        add_tactic: bool = False,
    ):
        self.appended_payload = appended_payload
        self.tactic_hash = tactic_hash
        self.timing_msec = timing_msec
        self.add_tactic = add_tactic
        self.calls = 0

    def build_serialized_network(self, network, config):
        self.calls += 1
        assert network == "network"
        if self.appended_payload:
            assert config.cache is not None
            config.cache.payload.extend(self.appended_payload)
        if self.tactic_hash is not None:
            assert config.cache is not None
            config.cache.tactics["0x01"] = SimpleNamespace(
                tacticHash=self.tactic_hash, timingMSec=2.0
            )
        if self.timing_msec is not None:
            assert config.cache is not None
            config.cache.tactics["0x01"] = SimpleNamespace(
                tacticHash=101, timingMSec=self.timing_msec
            )
        if self.add_tactic:
            assert config.cache is not None
            config.cache.tactics["0x02"] = SimpleNamespace(tacticHash=303, timingMSec=3.0)
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
    assert first_config.ignore_mismatches == [False]


@pytest.mark.parametrize("mutate_serialized_cache", [False, True])
def test_verified_mode_accepts_device_metadata_changes_when_tactics_are_unchanged(
    monkeypatch, tmp_path, fake_builder_flags, mutate_serialized_cache
) -> None:
    cache_path = tmp_path / "bark.cache"
    payload = b"verified-tactics"
    cache_path.write_bytes(payload)
    monkeypatch.setenv("TRTMC_BARK_TIMING_CACHE_MODE", "verified")
    monkeypatch.setenv("TRTMC_BARK_TIMING_CACHE_PATH", str(cache_path))
    monkeypatch.setenv("TRTMC_BARK_TIMING_CACHE_SHA256", hashlib.sha256(payload).hexdigest())
    config = _FakeConfig()
    builder = _FakeBuilder(b"-device-metadata" if mutate_serialized_cache else b"")

    plan = timing_cache.build_bark_serialized_network(builder, "network", config)

    assert plan == b"plan"

    assert config.created_payloads == [payload]
    assert config.flags == [
        "editable",
        "no-compilation-cache",
        "error-on-miss",
    ]
    assert config.ignore_mismatches == [True]
    assert config.cache is not None
    assert config.cache.updated_keys == ["0x01"]
    assert cache_path.read_bytes() == payload


def test_verified_mode_rejects_tactic_updates(monkeypatch, tmp_path, fake_builder_flags) -> None:
    cache_path = tmp_path / "bark.cache"
    payload = b"verified-tactics"
    cache_path.write_bytes(payload)
    monkeypatch.setenv("TRTMC_BARK_TIMING_CACHE_MODE", "verified")
    monkeypatch.setenv("TRTMC_BARK_TIMING_CACHE_PATH", str(cache_path))
    monkeypatch.setenv("TRTMC_BARK_TIMING_CACHE_SHA256", hashlib.sha256(payload).hexdigest())

    with pytest.raises(RuntimeError, match="tactic selection changed"):
        timing_cache.build_bark_serialized_network(
            _FakeBuilder(tactic_hash=202), "network", _FakeConfig()
        )


def test_verified_mode_accepts_updated_timing_for_the_same_tactic(
    monkeypatch, tmp_path, fake_builder_flags
) -> None:
    cache_path = tmp_path / "bark.cache"
    payload = b"verified-tactics"
    cache_path.write_bytes(payload)
    monkeypatch.setenv("TRTMC_BARK_TIMING_CACHE_MODE", "verified")
    monkeypatch.setenv("TRTMC_BARK_TIMING_CACHE_PATH", str(cache_path))
    monkeypatch.setenv("TRTMC_BARK_TIMING_CACHE_SHA256", hashlib.sha256(payload).hexdigest())

    plan = timing_cache.build_bark_serialized_network(
        _FakeBuilder(timing_msec=9.5), "network", _FakeConfig()
    )

    assert plan == b"plan"


def test_verified_mode_rejects_new_timing_cache_keys(
    monkeypatch, tmp_path, fake_builder_flags
) -> None:
    cache_path = tmp_path / "bark.cache"
    payload = b"verified-tactics"
    cache_path.write_bytes(payload)
    monkeypatch.setenv("TRTMC_BARK_TIMING_CACHE_MODE", "verified")
    monkeypatch.setenv("TRTMC_BARK_TIMING_CACHE_PATH", str(cache_path))
    monkeypatch.setenv("TRTMC_BARK_TIMING_CACHE_SHA256", hashlib.sha256(payload).hexdigest())

    with pytest.raises(RuntimeError, match=r"added=1, removed=0, changed=0"):
        timing_cache.build_bark_serialized_network(
            _FakeBuilder(add_tactic=True), "network", _FakeConfig()
        )


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
