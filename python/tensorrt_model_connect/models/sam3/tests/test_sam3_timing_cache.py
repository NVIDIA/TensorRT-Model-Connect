# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace
from typing import Iterator

import pytest

from tensorrt_model_connect.models.sam3 import timing_cache
from tensorrt_model_connect.models.sam3 import timing_cache_data
from tensorrt_model_connect.models.sam3 import vision_encoder_builder
from tensorrt_model_connect.models.sam3.tests import record_timing_cache


_L4_RUNTIME = timing_cache._RuntimeMetadata(
    gpu_name="NVIDIA L4",
    gpu_architecture="sm89",
    tensorrt_version="10.15.1.29",
    cuda_runtime_version=13010,
    cuda_driver_version=13020,
)
_RTX_5090_RUNTIME = timing_cache._RuntimeMetadata(
    gpu_name="NVIDIA GeForce RTX 5090",
    gpu_architecture="sm120",
    tensorrt_version="10.15.1.29",
    cuda_runtime_version=13010,
    cuda_driver_version=13020,
)
_L4_OTHER_DRIVER_RUNTIME = timing_cache._RuntimeMetadata(
    gpu_name="NVIDIA L4",
    gpu_architecture="sm89",
    tensorrt_version="10.15.1.29",
    cuda_runtime_version=13010,
    cuda_driver_version=13030,
)
_SYNTHETIC_GRAPH_PROFILE = {"precision": "fp32", "shape": [1, 32]}
_PACKAGED_GRAPH_PROFILES = {
    "core": {
        "decoder_hidden_act": "relu",
        "detr_decoder_heads": 8,
        "detr_decoder_intermediate_size": 2048,
        "detr_decoder_layers": 6,
        "detr_encoder_heads": 8,
        "detr_encoder_intermediate_size": 2048,
        "detr_encoder_layers": 6,
        "encoder_hidden_act": "relu",
        "fpn_hidden_size": 256,
        "fpn_shapes": ((288, 288), (144, 144), (72, 72)),
        "geometry_encoder_heads": 8,
        "geometry_encoder_hidden_act": "relu",
        "geometry_encoder_intermediate_size": 2048,
        "geometry_encoder_layer_norm_eps": 1e-5,
        "geometry_encoder_layers": 3,
        "hidden_size": 256,
        "layer_norm_eps": 1e-5,
        "mask_num_heads": 8,
        "mask_num_upsampling_stages": 3,
        "network_definition": "strongly_typed",
        "num_queries": 200,
        "precision": "fp32",
        "text_seq_len": 32,
        "workspace_bytes": 6 << 30,
    },
    "text-encoder": {
        "eps": 1e-5,
        "has_text_projection_bias": True,
        "hidden_act": "gelu",
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "max_seq_len": 32,
        "network_definition": "strongly_typed",
        "num_heads": 16,
        "num_layers": 24,
        "precision": "fp32",
        "projected_size": 256,
        "vocab_size": 49408,
        "workspace_bytes": 4 << 30,
    },
}
_VISION_GRAPH_PROFILE = vision_encoder_builder._timing_cache_graph_profile(
    image_size=1008,
    patch_size=14,
    pretrain_image_size=336,
    hidden_size=1024,
    intermediate_size=4736,
    num_layers=32,
    num_heads=16,
    window_size=24,
    global_attn_indexes=[7, 15, 23, 31],
    fpn_hidden_size=256,
    rope_theta=10000.0,
    eps=1e-6,
    precision="fp32",
    hidden_act="gelu",
    has_tracker_neck=True,
)
_GRAPH_PROFILES = {**_PACKAGED_GRAPH_PROFILES, "vision-encoder": _VISION_GRAPH_PROFILE}
_TACTICS = {
    "0x01": SimpleNamespace(tacticHash=101, timingMSec=1.0),
    "0x02": SimpleNamespace(tacticHash=202, timingMSec=2.0),
}


class _FakeCache:
    def __init__(self):
        self.tactics = {
            key: SimpleNamespace(tacticHash=value.tacticHash, timingMSec=value.timingMSec)
            for key, value in _TACTICS.items()
        }
        self.updated_keys: list[str] = []

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
    def __init__(self, *, accept_cache: bool = True):
        self.accept_cache = accept_cache
        self.flags: list[str] = []
        self.created_payloads: list[bytes] = []
        self.ignore_mismatches: list[bool] = []
        self.cache: _FakeCache | None = None

    def set_flag(self, flag: str) -> None:
        self.flags.append(flag)

    def create_timing_cache(self, payload: bytes) -> _FakeCache:
        self.created_payloads.append(bytes(payload))
        return _FakeCache()

    def set_timing_cache(self, cache: _FakeCache, ignore_mismatch: bool) -> bool:
        self.cache = cache
        self.ignore_mismatches.append(ignore_mismatch)
        return self.accept_cache

    def get_timing_cache(self) -> _FakeCache | None:
        return self.cache


class _FakeBuilder:
    def __init__(self, *, mutate_tactic: bool = False):
        self.calls = 0
        self.mutate_tactic = mutate_tactic

    def build_serialized_network(self, network, config):
        self.calls += 1
        assert network == "network"
        if self.mutate_tactic:
            assert config.cache is not None
            config.cache.tactics["0x01"] = SimpleNamespace(tacticHash=999, timingMSec=1.0)
        return b"plan"


@pytest.fixture
def fake_builder_flags(monkeypatch: pytest.MonkeyPatch) -> None:
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


def _graph_contracts(
    engine_kinds: frozenset[str] | None = None,
) -> dict[str, object]:
    if engine_kinds is None:
        contract_id = timing_cache._BUILTIN_TARGETS[_L4_RUNTIME.signature]
        engine_kinds = timing_cache._BUILTIN_ENGINE_KINDS[contract_id]
    return {
        "algorithm": timing_cache._GRAPH_CONTRACT_ALGORITHM,
        "engines": {
            engine_kind: timing_cache._graph_contract_fingerprint(
                engine_kind, _GRAPH_PROFILES[engine_kind]
            )
            for engine_kind in engine_kinds
        },
    }


@contextmanager
def _policy_scope(policy: timing_cache.Sam3TimingCachePolicy) -> Iterator[None]:
    """Exercise the owner-private policy state without restoring its retired API."""

    policy.validate()
    token = timing_cache._POLICY.set(policy)
    try:
        yield
    finally:
        timing_cache._POLICY.reset(token)


def _qualify_current_builtin_graphs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make a synthetic packaged contract exact for the current test graph."""

    contract_id = timing_cache._BUILTIN_TARGETS[_L4_RUNTIME.signature]
    monkeypatch.setattr(
        timing_cache,
        "_BUILTIN_GRAPH_FINGERPRINTS",
        {contract_id: dict(_graph_contracts()["engines"])},
    )


def _write_contract(
    root: Path,
    *,
    runtime: timing_cache._RuntimeMetadata = _L4_RUNTIME,
    payload: bytes = b"verified-cache",
    tactic_sha256: str | None = None,
    engine_kinds: frozenset[str] = frozenset({"core", "text-encoder"}),
) -> Path:
    directory = root / timing_cache._BUILTIN_TARGETS[_L4_RUNTIME.signature]
    directory.mkdir(parents=True)
    tactics = {key: value.tacticHash for key, value in _TACTICS.items()}
    entries = {}
    for engine_kind in engine_kinds:
        entry = {
            "file": f"{engine_kind}.cache",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "tactic_count": len(tactics),
            "tactic_sha256": tactic_sha256 or timing_cache._tactic_sha256(tactics),
        }
        (directory / entry["file"]).write_bytes(payload)
        entries[engine_kind] = entry
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "engines": entries,
                "graph_contract": _graph_contracts(engine_kinds),
                "schema_version": timing_cache._SCHEMA_VERSION,
                "target": runtime.as_dict(),
            }
        ),
        encoding="utf-8",
    )
    return directory


def _install_builtin_contract(
    monkeypatch: pytest.MonkeyPatch,
    *,
    payload: bytes = b"verified-cache",
    encoded_payload: str | None = None,
) -> None:
    tactics = {key: value.tacticHash for key, value in _TACTICS.items()}
    entry = {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "tactic_count": len(tactics),
        "tactic_sha256": timing_cache._tactic_sha256(tactics),
    }
    encoded = encoded_payload or base64.b64encode(payload).decode("ascii")
    contract_id = timing_cache._BUILTIN_TARGETS[_L4_RUNTIME.signature]
    monkeypatch.setattr(
        timing_cache_data,
        "CONTRACTS",
        {
            contract_id: {
                "manifest": {
                    "engines": {"core": dict(entry), "text-encoder": dict(entry)},
                    "graph_contract": _graph_contracts(),
                    "schema_version": timing_cache._SCHEMA_VERSION,
                    "target": _L4_RUNTIME.as_dict(),
                },
                "payloads": {"core": encoded, "text-encoder": encoded},
            }
        },
    )
    _qualify_current_builtin_graphs(monkeypatch)


def _build(
    builder,
    config,
    *,
    engine_kind: str = "text-encoder",
    graph_profile=None,
):
    return timing_cache.build_sam3_serialized_network(
        builder,
        "network",
        config,
        engine_kind=engine_kind,
        graph_profile=graph_profile or _GRAPH_PROFILES[engine_kind],
    )


def test_default_nonmatching_target_builds_normally(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(timing_cache, "_runtime_metadata", lambda: _RTX_5090_RUNTIME)
    builder = _FakeBuilder()
    config = _FakeConfig()

    plan = _build(builder, config)

    assert plan == b"plan"
    assert builder.calls == 1
    assert config.flags == []
    assert config.created_payloads == []


def test_default_same_l4_with_different_driver_builds_normally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        timing_cache,
        "_runtime_metadata",
        lambda: _L4_OTHER_DRIVER_RUNTIME,
    )
    builder = _FakeBuilder()
    config = _FakeConfig()

    plan = _build(builder, config)

    assert plan == b"plan"
    assert builder.calls == 1
    assert config.flags == []
    assert config.created_payloads == []


def test_runtime_target_includes_cuda_driver_api_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda_module = ModuleType("cuda")
    bindings_module = ModuleType("cuda.bindings")
    runtime_module = ModuleType("cuda.bindings.runtime")
    runtime_module.cudaError_t = SimpleNamespace(cudaSuccess=0)
    runtime_module.cudaGetDevice = lambda: (0, 0)
    runtime_module.cudaGetDeviceProperties = lambda device: (
        0,
        SimpleNamespace(name=b"NVIDIA L4\0", major=8, minor=9),
    )
    runtime_module.cudaRuntimeGetVersion = lambda: (0, 13010)
    runtime_module.cudaDriverGetVersion = lambda: (0, 13020)
    bindings_module.runtime = runtime_module
    cuda_module.bindings = bindings_module
    monkeypatch.setitem(sys.modules, "cuda", cuda_module)
    monkeypatch.setitem(sys.modules, "cuda.bindings", bindings_module)
    monkeypatch.setitem(sys.modules, "cuda.bindings.runtime", runtime_module)
    monkeypatch.setattr(timing_cache.trt_compat, "tensorrt_version", lambda: "10.15.1.29")

    runtime = timing_cache._runtime_metadata()

    assert runtime == _L4_RUNTIME
    assert runtime.signature == (
        "NVIDIA L4",
        "sm89",
        "10.15.1.29",
        13010,
        13020,
    )
    assert runtime.as_dict()["cuda_driver_version"] == 13020


def test_default_l4_noncanonical_graph_builds_normally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(timing_cache, "_runtime_metadata", lambda: _L4_RUNTIME)
    monkeypatch.setattr(timing_cache_data, "CONTRACTS", {})
    builder = _FakeBuilder()
    config = _FakeConfig()

    plan = _build(
        builder,
        config,
        graph_profile={
            **_PACKAGED_GRAPH_PROFILES["text-encoder"],
            "max_seq_len": 64,
        },
    )

    assert plan == b"plan"
    assert builder.calls == 1
    assert config.flags == []
    assert config.created_payloads == []


def test_default_l4_unqualified_vision_builds_normally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(timing_cache, "_runtime_metadata", lambda: _L4_RUNTIME)
    monkeypatch.setattr(timing_cache_data, "CONTRACTS", {})
    builder = _FakeBuilder()
    config = _FakeConfig()

    plan = _build(builder, config, engine_kind="vision-encoder")

    assert plan == b"plan"
    assert builder.calls == 1
    assert config.flags == []
    assert config.created_payloads == []


def test_default_l4_qualified_vision_noncanonical_graph_builds_normally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract_id = timing_cache._BUILTIN_TARGETS[_L4_RUNTIME.signature]
    monkeypatch.setattr(timing_cache, "_runtime_metadata", lambda: _L4_RUNTIME)
    monkeypatch.setattr(
        timing_cache,
        "_BUILTIN_ENGINE_KINDS",
        {contract_id: frozenset({"vision-encoder"})},
    )
    monkeypatch.setattr(
        timing_cache,
        "_BUILTIN_GRAPH_FINGERPRINTS",
        {
            contract_id: {
                "vision-encoder": timing_cache._graph_contract_fingerprint(
                    "vision-encoder", _VISION_GRAPH_PROFILE
                )
            }
        },
    )
    monkeypatch.setattr(timing_cache_data, "CONTRACTS", {})
    builder = _FakeBuilder()
    config = _FakeConfig()

    plan = _build(
        builder,
        config,
        engine_kind="vision-encoder",
        graph_profile={**_VISION_GRAPH_PROFILE, "image_size": 896},
    )

    assert plan == b"plan"
    assert builder.calls == 1
    assert config.flags == []
    assert config.created_payloads == []


def test_default_matching_target_current_unqualified_graph_builds_normally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(timing_cache, "_runtime_metadata", lambda: _L4_RUNTIME)
    monkeypatch.setattr(timing_cache_data, "CONTRACTS", {})
    builder = _FakeBuilder()
    config = _FakeConfig()

    plan = _build(builder, config)

    assert plan == b"plan"
    assert builder.calls == 1
    assert config.flags == []
    assert config.created_payloads == []


def test_default_matching_target_qualified_graph_missing_assets_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(timing_cache, "_runtime_metadata", lambda: _L4_RUNTIME)
    monkeypatch.setattr(timing_cache_data, "CONTRACTS", {})
    _qualify_current_builtin_graphs(monkeypatch)

    with pytest.raises(RuntimeError, match="packaged.*contract is missing"):
        _build(_FakeBuilder(), _FakeConfig())


def test_default_matching_target_replays_exact_tactics(
    monkeypatch: pytest.MonkeyPatch,
    fake_builder_flags,
) -> None:
    _install_builtin_contract(monkeypatch)
    monkeypatch.setattr(timing_cache, "_runtime_metadata", lambda: _L4_RUNTIME)
    builder = _FakeBuilder()
    config = _FakeConfig()

    plan = _build(builder, config)

    assert plan == b"plan"
    assert builder.calls == 1
    assert config.flags == ["editable", "no-compilation-cache", "error-on-miss"]
    assert config.created_payloads == [b"verified-cache"]
    assert config.ignore_mismatches == [False]
    assert config.cache is not None
    assert config.cache.updated_keys == ["0x01", "0x02"]


def test_verified_cache_rejects_digest_before_tensorrt_attach(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _write_contract(tmp_path)
    (directory / "text-encoder.cache").write_bytes(b"corrupt")
    monkeypatch.setattr(timing_cache, "_runtime_metadata", lambda: _L4_RUNTIME)
    config = _FakeConfig()
    policy = timing_cache.Sam3TimingCachePolicy("verified", directory)

    with _policy_scope(policy):
        with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
            _build(_FakeBuilder(), config)

    assert config.created_payloads == []


def test_verified_cache_rejects_tactic_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_builder_flags,
) -> None:
    directory = _write_contract(tmp_path, tactic_sha256="0" * 64)
    monkeypatch.setattr(timing_cache, "_runtime_metadata", lambda: _L4_RUNTIME)
    config = _FakeConfig()
    policy = timing_cache.Sam3TimingCachePolicy("verified", directory)

    with _policy_scope(policy):
        with pytest.raises(RuntimeError, match="tactic fingerprint mismatch"):
            _build(_FakeBuilder(), config)

    assert config.ignore_mismatches == []


def test_verified_cache_preserves_tensorrt_native_compatibility_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_builder_flags,
) -> None:
    directory = _write_contract(tmp_path)
    monkeypatch.setattr(timing_cache, "_runtime_metadata", lambda: _L4_RUNTIME)
    config = _FakeConfig(accept_cache=False)
    policy = timing_cache.Sam3TimingCachePolicy("verified", directory)

    with _policy_scope(policy):
        with pytest.raises(RuntimeError, match="TensorRT rejected verified"):
            _build(_FakeBuilder(), config)

    assert config.ignore_mismatches == [False]


def test_verified_cache_rejects_post_build_tactic_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_builder_flags,
) -> None:
    directory = _write_contract(tmp_path)
    monkeypatch.setattr(timing_cache, "_runtime_metadata", lambda: _L4_RUNTIME)
    policy = timing_cache.Sam3TimingCachePolicy("verified", directory)

    with _policy_scope(policy):
        with pytest.raises(RuntimeError, match=r"added=0, removed=0, changed=1"):
            _build(_FakeBuilder(mutate_tactic=True), _FakeConfig())


def test_verified_cache_rejects_graph_profile_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _write_contract(tmp_path)
    monkeypatch.setattr(timing_cache, "_runtime_metadata", lambda: _L4_RUNTIME)
    policy = timing_cache.Sam3TimingCachePolicy("verified", directory)

    with _policy_scope(policy):
        with pytest.raises(RuntimeError, match="graph contract mismatch"):
            timing_cache.build_sam3_serialized_network(
                _FakeBuilder(),
                "network",
                _FakeConfig(),
                engine_kind="text-encoder",
                graph_profile={
                    **_PACKAGED_GRAPH_PROFILES["text-encoder"],
                    "max_seq_len": 64,
                },
            )


def test_verified_isolated_vision_inventory_replays_strictly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_builder_flags,
) -> None:
    directory = _write_contract(
        tmp_path,
        engine_kinds=frozenset({"vision-encoder"}),
    )
    monkeypatch.setattr(timing_cache, "_runtime_metadata", lambda: _L4_RUNTIME)
    builder = _FakeBuilder()
    config = _FakeConfig()
    policy = timing_cache.Sam3TimingCachePolicy("verified", directory)

    with _policy_scope(policy):
        plan = _build(builder, config, engine_kind="vision-encoder")

    assert plan == b"plan"
    assert config.flags == ["editable", "no-compilation-cache", "error-on-miss"]
    assert config.created_payloads == [b"verified-cache"]
    assert config.ignore_mismatches == [False]


def test_graph_contract_is_deterministic_and_engine_specific() -> None:
    first = timing_cache._graph_contract_fingerprint("text-encoder", _SYNTHETIC_GRAPH_PROFILE)
    second = timing_cache._graph_contract_fingerprint(
        "text-encoder", {"shape": [1, 32], "precision": "fp32"}
    )
    core = timing_cache._graph_contract_fingerprint("core", _SYNTHETIC_GRAPH_PROFILE)

    assert first == second
    assert first != core
    assert timing_cache._SHA256.fullmatch(first)


def test_text_graph_contract_includes_projection_bias_presence() -> None:
    with_bias = _PACKAGED_GRAPH_PROFILES["text-encoder"]
    without_bias = {**with_bias, "has_text_projection_bias": False}

    assert timing_cache._graph_contract_fingerprint(
        "text-encoder", with_bias
    ) != timing_cache._graph_contract_fingerprint("text-encoder", without_bias)


def test_vision_graph_contract_includes_conditional_tracker_neck() -> None:
    without_tracker_neck = {**_VISION_GRAPH_PROFILE, "has_tracker_neck": False}

    assert timing_cache._graph_contract_fingerprint(
        "vision-encoder", _VISION_GRAPH_PROFILE
    ) != timing_cache._graph_contract_fingerprint("vision-encoder", without_tracker_neck)


def test_explicit_verified_policy_rejects_target_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _write_contract(tmp_path)
    monkeypatch.setattr(timing_cache, "_runtime_metadata", lambda: _RTX_5090_RUNTIME)
    policy = timing_cache.Sam3TimingCachePolicy("verified", directory)

    with _policy_scope(policy):
        with pytest.raises(RuntimeError, match="target mismatch"):
            _build(_FakeBuilder(), _FakeConfig(), engine_kind="core")


def test_context_override_disables_and_restores_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_calls = 0

    def runtime_metadata() -> timing_cache._RuntimeMetadata:
        nonlocal runtime_calls
        runtime_calls += 1
        return _L4_RUNTIME

    monkeypatch.setattr(timing_cache, "_runtime_metadata", runtime_metadata)
    monkeypatch.setattr(timing_cache_data, "CONTRACTS", {})
    builder = _FakeBuilder()
    policy = timing_cache.Sam3TimingCachePolicy("off")

    with _policy_scope(policy):
        assert _build(builder, _FakeConfig(), engine_kind="core") == b"plan"
    assert runtime_calls == 0

    assert _build(builder, _FakeConfig(), engine_kind="core") == b"plan"
    assert runtime_calls == 1


def test_policy_validation_and_engine_inventory_are_fail_closed() -> None:
    assert not hasattr(timing_cache, "use_sam3_timing_cache")
    with pytest.raises(ValueError, match="requires a directory"):
        timing_cache.Sam3TimingCachePolicy("verified").validate()
    with pytest.raises(ValueError, match="unknown.*mode"):
        timing_cache.Sam3TimingCachePolicy("unexpected").validate()  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unsupported.*engine kind"):
        timing_cache.build_sam3_serialized_network(
            _FakeBuilder(),
            "network",
            _FakeConfig(),
            engine_kind="tracker-step",
            graph_profile=_SYNTHETIC_GRAPH_PROFILE,
        )


def test_sam3_timing_cache_does_not_read_process_environment() -> None:
    source = inspect.getsource(timing_cache)

    assert "os.environ" not in source
    assert "getenv(" not in source


def test_recorder_generates_importable_sam3_owned_python_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(timing_cache, "_runtime_metadata", lambda: _L4_RUNTIME)
    cache_directory = tmp_path / "timing_cache"
    cache_directory.mkdir()
    recorder = record_timing_cache._Recorder(
        cache_directory,
        engine_kinds=frozenset({"core", "text-encoder"}),
    )
    tactics = {key: value.tacticHash for key, value in _TACTICS.items()}
    for engine_kind, payload in {
        "core": b"core-cache",
        "text-encoder": b"text-cache",
    }.items():
        file_name = f"{engine_kind}.cache"
        (cache_directory / file_name).write_bytes(payload)
        recorder.entries[engine_kind] = {
            "file": file_name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "tactic_count": len(tactics),
            "tactic_sha256": timing_cache._tactic_sha256(tactics),
        }
        recorder.graph_contracts[engine_kind] = timing_cache._graph_contract_fingerprint(
            engine_kind, _PACKAGED_GRAPH_PROFILES[engine_kind]
        )
    output = tmp_path / "timing_cache_data.py"

    recorder.write_python_module(output)

    namespace: dict[str, object] = {}
    exec(compile(output.read_text(encoding="utf-8"), str(output), "exec"), namespace)
    contracts = namespace["CONTRACTS"]
    contract_id = timing_cache._BUILTIN_TARGETS[_L4_RUNTIME.signature]
    assert isinstance(contracts, dict)
    assert contracts[contract_id]["manifest"]["graph_contract"] == _graph_contracts()
    assert base64.b64decode(contracts[contract_id]["payloads"]["core"]) == b"core-cache"


def test_recorder_writes_isolated_vision_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(timing_cache, "_runtime_metadata", lambda: _L4_RUNTIME)
    cache_directory = tmp_path / "timing_cache"
    recorder = record_timing_cache._Recorder(
        cache_directory,
        engine_kinds=frozenset({"vision-encoder"}),
    )
    tactics = {key: value.tacticHash for key, value in _TACTICS.items()}
    payload = b"vision-cache"
    cache_directory.mkdir()
    (cache_directory / "vision-encoder.cache").write_bytes(payload)
    recorder.entries["vision-encoder"] = {
        "file": "vision-encoder.cache",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "tactic_count": len(tactics),
        "tactic_sha256": timing_cache._tactic_sha256(tactics),
    }
    recorder.graph_contracts["vision-encoder"] = timing_cache._graph_contract_fingerprint(
        "vision-encoder", _VISION_GRAPH_PROFILE
    )

    recorder.write_manifest()
    manifest = json.loads((cache_directory / "manifest.json").read_text(encoding="utf-8"))

    assert set(manifest["engines"]) == {"vision-encoder"}
    assert set(manifest["graph_contract"]["engines"]) == {"vision-encoder"}


def test_model_snapshot_is_complete_deterministic_and_content_bound(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "z.bin").write_bytes(b"z")
    nested = model_dir / "nested"
    nested.mkdir()
    (nested / "a.json").write_bytes(b"a")
    (model_dir / "linked.bin").symlink_to(model_dir / "z.bin")

    first = record_timing_cache._directory_snapshot(model_dir)
    second = record_timing_cache._directory_snapshot(model_dir)

    assert first == second
    assert [record["path"] for record in first["records"]] == [
        "linked.bin",
        "nested/a.json",
        "z.bin",
    ]
    assert first["file_count"] == 3
    assert first["total_size_bytes"] == 3
    canonical = json.dumps(first["records"], separators=(",", ":"), sort_keys=True).encode("utf-8")
    assert first["aggregate_sha256"] == hashlib.sha256(canonical).hexdigest()

    (model_dir / "z.bin").write_bytes(b"y")
    changed = record_timing_cache._directory_snapshot(model_dir)

    assert changed["aggregate_sha256"] != first["aggregate_sha256"]


def test_model_snapshot_rejects_incomplete_symlinks_and_nested_output(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}\n", encoding="utf-8")
    (model_dir / "broken").symlink_to(model_dir / "missing")

    with pytest.raises(RuntimeError, match="broken or directory symlink"):
        record_timing_cache._directory_snapshot(model_dir)
    with pytest.raises(ValueError, match="must not be inside"):
        record_timing_cache._ensure_disjoint_output(model_dir, model_dir / "candidate")

    (model_dir / "broken").unlink()
    target_directory = tmp_path / "target-directory"
    target_directory.mkdir()
    (model_dir / "directory-link").symlink_to(target_directory, target_is_directory=True)
    with pytest.raises(RuntimeError, match="broken or directory symlink"):
        record_timing_cache._directory_snapshot(model_dir)


def test_source_commit_is_best_effort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        record_timing_cache.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="a" * 40 + "\n"),
    )
    assert record_timing_cache._git_commit_best_effort(tmp_path) == "a" * 40

    def unavailable(*args, **kwargs):
        del args, kwargs
        raise OSError("git unavailable")

    monkeypatch.setattr(record_timing_cache.subprocess, "run", unavailable)
    assert record_timing_cache._git_commit_best_effort(tmp_path) is None


def test_timing_cache_receipt_rejects_declared_cache_hash_mismatch(tmp_path: Path) -> None:
    output_directory = tmp_path / "candidate"
    cache_directory = output_directory / "timing_cache"
    cache_directory.mkdir(parents=True)
    cache_path = cache_directory / "vision-encoder.cache"
    cache_path.write_bytes(b"actual-cache")
    python_data_path = output_directory / "timing_cache_data.py"
    python_data_path.write_text("CONTRACTS = {}\n", encoding="utf-8")
    entry = {
        "file": cache_path.name,
        "sha256": "0" * 64,
        "tactic_count": 2,
        "tactic_sha256": "1" * 64,
    }
    (cache_directory / "manifest.json").write_text(
        json.dumps({"engines": {"vision-encoder": entry}}),
        encoding="utf-8",
    )
    recorder = SimpleNamespace(
        directory=cache_directory,
        engine_kinds=frozenset({"vision-encoder"}),
        entries={"vision-encoder": entry},
    )

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        record_timing_cache._timing_cache_receipt(
            recorder,
            output_directory=output_directory,
            python_data_path=python_data_path,
        )


def test_recorder_cli_builds_only_the_selected_vision_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "model"
    output_directory = tmp_path / "candidate"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"sam3_video"}\n', encoding="utf-8")
    recorders = []

    class Recorder:
        def __init__(self, directory, *, engine_kinds, verbose):
            self.directory = directory
            self.engine_kinds = engine_kinds
            self.verbose = verbose
            self.recorded_kind = None
            payload = b"vision-cache"
            self.entries = {
                "vision-encoder": {
                    "file": "vision-encoder.cache",
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "tactic_count": 2,
                    "tactic_sha256": "1" * 64,
                }
            }
            recorders.append(self)

        def build(self, builder, network, config, *, engine_kind, graph_profile):
            del builder, network, config, graph_profile
            assert engine_kind in self.engine_kinds
            self.recorded_kind = engine_kind
            return b"vision-plan"

        def write_manifest(self):
            self.directory.mkdir(parents=True)
            (self.directory / "vision-encoder.cache").write_bytes(b"vision-cache")
            (self.directory / "manifest.json").write_text(
                json.dumps({"engines": self.entries}) + "\n",
                encoding="utf-8",
            )

        def write_python_module(self, path):
            path.write_text("CONTRACTS = {}\n", encoding="utf-8")

    class Plugin:
        name = "sam3"

        def matches(self, config):
            del config
            return True

        def build_vision_engine(self, model_path, config, weights, **kwargs):
            assert model_path == str(model_dir)
            assert config.raw["_sam3_config"]["vision_image_size"] == 1008
            assert weights == {}
            assert kwargs == {"precision": "fp32", "verbose": True}
            # Exercise the symbol imported by the real vision builder rather
            # than the timing_cache module attribute used by older tests.
            return vision_encoder_builder.build_sam3_serialized_network(
                None,
                None,
                None,
                engine_kind="vision-encoder",
                graph_profile=_VISION_GRAPH_PROFILE,
            )

    monkeypatch.setattr(record_timing_cache, "_Recorder", Recorder)
    monkeypatch.setattr(
        record_timing_cache.ModelConfig,
        "from_dir",
        lambda path: SimpleNamespace(raw={}),
    )
    monkeypatch.setattr(record_timing_cache, "sam3_model", Plugin())
    monkeypatch.setattr(record_timing_cache, "_git_commit_best_effort", lambda root: "a" * 40)
    monkeypatch.setattr(timing_cache, "_runtime_metadata", lambda: _L4_RUNTIME)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "record_timing_cache.py",
            str(model_dir),
            str(output_directory),
            "--engine",
            "vision-encoder",
            "--verbose",
        ],
    )

    record_timing_cache.main()

    assert len(recorders) == 1
    assert recorders[0].engine_kinds == frozenset({"vision-encoder"})
    assert recorders[0].recorded_kind == "vision-encoder"
    assert (output_directory / "vision_engine_plan.bin").read_bytes() == b"vision-plan"
    assert not (output_directory / "engine_plan.bin").exists()
    receipt = json.loads((output_directory / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["artifact_type"] == "sam3_timing_cache_candidate_receipt"
    assert receipt["schema_version"] == 1
    assert receipt["engine_selection"] == "vision-encoder"
    assert set(receipt["plans"]) == {"vision_engine_plan"}
    assert receipt["plans"]["vision_engine_plan"] == {
        "path": "vision_engine_plan.bin",
        "sha256": hashlib.sha256(b"vision-plan").hexdigest(),
        "size_bytes": len(b"vision-plan"),
    }
    assert receipt["model_snapshot"]["file_count"] == 1
    assert receipt["model_snapshot"]["records"][0]["path"] == "config.json"
    assert receipt["source"]["git_commit"] == "a" * 40
    assert receipt["source"]["file_count"] >= 3
    source_paths = {record["path"] for record in receipt["source"]["records"]}
    assert "python/tensorrt_model_connect/models/sam3/tests/record_timing_cache.py" in source_paths
    assert "python/tensorrt_model_connect/models/sam3/timing_cache.py" in source_paths
    assert "python/tensorrt_model_connect/models/sam3/vision_encoder_builder.py" in source_paths
    assert receipt["target"]["cuda_driver_version"] == 13020
    assert receipt["timing_cache"]["manifest"]["path"] == "timing_cache/manifest.json"
    assert receipt["timing_cache"]["caches"]["vision-encoder"]["path"] == (
        "timing_cache/vision-encoder.cache"
    )
    assert receipt["timing_cache"]["python_data"]["path"] == "timing_cache_data.py"


def test_packaged_cache_payloads_match_manifest_hashes() -> None:
    contract_id = timing_cache._BUILTIN_TARGETS[_L4_RUNTIME.signature]
    contract = timing_cache_data.CONTRACTS[contract_id]
    provenance_path = Path(__file__).with_name("data") / "sam3_l4_timing_cache_qualification.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

    assert contract["manifest"]["schema_version"] == timing_cache._SCHEMA_VERSION
    assert set(contract["manifest"]["engines"]) == timing_cache._BUILTIN_ENGINE_KINDS[contract_id]
    assert "vision-encoder" not in contract["manifest"]["engines"]
    assert provenance["artifact_type"] == "sam3_l4_timing_cache_qualification_provenance"
    assert len(provenance["qualified_source_base_commit"]) == 40
    assert provenance["qualified_target"] == contract["manifest"]["target"]
    assert provenance["qualified_graph_contract"] == contract["manifest"]["graph_contract"]
    current_graph_contract = {
        "algorithm": timing_cache._GRAPH_CONTRACT_ALGORITHM,
        "engines": {
            engine_kind: timing_cache._graph_contract_fingerprint(engine_kind, graph_profile)
            for engine_kind, graph_profile in _PACKAGED_GRAPH_PROFILES.items()
        },
    }
    assert contract["manifest"]["graph_contract"]["engines"] == dict(
        timing_cache._BUILTIN_GRAPH_FINGERPRINTS[contract_id]
    )
    assert current_graph_contract["engines"] != contract["manifest"]["graph_contract"]["engines"]
    for engine_kind in timing_cache._BUILTIN_ENGINE_KINDS[contract_id]:
        payload = base64.b64decode(contract["payloads"][engine_kind], validate=True)
        manifest_entry = contract["manifest"]["engines"][engine_kind]
        provenance_entry = provenance["raw_caches"][engine_kind]
        assert hashlib.sha256(payload).hexdigest() == manifest_entry["sha256"]
        assert provenance_entry == {
            "sha256": manifest_entry["sha256"],
            "size_bytes": len(payload),
            "tactic_count": manifest_entry["tactic_count"],
            "tactic_sha256": manifest_entry["tactic_sha256"],
        }

    rounds = provenance["rounds"]
    assert [item["round"] for item in rounds] == [1, 2, 3]
    assert len({item["tactic_evidence_sha256"] for item in rounds}) == 3
    assert len({item["tracking_artifact_sha256"] for item in rounds}) == 3
    assert len({item["accuracy_evidence_sha256"] for item in rounds}) == 3
    assert len({item["mask_manifest_sha256"] for item in rounds}) == 1
    assert len({item["text_plan_sha256"] for item in rounds}) == 1
    for item in rounds:
        assert item["accuracy_passed"] is True
        assert item["structural_passed"] is True
        assert item["full_propagation_global_iou"] >= 0.995
        assert item["full_propagation_macro_iou"] >= 0.995
        assert item["minimum_track_spatiotemporal_iou"] >= 0.95
        assert item["tracker_tail_global_iou"] >= 0.995
        assert item["tracker_tail_macro_iou"] >= 0.995
    prompt_smoke = provenance["additional_prompt_smoke"]
    assert prompt_smoke["passed"] is True
    assert set(prompt_smoke["positive_prompts"].values()) == {"passed"}
    assert prompt_smoke["expected_negative_controls"] == {
        "curb": "not_applicable",
        "stair": "not_applicable",
    }
    assert provenance["tactic_reproducibility"] == {
        "all_rounds_sensitive_core_good": 22,
        "all_rounds_sensitive_text_good": 24,
        "full_fc_record_count": 171,
        "full_fc_record_difference_count": 0,
        "tactic_like_token_count": 281,
        "tactic_like_token_difference_count": 0,
        "text_plan_byte_identical_across_rounds": True,
    }
