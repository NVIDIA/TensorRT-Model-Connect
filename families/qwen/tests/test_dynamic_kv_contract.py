# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for the family-owned Qwen runtime-sized KV route."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("tensorrt")

from tensorrt_model_connect.build import BuildRequest  # noqa: E402

from .. import model as qwen_model  # noqa: E402
from .. import standard_decoder_builder  # noqa: E402
from ..config import ModelConfig  # noqa: E402
from ..dual_profile_decoder_builder import _resolve_prefill_lengths  # noqa: E402
from .test_e2e import _assert_runtime_sized_kv_receipt  # noqa: E402


class _Writer:
    def __init__(self) -> None:
        self.header: dict[str, str] = {}
        self.bytes: dict[str, bytes] = {}
        self.json: dict[str, dict] = {}

    def set_header(self, **values: str) -> None:
        self.header = values

    def add_bytes(self, name: str, value: bytes) -> None:
        self.bytes[name] = value

    def add_json(self, name: str, value: dict) -> None:
        self.json[name] = value


def _request(tmp_path: Path, **updates) -> BuildRequest:
    values = {
        "model_dir": tmp_path,
        "output_path": tmp_path / "model.bundle",
        "family": "qwen",
        "task": "text_generation",
        "precision": "fp16",
        "max_sequence_length": 64,
        "dynamic_kv_cache": True,
    }
    values.update(updates)
    return BuildRequest(**values)


def _config(model_type: str) -> ModelConfig:
    return ModelConfig.create_tiny(
        model_type,
        architectures=[f"{model_type.title()}ForCausalLM"],
        hidden_act="silu",
    )


def test_qwen2_dynamic_kv_builds_one_dual_profile_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config("qwen2")
    calls: list[dict] = []
    monkeypatch.setattr(qwen_model.ModelConfig, "from_dir", lambda _path: config)
    monkeypatch.setattr(qwen_model._QwenModel, "load_weights", lambda *_args, **_kwargs: {})

    def build_engine(_self, observed_config, _weights, max_cache_length, **kwargs):
        calls.append(
            {
                "raw": dict(observed_config.raw),
                "max_cache_length": max_cache_length,
                "kwargs": kwargs,
            }
        )
        return b"dynamic-plan"

    monkeypatch.setattr(qwen_model._QwenModel, "build_engine", build_engine)
    writer = _Writer()

    qwen_model.build(_request(tmp_path), writer)

    assert writer.header == {"family": "qwen", "task": "text_generation", "backend": "trt"}
    assert writer.bytes == {"engine.plan": b"dynamic-plan"}
    assert len(calls) == 1
    assert calls[0]["raw"]["dynamic_kv_cache"] is True
    assert calls[0]["raw"]["_decoder_engine_role"] == "dual_profile"
    assert calls[0]["max_cache_length"] == 64
    assert writer.json["runtime.json"]["dynamic_kv_cache"] is True
    assert writer.json["runtime.json"]["decoder_engine_layout"] == "dual_profile"


@pytest.mark.parametrize(
    ("model_type", "updates", "message"),
    [
        ("qwen3", {}, "fixed-capacity native KV"),
        ("qwen2", {"tensor_parallel_size": 2}, "tensor parallelism"),
        ("qwen2", {"quantization": "fp8"}, "quantization"),
    ],
)
def test_dynamic_kv_unsupported_routes_fail_before_loading_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_type: str,
    updates: dict,
    message: str,
) -> None:
    monkeypatch.setattr(qwen_model.ModelConfig, "from_dir", lambda _path: _config(model_type))

    def unexpected_load(*_args, **_kwargs):
        raise AssertionError("weights must not load for an unsupported dynamic KV request")

    monkeypatch.setattr(qwen_model._QwenModel, "load_weights", unexpected_load)

    with pytest.raises(NotImplementedError, match=message):
        qwen_model.build(_request(tmp_path, **updates), _Writer())


@pytest.mark.parametrize("role", ["dual_profile", "decode"])
def test_standard_decoder_routes_dynamic_kv_to_runtime_sized_profiles(
    monkeypatch: pytest.MonkeyPatch, role: str
) -> None:
    config = _config("qwen2")
    config.raw["dynamic_kv_cache"] = True
    config.raw["_decoder_engine_role"] = role
    calls: list[dict] = []

    def build_dual(_config, _weights, _max_cache_length, **kwargs):
        calls.append(kwargs)
        return b"plan"

    monkeypatch.setattr(
        standard_decoder_builder,
        "build_dual_profile_decoder_engine",
        build_dual,
    )

    assert standard_decoder_builder.build_standard_decoder_engine(config, {}, 64) == b"plan"
    assert len(calls) == 1
    assert calls[0]["profile_mode"] == role
    assert calls[0]["runtime_sized_kv_cache"] is True


def test_dynamic_kv_prefill_profile_keeps_the_bundle_limit() -> None:
    assert _resolve_prefill_lengths(
        256,
        64,
        None,
        native_kv_cache=False,
        profile_mode="dual_profile",
    ) == (64, 256)


def test_runtime_sized_receipt_requires_selected_rows() -> None:
    manifest = {"dynamic_kv_cache": True, "max_sequence_length": 256}
    case = {"kv_cache_size": "2359296", "expected_runtime_kv_cache_rows": 192}
    payload = {
        "runtime_command": ["trtmc", "run", "model.bundle", "--kv-cache-size", "2359296"],
        "runtime_stderr": "[trtmc] KV cache rows=192 (bundle max=256)\n",
    }

    _assert_runtime_sized_kv_receipt(payload, manifest, case)
    with pytest.raises(AssertionError):
        _assert_runtime_sized_kv_receipt(
            {**payload, "runtime_stderr": "[trtmc] KV cache rows=256 (bundle max=256)\n"},
            manifest,
            case,
        )
