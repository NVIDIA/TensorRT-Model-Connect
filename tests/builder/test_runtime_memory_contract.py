# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the versioned native runtime-memory contract."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_model_connect.config import ModelConfig
from tensorrt_model_connect.dynamic_memory_contract import (
    DynamicMemoryContractError,
    load_dynamic_memory_qualifications,
    validate_runtime_memory_contract,
)

pytestmark = pytest.mark.dynamic_memory


QWEN_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
QWEN_CONFIG_SHA256 = (
    "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd"
)


def _valid_contract() -> dict:
    return {
        "contract_version": 1,
        "qualified_model_id": "Qwen/Qwen3-0.6B",
        "qualified_model_revision": QWEN_REVISION,
        "qualified_config_sha256": QWEN_CONFIG_SHA256,
        "qualified_target": "gb300-trt-11.2",
        "qualified_runtime_stack": {
            "sm": "sm103",
            "tensorrt": "11.2.0.113",
            "cuda_runtime": "13.3",
            "cudnn_backend": "9.20.0",
            "cudnn_frontend_revision":
                "7b9b711c22b6823e87150213ecd8449260db8610",
            "nvrtc": "13.3",
            "driver": "580.105.08",
        },
        "native_kv_plugin_abi": 2,
        "model_context_limit": 40960,
        "prefill_chunk_limit": 2048,
        "kv_layout": "contiguous_runtime_v1",
        "kv_dtype": "bfloat16",
        "kv_bytes_per_token": 114688,
        "active_kv_profile_limits": [
            128,
            512,
            2048,
            8192,
            32768,
            40960,
        ],
        "runtime_owned": True,
    }


def test_version_one_runtime_memory_contract_is_normalized() -> None:
    contract = _valid_contract()
    assert validate_runtime_memory_contract(contract) == contract


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda value: value.update(contract_version=2), "contract_version"),
        (lambda value: value.pop("qualified_target"), "missing required"),
        (
            lambda value: value["qualified_runtime_stack"].pop("driver"),
            "missing required",
        ),
        (
            lambda value: value["qualified_runtime_stack"].update(
                cudnn_frontend_revision="not-a-revision"
            ),
            "cudnn_frontend_revision",
        ),
        (
            lambda value: value.update(
                active_kv_profile_limits=[128, 2048, 512, 40960]
            ),
            "strictly increasing",
        ),
        (
            lambda value: value.update(
                active_kv_profile_limits=[128, 512, 2048]
            ),
            "must end",
        ),
        (
            lambda value: value.update(runtime_owned=False),
            "runtime_owned",
        ),
        (
            lambda value: value.update(prefill_chunk_limit=40961),
            "cannot exceed",
        ),
    ),
)
def test_invalid_runtime_memory_contract_fails_closed(
    mutation,
    message: str,
) -> None:
    contract = _valid_contract()
    mutation(contract)
    with pytest.raises(DynamicMemoryContractError, match=message):
        validate_runtime_memory_contract(contract)


@pytest.mark.parametrize(
    ("model_id", "config", "expected_bytes"),
    (
        (
            "Qwen/Qwen3-0.6B",
            ModelConfig.create_tiny(
                "qwen3",
                max_position_embeddings=40960,
                num_hidden_layers=28,
                hidden_size=1024,
                num_attention_heads=16,
                num_key_value_heads=8,
                head_dim=128,
            ),
            114688,
        ),
        (
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            ModelConfig.create_tiny(
                "llama",
                max_position_embeddings=2048,
                num_hidden_layers=22,
                hidden_size=2048,
                num_attention_heads=32,
                num_key_value_heads=4,
            ),
            22528,
        ),
    ),
)
def test_b_is_derived_from_actual_config_and_bf16_precision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model_id: str,
    config: ModelConfig,
    expected_bytes: int,
) -> None:
    import tensorrt_model_connect.dynamic_memory_contract as contract_module

    qualification = next(
        record
        for record in load_dynamic_memory_qualifications()
        if record.qualified_model_id == model_id
    )
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setattr(
        contract_module,
        "_sha256_file",
        lambda _path: qualification.qualified_config_sha256,
    )

    contract = qualification.runtime_memory_contract(
        model_dir=tmp_path,
        config=config,
        precision="bf16",
    )
    assert contract["kv_bytes_per_token"] == expected_bytes
    assert contract["model_context_limit"] == config.max_position_embeddings
    assert contract["kv_dtype"] == "bfloat16"


def test_contract_rejects_precision_and_model_limit_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import tensorrt_model_connect.dynamic_memory_contract as contract_module

    qualification = next(
        record
        for record in load_dynamic_memory_qualifications()
        if record.family == "qwen"
    )
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setattr(
        contract_module,
        "_sha256_file",
        lambda _path: qualification.qualified_config_sha256,
    )
    config = ModelConfig.create_tiny(
        "qwen3",
        max_position_embeddings=40960,
        num_hidden_layers=28,
        hidden_size=1024,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
    )

    with pytest.raises(DynamicMemoryContractError, match="precision mismatch"):
        qualification.runtime_memory_contract(
            model_dir=tmp_path,
            config=config,
            precision="fp16",
        )

    config.max_position_embeddings = 4096
    with pytest.raises(DynamicMemoryContractError, match="context limit mismatch"):
        qualification.runtime_memory_contract(
            model_dir=tmp_path,
            config=config,
            precision="bf16",
        )


def test_engine_builder_injects_one_transient_graph_signal_only_when_qualified() -> None:
    from tensorrt_model_connect.engine_builder import (
        _prepare_runtime_memory_contract,
    )

    config = ModelConfig.create_tiny(
        "qwen3",
        max_position_embeddings=40960,
    )
    persistent = _valid_contract()
    qualification = SimpleNamespace(
        family="qwen",
        model_context_limit=40960,
        active_kv_profile_limits=(128, 512, 2048, 8192, 32768, 40960),
        runtime_memory_contract=lambda **_kwargs: dict(persistent),
    )

    returned = _prepare_runtime_memory_contract(
        config,
        qualification=qualification,
        family_name="qwen",
        precision="bf16",
        max_cache_length=40960,
        decoder_engine_layout="split",
        dynamic_kv_cache=True,
        profile_limits=[128, 512, 2048, 8192, 32768, 40960],
    )
    assert returned == persistent
    assert "precision" not in returned
    assert config.raw["_runtime_memory_contract"] == {
        **persistent,
        "precision": "bf16",
    }

    assert (
        _prepare_runtime_memory_contract(
            config,
            qualification=None,
            family_name="qwen",
            precision="bf16",
            max_cache_length=256,
            decoder_engine_layout="split",
            dynamic_kv_cache=False,
            profile_limits=None,
        )
        is None
    )
    assert "_runtime_memory_contract" not in config.raw
