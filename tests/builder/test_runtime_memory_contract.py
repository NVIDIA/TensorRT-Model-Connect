# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the versioned native runtime-memory contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_model_connect.config import ModelConfig
from tensorrt_model_connect.dynamic_memory_contract import (
    DynamicMemoryContractError,
    UnknownModuleResidencyCalibrationError,
    load_dynamic_memory_qualifications,
    module_residency_plan_set_sha256,
    qualified_runtime_stack_sha256,
    seal_runtime_memory_contract,
    seal_runtime_memory_contract_from_qualified_manifest,
    validate_runtime_memory_contract,
)

pytestmark = pytest.mark.dynamic_memory


QWEN_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
QWEN_CONFIG_SHA256 = "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd"
RUNTIME_CONFIG_BYTES = b'{"model_type":"qwen3","num_hidden_layers":28}'
RUNTIME_CONFIG_SHA256 = hashlib.sha256(RUNTIME_CONFIG_BYTES).hexdigest()


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
            "cudnn_frontend_revision": "7b9b711c22b6823e87150213ecd8449260db8610",
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


def _plan_sections() -> dict[str, bytes]:
    return {
        "engine_plan": b"serialized decode engine plan",
        "prefill_engine_plan": b"serialized prefill engine plan",
    }


def _valid_calibration(
    contract: dict | None = None,
) -> dict:
    base = _valid_contract() if contract is None else contract
    sections = _plan_sections()
    plans = [
        {
            "section_name": "engine_plan",
            "section_sha256": hashlib.sha256(sections["engine_plan"]).hexdigest(),
            "role": "decode",
            "optimization_profile_count": len(base["active_kv_profile_limits"]),
        },
        {
            "section_name": "prefill_engine_plan",
            "section_sha256": hashlib.sha256(
                sections["prefill_engine_plan"]
            ).hexdigest(),
            "role": "prefill",
            "optimization_profile_count": 1,
        },
    ]
    return {
        "schema_version": 1,
        "measurement_kind": "nvml_process_cumulative_first_use",
        "cuda_module_loading_mode": "lazy",
        "evidence_provenance": "external_manifest_v1",
        "qualified_runtime_stack_sha256": qualified_runtime_stack_sha256(
            base["qualified_runtime_stack"]
        ),
        "plan_set_sha256": module_residency_plan_set_sha256(plans),
        "plans": plans,
        "profile_reserves": [
            {
                "covering_profile_limit": limit,
                "cumulative_reserve_bytes": (index + 1) * 16 * 1024 * 1024,
            }
            for index, limit in enumerate(base["active_kv_profile_limits"])
        ],
        "evidence_sha256": hashlib.sha256(b"qualification evidence").hexdigest(),
    }


def test_version_one_provisional_runtime_memory_contract_is_normalized() -> None:
    contract = _valid_contract()
    assert validate_runtime_memory_contract(contract) == contract


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda value: value.update(contract_version=3), "contract_version"),
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
            lambda value: value.update(active_kv_profile_limits=[128, 512, 2048]),
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


def test_v2_sealing_binds_stack_plan_set_and_actual_plan_bytes() -> None:
    base = _valid_contract()
    calibration = _valid_calibration(base)

    sealed = seal_runtime_memory_contract(
        base,
        plan_sections=_plan_sections(),
        module_residency_calibration=calibration,
        runtime_config_bytes=RUNTIME_CONFIG_BYTES,
    )

    assert base["contract_version"] == 1
    assert sealed["contract_version"] == 2
    assert sealed["runtime_config_sha256"] == RUNTIME_CONFIG_SHA256
    assert sealed["module_residency_calibration"] == calibration
    assert validate_runtime_memory_contract(sealed) == sealed


def test_v2_runtime_config_digest_is_required_and_strict() -> None:
    base = _valid_contract()
    sealed = seal_runtime_memory_contract(
        base,
        plan_sections=_plan_sections(),
        module_residency_calibration=_valid_calibration(base),
        runtime_config_bytes=RUNTIME_CONFIG_BYTES,
    )

    missing = dict(sealed)
    missing.pop("runtime_config_sha256")
    with pytest.raises(
        DynamicMemoryContractError,
        match="runtime_config_sha256",
    ):
        validate_runtime_memory_contract(missing)

    malformed = dict(sealed)
    malformed["runtime_config_sha256"] = RUNTIME_CONFIG_SHA256.upper()
    with pytest.raises(
        DynamicMemoryContractError,
        match="runtime_config_sha256.*lowercase SHA-256",
    ):
        validate_runtime_memory_contract(malformed)


def test_legacy_v2_calibration_without_provenance_defaults_to_external() -> None:
    base = _valid_contract()
    calibration = _valid_calibration(base)
    calibration.pop("evidence_provenance")

    sealed = seal_runtime_memory_contract(
        base,
        plan_sections=_plan_sections(),
        module_residency_calibration=calibration,
        runtime_config_bytes=RUNTIME_CONFIG_BYTES,
    )

    assert (
        sealed["module_residency_calibration"]["evidence_provenance"]
        == "external_manifest_v1"
    )


def test_v2_calibration_rejects_unknown_evidence_provenance() -> None:
    base = _valid_contract()
    calibration = _valid_calibration(base)
    calibration["evidence_provenance"] = "downgraded"

    with pytest.raises(
        DynamicMemoryContractError,
        match="evidence_provenance",
    ):
        seal_runtime_memory_contract(
            base,
            plan_sections=_plan_sections(),
            module_residency_calibration=calibration,
            runtime_config_bytes=RUNTIME_CONFIG_BYTES,
        )


def test_qualified_manifest_rejects_embedded_evidence_provenance(
    tmp_path: Path,
) -> None:
    base = _valid_contract()
    calibration = _valid_calibration(base)
    calibration["evidence_provenance"] = "embedded_bundle_v1"
    manifest = tmp_path / "MODULE_RESIDENCY_CALIBRATIONS.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "records": [calibration]}),
        encoding="utf-8",
    )

    with pytest.raises(
        DynamicMemoryContractError,
        match="evidence_provenance",
    ):
        seal_runtime_memory_contract_from_qualified_manifest(
            base,
            family="qwen",
            plan_sections=_plan_sections(),
            runtime_config_bytes=RUNTIME_CONFIG_BYTES,
            manifest_path=manifest,
        )


def test_qualified_manifest_sealing_selects_the_exact_plan(
    tmp_path: Path,
) -> None:
    base = _valid_contract()
    selected = _valid_calibration(base)
    other = _valid_calibration(base)
    other["plans"][0]["section_sha256"] = "0" * 64
    other["plan_set_sha256"] = module_residency_plan_set_sha256(
        other["plans"]
    )
    manifest = tmp_path / "MODULE_RESIDENCY_CALIBRATIONS.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "records": [other, selected],
            }
        ),
        encoding="utf-8",
    )

    sealed = seal_runtime_memory_contract_from_qualified_manifest(
        base,
        family="qwen",
        plan_sections=_plan_sections(),
        runtime_config_bytes=RUNTIME_CONFIG_BYTES,
        manifest_path=manifest,
    )

    assert sealed["contract_version"] == 2
    assert sealed["module_residency_calibration"] == selected


def test_qualified_manifest_sealing_rejects_an_unqualified_plan(
    tmp_path: Path,
) -> None:
    base = _valid_contract()
    manifest = tmp_path / "MODULE_RESIDENCY_CALIBRATIONS.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "records": [_valid_calibration(base)],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        DynamicMemoryContractError,
        match="No unique exact-plan",
    ):
        seal_runtime_memory_contract_from_qualified_manifest(
            base,
            family="qwen",
            plan_sections={
                **_plan_sections(),
                "engine_plan": b"unqualified plan bytes",
            },
            runtime_config_bytes=RUNTIME_CONFIG_BYTES,
            manifest_path=manifest,
        )


def test_qualified_manifest_distinguishes_malformed_record_from_exact_miss(
    tmp_path: Path,
) -> None:
    base = _valid_contract()
    malformed = _valid_calibration(base)
    del malformed["plans"][0]["section_sha256"]
    manifest = tmp_path / "MODULE_RESIDENCY_CALIBRATIONS.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "records": [malformed]}),
        encoding="utf-8",
    )

    with pytest.raises(DynamicMemoryContractError) as caught:
        seal_runtime_memory_contract_from_qualified_manifest(
            base,
            family="qwen",
            plan_sections={
                **_plan_sections(),
                "engine_plan": b"unknown plan bytes",
            },
            runtime_config_bytes=RUNTIME_CONFIG_BYTES,
            manifest_path=manifest,
        )

    assert not isinstance(
        caught.value,
        UnknownModuleResidencyCalibrationError,
    )
    assert "missing required field" in str(caught.value)


def test_v2_accepts_eager_cuda_module_loading_calibration() -> None:
    base = _valid_contract()
    calibration = _valid_calibration(base)
    calibration["cuda_module_loading_mode"] = "eager"

    sealed = seal_runtime_memory_contract(
        base,
        plan_sections=_plan_sections(),
        module_residency_calibration=calibration,
        runtime_config_bytes=RUNTIME_CONFIG_BYTES,
    )

    assert (
        sealed["module_residency_calibration"]["cuda_module_loading_mode"]
        == "eager"
    )


def test_v2_requires_module_residency_calibration() -> None:
    contract = _valid_contract()
    contract["contract_version"] = 2
    contract["runtime_config_sha256"] = RUNTIME_CONFIG_SHA256

    with pytest.raises(
        DynamicMemoryContractError,
        match="module_residency_calibration",
    ):
        validate_runtime_memory_contract(contract)


def test_calibration_digest_preimages_are_language_neutral() -> None:
    contract = _valid_contract()
    stack = contract["qualified_runtime_stack"]
    stack_preimage = b"".join(
        (
            f"{len(key.encode('ascii'))}:{key}={len(stack[key].encode('utf-8'))}:{stack[key]}\n"
        ).encode("utf-8")
        for key in (
            "sm",
            "tensorrt",
            "cuda_runtime",
            "cudnn_backend",
            "cudnn_frontend_revision",
            "nvrtc",
            "driver",
        )
    )
    assert (
        qualified_runtime_stack_sha256(stack)
        == hashlib.sha256(stack_preimage).hexdigest()
    )

    plans = _valid_calibration(contract)["plans"]
    plan_preimage = b"".join(
        (
            f"{plan['section_name']}\0{plan['section_sha256']}\0"
            f"{plan['role']}\0{plan['optimization_profile_count']}\n"
        ).encode("ascii")
        for plan in plans
    )
    assert (
        module_residency_plan_set_sha256(plans)
        == hashlib.sha256(plan_preimage).hexdigest()
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda value: value.update(extra=True),
            "unsupported field",
        ),
        (
            lambda value: value.update(schema_version=2),
            "schema_version",
        ),
        (
            lambda value: value.update(measurement_kind="estimated"),
            "measurement_kind",
        ),
        (
            lambda value: value.update(cuda_module_loading_mode="default"),
            "cuda_module_loading_mode",
        ),
        (
            lambda value: value.update(qualified_runtime_stack_sha256="0" * 64),
            "does not bind the outer",
        ),
        (
            lambda value: value.update(plan_set_sha256="0" * 64),
            "does not bind the ordered plans",
        ),
        (
            lambda value: value["plans"].reverse(),
            "ordered exactly",
        ),
        (
            lambda value: value["plans"][0].update(extra=True),
            "unsupported field",
        ),
        (
            lambda value: value["plans"][1].update(optimization_profile_count=2),
            "exactly one prefill",
        ),
        (
            lambda value: value["plans"][0].update(
                section_sha256=value["plans"][0]["section_sha256"].upper()
            ),
            "lowercase SHA-256",
        ),
        (
            lambda value: value["profile_reserves"][0].update(extra=True),
            "unsupported field",
        ),
        (
            lambda value: value["profile_reserves"][1].update(
                covering_profile_limit=129
            ),
            "align exactly",
        ),
        (
            lambda value: value["profile_reserves"][1].update(
                cumulative_reserve_bytes=1
            ),
            "nondecreasing",
        ),
        (
            lambda value: value["profile_reserves"][-1].update(
                cumulative_reserve_bytes=0
            ),
            "positive integer",
        ),
        (
            lambda value: value.update(evidence_sha256="A" * 64),
            "lowercase SHA-256",
        ),
    ),
)
def test_invalid_v2_calibration_fails_closed(
    mutation,
    message: str,
) -> None:
    contract = _valid_contract()
    calibration = _valid_calibration(contract)
    mutation(calibration)
    contract.update(
        contract_version=2,
        runtime_config_sha256=RUNTIME_CONFIG_SHA256,
        module_residency_calibration=calibration,
    )

    with pytest.raises(DynamicMemoryContractError, match=message):
        validate_runtime_memory_contract(contract)


def test_v2_rejects_outer_stack_drift_after_sealing() -> None:
    base = _valid_contract()
    sealed = seal_runtime_memory_contract(
        base,
        plan_sections=_plan_sections(),
        module_residency_calibration=_valid_calibration(base),
        runtime_config_bytes=RUNTIME_CONFIG_BYTES,
    )
    sealed["qualified_runtime_stack"]["driver"] = "580.105.09"

    with pytest.raises(
        DynamicMemoryContractError,
        match="does not bind the outer",
    ):
        validate_runtime_memory_contract(sealed)


@pytest.mark.parametrize(
    ("sections", "message"),
    (
        (
            {"engine_plan": b"serialized decode engine plan"},
            "missing required",
        ),
        (
            {
                **_plan_sections(),
                "other_plan": b"not calibrated",
            },
            "unsupported field",
        ),
        (
            {
                **_plan_sections(),
                "engine_plan": b"different decode engine plan",
            },
            "hash mismatch for engine_plan",
        ),
    ),
)
def test_sealing_rejects_missing_extra_or_drifted_plan_bytes(
    sections: dict[str, bytes],
    message: str,
) -> None:
    base = _valid_contract()
    with pytest.raises(DynamicMemoryContractError, match=message):
        seal_runtime_memory_contract(
            base,
            plan_sections=sections,
            module_residency_calibration=_valid_calibration(base),
            runtime_config_bytes=RUNTIME_CONFIG_BYTES,
        )


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


def test_engine_builder_injects_one_transient_graph_signal_only_when_qualified() -> (
    None
):
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
