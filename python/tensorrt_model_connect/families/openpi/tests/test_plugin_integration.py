# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect.checkpoint_mapper import WeightDict
from tensorrt_model_connect.config import ModelConfig
from tensorrt_model_connect.families import find_plugin
from tensorrt_model_connect.families.openpi.model_config import (
    OPENPI_MODEL_TYPE,
    OPENPI_UPSTREAM_COMMIT,
    get_profile,
)


action_builder = importlib.import_module(
    "tensorrt_model_connect.families.openpi.action_expert_builder"
)
plugin_module = importlib.import_module("tensorrt_model_connect.families.openpi.plugin")


_CHECKPOINT_SHA256 = "a" * 64
_MANIFEST_SHA256 = "b" * 64
_TOKENIZER_SHA256 = "c" * 64
_NORMALIZATION_SHA256 = "d" * 64
_PREFILL_PLAN_SHA256 = "e" * 64
_ACTION_PLAN_SHA256 = "f" * 64


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _config(profile: str = "pi05_droid", **updates) -> ModelConfig:
    raw = {
        "openpi_profile": profile,
        "openpi_upstream_commit": OPENPI_UPSTREAM_COMMIT,
        "openpi_checkpoint_identity_sha256": _CHECKPOINT_SHA256,
        "openpi_conversion_manifest_sha256": _MANIFEST_SHA256,
        "openpi_tokenizer_sha256": _TOKENIZER_SHA256,
        "openpi_normalization_sha256": _NORMALIZATION_SHA256,
        "openpi_prefill_engine_sha256": _PREFILL_PLAN_SHA256,
        "openpi_action_engine_sha256": _ACTION_PLAN_SHA256,
    }
    raw.update(updates)
    return ModelConfig(model_type=OPENPI_MODEL_TYPE, raw=raw)


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _render_json(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _bundle_fixture(tmp_path: Path, profile_name: str = "pi05_droid"):
    model_dir = tmp_path / "prepared"
    weights_relative = "model.safetensors"
    tokenizer_relative = "assets/paligemma_tokenizer.trtmcbpe"
    normalization_relative = f"assets/{get_profile(profile_name).asset_id}/norm_stats.json"
    manifest_relative = "openpi_conversion_manifest.json"
    tokenizer = b"TRTMCBPE\x01\x00native-paligemma-bpe"
    weights = b"synthetic-safetensors-for-integrity-tests"
    normalization = _render_json(
        {
            "norm_stats": {
                "actions": {"q01": [-1.0], "q99": [1.0]},
                "state": {"q01": [-1.0], "q99": [1.0]},
            }
        }
    )
    _write(model_dir / weights_relative, weights)
    _write(model_dir / tokenizer_relative, tokenizer)
    _write(model_dir / normalization_relative, normalization)
    # The pinned legacy prepared snapshot keeps nested source paths.  Standard
    # aliases let the unchanged shared bundle writer embed the exact same bytes.
    _write(model_dir / "tokenizer.model", tokenizer)
    _write(model_dir / "preprocessor_config.json", normalization)
    manifest = {
        "schema_version": 1,
        "format": "trtmc.openpi.weight-conversion",
        "profile": profile_name,
        "upstream": {
            "repository": "https://github.com/Physical-Intelligence/openpi",
            "commit": OPENPI_UPSTREAM_COMMIT,
        },
        "source_checkpoint": {"identity_sha256": _CHECKPOINT_SHA256},
        "artifacts": {
            "weights": {
                "file": weights_relative,
                "sha256": _sha256(weights),
            },
            "tokenizer": {
                "file": tokenizer_relative,
                "sha256": _sha256(tokenizer),
                "asset_sha256": _sha256(tokenizer),
            },
            "normalization": {
                "file": normalization_relative,
                "sha256": _sha256(normalization),
            },
        },
    }
    manifest_path = model_dir / manifest_relative
    manifest_payload = _render_json(manifest)
    _write(manifest_path, manifest_payload)
    config = _config(
        profile_name,
        openpi_weights_file=weights_relative,
        openpi_tokenizer_file=tokenizer_relative,
        openpi_tokenizer_sha256=_sha256(tokenizer),
        openpi_normalization_file=normalization_relative,
        openpi_normalization_sha256=_sha256(normalization),
        openpi_conversion_manifest=manifest_relative,
        openpi_conversion_manifest_sha256=_sha256(manifest_payload),
    )
    return {
        "model_dir": model_dir,
        "config": config,
        "weights": weights,
        "weights_path": model_dir / weights_relative,
        "tokenizer": tokenizer,
        "tokenizer_path": model_dir / tokenizer_relative,
        "normalization": normalization,
        "normalization_path": model_dir / normalization_relative,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_payload": manifest_payload,
    }


@pytest.mark.parametrize(
    "model_type",
    [
        "openpi",
        "OPENPI",
        "openpi_pi05_flow",
        "OpenPI-PI05-Flow",
        "pi05_droid",
        "PI05-DROID",
    ],
)
def test_plugin_matches_every_owned_alias(model_type: str) -> None:
    assert plugin_module.plugin.matches(model_type)


@pytest.mark.parametrize("model_type", ["", "pi0", "qwen", "openpi_pi0_flow"])
def test_plugin_rejects_foreign_model_types(model_type: str) -> None:
    assert not plugin_module.plugin.matches(model_type)


def test_profile_binding_requires_an_explicit_profile_and_pinned_commit() -> None:
    nested = _config()
    nested.raw.pop("openpi_profile")
    nested.raw["openpi"] = {"name": "pi05_droid"}
    assert plugin_module.plugin.get_bundle_config_overrides(nested)["openpi_profile"] == (
        "pi05_droid"
    )

    with pytest.raises(ValueError, match="unsupported OpenPI profile"):
        plugin_module.plugin.get_bundle_config_overrides(
            ModelConfig(
                model_type=OPENPI_MODEL_TYPE,
                raw={"openpi_upstream_commit": OPENPI_UPSTREAM_COMMIT},
            )
        )
    with pytest.raises(ValueError, match="audited upstream commit"):
        plugin_module.plugin.get_bundle_config_overrides(_config(openpi_upstream_commit="0" * 40))


@pytest.mark.parametrize(
    ("profile_name", "action_horizon", "external_action_dim", "discrete_state_input"),
    [
        ("pi05_droid", 15, 8, True),
    ],
)
def test_bundle_config_overrides_are_the_complete_native_runtime_contract(
    profile_name: str,
    action_horizon: int,
    external_action_dim: int,
    discrete_state_input: bool,
) -> None:
    profile = get_profile(profile_name)
    assert plugin_module.plugin.get_bundle_config_overrides(_config(profile_name)) == {
        "engine_backend": "trt",
        "runtime_strategy": "openpi_vla",
        "task_strategy": "robot_action_generation",
        "user_contract": "robot_action_chunk",
        "model_type": OPENPI_MODEL_TYPE,
        "openpi_profile": profile_name,
        "openpi_upstream_commit": OPENPI_UPSTREAM_COMMIT,
        "openpi_action_horizon": action_horizon,
        "openpi_internal_action_dim": 32,
        "openpi_external_action_dim": external_action_dim,
        "openpi_external_state_dim": 8,
        "openpi_prefix_length": 968,
        "openpi_max_token_length": 200,
        "openpi_num_layers": 18,
        "openpi_num_heads": 8,
        "openpi_num_kv_heads": 1,
        "openpi_head_dim": 256,
        "openpi_denoise_steps": 10,
        "openpi_discrete_state_input": discrete_state_input,
        "openpi_camera_names": list(profile.camera_names),
        "openpi_camera_mask": list(profile.camera_mask),
        "openpi_batch_size": 1,
        "openpi_runtime_contract": "native_cpp_device_resident_flow",
        "openpi_parameter_dtype": "bfloat16",
        "openpi_tokenizer_sha256": _TOKENIZER_SHA256,
        "openpi_normalization_sha256": _NORMALIZATION_SHA256,
        "openpi_prefill_engine_sha256": _PREFILL_PLAN_SHA256,
        "openpi_action_engine_sha256": _ACTION_PLAN_SHA256,
    }


def test_weight_inventory_accepts_only_the_exact_names_and_shapes(monkeypatch) -> None:
    monkeypatch.setattr(
        plugin_module,
        "required_prefill_weight_shapes",
        lambda _profile: {"prefix.weight": (2, 3)},
    )
    monkeypatch.setattr(
        action_builder,
        "required_action_weight_shapes",
        lambda _profile: {"action.weight": (3,)},
    )
    profile = get_profile("pi05_droid")
    valid = WeightDict(
        {
            "prefix.weight": np.zeros((2, 3), dtype=np.float32),
            "action.weight": np.zeros(3, dtype=np.float16),
        }
    )
    plugin_module._validate_weight_inventory(valid, profile)

    with pytest.raises(ValueError, match=r"missing=action\.weight"):
        plugin_module._validate_weight_inventory(
            WeightDict({"prefix.weight": valid["prefix.weight"]}), profile
        )
    with pytest.raises(ValueError, match=r"unexpected=rogue\.weight"):
        plugin_module._validate_weight_inventory(
            WeightDict({**valid, "rogue.weight": np.zeros(1, dtype=np.float32)}),
            profile,
        )
    with pytest.raises(ValueError, match=r"prefix\.weight: expected \(2, 3\), got \(3, 2\)"):
        plugin_module._validate_weight_inventory(
            WeightDict(
                {
                    "prefix.weight": np.zeros((3, 2), dtype=np.float32),
                    "action.weight": valid["action.weight"],
                }
            ),
            profile,
        )


def test_extra_engine_uses_the_exact_runtime_section_name(monkeypatch) -> None:
    calls = []

    def fake_build(profile, weights, *, precision, verbose):
        calls.append((profile, weights, precision, verbose))
        return b"ACTION-PLAN"

    monkeypatch.setattr(action_builder, "build_action_expert_engine", fake_build)
    weights = WeightDict({"sentinel": np.zeros(1, dtype=np.float32)})
    config = _config("pi05_droid")
    sections = plugin_module.plugin.build_extra_engines(
        config,
        weights,
        1234,
        precision="bf16",
        verbose=True,
        build_timing={},
    )

    assert sections == {"openpi_action_step_engine_plan": b"ACTION-PLAN"}
    assert calls == [(get_profile("pi05_droid"), weights, "bf16", True)]
    assert config.raw["openpi_action_engine_sha256"] == _sha256(b"ACTION-PLAN")


def test_prefill_engine_hash_is_bound_into_the_runtime_config(monkeypatch) -> None:
    monkeypatch.setattr(
        plugin_module,
        "build_prefill_engine",
        lambda *_args, **_kwargs: b"PREFILL-PLAN",
    )
    config = _config("pi05_droid")

    assert (
        plugin_module.plugin.build_engine(
            config,
            WeightDict(),
            1234,
            precision="bf16",
        )
        == b"PREFILL-PLAN"
    )
    assert config.raw["openpi_prefill_engine_sha256"] == _sha256(b"PREFILL-PLAN")


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_plugin_rejects_unqualified_precision_before_load_or_build(
    monkeypatch, precision: str
) -> None:
    def unexpected_call(*_args, **_kwargs):
        raise AssertionError("OpenPI reached an implementation behind the precision gate")

    monkeypatch.setattr(plugin_module, "_load_prepared_weights", unexpected_call)
    monkeypatch.setattr(plugin_module, "build_prefill_engine", unexpected_call)
    monkeypatch.setattr(action_builder, "build_action_expert_engine", unexpected_call)
    config = _config("pi05_droid")
    weights = WeightDict()
    message = r"only precision='bf16'"

    with pytest.raises(ValueError, match=message):
        plugin_module.plugin.load_weights("unused", config, precision=precision)
    with pytest.raises(ValueError, match=message):
        plugin_module.plugin.build_engine(config, weights, 0, precision=precision)
    with pytest.raises(ValueError, match=message):
        plugin_module.plugin.build_extra_engines(config, weights, 0, precision=precision)


def test_prepared_weights_are_sha256_bound_to_the_conversion_manifest(tmp_path) -> None:
    fixture = _bundle_fixture(tmp_path)

    assert (
        plugin_module._validated_prepared_weight_path(fixture["model_dir"], fixture["config"])
        == fixture["weights_path"]
    )

    fixture["weights_path"].write_bytes(b"tampered-but-still-structurally-valid-safetensors")
    with pytest.raises(ValueError, match="manifest weights SHA-256"):
        plugin_module._validated_prepared_weight_path(fixture["model_dir"], fixture["config"])


def test_prepared_asset_validation_accepts_standard_aliases_for_legacy_paths(
    tmp_path,
) -> None:
    fixture = _bundle_fixture(tmp_path)
    assert (
        plugin_module._validated_prepared_weight_path(fixture["model_dir"], fixture["config"])
        == fixture["weights_path"]
    )
    assert fixture["config"].raw["openpi_tokenizer_sha256"] == _sha256(fixture["tokenizer"])
    assert fixture["config"].raw["openpi_normalization_sha256"] == _sha256(fixture["normalization"])


@pytest.mark.parametrize(
    ("standard_name", "message"),
    [
        ("tokenizer.model", "does not match the converted tokenizer"),
        ("preprocessor_config.json", "does not match the normalization"),
    ],
)
def test_prepared_asset_validation_rejects_mismatched_standard_aliases(
    tmp_path, standard_name: str, message: str
) -> None:
    fixture = _bundle_fixture(tmp_path)
    (fixture["model_dir"] / standard_name).write_bytes(b"tampered-alias")
    with pytest.raises(ValueError, match=message):
        plugin_module._validated_prepared_weight_path(fixture["model_dir"], fixture["config"])


@pytest.mark.parametrize(
    ("config_key", "label"),
    [
        ("openpi_tokenizer_file", "flattened tokenizer"),
        ("openpi_normalization_file", "normalization statistics"),
        ("openpi_conversion_manifest", "conversion manifest"),
    ],
)
def test_prepared_asset_validation_rejects_paths_outside_the_prepared_directory(
    tmp_path, config_key: str, label: str
) -> None:
    fixture = _bundle_fixture(tmp_path)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    fixture["config"].raw[config_key] = "../outside.bin"

    with pytest.raises(ValueError, match=rf"{label} escapes"):
        plugin_module._validated_prepared_weight_path(fixture["model_dir"], fixture["config"])


def test_prepared_asset_validation_rejects_tampered_payload_semantics(tmp_path) -> None:
    fixture = _bundle_fixture(tmp_path)
    fixture["tokenizer_path"].write_bytes(b"NOT-BPE")
    with pytest.raises(ValueError, match="invalid binary header"):
        plugin_module._validated_prepared_weight_path(fixture["model_dir"], fixture["config"])

    fixture = _bundle_fixture(tmp_path / "norm")
    fixture["normalization_path"].write_bytes(b"[]\n")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        plugin_module._validated_prepared_weight_path(fixture["model_dir"], fixture["config"])

    fixture = _bundle_fixture(tmp_path / "manifest")
    fixture["manifest"]["upstream"]["commit"] = "0" * 40
    tampered_manifest = _render_json(fixture["manifest"])
    fixture["manifest_path"].write_bytes(tampered_manifest)
    fixture["config"].raw["openpi_conversion_manifest_sha256"] = _sha256(tampered_manifest)
    with pytest.raises(ValueError, match="unaudited upstream repository or commit"):
        plugin_module._validated_prepared_weight_path(fixture["model_dir"], fixture["config"])


@pytest.mark.parametrize(
    "checkpoint_sha256",
    ["", "0" * 64, "A" * 64, "g" * 64, "a" * 63],
)
def test_prepared_asset_validation_rejects_invalid_checkpoint_hashes(
    tmp_path, checkpoint_sha256: str
) -> None:
    fixture = _bundle_fixture(tmp_path)
    fixture["config"].raw["openpi_checkpoint_identity_sha256"] = checkpoint_sha256
    with pytest.raises(ValueError, match="checkpoint identity SHA-256"):
        plugin_module._validated_prepared_weight_path(fixture["model_dir"], fixture["config"])


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("config_tokenizer", "tokenizer SHA-256"),
        ("config_manifest", "conversion manifest SHA-256"),
        ("manifest_tokenizer", "manifest tokenizer SHA-256"),
        ("manifest_normalization", "manifest normalization SHA-256"),
        ("manifest_checkpoint", "manifest checkpoint identity"),
    ],
)
def test_prepared_asset_validation_rejects_every_hash_binding_mismatch(
    tmp_path, tamper: str, message: str
) -> None:
    fixture = _bundle_fixture(tmp_path)
    if tamper == "config_tokenizer":
        fixture["config"].raw["openpi_tokenizer_sha256"] = "f" * 64
    elif tamper == "config_manifest":
        fixture["config"].raw["openpi_conversion_manifest_sha256"] = "f" * 64
    elif tamper == "manifest_tokenizer":
        fixture["manifest"]["artifacts"]["tokenizer"]["sha256"] = "f" * 64
    elif tamper == "manifest_normalization":
        fixture["manifest"]["artifacts"]["normalization"]["sha256"] = "f" * 64
    elif tamper == "manifest_checkpoint":
        fixture["manifest"]["source_checkpoint"]["identity_sha256"] = "f" * 64
    else:  # pragma: no cover - exhaustive parameter guard
        raise AssertionError(tamper)

    if tamper.startswith("manifest_"):
        manifest_payload = _render_json(fixture["manifest"])
        fixture["manifest_path"].write_bytes(manifest_payload)
        fixture["config"].raw["openpi_conversion_manifest_sha256"] = _sha256(manifest_payload)

    with pytest.raises(ValueError, match=message):
        plugin_module._validated_prepared_weight_path(fixture["model_dir"], fixture["config"])


def test_repository_family_index_drives_openpi_plugin_discovery() -> None:
    import tensorrt_model_connect.families as families

    metadata = next(meta for meta in families._load_family_metadata() if meta.id == "openpi")
    assert metadata.import_module == "openpi"
    assert metadata.aliases == {
        "openpi",
        "openpi_pi05_flow",
        "pi05_droid",
    }
    assert metadata.prefixes == {"openpi", "pi05"}
    assert metadata.config_adapter == "model_config.py|config_from_dir"
    assert set(metadata.hf_allow_patterns) == {
        "openpi_config.json",
        "openpi_conversion_manifest.json",
        "tokenizer.model",
        "preprocessor_config.json",
        "assets/paligemma_tokenizer.trtmcbpe",
        "assets/droid/norm_stats.json",
        "trtmc_openpi/**",
    }
    assert {
        "NVIDIA/TensorRT-Model-Connect-OpenPI-Pi05-DROID|trtmc_openpi/reference/reference-set.json",
        "NVIDIA/TensorRT-Model-Connect-OpenPI-Pi05-DROID|trtmc_openpi/request/request.json",
        "NVIDIA/TensorRT-Model-Connect-OpenPI-Pi05-DROID|"
        "trtmc_openpi/performance/pytorch-eager.json",
    }.issubset(metadata.hf_required_files)
    for alias in metadata.aliases:
        assert find_plugin(alias) is plugin_module.plugin


def test_repository_e2e_manifests_validate_as_openpi_cases(monkeypatch) -> None:
    from tests.e2e_harness import manifest_loader

    # Runtime-strategy registration is validated by the runtime model manifest
    # suite. This test isolates the Python family and its DROID E2E manifest.
    monkeypatch.setattr(
        manifest_loader,
        "_known_runtime_strategies",
        lambda: frozenset({"openpi_vla"}),
    )
    repository_root = Path(__file__).resolve().parents[5]
    cases = manifest_loader.load_all_manifests(
        repository_root / "tests" / "e2e" / "models" / "openpi"
    )

    assert [case.name for case in cases] == ["pi05-droid"]
    assert {case.family for case in cases} == {"openpi"}
    assert {case.runtime_strategy for case in cases} == {"openpi_vla"}
    assert {case.task_strategy for case in cases} == {"robot_action_generation"}
    assert {case.metadata["precision"] for case in cases} == {"bf16"}
