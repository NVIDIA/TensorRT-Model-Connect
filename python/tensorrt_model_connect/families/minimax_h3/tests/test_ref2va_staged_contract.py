# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_model_connect.families.minimax_h3 import (
    checkpoint,
    ref2va_checkpoint,
    staged_build,
)
from tensorrt_model_connect.families.minimax_h3.plugin import (
    plugin,
    write_path_free_effective_build_config,
)
from tensorrt_model_connect.families.minimax_h3.provenance import (
    validate_native_bundle_config,
)
from tensorrt_model_connect.families.minimax_h3.ref2va_checkpoint import (
    CHECKPOINT_REVISION,
    COMPONENT_NAME,
    MODEL_ID,
    TOTAL_TENSOR_BYTES,
    TransformerRefIdentity,
)
from tests.builder.conftest import read_bundle_file


SOURCE_REVISION = "a" * 40


@pytest.fixture(autouse=True)
def _source_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRTMC_MINIMAX_H3_SOURCE_REVISION", SOURCE_REVISION)


@pytest.fixture(autouse=True)
def _synthetic_checkpoint_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    def record(model: Path) -> dict:
        files = {}
        for path in sorted(model.rglob("*")):
            if not path.is_file() or COMPONENT_NAME in path.relative_to(model).parts:
                continue
            relative = path.relative_to(model).as_posix()
            payload = path.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            files[relative] = {
                "blob_id": digest,
                "bytes": len(payload),
                "sha256": digest,
            }
        payload = {
            "repository": "MiniMaxAI/MiniMax-H3",
            "revision": "48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc",
            "files": files,
        }
        return {
            **payload,
            "file_count": len(files),
            "inventory_sha256": hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }

    monkeypatch.setattr(staged_build, "checkpoint_snapshot_record", record)
    monkeypatch.setattr(staged_build, "validate_checkpoint_snapshot_record", lambda value: value)


def _identity() -> TransformerRefIdentity:
    return TransformerRefIdentity(
        model_id=MODEL_ID,
        revision=CHECKPOINT_REVISION,
        component=COMPONENT_NAME,
        tensor_bytes=TOTAL_TENSOR_BYTES,
        tensor_count=638,
        inventory_sha256="1" * 64,
        files={},
    )


def _adapter_identity():
    metadata = {
        "schema_version": 1,
        "adapter_model_id": "FastVideo/FastVideo-FastH3-4-step-v1",
        "adapter_source_revision": "2" * 40,
        "adapter_sha256": "3" * 64,
        "adapter_bytes": 5_339_117_712,
        "adapter_tensor_count": 856,
        "adapter_low_rank_tensor_count": 724,
        "adapter_diff_tensor_count": 82,
        "adapter_set_weight_tensor_count": 50,
        "adapter_gate_tensor_count": 50,
        "adapter_partition_tensor_counts": {
            "adaln_precompute": 156,
            "denoiser": 700,
        },
        "adapter_base_revision": "4" * 40,
        "adapter_finetuned_revision": "5" * 40,
    }
    return SimpleNamespace(bundle_metadata=lambda: dict(metadata))


def _model(root: Path) -> Path:
    model = root / "model"
    tokenizer = model / "tokenizer" / "tokenizer.json"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_text("{}", encoding="utf-8")
    audio = model / "audio_vae" / "config.json"
    audio.parent.mkdir(parents=True)
    audio.write_text(
        json.dumps(
            {
                "decoder_rates": [5, 5, 2, 2, 2, 2, 2],
                "sampling_rate": 32_000,
                "latents_mean": [0.0] * 32,
                "latents_std": [1.0] * 32,
            }
        ),
        encoding="utf-8",
    )
    return model


def test_staged_ref2va_is_opt_in_strict_and_adds_all_sections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _model(tmp_path)
    transformer_ref = model / COMPONENT_NAME
    transformer_ref.mkdir()
    output = tmp_path / "h3-ref.bundle"
    calls = []

    monkeypatch.setattr(
        ref2va_checkpoint,
        "validate_transformer_ref_checkpoint",
        lambda *_args, **_kwargs: _identity(),
    )

    def build(component, _model_path, plan, **kwargs):
        calls.append((component, kwargs))
        plan.write_bytes(component.encode())

    monkeypatch.setattr(staged_build, "_run_component", build)
    monkeypatch.setattr(staged_build.trt_compat, "configure_backend", lambda **_kwargs: None)
    monkeypatch.setattr(staged_build.trt_compat, "tensorrt_version", lambda: "1.6.1.120")
    monkeypatch.setattr(staged_build.trt_compat, "tensorrt_abi", lambda _version: "1.6")

    assert (
        staged_build.build_staged_bundle(
            model,
            output,
            transformer_ref=transformer_ref,
        )
        == output
    )
    expected = (*staged_build._COMPONENTS, *staged_build._REF2VA_COMPONENTS)
    assert [component for component, _kwargs in calls] == [item[0] for item in expected]
    assert all(
        kwargs["transformer_ref_path"] == transformer_ref.resolve() for _component, kwargs in calls
    )

    _header, sections = read_bundle_file(str(output))
    config = json.loads(sections["config.json"])
    assert config["public_workflows"] == ["t2va", "fl2va", "ref2va"]
    assert config["ref2va_supported"] is True
    assert config["ref2va_transformer_fallback_allowed"] is False
    assert config["ref2va_scheduler"]["sigma_grid_points"] == 50
    assert config["ref2va_scheduler"]["transformer_forwards"] == 49
    assert config["ref2va_scheduler"]["video_shift"] == 12.0
    assert config["ref2va_scheduler"]["audio_shift"] == 3.0
    assert config["ref2va_scheduler"]["guidance_distilled"] is True
    assert config["ref2va_limits"]["requires_image_or_video"] is True
    assert config["conditioning"]["text_sequence_profile"] == [1, 1_144, 262_144]
    assert config["conditioning"]["vision_patch_profile"] == [2_040, 4_032, 65_536]
    assert set(config["ref2va_plan_sections"].values()).issubset(
        config["bundle_loading"]["lazy_sections"]
    )
    assert config["ref2va_transformer_ref"]["revision"] == CHECKPOINT_REVISION
    assert str(tmp_path).lower() not in json.dumps(config).lower()


def test_plugin_passes_only_explicit_transformer_ref_to_staged_builder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transformer_ref = tmp_path / COMPONENT_NAME
    transformer_ref.mkdir()
    calls = []
    monkeypatch.setattr(
        staged_build,
        "build_staged_bundle",
        lambda *args, **kwargs: calls.append((args, kwargs)) or Path(args[1]),
    )
    model = tmp_path / "model"
    output = tmp_path / "model.bundle"
    plugin.build_staged_bundle(
        str(model),
        str(output),
        SimpleNamespace(raw={"transformer_ref": transformer_ref}),
        {"_model_dir": str(model)},
        precision="bf16",
        parallel_config=SimpleNamespace(mode="single"),
    )
    assert calls[0][1] == {
        "verbose": False,
        "transformer_ref": transformer_ref.resolve(),
    }
    with pytest.raises(ValueError, match="explicit checkpoint directory"):
        plugin.build_staged_bundle(
            str(model),
            str(output),
            SimpleNamespace(raw={"transformer_ref": True}),
            {"_model_dir": str(model)},
            precision="bf16",
            parallel_config=SimpleNamespace(mode="single"),
        )


def test_fast_h3_ref2va_build_is_exact_61_plan_resumable_and_path_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _model(tmp_path)
    transformer_ref = model / COMPONENT_NAME
    transformer_ref.mkdir()
    adapter = tmp_path / "adapter_model.safetensors"
    adapter.write_bytes(b"adapter")
    output = tmp_path / "h3-complete.bundle"
    calls: list[tuple[str, dict]] = []

    monkeypatch.setattr(
        ref2va_checkpoint,
        "validate_transformer_ref_checkpoint",
        lambda *_args, **_kwargs: _identity(),
    )
    monkeypatch.setattr(
        checkpoint,
        "validate_fast_h3_adapter",
        lambda *_args, **_kwargs: _adapter_identity(),
    )
    monkeypatch.setattr(staged_build, "_adapter_target_partitions", lambda _profile: {"all": ()})

    def build(component, _model_path, plan, **kwargs):
        calls.append((component, kwargs))
        plan.write_bytes(component.encode())

    monkeypatch.setattr(staged_build, "_run_component", build)
    monkeypatch.setattr(staged_build.trt_compat, "configure_backend", lambda **_kwargs: None)
    monkeypatch.setattr(staged_build.trt_compat, "tensorrt_version", lambda: "1.6.1.120")
    monkeypatch.setattr(staged_build.trt_compat, "tensorrt_abi", lambda _version: "1.6")

    assert (
        staged_build.build_staged_bundle(
            model,
            output,
            fast_h3_adapter=adapter,
            transformer_ref=transformer_ref,
        )
        == output
    )
    expected = (*staged_build._FASTH3_COMPONENTS, *staged_build._REF2VA_COMPONENTS)
    assert len(expected) == 61
    assert [component for component, _kwargs in calls] == [item[0] for item in expected]
    assert all(kwargs["adapter_path"] == adapter.resolve() for _component, kwargs in calls)
    assert all(
        kwargs["transformer_ref_path"] == transformer_ref.resolve() for _component, kwargs in calls
    )

    header, sections = read_bundle_file(str(output))
    assert len(header["sections"]) == len(sections) == 63
    config = json.loads(sections["config.json"])
    expected_plan_sections = {section for _component, _filename, section in expected}
    assert set(config["bundle_loading"]["lazy_sections"]) == expected_plan_sections
    assert len(config["bundle_loading"]["lazy_sections"]) == 61
    assert set(config["plan_sha256"]) == {filename for _component, filename, _section in expected}
    assert config["source_revision"] == SOURCE_REVISION
    assert config["public_workflows"] == ["t2va", "fl2va", "ref2va"]
    assert config["transformer_forwards"] == 4
    assert config["ref2va_scheduler"]["transformer_forwards"] == 49
    assert config["ref2va_limits"]["requires_image_or_video"] is True
    assert config["workspace_limit_bytes"] == {
        filename: staged_build._component_workspace_bytes(component, ref2va=True)
        for component, filename, _section in expected
    }
    serialized = json.dumps(config).lower()
    assert str(tmp_path).lower() not in serialized
    assert (
        validate_native_bundle_config(output, source_revision=SOURCE_REVISION)["ref2va_supported"]
        is True
    )

    calls.clear()
    staged_build.build_staged_bundle(
        model,
        output,
        fast_h3_adapter=adapter,
        transformer_ref=transformer_ref,
    )
    assert calls == []

    effective = SimpleNamespace(
        to_effective_dict=lambda: {
            "minimax_h3": {
                "fast_h3_adapter": {
                    "value": str(adapter.resolve()),
                    "source": "session_request",
                },
                "transformer_ref": {
                    "value": str(transformer_ref.resolve()),
                    "source": "session_request",
                },
            }
        }
    )
    sidecar = write_path_free_effective_build_config(effective, output)
    sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
    sidecar_text = json.dumps(sidecar_payload).lower()
    assert str(adapter.resolve()).lower() not in sidecar_text
    assert str(transformer_ref.resolve()).lower() not in sidecar_text
    namespace = sidecar_payload["minimax_h3"]
    assert namespace["fast_h3_adapter"]["value"]["logical_role"] == "fast_h3_adapter"
    assert namespace["transformer_ref"]["value"]["logical_role"] == "transformer_ref"
    assert namespace["build_provenance"]["value"]["source_revision"] == SOURCE_REVISION
