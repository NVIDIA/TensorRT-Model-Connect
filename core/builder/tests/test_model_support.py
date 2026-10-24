# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_model_connect.model_support import (
    AmbiguousFamilyError,
    FamilyResolutionError,
    FamilySupport,
    IncompleteMetadataError,
    ModelMetadata,
    decide_support,
    family_support,
    load_model_metadata,
    resolve_family,
    support_from_json,
)


def test_family_support_matches_exact_model_architecture_or_pipeline() -> None:
    describe = family_support(
        model_types=("example-model",),
        architectures=("ExampleForGeneration",),
        pipeline_classes=("ExamplePipeline",),
        required_files=("owner/config.yaml",),
        tasks=("generation", "editing"),
        default_task="generation",
    )

    expected = FamilySupport(("generation", "editing"), "generation")
    assert describe(ModelMetadata({"model_type": "example_model"}, {})) == expected
    assert describe(ModelMetadata({"architectures": ["ExampleForGeneration"]}, {})) == expected
    assert describe(ModelMetadata({}, {"_class_name": "ExamplePipeline"})) == expected
    assert describe(ModelMetadata({}, {}, ("owner/config.yaml",))) == expected
    assert describe(ModelMetadata({"model_type": "other"}, {})) is None


def test_family_support_rejects_invalid_declarations() -> None:
    with pytest.raises(ValueError, match="model identity"):
        family_support(tasks=("generation",), default_task="generation")
    with pytest.raises(ValueError, match="default_task"):
        FamilySupport(tasks=("generation",), default_task="editing")
    with pytest.raises(ValueError, match="lowercase identifiers"):
        FamilySupport(tasks=("not-valid",), default_task="not-valid")
    with pytest.raises(ValueError, match="default_precision"):
        FamilySupport(tasks=("generation",), default_task="generation", default_precision="int8")


def test_load_model_metadata_reads_only_standard_identity_files(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        '{"model_type":"gpt2","architectures":["GPT2LMHeadModel"]}',
        encoding="utf-8",
    )
    (tmp_path / "model_index.json").write_text(
        '{"_class_name":"ExamplePipeline"}', encoding="utf-8"
    )

    metadata = load_model_metadata(tmp_path)

    assert metadata.model_type == "gpt2"
    assert metadata.architectures == ("GPT2LMHeadModel",)
    assert metadata.pipeline_class == "ExamplePipeline"
    assert metadata.files == ("config.json", "model_index.json")


def test_load_model_metadata_keeps_exact_snapshot_files_without_root_json(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "manifest.ocdbt").write_text("owned", encoding="utf-8")

    metadata = load_model_metadata(tmp_path)

    assert metadata.config == {}
    assert metadata.model_index == {}
    assert metadata.files == ("checkpoint/manifest.ocdbt",)


def test_resolve_family_requires_exactly_one_match(monkeypatch) -> None:
    metadata = ModelMetadata({"model_type": "owned"}, {})
    directories = [Path("alpha"), Path("beta")]
    modules = {
        "families.alpha.support": SimpleNamespace(
            describe=lambda value: FamilySupport(("generation",), "generation")
        ),
        "families.beta.support": SimpleNamespace(describe=lambda value: None),
    }
    imported: list[str] = []

    def fake_import(name: str):
        imported.append(name)
        return modules[name]

    support_module = importlib.import_module("tensorrt_model_connect.model_support")
    monkeypatch.setattr(support_module, "_family_directories", lambda: directories)
    monkeypatch.setattr(importlib, "import_module", fake_import)

    assert resolve_family(metadata) == (
        "alpha",
        FamilySupport(("generation",), "generation"),
    )
    assert imported == ["families.alpha.support", "families.beta.support"]

    modules["families.alpha.support"] = SimpleNamespace(describe=lambda value: None)
    with pytest.raises(ValueError, match="no family supports"):
        resolve_family(metadata)

    modules["families.alpha.support"] = SimpleNamespace(
        describe=lambda value: FamilySupport(("generation",), "generation")
    )
    modules["families.beta.support"] = modules["families.alpha.support"]
    with pytest.raises(AmbiguousFamilyError, match="multiple families.*alpha, beta") as error:
        resolve_family(metadata)
    assert error.value.matches == (
        ("alpha", FamilySupport(("generation",), "generation")),
        ("beta", FamilySupport(("generation",), "generation")),
    )


def test_explicit_family_selects_only_that_compatible_owner(monkeypatch) -> None:
    metadata = ModelMetadata({"model_type": "shared"}, {})
    directories = [Path("alpha"), Path("beta")]
    modules = {
        "families.alpha.support": SimpleNamespace(
            describe=lambda value: FamilySupport(("generation",), "generation")
        ),
        "families.beta.support": SimpleNamespace(
            describe=lambda value: FamilySupport(("editing",), "editing")
        ),
    }
    imported = []

    support_module = importlib.import_module("tensorrt_model_connect.model_support")
    monkeypatch.setattr(support_module, "_family_directories", lambda: directories)

    def fake_import(name: str):
        imported.append(name)
        return modules[name]

    monkeypatch.setattr(importlib, "import_module", fake_import)

    assert resolve_family(metadata, "beta") == (
        "beta",
        FamilySupport(("editing",), "editing"),
    )
    assert imported == ["families.beta.support"]


def test_explicit_family_rejects_unknown_or_incompatible_selection(monkeypatch) -> None:
    metadata = ModelMetadata({"model_type": "shared"}, {})
    directories = [Path("alpha")]
    support_module = importlib.import_module("tensorrt_model_connect.model_support")
    monkeypatch.setattr(support_module, "_family_directories", lambda: directories)

    with pytest.raises(FamilyResolutionError, match="unknown family 'missing'.*omit --family"):
        resolve_family(metadata, "missing")

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: SimpleNamespace(describe=lambda value: None),
    )
    with pytest.raises(
        FamilyResolutionError, match="family 'alpha' does not support.*omit --family"
    ):
        resolve_family(metadata, "alpha")


@pytest.mark.parametrize(
    ("model_type", "expected_family"),
    [
        ("gpt_neox", "gpt_neox"),
        ("phi_moe", "phi_moe"),
        ("qwen2_moe", "qwen_moe"),
        ("sam2", "sam2"),
    ],
)
def test_specific_model_identity_never_falls_through_to_a_broad_family(
    model_type: str, expected_family: str
) -> None:
    family, _ = resolve_family(ModelMetadata({"model_type": model_type}, {}))
    assert family == expected_family


def test_rootless_moge_sentinel_does_not_steal_a_config_owned_checkpoint() -> None:
    family, _ = resolve_family(
        ModelMetadata(
            {"model_type": "gpt2"},
            {},
            ("config.json", "model.pt"),
        )
    )
    assert family == "gpt2"

    family, _ = resolve_family(ModelMetadata({}, {}, ("model.pt",)))
    assert family == "moge"


def test_qwen38_marker_has_one_owner() -> None:
    family, _ = resolve_family(
        ModelMetadata(
            {
                "model_type": "qwen3_5",
                "text_config": {"output_gate_type": "sigmoid"},
            },
            {},
        )
    )
    assert family == "qwen3_8"


@pytest.mark.parametrize(
    ("metadata", "expected_family"),
    [
        (
            ModelMetadata(
                {
                    "model_type": "t5",
                    "architectures": ["ChronosBoltModelForForecasting"],
                },
                {},
            ),
            "chronos_bolt",
        ),
        (
            ModelMetadata(
                {"model_type": "t5", "architectures": ["T5ForConditionalGeneration"]},
                {},
            ),
            "t5",
        ),
        (
            ModelMetadata({"model_name": "Lance", "organization": "bytedance-research"}, {}),
            "lance",
        ),
        (
            ModelMetadata(
                {
                    "_rnnt_merge_info": {},
                    "model": {"speech_generation": {}, "stt": {}},
                },
                {},
            ),
            "nemotron_voicechat",
        ),
        (
            ModelMetadata({"architecture": "vit_base_patch16_224"}, {}),
            "timm_vit",
        ),
        (
            ModelMetadata({"architecture": "vit_small_patch16_dinov3_qkvb"}, {}),
            "dinov3",
        ),
        (
            ModelMetadata(
                {},
                {},
                (
                    "config.yaml",
                    "dit/sana_wm_1600m_720p.safetensors",
                    "refiner/transformer/config.json",
                ),
            ),
            "sana_wm",
        ),
        (
            ModelMetadata({}, {}, ("sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt")),
            "sam2",
        ),
        (
            ModelMetadata({}, {}, ("magpie_tts_multilingual_357m.nemo",)),
            "magpie_tts",
        ),
        (
            ModelMetadata(
                {},
                {},
                (
                    "ELF-B-owt.yml",
                    "checkpoint_0/_CHECKPOINT_METADATA",
                    "checkpoint_0/manifest.ocdbt",
                ),
            ),
            "elf_flow",
        ),
    ],
)
def test_family_owned_exact_metadata_shapes_resolve_without_priority(
    metadata: ModelMetadata, expected_family: str
) -> None:
    family, _ = resolve_family(metadata)
    assert family == expected_family


def _write_rules(tmp_path: Path, rules: list, name: str = "support") -> Path:
    support_py = tmp_path / f"{name}.py"
    support_py.write_text("", encoding="utf-8")
    (tmp_path / f"{name}.json").write_text(
        json.dumps({"schema_version": 1, "rules": rules}), encoding="utf-8"
    )
    return support_py


GENERATION = {"tasks": ["generation"], "default_task": "generation"}


def test_support_rules_use_the_first_matching_rule_and_reject_stops(tmp_path) -> None:
    describe = support_from_json(
        _write_rules(
            tmp_path,
            [
                {"if": {"/config/variant": {"exists": True}}, "then": "reject"},
                {
                    "if": {"/config/model_type": {"in": ["alpha"]}, "files_all": ["extra.json"]},
                    "then": {"tasks": ["embedding"], "default_task": "embedding", "default_precision": "bf16"},
                },
                {"if": {"/config/model_type": {"in": ["alpha"]}}, "then": GENERATION},
            ],
        )
    )

    plain = describe(ModelMetadata({"model_type": "alpha"}, {}, ("config.json",)))
    extra = describe(ModelMetadata({"model_type": "alpha"}, {}, ("config.json", "extra.json")))
    rejected = describe(ModelMetadata({"model_type": "alpha", "variant": 1}, {}, ("config.json",)))

    assert plain == FamilySupport(("generation",), "generation")
    assert extra == FamilySupport(("embedding",), "embedding", "bf16")
    assert rejected is None
    assert describe(ModelMetadata({"model_type": "Alpha"}, {}, ("config.json",))) is None


@pytest.mark.parametrize(
    ("test", "value", "expected"),
    [
        ({"in": ["a", "b"]}, "b", True),
        ({"in": ["a"]}, ["a"], False),
        ({"contains_any": ["X"]}, ["Y", "X"], True),
        ({"contains_any": ["X"]}, "X", False),
        ({"equals": {"k": 1}}, {"k": 1}, True),
        ({"equals": 1}, 1.0, True),
        ({"equals": {"k": 1}}, {"k": True}, False),
        ({"equals": [1, "a"]}, [True, "a"], False),
        ({"exists": False}, None, True),
        ({"prefix": "yo"}, "yolo", True),
        ({"prefix": "yo"}, 3, False),
        ({"type": "object"}, {}, True),
        ({"type": "number"}, True, False),
        ({"nonempty": True}, {"a": 1}, True),
        ({"nonempty": False}, [], True),
    ],
)
def test_support_rule_tests_compare_exact_values(tmp_path, test, value, expected) -> None:
    describe = support_from_json(
        _write_rules(tmp_path, [{"if": {"/config/field": test}, "then": GENERATION}])
    )
    config = {} if value is None else {"field": value}

    assert (describe(ModelMetadata(config, {}, ("config.json",))) is not None) is expected


def test_support_rules_read_repo_identity_adapter_and_nested_pointers(tmp_path) -> None:
    describe = support_from_json(
        _write_rules(
            tmp_path,
            [
                {"if": {"/repo_id": {"in": ["owner/model"]}}, "then": GENERATION},
                {
                    "if": {
                        "/adapter_config/base_model_name_or_path": {"equals": "owner/base"},
                        "/config/a~1b/c": {"equals": 1},
                    },
                    "then": GENERATION,
                },
            ],
        )
    )

    assert describe(ModelMetadata({}, {}, (), repo_id="owner/model")) is not None
    assert describe(
        ModelMetadata(
            {"a/b": {"c": 1}},
            {},
            ("config.json", "adapter_config.json"),
            adapter_config={"base_model_name_or_path": "owner/base"},
        )
    ) is not None
    assert describe(ModelMetadata({}, {}, (), repo_id="owner/other")) is None


def test_support_rules_refuse_to_decide_when_a_listed_file_was_not_read(tmp_path) -> None:
    describe = support_from_json(
        _write_rules(
            tmp_path,
            [{"if": {"/config/text_config/gate": {"exists": False}}, "then": GENERATION}],
        )
    )

    with pytest.raises(IncompleteMetadataError, match="config.json"):
        describe(ModelMetadata(None, {}, ("config.json",)))
    assert describe(ModelMetadata(None, {}, ("model.safetensors",))) is not None


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ({"schema_version": 2, "rules": []}, "schema_version"),
        ({"schema_version": 1, "rules": [{"if": {"/config/x": {"like": "a"}}, "then": GENERATION}]}, "unknown test"),
        ({"schema_version": 1, "rules": [{"if": {"model_type": {"in": ["a"]}}, "then": GENERATION}]}, "unknown condition"),
        ({"schema_version": 1, "rules": [{"if": {"/weights/x": {"exists": True}}, "then": GENERATION}]}, "unknown document"),
        ({"schema_version": 1, "rules": [{"if": {}, "then": GENERATION}]}, "at least one condition"),
        ({"schema_version": 1, "rules": [{"if": {"/repo_id": {"in": ["a/b"]}}, "then": {"tasks": ["x"], "default_task": "y"}}]}, "default_task"),
    ],
)
def test_support_rules_reject_invalid_files(tmp_path, document, message) -> None:
    (tmp_path / "support.py").write_text("", encoding="utf-8")
    (tmp_path / "support.json").write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        support_from_json(tmp_path / "support.py")


def test_decide_support_reports_supported_unsupported_conflict_and_incomplete(monkeypatch) -> None:
    support_module = importlib.import_module("tensorrt_model_connect.model_support")
    alpha = family_support(model_types=("alpha",), tasks=("generation",), default_task="generation")

    def incomplete(metadata):
        raise IncompleteMetadataError("config.json was not read")

    modules = {
        "families.alpha.support": SimpleNamespace(describe=alpha),
        "families.beta.support": SimpleNamespace(describe=lambda metadata: None),
    }
    monkeypatch.setattr(
        support_module, "_family_directories", lambda: [Path("alpha"), Path("beta")]
    )
    monkeypatch.setattr(importlib, "import_module", lambda name: modules[name])

    supported = decide_support(ModelMetadata({"model_type": "alpha"}, {}))
    unsupported = decide_support(ModelMetadata({"model_type": "other"}, {}))
    modules["families.beta.support"] = SimpleNamespace(describe=alpha)
    conflict = decide_support(ModelMetadata({"model_type": "alpha"}, {}))
    modules["families.beta.support"] = SimpleNamespace(describe=incomplete)
    pending = decide_support(ModelMetadata({"model_type": "alpha"}, {}))

    assert (supported.status, supported.family) == ("supported", "alpha")
    assert supported.support == FamilySupport(("generation",), "generation")
    assert (unsupported.status, unsupported.family) == ("unsupported", None)
    assert (conflict.status, conflict.matches) == ("conflict", ("alpha", "beta"))
    assert (pending.status, pending.family) == ("incomplete", None)
    assert pending.missing == ("beta",)
    with pytest.raises(IncompleteMetadataError, match="beta"):
        resolve_family(ModelMetadata({"model_type": "alpha"}, {}))


def test_load_model_metadata_reads_adapter_and_remote_identity(tmp_path) -> None:
    (tmp_path / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "owner/base"}', encoding="utf-8"
    )

    metadata = load_model_metadata(tmp_path, repo_id="owner/adapter", revision="abc123")

    assert metadata.adapter_config == {"base_model_name_or_path": "owner/base"}
    assert (metadata.repo_id, metadata.revision) == ("owner/adapter", "abc123")
