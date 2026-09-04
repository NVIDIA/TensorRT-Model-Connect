# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_model_connect.model_support import (
    FamilySupport,
    ModelMetadata,
    family_support,
    load_model_metadata,
    resolve_family,
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
    with pytest.raises(ValueError, match="multiple families.*alpha, beta"):
        resolve_family(metadata)


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


def test_qwen_image_default_task_is_checkpoint_owned() -> None:
    _, generation = resolve_family(
        ModelMetadata({}, {"_class_name": "QwenImagePipeline"})
    )
    _, editing = resolve_family(
        ModelMetadata({}, {"_class_name": "QwenImageEditPipeline"})
    )

    assert generation.default_task == "image_generation"
    assert editing.default_task == "image_edit"


def test_eagle_vlm_default_task_is_checkpoint_owned() -> None:
    _, embedding = resolve_family(
        ModelMetadata(
            {
                "model_type": "llama_nemotron_vl",
                "architectures": ["LlamaNemotronVLModel"],
            },
            {},
        )
    )
    _, reranking = resolve_family(
        ModelMetadata(
            {
                "model_type": "llama_nemotron_vl_rerank",
                "architectures": ["LlamaNemotronVLForSequenceClassification"],
            },
            {},
        )
    )

    assert embedding.default_task == "embedding"
    assert reranking.default_task == "reranking"


def test_rootless_moge_sentinel_does_not_steal_a_config_owned_checkpoint() -> None:
    family, _ = resolve_family(
        ModelMetadata(
            {"model_type": "gpt2"},
            {},
            ("config.json", "model.pt"),
        )
    )
    assert family == "gpt2"

    family, support = resolve_family(ModelMetadata({}, {}, ("model.pt",)))
    assert family == "moge"
    assert support.default_task == "monocular_geometry"


def test_qwen38_marker_has_one_owner() -> None:
    family, support = resolve_family(
        ModelMetadata(
            {
                "model_type": "qwen3_5",
                "text_config": {"output_gate_type": "sigmoid"},
            },
            {},
        )
    )
    assert family == "qwen3_8"
    assert support.default_task == "text_generation"


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
            ModelMetadata(
                {"model_name": "Lance", "organization": "bytedance-research"}, {}
            ),
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
            ModelMetadata(
                {}, {}, ("sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt")
            ),
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
