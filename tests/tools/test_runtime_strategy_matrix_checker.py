# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for tools/check_runtime_strategy_matrix.py.

Trace: ARCH-E2E-001, UD-E2E-STRATEGY-MATRIX
Intent: Validate runtime strategy matrix checker C++ extraction, manifest scanning, and gap detection
Preconditions: Synthetic C++ files and manifest directories with strategy strings are created
Postconditions: Checker correctly extracts strategies from C++ source and identifies coverage gaps
"""

from __future__ import annotations

import importlib
from pathlib import Path


def _import_checker():
    return importlib.import_module("check_runtime_strategy_matrix")


def test_extract_runtime_strategies_from_cpp_filters_to_known_candidates(
    tmp_path: Path,
):
    mod = _import_checker()

    cpp = tmp_path / "audio_strategy_builder.cpp"
    cpp.write_text(
        """
        static constexpr std::array<std::string_view, 4> kStrategies = {
            "text_to_audio",
            "speech_to_text",
            "speech_to_speech",
            "omni_multimodal",
        };
        if (bundle_port.has_section("engine_plan")) {}
        if (bundle_port.has_section("codec_engine_plan")) {}
        """,
        encoding="utf-8",
    )

    strategies = mod.extract_runtime_strategies_from_cpp(
        cpp,
        {"text_to_audio", "speech_to_text", "speech_to_speech", "omni_multimodal", "diffusion"},
    )
    assert strategies == {
        "text_to_audio",
        "speech_to_text",
        "speech_to_speech",
        "omni_multimodal",
    }


def test_extract_runtime_strategies_from_cpp_files_aggregates_entrypoint_and_builders(
    tmp_path: Path,
):
    mod = _import_checker()

    trtmc_c = tmp_path / "trtmc_c.cpp"
    trtmc_c.write_text(
        """
        static const std::unordered_map<std::string, int> kStrategyFamilies = {
            {"qwen_decoder_kv_cache", 1},
            {"diffusion", 2},
        };
        """,
        encoding="utf-8",
    )

    builder = tmp_path / "vision_strategy_builder.cpp"
    builder.write_text(
        """
        static constexpr std::array<std::string_view, 2> kStrategies = {
            "qwen_vl_vision_language",
            "segformer_segmentation",
        };
        """,
        encoding="utf-8",
    )

    strategies = mod.extract_runtime_strategies_from_cpp_files(
        [trtmc_c, builder],
        {
            "qwen_decoder_kv_cache",
            "diffusion",
            "qwen_vl_vision_language",
            "segformer_segmentation",
        },
    )
    assert strategies == {
        "qwen_decoder_kv_cache",
        "diffusion",
        "qwen_vl_vision_language",
        "segformer_segmentation",
    }


def test_extract_runtime_strategies_from_model_manifests(tmp_path: Path):
    mod = _import_checker()

    model_dir = tmp_path / "src" / "runtime" / "models"
    (model_dir / "qwen").mkdir(parents=True)
    (model_dir / "qwen" / "MODEL.toml").write_text(
        'runtime_strategies = ["qwen_decoder_kv_cache"]\n',
        encoding="utf-8",
    )
    (model_dir / "media_runtime").mkdir(parents=True)
    (model_dir / "media_runtime" / "MODEL.toml").write_text(
        'runtime_strategy = "diffusion_primary"\n',
        encoding="utf-8",
    )

    assert mod.extract_runtime_strategies_from_model_manifests(model_dir) == {
        "qwen_decoder_kv_cache",
        "diffusion_primary",
    }


def test_model_local_e2e_plugin_discovery(tmp_path: Path):
    mod = _import_checker()

    runners_dir = (
        tmp_path / "tests" / "e2e" / "models" / "example_decoder" / "e2e_plugins" / "runners"
    )
    runners_dir.mkdir(parents=True)
    (runners_dir / "text_generation.py").write_text(
        """
class TextGenerationCausalRunner:
    def strategy_name(self):
        return "text_generation_causal"
        """,
        encoding="utf-8",
    )
    comparators_dir = (
        tmp_path / "tests" / "e2e" / "models" / "example_decoder" / "e2e_plugins" / "comparators"
    )
    comparators_dir.mkdir(parents=True)
    (comparators_dir / "text.py").write_text(
        """
class TextComparator:
    def task_strategy(self):
        return "text_generation_causal"
        """,
        encoding="utf-8",
    )

    models_dir = tmp_path / "tests" / "e2e" / "models"
    assert mod.extract_runner_classes_by_task_strategy(models_dir) == {
        "text_generation_causal": {"TextGenerationCausalRunner"},
    }
    assert mod.extract_comparator_classes_by_task_strategy(models_dir) == {
        "text_generation_causal": {"TextComparator"},
    }


def test_model_local_e2e_entrypoint_wrapper_discovery(tmp_path: Path):
    mod = _import_checker()

    plugin_root = tmp_path / "tests" / "e2e" / "models" / "example_speech" / "e2e_plugins"
    runners_dir = plugin_root / "runners"
    runners_dir.mkdir(parents=True)
    (runners_dir / "audio.py").write_text(
        """
class RuntimeSpecificRunner:
    @property
    def strategy_name(self):
        return "example_speech_runtime"
        """,
        encoding="utf-8",
    )
    (plugin_root / "runner.py").write_text(
        """
class ActiveSpeechRunner:
    @property
    def strategy_name(self):
        return "speech_to_text"
        """,
        encoding="utf-8",
    )

    models_dir = tmp_path / "tests" / "e2e" / "models"
    assert mod.extract_runner_classes_by_task_strategy(models_dir) == {
        "example_speech_runtime": {"RuntimeSpecificRunner"},
        "speech_to_text": {"ActiveSpeechRunner"},
    }


def test_validate_matrix_data_requires_exemption_when_no_diff_check():
    mod = _import_checker()
    errors = mod.validate_matrix_data(
        matrix={
            "unit_recurrent": {
                "task_strategy": "text_generation_causal",
                "cli_commands": ["run"],
                "runner_class": "TextGenerationCausalRunner",
                "comparator_class": "TextComparator",
                "diff_framework_check_classes": [],
            }
        },
        cpp_runtime_strategies={"unit_recurrent"},
        runtime_to_task_strategy={"unit_recurrent": "text_generation_causal"},
        diff_check_classes=set(),
        runner_classes_by_task={"text_generation_causal": {"TextGenerationCausalRunner"}},
        comparator_classes_by_task={"text_generation_causal": {"TextComparator"}},
    )

    assert any("diff_framework_exemption" in message for message in errors)


def test_validate_matrix_data_detects_runtime_source_mismatch():
    mod = _import_checker()
    errors = mod.validate_matrix_data(
        matrix={
            "qwen_vl_vision_language": {
                "task_strategy": "vision_language_generation",
                "cli_commands": ["run"],
                "runner_class": "VisionLanguageRunner",
                "comparator_class": "VisionLanguageComparator",
                "diff_framework_check_classes": ["VLPipelineTest"],
            }
        },
        cpp_runtime_strategies={"qwen_vl_vision_language", "future_runtime_strategy"},
        runtime_to_task_strategy={"qwen_vl_vision_language": "vision_language_generation"},
        diff_check_classes={"VLPipelineTest"},
        runner_classes_by_task={"vision_language_generation": {"VisionLanguageRunner"}},
        comparator_classes_by_task={"vision_language_generation": {"VisionLanguageComparator"}},
    )

    assert any("missing runtime strategies from runtime sources" in message for message in errors)


def test_validate_matrix_data_rejects_source_less_matrix_row():
    mod = _import_checker()
    errors = mod.validate_matrix_data(
        matrix={
            "retired_runtime": {
                "task_strategy": "text_generation_causal",
                "cli_commands": ["run"],
                "runner_class": "TextGenerationCausalRunner",
                "comparator_class": "TextComparator",
                "diff_framework_check_classes": [],
                "diff_framework_exemption": "No active diff check.",
                "performance_mode": "decode",
            }
        },
        cpp_runtime_strategies=set(),
        runtime_to_task_strategy={},
        diff_check_classes=set(),
        runner_classes_by_task={"text_generation_causal": {"TextGenerationCausalRunner"}},
        comparator_classes_by_task={"text_generation_causal": {"TextComparator"}},
    )

    assert any("absent from runtime sources and E2E manifests" in message for message in errors)


def test_validate_matrix_paths_supports_builder_source_extraction(tmp_path: Path):
    mod = _import_checker()

    matrix_path = tmp_path / "tests" / "runtime_strategy_matrix.yaml"
    matrix_path.parent.mkdir(parents=True)
    matrix_path.write_text(
        """
        {
          "runtime_strategies": {
            "qwen_decoder_kv_cache": {
              "task_strategy": "text_generation_causal",
              "performance_mode": "decode",
              "cli_commands": ["run"],
              "runner_class": "pkg.TextGenerationCausalRunner",
              "comparator_class": "pkg.TextComparator",
              "diff_framework_check_classes": ["LogitDiffTest"]
            },
            "qwen_vl_vision_language": {
              "task_strategy": "vision_language_generation",
              "performance_mode": "enc_dec",
              "cli_commands": ["run"],
              "runner_class": "pkg.VisionLanguageRunner",
              "comparator_class": "pkg.VisionLanguageComparator",
              "diff_framework_check_classes": ["VLPipelineTest"]
            }
          }
        }
        """,
        encoding="utf-8",
    )

    e2e_models_dir = tmp_path / "tests" / "e2e" / "models"
    text_manifest_dir = e2e_models_dir / "text" / "manifests"
    text_manifest_dir.mkdir(parents=True)
    (text_manifest_dir / "text.json").write_text(
        """
{
  "name": "text",
  "hf_id": "unit/text",
  "family": "text",
  "runtime_strategy": "qwen_decoder_kv_cache",
  "task_strategy": "text_generation_causal"
}
        """,
        encoding="utf-8",
    )
    vl_manifest_dir = e2e_models_dir / "vl" / "manifests"
    vl_manifest_dir.mkdir(parents=True)
    (vl_manifest_dir / "vl.json").write_text(
        """
{
  "name": "vl",
  "hf_id": "unit/vl",
  "family": "vl",
  "runtime_strategy": "qwen_vl_vision_language",
  "task_strategy": "vision_language_generation"
}
        """,
        encoding="utf-8",
    )

    cpp_path = tmp_path / "src" / "cabi" / "api" / "trtmc_c.cpp"
    cpp_path.parent.mkdir(parents=True)
    cpp_path.write_text(
        """
        static const std::unordered_map<std::string, int> kStrategyFamilies = {
            {"qwen_decoder_kv_cache", 1},
            {"qwen_vl_vision_language", 2},
        };
        """,
        encoding="utf-8",
    )

    builders_dir = tmp_path / "src" / "runtime" / "builders"
    (builders_dir / "text").mkdir(parents=True)
    (builders_dir / "text" / "text_strategy_builder.cpp").write_text(
        """
        static constexpr std::array<std::string_view, 1> kStrategies = {"qwen_decoder_kv_cache"};
        """,
        encoding="utf-8",
    )
    (builders_dir / "vision").mkdir(parents=True)
    (builders_dir / "vision" / "vision_strategy_builder.cpp").write_text(
        """
        static constexpr std::array<std::string_view, 1> kStrategies = {"qwen_vl_vision_language"};
        """,
        encoding="utf-8",
    )

    runners_dir = tmp_path / "tests" / "e2e_harness" / "runners"
    runners_dir.mkdir(parents=True)
    (runners_dir / "text_generation.py").write_text(
        """
class TextGenerationCausalRunner:
    def strategy_name(self):
        return "text_generation_causal"
        """,
        encoding="utf-8",
    )
    (runners_dir / "vision_language.py").write_text(
        """
class VisionLanguageRunner:
    def strategy_name(self):
        return "vision_language_generation"
        """,
        encoding="utf-8",
    )

    comparators_dir = tmp_path / "tests" / "e2e_harness" / "comparators"
    comparators_dir.mkdir(parents=True)
    (comparators_dir / "text.py").write_text(
        """
class TextComparator:
    def task_strategy(self):
        return "text_generation_causal"
        """,
        encoding="utf-8",
    )
    (comparators_dir / "vision_language.py").write_text(
        """
class VisionLanguageComparator:
    def task_strategy(self):
        return "vision_language_generation"
        """,
        encoding="utf-8",
    )

    diff_checks_dir = tmp_path / "tools" / "diff_framework" / "checks"
    diff_checks_dir.mkdir(parents=True)
    (diff_checks_dir / "text_checks.py").write_text(
        """
class LogitDiffTest:
    name = "logit_diff"
        """,
        encoding="utf-8",
    )
    (diff_checks_dir / "vision_checks.py").write_text(
        """
class VLPipelineTest:
    name = "vl_pipeline"
        """,
        encoding="utf-8",
    )

    errors = mod.validate_matrix_paths(
        matrix_path=matrix_path,
        cpp_path=cpp_path,
        builders_dir=builders_dir,
        runtime_registry_path=tmp_path / "missing_pipeline_factory.cpp",
        runtime_models_dir=tmp_path / "missing_runtime_models",
        torchtrt_strategies_dir=tmp_path / "missing_torchtrt_strategies",
        e2e_models_dir=e2e_models_dir,
        diff_checks_dir=diff_checks_dir,
        runners_dir=runners_dir,
        comparators_dir=comparators_dir,
    )
    assert errors == []
