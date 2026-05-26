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
            {"decoder_kv_cache", 1},
            {"diffusion", 2},
        };
        """,
        encoding="utf-8",
    )

    builder = tmp_path / "vision_strategy_builder.cpp"
    builder.write_text(
        """
        static constexpr std::array<std::string_view, 2> kStrategies = {
            "vision_language",
            "segmentation",
        };
        """,
        encoding="utf-8",
    )

    strategies = mod.extract_runtime_strategies_from_cpp_files(
        [trtmc_c, builder],
        {"decoder_kv_cache", "diffusion", "vision_language", "segmentation"},
    )
    assert strategies == {
        "decoder_kv_cache",
        "diffusion",
        "vision_language",
        "segmentation",
    }


def test_discover_runtime_cpp_files_includes_current_strategy_sources(tmp_path: Path):
    mod = _import_checker()

    cpp_path = tmp_path / "src" / "cabi" / "api" / "trtmc_c.cpp"
    cpp_path.parent.mkdir(parents=True)
    cpp_path.write_text('"decoder_kv_cache"', encoding="utf-8")

    models_dir = tmp_path / "src" / "runtime" / "models"
    model_toml = models_dir / "flux" / "MODEL.toml"
    model_toml.parent.mkdir(parents=True)
    model_toml.write_text('runtime_strategies = ["diffusion_flux"]', encoding="utf-8")

    registry_dir = tmp_path / "src" / "runtime" / "registry"
    registry_cpp = registry_dir / "pipeline_factory.cpp"
    registry_dir.mkdir(parents=True)
    registry_cpp.write_text('"text_to_audio_magpie"', encoding="utf-8")

    engine_defs_dir = tmp_path / "engine_defs"
    engine_def = engine_defs_dir / "strategies" / "decoder.py"
    engine_def.parent.mkdir(parents=True)
    engine_def.write_text('runtime_strategy = "torchtrt_decoder"', encoding="utf-8")

    discovered = mod.discover_runtime_cpp_files(
        cpp_path=cpp_path,
        builders_dir=models_dir,
        registry_dir=registry_dir,
        engine_defs_dir=engine_defs_dir,
    )

    assert discovered == [
        cpp_path.resolve(),
        model_toml.resolve(),
        registry_cpp.resolve(),
        engine_def.resolve(),
    ]


def test_validate_matrix_data_requires_exemption_when_no_diff_check():
    mod = _import_checker()
    errors = mod.validate_matrix_data(
        matrix={
            "rwkv_recurrent": {
                "task_strategy": "text_generation_causal",
                "cli_commands": ["run"],
                "runner_class": "TextGenerationCausalRunner",
                "comparator_class": "TextComparator",
                "diff_framework_check_classes": [],
            }
        },
        cpp_runtime_strategies={"rwkv_recurrent"},
        runtime_to_task_strategy={"rwkv_recurrent": "text_generation_causal"},
        diff_checks_by_strategy={},
        runner_classes_by_task={"text_generation_causal": {"TextGenerationCausalRunner"}},
        comparator_classes_by_task={"text_generation_causal": {"TextComparator"}},
    )

    assert any("diff_framework_exemption" in message for message in errors)


def test_validate_matrix_data_detects_runtime_source_mismatch():
    mod = _import_checker()
    errors = mod.validate_matrix_data(
        matrix={
            "vision_language": {
                "task_strategy": "vision_language_generation",
                "cli_commands": ["run"],
                "runner_class": "VisionLanguageRunner",
                "comparator_class": "VisionLanguageComparator",
                "diff_framework_check_classes": ["VLPipelineTest"],
            }
        },
        cpp_runtime_strategies=set(),
        runtime_to_task_strategy={"vision_language": "vision_language_generation"},
        diff_checks_by_strategy={"vision_language": {"VLPipelineTest"}},
        runner_classes_by_task={"vision_language_generation": {"VisionLanguageRunner"}},
        comparator_classes_by_task={"vision_language_generation": {"VisionLanguageComparator"}},
    )

    assert any("runtime builder sources strategy keys missing" in message for message in errors)


def test_validate_matrix_paths_supports_builder_source_extraction(tmp_path: Path):
    mod = _import_checker()

    matrix_path = tmp_path / "tests" / "runtime_strategy_matrix.yaml"
    matrix_path.parent.mkdir(parents=True)
    matrix_path.write_text(
        """
        {
          "runtime_strategies": {
            "decoder_kv_cache": {
              "task_strategy": "text_generation_causal",
              "cli_commands": ["run"],
              "runner_class": "pkg.TextGenerationCausalRunner",
              "comparator_class": "pkg.TextComparator",
              "diff_framework_check_classes": ["LogitDiffTest"]
            },
            "vision_language": {
              "task_strategy": "vision_language_generation",
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

    contracts_path = tmp_path / "tests" / "e2e_harness" / "contracts.py"
    contracts_path.parent.mkdir(parents=True)
    contracts_path.write_text(
        """
RUNTIME_TO_TASK_STRATEGY = {
    "decoder_kv_cache": "text_generation_causal",
    "vision_language": "vision_language_generation",
}
        """,
        encoding="utf-8",
    )

    cpp_path = tmp_path / "src" / "cabi" / "api" / "trtmc_c.cpp"
    cpp_path.parent.mkdir(parents=True)
    cpp_path.write_text(
        """
        static const std::unordered_map<std::string, int> kStrategyFamilies = {
            {"decoder_kv_cache", 1},
            {"vision_language", 2},
        };
        """,
        encoding="utf-8",
    )

    builders_dir = tmp_path / "src" / "runtime" / "builders"
    (builders_dir / "text").mkdir(parents=True)
    (builders_dir / "text" / "text_strategy_builder.cpp").write_text(
        """
        static constexpr std::array<std::string_view, 1> kStrategies = {"decoder_kv_cache"};
        """,
        encoding="utf-8",
    )
    (builders_dir / "vision").mkdir(parents=True)
    (builders_dir / "vision" / "vision_strategy_builder.cpp").write_text(
        """
        static constexpr std::array<std::string_view, 1> kStrategies = {"vision_language"};
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
    runtime_strategies = ["decoder_kv_cache"]
        """,
        encoding="utf-8",
    )
    (diff_checks_dir / "vision_checks.py").write_text(
        """
class VLPipelineTest:
    runtime_strategies = ["vision_language"]
        """,
        encoding="utf-8",
    )

    errors = mod.validate_matrix_paths(
        matrix_path=matrix_path,
        cpp_path=cpp_path,
        builders_dir=builders_dir,
        contracts_path=contracts_path,
        diff_checks_dir=diff_checks_dir,
        runners_dir=runners_dir,
        comparators_dir=comparators_dir,
    )
    assert errors == []
