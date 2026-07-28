# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for tools/check_runtime_strategy_matrix.py.

Trace: ARCH-E2E-001, UD-E2E-STRATEGY-MATRIX
Intent: Validate C++/CLI extraction, active runner contracts, manifest scanning, and gap detection
Preconditions: Synthetic runtime, CLI, runner, and manifest sources are created
Postconditions: Checker identifies strategy and executable-command coverage gaps
"""

from __future__ import annotations

import importlib
import json
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
        "text_generation_causal": {"runners.text_generation.TextGenerationCausalRunner"},
    }
    assert mod.extract_comparator_classes_by_task_strategy(models_dir) == {
        "text_generation_causal": {"comparators.text.TextComparator"},
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
        "example_speech_runtime": {"runners.audio.RuntimeSpecificRunner"},
        "speech_to_text": {"runner.ActiveSpeechRunner"},
    }


def test_extract_native_cli_commands_from_parser_table(tmp_path: Path):
    mod = _import_checker()
    cli_args = tmp_path / "args.cpp"
    cli_args.write_text(
        """
static const char* known_cmds[] = {
    "run", "segment", "segment-prompted", "encode", nullptr
};
        """,
        encoding="utf-8",
    )

    assert mod.extract_native_cli_commands(cli_args) == {
        "run",
        "segment",
        "segment-prompted",
        "encode",
    }


def test_extract_runtime_cli_commands_from_model_owned_command_builder(tmp_path: Path):
    mod = _import_checker()
    models_dir = tmp_path / "tests" / "e2e" / "models"
    owner_dir = models_dir / "sam3"
    manifest_dir = owner_dir / "manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "sam3.json").write_text(
        """
{
  "family": "sam3",
  "runtime_strategy": "sam3_prompted_segmentation",
  "task_strategy": "prompted_segmentation"
}
        """,
        encoding="utf-8",
    )
    plugin_dir = owner_dir / "e2e_plugins"
    plugin_dir.mkdir()
    (plugin_dir / "commands.py").write_text(
        """
def build_command(binary, bundle):
    return [binary, "segment-prompted", bundle, "--prompt", "ear"]
        """,
        encoding="utf-8",
    )
    (plugin_dir / "runner.py").write_text(
        """
from .commands import build_command

class PromptedSegmentationRunner:
    def run_stage(self, binary, bundle):
        return build_command(binary, bundle)

runner = PromptedSegmentationRunner()
        """,
        encoding="utf-8",
    )

    assert mod.extract_runtime_cli_commands_from_e2e_plugins(
        {"sam3_prompted_segmentation": {"runner_class": "runner.PromptedSegmentationRunner"}},
        models_dir,
        {"segment", "segment-prompted"},
    ) == {"sam3_prompted_segmentation": {"segment-prompted"}}


def _extract_conditional_runner_commands(
    tmp_path: Path,
    *,
    condition: str,
    runner_class: str,
) -> dict[str, set[str]]:
    mod = _import_checker()
    models_dir = tmp_path / "tests" / "e2e" / "models"
    owner_dir = models_dir / "conditional"
    manifest_dir = owner_dir / "manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "conditional.json").write_text(
        """
{
  "family": "conditional",
  "runtime_strategy": "conditional_runtime",
  "task_strategy": "text_generation_causal"
}
        """,
        encoding="utf-8",
    )
    plugin_dir = owner_dir / "e2e_plugins"
    plugin_dir.mkdir()
    (plugin_dir / "runner.py").write_text(
        f"""
import os

select_second = os.environ.get("SELECT_SECOND") == "1"

class FirstRunner:
    def run_stage(self, binary, bundle):
        return [binary, "run", bundle]

class SecondRunner:
    def run_stage(self, binary, bundle):
        return [binary, "generate-audio", bundle]

runner = FirstRunner()
if {condition}:
    runner = SecondRunner()
        """,
        encoding="utf-8",
    )

    return mod.extract_runtime_cli_commands_from_e2e_plugins(
        {
            "conditional_runtime": {
                "runner_class": f"runner.{runner_class}",
            }
        },
        models_dir,
        {"generate-audio", "run"},
    )


def test_runtime_command_discovery_uses_true_branch_runner_rebinding(
    tmp_path: Path,
):
    assert _extract_conditional_runner_commands(
        tmp_path,
        condition="True",
        runner_class="SecondRunner",
    ) == {"conditional_runtime": {"generate-audio"}}


def test_runtime_command_discovery_ignores_false_branch_runner_rebinding(
    tmp_path: Path,
):
    assert _extract_conditional_runner_commands(
        tmp_path,
        condition="False",
        runner_class="FirstRunner",
    ) == {"conditional_runtime": {"run"}}


def test_runtime_command_discovery_fails_closed_for_unknown_runner_rebinding(
    tmp_path: Path,
):
    assert _extract_conditional_runner_commands(
        tmp_path,
        condition="select_second",
        runner_class="FirstRunner",
    ) == {"conditional_runtime": set()}


def test_runtime_command_discovery_does_not_borrow_sibling_runner_commands(
    tmp_path: Path,
):
    mod = _import_checker()
    models_dir = tmp_path / "tests" / "e2e" / "models"
    owner_dir = models_dir / "dual_contract"
    manifest_dir = owner_dir / "manifests"
    manifest_dir.mkdir(parents=True)
    for runtime_strategy, task_strategy in (
        ("dual_embedding", "embedding"),
        ("dual_reranking", "reranking"),
    ):
        (manifest_dir / f"{runtime_strategy}.json").write_text(
            f"""
{{
  "family": "dual_contract",
  "runtime_strategy": "{runtime_strategy}",
  "task_strategy": "{task_strategy}"
}}
            """,
            encoding="utf-8",
        )
    plugin_dir = owner_dir / "e2e_plugins"
    plugin_dir.mkdir()
    (plugin_dir / "runner.py").write_text(
        """
class EmbeddingRunner:
    def run_stage(self, binary, bundle):
        return [binary, "embed", bundle]

class RerankingRunner:
    def run_stage(self, binary, bundle):
        return [binary, "rerank", bundle]

runner = [EmbeddingRunner(), RerankingRunner()]
        """,
        encoding="utf-8",
    )

    commands = mod.extract_runtime_cli_commands_from_e2e_plugins(
        {
            "dual_embedding": {"runner_class": "runner.EmbeddingRunner"},
            "dual_reranking": {"runner_class": "runner.RerankingRunner"},
        },
        models_dir,
        {"embed", "rerank"},
    )

    assert commands == {
        "dual_embedding": {"embed"},
        "dual_reranking": {"rerank"},
    }


def test_runtime_command_discovery_resolves_imported_runner_instance(tmp_path: Path):
    mod = _import_checker()
    models_dir = tmp_path / "tests" / "e2e" / "models"
    owner_dir = models_dir / "video"
    manifest_dir = owner_dir / "manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "video.json").write_text(
        """
{
  "family": "video",
  "runtime_strategy": "video_runtime",
  "task_strategy": "diffusion_media_generation"
}
        """,
        encoding="utf-8",
    )
    runner_dir = owner_dir / "e2e_plugins" / "runners"
    runner_dir.mkdir(parents=True)
    (runner_dir / "diffusion.py").write_text(
        """
class DiffusionMediaRunner:
    def run_stage(self, binary, bundle):
        return [binary, "generate-video", bundle]

plugin = DiffusionMediaRunner()
        """,
        encoding="utf-8",
    )
    (owner_dir / "e2e_plugins" / "runner.py").write_text(
        """
from .runners.diffusion import plugin

runner = plugin
        """,
        encoding="utf-8",
    )

    assert mod.extract_runtime_cli_commands_from_e2e_plugins(
        {"video_runtime": {"runner_class": "diffusion.DiffusionMediaRunner"}},
        models_dir,
        {"generate-video"},
    ) == {"video_runtime": {"generate-video"}}


def test_runtime_command_discovery_resolves_package_module_aliases(tmp_path: Path):
    mod = _import_checker()
    models_dir = tmp_path / "tests" / "e2e" / "models"
    owner_dir = models_dir / "module_alias"
    manifest_dir = owner_dir / "manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "module-alias.json").write_text(
        """
{
  "family": "module_alias",
  "runtime_strategy": "module_alias_runtime",
  "task_strategy": "text_generation_causal"
}
        """,
        encoding="utf-8",
    )
    plugin_dir = owner_dir / "e2e_plugins"
    runners_dir = plugin_dir / "runners"
    runners_dir.mkdir(parents=True)
    (runners_dir / "base.py").write_text(
        """
class Parent:
    def run_stage(self, binary, bundle):
        return [binary, "run", bundle]
        """,
        encoding="utf-8",
    )
    (runners_dir / "active.py").write_text(
        """
from . import base

class ActiveRunner(base.Parent):
    pass
        """,
        encoding="utf-8",
    )
    (plugin_dir / "runner.py").write_text(
        """
from .runners import active as selected

runner = selected.ActiveRunner()
        """,
        encoding="utf-8",
    )

    assert mod.extract_runtime_cli_commands_from_e2e_plugins(
        {"module_alias_runtime": {"runner_class": "active.ActiveRunner"}},
        models_dir,
        {"run"},
    ) == {"module_alias_runtime": {"run"}}


def test_runtime_command_discovery_resolves_recursive_package_reexport(
    tmp_path: Path,
):
    mod = _import_checker()
    models_dir = tmp_path / "tests" / "e2e" / "models"
    owner_dir = models_dir / "package_reexport"
    manifest_dir = owner_dir / "manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "package-reexport.json").write_text(
        """
{
  "family": "package_reexport",
  "runtime_strategy": "package_reexport_runtime",
  "task_strategy": "text_generation_causal"
}
        """,
        encoding="utf-8",
    )
    plugin_dir = owner_dir / "e2e_plugins"
    runners_dir = plugin_dir / "runners"
    runners_dir.mkdir(parents=True)
    (runners_dir / "__init__.py").write_text(
        "from . import active\n",
        encoding="utf-8",
    )
    (runners_dir / "active.py").write_text(
        """
class ActiveRunner:
    def run_stage(self, binary, bundle):
        return [binary, "run", bundle]
        """,
        encoding="utf-8",
    )
    (plugin_dir / "runner.py").write_text(
        """
from . import runners

runner = runners.active.ActiveRunner()
        """,
        encoding="utf-8",
    )

    assert mod.extract_runtime_cli_commands_from_e2e_plugins(
        {"package_reexport_runtime": {"runner_class": "active.ActiveRunner"}},
        models_dir,
        {"run"},
    ) == {"package_reexport_runtime": {"run"}}


def test_runtime_command_discovery_resolves_assignment_reexport_alias(
    tmp_path: Path,
):
    mod = _import_checker()
    models_dir = tmp_path / "tests" / "e2e" / "models"
    owner_dir = models_dir / "assignment_reexport"
    manifest_dir = owner_dir / "manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "assignment-reexport.json").write_text(
        """
{
  "family": "assignment_reexport",
  "runtime_strategy": "assignment_reexport_runtime",
  "task_strategy": "text_generation_causal"
}
        """,
        encoding="utf-8",
    )
    plugin_dir = owner_dir / "e2e_plugins"
    runners_dir = plugin_dir / "runners"
    runners_dir.mkdir(parents=True)
    (runners_dir / "active.py").write_text(
        """
class ActiveRunner:
    def run_stage(self, binary, bundle):
        return [binary, "run", bundle]
        """,
        encoding="utf-8",
    )
    (runners_dir / "__init__.py").write_text(
        """
from .active import ActiveRunner

PublicRunner = ActiveRunner
        """,
        encoding="utf-8",
    )
    (plugin_dir / "runner.py").write_text(
        """
from .runners import PublicRunner

runner = PublicRunner()
        """,
        encoding="utf-8",
    )

    assert mod.extract_runtime_cli_commands_from_e2e_plugins(
        {
            "assignment_reexport_runtime": {
                "runner_class": "active.ActiveRunner",
            }
        },
        models_dir,
        {"run"},
    ) == {"assignment_reexport_runtime": {"run"}}


def test_runtime_command_discovery_stops_assignment_reexport_cycles(
    tmp_path: Path,
):
    mod = _import_checker()
    models_dir = tmp_path / "tests" / "e2e" / "models"
    owner_dir = models_dir / "assignment_cycle"
    manifest_dir = owner_dir / "manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "assignment-cycle.json").write_text(
        """
{
  "family": "assignment_cycle",
  "runtime_strategy": "assignment_cycle_runtime",
  "task_strategy": "text_generation_causal"
}
        """,
        encoding="utf-8",
    )
    plugin_dir = owner_dir / "e2e_plugins"
    runners_dir = plugin_dir / "runners"
    runners_dir.mkdir(parents=True)
    (runners_dir / "__init__.py").write_text(
        """
FirstRunner = SecondRunner
SecondRunner = FirstRunner
        """,
        encoding="utf-8",
    )
    (plugin_dir / "runner.py").write_text(
        """
from .runners import FirstRunner

runner = FirstRunner()
        """,
        encoding="utf-8",
    )

    assert mod.extract_runtime_cli_commands_from_e2e_plugins(
        {
            "assignment_cycle_runtime": {
                "runner_class": "runners.FirstRunner",
            }
        },
        models_dir,
        {"run"},
    ) == {"assignment_cycle_runtime": set()}


def test_runtime_command_discovery_intersects_commands_across_runtime_owners(
    tmp_path: Path,
):
    mod = _import_checker()
    models_dir = tmp_path / "tests" / "e2e" / "models"
    for owner, owner_command in (
        ("alpha_owner", "embed"),
        ("beta_owner", "rerank"),
    ):
        owner_dir = models_dir / owner
        manifest_dir = owner_dir / "manifests"
        manifest_dir.mkdir(parents=True)
        (manifest_dir / f"{owner}.json").write_text(
            f"""
{{
  "family": "{owner}",
  "runtime_strategy": "shared_runtime",
  "task_strategy": "text_generation_causal"
}}
            """,
            encoding="utf-8",
        )
        plugin_dir = owner_dir / "e2e_plugins"
        runners_dir = plugin_dir / "runners"
        runners_dir.mkdir(parents=True)
        (runners_dir / "shared.py").write_text(
            f"""
class SharedRunner:
    def run_stage(self, binary, bundle):
        return [binary, "run", bundle]

    def owner_stage(self, binary, bundle):
        return [binary, "{owner_command}", bundle]
            """,
            encoding="utf-8",
        )
        (plugin_dir / "runner.py").write_text(
            """
from .runners.shared import SharedRunner

runner = SharedRunner()
            """,
            encoding="utf-8",
        )

    assert mod.extract_runtime_cli_commands_from_e2e_plugins(
        {"shared_runtime": {"runner_class": "shared.SharedRunner"}},
        models_dir,
        {"embed", "rerank", "run"},
    ) == {"shared_runtime": {"run"}}


def test_runtime_command_discovery_distinguishes_two_active_same_name_runners(
    tmp_path: Path,
):
    mod = _import_checker()
    models_dir = tmp_path / "tests" / "e2e" / "models"
    owner_dir = models_dir / "same_name"
    manifest_dir = owner_dir / "manifests"
    manifest_dir.mkdir(parents=True)
    for runtime_strategy in ("alpha_runtime", "beta_runtime"):
        (manifest_dir / f"{runtime_strategy}.json").write_text(
            f"""
{{
  "family": "same_name",
  "runtime_strategy": "{runtime_strategy}",
  "task_strategy": "text_generation_causal"
}}
            """,
            encoding="utf-8",
        )
    plugin_dir = owner_dir / "e2e_plugins"
    runners_dir = plugin_dir / "runners"
    runners_dir.mkdir(parents=True)
    (runners_dir / "alpha.py").write_text(
        """
class SameNameRunner:
    def run_stage(self, binary, bundle):
        return [binary, "run", bundle]
        """,
        encoding="utf-8",
    )
    (runners_dir / "beta.py").write_text(
        """
class SameNameRunner:
    def run_stage(self, binary, bundle):
        return [binary, "segment", bundle]
        """,
        encoding="utf-8",
    )
    (plugin_dir / "runner.py").write_text(
        """
from .runners import alpha, beta

runner = [alpha.SameNameRunner(), beta.SameNameRunner()]
        """,
        encoding="utf-8",
    )

    assert mod.extract_runtime_cli_commands_from_e2e_plugins(
        {
            "alpha_runtime": {"runner_class": "alpha.SameNameRunner"},
            "beta_runtime": {"runner_class": "beta.SameNameRunner"},
        },
        models_dir,
        {"run", "segment"},
    ) == {
        "alpha_runtime": {"run"},
        "beta_runtime": {"segment"},
    }


def test_runtime_command_discovery_is_module_qualified_for_duplicate_symbols(
    tmp_path: Path,
):
    mod = _import_checker()
    models_dir = tmp_path / "tests" / "e2e" / "models"
    owner_dir = models_dir / "duplicate_symbols"
    manifest_dir = owner_dir / "manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "active.json").write_text(
        """
{
  "family": "duplicate_symbols",
  "runtime_strategy": "active_runtime",
  "task_strategy": "text_generation_causal"
}
        """,
        encoding="utf-8",
    )
    runners_dir = owner_dir / "e2e_plugins" / "runners"
    runners_dir.mkdir(parents=True)
    (runners_dir / "active.py").write_text(
        """
def build_command(binary, bundle):
    return [binary, "run", bundle]

class DuplicateNameRunner:
    def run_stage(self, binary, bundle):
        return build_command(binary, bundle)
        """,
        encoding="utf-8",
    )
    (runners_dir / "inactive.py").write_text(
        """
def build_command(binary, bundle):
    return [binary, "segment", bundle]

class DuplicateNameRunner:
    def run_stage(self, binary, bundle):
        return build_command(binary, bundle)
        """,
        encoding="utf-8",
    )
    (owner_dir / "e2e_plugins" / "runner.py").write_text(
        """
from .runners.active import DuplicateNameRunner

runner = DuplicateNameRunner()
        """,
        encoding="utf-8",
    )

    assert mod.extract_runtime_cli_commands_from_e2e_plugins(
        {"active_runtime": {"runner_class": "active.DuplicateNameRunner"}},
        models_dir,
        {"run", "segment"},
    ) == {"active_runtime": {"run"}}
    assert mod.extract_runtime_cli_commands_from_e2e_plugins(
        {"active_runtime": {"runner_class": "inactive.DuplicateNameRunner"}},
        models_dir,
        {"run", "segment"},
    ) == {"active_runtime": set()}


def test_validate_matrix_data_requires_exemption_when_no_diff_check():
    mod = _import_checker()
    errors = mod.validate_matrix_data(
        matrix={
            "unit_recurrent": {
                "task_strategy": "text_generation_causal",
                "cli_commands": ["run"],
                "runner_class": "text_generation.TextGenerationCausalRunner",
                "comparator_class": "text.TextComparator",
                "diff_framework_check_classes": [],
            }
        },
        cpp_runtime_strategies={"unit_recurrent"},
        runtime_to_task_strategy={"unit_recurrent": "text_generation_causal"},
        diff_check_classes=set(),
        runner_classes_by_task={
            "text_generation_causal": {"runners.text_generation.TextGenerationCausalRunner"}
        },
        comparator_classes_by_task={"text_generation_causal": {"comparators.text.TextComparator"}},
        native_cli_commands={"run"},
        runner_cli_commands_by_runtime={"unit_recurrent": {"run"}},
        active_comparator_classes_by_runtime={
            "unit_recurrent": {"comparators.text.TextComparator"}
        },
    )

    assert any("diff_framework_exemption" in message for message in errors)


def test_validate_matrix_data_detects_runtime_source_mismatch():
    mod = _import_checker()
    errors = mod.validate_matrix_data(
        matrix={
            "qwen_vl_vision_language": {
                "task_strategy": "vision_language_generation",
                "cli_commands": ["run"],
                "runner_class": "vision_language.VisionLanguageRunner",
                "comparator_class": "vision_language.VisionLanguageComparator",
                "diff_framework_check_classes": ["VLPipelineTest"],
            }
        },
        cpp_runtime_strategies={"qwen_vl_vision_language", "future_runtime_strategy"},
        runtime_to_task_strategy={"qwen_vl_vision_language": "vision_language_generation"},
        diff_check_classes={"VLPipelineTest"},
        runner_classes_by_task={
            "vision_language_generation": {"runners.vision_language.VisionLanguageRunner"}
        },
        comparator_classes_by_task={
            "vision_language_generation": {"comparators.vision_language.VisionLanguageComparator"}
        },
        native_cli_commands={"run"},
        runner_cli_commands_by_runtime={"qwen_vl_vision_language": {"run"}},
        active_comparator_classes_by_runtime={
            "qwen_vl_vision_language": {"comparators.vision_language.VisionLanguageComparator"}
        },
    )

    assert any("missing runtime strategies from runtime sources" in message for message in errors)


def test_validate_matrix_data_rejects_source_less_matrix_row():
    mod = _import_checker()
    errors = mod.validate_matrix_data(
        matrix={
            "retired_runtime": {
                "task_strategy": "text_generation_causal",
                "cli_commands": ["run"],
                "runner_class": "text_generation.TextGenerationCausalRunner",
                "comparator_class": "text.TextComparator",
                "diff_framework_check_classes": [],
                "diff_framework_exemption": "No active diff check.",
                "performance_mode": "decode",
            }
        },
        cpp_runtime_strategies=set(),
        runtime_to_task_strategy={},
        diff_check_classes=set(),
        runner_classes_by_task={
            "text_generation_causal": {"runners.text_generation.TextGenerationCausalRunner"}
        },
        comparator_classes_by_task={"text_generation_causal": {"comparators.text.TextComparator"}},
        native_cli_commands={"run"},
        runner_cli_commands_by_runtime={},
        active_comparator_classes_by_runtime={},
    )

    assert any("absent from runtime sources and E2E manifests" in message for message in errors)


def test_validate_matrix_data_rejects_non_native_cli_command():
    mod = _import_checker()
    errors = mod.validate_matrix_data(
        matrix={
            "unit_recurrent": {
                "task_strategy": "text_generation_causal",
                "cli_commands": ["definitely-not-a-command"],
                "runner_class": "text_generation.TextGenerationCausalRunner",
                "comparator_class": "text.TextComparator",
                "diff_framework_check_classes": [],
                "diff_framework_exemption": "No active diff check.",
                "performance_mode": "decode",
            }
        },
        cpp_runtime_strategies={"unit_recurrent"},
        runtime_to_task_strategy={"unit_recurrent": "text_generation_causal"},
        diff_check_classes=set(),
        runner_classes_by_task={
            "text_generation_causal": {"runners.text_generation.TextGenerationCausalRunner"}
        },
        comparator_classes_by_task={"text_generation_causal": {"comparators.text.TextComparator"}},
        native_cli_commands={"run"},
        runner_cli_commands_by_runtime={"unit_recurrent": {"run"}},
        active_comparator_classes_by_runtime={
            "unit_recurrent": {"comparators.text.TextComparator"}
        },
    )

    assert any("not accepted by the native CLI" in message for message in errors)


def test_validate_matrix_data_rejects_command_not_used_by_runtime_runner():
    mod = _import_checker()
    errors = mod.validate_matrix_data(
        matrix={
            "sam3_prompted_segmentation": {
                "task_strategy": "prompted_segmentation",
                "cli_commands": ["segment"],
                "runner_class": "segmentation.PromptedSegmentationRunner",
                "comparator_class": "segmentation.PromptedSegmentationComparator",
                "diff_framework_check_classes": [],
                "diff_framework_exemption": "No active diff check.",
                "performance_mode": "single_pass",
            }
        },
        cpp_runtime_strategies={"sam3_prompted_segmentation"},
        runtime_to_task_strategy={"sam3_prompted_segmentation": "prompted_segmentation"},
        diff_check_classes=set(),
        runner_classes_by_task={
            "prompted_segmentation": {"runners.segmentation.PromptedSegmentationRunner"}
        },
        comparator_classes_by_task={
            "prompted_segmentation": {"comparators.segmentation.PromptedSegmentationComparator"}
        },
        native_cli_commands={"segment", "segment-prompted"},
        runner_cli_commands_by_runtime={"sam3_prompted_segmentation": {"segment-prompted"}},
        active_comparator_classes_by_runtime={
            "sam3_prompted_segmentation": {
                "comparators.segmentation.PromptedSegmentationComparator"
            }
        },
    )

    assert any(
        "not used by the model-owned E2E runner/command builders" in message for message in errors
    )


def test_validate_matrix_data_rejects_runner_command_missing_from_matrix():
    mod = _import_checker()
    errors = mod.validate_matrix_data(
        matrix={
            "diffusion_flux": {
                "task_strategy": "diffusion_media_generation",
                "cli_commands": ["generate-video"],
                "runner_class": "diffusion.DiffusionMediaRunner",
                "comparator_class": "diffusion.DiffusionComparator",
                "diff_framework_check_classes": [],
                "diff_framework_exemption": "No active diff check.",
                "performance_mode": "diffusion",
            }
        },
        cpp_runtime_strategies={"diffusion_flux"},
        runtime_to_task_strategy={"diffusion_flux": "diffusion_media_generation"},
        diff_check_classes=set(),
        runner_classes_by_task={
            "diffusion_media_generation": {"runners.diffusion.DiffusionMediaRunner"}
        },
        comparator_classes_by_task={
            "diffusion_media_generation": {"comparators.diffusion.DiffusionComparator"}
        },
        native_cli_commands={"generate-video", "run"},
        runner_cli_commands_by_runtime={"diffusion_flux": {"generate-video", "run"}},
        active_comparator_classes_by_runtime={
            "diffusion_flux": {"comparators.diffusion.DiffusionComparator"}
        },
    )

    assert any(
        "native commands missing from cli_commands: ['run']" in message for message in errors
    )


def test_validate_matrix_data_rejects_wrong_runner_module_with_same_class_name():
    mod = _import_checker()
    errors = mod.validate_matrix_data(
        matrix={
            "unit_runtime": {
                "task_strategy": "text_generation_causal",
                "cli_commands": ["run"],
                "runner_class": "inactive.TextGenerationCausalRunner",
                "comparator_class": "text.TextComparator",
                "diff_framework_check_classes": [],
                "diff_framework_exemption": "No active diff check.",
                "performance_mode": "decode",
            }
        },
        cpp_runtime_strategies={"unit_runtime"},
        runtime_to_task_strategy={"unit_runtime": "text_generation_causal"},
        diff_check_classes=set(),
        runner_classes_by_task={
            "text_generation_causal": {"runners.text_generation.TextGenerationCausalRunner"}
        },
        comparator_classes_by_task={"text_generation_causal": {"comparators.text.TextComparator"}},
        native_cli_commands={"run"},
        runner_cli_commands_by_runtime={"unit_runtime": {"run"}},
        active_comparator_classes_by_runtime={"unit_runtime": {"comparators.text.TextComparator"}},
    )

    assert any(
        "runner_class 'inactive.TextGenerationCausalRunner' not in discovered runner classes"
        in message
        for message in errors
    )


def test_validate_matrix_data_rejects_wrong_comparator_module_with_same_class_name():
    mod = _import_checker()
    errors = mod.validate_matrix_data(
        matrix={
            "unit_runtime": {
                "task_strategy": "text_generation_causal",
                "cli_commands": ["run"],
                "runner_class": "text_generation.TextGenerationCausalRunner",
                "comparator_class": "inactive.TextComparator",
                "diff_framework_check_classes": [],
                "diff_framework_exemption": "No active diff check.",
                "performance_mode": "decode",
            }
        },
        cpp_runtime_strategies={"unit_runtime"},
        runtime_to_task_strategy={"unit_runtime": "text_generation_causal"},
        diff_check_classes=set(),
        runner_classes_by_task={
            "text_generation_causal": {"runners.text_generation.TextGenerationCausalRunner"}
        },
        comparator_classes_by_task={"text_generation_causal": {"comparators.text.TextComparator"}},
        native_cli_commands={"run"},
        runner_cli_commands_by_runtime={"unit_runtime": {"run"}},
        active_comparator_classes_by_runtime={"unit_runtime": {"comparators.text.TextComparator"}},
    )

    assert any(
        "comparator_class 'inactive.TextComparator' not in discovered comparator classes" in message
        for message in errors
    )


def test_prompted_segmentation_matrix_uses_prompted_native_cli():
    mod = _import_checker()
    matrix = mod.load_runtime_strategy_matrix(mod.DEFAULT_MATRIX_PATH)

    assert matrix["sam_prompted_segmentation"]["cli_commands"] == ["segment-prompted"]
    assert matrix["sam3_prompted_segmentation"]["cli_commands"] == ["segment-prompted"]
    assert matrix["diffusion_flux"]["cli_commands"] == ["generate-video", "run"]


def test_checker_rejects_injected_non_native_command(tmp_path: Path):
    mod = _import_checker()
    matrix_data = mod.load_yaml_like(mod.DEFAULT_MATRIX_PATH)
    matrix_data["runtime_strategies"]["sam3_prompted_segmentation"]["cli_commands"] = [
        "definitely-not-a-command"
    ]
    matrix_path = tmp_path / "runtime_strategy_matrix.yaml"
    matrix_path.write_text(json.dumps(matrix_data), encoding="utf-8")

    assert mod.main(["--matrix", str(matrix_path)]) == 1


def test_validate_matrix_paths_requires_comparator_for_every_runtime_owner(
    tmp_path: Path,
):
    mod = _import_checker()
    matrix_path = tmp_path / "tests" / "runtime_strategy_matrix.yaml"
    matrix_path.parent.mkdir(parents=True)
    matrix_path.write_text(
        json.dumps(
            {
                "runtime_strategies": {
                    "diffusion_flux": {
                        "task_strategy": "diffusion_media_generation",
                        "performance_mode": "diffusion",
                        "cli_commands": ["generate-video"],
                        "runner_class": "diffusion.DiffusionMediaRunner",
                        "comparator_class": "borrowed.BorrowedComparator",
                        "diff_framework_check_classes": [],
                        "diff_framework_exemption": "Synthetic owner check.",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    models_dir = tmp_path / "tests" / "e2e" / "models"
    owner_dir = models_dir / "flux"
    manifest_dir = owner_dir / "manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "flux.json").write_text(
        """
{
  "family": "flux",
  "runtime_strategy": "diffusion_flux",
  "task_strategy": "diffusion_media_generation"
}
        """,
        encoding="utf-8",
    )
    plugin_dir = owner_dir / "e2e_plugins"
    runners_dir = plugin_dir / "runners"
    runners_dir.mkdir(parents=True)
    (runners_dir / "diffusion.py").write_text(
        """
class DiffusionMediaRunner:
    def strategy_name(self):
        return "diffusion_media_generation"

    def run_stage(self, binary, bundle):
        return [binary, "generate-video", bundle]
        """,
        encoding="utf-8",
    )
    (plugin_dir / "runner.py").write_text(
        """
from .runners.diffusion import DiffusionMediaRunner

runner = DiffusionMediaRunner()
        """,
        encoding="utf-8",
    )
    owner_comparators = plugin_dir / "comparators"
    owner_comparators.mkdir()
    (owner_comparators / "local.py").write_text(
        """
class LocalComparator:
    def task_strategy(self):
        return "diffusion_media_generation"
        """,
        encoding="utf-8",
    )
    (plugin_dir / "comparator.py").write_text(
        """
from .comparators.local import LocalComparator

comparator = LocalComparator()
        """,
        encoding="utf-8",
    )

    borrower_dir = models_dir / "sana_wm"
    borrower_manifest_dir = borrower_dir / "manifests"
    borrower_manifest_dir.mkdir(parents=True)
    (borrower_manifest_dir / "sana.json").write_text(
        """
{
  "family": "sana_wm",
  "runtime_strategy": "diffusion_flux",
  "task_strategy": "diffusion_media_generation"
}
        """,
        encoding="utf-8",
    )
    borrower_plugin_dir = borrower_dir / "e2e_plugins"
    borrower_comparators = borrower_plugin_dir / "comparators"
    borrower_comparators.mkdir(parents=True)
    (borrower_comparators / "borrowed.py").write_text(
        """
class BorrowedComparator:
    def task_strategy(self):
        return "diffusion_media_generation"
        """,
        encoding="utf-8",
    )
    (borrower_plugin_dir / "comparator.py").write_text(
        """
from .comparators.borrowed import BorrowedComparator

comparator = BorrowedComparator()
        """,
        encoding="utf-8",
    )

    cpp_path = tmp_path / "src" / "cabi" / "api" / "trtmc_c.cpp"
    cpp_path.parent.mkdir(parents=True)
    cpp_path.write_text(
        'static const char* runtime_strategy = "diffusion_flux";\n',
        encoding="utf-8",
    )
    cli_args_path = tmp_path / "src" / "cli" / "args.cpp"
    cli_args_path.parent.mkdir(parents=True)
    cli_args_path.write_text(
        'static const char* known_cmds[] = {"generate-video", nullptr};\n',
        encoding="utf-8",
    )

    errors = mod.validate_matrix_paths(
        matrix_path=matrix_path,
        cpp_path=cpp_path,
        builders_dir=tmp_path / "missing_builders",
        runtime_registry_path=tmp_path / "missing_pipeline_factory.cpp",
        runtime_models_dir=tmp_path / "missing_runtime_models",
        cli_args_path=cli_args_path,
        torchtrt_strategies_dir=tmp_path / "missing_torchtrt_strategies",
        e2e_models_dir=models_dir,
        diff_checks_dir=tmp_path / "missing_diff_checks",
        runners_dir=models_dir,
        comparators_dir=models_dir,
    )

    assert any(
        "comparator_class 'borrowed.BorrowedComparator' is not in the active "
        "comparator lineage for its manifest owner/runtime" in message
        for message in errors
    )
    assert not any("not in discovered comparator classes" in message for message in errors)


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
              "runner_class": "text_generation.TextGenerationCausalRunner",
              "comparator_class": "text.TextComparator",
              "diff_framework_check_classes": ["LogitDiffTest"]
            },
            "qwen_vl_vision_language": {
              "task_strategy": "vision_language_generation",
              "performance_mode": "enc_dec",
              "cli_commands": ["run"],
              "runner_class": "vision_language.VisionLanguageRunner",
              "comparator_class": "vision_language.VisionLanguageComparator",
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
    for (
        owner,
        runner_module,
        runner_class,
        comparator_module,
        comparator_class,
        task_strategy,
    ) in (
        (
            "text",
            "text_generation",
            "TextGenerationCausalRunner",
            "text",
            "TextComparator",
            "text_generation_causal",
        ),
        (
            "vl",
            "vision_language",
            "VisionLanguageRunner",
            "vision_language",
            "VisionLanguageComparator",
            "vision_language_generation",
        ),
    ):
        plugin_dir = e2e_models_dir / owner / "e2e_plugins"
        plugin_runners_dir = plugin_dir / "runners"
        plugin_runners_dir.mkdir(parents=True)
        (plugin_runners_dir / f"{runner_module}.py").write_text(
            f"""
class {runner_class}:
    def run_stage(self, binary, bundle):
        return [binary, "run", bundle]
            """,
            encoding="utf-8",
        )
        (plugin_dir / "runner.py").write_text(
            f"""
from .runners.{runner_module} import {runner_class}

runner = {runner_class}()
            """,
            encoding="utf-8",
        )
        plugin_comparators_dir = plugin_dir / "comparators"
        plugin_comparators_dir.mkdir()
        (plugin_comparators_dir / f"{comparator_module}.py").write_text(
            f"""
class {comparator_class}:
    def task_strategy(self):
        return "{task_strategy}"
            """,
            encoding="utf-8",
        )
        (plugin_dir / "comparator.py").write_text(
            f"""
from .comparators.{comparator_module} import {comparator_class}

comparator = {comparator_class}()
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
    cli_args_path = tmp_path / "src" / "cli" / "args.cpp"
    cli_args_path.parent.mkdir(parents=True)
    cli_args_path.write_text(
        """
static const char* known_cmds[] = {"run", "encode", nullptr};
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
        cli_args_path=cli_args_path,
        torchtrt_strategies_dir=tmp_path / "missing_torchtrt_strategies",
        e2e_models_dir=e2e_models_dir,
        diff_checks_dir=diff_checks_dir,
        runners_dir=runners_dir,
        comparators_dir=comparators_dir,
    )
    assert errors == []
