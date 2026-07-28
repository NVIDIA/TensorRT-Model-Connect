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
import runpy
import sys
from pathlib import Path

import pytest


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


def _extract_runner_commands_for_binding(
    tmp_path: Path,
    *,
    binding_source: str,
    runner_class: str,
    extra_module_sources: dict[str, str] | None = None,
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
    for relative_path, source in (extra_module_sources or {}).items():
        module_path = plugin_dir / relative_path
        module_path.parent.mkdir(parents=True, exist_ok=True)
        module_path.write_text(source, encoding="utf-8")
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

{binding_source}
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


def _extract_conditional_runner_commands(
    tmp_path: Path,
    *,
    condition: str,
    runner_class: str,
) -> dict[str, set[str]]:
    return _extract_runner_commands_for_binding(
        tmp_path,
        binding_source=f"""
runner = FirstRunner()
if {condition}:
    runner = SecondRunner()
""",
        runner_class=runner_class,
    )


def _execute_runner_binding(
    tmp_path: Path,
    *,
    binding_source: str,
) -> str:
    """Execute the synthetic binding with Python and return its runner type."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    module_path = tmp_path / "runner_runtime.py"
    module_path.write_text(
        f"""
import os

select_second = os.environ.get("SELECT_SECOND") == "1"

class FirstRunner:
    pass

class SecondRunner:
    pass

{binding_source}
""",
        encoding="utf-8",
    )
    namespace = runpy.run_path(str(module_path))
    return type(namespace["runner"]).__name__


def test_runtime_command_discovery_fails_closed_for_import_time_alias_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    package_name = "runtime_import_alias_case"
    runtime_root = tmp_path / "python"
    package_dir = runtime_root / package_name
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    mutator_source = """
import sys

runner_module = sys.modules[f"{__package__}.runner"]
runner_module.A = runner_module.SecondRunner
"""
    (package_dir / "mutator.py").write_text(mutator_source, encoding="utf-8")
    (package_dir / "runner.py").write_text(
        """
class FirstRunner:
    pass

class SecondRunner:
    pass

A = FirstRunner
from . import mutator
runner = A()
""",
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(runtime_root))
    try:
        runtime_module = importlib.import_module(f"{package_name}.runner")
        assert type(runtime_module.runner).__name__ == "SecondRunner"
    finally:
        for module_name in tuple(sys.modules):
            if module_name == package_name or module_name.startswith(f"{package_name}."):
                sys.modules.pop(module_name, None)

    binding_source = """
A = FirstRunner
from . import mutator
runner = A()
"""
    assert _extract_runner_commands_for_binding(
        tmp_path / "checker-first",
        binding_source=binding_source,
        runner_class="FirstRunner",
        extra_module_sources={"mutator.py": mutator_source},
    ) == {"conditional_runtime": set()}
    assert _extract_runner_commands_for_binding(
        tmp_path / "checker-second",
        binding_source=binding_source,
        runner_class="SecondRunner",
        extra_module_sources={"mutator.py": mutator_source},
    ) == {"conditional_runtime": set()}

    assert _extract_runner_commands_for_binding(
        tmp_path / "checker-rebound",
        binding_source="""
A = FirstRunner
from . import mutator
A = FirstRunner
runner = A()
""",
        runner_class="FirstRunner",
        extra_module_sources={"mutator.py": mutator_source},
    ) == {"conditional_runtime": {"run"}}


_RUNTIME_BINDING_DIFFERENTIAL_CASES = [
    (
        "old_eq_ifexp",
        """A = B = FirstRunner
class Selector:
    def __eq__(self, other):
        global A
        A = SecondRunner
        return True
flag = Selector() == 1
runner = A() if flag else B()
""",
        "closed",
    ),
    (
        "old_not_ifexp",
        """A = B = FirstRunner
class Selector:
    def __bool__(self):
        global A
        A = SecondRunner
        return True
flag = not Selector()
runner = B() if flag else A()
""",
        "closed",
    ),
    (
        "old_chain_false",
        "runner = FirstRunner()\n"
        "factory = lambda value=(2 < 1 < (runner := SecondRunner())): value\n",
        "exact",
    ),
    (
        "eq_magic_direct_alias",
        """A = FirstRunner
class Selector:
    def __eq__(self, other):
        global A
        A = SecondRunner
        return True
flag = Selector() == 1
runner = A()
""",
        "closed",
    ),
    (
        "order_magic_direct_alias",
        """A = FirstRunner
class Selector:
    def __lt__(self, other):
        global A
        A = SecondRunner
        return True
flag = Selector() < 1
runner = A()
""",
        "closed",
    ),
    (
        "reflected_order_magic_direct_alias",
        """A = FirstRunner
class Selector:
    def __gt__(self, other):
        global A
        A = SecondRunner
        return True
flag = 1 < Selector()
runner = A()
""",
        "closed",
    ),
    (
        "contains_magic_direct_alias",
        """A = FirstRunner
class Container:
    def __contains__(self, item):
        global A
        A = SecondRunner
        return True
flag = 1 in Container()
runner = A()
""",
        "closed",
    ),
    (
        "item_eq_membership_direct_alias",
        """A = FirstRunner
class Item:
    def __eq__(self, other):
        global A
        A = SecondRunner
        return True
flag = Item() in [1]
runner = A()
""",
        "closed",
    ),
    (
        "not_bool_magic_direct_alias",
        """A = FirstRunner
class Selector:
    def __bool__(self):
        global A
        A = SecondRunner
        return False
flag = not Selector()
runner = A()
""",
        "closed",
    ),
    (
        "identity_constructor_side_effect",
        """A = FirstRunner
class Selector:
    def __init__(self):
        global A
        A = SecondRunner
obj = Selector()
flag = obj is obj
runner = A()
""",
        "closed",
    ),
    (
        "safe_identity_true",
        "flag = None is None\nrunner = SecondRunner() if flag else FirstRunner()\n",
        "exact",
    ),
    (
        "safe_identity_false",
        "flag = None is not None\nrunner = SecondRunner() if flag else FirstRunner()\n",
        "exact",
    ),
    (
        "safe_eq_tuple",
        "flag = (1, 2) == (1, 2)\nrunner = SecondRunner() if flag else FirstRunner()\n",
        "exact",
    ),
    (
        "safe_order_int",
        "flag = 1 < 2\nrunner = SecondRunner() if flag else FirstRunner()\n",
        "exact",
    ),
    (
        "safe_membership_list",
        "flag = 2 in [1, 2]\nrunner = SecondRunner() if flag else FirstRunner()\n",
        "exact",
    ),
    (
        "safe_not_literal",
        "flag = not []\nrunner = SecondRunner() if flag else FirstRunner()\n",
        "exact",
    ),
    (
        "safe_truthy_literal",
        "runner = SecondRunner() if [0] else FirstRunner()\n",
        "exact",
    ),
    (
        "safe_chain_true",
        "flag = 1 < 2 < 3\nrunner = SecondRunner() if flag else FirstRunner()\n",
        "exact",
    ),
    (
        "safe_chain_false",
        "flag = 2 < 1 < 3\nrunner = SecondRunner() if flag else FirstRunner()\n",
        "exact",
    ),
    (
        "and_false_walrus",
        "runner = FirstRunner()\nflag = False and (runner := SecondRunner())\n",
        "exact",
    ),
    (
        "or_true_walrus",
        "runner = FirstRunner()\nflag = True or (runner := SecondRunner())\n",
        "exact",
    ),
    (
        "and_true_walrus",
        "runner = FirstRunner()\nflag = True and (runner := SecondRunner())\n",
        "exact",
    ),
    (
        "chain_true_walrus",
        "runner = FirstRunner()\nflag = 1 < 2 < ((runner := SecondRunner()) and 3)\n",
        "closed",
    ),
    (
        "lambda_default_false",
        "runner = FirstRunner()\n"
        "factory = lambda value=(False and (runner := SecondRunner())): value\n",
        "exact",
    ),
    (
        "lambda_default_alias_true",
        "Alias = FirstRunner\n"
        "factory = lambda value=(True and (Alias := SecondRunner)): value\n"
        "runner = Alias()\n",
        "exact",
    ),
    (
        "lambda_body_ignored",
        "runner = FirstRunner()\nfactory = lambda: (runner := SecondRunner())\n",
        "exact",
    ),
    (
        "starred_safe_sequence",
        "choices = [FirstRunner, SecondRunner]\nrunner = [*choices][1]()\n",
        "exact",
    ),
    (
        "starred_iter_magic_direct_alias",
        """A = FirstRunner
class Values:
    def __iter__(self):
        global A
        A = SecondRunner
        return iter([1])
values = Values()
expanded = [*values]
runner = A()
""",
        "closed",
    ),
    (
        "unknown_ifexp_same_alias",
        "A = B = FirstRunner\nrunner = A() if select_second else B()\n",
        "exact",
    ),
    (
        "unknown_ifexp_different",
        "runner = SecondRunner() if select_second else FirstRunner()\n",
        "closed",
    ),
    (
        "selector_binding_time",
        "selected = 0\nrunner = (FirstRunner(), SecondRunner())[selected]\nselected = 1\n",
        "exact",
    ),
    (
        "alias_rebind_before",
        "Alias = FirstRunner\nAlias = SecondRunner\nrunner = Alias()\n",
        "exact",
    ),
    (
        "alias_rebind_after",
        "Alias = FirstRunner\nrunner = Alias()\nAlias = SecondRunner\n",
        "exact",
    ),
    (
        "nested_rebind_ignored",
        "runner = FirstRunner()\ndef later():\n    global runner\n    runner = SecondRunner()\n",
        "exact",
    ),
    (
        "try_rebind",
        "runner = FirstRunner()\ntry:\n    runner = SecondRunner()\nexcept Exception:\n    pass\n",
        "closed",
    ),
    (
        "loop_rebind",
        "runner = FirstRunner()\nfor _ in [1]:\n    runner = SecondRunner()\n",
        "closed",
    ),
    (
        "decorator_magic_direct_alias",
        """A = FirstRunner
def decorate(cls):
    global A
    A = SecondRunner
    return cls
@decorate
class Marked:
    pass
runner = A()
""",
        "closed",
    ),
    (
        "function_default_magic_direct_alias",
        """A = FirstRunner
class Selector:
    def __eq__(self, other):
        global A
        A = SecondRunner
        return True
def factory(value=(Selector() == 1)):
    return value
runner = A()
""",
        "closed",
    ),
]


@pytest.mark.parametrize(
    ("case_name", "binding_source", "policy"),
    _RUNTIME_BINDING_DIFFERENTIAL_CASES,
    ids=[case[0] for case in _RUNTIME_BINDING_DIFFERENTIAL_CASES],
)
def test_runtime_command_discovery_matches_real_python_alias_effects(
    tmp_path: Path,
    case_name: str,
    binding_source: str,
    policy: str,
):
    actual = _execute_runner_binding(
        tmp_path / case_name / "python",
        binding_source=binding_source,
    )
    active = [
        runner_class
        for runner_class in ("FirstRunner", "SecondRunner")
        if _extract_runner_commands_for_binding(
            tmp_path / case_name / f"checker-{runner_class}",
            binding_source=binding_source,
            runner_class=runner_class,
        )["conditional_runtime"]
    ]
    if policy == "closed":
        assert active == []
    else:
        assert policy == "exact"
        assert active == [actual]


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


def test_runtime_command_discovery_selects_true_if_expression_branch(
    tmp_path: Path,
):
    assert _extract_runner_commands_for_binding(
        tmp_path,
        binding_source="runner = SecondRunner() if True else FirstRunner()",
        runner_class="SecondRunner",
    ) == {"conditional_runtime": {"generate-audio"}}


def test_runtime_command_discovery_selects_false_if_expression_branch(
    tmp_path: Path,
):
    assert _extract_runner_commands_for_binding(
        tmp_path,
        binding_source="runner = SecondRunner() if False else FirstRunner()",
        runner_class="FirstRunner",
    ) == {"conditional_runtime": {"run"}}


def test_runtime_command_discovery_fails_closed_for_unknown_if_expression(
    tmp_path: Path,
):
    assert _extract_runner_commands_for_binding(
        tmp_path,
        binding_source=("runner = SecondRunner() if select_second else FirstRunner()"),
        runner_class="FirstRunner",
    ) == {"conditional_runtime": set()}


def test_runtime_command_discovery_accepts_unknown_if_expression_same_runner(
    tmp_path: Path,
):
    assert _extract_runner_commands_for_binding(
        tmp_path,
        binding_source=("runner = FirstRunner() if select_second else FirstRunner()"),
        runner_class="FirstRunner",
    ) == {"conditional_runtime": {"run"}}


def test_runtime_command_discovery_accepts_unknown_if_expression_same_alias(
    tmp_path: Path,
):
    assert _extract_runner_commands_for_binding(
        tmp_path,
        binding_source="""
A = B = FirstRunner
runner = A() if select_second else B()
""",
        runner_class="FirstRunner",
    ) == {"conditional_runtime": {"run"}}


def test_runtime_command_discovery_resolves_safe_constant_if_expression(
    tmp_path: Path,
):
    assert _extract_runner_commands_for_binding(
        tmp_path,
        binding_source="runner = SecondRunner() if [0] else FirstRunner()",
        runner_class="SecondRunner",
    ) == {"conditional_runtime": {"generate-audio"}}


def test_runtime_command_discovery_rejects_shadowed_set_call_condition(
    tmp_path: Path,
):
    assert _extract_runner_commands_for_binding(
        tmp_path,
        binding_source="""
def set():
    return [1]

runner = SecondRunner() if set() else FirstRunner()
""",
        runner_class="FirstRunner",
    ) == {"conditional_runtime": set()}


def test_runtime_command_discovery_rejects_if_expression_condition_walrus(
    tmp_path: Path,
):
    assert _extract_runner_commands_for_binding(
        tmp_path,
        binding_source="""
A = B = FirstRunner
runner = A() if (A := SecondRunner) else B()
""",
        runner_class="FirstRunner",
    ) == {"conditional_runtime": set()}


def test_runtime_command_discovery_rejects_if_expression_name_truthiness(
    tmp_path: Path,
):
    assert _extract_runner_commands_for_binding(
        tmp_path,
        binding_source="""
A = B = FirstRunner

class Selector:
    def __bool__(self):
        global A
        A = SecondRunner
        return True

selector = Selector()
runner = A() if selector else B()
""",
        runner_class="FirstRunner",
    ) == {"conditional_runtime": set()}


def test_runtime_command_discovery_rejects_if_expression_starred_condition(
    tmp_path: Path,
):
    assert _extract_runner_commands_for_binding(
        tmp_path,
        binding_source="""
A = B = FirstRunner

class Values:
    def __iter__(self):
        global A
        A = SecondRunner
        return iter([1])

values = Values()
runner = A() if [*values] else B()
""",
        runner_class="FirstRunner",
    ) == {"conditional_runtime": set()}


def test_runtime_command_discovery_rejects_compare_magic_method_alias_side_effect(
    tmp_path: Path,
):
    binding_source = """
A = B = FirstRunner

class Selector:
    def __eq__(self, other):
        global A
        A = SecondRunner
        return True

select_second = Selector() == 1
runner = A() if select_second else B()
"""
    assert (
        _execute_runner_binding(
            tmp_path / "python",
            binding_source=binding_source,
        )
        == "SecondRunner"
    )
    assert _extract_runner_commands_for_binding(
        tmp_path / "checker",
        binding_source=binding_source,
        runner_class="FirstRunner",
    ) == {"conditional_runtime": set()}


def test_runtime_command_discovery_rejects_not_magic_method_alias_side_effect(
    tmp_path: Path,
):
    binding_source = """
A = B = FirstRunner

class Selector:
    def __bool__(self):
        global A
        A = SecondRunner
        return False

select_second = not Selector()
runner = A() if select_second else B()
"""
    assert (
        _execute_runner_binding(
            tmp_path / "python",
            binding_source=binding_source,
        )
        == "SecondRunner"
    )
    assert _extract_runner_commands_for_binding(
        tmp_path / "checker",
        binding_source=binding_source,
        runner_class="FirstRunner",
    ) == {"conditional_runtime": set()}


def test_runtime_command_discovery_fails_closed_for_lambda_default_walrus(
    tmp_path: Path,
):
    assert _extract_runner_commands_for_binding(
        tmp_path,
        binding_source="""
runner = FirstRunner()
factory = lambda value=(runner := SecondRunner()): value
""",
        runner_class="FirstRunner",
    ) == {"conditional_runtime": set()}


def test_runtime_command_discovery_honors_lambda_default_short_circuit(
    tmp_path: Path,
):
    assert _extract_runner_commands_for_binding(
        tmp_path,
        binding_source="""
runner = FirstRunner()
factory = lambda value=(False and (runner := SecondRunner())): value
""",
        runner_class="FirstRunner",
    ) == {"conditional_runtime": {"run"}}


def test_runtime_command_discovery_honors_assigned_lambda_short_circuit(
    tmp_path: Path,
):
    assert _extract_runner_commands_for_binding(
        tmp_path,
        binding_source="""
flag = False
runner = FirstRunner()
factory = lambda value=(flag and (runner := SecondRunner())): value
""",
        runner_class="FirstRunner",
    ) == {"conditional_runtime": {"run"}}


def test_runtime_command_discovery_honors_comparison_lambda_short_circuit(
    tmp_path: Path,
):
    assert _extract_runner_commands_for_binding(
        tmp_path,
        binding_source="""
runner = FirstRunner()
factory = lambda value=((1 == 2) and (runner := SecondRunner())): value
""",
        runner_class="FirstRunner",
    ) == {"conditional_runtime": {"run"}}


def test_runtime_command_discovery_honors_false_chained_comparison_short_circuit(
    tmp_path: Path,
):
    binding_source = """
runner = FirstRunner()
comparison = 2 < 1 < (runner := SecondRunner())
"""
    assert (
        _execute_runner_binding(
            tmp_path / "python",
            binding_source=binding_source,
        )
        == "FirstRunner"
    )
    assert _extract_runner_commands_for_binding(
        tmp_path / "checker",
        binding_source=binding_source,
        runner_class="FirstRunner",
    ) == {"conditional_runtime": {"run"}}


def test_runtime_command_discovery_checks_true_chained_comparison_rebinding(
    tmp_path: Path,
):
    binding_source = """
runner = FirstRunner()
comparison = 1 < 2 < ((runner := SecondRunner()) and 3)
"""
    assert (
        _execute_runner_binding(
            tmp_path / "python",
            binding_source=binding_source,
        )
        == "SecondRunner"
    )
    assert _extract_runner_commands_for_binding(
        tmp_path / "checker",
        binding_source=binding_source,
        runner_class="SecondRunner",
    ) == {"conditional_runtime": set()}


def test_runtime_command_discovery_applies_true_lambda_default_alias_rebinding(
    tmp_path: Path,
):
    binding_source = """
Alias = FirstRunner
factory = lambda value=((1 == 1) and (Alias := SecondRunner)): value
runner = Alias()
"""
    assert _extract_runner_commands_for_binding(
        tmp_path / "first",
        binding_source=binding_source,
        runner_class="FirstRunner",
    ) == {"conditional_runtime": set()}
    assert _extract_runner_commands_for_binding(
        tmp_path / "second",
        binding_source=binding_source,
        runner_class="SecondRunner",
    ) == {"conditional_runtime": {"generate-audio"}}


def test_runtime_command_discovery_skips_false_lambda_default_alias_rebinding(
    tmp_path: Path,
):
    assert _extract_runner_commands_for_binding(
        tmp_path,
        binding_source="""
Alias = FirstRunner
factory = lambda value=((1 == 2) and (Alias := SecondRunner)): value
runner = Alias()
""",
        runner_class="FirstRunner",
    ) == {"conditional_runtime": {"run"}}


def test_runtime_command_discovery_selects_lambda_default_if_expression(
    tmp_path: Path,
):
    assert _extract_runner_commands_for_binding(
        tmp_path,
        binding_source="""
runner = FirstRunner()
factory = lambda value=((runner := SecondRunner()) if False else None): value
""",
        runner_class="FirstRunner",
    ) == {"conditional_runtime": {"run"}}


def test_runtime_command_discovery_conservatively_checks_unknown_lambda_default(
    tmp_path: Path,
):
    assert _extract_runner_commands_for_binding(
        tmp_path,
        binding_source="""
runner = FirstRunner()
factory = lambda value=((runner := SecondRunner()) if select_second else None): value
""",
        runner_class="FirstRunner",
    ) == {"conditional_runtime": set()}


def test_runtime_command_discovery_checks_lambda_keyword_default_walrus(
    tmp_path: Path,
):
    assert _extract_runner_commands_for_binding(
        tmp_path,
        binding_source="""
runner = FirstRunner()
factory = lambda *, value=(runner := SecondRunner()): value
""",
        runner_class="FirstRunner",
    ) == {"conditional_runtime": set()}


def test_runtime_command_discovery_ignores_lambda_body_walrus(
    tmp_path: Path,
):
    assert _extract_runner_commands_for_binding(
        tmp_path,
        binding_source="""
runner = FirstRunner()
factory = lambda: (runner := SecondRunner())
""",
        runner_class="FirstRunner",
    ) == {"conditional_runtime": {"run"}}


def test_runtime_command_discovery_selects_only_constant_subscript_runner(
    tmp_path: Path,
):
    binding_source = "runner = (FirstRunner(), SecondRunner())[0]"
    assert _extract_runner_commands_for_binding(
        tmp_path / "selected",
        binding_source=binding_source,
        runner_class="FirstRunner",
    ) == {"conditional_runtime": {"run"}}
    assert _extract_runner_commands_for_binding(
        tmp_path / "unselected",
        binding_source=binding_source,
        runner_class="SecondRunner",
    ) == {"conditional_runtime": set()}


def test_runtime_command_discovery_resolves_negative_constant_subscript(
    tmp_path: Path,
):
    binding_source = "runner = [FirstRunner(), SecondRunner()][-1]"
    assert _extract_runner_commands_for_binding(
        tmp_path / "selected",
        binding_source=binding_source,
        runner_class="SecondRunner",
    ) == {"conditional_runtime": {"generate-audio"}}
    assert _extract_runner_commands_for_binding(
        tmp_path / "unselected",
        binding_source=binding_source,
        runner_class="FirstRunner",
    ) == {"conditional_runtime": set()}


def test_runtime_command_discovery_resolves_safe_constant_subscript_selector(
    tmp_path: Path,
):
    assert _extract_runner_commands_for_binding(
        tmp_path,
        binding_source=("runner = (FirstRunner(), SecondRunner())[0 if not False else 1]"),
        runner_class="FirstRunner",
    ) == {"conditional_runtime": {"run"}}


def test_runtime_command_discovery_resolves_assigned_constant_subscript_selector(
    tmp_path: Path,
):
    binding_source = """
selected = 0
runner = (FirstRunner(), SecondRunner())[selected]
"""
    assert _extract_runner_commands_for_binding(
        tmp_path / "selected",
        binding_source=binding_source,
        runner_class="FirstRunner",
    ) == {"conditional_runtime": {"run"}}
    assert _extract_runner_commands_for_binding(
        tmp_path / "unselected",
        binding_source=binding_source,
        runner_class="SecondRunner",
    ) == {"conditional_runtime": set()}


def test_runtime_command_discovery_uses_selector_value_at_runner_binding_time(
    tmp_path: Path,
):
    binding_source = """
selected = 0
runner = (FirstRunner(), SecondRunner())[selected]
selected = 1
"""
    assert _extract_runner_commands_for_binding(
        tmp_path / "first",
        binding_source=binding_source,
        runner_class="FirstRunner",
    ) == {"conditional_runtime": {"run"}}
    assert _extract_runner_commands_for_binding(
        tmp_path / "second",
        binding_source=binding_source,
        runner_class="SecondRunner",
    ) == {"conditional_runtime": set()}


def test_runtime_command_discovery_uses_latest_selector_before_runner_binding(
    tmp_path: Path,
):
    assert _extract_runner_commands_for_binding(
        tmp_path,
        binding_source="""
selected = 1
selected = 0
runner = (FirstRunner(), SecondRunner())[selected]
""",
        runner_class="FirstRunner",
    ) == {"conditional_runtime": {"run"}}


def test_runtime_command_discovery_uses_class_alias_at_instantiation_time(
    tmp_path: Path,
):
    binding_source = """
Alias = FirstRunner
runner = Alias()
Alias = SecondRunner
"""
    assert _extract_runner_commands_for_binding(
        tmp_path / "first",
        binding_source=binding_source,
        runner_class="FirstRunner",
    ) == {"conditional_runtime": {"run"}}
    assert _extract_runner_commands_for_binding(
        tmp_path / "second",
        binding_source=binding_source,
        runner_class="SecondRunner",
    ) == {"conditional_runtime": set()}


def test_runtime_command_discovery_uses_latest_class_alias_before_instantiation(
    tmp_path: Path,
):
    assert _extract_runner_commands_for_binding(
        tmp_path,
        binding_source="""
Alias = SecondRunner
Alias = FirstRunner
runner = Alias()
""",
        runner_class="FirstRunner",
    ) == {"conditional_runtime": {"run"}}


def test_runtime_command_discovery_fails_closed_for_out_of_range_subscript(
    tmp_path: Path,
):
    assert _extract_runner_commands_for_binding(
        tmp_path,
        binding_source="runner = (FirstRunner(),)[1]",
        runner_class="FirstRunner",
    ) == {"conditional_runtime": set()}


def test_runtime_command_discovery_expands_starred_sequence_subscript(
    tmp_path: Path,
):
    binding_source = "runner = (*[FirstRunner(), SecondRunner()], FirstRunner())[1]"
    assert _extract_runner_commands_for_binding(
        tmp_path / "first",
        binding_source=binding_source,
        runner_class="FirstRunner",
    ) == {"conditional_runtime": set()}
    assert _extract_runner_commands_for_binding(
        tmp_path / "second",
        binding_source=binding_source,
        runner_class="SecondRunner",
    ) == {"conditional_runtime": {"generate-audio"}}


def test_runtime_command_discovery_expands_starred_sequence_negative_subscript(
    tmp_path: Path,
):
    binding_source = """
choices = [FirstRunner(), SecondRunner()]
runner = (*choices,)[-1]
"""
    assert _extract_runner_commands_for_binding(
        tmp_path / "first",
        binding_source=binding_source,
        runner_class="FirstRunner",
    ) == {"conditional_runtime": set()}
    assert _extract_runner_commands_for_binding(
        tmp_path / "second",
        binding_source=binding_source,
        runner_class="SecondRunner",
    ) == {"conditional_runtime": {"generate-audio"}}


def test_runtime_command_discovery_fails_closed_for_starred_sequence_cycle(
    tmp_path: Path,
):
    assert _extract_runner_commands_for_binding(
        tmp_path,
        binding_source="""
first_choices = second_choices
second_choices = first_choices
runner = (*second_choices,)[0]
""",
        runner_class="FirstRunner",
    ) == {"conditional_runtime": set()}


def test_runtime_command_discovery_accepts_dynamic_same_class_candidates(
    tmp_path: Path,
):
    assert _extract_runner_commands_for_binding(
        tmp_path,
        binding_source=("runner = (FirstRunner(), FirstRunner())[select_second]"),
        runner_class="FirstRunner",
    ) == {"conditional_runtime": {"run"}}


def test_runtime_command_discovery_rejects_dynamic_mixed_class_candidates(
    tmp_path: Path,
):
    binding_source = "runner = (FirstRunner(), SecondRunner())[select_second]"
    assert _extract_runner_commands_for_binding(
        tmp_path / "first",
        binding_source=binding_source,
        runner_class="FirstRunner",
    ) == {"conditional_runtime": set()}
    assert _extract_runner_commands_for_binding(
        tmp_path / "second",
        binding_source=binding_source,
        runner_class="SecondRunner",
    ) == {"conditional_runtime": set()}


def test_runtime_command_discovery_rejects_dynamic_unknown_candidate(
    tmp_path: Path,
):
    assert _extract_runner_commands_for_binding(
        tmp_path,
        binding_source="""
def build_runner():
    return FirstRunner()

runner = (FirstRunner(), build_runner())[select_second]
""",
        runner_class="FirstRunner",
    ) == {"conditional_runtime": set()}


def test_runtime_command_discovery_rejects_side_effecting_subscript_selector(
    tmp_path: Path,
):
    assert _extract_runner_commands_for_binding(
        tmp_path,
        binding_source="""
def choose_runner():
    return 0

runner = (FirstRunner(), FirstRunner())[choose_runner()]
""",
        runner_class="FirstRunner",
    ) == {"conditional_runtime": set()}


def test_runtime_command_discovery_rejects_runner_slice(
    tmp_path: Path,
):
    assert _extract_runner_commands_for_binding(
        tmp_path,
        binding_source="runner = (FirstRunner(), FirstRunner())[:]",
        runner_class="FirstRunner",
    ) == {"conditional_runtime": set()}


def test_runtime_command_discovery_fails_closed_for_try_rebinding(
    tmp_path: Path,
):
    assert _extract_runner_commands_for_binding(
        tmp_path,
        binding_source="""
runner = FirstRunner()
try:
    runner = SecondRunner()
except RuntimeError:
    runner = FirstRunner()
""",
        runner_class="FirstRunner",
    ) == {"conditional_runtime": set()}


def test_runtime_command_discovery_fails_closed_for_loop_rebinding(
    tmp_path: Path,
):
    assert _extract_runner_commands_for_binding(
        tmp_path,
        binding_source="""
runner = FirstRunner()
for _candidate in candidates:
    runner = SecondRunner()
""",
        runner_class="FirstRunner",
    ) == {"conditional_runtime": set()}


def test_runtime_command_discovery_later_assignment_clears_ambiguity(
    tmp_path: Path,
):
    assert _extract_runner_commands_for_binding(
        tmp_path,
        binding_source="""
runner = FirstRunner()
try:
    runner = SecondRunner()
except RuntimeError:
    runner = FirstRunner()
for _candidate in candidates:
    runner = FirstRunner()
runner = SecondRunner()
""",
        runner_class="SecondRunner",
    ) == {"conditional_runtime": {"generate-audio"}}


def test_runtime_command_discovery_ignores_nested_scope_assignments(
    tmp_path: Path,
):
    assert _extract_runner_commands_for_binding(
        tmp_path,
        binding_source="""
runner = FirstRunner()

def configure_runner():
    runner = SecondRunner()
    return runner

class RunnerHolder:
    runner = SecondRunner()
""",
        runner_class="FirstRunner",
    ) == {"conditional_runtime": {"run"}}


def test_runtime_command_discovery_fails_closed_for_comprehension_walrus(
    tmp_path: Path,
):
    assert _extract_runner_commands_for_binding(
        tmp_path,
        binding_source="""
runner = FirstRunner()
[runner := SecondRunner() for _candidate in candidates]
""",
        runner_class="FirstRunner",
    ) == {"conditional_runtime": set()}


def test_runtime_command_discovery_fails_closed_for_comprehension_callbacks(
    tmp_path: Path,
):
    assert _extract_runner_commands_for_binding(
        tmp_path,
        binding_source="""
runner = FirstRunner()
[FirstRunner() for runner in candidates]
""",
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
