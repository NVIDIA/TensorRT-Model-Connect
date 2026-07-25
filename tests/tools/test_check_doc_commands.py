# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the non-executing documentation command gate.

Intent:
    Keep every documented shell block parseable and prevent commands from
    naming deleted repository-local scripts or search/test inputs.
Preconditions:
    ``bash`` and the pure-stdlib checker are available.
Postconditions:
    Nested/indented Markdown fences, placeholders, syntax failures, and local
    command inputs are classified without executing the examples. Static
    native, argparse, and Bash-wrapper contracts enforce subcommand scope,
    option arity/choices, and required inputs.

Trace IDs:
    - ARCH-CI-QUALITY-GATES
    - UD-TOOLS-DOC-COMMAND-CHECKER
    - UT-TOOLS-DOC-COMMAND-PARSER
"""

from __future__ import annotations

from pathlib import Path

from tools import check_doc_commands as cdc


def test_vendored_hf_model_card_is_not_project_documentation() -> None:
    assert cdc.is_vendored_fixture_document(
        Path("tests/e2e/models/qwen/data/hf/Qwen__Qwen3-0.6B/README.md")
    )
    assert not cdc.is_vendored_fixture_document(Path("tests/e2e/models/qwen/README.md"))


def test_extract_shell_blocks_includes_indented_fence() -> None:
    content = "1. Example:\n\n   ```bash\n   python3 tools/check.py --help\n   ```\n"

    blocks = cdc.extract_shell_blocks(Path("README.md"), content)

    assert len(blocks) == 1
    assert blocks[0].line == 3
    assert "python3 tools/check.py" in blocks[0].body


def test_four_backtick_markdown_wrapper_does_not_create_inner_shell_block() -> None:
    content = "````markdown\n```bash\npython3 tools/example.py\n```\n````\n"

    assert cdc.extract_shell_blocks(Path("README.md"), content) == []


def test_extract_inline_commands_skips_fences_and_non_commands() -> None:
    content = (
        "Run `python3 tools/check.py --help`, not `PipelineFactory::load()`.\n"
        "```bash\n"
        "python3 tools/fenced.py\n"
        "```\n"
    )

    commands = cdc.extract_inline_commands(Path("README.md"), content)

    assert [command.body for command in commands] == ["python3 tools/check.py --help\n"]


def test_placeholder_is_safe_for_shell_syntax_check() -> None:
    block = cdc.ShellBlock(
        path=Path("README.md"),
        line=10,
        language="bash",
        body="python3 tools/run.py --model <model-name>\n",
    )

    assert cdc.check_shell_syntax(block) is None


def test_invalid_shell_syntax_is_reported() -> None:
    block = cdc.ShellBlock(
        path=Path("README.md"),
        line=4,
        language="bash",
        body="if true; then\n",
    )

    finding = cdc.check_shell_syntax(block)

    assert finding is not None
    assert "invalid shell syntax" in finding.message


def test_missing_python_script_is_reported_without_execution(tmp_path: Path) -> None:
    block = cdc.ShellBlock(
        path=Path("README.md"),
        line=7,
        language="bash",
        body="python3 tools/missing.py --help\n",
    )

    findings = cdc.check_local_inputs(block, tmp_path)

    assert [finding.message for finding in findings] == [
        "command input does not exist: tools/missing.py"
    ]


def test_every_command_in_shell_chain_is_checked(tmp_path: Path) -> None:
    existing = tmp_path / "tools" / "existing.py"
    existing.parent.mkdir(parents=True)
    existing.write_text("", encoding="utf-8")
    block = cdc.ShellBlock(
        path=Path("README.md"),
        line=7,
        language="bash",
        body=(
            "python3 tools/existing.py && "
            "python3 tools/missing.py | python3 tools/also-missing.py\n"
        ),
    )

    findings = cdc.check_local_inputs(block, tmp_path)

    assert [finding.message for finding in findings] == [
        "command input does not exist: tools/missing.py",
        "command input does not exist: tools/also-missing.py",
    ]


def test_existing_pytest_node_path_is_accepted(tmp_path: Path) -> None:
    test_file = tmp_path / "tests" / "tools" / "test_example.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("def test_example(): pass\n", encoding="utf-8")
    block = cdc.ShellBlock(
        path=Path("README.md"),
        line=2,
        language="bash",
        body="pytest tests/tools/test_example.py::test_example -q\n",
    )

    assert cdc.check_local_inputs(block, tmp_path) == []


def test_placeholder_pytest_path_is_not_treated_as_a_real_input(
    tmp_path: Path,
) -> None:
    block = cdc.ShellBlock(
        path=Path("README.md"),
        line=2,
        language="bash",
        body="pytest tests/e2e/models/<family> --e2e-model <model-name>\n",
    )

    assert cdc.check_local_inputs(block, tmp_path) == []


def test_trtmc_contract_rejects_unknown_subcommand_and_option(
    tmp_path: Path,
) -> None:
    args_cpp = tmp_path / "src" / "cli" / "args.cpp"
    args_cpp.parent.mkdir(parents=True)
    args_cpp.write_text(
        'static const char* known_cmds[] = {"run", "inspect", nullptr};\n'
        'if (arg == "--prompt") {}\n',
        encoding="utf-8",
    )
    build_cli = tmp_path / "python" / "tensorrt_model_connect" / "build_cli.py"
    build_cli.parent.mkdir(parents=True)
    build_cli.write_text(
        'parser.add_argument("-o", "--output")\n',
        encoding="utf-8",
    )
    block = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body=("trtmc missing bundle.trtfb\ntrtmc run bundle.trtfb --not-a-flag\n"),
    )

    findings = cdc.check_trtmc_contract(block, tmp_path)

    assert [finding.message for finding in findings] == [
        "unknown trtmc subcommand: missing",
        "unknown option for `trtmc run`: --not-a-flag",
        "missing required input for `trtmc run`: "
        "--prompt, --prompts-file, or --initial-latents-raw",
    ]


def test_python_module_contract_reads_argparse_commands_and_flags(
    tmp_path: Path,
) -> None:
    build_cli = tmp_path / "python" / "tensorrt_model_connect" / "build_cli.py"
    build_cli.parent.mkdir(parents=True)
    build_cli.write_text(
        'build_p = sub.add_parser("build")\nbuild_p.add_argument("-o", "--output")\n',
        encoding="utf-8",
    )
    block = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body=(
            "python3 -m tensorrt_model_connect build model -o out.trtfb\n"
            "python3 -m tensorrt_model_connect unknown --bad\n"
        ),
    )

    findings = cdc.check_python_module_contract(block, tmp_path)

    assert [finding.message for finding in findings] == [
        "unknown `python -m tensorrt_model_connect` subcommand: unknown"
    ]


def test_python_module_flags_are_scoped_to_the_selected_subcommand(
    tmp_path: Path,
) -> None:
    build_cli = tmp_path / "python" / "tensorrt_model_connect" / "build_cli.py"
    build_cli.parent.mkdir(parents=True)
    build_cli.write_text(
        'build_p = sub.add_parser("build")\n'
        'build_p.add_argument("--precision")\n'
        'inspect_p = sub.add_parser("inspect")\n'
        'inspect_p.add_argument("--list-engines", action="store_true")\n',
        encoding="utf-8",
    )
    block = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body=("python3 -m tensorrt_model_connect inspect model.trtfb --precision fp16\n"),
    )

    findings = cdc.check_python_module_contract(block, tmp_path)

    assert [finding.message for finding in findings] == [
        "unknown option for Python `inspect` command: --precision"
    ]


def test_python_module_contract_checks_required_args_choices_and_arity(
    tmp_path: Path,
) -> None:
    build_cli = tmp_path / "python" / "tensorrt_model_connect" / "build_cli.py"
    build_cli.parent.mkdir(parents=True)
    build_cli.write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        'sub = parser.add_subparsers(dest="command")\n'
        'build = sub.add_parser("build")\n'
        'build.add_argument("model")\n'
        'build.add_argument("-o", "--output", required=True)\n'
        'build.add_argument("--precision", choices=["fp16", "fp32"])\n',
        encoding="utf-8",
    )
    invalid_choice = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body="python3 -m tensorrt_model_connect build model --precision int8\n",
    )
    missing_value = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body=("python3 -m tensorrt_model_connect build model --output --precision fp16\n"),
    )

    assert [
        finding.message for finding in cdc.check_python_module_contract(invalid_choice, tmp_path)
    ] == [
        "invalid value for Python `build` command --precision: int8; expected one of fp16, fp32",
        "missing required option for Python `build` command: --output",
    ]
    assert [
        finding.message for finding in cdc.check_python_module_contract(missing_value, tmp_path)
    ] == ["option for Python `build` command requires a value: --output"]


def test_native_contract_checks_required_inputs_scope_arity_and_choices(
    tmp_path: Path,
) -> None:
    args_cpp = tmp_path / "src" / "cli" / "args.cpp"
    args_cpp.parent.mkdir(parents=True)
    args_cpp.write_text(
        "void print_usage() {\n"
        '  print("  trtmc run <bundle> --prompt TEXT [--config FILE]\\n");\n'
        '  print("  trtmc inspect <bundle> [--list-engines]\\n");\n'
        '  print("  trtmc transcribe <bundle> --audio FILE '
        '[--task transcribe|translate]\\n");\n'
        "}\n"
        'static const char* known_cmds[] = {"run", "inspect", '
        '"transcribe", nullptr};\n'
        'if (arg == "--prompt" && need_value(arg)) {}\n'
        'if (arg == "--config" && need_value(arg)) {}\n'
        'if (arg == "--list-engines") {}\n'
        'if (arg == "--audio" && need_value(arg)) {}\n'
        'if (arg == "--task" && need_value(arg)) {}\n',
        encoding="utf-8",
    )

    missing = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body="trtmc run\n",
    )
    wrong_scope = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body="trtmc inspect bundle.trtfb --prompt text\n",
    )
    missing_value = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body="trtmc run bundle.trtfb --prompt --config profile.json\n",
    )
    invalid_choice = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body="trtmc transcribe bundle.trtfb --audio in.wav --task summarize\n",
    )

    assert [finding.message for finding in cdc.check_trtmc_contract(missing, tmp_path)] == [
        "missing required input for `trtmc run`: "
        "--prompt, --prompts-file, or --initial-latents-raw",
        "missing required positional for `trtmc run`: bundle",
    ]
    assert [finding.message for finding in cdc.check_trtmc_contract(wrong_scope, tmp_path)] == [
        "unknown option for `trtmc inspect`: --prompt"
    ]
    assert [finding.message for finding in cdc.check_trtmc_contract(missing_value, tmp_path)] == [
        "option for `trtmc run` requires a value: --prompt"
    ]
    assert [finding.message for finding in cdc.check_trtmc_contract(invalid_choice, tmp_path)] == [
        "invalid value for `trtmc transcribe --task`: summarize; "
        "expected one of transcribe, translate"
    ]


def test_native_option_value_starting_with_dash_is_not_treated_as_a_flag(
    tmp_path: Path,
) -> None:
    args_cpp = tmp_path / "src" / "cli" / "args.cpp"
    args_cpp.parent.mkdir(parents=True)
    args_cpp.write_text(
        "void print_usage() {\n"
        '  print("  trtmc run <bundle> --prompt TEXT [--temperature F]\\n");\n'
        "}\n"
        'static const char* known_cmds[] = {"run", nullptr};\n'
        'if (arg == "--prompt" && need_value(arg)) {}\n'
        'if (arg == "--temperature" && need_value(arg)) {}\n',
        encoding="utf-8",
    )
    block = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body="trtmc run bundle.trtfb --prompt -leading --temperature -0.5\n",
    )

    assert cdc.check_trtmc_contract(block, tmp_path) == []


def test_native_optional_outputs_follow_runtime_guards_not_usage_brackets(
    tmp_path: Path,
) -> None:
    args_cpp = tmp_path / "src" / "cli" / "args.cpp"
    args_cpp.parent.mkdir(parents=True)
    args_cpp.write_text(
        "void print_usage() {\n"
        '  print("  trtmc generate-video <bundle> --prompt TEXT --output DIR\\n");\n'
        "}\n"
        'static const char* known_cmds[] = {"generate-video", nullptr};\n'
        'if (arg == "--prompt" && need_value(arg)) {}\n'
        'if (arg == "--output" && need_value(arg)) {}\n',
        encoding="utf-8",
    )
    block = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body="trtmc generate-video bundle.trtfb --prompt demo\n",
    )

    # src/cli/main.cpp supplies a /tmp output default; only its runtime guard
    # (bundle + prompt) is a required-input contract.
    assert cdc.check_trtmc_contract(block, tmp_path) == []


def test_python_script_contract_rejects_removed_argparse_flag(
    tmp_path: Path,
) -> None:
    script = tmp_path / "tools" / "diff.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        'import argparse\nparser = argparse.ArgumentParser()\nparser.add_argument("--model")\n',
        encoding="utf-8",
    )
    block = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body="python3 tools/diff.py --model model --removed\n",
    )

    findings = cdc.check_python_script_contract(block, tmp_path)

    assert [finding.message for finding in findings] == [
        "unknown option for `tools/diff.py`: --removed"
    ]


def test_python_script_contract_reads_required_mutually_exclusive_group(
    tmp_path: Path,
) -> None:
    script = tmp_path / "tools" / "select.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "source = parser.add_mutually_exclusive_group(required=True)\n"
        'source.add_argument("--files", nargs="+")\n'
        'source.add_argument("--base")\n'
        'source.add_argument("--all", action="store_true")\n',
        encoding="utf-8",
    )
    files = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body="python3 tools/select.py --files one.py two.py\n",
    )
    select_all = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body="python3 tools/select.py --all\n",
    )
    missing = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body="python3 tools/select.py\n",
    )

    assert cdc.check_python_script_contract(files, tmp_path) == []
    assert cdc.check_python_script_contract(select_all, tmp_path) == []
    assert [finding.message for finding in cdc.check_python_script_contract(missing, tmp_path)] == [
        "missing required input for `tools/select.py`: --files, --base, or --all"
    ]


def test_python_subcommand_contract_reads_argument_group_options(
    tmp_path: Path,
) -> None:
    script = tmp_path / "tools" / "queue.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        'sub = parser.add_subparsers(dest="command", required=True)\n'
        'list_cmd = sub.add_parser("list")\n'
        'output = list_cmd.add_argument_group("output")\n'
        'output.add_argument("--json", action="store_true")\n',
        encoding="utf-8",
    )
    block = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body="python3 tools/queue.py list --json\n",
    )

    assert cdc.check_python_script_contract(block, tmp_path) == []


def test_python_script_contract_rejects_invalid_literal_choice(
    tmp_path: Path,
) -> None:
    script = tmp_path / "tools" / "profile.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        'parser.add_argument("--runner", choices=["decoder", "family"])\n',
        encoding="utf-8",
    )
    block = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body="python3 tools/profile.py --runner mamba\n",
    )

    findings = cdc.check_python_script_contract(block, tmp_path)

    assert [finding.message for finding in findings] == [
        "invalid value for `tools/profile.py --runner`: mamba; expected one of decoder, family"
    ]


def test_direct_python_wrapper_contract_is_checked(tmp_path: Path) -> None:
    script = tmp_path / "scripts" / "bench"
    script.parent.mkdir(parents=True)
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        'parser.add_argument("--iterations")\n',
        encoding="utf-8",
    )
    block = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body="./scripts/bench --unknown\n",
    )

    findings = cdc.check_direct_script_contract(block, tmp_path)

    assert [finding.message for finding in findings] == [
        "unknown option for `scripts/bench`: --unknown"
    ]


def test_direct_shell_wrapper_contract_checks_engine_dir_and_arity(
    tmp_path: Path,
) -> None:
    script = tmp_path / "scripts" / "validate_family.sh"
    script.parent.mkdir(parents=True)
    script.write_text(
        "#!/usr/bin/env bash\n"
        "while [[ $# -gt 0 ]]; do\n"
        '  case "$1" in\n'
        '    --engine-dir) ENGINE_DIR="$2"; shift 2 ;;\n'
        '    --isolate-model-plugin) ISOLATE="true"; shift ;;\n'
        "    -h|--help) exit 0 ;;\n"
        "  esac\n"
        "done\n",
        encoding="utf-8",
    )
    valid = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body=(
            "./scripts/validate_family.sh model --engine-dir /tmp/engines --isolate-model-plugin\n"
        ),
    )
    missing_value = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body="./scripts/validate_family.sh model --engine-dir\n",
    )
    unknown = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body="./scripts/validate_family.sh model --engines-dir /tmp/engines\n",
    )

    assert cdc.check_direct_script_contract(valid, tmp_path) == []
    assert [
        finding.message for finding in cdc.check_direct_script_contract(missing_value, tmp_path)
    ] == ["option for `scripts/validate_family.sh` requires a value: --engine-dir"]
    assert [finding.message for finding in cdc.check_direct_script_contract(unknown, tmp_path)] == [
        "unknown option for `scripts/validate_family.sh`: --engines-dir"
    ]


def test_python_script_subcommand_scope_skips_root_option_values(
    tmp_path: Path,
) -> None:
    script = tmp_path / "tools" / "queue.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        'parser.add_argument("--remote")\n'
        'sub = parser.add_subparsers(dest="command", required=True)\n'
        'list_cmd = sub.add_parser("list")\n'
        'list_cmd.add_argument("--json", action="store_true")\n'
        'create = sub.add_parser("create")\n'
        'create.add_argument("--title", required=True)\n',
        encoding="utf-8",
    )
    block = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body="python3 tools/queue.py --remote $DOC_REMOTE list --title demo\n",
    )
    missing_subcommand = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body="python3 tools/queue.py --remote $DOC_REMOTE\n",
    )

    assert [finding.message for finding in cdc.check_python_script_contract(block, tmp_path)] == [
        "unknown option for `tools/queue.py list`: --title"
    ]
    assert [
        finding.message
        for finding in cdc.check_python_script_contract(missing_subcommand, tmp_path)
    ] == ["missing required subcommand for `tools/queue.py`"]


def test_explicit_repo_local_positional_input_must_exist(tmp_path: Path) -> None:
    args_cpp = tmp_path / "src" / "cli" / "args.cpp"
    args_cpp.parent.mkdir(parents=True)
    args_cpp.write_text(
        "void print_usage() {\n"
        '  print("  trtmc inspect <bundle>\\n");\n'
        "}\n"
        'static const char* known_cmds[] = {"inspect", nullptr};\n',
        encoding="utf-8",
    )
    block = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body="trtmc inspect tests/fixtures/missing.trtfb\n",
    )

    assert [finding.message for finding in cdc.check_positional_inputs(block, tmp_path)] == [
        "positional command input does not exist: tests/fixtures/missing.trtfb"
    ]


def test_filtered_ctest_requires_nonempty_selection_guard() -> None:
    block = cdc.ShellBlock(
        path=Path("README.md"),
        line=4,
        language="bash",
        body="ctest --test-dir build --output-on-failure -R 'example'\n",
    )

    findings = cdc.check_ctest_contract(block)

    assert [finding.message for finding in findings] == [
        "filtered ctest command must include --no-tests=error"
    ]


def test_filtered_ctest_with_nonempty_selection_guard_is_accepted() -> None:
    block = cdc.ShellBlock(
        path=Path("README.md"),
        line=4,
        language="bash",
        body=("ctest --test-dir build --output-on-failure --no-tests=error -R 'example'\n"),
    )

    assert cdc.check_ctest_contract(block) == []
