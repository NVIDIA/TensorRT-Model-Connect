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
    command inputs are classified without executing the examples.

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
