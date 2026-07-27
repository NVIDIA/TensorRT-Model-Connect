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
    option arity/choices, and required inputs. Literal nested shell payloads
    reuse the same checks under depth and cycle guards.

Trace IDs:
    - ARCH-CI-QUALITY-GATES
    - UD-TOOLS-DOC-COMMAND-CHECKER
    - UT-TOOLS-DOC-COMMAND-PARSER
"""

from __future__ import annotations

from pathlib import Path
import shlex

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


def test_unclosed_shell_fence_extends_through_end_of_file() -> None:
    content = "Intro.\n\n```bash\npython3 tools/check.py --help\n"

    blocks = cdc.extract_shell_blocks(Path("README.md"), content)

    assert len(blocks) == 1
    assert blocks[0].line == 3
    assert blocks[0].body == "python3 tools/check.py --help\n"


def test_commonmark_fences_support_containers_and_reject_root_four_space_indent() -> None:
    content = (
        "> ```bash\n"
        "> python3 tools/quoted.py\n"
        "> ```\n"
        "\n"
        "- ```sh\n"
        "  python3 tools/listed.py\n"
        "  ```\n"
        "\n"
        "   ~~~shell\n"
        "   python3 tools/three-space.py\n"
        "   ~~~\n"
        "\n"
        "    ```bash\n"
        "    python3 tools/indented-code.py\n"
        "    ```\n"
    )

    blocks = cdc.extract_shell_blocks(Path("README.md"), content)

    assert [(block.line, block.language, block.body) for block in blocks] == [
        (1, "bash", "python3 tools/quoted.py\n"),
        (5, "sh", "python3 tools/listed.py\n"),
        (9, "shell", "python3 tools/three-space.py\n"),
    ]


def test_commonmark_four_space_list_continuation_fences_are_shell_blocks() -> None:
    content = (
        "- Run the check:\n"
        "\n"
        "    ```bash\n"
        "    python3 tools/listed.py\n"
        "    ```\n"
        "\n"
        "> - Run the quoted check:\n"
        ">\n"
        ">     ```sh\n"
        ">     python3 tools/quoted-list.py\n"
        ">     ```\n"
    )

    blocks = cdc.extract_shell_blocks(Path("README.md"), content)

    assert [(block.line, block.language, block.body) for block in blocks] == [
        (3, "bash", "python3 tools/listed.py\n"),
        (9, "sh", "python3 tools/quoted-list.py\n"),
    ]


def test_extract_inline_commands_skips_fences_and_non_commands() -> None:
    content = (
        "Run `python3 tools/check.py --help`, not `PipelineFactory::load()`.\n"
        "```bash\n"
        "python3 tools/fenced.py\n"
        "```\n"
    )

    commands = cdc.extract_inline_commands(Path("README.md"), content)

    assert [command.body for command in commands] == ["python3 tools/check.py --help\n"]


def test_extract_inline_commands_supports_arbitrary_backtick_run_lengths() -> None:
    content = (
        "Run ``python3 tools/check.py --value `literal` tail`` and "
        "```python3 tools/other.py --value ``nested`` tail```.\n"
    )

    commands = cdc.extract_inline_commands(Path("README.md"), content)

    assert [command.body for command in commands] == [
        "python3 tools/check.py --value `literal` tail\n",
        "python3 tools/other.py --value ``nested`` tail\n",
    ]


def test_extract_inline_commands_supports_multiline_code_spans() -> None:
    content = (
        "Intro.\n"
        "Run ``python3 tools/check.py\n"
        "--value `literal` tail`` after the build.\n"
        "```bash\n"
        "python3 tools/fenced.py\n"
        "```\n"
    )

    commands = cdc.extract_inline_commands(Path("README.md"), content)

    assert [(command.line, command.body) for command in commands] == [
        (2, "python3 tools/check.py --value `literal` tail\n")
    ]


def test_extract_inline_commands_does_not_cross_paragraph_boundaries() -> None:
    content = (
        "Run `python3 tools/definitely-missing.py\n"
        "\n"
        "This is a separate paragraph ending with `ordinary prose.\n"
    )

    assert cdc.extract_inline_commands(Path("README.md"), content) == []


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


def test_literal_shell_c_payloads_reuse_cli_contracts_and_source_lines() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    direct = cdc.ShellBlock(
        path=Path("README.md"),
        line=40,
        language="bash",
        body=(
            "# Run in a clean shell.\n"
            "bash -c 'python3 tools/trtmc_validate.py --definitely-invalid'\n"
        ),
    )
    docker = cdc.ShellBlock(
        path=Path("README.md"),
        line=70,
        language="bash",
        body=(
            "docker exec --env MODE=test trtmc-dev /bin/sh -c "
            "'python3 tools/trtmc_validate.py --definitely-invalid'\n"
        ),
    )

    direct_findings = cdc.check_command_block(direct, repo_root)
    docker_findings = cdc.check_command_block(docker, repo_root)

    assert [(finding.line, finding.message) for finding in direct_findings] == [
        (
            42,
            "unknown option for `tools/trtmc_validate.py`: --definitely-invalid",
        )
    ]
    assert [(finding.line, finding.message) for finding in docker_findings] == [
        (
            71,
            "unknown option for `tools/trtmc_validate.py`: --definitely-invalid",
        )
    ]


def test_literal_shell_c_payload_syntax_and_local_inputs_are_checked(
    tmp_path: Path,
) -> None:
    syntax = cdc.ShellBlock(
        path=Path("README.md"),
        line=20,
        language="bash",
        body="bash -lc 'if true; then'\n",
    )
    missing = cdc.ShellBlock(
        path=Path("README.md"),
        line=30,
        language="bash",
        body="sh -c 'python3 tools/definitely-missing.py --help'\n",
    )

    syntax_findings = cdc.check_command_block(syntax, tmp_path)
    missing_findings = cdc.check_command_block(missing, tmp_path)

    assert len(syntax_findings) == 1
    assert syntax_findings[0].line == 21
    assert syntax_findings[0].message.startswith("invalid shell syntax:")
    assert [(finding.line, finding.message) for finding in missing_findings] == [
        (31, "command input does not exist: tools/definitely-missing.py")
    ]


def test_valid_and_dynamic_shell_c_wrappers_are_safe() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    valid = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body=(
            "bash -lc 'python3 tools/trtmc_validate.py --list'\n"
            "docker exec trtmc-dev bash -c "
            "'python3 tools/trtmc_validate.py --all --dry-run'\n"
        ),
    )
    dynamic = cdc.ShellBlock(
        path=Path("README.md"),
        line=10,
        language="bash",
        body=(
            'bash -c "$VALIDATION_COMMAND"\ndocker exec trtmc-dev sh -c "${VALIDATION_COMMAND}"\n'
        ),
    )

    assert cdc.check_command_block(valid, repo_root) == []
    assert cdc.check_command_block(dynamic, repo_root) == []
    assert list(cdc.shell_validation_blocks(dynamic)) == [dynamic]


def test_nested_shell_c_recursion_is_depth_bounded() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    payload = "python3 tools/trtmc_validate.py --definitely-invalid"
    for _level in range(3):
        payload = f"bash -c {shlex.quote(payload)}"
    block = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body=f"{payload}\n",
    )

    bounded = list(cdc.shell_validation_blocks(block, max_nested_depth=2))

    assert len(bounded) == 3
    assert cdc.check_command_block(block, repo_root, max_nested_depth=2) == []
    assert [
        finding.message
        for finding in cdc.check_command_block(
            block,
            repo_root,
            max_nested_depth=3,
        )
    ] == ["unknown option for `tools/trtmc_validate.py`: --definitely-invalid"]


def test_nested_shell_c_recursion_rejects_an_ancestor_cycle(monkeypatch) -> None:
    block = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body="echo literal\n",
    )
    monkeypatch.setattr(
        cdc,
        "_nested_shell_payload",
        lambda _tokens: block.body,
    )

    assert list(cdc.shell_validation_blocks(block)) == [block]


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


def test_command_finding_uses_physical_line_across_blanks_comments_and_continuation(
    tmp_path: Path,
) -> None:
    block = cdc.ShellBlock(
        path=Path("README.md"),
        line=10,
        language="bash",
        body=("# explanatory comment\n\npython3 tools/missing.py \\\n  --help\n"),
    )

    findings = cdc.check_local_inputs(block, tmp_path)

    assert len(findings) == 1
    assert findings[0].line == 13


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
        'build_p = sub.add_parser("build")\n'
        'build_p.add_argument("model")\n'
        'build_p.add_argument("-o", "--output")\n',
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


def test_python_script_contract_checks_nargs_positional_choices(
    tmp_path: Path,
) -> None:
    script = tmp_path / "tools" / "profile.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        'parser.add_argument("modes", nargs="+", choices=["fast", "safe"])\n'
        'parser.add_argument("output")\n',
        encoding="utf-8",
    )
    block = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body="python3 tools/profile.py fast broken reports/result.json\n",
    )

    findings = cdc.check_python_script_contract(block, tmp_path)

    assert [finding.message for finding in findings] == [
        "invalid value for positional `modes` for `tools/profile.py`: "
        "broken; expected one of fast, safe"
    ]


def test_python_script_negative_numbers_match_default_argparse(
    tmp_path: Path,
) -> None:
    script = tmp_path / "tools" / "profile.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        'import argparse\nparser = argparse.ArgumentParser()\nparser.add_argument("--threshold")\n',
        encoding="utf-8",
    )
    decimal = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body="python3 tools/profile.py --threshold -.5\n",
    )
    scientific = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body="python3 tools/profile.py --threshold -1e-3\n",
    )

    assert cdc.check_python_script_contract(decimal, tmp_path) == []
    assert [
        finding.message for finding in cdc.check_python_script_contract(scientific, tmp_path)
    ] == [
        "option for `tools/profile.py` requires a value: --threshold",
        "unknown option for `tools/profile.py`: -1e-3",
    ]


def test_python_script_contract_accepts_attached_short_option_value(
    tmp_path: Path,
) -> None:
    script = tmp_path / "tools" / "generate.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        'parser.add_argument("-o", "--output")\n',
        encoding="utf-8",
    )
    block = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body='python3 tools/generate.py -o"reports/generated.json"\n',
    )

    assert cdc.check_python_script_contract(block, tmp_path) == []


def test_python_script_inline_values_match_argparse_arity_and_empty_string(
    tmp_path: Path,
) -> None:
    script = tmp_path / "tools" / "inline_values.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        'parser.add_argument("--pair", nargs=2)\n'
        'parser.add_argument("--name", choices=("", "set"))\n'
        'parser.add_argument("--items", nargs="+")\n'
        'parser.add_argument("--flag", action="store_true")\n'
        "parser.parse_args()\n",
        encoding="utf-8",
    )

    valid = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/inline_values.py --pair left right --name= --items=\n",
    )
    short_pair = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/inline_values.py --pair=left\n",
    )
    flag_value = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/inline_values.py --flag=\n",
    )

    assert cdc.check_python_script_contract(valid, tmp_path) == []
    assert [
        finding.message for finding in cdc.check_python_script_contract(short_pair, tmp_path)
    ] == ["option for `tools/inline_values.py` requires a value: --pair"]
    assert [
        finding.message for finding in cdc.check_python_script_contract(flag_value, tmp_path)
    ] == ["option for `tools/inline_values.py` does not take a value: --flag"]


def test_argparse_subcommand_aliases_share_options_and_nested_contracts(
    tmp_path: Path,
) -> None:
    script = tmp_path / "tools" / "aliases.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "commands = parser.add_subparsers(required=True)\n"
        'checkout = commands.add_parser("checkout", aliases=["co"])\n'
        'checkout.add_argument("--mode", required=True)\n'
        "children = checkout.add_subparsers(required=True)\n"
        'add = children.add_parser("add", aliases=["a"])\n'
        'add.add_argument("--name", required=True)\n'
        "parser.parse_args()\n",
        encoding="utf-8",
    )
    valid = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/aliases.py co --mode safe a --name item\n",
    )
    missing = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/aliases.py co a --name item\n",
    )

    assert cdc.check_python_script_contract(valid, tmp_path) == []
    assert [finding.message for finding in cdc.check_python_script_contract(missing, tmp_path)] == [
        "missing required option for `tools/aliases.py co`: --mode"
    ]


def test_argparse_parent_parsers_are_merged_at_root_and_subcommand(
    tmp_path: Path,
) -> None:
    script = tmp_path / "tools" / "parents.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import argparse\n"
        "common = argparse.ArgumentParser(add_help=False)\n"
        'common.add_argument("--profile", choices=("fast", "safe"), required=True)\n'
        "run_common = argparse.ArgumentParser(add_help=False)\n"
        'run_common.add_argument("--count", type=int, required=True)\n'
        "parser = argparse.ArgumentParser(parents=[common])\n"
        "commands = parser.add_subparsers(required=True)\n"
        'commands.add_parser("run", parents=[run_common])\n'
        "parser.parse_args()\n",
        encoding="utf-8",
    )
    valid = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/parents.py --profile fast run --count 2\n",
    )
    invalid_choice = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/parents.py --profile broken run --count 2\n",
    )
    missing_child = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/parents.py --profile fast run\n",
    )

    assert cdc.check_python_script_contract(valid, tmp_path) == []
    assert [
        finding.message for finding in cdc.check_python_script_contract(invalid_choice, tmp_path)
    ] == ["invalid value for `tools/parents.py --profile`: broken; expected one of fast, safe"]
    assert [
        finding.message for finding in cdc.check_python_script_contract(missing_child, tmp_path)
    ] == ["missing required option for `tools/parents.py run`: --count"]


def test_dynamic_curly_paths_are_not_treated_as_literal_inputs(
    tmp_path: Path,
) -> None:
    block = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        ("python3 tools/{family}/validate.py --help\npython3 tools/definitely-missing.py --help\n"),
    )

    assert [finding.message for finding in cdc.check_local_inputs(block, tmp_path)] == [
        "command input does not exist: tools/definitely-missing.py"
    ]


def test_local_python_module_contract_checks_nested_subcommands(
    tmp_path: Path,
) -> None:
    package = tmp_path / "tools" / "ci"
    package.mkdir(parents=True)
    (tmp_path / "tools" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "__main__.py").write_text(
        "import argparse\n"
        "class Command:\n"
        "    def __init__(self):\n"
        "        self.parser = argparse.ArgumentParser()\n"
        '        commands = self.parser.add_subparsers(dest="command", required=True)\n'
        '        image = commands.add_parser("image")\n'
        "        image_commands = image.add_subparsers(required=True)\n"
        '        image_commands.add_parser("ensure")\n'
        '        coverage = commands.add_parser("coverage")\n'
        '        coverage.add_argument("language", choices=("cpp", "python"))\n',
        encoding="utf-8",
    )
    valid = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body="python3 -m tools.ci image ensure\n",
    )
    invalid_nested = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body="python3 -m tools.ci image removed\n",
    )
    invalid_choice = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body="python3 -m tools.ci coverage rust\n",
    )

    assert cdc.check_python_module_contract(valid, tmp_path) == []
    assert [
        finding.message for finding in cdc.check_python_module_contract(invalid_nested, tmp_path)
    ] == ["unknown subcommand for `python -m tools.ci image`: removed"]
    assert [
        finding.message for finding in cdc.check_python_module_contract(invalid_choice, tmp_path)
    ] == [
        "invalid value for positional `language` for `python -m tools.ci coverage`: "
        "rust; expected one of cpp, python"
    ]


def test_argparse_contract_expands_literal_loop_subcommands(
    tmp_path: Path,
) -> None:
    script = tmp_path / "tools" / "matrix.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "commands = parser.add_subparsers(required=True)\n"
        "for name, help_text in (\n"
        '    ("check", "validate entries"),\n'
        '    ("run", "execute entries"),\n'
        "):\n"
        "    command = commands.add_parser(name, help=help_text)\n"
        '    command.add_argument("suite")\n'
        '    command.add_argument("--environment", required=True)\n'
        '    command.add_argument("--entry", action="append")\n'
        'resume = commands.add_parser("resume")\n'
        'resume.add_argument("run_directory")\n',
        encoding="utf-8",
    )
    check = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body=(
            "python3 tools/matrix.py check benchmarks/suite.yaml "
            "--environment benchmarks/environment.yaml --entry alpha\n"
        ),
    )
    run = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body=(
            "python3 tools/matrix.py run benchmarks/suite.yaml "
            "--environment benchmarks/environment.yaml\n"
        ),
    )
    missing_environment = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body="python3 tools/matrix.py check benchmarks/suite.yaml\n",
    )
    missing_suite = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body=("python3 tools/matrix.py run --environment benchmarks/environment.yaml\n"),
    )
    wrong_scope = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body="python3 tools/matrix.py resume artifacts/run --entry alpha\n",
    )

    assert cdc.check_python_script_contract(check, tmp_path) == []
    assert cdc.check_python_script_contract(run, tmp_path) == []
    assert [
        finding.message
        for finding in cdc.check_python_script_contract(
            missing_environment,
            tmp_path,
        )
    ] == ["missing required option for `tools/matrix.py check`: --environment"]
    assert [
        finding.message for finding in cdc.check_python_script_contract(missing_suite, tmp_path)
    ] == ["missing required positional for `tools/matrix.py run`: suite"]
    assert [
        finding.message for finding in cdc.check_python_script_contract(wrong_scope, tmp_path)
    ] == ["unknown option for `tools/matrix.py resume`: --entry"]


def test_argparse_literal_loop_preserves_each_argument_binding(
    tmp_path: Path,
) -> None:
    script = tmp_path / "tools" / "bound_matrix.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "commands = parser.add_subparsers(required=True)\n"
        "for name, flag, values, needed, arity in (\n"
        '    ("check", "--mode", ("fast", "safe"), True, 1),\n'
        '    ("run", "--tag", ("gpu", "cpu"), False, "+"),\n'
        "):\n"
        "    command = commands.add_parser(name)\n"
        "    command.add_argument(\n"
        "        flag, choices=values, required=needed, nargs=arity\n"
        "    )\n"
        "parser.parse_args()\n",
        encoding="utf-8",
    )
    valid_check = cdc.ShellBlock(
        Path("README.md"), 1, "bash", "python3 tools/bound_matrix.py check --mode fast\n"
    )
    valid_run = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/bound_matrix.py run --tag gpu cpu\n",
    )
    missing = cdc.ShellBlock(Path("README.md"), 1, "bash", "python3 tools/bound_matrix.py check\n")
    wrong_choice = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/bound_matrix.py run --tag safe\n",
    )
    crossed = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/bound_matrix.py run --mode fast\n",
    )

    assert cdc.check_python_script_contract(valid_check, tmp_path) == []
    assert cdc.check_python_script_contract(valid_run, tmp_path) == []
    assert [finding.message for finding in cdc.check_python_script_contract(missing, tmp_path)] == [
        "missing required option for `tools/bound_matrix.py check`: --mode"
    ]
    assert [
        finding.message for finding in cdc.check_python_script_contract(wrong_choice, tmp_path)
    ] == ["invalid value for `tools/bound_matrix.py run --tag`: safe; expected one of cpu, gpu"]
    assert [finding.message for finding in cdc.check_python_script_contract(crossed, tmp_path)] == [
        "unknown option for `tools/bound_matrix.py run`: --mode"
    ]


def test_argparse_selects_reachable_parse_root_without_dead_parser_pollution(
    tmp_path: Path,
) -> None:
    script = tmp_path / "tools" / "scoped.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import argparse\n"
        "def dead():\n"
        "    parser = argparse.ArgumentParser()\n"
        '    parser.add_argument("--dead", required=True)\n'
        "    parser.parse_args()\n"
        "    return parser\n"
        "def build_parser():\n"
        "    parser = argparse.ArgumentParser()\n"
        '    parser.add_argument("--live", required=True)\n'
        "    return parser\n"
        "def main():\n"
        "    selected = build_parser()\n"
        "    return selected.parse_args()\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )
    live = cdc.ShellBlock(Path("README.md"), 1, "bash", "python3 tools/scoped.py --live yes\n")
    dead = cdc.ShellBlock(Path("README.md"), 1, "bash", "python3 tools/scoped.py --dead yes\n")

    assert cdc.check_python_script_contract(live, tmp_path) == []
    assert [finding.message for finding in cdc.check_python_script_contract(dead, tmp_path)] == [
        "unknown option for `tools/scoped.py`: --dead",
        "missing required option for `tools/scoped.py`: --live",
    ]


def test_argparse_helper_arguments_attach_to_the_called_parser(
    tmp_path: Path,
) -> None:
    script = tmp_path / "tools" / "helper_args.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import argparse\n"
        "def add_common_args(parser):\n"
        '    parser.add_argument("--remote")\n'
        "def build_parser():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    add_common_args(parser)\n"
        "    commands = parser.add_subparsers(required=True)\n"
        '    commands.add_parser("list")\n'
        "    return parser\n"
        "def main():\n"
        "    return build_parser().parse_args()\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )
    block = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/helper_args.py --remote $DOC_REMOTE list\n",
    )

    assert cdc.check_python_script_contract(block, tmp_path) == []


def test_parse_known_args_allows_unknowns_but_checks_known_contracts(
    tmp_path: Path,
) -> None:
    script = tmp_path / "tools" / "known.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        'parser.add_argument("--mode", choices=("fast", "safe"), required=True)\n'
        "parser.parse_known_args()\n",
        encoding="utf-8",
    )
    extras = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/known.py --mode fast extra --future value\n",
    )
    invalid = cdc.ShellBlock(Path("README.md"), 1, "bash", "python3 tools/known.py --mode broken\n")

    assert cdc.check_python_script_contract(extras, tmp_path) == []
    assert [finding.message for finding in cdc.check_python_script_contract(invalid, tmp_path)] == [
        "invalid value for `tools/known.py --mode`: broken; expected one of fast, safe"
    ]


def test_argparse_subparsers_required_assignment_covers_nested_levels(
    tmp_path: Path,
) -> None:
    script = tmp_path / "tools" / "assigned_required.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "commands = parser.add_subparsers()\n"
        "commands.required = True\n"
        'parent = commands.add_parser("parent")\n'
        "children = parent.add_subparsers()\n"
        "children.required = True\n"
        'children.add_parser("child")\n'
        "parser.parse_args()\n",
        encoding="utf-8",
    )
    root_missing = cdc.ShellBlock(
        Path("README.md"), 1, "bash", "python3 tools/assigned_required.py\n"
    )
    child_missing = cdc.ShellBlock(
        Path("README.md"), 1, "bash", "python3 tools/assigned_required.py parent\n"
    )
    valid = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/assigned_required.py parent child\n",
    )

    assert [
        finding.message for finding in cdc.check_python_script_contract(root_missing, tmp_path)
    ] == ["missing required subcommand for `tools/assigned_required.py`"]
    assert [
        finding.message for finding in cdc.check_python_script_contract(child_missing, tmp_path)
    ] == ["missing required subcommand for `tools/assigned_required.py parent`"]
    assert cdc.check_python_script_contract(valid, tmp_path) == []


def test_argparse_strict_parser_rejects_extras_without_positionals(
    tmp_path: Path,
) -> None:
    script = tmp_path / "tools" / "strict.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import argparse\nparser = argparse.ArgumentParser()\nparser.parse_args()\n",
        encoding="utf-8",
    )

    for suffix, expected in (
        ("extra", "extra"),
        ("--", "--"),
        ("-- extra", "--"),
    ):
        block = cdc.ShellBlock(
            Path("README.md"),
            1,
            "bash",
            f"python3 tools/strict.py {suffix}\n",
        )
        assert [
            finding.message for finding in cdc.check_python_script_contract(block, tmp_path)
        ] == [f"unexpected positional argument for `tools/strict.py`: {expected}"]


def test_argparse_short_clusters_and_attached_value_match_runtime(
    tmp_path: Path,
) -> None:
    script = tmp_path / "tools" / "clustered.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        'parser.add_argument("-v", action="store_true", required=True)\n'
        'parser.add_argument("-q", action="store_true")\n'
        'parser.add_argument("-o", required=True)\n'
        "parser.parse_args()\n",
        encoding="utf-8",
    )
    valid = cdc.ShellBlock(
        Path("README.md"), 1, "bash", "python3 tools/clustered.py -vqoout.json\n"
    )
    unknown = cdc.ShellBlock(
        Path("README.md"), 1, "bash", "python3 tools/clustered.py -vxoout.json\n"
    )

    assert cdc.check_python_script_contract(valid, tmp_path) == []
    assert [finding.message for finding in cdc.check_python_script_contract(unknown, tmp_path)] == [
        "unknown option for `tools/clustered.py`: -vxoout.json",
        "missing required option for `tools/clustered.py`: -v",
        "missing required option for `tools/clustered.py`: -o",
    ]


def test_shell_output_redirections_do_not_reach_cli_contract_and_input_is_checked(
    tmp_path: Path,
) -> None:
    script = tmp_path / "tools" / "redirected.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        'parser.add_argument("--value", required=True)\n'
        "parser.parse_args()\n",
        encoding="utf-8",
    )
    outputs = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/redirected.py --value ok "
        '> "reports/out.txt" 1>>reports/one.txt '
        "2> reports/two.txt &>reports/all.txt\n",
    )
    missing_input = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/redirected.py --value ok < tests/fixtures/missing-input.txt\n",
    )

    assert cdc.check_python_script_contract(outputs, tmp_path) == []
    assert cdc.check_local_inputs(outputs, tmp_path) == []
    assert [finding.message for finding in cdc.check_local_inputs(missing_input, tmp_path)] == [
        "command input does not exist: tests/fixtures/missing-input.txt"
    ]


def test_process_substitution_inner_commands_and_inputs_are_checked(
    tmp_path: Path,
) -> None:
    block = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "diff <(python3 tools/definitely-missing.py) <(cat tests/also-missing.json)\n",
    )

    assert [finding.message for finding in cdc.check_local_inputs(block, tmp_path)] == [
        "command input does not exist: tools/definitely-missing.py",
        "command input does not exist: tests/also-missing.json",
    ]


def test_nested_argparse_checks_parent_required_options_only_at_parent_level(
    tmp_path: Path,
) -> None:
    script = tmp_path / "tools" / "nested.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        'commands = parser.add_subparsers(dest="command", required=True)\n'
        'parent = commands.add_parser("parent")\n'
        'parent.add_argument("--profile", required=True, choices=("fast", "safe"))\n'
        "children = parent.add_subparsers(required=True)\n"
        'child = children.add_parser("child")\n'
        'child.add_argument("--count")\n',
        encoding="utf-8",
    )
    exact = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body="python3 tools/nested.py parent --profile fast child\n",
    )
    combined = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body=("python3 tools/nested.py parent --profile safe child --count 2\n"),
    )
    missing_parent = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body="python3 tools/nested.py parent child --count 2\n",
    )

    assert cdc.check_python_script_contract(exact, tmp_path) == []
    assert cdc.check_python_script_contract(combined, tmp_path) == []
    assert [
        finding.message for finding in cdc.check_python_script_contract(missing_parent, tmp_path)
    ] == ["missing required option for `tools/nested.py parent`: --profile"]


def test_argparse_root_positionals_are_consumed_before_subcommand(
    tmp_path: Path,
) -> None:
    script = tmp_path / "tools" / "root_positionals.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        'parser.add_argument("--profile", choices=("fast", "safe"))\n'
        'parser.add_argument("environments", nargs="+", '
        'choices=("default", "staging"))\n'
        "commands = parser.add_subparsers(required=True)\n"
        'run = commands.add_parser("run")\n'
        'run.add_argument("target", choices=("deploy", "test"))\n',
        encoding="utf-8",
    )
    exact = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body="python3 tools/root_positionals.py default run deploy\n",
    )
    combined = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body=("python3 tools/root_positionals.py --profile fast default staging run deploy\n"),
    )
    invalid_choice = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body=("python3 tools/root_positionals.py default broken --profile safe run test\n"),
    )

    assert cdc.check_python_script_contract(exact, tmp_path) == []
    assert cdc.check_python_script_contract(combined, tmp_path) == []
    assert [
        finding.message for finding in cdc.check_python_script_contract(invalid_choice, tmp_path)
    ] == [
        "invalid value for positional `environments` for "
        "`tools/root_positionals.py`: broken; expected one of default, staging"
    ]


def test_argparse_negative_root_positional_is_consumed_before_subcommand(
    tmp_path: Path,
) -> None:
    script = tmp_path / "tools" / "root_threshold.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        'parser.add_argument("threshold", type=float)\n'
        "commands = parser.add_subparsers(required=True)\n"
        'run = commands.add_parser("run")\n'
        'run.add_argument("target", choices=("deploy", "test"))\n',
        encoding="utf-8",
    )
    block = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body="python3 tools/root_threshold.py -.5 run test\n",
    )

    assert cdc.check_python_script_contract(block, tmp_path) == []


def test_argparse_double_dash_is_not_skipped_before_required_subcommand(
    tmp_path: Path,
) -> None:
    script = tmp_path / "tools" / "queue.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "commands = parser.add_subparsers(required=True)\n"
        'commands.add_parser("run")\n',
        encoding="utf-8",
    )
    block = cdc.ShellBlock(
        path=Path("README.md"),
        line=1,
        language="bash",
        body="python3 tools/queue.py -- run\n",
    )

    assert [finding.message for finding in cdc.check_python_script_contract(block, tmp_path)] == [
        "unknown subcommand for `tools/queue.py`: --"
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


def test_docs_validation_workflow_gates_are_discoverable_in_docs_and_skill() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    workflow = (repo_root / ".github/workflows/docs-validation.yml").read_text(encoding="utf-8")
    human_docs = (repo_root / "website/docs/reference/testing.md").read_text(encoding="utf-8")
    agent_skill = (repo_root / "plugins/trtmc-agent-skills/skills/doc-sync/SKILL.md").read_text(
        encoding="utf-8"
    )
    gates = (
        "tests/tools/test_check_doc_file_references.py",
        "tests/tools/test_check_doc_commands.py",
        "tests/tools/test_runtime_strategy_matrix_checker.py",
        "tests/tools/test_model_owned_validation_scripts.py",
        "tests/tools/test_test_impact.py",
        "tests/tools/test_trtmc_validate.py",
        "tests/tools/test_perf_matrix.py::test_release_suite_covers_every_non_l0_ready_model_profile",
        "tools/test_impact.py --validate",
        "tools/check_doc_file_references.py --strict --tracked",
        "tools/check_doc_commands.py",
        "tools/check_runtime_strategy_matrix.py",
    )

    for gate in gates:
        assert gate in workflow, f"workflow is missing documentation gate: {gate}"
        assert gate in human_docs, f"testing reference is missing workflow gate: {gate}"
        assert gate in agent_skill, f"doc-sync skill is missing workflow gate: {gate}"
    assert "npm ci" in workflow
    assert "npm run build" in workflow
    for surface in (human_docs, agent_skill):
        assert "npm --prefix website ci" in surface
        assert "npm --prefix website run build" in surface


def test_public_config_docs_match_pipeline_factory_contribution_layers() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    resolver_implementation = (repo_root / "src/runtime/config/cli_support.cpp").read_text(
        encoding="utf-8"
    )
    function_start = resolver_implementation.index(
        "PipelineConfigResolution resolve_pipeline_config"
    )
    function_end = resolver_implementation.index(
        "std::string write_effective_config_next_to",
        function_start,
    )
    resolver = resolver_implementation[function_start:function_end]

    assert "bundle_defaults_contribution" in resolver
    assert "Layer::SessionRequest" in resolver
    assert "Layer::BuildTime" not in resolver
    assert "Layer::PlatformProfile" not in resolver

    factory_implementation = (repo_root / "src/runtime/registry/pipeline_factory.cpp").read_text(
        encoding="utf-8"
    )
    materialization_start = factory_implementation.index(
        "const BundleSectionInfo* find_nonempty_config_section"
    )
    materialization_end = factory_implementation.index(
        "} // namespace detail",
        materialization_start,
    )
    materialization = factory_implementation[materialization_start:materialization_end]
    assert 'entry.name == "config.json"' in materialization
    assert "const auto info = ReadBundleHeader(bundle_path)" in materialization
    assert "ReadBundleSection(bundle_path, *config_info)" in materialization
    assert "std::string config_text(config_data.begin(), config_data.end())" in materialization

    runtime_resolution_start = factory_implementation.index("detail::resolve_runtime_config")
    runtime_resolution_end = factory_implementation.index(
        "std::unique_ptr<IPipeline> PipelineFactory::from_bundle",
        runtime_resolution_start,
    )
    runtime_resolution = factory_implementation[runtime_resolution_start:runtime_resolution_end]
    assert "resolve_pipeline_config(config_text, config_path, set_tokens)" in runtime_resolution

    factory_start = factory_implementation.index(
        "std::unique_ptr<IPipeline> PipelineFactory::from_bundle"
    )
    factory_body = factory_implementation[factory_start:]
    assert "std::string config_text = std::move(materialized.config_text)" in factory_body
    assert "detail::resolve_runtime_config(config_text" in factory_body

    public_surfaces = (
        repo_root / "include/trtmc/config/cli_support.h",
        repo_root / "include/trtmc/config/schema_registry.h",
        repo_root / "website/docs/features/config-and-backends.md",
        repo_root / "website/docs/unit-design/python-builder.md",
        repo_root / "website/docs/unit-design/building-blocks.md",
    )
    for path in public_surfaces:
        compact = " ".join(path.read_text(encoding="utf-8").split())
        assert "config.json" in compact
        assert "BundleDefault" in compact
        assert "SessionRequest" in compact
        assert "binary header" in compact
        assert "PlatformProfile" in compact
    cli_support = " ".join(public_surfaces[0].read_text(encoding="utf-8").split())
    assert "materialized ``config.json`` section" in cli_support
    assert "bundle's raw header JSON" not in cli_support
    assert "resolve_pipeline_config(const std::string& config_json" in cli_support

    for path in public_surfaces[3:]:
        compact = " ".join(path.read_text(encoding="utf-8").split())
        assert "does not automatically" in compact

    architecture = " ".join(
        (repo_root / "website/docs/wiki/Architecture-Overview.md")
        .read_text(encoding="utf-8")
        .split()
    )
    assert "passes the materialized `config.json` section" in architecture
    assert "does not pass `BundleInfo.defaults` from the binary header" in architecture
    assert "No separate build-time or platform-profile contribution is wired" in architecture


def test_public_visual_docs_cover_native_and_optimized_bundle_paths() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    system_map_path = repo_root / "website/static/img/diagrams/trtmc-system-map.svg"
    system_map = system_map_path.read_text(encoding="utf-8")

    assert "website/static/img/diagrams/trtmc-system-map.svg" in readme
    assert "trtmc-landing.png" not in readme
    assert not (repo_root / "website/static/img/trtmc-landing.png").exists()
    for required in (
        "trtmc build",
        "runtime_strategy",
        "optimized_runtime.json",
        "model + backend DSO",
        "implementation DSO",
        "NVIDIA driver",
        "CUDA/TensorRT",
        "system libraries",
    ):
        assert required in system_map
    assert "trtmc-build" not in system_map
    assert "everything the runtime needs" not in system_map


def test_diagram_policy_and_beginner_block_count_are_self_consistent() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    research = " ".join(
        (repo_root / "website/docs/reference/documentation-research.md")
        .read_text(encoding="utf-8")
        .split()
    )
    building_blocks = " ".join(
        (repo_root / "website/docs/unit-design/building-blocks.md")
        .read_text(encoding="utf-8")
        .split()
    )

    assert "not automatic fallbacks for Mermaid blocks" in research
    assert "Core diagrams have static SVG versions" not in research
    assert "these eight blocks" in building_blocks
    assert "those seven blocks" not in building_blocks
