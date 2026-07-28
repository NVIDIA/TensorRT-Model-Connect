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
import signal
import subprocess
import sys
import textwrap

import pytest

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


def test_literal_env_wrapped_shell_payloads_reuse_cli_contracts() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    payload = "python3 tools/trtmc_validate.py --definitely-invalid"
    bodies = (
        f"env -i bash -c {shlex.quote(payload)}\n",
        f"env - sh -c {shlex.quote(payload)}\n",
        (f"/usr/bin/env --ignore-environment MODE=test sh -c {shlex.quote(payload)}\n"),
        f"env -u HOME bash -c {shlex.quote(payload)}\n",
        f"env --unset=HOME sh -c {shlex.quote(payload)}\n",
        (f"MODE=test env -i /usr/bin/env --unset HOME -- bash -c {shlex.quote(payload)}\n"),
        (f"docker exec trtmc-dev /usr/bin/env --unset HOME bash -c {shlex.quote(payload)}\n"),
    )

    for body in bodies:
        block = cdc.ShellBlock(
            path=Path("README.md"),
            line=10,
            language="bash",
            body=body,
        )
        assert [finding.message for finding in cdc.check_command_block(block, repo_root)] == [
            "unknown option for `tools/trtmc_validate.py`: --definitely-invalid"
        ]


def test_dynamic_and_unknown_env_wrappers_fail_closed() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    payload = shlex.quote("python3 tools/trtmc_validate.py --definitely-invalid")
    block = cdc.ShellBlock(
        path=Path("README.md"),
        line=10,
        language="bash",
        body=(
            f'env -S "$ENV_ARGS" bash -c {payload}\n'
            f'env --unset "$ENV_NAME" bash -c {payload}\n'
            f'env --chdir "$WORKDIR" bash -c {payload}\n'
            f"env --implementation-specific bash -c {payload}\n"
        ),
    )

    assert [finding.message for finding in cdc.check_command_block(block, repo_root)] == [
        "cannot statically resolve shell wrapper: unsupported or dynamic `env` wrapper options",
        "cannot statically resolve shell wrapper: unsupported or dynamic `env` wrapper options",
        "cannot statically resolve shell wrapper: unsupported or dynamic `env` wrapper options",
        "cannot statically resolve shell wrapper: unsupported or dynamic `env` wrapper options",
    ]
    assert list(cdc.shell_validation_blocks(block)) == [block]

    inline = cdc.extract_inline_commands(
        Path("README.md"),
        'Run `env --split-string="$ENV_ARGS"` after setup.',
    )
    assert len(inline) == 1
    assert [finding.message for finding in cdc.check_command_block(inline[0], repo_root)] == [
        "cannot statically resolve shell wrapper: unsupported or dynamic `env` wrapper options"
    ]


def test_static_env_chdir_wrappers_reuse_nested_cli_contracts() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    payload = shlex.quote("python3 tools/trtmc_validate.py --definitely-invalid")
    root = shlex.quote(str(repo_root))
    bodies = (
        f"env --chdir {root} bash -c {payload}\n",
        f"env --chdir={root} sh -c {payload}\n",
        f"env -C {root} bash -c {payload}\n",
        f"env -C{root} sh -c {payload}\n",
        (f"docker exec trtmc-dev env --chdir {root} bash -c {payload}\n"),
        (f"docker exec trtmc-dev /usr/bin/env -C{root} sh -c {payload}\n"),
    )

    for body in bodies:
        block = cdc.ShellBlock(
            path=Path("README.md"),
            line=10,
            language="bash",
            body=body,
        )
        assert [finding.message for finding in cdc.check_command_block(block, repo_root)] == [
            "unknown option for `tools/trtmc_validate.py`: --definitely-invalid"
        ]


def test_env_chdir_propagates_static_cwd_to_relative_scripts_and_inputs(
    tmp_path: Path,
) -> None:
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    (tools_dir / "cwd_cli.py").write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        'parser.add_argument("--known")\n'
        "parser.parse_args()\n",
        encoding="utf-8",
    )
    (tools_dir / "present.json").write_text("{}\n", encoding="utf-8")
    block = cdc.ShellBlock(
        Path("README.md"),
        10,
        "bash",
        "env -C tools python3 cwd_cli.py --removed\n"
        "env --chdir=tools cat present.json missing.json\n"
        "env -C tools bash -c 'python3 cwd_cli.py --nested-removed'\n",
    )

    assert [
        (finding.line, finding.message) for finding in cdc.check_command_block(block, tmp_path)
    ] == [
        (12, "command input does not exist: tools/missing.json"),
        (11, "unknown option for `tools/cwd_cli.py`: --removed"),
        (13, "unknown option for `tools/cwd_cli.py`: --nested-removed"),
    ]


def test_static_env_chdir_outside_repo_fails_closed() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    block = cdc.ShellBlock(
        Path("README.md"),
        10,
        "bash",
        "env --chdir /tmp python3 relative.py --invalid\n",
    )

    assert [
        (finding.line, finding.message) for finding in cdc.check_command_block(block, repo_root)
    ] == [
        (
            11,
            "command input resolves outside repository and cannot be validated: /tmp/relative.py",
        )
    ]


def test_literal_env_split_string_is_recursively_resolved() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    bodies = (
        ("env -S 'python3 tools/trtmc_validate.py --definitely-invalid'\n"),
        ("env --split-string='env -S \"python3 tools/trtmc_validate.py --definitely-invalid\"'\n"),
        ("env -S '-C . python3 tools/trtmc_validate.py --definitely-invalid'\n"),
    )

    for body in bodies:
        block = cdc.ShellBlock(Path("README.md"), 1, "bash", body)
        assert [finding.message for finding in cdc.check_command_block(block, repo_root)] == [
            "unknown option for `tools/trtmc_validate.py`: --definitely-invalid"
        ]


def test_gnu_env_split_string_escape_boundary_and_unknown_escape(
    tmp_path: Path,
) -> None:
    script = tmp_path / "tools" / "cli.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "from argparse import ArgumentParser\n"
        "parser = ArgumentParser()\n"
        'parser.add_argument("--live", required=True)\n'
        "parser.parse_args()\n",
        encoding="utf-8",
    )
    escaped_boundary = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        r"env -S 'python3\_tools/cli.py --dead yes'" "\n",
    )
    unknown_escape = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        r"env -S 'python3\q tools/cli.py --live yes'" "\n",
    )

    assert [finding.message for finding in cdc.check_command_block(escaped_boundary, tmp_path)] == [
        "unknown option for `tools/cli.py`: --dead",
        "missing required option for `tools/cli.py`: --live",
    ]
    assert [finding.message for finding in cdc.check_command_block(unknown_escape, tmp_path)] == [
        "cannot statically resolve shell wrapper: unsupported or dynamic `env` wrapper options"
    ]


def test_env_chdir_missing_repo_directory_fails_closed(
    tmp_path: Path,
) -> None:
    block = cdc.ShellBlock(
        Path("README.md"),
        4,
        "bash",
        "env -C missing true\n",
    )

    assert [
        (finding.line, finding.message) for finding in cdc.check_command_block(block, tmp_path)
    ] == [
        (5, "command working directory does not exist: missing"),
    ]


def test_env_chdir_missing_or_non_directory_external_hops_fail_closed(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    missing = tmp_path / "missing-cwd"
    not_directory = tmp_path / "cwd-file"
    not_directory.write_text("not a directory\n", encoding="utf-8")
    block = cdc.ShellBlock(
        Path("README.md"),
        4,
        "bash",
        f"env -C {shlex.quote(str(missing))} true\n"
        f"env --chdir={shlex.quote(str(not_directory))} true\n",
    )

    assert [
        (finding.line, finding.message) for finding in cdc.check_command_block(block, repo_root)
    ] == [
        (
            5,
            f"command working directory does not exist: {missing}",
        ),
        (
            6,
            f"command working directory is not a directory: {not_directory}",
        ),
    ]


def test_env_chdir_rejects_invalid_outer_hop_before_valid_final_cwd(
    tmp_path: Path,
) -> None:
    resolution = cdc._resolve_shell_wrappers(["env", "-C", "missing", "env", "-C", "..", "true"])
    block = cdc.ShellBlock(
        Path("README.md"),
        4,
        "bash",
        "env -C missing env -C .. true\n",
    )

    assert resolution.cwd == Path(".")
    assert resolution.cwd_hops == (Path("missing"), Path("."))
    assert [
        (finding.line, finding.message) for finding in cdc.check_command_block(block, tmp_path)
    ] == [
        (5, "command working directory does not exist: missing"),
    ]


def test_env_chdir_valid_external_directory_keeps_input_policy(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    external_cwd = tmp_path / "external-cwd"
    external_cwd.mkdir()
    block = cdc.ShellBlock(
        Path("README.md"),
        4,
        "bash",
        f"env -C {shlex.quote(str(external_cwd))} python3 relative.py\n",
    )

    assert [
        (finding.line, finding.message) for finding in cdc.check_command_block(block, repo_root)
    ] == [
        (
            5,
            "command input resolves outside repository and cannot be "
            f"validated: {external_cwd}/relative.py",
        ),
    ]


def test_python_module_under_static_cwd_is_not_treated_as_a_script_path(
    tmp_path: Path,
) -> None:
    package = tmp_path / "tools" / "pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "__main__.py").write_text("", encoding="utf-8")
    (package / "cli.py").write_text("", encoding="utf-8")
    block = cdc.ShellBlock(
        Path("README.md"),
        4,
        "bash",
        "timeout 5 env -C tools python3 -m pkg.cli\nenv --chdir=tools python3 -m pkg\n",
    )

    assert cdc.check_local_inputs(block, tmp_path) == []


@pytest.mark.parametrize(
    "tokens",
    [
        ["env", "-C", "missing", "-C", ".", "pwd"],
        ["env", "--chdir", "missing", "--chdir", ".", "pwd"],
        ["env", "-Cmissing", "--chdir=.", "pwd"],
        ["env", "-S", "-C missing -C . pwd"],
    ],
)
def test_same_env_layer_uses_only_final_chdir_like_gnu_env(
    tmp_path: Path,
    tokens: list[str],
) -> None:
    completed = subprocess.run(
        tokens,
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
    )
    resolution = cdc._resolve_shell_wrappers(tokens)
    block = cdc.ShellBlock(
        Path("README.md"),
        4,
        "bash",
        f"{shlex.join(tokens)}\n",
    )

    assert completed.returncode == 0, completed.stderr
    assert Path(completed.stdout.strip()) == tmp_path
    assert resolution.cwd == Path(".")
    assert resolution.cwd_hops == (Path("."),)
    assert cdc.check_local_inputs(block, tmp_path) == []


def test_env_chdir_layers_match_gnu_for_override_and_nested_env(
    tmp_path: Path,
) -> None:
    (tmp_path / "x" / "y").mkdir(parents=True)
    (tmp_path / "y").mkdir()
    same_layer = ["env", "-C", "x", "-C", "y", "pwd"]
    nested = ["env", "-C", "x", "env", "-C", "y", "pwd"]
    split_nested = ["env", "-S", "-C x env -C y pwd"]

    for tokens, expected, expected_hops in [
        (same_layer, tmp_path / "y", (Path("y"),)),
        (
            nested,
            tmp_path / "x" / "y",
            (Path("x"), Path("x/y")),
        ),
        (
            split_nested,
            tmp_path / "x" / "y",
            (Path("x"), Path("x/y")),
        ),
    ]:
        completed = subprocess.run(
            tokens,
            cwd=tmp_path,
            capture_output=True,
            check=False,
            text=True,
        )
        resolution = cdc._resolve_shell_wrappers(tokens)
        block = cdc.ShellBlock(
            Path("README.md"),
            4,
            "bash",
            f"{shlex.join(tokens)}\n",
        )

        assert completed.returncode == 0, completed.stderr
        assert Path(completed.stdout.strip()) == expected
        assert resolution.cwd_hops == expected_hops
        assert cdc.check_local_inputs(block, tmp_path) == []


def test_env_chdir_final_hop_and_external_override_match_gnu(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    missing = tmp_path / "missing"
    invalid = ["env", "-C", ".", "-C", str(missing), "pwd"]
    overridden = [
        "env",
        "-C",
        str(missing),
        "-C",
        str(external),
        "pwd",
    ]

    invalid_run = subprocess.run(
        invalid,
        cwd=repo_root,
        capture_output=True,
        check=False,
        text=True,
    )
    overridden_run = subprocess.run(
        overridden,
        cwd=repo_root,
        capture_output=True,
        check=False,
        text=True,
    )
    invalid_block = cdc.ShellBlock(
        Path("README.md"),
        4,
        "bash",
        f"{shlex.join(invalid)}\n",
    )
    overridden_block = cdc.ShellBlock(
        Path("README.md"),
        4,
        "bash",
        f"{shlex.join(overridden)}\n",
    )

    assert invalid_run.returncode != 0
    assert [finding.message for finding in cdc.check_local_inputs(invalid_block, repo_root)] == [
        f"command working directory does not exist: {missing}",
    ]
    assert overridden_run.returncode == 0, overridden_run.stderr
    assert Path(overridden_run.stdout.strip()) == external
    assert cdc.check_local_inputs(overridden_block, repo_root) == []


def test_static_env_wrappers_preserve_valid_nested_commands(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    external_cwd = tmp_path / "work tree"
    external_cwd.mkdir()
    block = cdc.ShellBlock(
        path=Path("README.md"),
        line=10,
        language="bash",
        body=(
            "MODE=test /usr/bin/env -i --unset HOME bash -c "
            "'python3 tools/trtmc_validate.py --list'\n"
            "docker exec trtmc-dev env -uHOME sh -c "
            "'python3 tools/trtmc_validate.py --all --dry-run'\n"
        ),
    )

    assert cdc.check_command_block(block, repo_root) == []
    assert len(list(cdc.shell_validation_blocks(block))) == 3
    inline = cdc.extract_inline_commands(
        Path("README.md"),
        (
            "Run `EMPTY= LABEL='two words' env -i env --chdir "
            f"{shlex.quote(str(external_cwd))} "
            "--unset HOME -- bash -c "
            "'python3 tools/trtmc_validate.py --definitely-invalid'`."
        ),
    )
    assert len(inline) == 1
    assert [finding.message for finding in cdc.check_command_block(inline[0], repo_root)] == [
        "command input resolves outside repository and cannot be validated: "
        f"{external_cwd}/tools/trtmc_validate.py"
    ]


def test_command_and_time_wrappers_reuse_nested_cli_contracts() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    bodies = (
        "command python3 tools/trtmc_validate.py --definitely-invalid\n",
        "command -p -- python3 tools/trtmc_validate.py --definitely-invalid\n",
        "time python3 tools/trtmc_validate.py --definitely-invalid\n",
        "time -p command -- python3 tools/trtmc_validate.py --definitely-invalid\n",
        (
            "/usr/bin/time --verbose --output=/tmp/trtmc-time.txt "
            "python3 tools/trtmc_validate.py --definitely-invalid\n"
        ),
        ("{ time -p command python3 tools/trtmc_validate.py --definitely-invalid; }\n"),
    )

    for body in bodies:
        block = cdc.ShellBlock(
            path=Path("README.md"),
            line=10,
            language="bash",
            body=body,
        )
        assert [finding.message for finding in cdc.check_command_block(block, repo_root)] == [
            "unknown option for `tools/trtmc_validate.py`: --definitely-invalid"
        ]


def test_uncertain_command_and_time_wrapper_options_fail_closed() -> None:
    block = cdc.ShellBlock(
        Path("README.md"),
        3,
        "bash",
        "command --implementation-specific python3 tools/trtmc_validate.py\n"
        "time --implementation-specific python3 tools/trtmc_validate.py\n",
    )
    query = cdc.ShellBlock(
        Path("README.md"),
        8,
        "bash",
        "command -v python3\n",
    )

    assert [
        (finding.line, finding.message) for finding in cdc.check_shell_wrapper_contract(block)
    ] == [
        (
            4,
            "cannot statically resolve shell wrapper: unsupported `command` wrapper options",
        ),
        (
            5,
            "cannot statically resolve shell wrapper: unsupported `time` wrapper options",
        ),
    ]
    assert cdc.check_shell_wrapper_contract(query) == []


def test_timeout_wrapper_resolves_common_options_and_command_boundary() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    bodies = (
        "timeout 30s python3 tools/trtmc_validate.py --definitely-invalid\n",
        "timeout 1e3s python3 tools/trtmc_validate.py --definitely-invalid\n",
        "timeout +1s python3 tools/trtmc_validate.py --definitely-invalid\n",
        "timeout inf python3 tools/trtmc_validate.py --definitely-invalid\n",
        (
            "/usr/bin/timeout --foreground --preserve-status "
            "--signal=TERM --kill-after=5s 2m "
            "python3 tools/trtmc_validate.py --definitely-invalid\n"
        ),
        (
            "timeout -v -s TERM -k 1s -- 10 "
            "command python3 tools/trtmc_validate.py --definitely-invalid\n"
        ),
    )

    for body in bodies:
        block = cdc.ShellBlock(Path("README.md"), 1, "bash", body)
        assert [finding.message for finding in cdc.check_command_block(block, repo_root)] == [
            "unknown option for `tools/trtmc_validate.py`: --definitely-invalid"
        ]


def test_unknown_timeout_shapes_fail_closed() -> None:
    block = cdc.ShellBlock(
        Path("README.md"),
        3,
        "bash",
        "timeout --implementation-specific 5 python3 tools/check.py\n"
        "timeout -p 5 python3 tools/check.py\n"
        "timeout soon python3 tools/check.py\n"
        "timeout NaN python3 tools/check.py\n"
        "timeout 5\n",
    )

    assert [
        (finding.line, finding.message) for finding in cdc.check_shell_wrapper_contract(block)
    ] == [
        (
            4,
            "cannot statically resolve shell wrapper: "
            "unsupported `timeout` wrapper options or command boundary",
        ),
        (
            5,
            "cannot statically resolve shell wrapper: "
            "unsupported `timeout` wrapper options or command boundary",
        ),
        (
            6,
            "cannot statically resolve shell wrapper: "
            "unsupported `timeout` wrapper options or command boundary",
        ),
        (
            7,
            "cannot statically resolve shell wrapper: "
            "unsupported `timeout` wrapper options or command boundary",
        ),
        (
            8,
            "cannot statically resolve shell wrapper: "
            "unsupported `timeout` wrapper options or command boundary",
        ),
    ]


def test_timeout_signal_values_are_validated_without_execution() -> None:
    valid = cdc.ShellBlock(
        Path("README.md"),
        3,
        "bash",
        "timeout --signal=TERM 1 true\n"
        "timeout --signal SIGTERM 1 true\n"
        "timeout -s15 1 true\n"
        "timeout -s 0 1 true\n"
        "timeout --signal=RTMIN+1 1 true\n",
    )
    invalid = cdc.ShellBlock(
        Path("README.md"),
        8,
        "bash",
        "timeout --signal=NOT_A_REAL_SIGNAL 1 true\ntimeout -s999999 1 true\n",
    )

    assert cdc.check_shell_wrapper_contract(valid) == []
    assert [
        (finding.line, finding.message) for finding in cdc.check_shell_wrapper_contract(invalid)
    ] == [
        (
            9,
            "cannot statically resolve shell wrapper: "
            "unsupported `timeout` wrapper options or command boundary",
        ),
        (
            10,
            "cannot statically resolve shell wrapper: "
            "unsupported `timeout` wrapper options or command boundary",
        ),
    ]


@pytest.mark.parametrize(
    "value",
    [
        "0",
        "32",
        "33",
        "64",
        "TERM",
        "SIGTERM",
        "RTMIN+1",
        "NOT_A_REAL_SIGNAL",
        "١٥",
    ],
)
def test_timeout_signal_acceptance_matches_gnu_timeout(
    value: str,
) -> None:
    completed = subprocess.run(
        ["timeout", f"--signal={value}", "1", "true"],
        capture_output=True,
        check=False,
        text=True,
    )
    block = cdc.ShellBlock(
        Path("README.md"),
        3,
        "bash",
        f"timeout --signal={value} 1 true\n",
    )
    checker_accepts = cdc.check_shell_wrapper_contract(block) == []

    assert checker_accepts == (completed.returncode != 125), completed.stderr
    if value.isascii() and value.isdecimal():
        number = int(value)
        assert checker_accepts == (
            number == 0 or number in {int(candidate) for candidate in signal.valid_signals()}
        )


@pytest.mark.parametrize(
    "options",
    [
        ["-s=TERM"],
        ["-vsTERM"],
        ["-vk.01s"],
    ],
)
def test_timeout_short_option_clusters_match_gnu_timeout(
    options: list[str],
) -> None:
    tokens = ["timeout", *options, "1", "true"]
    completed = subprocess.run(
        tokens,
        capture_output=True,
        check=False,
        text=True,
    )
    block = cdc.ShellBlock(
        Path("README.md"),
        3,
        "bash",
        f"{shlex.join(tokens)}\n",
    )

    assert (cdc.check_shell_wrapper_contract(block) == []) == (completed.returncode != 125), (
        completed.stderr
    )


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
        lambda _tokens, *, cwd=Path("."): cdc._NestedShellPayload(
            block.body,
            cwd,
        ),
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


def test_python_contract_inside_bash_brace_group_is_checked(tmp_path: Path) -> None:
    script = tmp_path / "tools" / "cli.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        'parser.add_argument("--good", required=True)\n'
        "parser.parse_args()\n",
        encoding="utf-8",
    )
    block = cdc.ShellBlock(
        path=Path("README.md"),
        line=7,
        language="bash",
        body="{ python3 tools/cli.py --bad value; }\n",
    )

    findings = cdc.check_python_script_contract(block, tmp_path)

    assert [finding.message for finding in findings] == [
        "unknown option for `tools/cli.py`: --bad",
        "missing required option for `tools/cli.py`: --good",
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


def test_argparse_ignores_uncalled_class_main_and_follows_module_main_call_graph(
    tmp_path: Path,
) -> None:
    script = tmp_path / "tools" / "scoped_class.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import argparse\n"
        "class Unused:\n"
        "    def main(self):\n"
        "        parser = argparse.ArgumentParser()\n"
        '        parser.add_argument("--dead", required=True)\n'
        "        return parser.parse_args()\n"
        "def build_parser():\n"
        "    parser = argparse.ArgumentParser()\n"
        '    parser.add_argument("--live", required=True)\n'
        "    return parser\n"
        "def parse_cli():\n"
        "    return build_parser().parse_args()\n"
        "def main():\n"
        "    return parse_cli()\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )
    live = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/scoped_class.py --live yes\n",
    )
    dead = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/scoped_class.py --dead yes\n",
    )

    assert cdc.check_python_script_contract(live, tmp_path) == []
    assert [finding.message for finding in cdc.check_python_script_contract(dead, tmp_path)] == [
        "unknown option for `tools/scoped_class.py`: --dead",
        "missing required option for `tools/scoped_class.py`: --live",
    ]


@pytest.mark.parametrize(
    "entrypoint",
    [
        "if __name__ == '__main__':\n    App().main()\n",
        "def run():\n"
        "    app = App()\n"
        "    return app.main()\n"
        "if __name__ == '__main__':\n"
        "    run()\n",
    ],
)
def test_argparse_follows_statically_bound_class_method_entrypoints(
    tmp_path: Path,
    entrypoint: str,
) -> None:
    script = tmp_path / "tools" / "class_entrypoint.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import argparse\n"
        "class Unused:\n"
        "    def main(self):\n"
        "        parser = argparse.ArgumentParser()\n"
        '        parser.add_argument("--dead", required=True)\n'
        "        return parser.parse_args()\n"
        "class App:\n"
        "    def main(self):\n"
        "        parser = argparse.ArgumentParser()\n"
        '        parser.add_argument("--live", required=True)\n'
        "        return parser.parse_args()\n"
        f"{entrypoint}",
        encoding="utf-8",
    )
    valid = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/class_entrypoint.py --live yes\n",
    )
    invalid = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "{ time -p command python3 tools/class_entrypoint.py --dead yes; }\n",
    )

    assert cdc.check_command_block(valid, tmp_path) == []
    assert [finding.message for finding in cdc.check_command_block(invalid, tmp_path)] == [
        "unknown option for `tools/class_entrypoint.py`: --dead",
        "missing required option for `tools/class_entrypoint.py`: --live",
    ]


def test_argparse_import_alias_and_named_expression_select_root(
    tmp_path: Path,
) -> None:
    script = tmp_path / "tools" / "named_root.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "from argparse import ArgumentParser as AP\n"
        "parser = (selected := AP())\n"
        'selected.add_argument("--live", required=True)\n'
        "parser.parse_args()\n",
        encoding="utf-8",
    )
    valid = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/named_root.py --live yes\n",
    )
    invalid = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/named_root.py --dead yes\n",
    )

    assert cdc.check_python_script_contract(valid, tmp_path) == []
    assert [finding.message for finding in cdc.check_python_script_contract(invalid, tmp_path)] == [
        "unknown option for `tools/named_root.py`: --dead",
        "missing required option for `tools/named_root.py`: --live",
    ]


def test_argparse_constructor_alias_chain_selects_root(
    tmp_path: Path,
) -> None:
    script = tmp_path / "tools" / "alias_chain.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "from argparse import ArgumentParser as AP\n"
        "Parser = AP\n"
        "parser = Parser()\n"
        'parser.add_argument("--live", required=True)\n'
        "parser.parse_args()\n",
        encoding="utf-8",
    )
    valid = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/alias_chain.py --live yes\n",
    )
    invalid = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/alias_chain.py --dead yes\n",
    )

    assert cdc.check_python_script_contract(valid, tmp_path) == []
    assert [finding.message for finding in cdc.check_python_script_contract(invalid, tmp_path)] == [
        "unknown option for `tools/alias_chain.py`: --dead",
        "missing required option for `tools/alias_chain.py`: --live",
    ]


def test_argparse_module_attribute_constructor_alias_selects_root(
    tmp_path: Path,
) -> None:
    script = tmp_path / "tools" / "module_alias.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import argparse as ap\n"
        "module = ap\n"
        "Parser = module.ArgumentParser\n"
        "parser = Parser()\n"
        'parser.add_argument("--live", required=True)\n'
        "parser.parse_args()\n",
        encoding="utf-8",
    )
    invalid = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/module_alias.py --dead yes\n",
    )

    assert [finding.message for finding in cdc.check_python_script_contract(invalid, tmp_path)] == [
        "unknown option for `tools/module_alias.py`: --dead",
        "missing required option for `tools/module_alias.py`: --live",
    ]


def test_argparse_constructor_alias_rebinding_is_ordered(
    tmp_path: Path,
) -> None:
    tools = tmp_path / "tools"
    tools.mkdir(parents=True)
    custom_parser = (
        "class CustomParser:\n"
        "    def add_argument(self, *_args, **_kwargs):\n"
        "        return None\n"
        "    def parse_args(self):\n"
        "        return None\n"
    )
    before_rebind = tools / "before_rebind.py"
    before_rebind.write_text(
        "from argparse import ArgumentParser as AP\n"
        f"{custom_parser}"
        "Parser = AP\n"
        "parser = Parser()\n"
        'parser.add_argument("--live", required=True)\n'
        "Parser = CustomParser\n"
        "parser.parse_args()\n",
        encoding="utf-8",
    )
    after_rebind = tools / "after_rebind.py"
    after_rebind.write_text(
        "from argparse import ArgumentParser as AP\n"
        f"{custom_parser}"
        "Parser = AP\n"
        "Parser = CustomParser\n"
        "parser = Parser()\n"
        'parser.add_argument("--custom")\n'
        "parser.parse_args()\n",
        encoding="utf-8",
    )
    before_block = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/before_rebind.py --dead yes\n",
    )
    after_block = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/after_rebind.py --anything\n",
    )

    assert [
        finding.message for finding in cdc.check_python_script_contract(before_block, tmp_path)
    ] == [
        "unknown option for `tools/before_rebind.py`: --dead",
        "missing required option for `tools/before_rebind.py`: --live",
    ]
    assert cdc.check_python_script_contract(after_block, tmp_path) == []


def test_argparse_constructor_parameter_shadows_outer_alias(
    tmp_path: Path,
) -> None:
    script = tmp_path / "tools" / "scoped_alias.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "from argparse import ArgumentParser as Parser\n"
        "class CustomParser:\n"
        "    def add_argument(self, *_args, **_kwargs):\n"
        "        return None\n"
        "    def parse_args(self):\n"
        "        return None\n"
        "def run(Parser):\n"
        "    parser = Parser()\n"
        '    parser.add_argument("--custom")\n'
        "    parser.parse_args()\n"
        "run(CustomParser)\n",
        encoding="utf-8",
    )
    block = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/scoped_alias.py --anything\n",
    )

    assert cdc.check_python_script_contract(block, tmp_path) == []


@pytest.mark.parametrize(
    ("expression", "expects_argparse"),
    [
        ("AP if True else CustomParser", True),
        ("CustomParser if False else AP", True),
        ("AP if False else CustomParser", False),
        ("AP if condition else AP", True),
        ("AP if condition else CustomParser", False),
    ],
)
def test_argparse_constructor_alias_static_conditional_expressions(
    tmp_path: Path,
    expression: str,
    expects_argparse: bool,
) -> None:
    script = tmp_path / "tools" / "conditional_alias.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "from argparse import ArgumentParser as AP\n"
        "class CustomParser:\n"
        "    def add_argument(self, *_args, **_kwargs):\n"
        "        return None\n"
        "    def parse_args(self):\n"
        "        return None\n"
        "condition = bool()\n"
        f"Parser = {expression}\n"
        "parser = Parser()\n"
        'parser.add_argument("--live", required=True)\n'
        "parser.parse_args()\n",
        encoding="utf-8",
    )
    block = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/conditional_alias.py --dead yes\n",
    )

    messages = [finding.message for finding in cdc.check_python_script_contract(block, tmp_path)]
    if expects_argparse:
        assert messages == [
            "unknown option for `tools/conditional_alias.py`: --dead",
            "missing required option for `tools/conditional_alias.py`: --live",
        ]
    else:
        assert messages == []


def test_argparse_constructor_named_expression_alias_side_effect(
    tmp_path: Path,
) -> None:
    script = tmp_path / "tools" / "named_constructor_alias.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "from argparse import ArgumentParser as AP\n"
        "Parser = (Alias := AP)\n"
        "parser = Alias()\n"
        'parser.add_argument("--live", required=True)\n'
        "parser.parse_args()\n",
        encoding="utf-8",
    )
    block = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/named_constructor_alias.py --dead yes\n",
    )

    assert [finding.message for finding in cdc.check_python_script_contract(block, tmp_path)] == [
        "unknown option for `tools/named_constructor_alias.py`: --dead",
        "missing required option for `tools/named_constructor_alias.py`: --live",
    ]


def test_argparse_parser_instance_rebinding_and_parameter_shadowing(
    tmp_path: Path,
) -> None:
    tools = tmp_path / "tools"
    tools.mkdir(parents=True)
    custom_parser = (
        "class CustomParser:\n"
        "    def add_argument(self, *_args, **_kwargs):\n"
        "        return None\n"
        "    def parse_args(self):\n"
        "        return None\n"
    )
    rebound = tools / "rebound_instance.py"
    rebound.write_text(
        "from argparse import ArgumentParser as AP\n"
        f"{custom_parser}"
        "parser = AP()\n"
        'parser.add_argument("--dead", required=True)\n'
        "parser = CustomParser()\n"
        "parser.parse_args()\n",
        encoding="utf-8",
    )
    shadowed = tools / "shadowed_instance.py"
    shadowed.write_text(
        "from argparse import ArgumentParser as AP\n"
        f"{custom_parser}"
        "parser = AP()\n"
        'parser.add_argument("--dead", required=True)\n'
        "def run(parser):\n"
        "    parser.parse_args()\n"
        "run(CustomParser())\n",
        encoding="utf-8",
    )

    rebound_block = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/rebound_instance.py --anything\n",
    )
    shadowed_block = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/shadowed_instance.py --anything\n",
    )

    assert cdc.check_python_script_contract(rebound_block, tmp_path) == []
    assert [
        finding.message
        for finding in cdc.check_python_script_contract(
            shadowed_block,
            tmp_path,
        )
    ] == [
        "cannot statically validate argparse contract for "
        "`tools/shadowed_instance.py`: "
        "a reachable parse call has an unresolved parser identity"
    ]


def test_argparse_named_expression_receiver_attaches_arguments(
    tmp_path: Path,
) -> None:
    script = tmp_path / "tools" / "named_receiver.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "from argparse import ArgumentParser as AP\n"
        "(parser := AP()).add_argument('--live', required=True)\n"
        "parser.parse_args()\n",
        encoding="utf-8",
    )
    valid = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/named_receiver.py --live yes\n",
    )
    invalid = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/named_receiver.py --dead yes\n",
    )

    assert cdc.check_python_script_contract(valid, tmp_path) == []
    assert [finding.message for finding in cdc.check_python_script_contract(invalid, tmp_path)] == [
        "unknown option for `tools/named_receiver.py`: --dead",
        "missing required option for `tools/named_receiver.py`: --live",
    ]


def test_argparse_if_expression_preserves_or_rejects_root_identity(
    tmp_path: Path,
) -> None:
    same = tmp_path / "tools" / "same_root.py"
    same.parent.mkdir(parents=True)
    same.write_text(
        "from argparse import ArgumentParser as AP\n"
        "parser = AP()\n"
        'parser.add_argument("--live", required=True)\n'
        "selected = parser if object() else parser\n"
        "selected.parse_args()\n",
        encoding="utf-8",
    )
    ambiguous = tmp_path / "tools" / "ambiguous_root.py"
    ambiguous.write_text(
        "from argparse import ArgumentParser as AP\n"
        "first = AP()\n"
        'first.add_argument("--first")\n'
        "second = AP()\n"
        'second.add_argument("--second")\n'
        "selected = first if object() else second\n"
        "selected.parse_args()\n",
        encoding="utf-8",
    )
    valid = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/same_root.py --live yes\n",
    )
    uncertain = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/ambiguous_root.py --first yes\n",
    )

    assert cdc.check_python_script_contract(valid, tmp_path) == []
    assert [
        finding.message for finding in cdc.check_python_script_contract(uncertain, tmp_path)
    ] == [
        "cannot statically validate argparse contract for "
        "`tools/ambiguous_root.py`: "
        "a reachable parse call has an unresolved parser identity"
    ]


@pytest.mark.parametrize("method", ["__new__", "__init__"])
def test_argparse_reaches_parser_in_called_class_constructor(
    tmp_path: Path,
    method: str,
) -> None:
    script = tmp_path / "tools" / f"constructor_{method.strip('_')}.py"
    script.parent.mkdir(parents=True)
    receiver = "cls" if method == "__new__" else "self"
    tail = "        return object.__new__(cls)\n" if method == "__new__" else ""
    script.write_text(
        "from argparse import ArgumentParser as AP\n"
        "class App:\n"
        f"    def {method}({receiver}):\n"
        "        parser = AP()\n"
        '        parser.add_argument("--live", required=True)\n'
        "        parser.parse_args()\n"
        f"{tail}"
        "def main():\n"
        "    return App()\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )
    invalid = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        f"python3 tools/{script.name} --dead yes\n",
    )

    assert [finding.message for finding in cdc.check_python_script_contract(invalid, tmp_path)] == [
        f"unknown option for `tools/{script.name}`: --dead",
        f"missing required option for `tools/{script.name}`: --live",
    ]


def test_argparse_reaches_inherited_class_constructor(
    tmp_path: Path,
) -> None:
    script = tmp_path / "tools" / "inherited_constructor.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "from argparse import ArgumentParser as AP\n"
        "class Base:\n"
        "    def __init__(self):\n"
        "        parser = AP()\n"
        '        parser.add_argument("--live", required=True)\n'
        "        parser.parse_args()\n"
        "class App(Base):\n"
        "    pass\n"
        "def main():\n"
        "    return App()\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )
    invalid = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/inherited_constructor.py --dead yes\n",
    )

    assert [finding.message for finding in cdc.check_python_script_contract(invalid, tmp_path)] == [
        "unknown option for `tools/inherited_constructor.py`: --dead",
        "missing required option for `tools/inherited_constructor.py`: --live",
    ]


def test_argparse_local_constructor_reaches_base_only_with_explicit_super(
    tmp_path: Path,
) -> None:
    tools = tmp_path / "tools"
    tools.mkdir(parents=True)
    base = (
        "from argparse import ArgumentParser\n"
        "class Base:\n"
        "    def __init__(self):\n"
        "        parser = ArgumentParser()\n"
        '        parser.add_argument("--live", required=True)\n'
        "        parser.parse_args()\n"
    )
    with_super = tools / "with_super.py"
    with_super.write_text(
        f"{base}"
        "class App(Base):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "def main():\n"
        "    return App()\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )
    without_super = tools / "without_super.py"
    without_super.write_text(
        f"{base}"
        "class App(Base):\n"
        "    def __init__(self):\n"
        "        self.ready = True\n"
        "def main():\n"
        "    return App()\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )
    with_super_block = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/with_super.py --dead yes\n",
    )
    without_super_block = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/without_super.py --dead yes\n",
    )

    assert [
        finding.message
        for finding in cdc.check_python_script_contract(
            with_super_block,
            tmp_path,
        )
    ] == [
        "unknown option for `tools/with_super.py`: --dead",
        "missing required option for `tools/with_super.py`: --live",
    ]
    assert (
        cdc.check_python_script_contract(
            without_super_block,
            tmp_path,
        )
        == []
    )


def test_argparse_constructor_lookup_follows_python_mro(
    tmp_path: Path,
) -> None:
    script = tmp_path / "tools" / "mro_constructor.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "from argparse import ArgumentParser\n"
        "class First:\n"
        "    def __init__(self):\n"
        "        parser = ArgumentParser()\n"
        '        parser.add_argument("--live", required=True)\n'
        "        parser.parse_args()\n"
        "class DeadSecond:\n"
        "    def __init__(self):\n"
        "        parser = ArgumentParser()\n"
        '        parser.add_argument("--dead", required=True)\n'
        "        parser.parse_args()\n"
        "class App(First, DeadSecond):\n"
        "    pass\n"
        "App()\n",
        encoding="utf-8",
    )
    valid = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/mro_constructor.py --live yes\n",
    )

    assert cdc.check_python_script_contract(valid, tmp_path) == []


@pytest.mark.parametrize("cooperative", [False, True])
def test_argparse_constructor_super_uses_runtime_mro_context(
    tmp_path: Path,
    cooperative: bool,
) -> None:
    script = tmp_path / "tools" / f"cooperative_{cooperative}.py"
    script.parent.mkdir(parents=True)
    super_call = "        super().__init__()\n" if cooperative else ""
    script.write_text(
        "from argparse import ArgumentParser\n"
        "class First:\n"
        "    def __init__(self):\n"
        "        self.ready = True\n"
        f"{super_call}"
        "class Second:\n"
        "    def __init__(self):\n"
        "        parser = ArgumentParser()\n"
        '        parser.add_argument("--live", required=True)\n'
        "        parser.parse_args()\n"
        "class App(First, Second):\n"
        "    pass\n"
        "App()\n",
        encoding="utf-8",
    )
    block = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        f"python3 tools/{script.name} --dead yes\n",
    )

    messages = [finding.message for finding in cdc.check_python_script_contract(block, tmp_path)]
    if cooperative:
        assert messages == [
            f"unknown option for `tools/{script.name}`: --dead",
            f"missing required option for `tools/{script.name}`: --live",
        ]
    else:
        assert messages == []


@pytest.mark.parametrize(
    ("super_call", "reaches_base"),
    [
        ("super(App, self).__init__()", True),
        ("super(Base, self).__init__()", False),
        ("super(App, object()).__init__()", False),
    ],
)
def test_argparse_two_argument_super_binding(
    tmp_path: Path,
    super_call: str,
    reaches_base: bool,
) -> None:
    script = tmp_path / "tools" / "two_arg_super.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "from argparse import ArgumentParser\n"
        "class Base:\n"
        "    def __init__(self):\n"
        "        parser = ArgumentParser()\n"
        '        parser.add_argument("--live", required=True)\n'
        "        parser.parse_args()\n"
        "class App(Base):\n"
        "    def __init__(self):\n"
        f"        {super_call}\n"
        "App()\n",
        encoding="utf-8",
    )
    block = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/two_arg_super.py --dead yes\n",
    )

    messages = [finding.message for finding in cdc.check_python_script_contract(block, tmp_path)]
    if reaches_base:
        assert messages == [
            "unknown option for `tools/two_arg_super.py`: --dead",
            "missing required option for `tools/two_arg_super.py`: --live",
        ]
    else:
        assert messages == []


@pytest.mark.parametrize(
    ("case_id", "source"),
    [
        (
            "A04",
            "class Base:\n"
            "    def __init__(self):\n"
            "        parser = AP()\n"
            "        parser.add_argument('--live', required=True)\n"
            "        parser.parse_args()\n"
            "FrozenBase = Base\n"
            "class App(FrozenBase):\n"
            "    pass\n"
            "class Base:\n"
            "    def __init__(self):\n"
            "        parser = AP()\n"
            "        parser.add_argument('--dead', required=True)\n"
            "        parser.parse_args()\n"
            "App()\n",
        ),
        (
            "A05",
            "class App:\n"
            "    def __init__(self):\n"
            "        parser = AP()\n"
            "        parser.add_argument('--live', required=True)\n"
            "        parser.parse_args()\n"
            "App()\n"
            "class App:\n"
            "    def __init__(self):\n"
            "        parser = AP()\n"
            "        parser.add_argument('--dead', required=True)\n"
            "        parser.parse_args()\n",
        ),
        (
            "A06",
            "class App:\n"
            "    def __init__(self):\n"
            "        parser = AP()\n"
            "        parser.add_argument('--live', required=True)\n"
            "        parser.parse_args()\n"
            "Alias = App\n"
            "class App:\n"
            "    def __init__(self):\n"
            "        parser = AP()\n"
            "        parser.add_argument('--dead', required=True)\n"
            "        parser.parse_args()\n"
            "Alias()\n",
        ),
        (
            "A07",
            "def run():\n"
            "    parser = AP()\n"
            "    parser.add_argument('--live', required=True)\n"
            "    parser.parse_args()\n"
            "run()\n"
            "def run():\n"
            "    parser = AP()\n"
            "    parser.add_argument('--dead', required=True)\n"
            "    parser.parse_args()\n",
        ),
        (
            "A08",
            "def run():\n"
            "    parser = AP()\n"
            "    parser.add_argument('--live', required=True)\n"
            "    parser.parse_args()\n"
            "Alias = run\n"
            "def run():\n"
            "    parser = AP()\n"
            "    parser.add_argument('--dead', required=True)\n"
            "    parser.parse_args()\n"
            "Alias()\n",
        ),
        (
            "A09",
            "class First:\n"
            "    def run(self):\n"
            "        parser = AP()\n"
            "        parser.add_argument('--dead', required=True)\n"
            "        parser.parse_args()\n"
            "class Second:\n"
            "    def run(self):\n"
            "        parser = AP()\n"
            "        parser.add_argument('--live', required=True)\n"
            "        parser.parse_args()\n"
            "app = First()\n"
            "app = Second()\n"
            "app.run()\n",
        ),
        (
            "A10",
            "class First:\n"
            "    def run(self):\n"
            "        parser = AP()\n"
            "        parser.add_argument('--live', required=True)\n"
            "        parser.parse_args()\n"
            "class Second:\n"
            "    def run(self):\n"
            "        parser = AP()\n"
            "        parser.add_argument('--dead', required=True)\n"
            "        parser.parse_args()\n"
            "app = First()\n"
            "alias = app\n"
            "app = Second()\n"
            "alias.run()\n",
        ),
        (
            "A11",
            "class App:\n"
            "    def run(self):\n"
            "        parser = AP()\n"
            "        parser.add_argument('--dead', required=True)\n"
            "        parser.parse_args()\n"
            "def main():\n"
            "    class App:\n"
            "        def run(self):\n"
            "            parser = AP()\n"
            "            parser.add_argument('--live', required=True)\n"
            "            parser.parse_args()\n"
            "    App().run()\n"
            "main()\n",
        ),
        (
            "A12",
            "class Base:\n"
            "    def __init__(self):\n"
            "        parser = AP()\n"
            "        parser.add_argument('--live', required=True)\n"
            "        parser.parse_args()\n"
            "class App(Base):\n"
            "    def __init__(self):\n"
            "        ThisClass = App\n"
            "        super(ThisClass, self).__init__()\n"
            "App()\n",
        ),
        (
            "A13",
            "class Base:\n"
            "    def __init__(self):\n"
            "        parser = AP()\n"
            "        parser.add_argument('--live', required=True)\n"
            "        parser.parse_args()\n"
            "class App(Base):\n"
            "    def __init__(self):\n"
            "        receiver = self\n"
            "        super(App, receiver).__init__()\n"
            "App()\n",
        ),
    ],
    ids=lambda value: value if isinstance(value, str) and value.startswith("A") else None,
)
def test_argparse_reachability_identity_matches_python_execution(
    tmp_path: Path,
    case_id: str,
    source: str,
) -> None:
    script = tmp_path / "tools" / f"identity_{case_id}.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        f"from argparse import ArgumentParser as AP\n{source}",
        encoding="utf-8",
    )
    valid_run = subprocess.run(
        [sys.executable, str(script), "--live", "yes"],
        capture_output=True,
        check=False,
        text=True,
    )
    invalid_run = subprocess.run(
        [sys.executable, str(script), "--dead", "yes"],
        capture_output=True,
        check=False,
        text=True,
    )
    valid = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        f"python3 tools/{script.name} --live yes\n",
    )
    invalid = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        f"python3 tools/{script.name} --dead yes\n",
    )

    assert valid_run.returncode == 0, valid_run.stderr
    assert invalid_run.returncode != 0
    assert cdc.check_python_script_contract(valid, tmp_path) == []
    assert [finding.message for finding in cdc.check_python_script_contract(invalid, tmp_path)] == [
        f"unknown option for `tools/{script.name}`: --dead",
        f"missing required option for `tools/{script.name}`: --live",
    ]


def test_argparse_dynamic_reachability_identity_fails_closed(
    tmp_path: Path,
) -> None:
    script = tmp_path / "tools" / "dynamic_identity.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "from argparse import ArgumentParser as AP\n"
        "class First:\n"
        "    def run(self):\n"
        "        parser = AP()\n"
        "        parser.add_argument('--first')\n"
        "        parser.parse_args()\n"
        "class Second:\n"
        "    def run(self):\n"
        "        parser = AP()\n"
        "        parser.add_argument('--second')\n"
        "        parser.parse_args()\n"
        "Target = First if object() else Second\n"
        "Target().run()\n",
        encoding="utf-8",
    )
    block = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/dynamic_identity.py --first yes\n",
    )

    assert [finding.message for finding in cdc.check_python_script_contract(block, tmp_path)] == [
        "cannot statically validate argparse contract for "
        "`tools/dynamic_identity.py`: "
        "a reachable call has an unresolved class, function, or instance identity"
    ]


@pytest.mark.parametrize(
    ("case_id", "source", "runtime_flag", "statically_exact"),
    [
        (
            "V01_class_alias_chain",
            """
            class Live:
                def run(self):
                    p = AP(); p.add_argument("--live", required=True); p.parse_args()
            class Dead:
                def run(self):
                    p = AP(); p.add_argument("--dead", required=True); p.parse_args()
            A = Live
            B = A
            A = Dead
            B().run()
            """,
            "live",
            True,
        ),
        (
            "V02_function_alias_chain",
            """
            def target():
                p = AP(); p.add_argument("--live", required=True); p.parse_args()
            A = target
            B = A
            def target():
                p = AP(); p.add_argument("--dead", required=True); p.parse_args()
            B()
            """,
            "live",
            True,
        ),
        (
            "V03_instance_alias_chain",
            """
            class Live:
                def run(self):
                    p = AP(); p.add_argument("--live", required=True); p.parse_args()
            class Dead:
                def run(self):
                    p = AP(); p.add_argument("--dead", required=True); p.parse_args()
            app = Live()
            A = app
            B = A
            app = Dead()
            B.run()
            """,
            "live",
            True,
        ),
        (
            "V04_base_alias_chain_freeze",
            """
            class Base:
                def __init__(self):
                    p = AP(); p.add_argument("--live", required=True); p.parse_args()
            B1 = Base
            B2 = B1
            class App(B2):
                pass
            class Base:
                def __init__(self):
                    p = AP(); p.add_argument("--dead", required=True); p.parse_args()
            App()
            """,
            "live",
            True,
        ),
        (
            "V05_local_shadow_alias_chain",
            """
            class App:
                def run(self):
                    p = AP(); p.add_argument("--dead", required=True); p.parse_args()
            def main():
                class App:
                    def run(self):
                        p = AP(); p.add_argument("--live", required=True); p.parse_args()
                A = App
                B = A
                B().run()
            main()
            """,
            "live",
            True,
        ),
        (
            "V06_super_alias_chains",
            """
            class Base:
                def __init__(self):
                    p = AP(); p.add_argument("--live", required=True); p.parse_args()
            class App(Base):
                def __init__(self):
                    C1 = App
                    C2 = C1
                    r1 = self
                    r2 = r1
                    super(C2, r2).__init__()
            App()
            """,
            "live",
            True,
        ),
        (
            "V07_deleted_class_alias",
            """
            class Live:
                def __init__(self):
                    p = AP(); p.add_argument("--live", required=True); p.parse_args()
            Alias = Live
            del Alias
            Alias()
            """,
            "none",
            False,
        ),
        (
            "V08_deleted_function_alias",
            """
            def live():
                p = AP(); p.add_argument("--live", required=True); p.parse_args()
            Alias = live
            del Alias
            Alias()
            """,
            "none",
            False,
        ),
        (
            "V09_deleted_instance_alias",
            """
            class Live:
                def run(self):
                    p = AP(); p.add_argument("--live", required=True); p.parse_args()
            alias = Live()
            del alias
            alias.run()
            """,
            "none",
            False,
        ),
        (
            "V10_global_instance_rebind",
            """
            class Live:
                def run(self):
                    p = AP(); p.add_argument("--live", required=True); p.parse_args()
            class Dead:
                def run(self):
                    p = AP(); p.add_argument("--dead", required=True); p.parse_args()
            target = Live()
            def switch():
                global target
                target = Dead()
            switch()
            target.run()
            """,
            "dead",
            False,
        ),
        (
            "V11_global_function_rebind",
            """
            def live():
                p = AP(); p.add_argument("--live", required=True); p.parse_args()
            def dead():
                p = AP(); p.add_argument("--dead", required=True); p.parse_args()
            target = live
            def switch():
                global target
                target = dead
            switch()
            target()
            """,
            "dead",
            False,
        ),
        (
            "V12_global_class_rebind",
            """
            class Live:
                def run(self):
                    p = AP(); p.add_argument("--live", required=True); p.parse_args()
            class Dead:
                def run(self):
                    p = AP(); p.add_argument("--dead", required=True); p.parse_args()
            Target = Live
            def switch():
                global Target
                Target = Dead
            switch()
            Target().run()
            """,
            "dead",
            False,
        ),
        (
            "V13_nonlocal_instance_rebind",
            """
            def outer():
                class Live:
                    def run(self):
                        p = AP(); p.add_argument("--live", required=True); p.parse_args()
                class Dead:
                    def run(self):
                        p = AP(); p.add_argument("--dead", required=True); p.parse_args()
                target = Live()
                def switch():
                    nonlocal target
                    target = Dead()
                switch()
                target.run()
            outer()
            """,
            "dead",
            False,
        ),
        (
            "V14_nonlocal_function_rebind",
            """
            def outer():
                def live():
                    p = AP(); p.add_argument("--live", required=True); p.parse_args()
                def dead():
                    p = AP(); p.add_argument("--dead", required=True); p.parse_args()
                target = live
                def switch():
                    nonlocal target
                    target = dead
                switch()
                target()
            outer()
            """,
            "dead",
            False,
        ),
        (
            "V15_class_decorator_replacement",
            """
            class Dead:
                def run(self):
                    p = AP(); p.add_argument("--dead", required=True); p.parse_args()
            def replace(_):
                return Dead
            @replace
            class Target:
                def run(self):
                    p = AP(); p.add_argument("--live", required=True); p.parse_args()
            Target().run()
            """,
            "dead",
            False,
        ),
        (
            "V16_function_decorator_replacement",
            """
            def dead():
                p = AP(); p.add_argument("--dead", required=True); p.parse_args()
            def replace(_):
                return dead
            @replace
            def target():
                p = AP(); p.add_argument("--live", required=True); p.parse_args()
            target()
            """,
            "dead",
            False,
        ),
        (
            "V17_decorator_global_side_effect",
            """
            class Live:
                def run(self):
                    p = AP(); p.add_argument("--live", required=True); p.parse_args()
            class Dead:
                def run(self):
                    p = AP(); p.add_argument("--dead", required=True); p.parse_args()
            target = Live()
            def mutate(value):
                global target
                target = Dead()
                return value
            @mutate
            def marker():
                pass
            target.run()
            """,
            "dead",
            False,
        ),
        (
            "V18_default_global_side_effect",
            """
            class Live:
                def run(self):
                    p = AP(); p.add_argument("--live", required=True); p.parse_args()
            class Dead:
                def run(self):
                    p = AP(); p.add_argument("--dead", required=True); p.parse_args()
            target = Live()
            def switch():
                global target
                target = Dead()
            def marker(value=switch()):
                pass
            target.run()
            """,
            "dead",
            False,
        ),
        (
            "V19_default_function_freeze",
            """
            def live():
                p = AP(); p.add_argument("--live", required=True); p.parse_args()
            def invoke(fn=live):
                fn()
            def live():
                p = AP(); p.add_argument("--dead", required=True); p.parse_args()
            invoke()
            """,
            "live",
            False,
        ),
        (
            "V20_default_class_freeze",
            """
            class Live:
                def run(self):
                    p = AP(); p.add_argument("--live", required=True); p.parse_args()
            def invoke(cls=Live):
                cls().run()
            class Live:
                def run(self):
                    p = AP(); p.add_argument("--dead", required=True); p.parse_args()
            invoke()
            """,
            "live",
            False,
        ),
        (
            "V21_chained_decorators_replace",
            """
            def dead():
                p = AP(); p.add_argument("--dead", required=True); p.parse_args()
            def identity(value):
                return value
            def replace(value):
                return dead
            @identity
            @replace
            def target():
                p = AP(); p.add_argument("--live", required=True); p.parse_args()
            target()
            """,
            "dead",
            False,
        ),
        (
            "V22_delete_then_except_rebind",
            """
            class Live:
                def run(self):
                    p = AP(); p.add_argument("--live", required=True); p.parse_args()
            class Dead:
                def run(self):
                    p = AP(); p.add_argument("--dead", required=True); p.parse_args()
            Target = Live
            del Target
            try:
                Target
            except NameError:
                Target = Dead
            Target().run()
            """,
            "dead",
            False,
        ),
        (
            "V23_identity_class_decorator",
            """
            def identity(value):
                return value
            @identity
            class Target:
                def run(self):
                    p = AP(); p.add_argument("--live", required=True); p.parse_args()
            Target().run()
            """,
            "live",
            False,
        ),
        (
            "V24_default_direct_parser_call",
            """
            def live():
                p = AP(); p.add_argument("--live", required=True); p.parse_args()
            def marker(value=live()):
                pass
            """,
            "live",
            False,
        ),
    ],
    ids=lambda value: value if isinstance(value, str) and value.startswith("V") else None,
)
def test_argparse_adjacent_identity_graph_matches_python(
    tmp_path: Path,
    case_id: str,
    source: str,
    runtime_flag: str,
    statically_exact: bool,
) -> None:
    script = tmp_path / "tools" / f"{case_id}.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "from argparse import ArgumentParser as AP\n" + textwrap.dedent(source),
        encoding="utf-8",
    )
    outcomes: dict[str, tuple[int, list[str]]] = {}
    for flag_name in ("live", "dead"):
        flag = f"--{flag_name}"
        completed = subprocess.run(
            [sys.executable, str(script), flag, "yes"],
            capture_output=True,
            check=False,
            text=True,
        )
        command = cdc.ShellBlock(
            Path("README.md"),
            1,
            "bash",
            f"python3 tools/{script.name} {flag} yes\n",
        )
        findings = [
            finding.message
            for finding in cdc.check_python_script_contract(
                command,
                tmp_path,
            )
        ]
        outcomes[flag_name] = completed.returncode, findings
        if completed.returncode != 0:
            assert findings, f"{case_id} false pass for {flag}: {completed.stderr}"

    if runtime_flag == "none":
        assert outcomes["live"][0] != 0
        assert outcomes["dead"][0] != 0
    else:
        rejected = "dead" if runtime_flag == "live" else "live"
        assert outcomes[runtime_flag][0] == 0
        assert outcomes[rejected][0] != 0
        if statically_exact:
            assert outcomes[runtime_flag][1] == []
            assert outcomes[rejected][1]


def test_argparse_opaque_effect_is_recovered_by_static_rebind(
    tmp_path: Path,
) -> None:
    script = tmp_path / "tools" / "recovered_identity.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "from argparse import ArgumentParser as AP\n"
        "class Live:\n"
        "    def run(self):\n"
        "        p = AP(); p.add_argument('--live', required=True); p.parse_args()\n"
        "class Dead:\n"
        "    def run(self):\n"
        "        p = AP(); p.add_argument('--dead', required=True); p.parse_args()\n"
        "target = Live()\n"
        "def switch():\n"
        "    global target\n"
        "    target = Live()\n"
        "switch()\n"
        "target = Dead()\n"
        "target.run()\n",
        encoding="utf-8",
    )
    valid = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/recovered_identity.py --dead yes\n",
    )
    invalid = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/recovered_identity.py --live yes\n",
    )

    assert cdc.check_python_script_contract(valid, tmp_path) == []
    assert cdc.check_python_script_contract(invalid, tmp_path)


def test_direct_argparse_constructor_parse_is_not_silently_skipped(
    tmp_path: Path,
) -> None:
    script = tmp_path / "tools" / "direct_parse.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import argparse\nargparse.ArgumentParser().parse_args()\n",
        encoding="utf-8",
    )
    block = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/direct_parse.py --implementation-specific\n",
    )

    assert [finding.message for finding in cdc.check_python_script_contract(block, tmp_path)] == [
        "unknown option for `tools/direct_parse.py`: --implementation-specific"
    ]


def test_uncalled_argparse_class_and_custom_parser_remain_ignored(
    tmp_path: Path,
) -> None:
    uncalled = tmp_path / "tools" / "uncalled_constructor.py"
    uncalled.parent.mkdir(parents=True)
    uncalled.write_text(
        "import argparse\n"
        "class Unused:\n"
        "    def __init__(self):\n"
        "        parser = argparse.ArgumentParser()\n"
        '        parser.add_argument("--dead", required=True)\n'
        "        parser.parse_args()\n"
        "def main():\n"
        "    return 0\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )
    custom = tmp_path / "tools" / "custom_single_root.py"
    custom.write_text(
        "class ArgumentParser:\n"
        "    def add_argument(self, *_args, **_kwargs):\n"
        "        return None\n"
        "    def parse_args(self):\n"
        "        return None\n"
        "def main():\n"
        "    parser = ArgumentParser()\n"
        '    parser.add_argument("--custom")\n'
        "    parser.parse_args()\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )
    uncalled_block = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/uncalled_constructor.py --dead yes\n",
    )
    custom_block = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/custom_single_root.py --implementation-specific\n",
    )

    assert cdc.check_python_script_contract(uncalled_block, tmp_path) == []
    assert cdc.check_python_script_contract(custom_block, tmp_path) == []


def test_multiple_reachable_argparse_roots_report_conservative_finding(
    tmp_path: Path,
) -> None:
    script = tmp_path / "tools" / "multiple_roots.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import argparse\n"
        "def main():\n"
        "    first = argparse.ArgumentParser()\n"
        '    first.add_argument("--first")\n'
        "    first.parse_args()\n"
        "    second = argparse.ArgumentParser()\n"
        '    second.add_argument("--second")\n'
        "    second.parse_args()\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )
    block = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/multiple_roots.py --first yes\n",
    )

    assert [finding.message for finding in cdc.check_command_block(block, tmp_path)] == [
        "cannot statically validate argparse contract for "
        "`tools/multiple_roots.py`: multiple reachable parser roots were discovered"
    ]


def test_unresolved_reachable_argparse_identity_reports_conservative_finding(
    tmp_path: Path,
) -> None:
    script = tmp_path / "tools" / "selected_root.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import argparse\n"
        "def select_parser(use_first):\n"
        "    first = argparse.ArgumentParser()\n"
        '    first.add_argument("--first")\n'
        "    second = argparse.ArgumentParser()\n"
        '    second.add_argument("--second")\n'
        "    return first if use_first else second\n"
        "def main():\n"
        "    return select_parser(True).parse_args()\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )
    block = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/selected_root.py --first yes\n",
    )

    assert [finding.message for finding in cdc.check_command_block(block, tmp_path)] == [
        "cannot statically validate argparse contract for "
        "`tools/selected_root.py`: "
        "a reachable parse call has an unresolved parser identity"
    ]


def test_no_argparse_script_is_not_reported_as_an_ambiguous_contract(
    tmp_path: Path,
) -> None:
    script = tmp_path / "tools" / "custom_parser.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "class ArgumentParser:\n"
        "    def add_argument(self, *_args, **_kwargs):\n"
        "        return None\n"
        "    def parse_args(self):\n"
        "        return None\n"
        "def main():\n"
        "    first = ArgumentParser()\n"
        '    first.add_argument("--first")\n'
        "    first.parse_args()\n"
        "    second = ArgumentParser()\n"
        '    second.add_argument("--second")\n'
        "    second.parse_args()\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )
    block = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        "python3 tools/custom_parser.py --implementation-specific\n",
    )

    assert cdc.check_command_block(block, tmp_path) == []


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


_FIFTH_IDENTITY_CASES = [
    (
        "I01_factory_returns_function",
        """
        def live():
            p = AP(); p.add_argument('--live', required=True); p.parse_args()
        def dead():
            p = AP(); p.add_argument('--dead', required=True); p.parse_args()
        def choose(first, second):
            return second
        target = choose(live, dead)
        target()
        """,
        "dead",
        False,
    ),
    (
        "I02_factory_returns_class",
        """
        class Live:
            def run(self):
                p = AP(); p.add_argument('--live', required=True); p.parse_args()
        class Dead:
            def run(self):
                p = AP(); p.add_argument('--dead', required=True); p.parse_args()
        def choose(first, second):
            return second
        Target = choose(Live, Dead)
        Target().run()
        """,
        "dead",
        False,
    ),
    (
        "I03_factory_returns_instance",
        """
        class Live:
            def run(self):
                p = AP(); p.add_argument('--live', required=True); p.parse_args()
        class Dead:
            def run(self):
                p = AP(); p.add_argument('--dead', required=True); p.parse_args()
        def choose(first, second):
            return second
        target = choose(Live(), Dead())
        target.run()
        """,
        "dead",
        False,
    ),
    (
        "I04_imported_factory",
        """
        from factory_helper import choose
        def live():
            p = AP(); p.add_argument('--live', required=True); p.parse_args()
        def dead():
            p = AP(); p.add_argument('--dead', required=True); p.parse_args()
        target = choose(live, dead)
        target()
        """,
        "dead",
        False,
    ),
    (
        "I05_default_factory",
        """
        def live():
            p = AP(); p.add_argument('--live', required=True); p.parse_args()
        def dead():
            p = AP(); p.add_argument('--dead', required=True); p.parse_args()
        def choose(first, second):
            return second
        def invoke(fn=choose(live, dead)):
            fn()
        invoke()
        """,
        "dead",
        False,
    ),
    (
        "I06_method_factory",
        """
        def live():
            p = AP(); p.add_argument('--live', required=True); p.parse_args()
        def dead():
            p = AP(); p.add_argument('--dead', required=True); p.parse_args()
        class Chooser:
            def choose(self, first, second):
                return second
        target = Chooser().choose(live, dead)
        target()
        """,
        "dead",
        False,
    ),
    (
        "I07_factory_then_direct_recovery",
        """
        def live():
            p = AP(); p.add_argument('--live', required=True); p.parse_args()
        def dead():
            p = AP(); p.add_argument('--dead', required=True); p.parse_args()
        def choose(first, second):
            return second
        target = choose(live, dead)
        target = live
        target()
        """,
        "live",
        True,
    ),
    (
        "I08_unrelated_factory_result",
        """
        class Live:
            def run(self):
                p = AP(); p.add_argument('--live', required=True); p.parse_args()
        def choose(first, second):
            return second
        unrelated = choose(1, 2)
        Live().run()
        """,
        "live",
        True,
    ),
    (
        "I09_imported_globals_mutation",
        """
        from factory_helper import switch
        class Live:
            def run(self):
                p = AP(); p.add_argument('--live', required=True); p.parse_args()
        class Dead:
            def run(self):
                p = AP(); p.add_argument('--dead', required=True); p.parse_args()
        target = Live()
        switch(globals())
        target.run()
        """,
        "dead",
        False,
    ),
    (
        "I10_imported_instance_mutation",
        """
        from factory_helper import mutate
        class Live:
            def run(self):
                p = AP(); p.add_argument('--live', required=True); p.parse_args()
        class Dead:
            def run(self):
                p = AP(); p.add_argument('--dead', required=True); p.parse_args()
        target = Live()
        mutate(target, Dead)
        target.run()
        """,
        "dead",
        False,
    ),
    (
        "I11_shadowed_property_decorator",
        """
        def dead():
            p = AP(); p.add_argument('--dead', required=True); p.parse_args()
        def property(value):
            return dead
        @property
        def target():
            p = AP(); p.add_argument('--live', required=True); p.parse_args()
        target()
        """,
        "dead",
        False,
    ),
    (
        "I12_shadowed_cache_decorator",
        """
        class Dead:
            def run(self):
                p = AP(); p.add_argument('--dead', required=True); p.parse_args()
        def cache(value):
            return Dead
        @cache
        class Target:
            def run(self):
                p = AP(); p.add_argument('--live', required=True); p.parse_args()
        Target().run()
        """,
        "dead",
        False,
    ),
    (
        "I13_mutated_functools_decorator",
        """
        import functools
        def dead():
            p = AP(); p.add_argument('--dead', required=True); p.parse_args()
        def replacement():
            return lambda value: dead
        functools.lru_cache = replacement
        @functools.lru_cache()
        def target():
            p = AP(); p.add_argument('--live', required=True); p.parse_args()
        target()
        """,
        "dead",
        False,
    ),
    (
        "I14_opaque_decorator_new_symbol_recovery",
        """
        def replace(value):
            return object()
        @replace
        class Target:
            pass
        class Dead:
            def run(self):
                p = AP(); p.add_argument('--dead', required=True); p.parse_args()
        Target = Dead
        Target().run()
        """,
        "dead",
        True,
    ),
    (
        "I15_opaque_decorator_existing_symbol_recovery",
        """
        class Dead:
            def run(self):
                p = AP(); p.add_argument('--dead', required=True); p.parse_args()
        def replace(value):
            return object()
        @replace
        class Target:
            pass
        Target = Dead
        Target().run()
        """,
        "dead",
        True,
    ),
    (
        "I16_global_alias_recovery",
        """
        class Live:
            def run(self):
                p = AP(); p.add_argument('--live', required=True); p.parse_args()
        class Dead:
            def run(self):
                p = AP(); p.add_argument('--dead', required=True); p.parse_args()
        target = Live()
        def switch():
            global target
            target = Live()
        switch()
        fresh = Dead()
        target = fresh
        target.run()
        """,
        "dead",
        True,
    ),
    (
        "I17_nonlocal_direct_recovery",
        """
        def outer():
            class Live:
                def run(self):
                    p = AP(); p.add_argument('--live', required=True); p.parse_args()
            class Dead:
                def run(self):
                    p = AP(); p.add_argument('--dead', required=True); p.parse_args()
            target = Live()
            def switch():
                nonlocal target
                target = Live()
            switch()
            target = Dead()
            target.run()
        outer()
        """,
        "dead",
        True,
    ),
    (
        "I18_delete_direct_recovery",
        """
        class Live:
            def run(self):
                p = AP(); p.add_argument('--live', required=True); p.parse_args()
        class Dead:
            def run(self):
                p = AP(); p.add_argument('--dead', required=True); p.parse_args()
        target = Live()
        del target
        target = Dead()
        target.run()
        """,
        "dead",
        True,
    ),
    (
        "I19_unrelated_global_write",
        """
        class Live:
            def run(self):
                p = AP(); p.add_argument('--live', required=True); p.parse_args()
        other = 0
        def switch():
            global other
            other = 1
        switch()
        Live().run()
        """,
        "live",
        True,
    ),
    (
        "I20_unrelated_nonlocal_write",
        """
        def outer():
            class Live:
                def run(self):
                    p = AP(); p.add_argument('--live', required=True); p.parse_args()
            other = 0
            def switch():
                nonlocal other
                other = 1
            switch()
            Live().run()
        outer()
        """,
        "live",
        True,
    ),
    (
        "I21_same_line_global_recovery",
        """
        class Live:
            def run(self):
                p = AP(); p.add_argument('--live', required=True); p.parse_args()
        class Dead:
            def run(self):
                p = AP(); p.add_argument('--dead', required=True); p.parse_args()
        target = Live()
        def switch():
            global target
            target = Live()
        switch(); target = Dead(); target.run()
        """,
        "dead",
        True,
    ),
    (
        "I22_real_lru_cache",
        """
        from functools import lru_cache
        @lru_cache()
        def target():
            p = AP(); p.add_argument('--live', required=True); p.parse_args()
        target()
        """,
        "live",
        True,
    ),
    (
        "I23_real_classmethod",
        """
        class Target:
            @classmethod
            def run(cls):
                p = AP(); p.add_argument('--live', required=True); p.parse_args()
        Target.run()
        """,
        "live",
        True,
    ),
    (
        "I24_supplied_default_function",
        """
        def live():
            p = AP(); p.add_argument('--live', required=True); p.parse_args()
        def dead():
            p = AP(); p.add_argument('--dead', required=True); p.parse_args()
        def invoke(fn=live):
            fn()
        invoke(dead)
        """,
        "dead",
        False,
    ),
    (
        "I25_supplied_function_parameter",
        """
        def live():
            p = AP(); p.add_argument('--live', required=True); p.parse_args()
        def dead():
            p = AP(); p.add_argument('--dead', required=True); p.parse_args()
        def invoke(fn):
            fn()
        invoke(dead)
        """,
        "dead",
        False,
    ),
]


def _write_fifth_factory_helper(tmp_path: Path) -> None:
    helper = tmp_path / "tools" / "factory_helper.py"
    helper.parent.mkdir(parents=True, exist_ok=True)
    helper.write_text(
        "def choose(first, second):\n"
        "    return second\n"
        "def switch(namespace):\n"
        "    namespace['target'] = namespace['Dead']()\n"
        "def mutate(instance, replacement):\n"
        "    instance.__class__ = replacement\n",
        encoding="utf-8",
    )


def _fifth_contract_outcome(
    tmp_path: Path,
    script: Path,
    argv: list[str],
) -> tuple[int, list[str]]:
    completed = subprocess.run(
        [sys.executable, str(script), *argv],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
    )
    command = cdc.ShellBlock(
        Path("README.md"),
        1,
        "bash",
        f"python3 tools/{script.name} {' '.join(argv)}\n",
    )
    findings = [finding.message for finding in cdc.check_python_script_contract(command, tmp_path)]
    return completed.returncode, findings


@pytest.mark.parametrize(
    ("case_id", "source", "runtime_flag", "statically_exact"),
    _FIFTH_IDENTITY_CASES,
    ids=[case[0] for case in _FIFTH_IDENTITY_CASES],
)
def test_argparse_fifth_identity_graph_matches_python(
    tmp_path: Path,
    case_id: str,
    source: str,
    runtime_flag: str,
    statically_exact: bool,
) -> None:
    _write_fifth_factory_helper(tmp_path)
    script = tmp_path / "tools" / f"{case_id}.py"
    script.write_text(
        "from argparse import ArgumentParser as AP\n" + textwrap.dedent(source),
        encoding="utf-8",
    )

    outcomes: dict[str, tuple[int, list[str]]] = {}
    for flag_name in ("live", "dead"):
        outcomes[flag_name] = _fifth_contract_outcome(
            tmp_path,
            script,
            [f"--{flag_name}", "yes"],
        )
        returncode, findings = outcomes[flag_name]
        if returncode != 0:
            assert findings, f"{case_id} false pass for --{flag_name}"

    rejected = "dead" if runtime_flag == "live" else "live"
    assert outcomes[runtime_flag][0] == 0
    assert outcomes[rejected][0] != 0
    if statically_exact:
        assert outcomes[runtime_flag][1] == []
        assert outcomes[rejected][1]


_FIFTH_SUBCOMMAND_CASES = [
    (
        "S01_literal_loop",
        """
        p = AP()
        sub = p.add_subparsers(required=True)
        for name in ('run', 'check'):
            cmd = sub.add_parser(name)
            cmd.add_argument('--mode', choices=('fast', 'safe'), required=True)
        p.parse_args()
        """,
        ["run", "--mode", "fast"],
        ["removed", "--mode", "fast"],
        True,
    ),
    (
        "S02_if_false_subcommand",
        """
        p = AP()
        sub = p.add_subparsers(required=True)
        if False:
            sub.add_parser('dead')
        sub.add_parser('live')
        p.parse_args()
        """,
        ["live"],
        ["dead"],
        False,
    ),
    (
        "S03_if_true_subcommand",
        """
        p = AP()
        sub = p.add_subparsers(required=True)
        if True:
            sub.add_parser('live')
        p.parse_args()
        """,
        ["live"],
        ["dead"],
        True,
    ),
    (
        "S04_empty_literal_loop",
        """
        p = AP()
        sub = p.add_subparsers(required=True)
        for name in ():
            sub.add_parser(name)
        p.parse_args()
        """,
        ["--help"],
        ["ghost"],
        False,
    ),
    (
        "S05_loop_with_static_filter",
        """
        p = AP()
        sub = p.add_subparsers(required=True)
        for name in ('live', 'dead'):
            if name == 'live':
                sub.add_parser(name)
        p.parse_args()
        """,
        ["live"],
        ["dead"],
        True,
    ),
    (
        "S06_two_literal_loops",
        """
        p = AP()
        sub = p.add_subparsers(required=True)
        for name in ('one',):
            sub.add_parser(name)
        for name in ('two',):
            sub.add_parser(name)
        p.parse_args()
        """,
        ["two"],
        ["three"],
        True,
    ),
    (
        "S07_if_false_option",
        """
        p = AP()
        if False:
            p.add_argument('--dead', required=True)
        p.add_argument('--live', required=True)
        p.parse_args()
        """,
        ["--live", "yes"],
        ["--dead", "yes", "--live", "yes"],
        False,
    ),
    (
        "S08_if_true_required_option",
        """
        p = AP()
        if True:
            p.add_argument('--live', required=True)
        p.parse_args()
        """,
        ["--live", "yes"],
        ["--dead", "yes"],
        True,
    ),
]


@pytest.mark.parametrize(
    ("case_id", "source", "accepted", "rejected", "statically_exact"),
    _FIFTH_SUBCOMMAND_CASES,
    ids=[case[0] for case in _FIFTH_SUBCOMMAND_CASES],
)
def test_argparse_fifth_subcommand_contract_matches_python(
    tmp_path: Path,
    case_id: str,
    source: str,
    accepted: list[str],
    rejected: list[str],
    statically_exact: bool,
) -> None:
    script = tmp_path / "tools" / f"{case_id}.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "from argparse import ArgumentParser as AP\n" + textwrap.dedent(source),
        encoding="utf-8",
    )

    good = _fifth_contract_outcome(tmp_path, script, accepted)
    bad = _fifth_contract_outcome(tmp_path, script, rejected)
    assert good[0] == 0
    assert bad[0] != 0
    assert bad[1], f"{case_id} false pass for {' '.join(rejected)}"
    if statically_exact:
        assert good[1] == []
        assert bad[1]

    if case_id == "S08_if_true_required_option":
        dynamic = tmp_path / "tools" / "dynamic_declaration.py"
        dynamic.write_text(
            "from argparse import ArgumentParser as AP\n"
            "import os\n"
            "p = AP()\n"
            "sub = p.add_subparsers(required=True)\n"
            "enabled = bool(os.environ.get('ENABLE_MAYBE'))\n"
            "if enabled:\n"
            "    sub.add_parser('maybe')\n"
            "sub.add_parser('live')\n"
            "p.parse_args()\n",
            encoding="utf-8",
        )
        dynamic_findings = _fifth_contract_outcome(
            tmp_path,
            dynamic,
            ["live"],
        )[1]
        assert any(
            "a reachable argparse declaration has dynamic control-flow reachability" in finding
            for finding in dynamic_findings
        )


_SEVENTH_BASE_CLASSES = """
        class Live:
            def run(self):
                p = AP(); p.add_argument('--live', required=True); p.parse_args()
        class Dead:
            def run(self):
                p = AP(); p.add_argument('--dead', required=True); p.parse_args()
"""


_SEVENTH_IDENTITY_CASES = [
    (
        "U01_keyword_instance_mutation",
        _SEVENTH_BASE_CLASSES
        + """
        from mutation_helper import mutate
        target = Live()
        mutate(instance=target, replacement=Dead)
        target.run()
        """,
        "dead",
        False,
    ),
    (
        "U02_starred_instance_mutation",
        _SEVENTH_BASE_CLASSES
        + """
        from mutation_helper import mutate
        target = Live()
        mutate(*(target, Dead))
        target.run()
        """,
        "dead",
        False,
    ),
    (
        "U03_alias_instance_mutation",
        _SEVENTH_BASE_CLASSES
        + """
        from mutation_helper import mutate
        target = Live()
        alias = target
        mutate(alias, Dead)
        target.run()
        """,
        "dead",
        False,
    ),
    (
        "U04_container_instance_mutation",
        _SEVENTH_BASE_CLASSES
        + """
        from mutation_helper import mutate_box
        target = Live()
        mutate_box([target], Dead)
        target.run()
        """,
        "dead",
        False,
    ),
    (
        "U05_lambda_factory",
        """
        def live():
            p = AP(); p.add_argument('--live', required=True); p.parse_args()
        def dead():
            p = AP(); p.add_argument('--dead', required=True); p.parse_args()
        choose = lambda first, second: second
        target = choose(live, dead)
        target()
        """,
        "dead",
        False,
    ),
    (
        "U06_partial_callable",
        """
        from functools import partial
        def live():
            p = AP(); p.add_argument('--live', required=True); p.parse_args()
        def dead(prefix=''):
            p = AP(); p.add_argument('--dead', required=True); p.parse_args()
        target = partial(dead)
        target()
        """,
        "dead",
        False,
    ),
    (
        "U07_static_false_rebind",
        _SEVENTH_BASE_CLASSES
        + """
        target = Live()
        if False:
            target = Dead()
        target.run()
        """,
        "live",
        True,
    ),
    (
        "U08_static_true_factory_recovery",
        _SEVENTH_BASE_CLASSES
        + """
        def choose(first, second):
            return second
        target = choose(Live(), Dead())
        if True:
            target = Live()
        target.run()
        """,
        "live",
        True,
    ),
    (
        "D01_global_decorator_rebind",
        """
        from functools import lru_cache as memo
        def dead():
            p = AP(); p.add_argument('--dead', required=True); p.parse_args()
        def replacement():
            return lambda value: dead
        def switch():
            global memo
            memo = replacement
        switch()
        @memo()
        def target():
            p = AP(); p.add_argument('--live', required=True); p.parse_args()
        target()
        """,
        "dead",
        False,
    ),
    (
        "D02_setattr_module_decorator",
        """
        import functools
        def dead():
            p = AP(); p.add_argument('--dead', required=True); p.parse_args()
        def replacement():
            return lambda value: dead
        setattr(functools, 'lru_cache', replacement)
        @functools.lru_cache()
        def target():
            p = AP(); p.add_argument('--live', required=True); p.parse_args()
        target()
        """,
        "dead",
        False,
    ),
    (
        "D03_mutated_builtin_property",
        """
        import builtins
        def dead():
            p = AP(); p.add_argument('--dead', required=True); p.parse_args()
        def replacement(value):
            return dead
        builtins.property = replacement
        @property
        def target():
            p = AP(); p.add_argument('--live', required=True); p.parse_args()
        target()
        """,
        "dead",
        False,
    ),
    (
        "D04_unreachable_import_provenance",
        """
        def dead():
            p = AP(); p.add_argument('--dead', required=True); p.parse_args()
        class Fake:
            @staticmethod
            def lru_cache():
                return lambda value: dead
        functools = Fake
        if False:
            import functools
        @functools.lru_cache()
        def target():
            p = AP(); p.add_argument('--live', required=True); p.parse_args()
        target()
        """,
        "dead",
        False,
    ),
    (
        "D05_deleted_decorator_fallback",
        """
        from functools import lru_cache as memo
        def dead():
            p = AP(); p.add_argument('--dead', required=True); p.parse_args()
        def replacement():
            return lambda value: dead
        del memo
        try:
            memo
        except NameError:
            memo = replacement
        @memo()
        def target():
            p = AP(); p.add_argument('--live', required=True); p.parse_args()
        target()
        """,
        "dead",
        False,
    ),
    (
        "D06_real_import_alias",
        """
        from functools import lru_cache as memo
        @memo()
        def target():
            p = AP(); p.add_argument('--live', required=True); p.parse_args()
        target()
        """,
        "live",
        True,
    ),
    (
        "D07_module_alias_direct_mutation",
        """
        import functools as ft
        def dead():
            p = AP(); p.add_argument('--dead', required=True); p.parse_args()
        def replacement():
            return lambda value: dead
        ft.lru_cache = replacement
        @ft.lru_cache()
        def target():
            p = AP(); p.add_argument('--live', required=True); p.parse_args()
        target()
        """,
        "dead",
        False,
    ),
    (
        "D08_decorated_symbol_class_recovery",
        """
        def replacement(value):
            return object()
        @replacement
        class Target:
            pass
        class Target:
            def run(self):
                p = AP(); p.add_argument('--dead', required=True); p.parse_args()
        Target().run()
        """,
        "dead",
        True,
    ),
]


def _write_seventh_mutation_helper(tmp_path: Path) -> None:
    helper = tmp_path / "tools" / "mutation_helper.py"
    helper.parent.mkdir(parents=True, exist_ok=True)
    helper.write_text(
        "def mutate(instance, replacement):\n"
        "    instance.__class__ = replacement\n"
        "def mutate_box(box, replacement):\n"
        "    box[0].__class__ = replacement\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("case_id", "source", "runtime_flag", "statically_exact"),
    _SEVENTH_IDENTITY_CASES,
    ids=[case[0] for case in _SEVENTH_IDENTITY_CASES],
)
def test_argparse_seventh_identity_and_decorator_graph_matches_python(
    tmp_path: Path,
    case_id: str,
    source: str,
    runtime_flag: str,
    statically_exact: bool,
) -> None:
    _write_seventh_mutation_helper(tmp_path)
    script = tmp_path / "tools" / f"{case_id}.py"
    script.write_text(
        "from argparse import ArgumentParser as AP\n" + textwrap.dedent(source),
        encoding="utf-8",
    )

    outcomes = {
        flag_name: _fifth_contract_outcome(
            tmp_path,
            script,
            [f"--{flag_name}", "yes"],
        )
        for flag_name in ("live", "dead")
    }
    rejected = "dead" if runtime_flag == "live" else "live"
    assert outcomes[runtime_flag][0] == 0
    assert outcomes[rejected][0] != 0
    assert outcomes[rejected][1], f"{case_id} false pass for --{rejected}"
    if statically_exact:
        assert outcomes[runtime_flag][1] == []
        assert outcomes[rejected][1]


_SEVENTH_CONTROL_CASES = [
    (
        "C01_raise_before_option",
        """
        p = AP()
        try:
            raise RuntimeError()
            p.add_argument('--dead', required=True)
        except RuntimeError:
            pass
        p.add_argument('--live', required=True)
        p.parse_args()
        """,
        ["--live", "yes"],
        ["--dead", "yes", "--live", "yes"],
        True,
    ),
    (
        "C02_unreached_except_option",
        """
        p = AP()
        try:
            pass
        except Exception:
            p.add_argument('--dead', required=True)
        p.add_argument('--live', required=True)
        p.parse_args()
        """,
        ["--live", "yes"],
        ["--dead", "yes", "--live", "yes"],
        True,
    ),
    (
        "C03_static_match_branch",
        """
        p = AP()
        match 1:
            case 0:
                p.add_argument('--dead', required=True)
            case 1:
                p.add_argument('--live', required=True)
        p.parse_args()
        """,
        ["--live", "yes"],
        ["--dead", "yes", "--live", "yes"],
        True,
    ),
    (
        "C04_while_break_else_subcommand",
        """
        p = AP()
        sub = p.add_subparsers(required=True)
        while True:
            break
        else:
            sub.add_parser('dead')
        sub.add_parser('live')
        p.parse_args()
        """,
        ["live"],
        ["dead"],
        True,
    ),
    (
        "C05_continue_before_subcommand",
        """
        p = AP()
        sub = p.add_subparsers(required=True)
        for name in ('dead',):
            continue
            sub.add_parser(name)
        sub.add_parser('live')
        p.parse_args()
        """,
        ["live"],
        ["dead"],
        True,
    ),
    (
        "C06_dynamic_if_fails_closed",
        """
        import os
        p = AP()
        if os.getenv('ENABLE_DEAD'):
            p.add_argument('--dead')
        p.add_argument('--live', required=True)
        p.parse_args()
        """,
        ["--live", "yes"],
        ["--dead", "yes", "--live", "yes"],
        False,
    ),
    (
        "C07_dynamic_loop_fails_closed",
        """
        import os
        p = AP()
        sub = p.add_subparsers(required=True)
        for name in os.getenv('EXTRA_COMMANDS', '').split():
            sub.add_parser(name)
        sub.add_parser('live')
        p.parse_args()
        """,
        ["live"],
        ["dead"],
        False,
    ),
    (
        "C08_literal_loop_membership_filter",
        """
        p = AP()
        sub = p.add_subparsers(required=True)
        for name in ('one', 'two', 'skip'):
            if name in ('one', 'two'):
                sub.add_parser(name)
        p.parse_args()
        """,
        ["two"],
        ["skip"],
        True,
    ),
]


@pytest.mark.parametrize(
    ("case_id", "source", "accepted", "rejected", "statically_exact"),
    _SEVENTH_CONTROL_CASES,
    ids=[case[0] for case in _SEVENTH_CONTROL_CASES],
)
def test_argparse_seventh_control_flow_matches_python(
    tmp_path: Path,
    case_id: str,
    source: str,
    accepted: list[str],
    rejected: list[str],
    statically_exact: bool,
) -> None:
    script = tmp_path / "tools" / f"{case_id}.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "from argparse import ArgumentParser as AP\n" + textwrap.dedent(source),
        encoding="utf-8",
    )

    good = _fifth_contract_outcome(tmp_path, script, accepted)
    bad = _fifth_contract_outcome(tmp_path, script, rejected)
    assert good[0] == 0
    assert bad[0] != 0
    assert bad[1], f"{case_id} false pass for {' '.join(rejected)}"
    if statically_exact:
        assert good[1] == []
        assert bad[1]


_EIGHTH_BASE_CLASSES = """
        class Live:
            def run(self):
                p = AP()
                p.add_argument('--live', required=True)
                p.parse_args()

        class Dead:
            def run(self):
                p = AP()
                p.add_argument('--dead', required=True)
                p.parse_args()
"""


_EIGHTH_ARGPARSE_CASES = [
    (
        "T01_return_runs_finally_parser",
        """
        def run():
            try:
                return
            finally:
                p = AP()
                p.add_argument('--live', required=True)
                p.parse_args()
        run()
        """,
        ["--live", "yes"],
        ["--dead", "yes"],
        True,
    ),
    (
        "T02_return_skips_try_tail",
        """
        def run():
            try:
                return
                p = AP()
                p.add_argument('--dead', required=True)
                p.parse_args()
            finally:
                pass
        run()
        """,
        ["--live", "yes"],
        None,
        True,
    ),
    (
        "T03_finally_return_skips_following_parser",
        """
        def run():
            try:
                pass
            finally:
                return
            p = AP()
            p.add_argument('--dead', required=True)
            p.parse_args()
        run()
        """,
        ["--live", "yes"],
        None,
        True,
    ),
    (
        "T04_raise_runs_finally_before_handler",
        """
        def run():
            try:
                try:
                    raise RuntimeError('boom')
                finally:
                    p = AP()
                    p.add_argument('--live', required=True)
                    p.parse_args()
            except RuntimeError:
                pass
        run()
        """,
        ["--live", "yes"],
        ["--dead", "yes"],
        False,
    ),
    (
        "M01_false_guard_falls_through",
        """
        p = AP()
        match 1:
            case 1 if False:
                p.add_argument('--dead', required=True)
            case 1:
                p.add_argument('--live', required=True)
        p.parse_args()
        """,
        ["--live", "yes"],
        ["--dead", "yes", "--live", "yes"],
        True,
    ),
    (
        "M02_dynamic_guard_fails_closed",
        """
        import os
        p = AP()
        match 1:
            case 1 if os.getenv('ENABLE_DEAD'):
                p.add_argument('--dead', required=True)
            case _:
                p.add_argument('--live', required=True)
        p.parse_args()
        """,
        ["--live", "yes"],
        ["--dead", "yes", "--live", "yes"],
        False,
    ),
    (
        "M03_true_guard_stops_fallthrough",
        """
        p = AP()
        match 1:
            case 1 if True:
                p.add_argument('--live', required=True)
            case _:
                p.add_argument('--dead', required=True)
        p.parse_args()
        """,
        ["--live", "yes"],
        ["--dead", "yes", "--live", "yes"],
        True,
    ),
    (
        "W01_false_while_runs_else",
        """
        p = AP()
        while False:
            p.add_argument('--dead', required=True)
        else:
            p.add_argument('--live', required=True)
        p.parse_args()
        """,
        ["--live", "yes"],
        ["--dead", "yes", "--live", "yes"],
        True,
    ),
    (
        "W02_stateful_while_runs_else",
        """
        p = AP()
        remaining = 1
        while remaining:
            remaining -= 1
        else:
            p.add_argument('--live', required=True)
        p.parse_args()
        """,
        ["--live", "yes"],
        ["--dead", "yes", "--live", "yes"],
        False,
    ),
    (
        "W03_callable_false_while_runs_else",
        """
        def keep_going():
            return False
        p = AP()
        while keep_going():
            p.add_argument('--dead', required=True)
        else:
            p.add_argument('--live', required=True)
        p.parse_args()
        """,
        ["--live", "yes"],
        ["--dead", "yes", "--live", "yes"],
        False,
    ),
    (
        "C01_enter_rebinds_global_target",
        _EIGHTH_BASE_CLASSES
        + """
        target = Live()
        class Swap:
            def __enter__(self):
                global target
                target = Dead()
            def __exit__(self, exc_type, exc, tb):
                return False
        with Swap():
            pass
        target.run()
        """,
        ["--dead", "yes"],
        ["--live", "yes"],
        False,
    ),
    (
        "C02_exit_rebinds_global_target",
        _EIGHTH_BASE_CLASSES
        + """
        target = Live()
        class Swap:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                global target
                target = Dead()
                return False
        with Swap():
            pass
        target.run()
        """,
        ["--dead", "yes"],
        ["--live", "yes"],
        False,
    ),
    (
        "C03_enter_mutates_alias_identity",
        _EIGHTH_BASE_CLASSES
        + """
        target = Live()
        alias = target
        class Swap:
            def __enter__(self):
                alias.__class__ = Dead
            def __exit__(self, exc_type, exc, tb):
                return False
        with Swap():
            pass
        target.run()
        """,
        ["--dead", "yes"],
        ["--live", "yes"],
        False,
    ),
    (
        "K01_direct_nested_kwargs",
        _EIGHTH_BASE_CLASSES
        + """
        from mutation_helper import mutate_kwargs
        target = Live()
        mutate_kwargs(**{
            'payload': {'layers': [target]},
            'replacement': Dead,
        })
        target.run()
        """,
        ["--dead", "yes"],
        ["--live", "yes"],
        False,
    ),
    (
        "K02_bound_nested_kwargs",
        _EIGHTH_BASE_CLASSES
        + """
        from mutation_helper import mutate_kwargs
        target = Live()
        payload = {
            'payload': {'layers': [target]},
            'replacement': Dead,
        }
        mutate_kwargs(**payload)
        target.run()
        """,
        ["--dead", "yes"],
        ["--live", "yes"],
        False,
    ),
    (
        "K03_call_built_nested_kwargs",
        _EIGHTH_BASE_CLASSES
        + """
        from mutation_helper import mutate_kwargs
        target = Live()
        mutate_kwargs(**dict(
            payload={'layers': [target]},
            replacement=Dead,
        ))
        target.run()
        """,
        ["--dead", "yes"],
        ["--live", "yes"],
        False,
    ),
    (
        "D01_module_dict_subscript_mutation",
        """
        import functools
        def dead():
            p = AP()
            p.add_argument('--dead', required=True)
            p.parse_args()
        def replacement():
            return lambda value: dead
        functools.__dict__['lru_cache'] = replacement
        @functools.lru_cache()
        def target():
            p = AP()
            p.add_argument('--live', required=True)
            p.parse_args()
        target()
        """,
        ["--dead", "yes"],
        ["--live", "yes"],
        False,
    ),
    (
        "D02_vars_subscript_mutation",
        """
        import functools
        def dead():
            p = AP()
            p.add_argument('--dead', required=True)
            p.parse_args()
        def replacement():
            return lambda value: dead
        vars(functools)['lru_cache'] = replacement
        @functools.lru_cache()
        def target():
            p = AP()
            p.add_argument('--live', required=True)
            p.parse_args()
        target()
        """,
        ["--dead", "yes"],
        ["--live", "yes"],
        False,
    ),
    (
        "D03_namespace_update_mutation",
        """
        import functools
        def dead():
            p = AP()
            p.add_argument('--dead', required=True)
            p.parse_args()
        def replacement():
            return lambda value: dead
        functools.__dict__.update({'lru_cache': replacement})
        @functools.lru_cache()
        def target():
            p = AP()
            p.add_argument('--live', required=True)
            p.parse_args()
        target()
        """,
        ["--dead", "yes"],
        ["--live", "yes"],
        False,
    ),
    (
        "D04_builtin_descriptor_dict_mutation",
        """
        import builtins
        def dead():
            p = AP()
            p.add_argument('--dead', required=True)
            p.parse_args()
        def replacement(value):
            return dead
        vars(builtins)['property'] = replacement
        @property
        def target():
            p = AP()
            p.add_argument('--live', required=True)
            p.parse_args()
        target()
        """,
        ["--dead", "yes"],
        ["--live", "yes"],
        False,
    ),
]


def _write_eighth_mutation_helper(tmp_path: Path) -> None:
    helper = tmp_path / "tools" / "mutation_helper.py"
    helper.parent.mkdir(parents=True, exist_ok=True)
    helper.write_text(
        "def mutate_kwargs(**kwargs):\n"
        "    target = kwargs['payload']['layers'][0]\n"
        "    target.__class__ = kwargs['replacement']\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("case_id", "source", "accepted", "rejected", "statically_exact"),
    _EIGHTH_ARGPARSE_CASES,
    ids=[case[0] for case in _EIGHTH_ARGPARSE_CASES],
)
def test_argparse_eighth_control_and_identity_graph_matches_python(
    tmp_path: Path,
    case_id: str,
    source: str,
    accepted: list[str],
    rejected: list[str] | None,
    statically_exact: bool,
) -> None:
    _write_eighth_mutation_helper(tmp_path)
    script = tmp_path / "tools" / f"{case_id}.py"
    script.write_text(
        "from argparse import ArgumentParser as AP\n" + textwrap.dedent(source),
        encoding="utf-8",
    )

    good = _fifth_contract_outcome(tmp_path, script, accepted)
    assert good[0] == 0
    if rejected is None:
        alternate = _fifth_contract_outcome(
            tmp_path,
            script,
            ["--dead", "yes"],
        )
        assert alternate[0] == 0
        if statically_exact:
            assert good[1] == []
            assert alternate[1] == []
        return

    bad = _fifth_contract_outcome(tmp_path, script, rejected)
    assert bad[0] != 0
    assert bad[1], f"{case_id} false pass for {' '.join(rejected)}"
    if statically_exact:
        assert good[1] == []
        assert bad[1]


_CONTEXTMANAGER_SCANDIR_CASES = [
    (
        "G01_contextmanager_rebind_before_yield",
        _EIGHTH_BASE_CLASSES
        + """
        from contextlib import contextmanager
        parser = Live()
        @contextmanager
        def swap():
            global parser
            parser = Dead()
            yield
        with swap():
            pass
        parser.run()
        """,
        ["--dead", "yes"],
        ["--live", "yes"],
        False,
    ),
    (
        "G02_contextmanager_rebind_after_yield",
        _EIGHTH_BASE_CLASSES
        + """
        from contextlib import contextmanager
        parser = Live()
        @contextmanager
        def swap():
            global parser
            yield
            parser = Dead()
        with swap():
            pass
        parser.run()
        """,
        ["--dead", "yes"],
        ["--live", "yes"],
        False,
    ),
    (
        "S01_unmodified_os_scandir",
        """
        import os
        with os.scandir('.') as entries:
            list(entries)
        p = AP()
        p.add_argument('--live', required=True)
        p.parse_args()
        """,
        ["--live", "yes"],
        ["--dead", "yes", "--live", "yes"],
        True,
    ),
    (
        "S02_directly_rewritten_os_scandir",
        _EIGHTH_BASE_CLASSES
        + """
        import os
        from contextlib import contextmanager
        parser = Live()
        @contextmanager
        def replacement(path):
            global parser
            parser = Dead()
            yield ()
        os.scandir = replacement
        with os.scandir('.') as entries:
            pass
        parser.run()
        """,
        ["--dead", "yes"],
        ["--live", "yes"],
        False,
    ),
    (
        "S03_indirectly_rewritten_os_scandir",
        _EIGHTH_BASE_CLASSES
        + """
        import os
        from contextlib import contextmanager
        parser = Live()
        @contextmanager
        def replacement(path):
            global parser
            parser = Dead()
            yield ()
        setattr(os, 'scandir', replacement)
        with os.scandir('.') as entries:
            pass
        parser.run()
        """,
        ["--dead", "yes"],
        ["--live", "yes"],
        False,
    ),
]


@pytest.mark.parametrize(
    ("case_id", "source", "accepted", "rejected", "statically_exact"),
    _CONTEXTMANAGER_SCANDIR_CASES,
    ids=[case[0] for case in _CONTEXTMANAGER_SCANDIR_CASES],
)
def test_argparse_contextmanager_and_external_context_provenance(
    tmp_path: Path,
    case_id: str,
    source: str,
    accepted: list[str],
    rejected: list[str],
    statically_exact: bool,
) -> None:
    script = tmp_path / "tools" / f"{case_id}.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "from argparse import ArgumentParser as AP\n" + textwrap.dedent(source),
        encoding="utf-8",
    )

    good = _fifth_contract_outcome(tmp_path, script, accepted)
    bad = _fifth_contract_outcome(tmp_path, script, rejected)
    assert good[0] == 0
    assert bad[0] != 0
    if statically_exact:
        assert good[1] == []
        assert bad[1]
    else:
        assert good[1], f"{case_id} did not fail closed for {' '.join(accepted)}"
        assert bad[1], f"{case_id} did not fail closed for {' '.join(rejected)}"
