#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate shell examples in Git-tracked Markdown documentation.

The checker is deliberately non-executing: it parses every ``bash``, ``sh``,
or ``shell`` fenced block with ``bash -n`` and verifies repository-local
scripts and search/test input paths. Commands that need a GPU, model download,
credentials, services, or generated artifacts remain runtime validation work.

Usage:
    python3 tools/check_doc_commands.py
    python3 tools/check_doc_commands.py website/docs plugins/
"""

from __future__ import annotations

import argparse
import ast
import re
import shlex
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


SHELL_LANGUAGES = {"bash", "sh", "shell"}
SKIPPED_DIRS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "build",
    "node_modules",
}
_OPEN_FENCE_RE = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>`{3,}|~{3,})[ \t]*(?P<info>.*)$")
_PLACEHOLDER_RE = re.compile(r"<[^>\n]+>")
_PROMPT_RE = re.compile(r"^(?P<indent>[ \t]*)\$\s+")
_ENV_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_INLINE_CODE_RE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
_COMMAND_HEAD_RE = re.compile(
    r"^(?:\$\s+)?(?:env\s+)?"
    r"(?:[A-Za-z_][A-Za-z0-9_]*=\S+\s+)*"
    r"(?:python3?|pytest|py\.test|pip3?|cmake|ctest|git|gh|docker|npm|npx|"
    r"bash|sh|source|rg|find|make|trtmc|curl|wget|nvidia-smi|"
    r"ls|mkdir|touch|cp|mv|rm|chmod|sed|awk|jq|head|tail|cat|echo|"
    r"printf|test|timeout|tee|tar|export|unset|"
    r"\$[A-Za-z_][A-Za-z0-9_]*|\./[A-Za-z0-9_.\-/]+)"
    r"(?:\s|$)"
)
_LOCAL_PREFIXES = (
    ".github/",
    "cmake/",
    "examples/",
    "include/",
    "plugins/",
    "python/",
    "reports/",
    "scripts/",
    "src/",
    "tests/",
    "tools/",
    "website/",
)
_SHELL_SEPARATORS = {";", ";;", "&&", "||", "|", "&", "(", ")"}
_SHELL_PREFIX_WORDS = {
    "!",
    "do",
    "elif",
    "if",
    "then",
    "until",
    "while",
}
_SHELL_ONLY_WORDS = {
    "case",
    "done",
    "else",
    "esac",
    "fi",
    "for",
    "in",
    "select",
}
_VENDORED_FIXTURE_DOC_RE = re.compile(r"^tests/e2e/models/[^/]+/data/hf/[^/]+/README\.md$")


@dataclass(frozen=True)
class ShellBlock:
    path: Path
    line: int
    language: str
    body: str


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.message}"


def is_vendored_fixture_document(path: Path) -> bool:
    """Return whether ``path`` is an upstream model-card test fixture."""
    return _VENDORED_FIXTURE_DOC_RE.fullmatch(path.as_posix()) is not None


def _language(info: str) -> str:
    """Return a normalized first info-string token."""
    token = info.strip().split(maxsplit=1)[0] if info.strip() else ""
    return token.strip("{}").removeprefix(".").lower()


def extract_shell_blocks(path: Path, content: str) -> list[ShellBlock]:
    """Parse shell fences while respecting the opening fence length.

    Matching by delimiter length matters for Markdown examples that use four
    backticks around an inner three-backtick example.
    """
    blocks: list[ShellBlock] = []
    opener_char = ""
    opener_length = 0
    opener_line = 0
    language = ""
    body: list[str] = []

    for line_no, line in enumerate(content.splitlines(), start=1):
        if not opener_char:
            match = _OPEN_FENCE_RE.match(line)
            if not match:
                continue
            fence = match.group("fence")
            opener_char = fence[0]
            opener_length = len(fence)
            opener_line = line_no
            language = _language(match.group("info"))
            body = []
            continue

        stripped = line.lstrip()
        closing = re.match(
            rf"^{re.escape(opener_char)}{{{opener_length},}}[ \t]*$",
            stripped,
        )
        if closing:
            if language in SHELL_LANGUAGES:
                blocks.append(
                    ShellBlock(
                        path=path,
                        line=opener_line,
                        language=language,
                        body="\n".join(body) + "\n",
                    )
                )
            opener_char = ""
            opener_length = 0
            opener_line = 0
            language = ""
            body = []
            continue

        body.append(line)

    return blocks


def extract_inline_commands(path: Path, content: str) -> list[ShellBlock]:
    """Return command-like single-backtick spans outside fenced blocks."""
    commands: list[ShellBlock] = []
    opener_char = ""
    opener_length = 0

    for line_no, line in enumerate(content.splitlines(), start=1):
        if opener_char:
            stripped = line.lstrip()
            if re.match(
                rf"^{re.escape(opener_char)}{{{opener_length},}}[ \t]*$",
                stripped,
            ):
                opener_char = ""
                opener_length = 0
            continue

        fence_match = _OPEN_FENCE_RE.match(line)
        if fence_match:
            fence = fence_match.group("fence")
            opener_char = fence[0]
            opener_length = len(fence)
            continue

        for match in _INLINE_CODE_RE.finditer(line):
            body = match.group(1).strip()
            if _COMMAND_HEAD_RE.match(body):
                commands.append(
                    ShellBlock(
                        path=path,
                        line=line_no,
                        language="inline",
                        body=body + "\n",
                    )
                )
    return commands


def normalize_shell(body: str) -> str:
    """Make documentation placeholders parseable without changing behavior."""
    normalized = textwrap.dedent(body)
    normalized = _PLACEHOLDER_RE.sub("DOC_PLACEHOLDER", normalized)
    normalized = "\n".join(_PROMPT_RE.sub(r"\g<indent>", line) for line in normalized.splitlines())
    return normalized + ("\n" if normalized and not normalized.endswith("\n") else "")


def check_shell_syntax(block: ShellBlock) -> Finding | None:
    """Run the shell parser only; the command body is never executed."""
    result = subprocess.run(
        ["bash", "-n"],
        input=normalize_shell(block.body),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        return None
    detail = result.stderr.strip().splitlines()
    message = detail[-1] if detail else f"bash -n exited {result.returncode}"
    return Finding(block.path, block.line, f"invalid shell syntax: {message}")


def _logical_lines(body: str) -> Iterable[str]:
    normalized = normalize_shell(body).replace("\\\n", " ")
    for line in normalized.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            yield stripped


def _source_line(block: ShellBlock, offset: int) -> int:
    return block.line if block.language == "inline" else block.line + offset


def _tokenized_commands(line: str) -> list[list[str]]:
    """Split one shell line into simple commands without executing it.

    ``shlex.split`` returns one token list for an entire pipeline or ``&&``
    chain. Using shell punctuation lets the contract checks inspect every
    command in the chain while ``bash -n`` remains the syntax authority.
    """
    lexer = shlex.shlex(line, posix=True, punctuation_chars=";&|()")
    lexer.whitespace_split = True
    lexer.commenters = "#"
    try:
        stream = list(lexer)
    except ValueError:
        return []

    groups: list[list[str]] = []
    current: list[str] = []
    for token in stream:
        if token in _SHELL_SEPARATORS:
            if current:
                groups.append(current)
                current = []
            continue
        current.append(token)
    if current:
        groups.append(current)

    commands: list[list[str]] = []
    for tokens in groups:
        while tokens and tokens[0] in _SHELL_PREFIX_WORDS:
            tokens = tokens[1:]
        while tokens and tokens[-1] in {"do", "then"}:
            tokens = tokens[:-1]
        if not tokens or tokens[0] in _SHELL_ONLY_WORDS:
            continue
        commands.append(tokens)
    return commands


def _block_commands(block: ShellBlock) -> Iterable[tuple[int, list[str]]]:
    """Yield source-line offsets and tokens for every simple shell command."""
    for offset, line in enumerate(_logical_lines(block.body), start=1):
        for tokens in _tokenized_commands(line):
            yield offset, tokens


def _strip_shell_wrappers(tokens: list[str]) -> list[str]:
    """Drop leading assignments and the portable ``env`` wrapper."""
    remaining = list(tokens)
    if remaining and remaining[0] == "env":
        remaining.pop(0)
    while remaining and _ENV_ASSIGNMENT_RE.match(remaining[0]):
        remaining.pop(0)
    return remaining


def _clean_local_path(token: str) -> str | None:
    token = token.strip("'\"")
    if token.startswith("./"):
        token = token[2:]
    token = token.split("::", 1)[0]
    token = token.rstrip(".,;:)")
    if (
        not token
        or token == "."
        or token.startswith("/")
        or "$" in token
        or "*" in token
        or "?" in token
        or "[" in token
        or "DOC_PLACEHOLDER" in token
    ):
        return None
    if token.startswith(_LOCAL_PREFIXES):
        return token
    return None


def _candidate_input_paths(tokens: list[str]) -> list[str]:
    """Return repo-local inputs whose existence is required by the command."""
    tokens = _strip_shell_wrappers(tokens)
    if not tokens:
        return []

    command = tokens[0]
    command_name = Path(command).name
    candidates: list[str] = []

    if command.startswith("./"):
        candidates.append(command)

    if command_name in {"python", "python3", "bash", "sh"}:
        for token in tokens[1:]:
            if token.startswith("-"):
                continue
            local = _clean_local_path(token)
            if local:
                candidates.append(local)
            break

    if command_name == "find" and len(tokens) > 1:
        candidates.append(tokens[1])

    if command_name == "rg":
        # Search roots are repo inputs. Quoted patterns and option values do
        # not start with a known repository prefix and are ignored.
        candidates.extend(tokens[1:])

    is_pytest = command_name in {"pytest", "py.test"}
    if command_name in {"python", "python3"}:
        is_pytest = len(tokens) > 2 and tokens[1:3] == ["-m", "pytest"]
    if is_pytest:
        candidates.extend(tokens[1:])

    results: list[str] = []
    for candidate in candidates:
        local = _clean_local_path(candidate)
        if local and local not in results:
            results.append(local)
    return results


def check_local_inputs(block: ShellBlock, repo_root: Path) -> list[Finding]:
    """Check commands whose repo-local inputs must already exist."""
    findings: list[Finding] = []
    for offset, tokens in _block_commands(block):
        for local in _candidate_input_paths(tokens):
            if not (repo_root / local).exists():
                findings.append(
                    Finding(
                        block.path,
                        _source_line(block, offset),
                        f"command input does not exist: {local}",
                    )
                )
    return findings


def _python_cli_contract(
    repo_root: Path,
) -> tuple[set[str], dict[str, set[str]]]:
    source = repo_root / "python" / "tensorrt_model_connect" / "build_cli.py"
    if not source.is_file():
        return set(), {}
    tree = ast.parse(source.read_text(encoding="utf-8"))
    parser_variables: dict[str, str] = {}
    flags: dict[str, set[str]] = {}
    commands: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "add_parser"
            and node.value.args
            and isinstance(node.value.args[0], ast.Constant)
            and isinstance(node.value.args[0].value, str)
        ):
            command = node.value.args[0].value
            commands.add(command)
            parser_variables[node.targets[0].id] = command
            flags.setdefault(command, set())
            continue
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr == "add_parser" and node.args:
            argument = node.args[0]
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                commands.add(argument.value)
        if node.func.attr == "add_argument":
            owner = node.func.value
            if not isinstance(owner, ast.Name):
                continue
            command = parser_variables.get(owner.id)
            if command is None:
                continue
            for argument in node.args:
                if (
                    isinstance(argument, ast.Constant)
                    and isinstance(argument.value, str)
                    and argument.value.startswith("-")
                ):
                    flags.setdefault(command, set()).add(argument.value)
    return commands, flags


def _runtime_cli_contract(repo_root: Path) -> tuple[set[str], set[str]]:
    source = repo_root / "src" / "cli" / "args.cpp"
    if not source.is_file():
        return set(), set()
    content = source.read_text(encoding="utf-8")
    known_match = re.search(
        r"static const char\* known_cmds\[\]\s*=\s*\{(?P<body>.*?)nullptr\s*\};",
        content,
        re.DOTALL,
    )
    commands = {"build", "help", "version"}
    if known_match:
        commands.update(re.findall(r'"([a-z][a-z0-9-]+)"', known_match.group("body")))
    flags = set(re.findall(r'"(--?[A-Za-z][A-Za-z0-9-]*)"', content))
    flags.update({"-h", "--help", "-v", "--version"})
    return commands, flags


def _trtmc_tokens(tokens: list[str]) -> list[str] | None:
    tokens = _strip_shell_wrappers(tokens)
    if not tokens:
        return None
    executable = tokens[0]
    if executable in {"trtmc", "$TRTMC"} or Path(executable).name == "trtmc":
        return tokens
    return None


def check_trtmc_contract(block: ShellBlock, repo_root: Path) -> list[Finding]:
    """Validate documented native CLI subcommands and option spellings."""
    commands, runtime_flags = _runtime_cli_contract(repo_root)
    _python_commands, python_flags = _python_cli_contract(repo_root)
    if not commands:
        return []

    findings: list[Finding] = []
    for offset, tokens in _block_commands(block):
        invocation = _trtmc_tokens(tokens)
        if not invocation or len(invocation) < 2:
            continue
        command = invocation[1]
        if command.startswith("-"):
            continue
        if "DOC_PLACEHOLDER" in command:
            continue
        if command not in commands:
            findings.append(
                Finding(
                    block.path,
                    _source_line(block, offset),
                    f"unknown trtmc subcommand: {command}",
                )
            )
            continue

        allowed_flags = (
            python_flags.get("build", set()) | {"-h", "--help"}
            if command == "build"
            else runtime_flags
        )
        for token in invocation[2:]:
            option = token.split("=", 1)[0]
            if not option.startswith("-") or re.match(r"^-\d", option):
                continue
            if "DOC_PLACEHOLDER" in option or option.startswith("["):
                continue
            if option not in allowed_flags:
                findings.append(
                    Finding(
                        block.path,
                        _source_line(block, offset),
                        f"unknown option for `trtmc {command}`: {option}",
                    )
                )
    return findings


def check_python_module_contract(
    block: ShellBlock,
    repo_root: Path,
) -> list[Finding]:
    """Validate ``python -m tensorrt_model_connect`` examples."""
    commands, flags_by_command = _python_cli_contract(repo_root)
    if not commands:
        return []

    findings: list[Finding] = []
    for offset, raw_tokens in _block_commands(block):
        tokens = _strip_shell_wrappers(raw_tokens)
        if len(tokens) < 4:
            continue
        if Path(tokens[0]).name not in {"python", "python3"}:
            continue
        if tokens[1:3] != ["-m", "tensorrt_model_connect"]:
            continue
        command = tokens[3]
        if command.startswith("-") or "DOC_PLACEHOLDER" in command:
            continue
        if command not in commands:
            findings.append(
                Finding(
                    block.path,
                    _source_line(block, offset),
                    f"unknown `python -m tensorrt_model_connect` subcommand: {command}",
                )
            )
            continue
        allowed_flags = flags_by_command.get(command, set()) | {"-h", "--help"}
        for token in tokens[4:]:
            option = token.split("=", 1)[0]
            if not option.startswith("-") or re.match(r"^-\d", option):
                continue
            if "DOC_PLACEHOLDER" in option or option.startswith("["):
                continue
            if option not in allowed_flags:
                findings.append(
                    Finding(
                        block.path,
                        _source_line(block, offset),
                        f"unknown option for Python `{command}` command: {option}",
                    )
                )
    return findings


def _literal_choices(node: ast.AST) -> set[str]:
    if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return set()
    values: set[str] = set()
    for element in node.elts:
        if isinstance(element, ast.Constant):
            values.add(str(element.value))
    return values


def _argparse_contract(
    script_path: Path,
) -> tuple[set[str], dict[str, set[str]]] | None:
    """Return argparse flags and literal choices, or ``None`` if not argparse."""
    if not script_path.is_file():
        return None
    try:
        tree = ast.parse(script_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None

    uses_argparse = False
    flags: set[str] = set()
    choices: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "ArgumentParser":
                uses_argparse = True
            if node.func.attr == "add_argument":
                option_names: list[str] = []
                for argument in node.args:
                    if (
                        isinstance(argument, ast.Constant)
                        and isinstance(argument.value, str)
                        and argument.value.startswith("-")
                    ):
                        flags.add(argument.value)
                        option_names.append(argument.value)
                literal_choices: set[str] = set()
                for keyword in node.keywords:
                    if keyword.arg == "choices":
                        literal_choices = _literal_choices(keyword.value)
                        break
                if literal_choices:
                    for option_name in option_names:
                        choices[option_name] = literal_choices
        elif isinstance(node.func, ast.Name) and node.func.id == "ArgumentParser":
            uses_argparse = True
    return (flags, choices) if uses_argparse else None


def _check_argparse_options(
    block: ShellBlock,
    offset: int,
    tokens: list[str],
    local: str,
    contract: tuple[set[str], dict[str, set[str]]],
) -> list[Finding]:
    """Validate option spellings and statically declared literal choices."""
    flags, choices = contract
    allowed_flags = flags | {"-h", "--help"}
    findings: list[Finding] = []
    for index, token in enumerate(tokens):
        option, separator, inline_value = token.partition("=")
        if not option.startswith("-") or re.match(r"^-\d", option):
            continue
        if "DOC_PLACEHOLDER" in option or option.startswith("["):
            continue
        if option not in allowed_flags:
            findings.append(
                Finding(
                    block.path,
                    _source_line(block, offset),
                    f"unknown option for `{local}`: {option}",
                )
            )
            continue
        allowed_values = choices.get(option)
        if not allowed_values:
            continue
        value = (
            inline_value if separator else (tokens[index + 1] if index + 1 < len(tokens) else "")
        )
        if (
            value
            and not value.startswith("-")
            and "DOC_PLACEHOLDER" not in value
            and value not in allowed_values
        ):
            findings.append(
                Finding(
                    block.path,
                    _source_line(block, offset),
                    f"invalid value for `{local} {option}`: {value}; "
                    f"expected one of {', '.join(sorted(allowed_values))}",
                )
            )
    return findings


def check_python_script_contract(
    block: ShellBlock,
    repo_root: Path,
) -> list[Finding]:
    """Validate flags for directly invoked, local argparse scripts."""
    findings: list[Finding] = []
    for offset, raw_tokens in _block_commands(block):
        tokens = _strip_shell_wrappers(raw_tokens)
        if len(tokens) < 2 or Path(tokens[0]).name not in {"python", "python3"}:
            continue
        local = _clean_local_path(tokens[1])
        if not local or not local.endswith(".py"):
            continue
        contract = _argparse_contract(repo_root / local)
        if contract is None:
            continue
        findings.extend(
            _check_argparse_options(
                block,
                offset,
                tokens[2:],
                local,
                contract,
            )
        )
    return findings


def check_direct_script_contract(
    block: ShellBlock,
    repo_root: Path,
) -> list[Finding]:
    """Validate flags for executable repo-local Python wrappers."""
    findings: list[Finding] = []
    for offset, raw_tokens in _block_commands(block):
        tokens = _strip_shell_wrappers(raw_tokens)
        if not tokens or not tokens[0].startswith("./"):
            continue
        local = _clean_local_path(tokens[0])
        if not local:
            continue
        contract = _argparse_contract(repo_root / local)
        if contract is None:
            continue
        findings.extend(
            _check_argparse_options(
                block,
                offset,
                tokens[1:],
                local,
                contract,
            )
        )
    return findings


def check_ctest_contract(block: ShellBlock) -> list[Finding]:
    """Prevent filtered CTest examples from succeeding with zero matches."""
    if block.language == "inline":
        return []

    findings: list[Finding] = []
    for offset, raw_tokens in _block_commands(block):
        tokens = _strip_shell_wrappers(raw_tokens)
        if not tokens or Path(tokens[0]).name != "ctest":
            continue
        uses_filter = "-R" in tokens or "--tests-regex" in tokens
        rejects_empty_selection = any(token == "--no-tests=error" for token in tokens)
        if uses_filter and not rejects_empty_selection:
            findings.append(
                Finding(
                    block.path,
                    _source_line(block, offset),
                    "filtered ctest command must include --no-tests=error",
                )
            )
    return findings


def _fallback_markdown_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".md", ".mdx"}
        and not any(part in SKIPPED_DIRS for part in path.parts)
    )


def tracked_markdown_files(repo_root: Path) -> list[Path]:
    """Return Git-tracked Markdown, with a source-archive fallback."""
    result = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "-z", "--", "*.md", "*.mdx"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return _fallback_markdown_files(repo_root)
    return sorted(repo_root / raw.decode("utf-8") for raw in result.stdout.split(b"\0") if raw)


def selected_markdown_files(repo_root: Path, selections: Sequence[str]) -> list[Path]:
    if not selections:
        return tracked_markdown_files(repo_root)

    files: set[Path] = set()
    for selection in selections:
        path = (repo_root / selection).resolve()
        if path.is_file() and path.suffix.lower() in {".md", ".mdx"}:
            files.add(path)
        elif path.is_dir():
            files.update(_fallback_markdown_files(path))
        else:
            raise FileNotFoundError(f"documentation path does not exist: {selection}")
    return sorted(files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        help="Markdown files/directories (default: all Git-tracked Markdown)",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository root (default: parent of this script's directory)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = (
        Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parent.parent
    )
    try:
        markdown_files = selected_markdown_files(repo_root, args.paths)
    except FileNotFoundError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    findings: list[Finding] = []
    fenced_blocks: list[ShellBlock] = []
    inline_commands: list[ShellBlock] = []
    vendored_fixture_docs = 0
    for path in markdown_files:
        if not path.is_file():
            findings.append(
                Finding(
                    path.relative_to(repo_root),
                    1,
                    "Git-tracked Markdown file is missing from the worktree",
                )
            )
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        rel_path = path.relative_to(repo_root)
        if is_vendored_fixture_document(rel_path):
            vendored_fixture_docs += 1
            continue
        document_fences = extract_shell_blocks(rel_path, content)
        document_inline = extract_inline_commands(rel_path, content)
        fenced_blocks.extend(document_fences)
        inline_commands.extend(document_inline)
        for block in [*document_fences, *document_inline]:
            syntax_finding = check_shell_syntax(block)
            if syntax_finding:
                findings.append(syntax_finding)
            findings.extend(check_local_inputs(block, repo_root))
            findings.extend(check_trtmc_contract(block, repo_root))
            findings.extend(check_python_module_contract(block, repo_root))
            findings.extend(check_python_script_contract(block, repo_root))
            findings.extend(check_direct_script_contract(block, repo_root))
            findings.extend(check_ctest_contract(block))

    all_commands = [*fenced_blocks, *inline_commands]
    unique_bodies = {normalize_shell(block.body).strip() for block in all_commands}
    print(f"Project Markdown documents checked: {len(markdown_files) - vendored_fixture_docs}")
    print(f"Vendored fixture documents excluded: {vendored_fixture_docs}")
    print(f"Shell fenced blocks: {len(fenced_blocks)}")
    print(f"Inline commands: {len(inline_commands)}")
    print(f"Unique command bodies: {len(unique_bodies)}")
    print(f"Findings: {len(findings)}")
    for finding in findings:
        print(f"  [ERROR] {finding}")

    if findings:
        print("\nFAILED: documentation shell examples are not self-consistent.")
        return 1
    print("\nAll documentation shell examples passed syntax and local-input checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
