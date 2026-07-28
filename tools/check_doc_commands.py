#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate shell examples in Git-tracked Markdown documentation.

The checker is deliberately non-executing: it parses every ``bash``, ``sh``,
or ``shell`` fenced block with ``bash -n``; verifies repository-local scripts,
search/test inputs, and explicit positional inputs; and checks statically
discoverable CLI subcommands, option scope/arity, choices, and required inputs.
Literal ``bash``/``sh -c`` payloads, including ``docker exec`` wrappers, are
checked recursively with bounded depth.
Commands that need a GPU, model download, credentials, services, or generated
artifacts remain runtime validation work.

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
from dataclasses import dataclass, field
from functools import lru_cache
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
_FENCE_OPEN_RE = re.compile(r"^(?P<indent> {0,3})(?P<fence>`{3,}|~{3,})[ \t]*(?P<info>.*)$")
_BLOCK_QUOTE_RE = re.compile(r"^ {0,3}>[ \t]?")
_LIST_MARKER_RE = re.compile(
    r"^(?P<indent> {0,3})(?P<marker>[-+*]|\d{1,9}[.)])"
    r"(?P<spacing>[ \t]{1,4})"
)
_PLACEHOLDER_RE = re.compile(r"<[^>\n]+>|(?<!\$)\{[A-Za-z][A-Za-z0-9_.-]*\}")
_PROMPT_RE = re.compile(r"^(?P<indent>[ \t]*)\$\s+")
_ENV_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_BACKTICK_RUN_RE = re.compile(r"`+")
_ARGPARSE_NEGATIVE_NUMBER_RE = re.compile(r"(?:-\d+|-\d*\.\d+)\Z")
_INLINE_COMMAND_NAMES = frozenset(
    {
        "awk",
        "bash",
        "cat",
        "chmod",
        "cmake",
        "cp",
        "ctest",
        "curl",
        "docker",
        "echo",
        "export",
        "find",
        "gh",
        "git",
        "head",
        "jq",
        "ls",
        "make",
        "mkdir",
        "mv",
        "npm",
        "npx",
        "nvidia-smi",
        "pip",
        "pip3",
        "printf",
        "py.test",
        "pytest",
        "python",
        "python3",
        "rg",
        "rm",
        "sed",
        "sh",
        "source",
        "tail",
        "tar",
        "tee",
        "test",
        "timeout",
        "touch",
        "trtmc",
        "unset",
        "wget",
    }
)
_SHELL_VARIABLE_COMMAND_RE = re.compile(r"^\$[A-Za-z_][A-Za-z0-9_]*$")
_RELATIVE_COMMAND_RE = re.compile(r"^\./[A-Za-z0-9_.\-/]+$")
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
    "{",
    "do",
    "elif",
    "if",
    "then",
    "until",
    "while",
}
_SHELL_ONLY_WORDS = {
    "}",
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
_NESTED_SHELL_LANGUAGE = "_nested-shell"
_MAX_NESTED_SHELL_DEPTH = 4


@dataclass(frozen=True)
class ShellBlock:
    path: Path
    line: int
    language: str
    body: str


@dataclass(frozen=True)
class MarkdownFenceBlock:
    """One CommonMark fenced block with container prefixes removed."""

    line: int
    end_line: int
    language: str
    body: str


@dataclass(frozen=True)
class _MarkdownContainer:
    kind: str
    indent: int = 0


@dataclass(frozen=True)
class _ShellCommand:
    tokens: tuple[str, ...]
    input_redirections: tuple[str, ...] = ()


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.message}"


@dataclass(frozen=True)
class OptionSpec:
    """Static command-line option contract."""

    aliases: tuple[str, ...]
    min_values: int = 1
    max_values: int | None = 1
    choices: frozenset[str] = frozenset()
    required: bool = False
    allow_inline_value: bool = True
    consume_option_like_value: bool = False
    output_path: bool = False


@dataclass(frozen=True)
class PositionalSpec:
    """Static positional-argument contract."""

    name: str
    min_values: int = 1
    max_values: int | None = 1
    choices: frozenset[str] = frozenset()


@dataclass
class CommandSpec:
    """Arguments accepted by one command or subcommand."""

    options: dict[str, OptionSpec]
    positionals: list[PositionalSpec]
    required_any: list[tuple[frozenset[str], str]]

    @classmethod
    def empty(cls) -> CommandSpec:
        return cls(options={}, positionals=[], required_any=[])


@dataclass
class ProgramSpec:
    """Root and subcommand contracts extracted from one CLI implementation."""

    root: CommandSpec
    commands: dict[str, CommandSpec]
    command_required: bool = False
    nested: dict[str, ProgramSpec] = field(default_factory=dict)
    allow_extras: bool = False
    uncertain_reason: str | None = None


def is_vendored_fixture_document(path: Path) -> bool:
    """Return whether ``path`` is an upstream model-card test fixture."""
    return _VENDORED_FIXTURE_DOC_RE.fullmatch(path.as_posix()) is not None


def _language(info: str) -> str:
    """Return a normalized first info-string token."""
    token = info.strip().split(maxsplit=1)[0] if info.strip() else ""
    return token.strip("{}").removeprefix(".").lower()


def _fence_opener(
    line: str,
) -> tuple[re.Match[str], tuple[_MarkdownContainer, ...]] | None:
    """Return a CommonMark fence opener after blockquote/list containers."""
    rest = line
    containers: list[_MarkdownContainer] = []
    while True:
        quote = _BLOCK_QUOTE_RE.match(rest)
        if quote is not None:
            containers.append(_MarkdownContainer("quote"))
            rest = rest[quote.end() :]
            continue
        marker = _LIST_MARKER_RE.match(rest)
        if marker is not None:
            containers.append(
                _MarkdownContainer(
                    "list",
                    len(marker.group("indent"))
                    + len(marker.group("marker"))
                    + len(marker.group("spacing")),
                )
            )
            rest = rest[marker.end() :]
            continue
        break
    match = _FENCE_OPEN_RE.match(rest)
    if match is None:
        return None
    return match, tuple(containers)


def _quote_containers(line: str) -> tuple[str, tuple[_MarkdownContainer, ...]]:
    """Strip leading blockquote markers and return their container shape."""
    rest = line
    containers: list[_MarkdownContainer] = []
    while (quote := _BLOCK_QUOTE_RE.match(rest)) is not None:
        containers.append(_MarkdownContainer("quote"))
        rest = rest[quote.end() :]
    return rest, tuple(containers)


def _list_continuation_fence_opener(
    lines: Sequence[str],
    index: int,
) -> tuple[re.Match[str], tuple[_MarkdownContainer, ...]] | None:
    """Find a fence indented as content of a preceding CommonMark list item."""
    rest, quote_containers = _quote_containers(lines[index])
    leading = len(rest) - len(rest.lstrip(" "))
    if leading < 2:
        return None

    for previous_index in range(index - 1, -1, -1):
        previous, previous_quotes = _quote_containers(lines[previous_index])
        if not previous.strip():
            continue
        if previous_quotes != quote_containers:
            break
        marker = _LIST_MARKER_RE.match(previous)
        if marker is None and _FENCE_OPEN_RE.match(previous) is not None:
            break
        if marker is not None:
            if _FENCE_OPEN_RE.match(previous[marker.end() :]) is not None:
                break
            content_indent = (
                len(marker.group("indent"))
                + len(marker.group("marker"))
                + len(marker.group("spacing"))
            )
            if leading >= content_indent:
                opener = _FENCE_OPEN_RE.match(rest[content_indent:])
                if opener is not None:
                    return opener, (
                        *quote_containers,
                        _MarkdownContainer("list", content_indent),
                    )
        previous_leading = len(previous) - len(previous.lstrip(" "))
        if previous_leading == 0 and marker is None:
            break
    return None


def _strip_fence_containers(
    line: str,
    containers: Sequence[_MarkdownContainer],
) -> str | None:
    """Remove the containers that own a fenced block from one physical line."""
    rest = line
    for container in containers:
        if not rest.strip():
            return ""
        if container.kind == "quote":
            match = _BLOCK_QUOTE_RE.match(rest)
            if match is None:
                return None
            rest = rest[match.end() :]
            continue
        leading = len(rest) - len(rest.lstrip(" "))
        if leading < container.indent:
            return None
        rest = rest[container.indent :]
    return rest


def markdown_fence_blocks(content: str) -> list[MarkdownFenceBlock]:
    """Parse CommonMark-style fenced blocks, including list/quote containers."""
    lines = content.splitlines()
    blocks: list[MarkdownFenceBlock] = []
    index = 0
    while index < len(lines):
        parsed = _fence_opener(lines[index])
        if parsed is None:
            parsed = _list_continuation_fence_opener(lines, index)
        if parsed is None:
            index += 1
            continue
        opener, containers = parsed
        fence = opener.group("fence")
        opener_char = fence[0]
        opener_length = len(fence)
        opener_indent = len(opener.group("indent"))
        start_line = index + 1
        language = _language(opener.group("info"))
        body: list[str] = []
        cursor = index + 1
        end_line = len(lines)
        while cursor < len(lines):
            normalized = _strip_fence_containers(lines[cursor], containers)
            if normalized is None:
                end_line = cursor
                break
            closing = re.match(
                rf"^ {{0,3}}{re.escape(opener_char)}"
                rf"{{{opener_length},}}[ \t]*$",
                normalized,
            )
            if closing is not None:
                end_line = cursor + 1
                cursor += 1
                break
            removable = min(
                opener_indent,
                len(normalized) - len(normalized.lstrip(" ")),
            )
            normalized = normalized[removable:]
            body.append(normalized)
            cursor += 1
        blocks.append(
            MarkdownFenceBlock(
                line=start_line,
                end_line=end_line,
                language=language,
                body="\n".join(body) + "\n",
            )
        )
        index = max(cursor, index + 1)
    return blocks


def extract_shell_blocks(path: Path, content: str) -> list[ShellBlock]:
    """Parse shell fences while respecting the opening fence length.

    Matching by delimiter length matters for Markdown examples that use four
    backticks around an inner three-backtick example.
    """
    return [
        ShellBlock(path=path, line=block.line, language=block.language, body=block.body)
        for block in markdown_fence_blocks(content)
        if block.language in SHELL_LANGUAGES
    ]


def _inline_code_spans(content: str) -> Iterable[tuple[int, str]]:
    """Yield opening offsets and CommonMark code spans for each run length."""
    runs = list(_BACKTICK_RUN_RE.finditer(content))
    index = 0
    while index < len(runs):
        opener = runs[index]
        closer_index = index + 1
        while closer_index < len(runs):
            closer = runs[closer_index]
            if len(closer.group(0)) == len(opener.group(0)):
                yield opener.start(), content[opener.end() : closer.start()]
                index = closer_index + 1
                break
            closer_index += 1
        else:
            index += 1


def extract_inline_commands(path: Path, content: str) -> list[ShellBlock]:
    """Return command-like inline code spans outside fenced blocks."""
    commands: list[ShellBlock] = []
    chunk_start_line = 0
    chunk_lines: list[str] = []
    fenced_lines = {
        line
        for block in markdown_fence_blocks(content)
        for line in range(block.line, block.end_line + 1)
    }

    def append_chunk_commands() -> None:
        if not chunk_lines:
            return
        chunk = "\n".join(chunk_lines)
        for opening_offset, span in _inline_code_spans(chunk):
            body = span.replace("\n", " ").strip()
            if _is_inline_command(body):
                commands.append(
                    ShellBlock(
                        path=path,
                        line=chunk_start_line + chunk.count("\n", 0, opening_offset),
                        language="inline",
                        body=body + "\n",
                    )
                )

    for line_no, line in enumerate(content.splitlines(), start=1):
        if line_no in fenced_lines:
            append_chunk_commands()
            chunk_lines = []
            chunk_start_line = 0
            continue

        if not line.strip():
            append_chunk_commands()
            chunk_lines = []
            chunk_start_line = 0
            continue
        if not chunk_lines:
            chunk_start_line = line_no
        chunk_lines.append(line)

    append_chunk_commands()
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


def _logical_lines(body: str) -> Iterable[tuple[int, str]]:
    """Yield physical start offsets and continuation-joined shell lines."""
    pending = ""
    start_offset = 0
    for offset, line in enumerate(normalize_shell(body).splitlines(), start=1):
        if not pending:
            start_offset = offset
        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue
        stripped = (pending + line).strip()
        pending = ""
        if stripped and not stripped.startswith("#"):
            yield start_offset, stripped
    if pending:
        stripped = pending.strip()
        if stripped and not stripped.startswith("#"):
            yield start_offset, stripped


def _source_line(block: ShellBlock, offset: int) -> int:
    if block.language == "inline":
        return block.line
    if block.language == _NESTED_SHELL_LANGUAGE:
        return block.line + offset - 1
    return block.line + offset


_OUTPUT_REDIRECTS = {">", ">>", ">|", "&>", "&>>", ">&"}
_INPUT_REDIRECTS = {"<", "<>"}
_NON_FILE_INPUT_REDIRECTS = {"<<", "<<<"}
_PROCESS_SUBSTITUTION_OPENERS = {"<(", ">("}
_PROCESS_SUBSTITUTION_PLACEHOLDER = "DOC_PROCESS_SUBSTITUTION"
_SHELL_PUNCTUATION_OPERATORS = (
    "<(",
    ">(",
    "<<<",
    "&>>",
    ";;&",
    "&&",
    "||",
    ";;",
    ";&",
    "|&",
    "<<",
    ">>",
    ">|",
    "&>",
    ">&",
    "<&",
    "<>",
    ";",
    "|",
    "&",
    "(",
    ")",
    "<",
    ">",
)


def _strip_redirections(tokens: Sequence[str]) -> _ShellCommand:
    """Remove shell redirections while retaining file-backed input paths."""
    cleaned: list[str] = []
    inputs: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        operator_index = index
        if (
            token.isdigit()
            and index + 1 < len(tokens)
            and tokens[index + 1] in _OUTPUT_REDIRECTS | _INPUT_REDIRECTS
        ):
            operator_index = index + 1
            token = tokens[operator_index]
        if token in _OUTPUT_REDIRECTS | _INPUT_REDIRECTS | _NON_FILE_INPUT_REDIRECTS:
            destination_index = operator_index + 1
            if destination_index < len(tokens):
                destination = tokens[destination_index]
                if token in _INPUT_REDIRECTS:
                    inputs.append(destination)
                index = destination_index + 1
            else:
                index = destination_index
            continue
        cleaned.append(token)
        index += 1
    return _ShellCommand(tuple(cleaned), tuple(inputs))


def _normalize_shell_punctuation(tokens: Sequence[str]) -> list[str]:
    """Split compound shlex punctuation runs into shell operator tokens."""
    normalized: list[str] = []
    punctuation = set(";&|()<>")
    for token in tokens:
        if not token or any(character not in punctuation for character in token):
            normalized.append(token)
            continue
        remainder = token
        while remainder:
            operator = next(
                (
                    candidate
                    for candidate in _SHELL_PUNCTUATION_OPERATORS
                    if remainder.startswith(candidate)
                ),
                remainder[0],
            )
            normalized.append(operator)
            remainder = remainder[len(operator) :]
    return normalized


def _shell_command_groups(
    stream: Sequence[str],
    start: int = 0,
    *,
    stop_at_close: bool = False,
) -> tuple[list[list[str]], int]:
    """Split simple commands while recursively exposing process substitutions."""
    groups: list[list[str]] = []
    nested_groups: list[list[str]] = []
    current: list[str] = []
    index = start
    while index < len(stream):
        token = stream[index]
        if token in _PROCESS_SUBSTITUTION_OPENERS:
            inner, index = _shell_command_groups(stream, index + 1, stop_at_close=True)
            nested_groups.extend(inner)
            current.append(_PROCESS_SUBSTITUTION_PLACEHOLDER)
            continue
        if token == ")" and stop_at_close:
            if current:
                groups.append(current)
            return [*groups, *nested_groups], index + 1
        if token in _SHELL_SEPARATORS:
            if current:
                groups.append(current)
                current = []
            index += 1
            continue
        current.append(token)
        index += 1
    if current:
        groups.append(current)
    return [*groups, *nested_groups], index


def _shell_commands(line: str) -> list[_ShellCommand]:
    """Split one shell line into simple commands without executing it.

    ``shlex.split`` returns one token list for an entire pipeline or ``&&``
    chain. Using shell punctuation lets the contract checks inspect every
    command in the chain while ``bash -n`` remains the syntax authority.
    """
    lexer = shlex.shlex(line, posix=True, punctuation_chars=";&|()<>")
    lexer.whitespace_split = True
    lexer.commenters = "#"
    try:
        stream = _normalize_shell_punctuation(list(lexer))
    except ValueError:
        return []

    groups, _index = _shell_command_groups(stream)

    commands: list[_ShellCommand] = []
    for tokens in groups:
        while tokens and tokens[0] in _SHELL_PREFIX_WORDS:
            tokens = tokens[1:]
        while tokens and tokens[-1] in {"do", "then"}:
            tokens = tokens[:-1]
        if not tokens or tokens[0] in _SHELL_ONLY_WORDS:
            continue
        command = _strip_redirections(tokens)
        if command.tokens or command.input_redirections:
            commands.append(command)
    return commands


def _tokenized_commands(line: str) -> list[list[str]]:
    """Compatibility wrapper returning redirection-free command tokens."""
    return [list(command.tokens) for command in _shell_commands(line)]


def _block_shell_commands(block: ShellBlock) -> Iterable[tuple[int, _ShellCommand]]:
    for offset, line in _logical_lines(block.body):
        for command in _shell_commands(line):
            yield offset, command


def _block_commands(block: ShellBlock) -> Iterable[tuple[int, list[str]]]:
    """Yield source-line offsets and tokens for every simple shell command."""
    for offset, command in _block_shell_commands(block):
        yield offset, list(command.tokens)


def _is_static_env_operand(token: str) -> bool:
    """Return whether an ``env`` option operand is statically knowable."""
    return bool(token) and not any(
        marker in token
        for marker in ("$", "`", "\x00", "DOC_PLACEHOLDER")
    )


def _strip_command_wrapper(tokens: Sequence[str]) -> list[str] | None:
    """Return the executable following a statically understood ``command``."""
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            return list(tokens[index + 1 :])
        if token == "-p":
            index += 1
            continue
        if token in {"-v", "-V"}:
            return []
        if token.startswith("-") and not token.startswith("--"):
            options = token[1:]
            if options and set(options) <= {"p"}:
                index += 1
                continue
            if options and set(options) <= {"p", "v", "V"}:
                return []
            return None
        if token.startswith("-"):
            return None
        return list(tokens[index:])
    return []


_TIME_FLAG_OPTIONS = {
    "-a",
    "--append",
    "-p",
    "--portability",
    "-q",
    "--quiet",
    "-v",
    "--verbose",
}
_TIME_VALUE_OPTIONS = {
    "-f",
    "--format",
    "-o",
    "--output",
}


def _strip_time_wrapper(tokens: Sequence[str]) -> list[str] | None:
    """Return the executable following statically understood shell/GNU ``time``."""
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            return list(tokens[index + 1 :])
        if token in {"--help", "--version", "-V"}:
            return []
        if token in _TIME_FLAG_OPTIONS:
            index += 1
            continue
        option, separator, value = token.partition("=")
        if option in _TIME_VALUE_OPTIONS and separator:
            if not value:
                return None
            index += 1
            continue
        if token in _TIME_VALUE_OPTIONS:
            if index + 1 >= len(tokens):
                return None
            index += 2
            continue
        if token.startswith(("-f", "-o")) and len(token) > 2:
            index += 1
            continue
        if token.startswith("-") and not token.startswith("--"):
            options = token[1:]
            if options and set(options) <= {"a", "p", "q", "v"}:
                index += 1
                continue
            return None
        if token.startswith("-"):
            return None
        return list(tokens[index:])
    return []


def _strip_shell_wrappers(
    tokens: list[str],
    *,
    uncertainty: list[str] | None = None,
) -> list[str]:
    """Drop understood assignments plus ``env``, ``command``, and ``time``.

    Unknown or dynamic wrapper options return no command so callers fail closed
    instead of treating an option operand as an executable.
    """
    remaining = list(tokens)
    while True:
        while remaining and _ENV_ASSIGNMENT_RE.match(remaining[0]):
            remaining.pop(0)
        if not remaining:
            return remaining
        if remaining[0] in {"command", "/usr/bin/command"}:
            stripped = _strip_command_wrapper(remaining)
            if stripped is None:
                if uncertainty is not None:
                    uncertainty.append("unsupported `command` wrapper options")
                return []
            remaining = stripped
            continue
        if remaining[0] in {"time", "/bin/time", "/usr/bin/time"}:
            stripped = _strip_time_wrapper(remaining)
            if stripped is None:
                if uncertainty is not None:
                    uncertainty.append("unsupported `time` wrapper options")
                return []
            remaining = stripped
            continue
        if remaining[0] not in {"env", "/usr/bin/env"}:
            return remaining

        index = 1
        while index < len(remaining):
            token = remaining[index]
            if token == "--":
                index += 1
                break
            if token in {"-i", "--ignore-environment", "-"}:
                index += 1
                continue
            if token in {"-u", "--unset"}:
                if (
                    index + 1 >= len(remaining)
                    or _ENV_NAME_RE.fullmatch(remaining[index + 1]) is None
                ):
                    return []
                index += 2
                continue
            if token.startswith("-u") and token != "-u":
                if _ENV_NAME_RE.fullmatch(token[2:]) is None:
                    return []
                index += 1
                continue
            if token.startswith("--unset="):
                if _ENV_NAME_RE.fullmatch(token.partition("=")[2]) is None:
                    return []
                index += 1
                continue
            if token in {"-C", "--chdir"}:
                if (
                    index + 1 >= len(remaining)
                    or not _is_static_env_operand(remaining[index + 1])
                ):
                    return []
                index += 2
                continue
            if token.startswith("-C") and token != "-C":
                if not _is_static_env_operand(token[2:]):
                    return []
                index += 1
                continue
            if token.startswith("--chdir="):
                if not _is_static_env_operand(token.partition("=")[2]):
                    return []
                index += 1
                continue
            if token.startswith("-"):
                return []
            break

        remaining = remaining[index:]


def _is_inline_command(body: str) -> bool:
    """Return whether an inline code span begins with a supported command.

    Reusing the shell tokenizer and wrapper parser keeps inline discovery in
    sync with fenced-command validation. In particular, quoted or empty
    assignments and repeated GNU ``env`` wrappers cannot be represented
    reliably by a coarse regular expression.
    """
    commands = _shell_commands(body)
    if not commands:
        return False
    tokens = list(commands[0].tokens)
    if tokens and tokens[0] == "$":
        tokens.pop(0)
    tokens = _strip_shell_wrappers(tokens)
    if not tokens:
        return False
    command = tokens[0]
    return (
        command in _INLINE_COMMAND_NAMES
        or _RELATIVE_COMMAND_RE.fullmatch(command) is not None
        or _SHELL_VARIABLE_COMMAND_RE.fullmatch(command) is not None
    )


def _shell_c_payload(tokens: Sequence[str]) -> str | None:
    """Return the command string from a literal ``bash``/``sh -c`` invocation."""
    if not tokens or Path(tokens[0]).name not in {"bash", "sh"}:
        return None

    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            return None
        if token == "-c" or (
            token.startswith("-") and not token.startswith("--") and "c" in token.removeprefix("-")
        ):
            return tokens[index + 1] if index + 1 < len(tokens) else None
        if token in {"-o", "-O"}:
            index += 2
            continue
        if token in {"--init-file", "--rcfile"}:
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        return None
    return None


_DOCKER_EXEC_VALUE_OPTIONS = {
    "--detach-keys",
    "--env",
    "--env-file",
    "--user",
    "--workdir",
    "-e",
    "-u",
    "-w",
}


def _docker_exec_command(tokens: Sequence[str]) -> list[str] | None:
    """Return ``docker exec``'s command tokens after options and container."""
    try:
        index = tokens.index("exec", 1) + 1
    except ValueError:
        return None

    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            index += 1
            break
        option, separator, _value = token.partition("=")
        if option in _DOCKER_EXEC_VALUE_OPTIONS:
            index += 1 if separator or len(option) == 2 and token != option else 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        break

    # The first non-option is the container; the remainder is its command.
    return list(tokens[index + 1 :]) if index + 1 < len(tokens) else None


def _nested_shell_payload(tokens: Sequence[str]) -> str | None:
    """Extract a statically knowable shell payload from supported wrappers."""
    remaining = _strip_shell_wrappers(list(tokens))
    if not remaining:
        return None

    payload = _shell_c_payload(remaining)
    if payload is None and Path(remaining[0]).name == "docker":
        docker_command = _docker_exec_command(remaining)
        payload = (
            _shell_c_payload(_strip_shell_wrappers(docker_command))
            if docker_command is not None
            else None
        )

    if (
        payload is None
        or not payload.strip()
        or "\x00" in payload
        or "$" in payload
        or "`" in payload
        or "DOC_PLACEHOLDER" in payload
    ):
        return None
    return payload


def shell_validation_blocks(
    block: ShellBlock,
    *,
    max_nested_depth: int = _MAX_NESTED_SHELL_DEPTH,
) -> Iterable[ShellBlock]:
    """Yield ``block`` and literal nested ``bash``/``sh -c`` payloads.

    Payloads are parsed but never executed. Dynamic strings are skipped because
    their runtime command cannot be reconstructed safely. The depth and
    ancestry guards keep adversarial wrapper chains bounded.
    """
    if max_nested_depth < 0:
        raise ValueError("max_nested_depth must be non-negative")

    def visit(
        current: ShellBlock,
        depth: int,
        ancestors: frozenset[str],
    ) -> Iterable[ShellBlock]:
        yield current
        if depth >= max_nested_depth:
            return
        for offset, command in _block_shell_commands(current):
            payload = _nested_shell_payload(command.tokens)
            if payload is None:
                continue
            body = payload + ("" if payload.endswith("\n") else "\n")
            signature = normalize_shell(body).strip()
            if not signature or signature in ancestors:
                continue
            nested = ShellBlock(
                path=current.path,
                line=_source_line(current, offset),
                language=_NESTED_SHELL_LANGUAGE,
                body=body,
            )
            yield from visit(
                nested,
                depth + 1,
                ancestors | {signature},
            )

    root_signature = normalize_shell(block.body).strip()
    yield from visit(block, 0, frozenset({root_signature}))


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
        or "{" in token
        or "}" in token
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

    if command_name == "cat":
        candidates.extend(tokens[1:])

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
    for offset, command in _block_shell_commands(block):
        candidates = [
            *_candidate_input_paths(list(command.tokens)),
            *command.input_redirections,
        ]
        for candidate in candidates:
            local = _clean_local_path(candidate)
            if local is None:
                continue
            if not (repo_root / local).exists():
                findings.append(
                    Finding(
                        block.path,
                        _source_line(block, offset),
                        f"command input does not exist: {local}",
                    )
                )
    return findings


def _constant(node: ast.AST | None) -> object | None:
    if isinstance(node, ast.Constant):
        return node.value
    return None


def _call_keyword(call: ast.Call, name: str) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


_UNRESOLVED = object()


def _literal_value(
    node: ast.AST | None,
    binding: dict[str, object] | None = None,
) -> object:
    """Evaluate the literal subset used by static argparse declarations."""
    if node is None:
        return _UNRESOLVED
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and binding is not None and node.id in binding:
        return binding[node.id]
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        values = [_literal_value(element, binding) for element in node.elts]
        if any(value is _UNRESOLVED for value in values):
            return _UNRESOLVED
        if isinstance(node, ast.Tuple):
            return tuple(values)
        if isinstance(node, ast.Set):
            return frozenset(values)
        return values
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        operand = _literal_value(node.operand, binding)
        if isinstance(operand, (int, float, complex)):
            return -operand if isinstance(node.op, ast.USub) else operand
    return _UNRESOLVED


def _argument_spec(
    call: ast.Call,
    binding: dict[str, object] | None = None,
) -> OptionSpec | PositionalSpec | None:
    names = tuple(
        value
        for argument in call.args
        if isinstance((value := _literal_value(argument, binding)), str)
    )
    if not names:
        return None

    action = _literal_value(_call_keyword(call, "action"), binding)
    nargs = _literal_value(_call_keyword(call, "nargs"), binding)
    if action in {
        "append_const",
        "count",
        "help",
        "store_const",
        "store_false",
        "store_true",
        "version",
    }:
        min_values, max_values = 0, 0
    elif isinstance(nargs, int):
        min_values = max_values = nargs
    elif nargs == "?":
        min_values, max_values = 0, 1
    elif nargs == "*":
        min_values, max_values = 0, None
    elif nargs == "+":
        min_values, max_values = 1, None
    else:
        min_values = max_values = 1

    raw_choices = _literal_value(_call_keyword(call, "choices"), binding)
    literal_choices = (
        {str(value) for value in raw_choices}
        if isinstance(raw_choices, (list, tuple, set, frozenset))
        else set()
    )
    is_option = any(name.startswith("-") for name in names)
    if not is_option:
        return PositionalSpec(
            names[0],
            min_values,
            max_values,
            frozenset(literal_choices),
        )

    aliases = tuple(name for name in names if name.startswith("-"))
    destination = _literal_value(_call_keyword(call, "dest"), binding)
    help_text = _literal_value(_call_keyword(call, "help"), binding)
    destination_text = destination if isinstance(destination, str) else ""
    help_semantics = help_text if isinstance(help_text, str) else ""
    explicit_output_destination = re.search(
        r"(?:^|_)(?:output|write|save|emit)(?:_|$)",
        destination_text,
        re.IGNORECASE,
    )
    output_help = re.search(
        r"\b(?:output|write|save|emit)\b",
        help_semantics,
        re.IGNORECASE,
    )
    input_help = re.search(
        r"\b(?:input|read|load|source)\b",
        help_semantics,
        re.IGNORECASE,
    )
    output_path = bool(explicit_output_destination or (output_help and not input_help))
    return OptionSpec(
        aliases=aliases,
        min_values=min_values,
        max_values=max_values,
        choices=frozenset(literal_choices),
        required=_literal_value(_call_keyword(call, "required"), binding) is True,
        output_path=output_path,
    )


def _expression_reference(node: ast.AST) -> str | None:
    """Return a stable dotted reference for a simple name or attribute."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        owner = _expression_reference(node.value)
        return f"{owner}.{node.attr}" if owner else None
    return None


def _assigned_name(node: ast.Assign | ast.AnnAssign) -> str | None:
    if isinstance(node, ast.Assign):
        if len(node.targets) != 1:
            return None
        return _expression_reference(node.targets[0])
    return _expression_reference(node.target)


def _assigned_call(node: ast.AST) -> tuple[str, ast.Call] | None:
    if not isinstance(node, (ast.Assign, ast.AnnAssign)):
        return None
    name = _assigned_name(node)
    value = node.value
    if name is None or not isinstance(value, ast.Call):
        return None
    return name, value


def _is_call_named(call: ast.Call, name: str) -> bool:
    if isinstance(call.func, ast.Name):
        return call.func.id == name
    return isinstance(call.func, ast.Attribute) and call.func.attr == name


def _literal_loop_bindings(node: ast.For) -> list[dict[str, object]]:
    """Expand a finite loop whose targets and iterable are literal values."""
    if isinstance(node.target, ast.Name):
        target_names = (node.target.id,)
    elif isinstance(node.target, (ast.Tuple, ast.List)) and all(
        isinstance(element, ast.Name) for element in node.target.elts
    ):
        target_names = tuple(element.id for element in node.target.elts)
    else:
        return []
    if not isinstance(node.iter, (ast.Tuple, ast.List)):
        return []

    bindings: list[dict[str, object]] = []
    for item in node.iter.elts:
        if len(target_names) == 1:
            raw_values = (_literal_value(item),)
        elif isinstance(item, (ast.Tuple, ast.List)):
            raw_values = tuple(_literal_value(value) for value in item.elts)
        else:
            return []
        if len(raw_values) != len(target_names) or any(
            value is _UNRESOLVED for value in raw_values
        ):
            return []
        bindings.append(dict(zip(target_names, raw_values)))
    return bindings


def _bound_loop_value(node: ast.AST, binding: dict[str, object]) -> object | None:
    value = _literal_value(node, binding)
    return None if value is _UNRESOLVED else value


@lru_cache(maxsize=None)
def _argparse_program_contract(script_path: Path) -> ProgramSpec | None:
    """Extract root/subcommand argparse contracts without importing a script."""
    if not script_path.is_file():
        return None
    try:
        tree = ast.parse(script_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None
    argparse_evidence = any(
        (
            isinstance(node, ast.Import)
            and any(alias.name == "argparse" for alias in node.names)
        )
        or (isinstance(node, ast.ImportFrom) and node.module == "argparse")
        or (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "argparse"
            and node.func.attr == "ArgumentParser"
        )
        for node in ast.walk(tree)
    )

    scope_by_node: dict[int, tuple[tuple[str, str, int], ...]] = {}
    function_scopes: dict[
        tuple[tuple[tuple[str, str, int], ...], str],
        tuple[tuple[str, str, int], ...],
    ] = {}
    function_parameters: dict[
        tuple[tuple[str, str, int], ...],
        tuple[str, ...],
    ] = {}
    class_scopes: dict[
        tuple[tuple[tuple[str, str, int], ...], str],
        tuple[tuple[str, str, int], ...],
    ] = {}

    def normalize_reference(reference: str) -> str:
        if reference.startswith(("self.", "cls.")):
            return "$class." + reference.split(".", 1)[1]
        return reference

    def class_scope(
        scope: tuple[tuple[str, str, int], ...],
    ) -> tuple[tuple[str, str, int], ...]:
        last_class = next(
            (index for index in range(len(scope) - 1, -1, -1) if scope[index][0] == "class"),
            None,
        )
        return scope if last_class is None else scope[: last_class + 1]

    def binding_key(
        reference: str,
        scope: tuple[tuple[str, str, int], ...],
    ) -> tuple[tuple[tuple[str, str, int], ...], str]:
        normalized = normalize_reference(reference)
        if normalized.startswith("$class."):
            scope = class_scope(scope)
        return scope, normalized

    def walk_scopes(
        node: ast.AST,
        scope: tuple[tuple[str, str, int], ...] = (),
    ) -> None:
        scope_by_node[id(node)] = scope
        if isinstance(node, ast.ClassDef):
            nested_scope = (*scope, ("class", node.name, node.lineno))
            class_scopes[binding_key(node.name, scope)] = nested_scope
            for expression in [
                *node.decorator_list,
                *node.bases,
                *(keyword.value for keyword in node.keywords),
            ]:
                walk_scopes(expression, scope)
            for statement in node.body:
                walk_scopes(statement, nested_scope)
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nested_scope = (*scope, ("function", node.name, node.lineno))
            function_scopes[binding_key(node.name, scope)] = nested_scope
            function_parameters[nested_scope] = tuple(
                argument.arg
                for argument in [
                    *node.args.posonlyargs,
                    *node.args.args,
                    *node.args.kwonlyargs,
                ]
            )
            if scope and scope[-1][0] == "class":
                function_scopes[binding_key(f"$class.{node.name}", scope)] = nested_scope
            for expression in [*node.decorator_list, node.args]:
                walk_scopes(expression, scope)
            if node.returns is not None:
                walk_scopes(node.returns, scope)
            for statement in node.body:
                walk_scopes(statement, nested_scope)
            return
        for child in ast.iter_child_nodes(node):
            walk_scopes(child, scope)

    walk_scopes(tree)

    def resolve_reference(
        mapping: dict[tuple[tuple[tuple[str, str, int], ...], str], object],
        reference: str,
        scope: tuple[tuple[str, str, int], ...],
    ) -> object | None:
        normalized = normalize_reference(reference)
        if normalized.startswith("$class."):
            return mapping.get(binding_key(normalized, scope))
        for length in range(len(scope), -1, -1):
            key = (scope[:length], normalized)
            if key in mapping:
                return mapping[key]
        return None

    assignment_nodes = [
        node for node in ast.walk(tree) if isinstance(node, (ast.Assign, ast.AnnAssign))
    ]
    assignments = [
        (
            binding_key(variable, scope_by_node[id(node)]),
            call,
            node,
        )
        for node in assignment_nodes
        if (assigned := _assigned_call(node)) is not None
        for variable, call in (assigned,)
    ]
    instance_binding_candidates: dict[
        tuple[tuple[tuple[str, str, int], ...], str],
        set[tuple[tuple[str, str, int], ...] | None],
    ] = {}
    for node in assignment_nodes:
        variable = _assigned_name(node)
        if variable is None:
            continue
        scope = scope_by_node[id(node)]
        class_scope_value: tuple[tuple[str, str, int], ...] | None = None
        value = node.value
        if isinstance(value, ast.Call):
            class_reference = _expression_reference(value.func)
            if class_reference is not None:
                resolved_class = resolve_reference(class_scopes, class_reference, scope)
                if isinstance(resolved_class, tuple):
                    class_scope_value = resolved_class
        instance_binding_candidates.setdefault(
            binding_key(variable, scope),
            set(),
        ).add(class_scope_value)
    instance_class_scopes = {
        variable: next(iter(candidates))
        for variable, candidates in instance_binding_candidates.items()
        if len(candidates) == 1 and None not in candidates
    }
    calls = sorted(
        (node for node in ast.walk(tree) if isinstance(node, ast.Call)),
        key=lambda node: (node.lineno, node.col_offset),
    )

    # A parser path is ``(root parser identity, command-name tuple)``.
    parser_paths: dict[
        tuple[tuple[tuple[str, str, int], ...], str],
        tuple[tuple[object, ...], tuple[str, ...]],
    ] = {}
    root_ids: set[tuple[object, ...]] = set()
    for variable, call, _node in assignments:
        if not _is_call_named(call, "ArgumentParser"):
            continue
        root_id = (variable, call.lineno, call.col_offset)
        root_ids.add(root_id)
        parser_paths[variable] = (root_id, ())

    returns_by_scope: dict[
        tuple[tuple[str, str, int], ...],
        list[ast.AST],
    ] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and node.value is not None:
            returns_by_scope.setdefault(scope_by_node[id(node)], []).append(node.value)

    def resolve_function(
        expression: ast.AST,
        scope: tuple[tuple[str, str, int], ...],
    ) -> tuple[tuple[str, str, int], ...] | None:
        reference = _expression_reference(expression)
        if reference is not None:
            resolved = resolve_reference(function_scopes, reference, scope)
            if isinstance(resolved, tuple):
                return resolved
        if not isinstance(expression, ast.Attribute):
            return None

        owner_scope: tuple[tuple[str, str, int], ...] | None = None
        if isinstance(expression.value, ast.Call):
            class_reference = _expression_reference(expression.value.func)
            if class_reference is not None:
                resolved_class = resolve_reference(class_scopes, class_reference, scope)
                if isinstance(resolved_class, tuple):
                    owner_scope = resolved_class
        else:
            owner_reference = _expression_reference(expression.value)
            if owner_reference is not None:
                resolved_instance = resolve_reference(
                    instance_class_scopes,
                    owner_reference,
                    scope,
                )
                if isinstance(resolved_instance, tuple):
                    owner_scope = resolved_instance
        if owner_scope is None:
            return None
        resolved_method = function_scopes.get((owner_scope, expression.attr))
        return resolved_method if isinstance(resolved_method, tuple) else None

    function_return_paths: dict[
        tuple[tuple[str, str, int], ...],
        tuple[tuple[object, ...], tuple[str, ...]],
    ] = {}

    def resolve_parser_expression(
        expression: ast.AST,
        scope: tuple[tuple[str, str, int], ...],
    ) -> tuple[tuple[object, ...], tuple[str, ...]] | None:
        reference = _expression_reference(expression)
        if reference is not None:
            resolved = resolve_reference(parser_paths, reference, scope)
            if isinstance(resolved, tuple):
                return resolved
        if isinstance(expression, ast.Call):
            function_scope = resolve_function(expression.func, scope)
            if function_scope is not None:
                return function_return_paths.get(function_scope)
        return None

    subparser_paths: dict[
        tuple[tuple[tuple[str, str, int], ...], str],
        tuple[tuple[object, ...], tuple[str, ...]],
    ] = {}
    declared_subparser_symbols = {
        variable
        for variable, call, _node in assignments
        if isinstance(call.func, ast.Attribute) and call.func.attr == "add_subparsers"
    }
    # Tiny compatibility fixtures and generated entry points sometimes expose
    # only ``sub.add_parser(...)``. Keep that permissive shape isolated to the
    # exact scoped ``sub`` object instead of merging every unresolved owner.
    for call in calls:
        if not (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "add_parser"
            and (owner := _expression_reference(call.func.value)) is not None
        ):
            continue
        owner_key = binding_key(owner, scope_by_node[id(call)])
        if owner_key in subparser_paths or owner_key in declared_subparser_symbols:
            continue
        synthetic_root = ("synthetic", owner_key)
        root_ids.add(synthetic_root)
        subparser_paths[owner_key] = (synthetic_root, ())

    changed = True
    while changed:
        changed = False
        for function_scope, expressions in returns_by_scope.items():
            resolved = {
                path
                for expression in expressions
                if (path := resolve_parser_expression(expression, function_scope)) is not None
            }
            if len(resolved) == 1:
                path = next(iter(resolved))
                if function_return_paths.get(function_scope) != path:
                    function_return_paths[function_scope] = path
                    changed = True
        for variable, call, node in assignments:
            scope = scope_by_node[id(node)]
            if not isinstance(call.func, ast.Attribute):
                function_scope = resolve_function(call.func, scope)
                if function_scope is not None and function_scope in function_return_paths:
                    path = function_return_paths[function_scope]
                    if parser_paths.get(variable) != path:
                        parser_paths[variable] = path
                        changed = True
                continue
            owner = _expression_reference(call.func.value)
            if owner is None:
                continue
            if call.func.attr == "add_subparsers":
                path = resolve_reference(parser_paths, owner, scope)
                if isinstance(path, tuple) and subparser_paths.get(variable) != path:
                    subparser_paths[variable] = path
                    changed = True
            elif call.func.attr == "add_parser" and call.args:
                parent = resolve_reference(subparser_paths, owner, scope)
                name = _literal_value(call.args[0])
                if isinstance(parent, tuple) and isinstance(name, str):
                    path = (parent[0], (*parent[1], name))
                    if parser_paths.get(variable) != path:
                        parser_paths[variable] = path
                        changed = True

    command_paths: set[tuple[tuple[object, ...], tuple[str, ...]]] = set()
    command_aliases: dict[
        tuple[tuple[object, ...], tuple[str, ...]],
        list[str],
    ] = {}
    parent_paths: dict[
        tuple[tuple[object, ...], tuple[str, ...]],
        list[tuple[tuple[object, ...], tuple[str, ...]]],
    ] = {}

    def record_command_aliases(
        path: tuple[tuple[object, ...], tuple[str, ...]],
        call: ast.Call,
        binding: dict[str, object] | None = None,
    ) -> None:
        raw_aliases = _literal_value(_call_keyword(call, "aliases"), binding)
        if not isinstance(raw_aliases, (list, tuple)):
            return
        aliases = command_aliases.setdefault(path, [])
        for alias in raw_aliases:
            if isinstance(alias, str) and alias and alias not in aliases:
                aliases.append(alias)

    def resolved_parent_paths(
        call: ast.Call,
        scope: tuple[tuple[str, str, int], ...],
    ) -> list[tuple[tuple[object, ...], tuple[str, ...]]]:
        parents = _call_keyword(call, "parents")
        if not isinstance(parents, (ast.List, ast.Tuple)):
            return []
        resolved: list[tuple[tuple[object, ...], tuple[str, ...]]] = []
        for expression in parents.elts:
            path = resolve_parser_expression(expression, scope)
            if path is not None and path not in resolved:
                resolved.append(path)
        return resolved

    for call in calls:
        if not (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "add_parser"
            and call.args
            and (owner := _expression_reference(call.func.value)) is not None
        ):
            continue
        parent = resolve_reference(
            subparser_paths,
            owner,
            scope_by_node[id(call)],
        )
        name = _literal_value(call.args[0])
        if isinstance(parent, tuple) and isinstance(name, str):
            path = (parent[0], (*parent[1], name))
            command_paths.add(path)
            record_command_aliases(path, call)
            inherited = resolved_parent_paths(call, scope_by_node[id(call)])
            if inherited:
                parent_paths[path] = inherited

    for variable, call, node in assignments:
        if not _is_call_named(call, "ArgumentParser"):
            continue
        path = parser_paths.get(variable)
        if path is None:
            continue
        inherited = resolved_parent_paths(call, scope_by_node[id(node)])
        if inherited:
            parent_paths[path] = inherited

    loop_generated: dict[
        int,
        dict[
            tuple[tuple[tuple[str, str, int], ...], str],
            list[
                tuple[
                    dict[str, object],
                    tuple[tuple[object, ...], tuple[str, ...]],
                ]
            ],
        ],
    ] = {}
    for loop in (node for node in ast.walk(tree) if isinstance(node, ast.For)):
        bindings = _literal_loop_bindings(loop)
        if not bindings:
            continue
        generated: dict[
            tuple[tuple[tuple[str, str, int], ...], str],
            list[
                tuple[
                    dict[str, object],
                    tuple[tuple[object, ...], tuple[str, ...]],
                ]
            ],
        ] = {}
        for statement in loop.body:
            assigned = _assigned_call(statement)
            if assigned is None:
                continue
            variable_name, call = assigned
            if not (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "add_parser"
                and call.args
                and (owner := _expression_reference(call.func.value)) is not None
            ):
                continue
            parent = resolve_reference(
                subparser_paths,
                owner,
                scope_by_node[id(statement)],
            )
            if not isinstance(parent, tuple):
                continue
            variable = binding_key(variable_name, scope_by_node[id(statement)])
            for binding in bindings:
                name = _bound_loop_value(call.args[0], binding)
                if not isinstance(name, str):
                    continue
                path = (parent[0], (*parent[1], name))
                generated.setdefault(variable, []).append((binding, path))
                command_paths.add(path)
                record_command_aliases(path, call, binding)
                inherited = resolved_parent_paths(
                    call,
                    scope_by_node[id(statement)],
                )
                if inherited:
                    parent_paths[path] = inherited
            if generated.get(variable):
                parser_paths[variable] = generated[variable][-1][1]
        loop_generated[id(loop)] = generated

    argument_containers = dict(parser_paths)
    mutually_exclusive_groups: dict[
        tuple[tuple[tuple[str, str, int], ...], str],
        tuple[tuple[tuple[object, ...], tuple[str, ...]], bool],
    ] = {}
    unresolved_groups = True
    while unresolved_groups:
        unresolved_groups = False
        for variable, call, node in assignments:
            if variable in argument_containers or not isinstance(call.func, ast.Attribute):
                continue
            if call.func.attr not in {"add_argument_group", "add_mutually_exclusive_group"}:
                continue
            owner = _expression_reference(call.func.value)
            if owner is None:
                continue
            path = resolve_reference(
                argument_containers,
                owner,
                scope_by_node[id(node)],
            )
            if not isinstance(path, tuple):
                continue
            argument_containers[variable] = path
            if call.func.attr == "add_mutually_exclusive_group":
                mutually_exclusive_groups[variable] = (
                    path,
                    _literal_value(_call_keyword(call, "required")) is True,
                )
            unresolved_groups = True

    if not root_ids:
        return None

    loop_arguments: dict[
        int,
        list[
            tuple[
                tuple[tuple[object, ...], tuple[str, ...]],
                dict[str, object],
            ]
        ],
    ] = {}
    for loop in (node for node in ast.walk(tree) if isinstance(node, ast.For)):
        bindings = _literal_loop_bindings(loop)
        if not bindings:
            continue
        generated = loop_generated.get(id(loop), {})
        for statement in loop.body:
            for node in ast.walk(statement):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"
                    and (owner := _expression_reference(node.func.value)) is not None
                ):
                    continue
                owner_key = binding_key(owner, scope_by_node[id(node)])
                if owner_key in generated:
                    loop_arguments[id(node)] = [
                        (path, binding) for binding, path in generated[owner_key]
                    ]
                    continue
                path = resolve_reference(
                    argument_containers,
                    owner,
                    scope_by_node[id(node)],
                )
                if isinstance(path, tuple):
                    loop_arguments[id(node)] = [(path, binding) for binding in bindings]

    helper_argument_paths: dict[
        int,
        set[tuple[tuple[object, ...], tuple[str, ...]]],
    ] = {}
    for node in calls:
        if not (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and isinstance(node.func.value, ast.Name)
        ):
            continue
        scope = scope_by_node[id(node)]
        parameters = function_parameters.get(scope, ())
        if node.func.value.id not in parameters:
            continue
        parameter_index = parameters.index(node.func.value.id)
        for invocation in calls:
            function_scope = resolve_function(
                invocation.func,
                scope_by_node[id(invocation)],
            )
            if function_scope != scope:
                continue
            argument_expression: ast.AST | None = None
            if parameter_index < len(invocation.args):
                argument_expression = invocation.args[parameter_index]
            else:
                argument_expression = next(
                    (
                        keyword.value
                        for keyword in invocation.keywords
                        if keyword.arg == node.func.value.id
                    ),
                    None,
                )
            if argument_expression is None:
                continue
            path = resolve_parser_expression(
                argument_expression,
                scope_by_node[id(invocation)],
            )
            if path is not None:
                helper_argument_paths.setdefault(id(node), set()).add(path)

    all_paths = {
        *((root_id, ()) for root_id in root_ids),
        *command_paths,
    }
    specs: dict[
        tuple[tuple[object, ...], tuple[str, ...]],
        CommandSpec,
    ] = {path: CommandSpec.empty() for path in all_paths}
    for path, spec in specs.items():
        if path[0] and path[0][0] == "synthetic" and path[1]:
            spec.positionals.append(PositionalSpec("arguments", min_values=0, max_values=None))
    mutually_exclusive_options: dict[
        tuple[tuple[tuple[str, str, int], ...], str],
        list[OptionSpec],
    ] = {variable: [] for variable in mutually_exclusive_groups}
    for node in calls:
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"):
            continue
        owner = _expression_reference(node.func.value)
        if owner is None:
            continue
        declarations: list[
            tuple[
                tuple[tuple[object, ...], tuple[str, ...]],
                dict[str, object] | None,
            ]
        ]
        if id(node) in loop_arguments:
            declarations = list(loop_arguments[id(node)])
        elif id(node) in helper_argument_paths:
            declarations = [
                (path, None) for path in sorted(helper_argument_paths[id(node)], key=repr)
            ]
        else:
            path = resolve_reference(
                argument_containers,
                owner,
                scope_by_node[id(node)],
            )
            declarations = [(path, None)] if isinstance(path, tuple) else []
        for path, binding in declarations:
            argument = _argument_spec(node, binding)
            if argument is None:
                continue
            target = specs.setdefault(path, CommandSpec.empty())
            if isinstance(argument, OptionSpec):
                for alias in argument.aliases:
                    target.options[alias] = argument
                group = binding_key(owner, scope_by_node[id(node)])
                if group in mutually_exclusive_options:
                    mutually_exclusive_options[group].append(argument)
            else:
                target.positionals.append(argument)

    for variable, (path, required) in mutually_exclusive_groups.items():
        options = list(dict.fromkeys(mutually_exclusive_options[variable]))
        if not required or not options:
            continue
        aliases = frozenset(alias for option in options for alias in option.aliases)
        preferred = [option.aliases[-1] for option in options]
        if len(preferred) == 1:
            description = preferred[0]
        elif len(preferred) == 2:
            description = f"{preferred[0]} or {preferred[1]}"
        else:
            description = f"{', '.join(preferred[:-1])}, or {preferred[-1]}"
        specs.setdefault(path, CommandSpec.empty()).required_any.append((aliases, description))

    help_option = OptionSpec(aliases=("-h", "--help"), min_values=0, max_values=0)
    for spec in specs.values():
        spec.options.setdefault("-h", help_option)
        spec.options.setdefault("--help", help_option)

    required_state: dict[
        tuple[tuple[object, ...], tuple[str, ...]],
        bool,
    ] = {}
    required_events: list[tuple[int, int, tuple[tuple[object, ...], tuple[str, ...]], bool]] = []
    for call in calls:
        if not (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "add_subparsers"
            and (owner := _expression_reference(call.func.value)) is not None
        ):
            continue
        path = resolve_reference(parser_paths, owner, scope_by_node[id(call)])
        required = _literal_value(_call_keyword(call, "required"))
        if isinstance(path, tuple) and isinstance(required, bool):
            required_events.append((call.lineno, call.col_offset, path, required))
    for node in assignment_nodes:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = _literal_value(node.value)
        if not isinstance(value, bool):
            continue
        for target in targets:
            if not isinstance(target, ast.Attribute) or target.attr != "required":
                continue
            owner = _expression_reference(target.value)
            if owner is None:
                continue
            path = resolve_reference(
                subparser_paths,
                owner,
                scope_by_node[id(node)],
            )
            if isinstance(path, tuple):
                required_events.append((node.lineno, node.col_offset, path, value))
    for _line, _column, path, required in sorted(required_events):
        required_state[path] = required

    parse_modes: dict[tuple[object, ...], set[bool]] = {}
    parse_calls: list[
        tuple[
            tuple[tuple[str, str, int], ...],
            tuple[object, ...],
            bool,
        ]
    ] = []
    unresolved_parse_scopes: list[tuple[tuple[str, str, int], ...]] = []
    parse_method_names = {
        "parse_args": False,
        "parse_intermixed_args": False,
        "parse_known_args": True,
        "parse_known_intermixed_args": True,
    }
    for call in calls:
        if not isinstance(call.func, ast.Attribute):
            continue
        if call.func.attr not in parse_method_names:
            continue
        path = resolve_parser_expression(
            call.func.value,
            scope_by_node[id(call)],
        )
        if path is not None:
            parse_calls.append(
                (
                    scope_by_node[id(call)],
                    path[0],
                    parse_method_names[call.func.attr],
                )
            )
        else:
            unresolved_parse_scopes.append(scope_by_node[id(call)])

    reachable_scopes = {
        (),
        *(
            scope
            for scope in function_scopes.values()
            if (len(scope) == 1 and scope[-1][0] == "function" and scope[-1][1] == "main")
        ),
    }
    reachability_changed = True
    while reachability_changed:
        reachability_changed = False
        for call in calls:
            if scope_by_node[id(call)] not in reachable_scopes:
                continue
            function_scope = resolve_function(call.func, scope_by_node[id(call)])
            if function_scope is not None and function_scope not in reachable_scopes:
                reachable_scopes.add(function_scope)
                reachability_changed = True

    def uncertain_contract(reason: str) -> ProgramSpec:
        return ProgramSpec(
            root=CommandSpec.empty(),
            commands={},
            uncertain_reason=reason,
        )

    if (
        argparse_evidence
        and root_ids
        and any(scope in reachable_scopes for scope in unresolved_parse_scopes)
    ):
        return uncertain_contract("a reachable parse call has an unresolved parser identity")

    selected_calls = [item for item in parse_calls if item[0] in reachable_scopes]
    if parse_calls and not selected_calls:
        return None
    for _scope, root_id, allow_extras in selected_calls:
        parse_modes.setdefault(root_id, set()).add(allow_extras)

    selected_roots = set(parse_modes)
    if not selected_roots:
        inherited_roots = {parent[0] for inherited in parent_paths.values() for parent in inherited}
        candidate_roots = root_ids - inherited_roots
        if len(candidate_roots) == 1:
            selected_roots = candidate_roots
        else:

            def root_scope(
                root: tuple[object, ...],
            ) -> tuple[tuple[str, str, int], ...] | None:
                variable = root[1] if root and root[0] == "synthetic" else root[0]
                if (
                    isinstance(variable, tuple)
                    and len(variable) == 2
                    and isinstance(variable[0], tuple)
                ):
                    return variable[0]
                return None

            selected_roots = {
                root for root in candidate_roots if root_scope(root) in reachable_scopes
            }
        if not selected_roots:
            return None
    if len(selected_roots) > 1:
        if not argparse_evidence:
            return None
        return uncertain_contract("multiple reachable parser roots were discovered")
    selected_root = next(iter(selected_roots))
    modes = parse_modes.get(selected_root, {False})
    allow_extras = modes == {True}

    selected_commands = {path for path in command_paths if path[0] == selected_root}

    effective_specs: dict[
        tuple[tuple[object, ...], tuple[str, ...]],
        CommandSpec,
    ] = {}
    resolving_specs: set[tuple[tuple[object, ...], tuple[str, ...]],] = set()

    def effective_spec(
        full_path: tuple[tuple[object, ...], tuple[str, ...]],
    ) -> CommandSpec:
        cached = effective_specs.get(full_path)
        if cached is not None:
            return cached
        if full_path in resolving_specs:
            return specs.setdefault(full_path, CommandSpec.empty())
        resolving_specs.add(full_path)
        merged = CommandSpec.empty()
        for parent in parent_paths.get(full_path, []):
            merged = _merge_command_specs(merged, effective_spec(parent))
        merged = _merge_command_specs(
            merged,
            specs.setdefault(full_path, CommandSpec.empty()),
        )
        resolving_specs.remove(full_path)
        effective_specs[full_path] = merged
        return merged

    def build_program(path: tuple[str, ...]) -> ProgramSpec:
        full_path = (selected_root, path)
        children = sorted(
            command_path
            for command_path in selected_commands
            if len(command_path[1]) == len(path) + 1 and command_path[1][:-1] == path
        )
        commands = {child[1][-1]: effective_spec(child) for child in children}
        nested = {
            child[1][-1]: build_program(child[1])
            for child in children
            if any(
                len(descendant[1]) > len(child[1]) and descendant[1][: len(child[1])] == child[1]
                for descendant in selected_commands
            )
        }
        for child in children:
            canonical = child[1][-1]
            for alias in command_aliases.get(child, []):
                commands.setdefault(alias, commands[canonical])
                if canonical in nested:
                    nested.setdefault(alias, nested[canonical])
        return ProgramSpec(
            root=effective_spec(full_path),
            commands=commands,
            command_required=required_state.get(full_path, False),
            nested=nested,
            allow_extras=allow_extras,
        )

    return build_program(())


def _python_cli_contract(repo_root: Path) -> ProgramSpec | None:
    source = repo_root / "python" / "tensorrt_model_connect" / "build_cli.py"
    return _argparse_program_contract(source)


_RUNTIME_FLAG_OPTIONS = {
    "--background",
    "--chat-template",
    "--cuda-graphs",
    "--greedy",
    "--list-engines",
    "--no-punctuation",
    "--no-thinking",
    "--no-timestamps",
    "--pad-and-drop-preencoded",
    "--punctuation",
    "--stream",
    "--timestamps",
    "-h",
    "--help",
    "-v",
    "--version",
}
_RUNTIME_COMMON_OPTIONS = {
    "--backend-dir",
    "--config",
    "--cuda-graphs",
    "--model-plugin-dir",
    "--runtime-cache",
    "--set",
}
_RUNTIME_REQUIRED_OPTIONS: dict[str, tuple[tuple[set[str], str], ...]] = {
    "run": (
        (
            {"--prompt", "-p", "--prompts-file", "--initial-latents-raw"},
            "--prompt, --prompts-file, or --initial-latents-raw",
        ),
    ),
    "encode": (({"--prompt", "-p"}, "--prompt"),),
    "segment": (({"--image"}, "--image"),),
    "segment-prompted": (({"--image"}, "--image"),),
    "classify": (({"--image"}, "--image"),),
    "detect": (({"--image"}, "--image"),),
    "generate-audio": (({"--prompt", "-p"}, "--prompt"),),
    "generate-video": (({"--prompt", "-p"}, "--prompt"),),
    "embed": (({"--prompt", "-p"}, "--prompt"),),
    "rerank": (
        ({"--prompt", "-p"}, "--prompt"),
        ({"--document"}, "--document"),
    ),
    "solve": (({"--field-input", "--branch-input"}, "--field-input or --branch-input"),),
    "speak": (({"--audio-in", "--audio"}, "--audio-in or --audio"),),
    "transcribe": (({"--audio", "--audio-in"}, "--audio or --audio-in"),),
}


def _runtime_help_scopes(content: str) -> dict[str, set[str]]:
    """Derive command option scopes from the native CLI's usage text."""
    usage_function = re.search(
        r"void print_usage\(\)\s*\{(?P<body>.*?)\n\}",
        content,
        re.DOTALL,
    )
    if usage_function is None:
        return {}
    fragments = re.findall(r'"((?:\\.|[^"\\])*)"', usage_function.group("body"))
    help_text = "".join(bytes(fragment, "utf-8").decode("unicode_escape") for fragment in fragments)
    scopes: dict[str, set[str]] = {}
    current: str | None = None
    in_global_options = False
    global_options: set[str] = set()
    for line in help_text.splitlines():
        command_match = re.match(r"\s*trtmc\s+([a-z][a-z0-9-]*)\b", line)
        if command_match:
            current = command_match.group(1)
            in_global_options = False
            scopes.setdefault(current, set()).update(
                re.findall(r"(?<![A-Za-z0-9_])--?[A-Za-z][A-Za-z0-9_-]*", line)
            )
            continue
        if line.strip() == "Options:":
            current = None
            in_global_options = True
            continue
        options = set(re.findall(r"(?<![A-Za-z0-9_])--?[A-Za-z][A-Za-z0-9_-]*", line))
        if in_global_options:
            global_options.update(options)
        elif current is not None:
            scopes.setdefault(current, set()).update(options)

    for options in scopes.values():
        options.update(global_options)
        options.update(_RUNTIME_COMMON_OPTIONS)
        if "--prompt" in options:
            options.add("-p")
        if "--output" in options:
            options.add("-o")
        if "--kv-cache-size" in options:
            options.add("--kv_cache_size")
    scopes.setdefault("run", set()).add("--greedy")
    scopes.setdefault("generate-video", set()).add("--seed")
    scopes.setdefault("transcribe", set()).update({"--audio-in", "--no-timestamps"})
    scopes.setdefault("speak", set()).add("--audio")
    return scopes


@lru_cache(maxsize=None)
def _runtime_cli_contract(repo_root: Path) -> ProgramSpec | None:
    source = repo_root / "src" / "cli" / "args.cpp"
    if not source.is_file():
        return None
    content = source.read_text(encoding="utf-8")
    known_match = re.search(
        r"static const char\* known_cmds\[\]\s*=\s*\{(?P<body>.*?)nullptr\s*\};",
        content,
        re.DOTALL,
    )
    commands = {"build", "help", "version"}
    if known_match:
        commands.update(re.findall(r'"([a-z][a-z0-9-]+)"', known_match.group("body")))
    all_options = set(re.findall(r'"(--?[A-Za-z][A-Za-z0-9_-]*)"', content))
    all_options.update({"-h", "--help", "-v", "--version"})
    option_specs = {
        option: OptionSpec(
            aliases=(option,),
            min_values=0 if option in _RUNTIME_FLAG_OPTIONS else 1,
            max_values=0 if option in _RUNTIME_FLAG_OPTIONS else 1,
            choices=(frozenset({"transcribe", "translate"}) if option == "--task" else frozenset()),
            allow_inline_value=option in {"--kv-cache-size", "--kv_cache_size"},
            consume_option_like_value=True,
        )
        for option in all_options
    }
    scopes = _runtime_help_scopes(content)
    specs: dict[str, CommandSpec] = {}
    for command in commands:
        scoped_options = scopes.get(command, set())
        spec = CommandSpec(
            options={
                option: option_specs[option] for option in scoped_options if option in option_specs
            },
            positionals=[],
            required_any=[],
        )
        if command not in {"build", "help", "version"}:
            spec.positionals.append(PositionalSpec("bundle"))
        for aliases, description in _RUNTIME_REQUIRED_OPTIONS.get(command, ()):
            spec.required_any.append((frozenset(aliases), description))
        specs[command] = spec
    return ProgramSpec(root=CommandSpec.empty(), commands=specs)


def _trtmc_tokens(tokens: list[str]) -> list[str] | None:
    tokens = _strip_shell_wrappers(tokens)
    if not tokens:
        return None
    executable = tokens[0]
    if executable in {"trtmc", "$TRTMC"} or Path(executable).name == "trtmc":
        return tokens
    return None


def _merge_command_specs(root: CommandSpec, command: CommandSpec) -> CommandSpec:
    options = dict(root.options)
    options.update(command.options)
    return CommandSpec(
        options=options,
        positionals=[*root.positionals, *command.positionals],
        required_any=[*root.required_any, *command.required_any],
    )


def _match_option_token(
    token: str,
    spec: CommandSpec,
    *,
    positional_only: bool = False,
) -> tuple[str, OptionSpec | None, bool, str]:
    """Resolve exact, ``--long=value``, and argparse ``-ovalue`` forms."""
    option_name, separator, inline_value = token.partition("=")
    option = None if positional_only else spec.options.get(option_name)
    if option is not None:
        return option_name, option, bool(separator), inline_value
    if positional_only or not token.startswith("-") or token.startswith("--"):
        return option_name, None, False, ""
    for alias, candidate in spec.options.items():
        if (
            len(alias) == 2
            and token.startswith(alias)
            and len(token) > len(alias)
            and candidate.max_values != 0
            and candidate.allow_inline_value
        ):
            return alias, candidate, True, token[len(alias) :]
    return option_name, None, False, ""


def _expand_short_option_token(token: str, spec: CommandSpec) -> list[str]:
    """Expand argparse short-flag clusters while preserving attached values."""
    if (
        token in spec.options
        or not token.startswith("-")
        or token.startswith("--")
        or len(token) <= 2
        or _ARGPARSE_NEGATIVE_NUMBER_RE.fullmatch(token) is not None
        or "=" in token
    ):
        return [token]
    expanded: list[str] = []
    cursor = 1
    while cursor < len(token):
        alias = f"-{token[cursor]}"
        option = spec.options.get(alias)
        if option is None:
            return [token]
        if option.max_values == 0:
            expanded.append(alias)
            cursor += 1
            continue
        remainder = token[cursor + 1 :]
        expanded.append(f"{alias}{remainder}" if remainder else alias)
        return expanded
    return expanded or [token]


def _expand_short_options(tokens: Sequence[str], spec: CommandSpec) -> list[str]:
    return [expanded for token in tokens for expanded in _expand_short_option_token(token, spec)]


def _parse_command_arguments(
    tokens: list[str],
    spec: CommandSpec,
) -> tuple[
    list[str],
    set[str],
    list[tuple[str, str, frozenset[str], bool]],
    list[str],
]:
    """Parse tokens by static arity.

    Returns positional values, seen option aliases, choice violations, and
    human-readable arity/unknown-option errors. Known option values are
    consumed even when they begin with ``-``; an exact known option spelling
    still starts the next option.
    """
    tokens = _expand_short_options(tokens, spec)
    positionals: list[str] = []
    seen_options: set[str] = set()
    choice_errors: list[tuple[str, str, frozenset[str], bool]] = []
    errors: list[str] = []
    index = 0
    positional_only = False
    while index < len(tokens):
        token = tokens[index]
        if token == "--" and not positional_only:
            if not spec.positionals:
                positionals.append(token)
                positional_only = True
                index += 1
                continue
            positional_only = True
            index += 1
            continue
        option_name, option, inline_form, inline_value = _match_option_token(
            token,
            spec,
            positional_only=positional_only,
        )
        if option is None:
            if (
                not positional_only
                and token.startswith("-")
                and _ARGPARSE_NEGATIVE_NUMBER_RE.fullmatch(token) is None
                and "DOC_PLACEHOLDER" not in token
                and not token.startswith("[")
            ):
                errors.append(f"unknown option:{option_name}")
            else:
                positionals.append(token)
            index += 1
            continue

        seen_options.update(option.aliases)
        values: list[str] = []
        if inline_form:
            if not option.allow_inline_value:
                errors.append(f"does not support inline value:{option_name}")
            elif option.max_values == 0:
                errors.append(f"does not take a value:{option_name}")
            else:
                # argparse treats ``--option=value`` as exactly one attached
                # value.  The value may be an empty string, and it is not
                # supplemented with following tokens when nargs requires more.
                values.append(inline_value)
                if len(values) < option.min_values:
                    errors.append(f"requires a value:{option_name}")
        elif option.max_values != 0:
            cursor = index + 1
            while cursor < len(tokens) and (
                option.max_values is None or len(values) < option.max_values
            ):
                candidate = tokens[cursor]
                _candidate_name, candidate_option, _inline, _value = _match_option_token(
                    candidate, spec
                )
                if candidate_option is not None:
                    break
                if (
                    candidate.startswith("-")
                    and _ARGPARSE_NEGATIVE_NUMBER_RE.fullmatch(candidate) is None
                    and not option.consume_option_like_value
                ):
                    break
                values.append(candidate)
                cursor += 1
            if len(values) < option.min_values:
                errors.append(f"requires a value:{option_name}")
            else:
                index = cursor - 1

        if option.choices:
            for value in values:
                if "DOC_PLACEHOLDER" not in value and value not in option.choices:
                    choice_errors.append((option_name, value, option.choices, True))
        index += 1

    assigned, _extras = _assign_positional_values(positionals, spec.positionals)
    for positional, values in zip(spec.positionals, assigned):
        if not positional.choices:
            continue
        for value in values:
            if "DOC_PLACEHOLDER" not in value and value not in positional.choices:
                choice_errors.append((positional.name, value, positional.choices, False))

    return positionals, seen_options, choice_errors, errors


def _assign_positional_values(
    values: Sequence[str],
    specs: Sequence[PositionalSpec],
) -> tuple[list[list[str]], list[str]]:
    """Assign flat positional values using argparse-compatible ``nargs`` greed."""
    assigned: list[list[str]] = []
    cursor = 0
    for index, positional in enumerate(specs):
        available = len(values) - cursor
        later_minimum = sum(item.min_values for item in specs[index + 1 :])
        minimum_here = min(positional.min_values, available)
        extra = max(0, available - minimum_here - later_minimum)
        if positional.max_values is None:
            count = minimum_here + extra
        else:
            count = min(positional.max_values, minimum_here + extra)
        assigned.append(list(values[cursor : cursor + count]))
        cursor += count
    return assigned, list(values[cursor:])


def _check_command_spec(
    block: ShellBlock,
    offset: int,
    tokens: list[str],
    spec: CommandSpec,
    *,
    label: str,
    unknown_template: str,
    allow_extras: bool = False,
) -> list[Finding]:
    positionals, seen_options, choice_errors, errors = _parse_command_arguments(tokens, spec)
    line = _source_line(block, offset)
    findings: list[Finding] = []
    for error in errors:
        kind, option = error.split(":", 1)
        if kind == "unknown option" and allow_extras:
            continue
        if kind == "unknown option":
            message = unknown_template.format(option=option)
        elif kind == "does not support inline value":
            message = f"option for {label} does not support `=` form: {option}"
        elif kind == "does not take a value":
            message = f"option for {label} does not take a value: {option}"
        else:
            message = f"option for {label} requires a value: {option}"
        findings.append(Finding(block.path, line, message))

    for argument, value, choices, is_option in choice_errors:
        if is_option:
            choice_label = (
                f"`{label.strip('`')} {argument}`"
                if label.startswith("`") and label.endswith("`")
                else f"{label} {argument}"
            )
        else:
            choice_label = f"positional `{argument}` for {label}"
        findings.append(
            Finding(
                block.path,
                line,
                f"invalid value for {choice_label}: {value}; "
                f"expected one of {', '.join(sorted(choices))}",
            )
        )

    is_abstract = any(
        token == "..." or token.startswith("[") or token.endswith("...]") for token in tokens
    )
    skip_required = (
        block.language == "inline" or is_abstract or bool({"-h", "--help"} & seen_options)
    )
    if skip_required:
        return findings

    for option in dict.fromkeys(spec.options.values()):
        if option.required and not set(option.aliases) & seen_options:
            findings.append(
                Finding(
                    block.path,
                    line,
                    f"missing required option for {label}: {option.aliases[-1]}",
                )
            )
    for aliases, description in spec.required_any:
        if not aliases & seen_options:
            findings.append(
                Finding(
                    block.path,
                    line,
                    f"missing required input for {label}: {description}",
                )
            )

    assigned_positionals, extra_positionals = _assign_positional_values(
        positionals,
        spec.positionals,
    )
    missing = next(
        (
            positional.name
            for positional, values in zip(spec.positionals, assigned_positionals)
            if len(values) < positional.min_values
        ),
        None,
    )
    if missing is not None:
        findings.append(
            Finding(
                block.path,
                line,
                f"missing required positional for {label}: {missing}",
            )
        )
    if (
        not allow_extras
        and extra_positionals
        and not any(error.startswith("unknown option:") for error in errors)
    ):
        findings.append(
            Finding(
                block.path,
                line,
                f"unexpected positional argument for {label}: {extra_positionals[0]}",
            )
        )
    return findings


def check_trtmc_contract(block: ShellBlock, repo_root: Path) -> list[Finding]:
    """Validate native CLI subcommands, option scope, arity, and requirements."""
    runtime = _runtime_cli_contract(repo_root)
    python_cli = _python_cli_contract(repo_root)
    if runtime is None:
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
        if command not in runtime.commands:
            findings.append(
                Finding(
                    block.path,
                    _source_line(block, offset),
                    f"unknown trtmc subcommand: {command}",
                )
            )
            continue

        if (
            command == "build"
            and python_cli is not None
            and python_cli.uncertain_reason is not None
        ):
            findings.extend(
                _check_argparse_invocation(
                    block,
                    offset,
                    invocation[2:],
                    "trtmc build",
                    python_cli,
                )
            )
            continue
        if command == "build" and python_cli is not None:
            spec = _merge_command_specs(
                python_cli.root,
                python_cli.commands.get("build", CommandSpec.empty()),
            )
        else:
            spec = runtime.commands[command]
        findings.extend(
            _check_command_spec(
                block,
                offset,
                invocation[2:],
                spec,
                label=f"`trtmc {command}`",
                unknown_template=f"unknown option for `trtmc {command}`: {{option}}",
            )
        )
    return findings


def check_python_module_contract(
    block: ShellBlock,
    repo_root: Path,
) -> list[Finding]:
    """Validate project-local ``python -m`` argparse examples."""
    tensorrt_program = _python_cli_contract(repo_root)
    findings: list[Finding] = []
    for offset, raw_tokens in _block_commands(block):
        tokens = _strip_shell_wrappers(raw_tokens)
        if len(tokens) < 3:
            continue
        if Path(tokens[0]).name not in {"python", "python3"}:
            continue
        if tokens[1] != "-m":
            continue
        module = tokens[2]
        if module == "tensorrt_model_connect":
            if tensorrt_program is None or len(tokens) < 4:
                continue
            if tensorrt_program.uncertain_reason is not None:
                findings.extend(
                    _check_argparse_invocation(
                        block,
                        offset,
                        tokens[3:],
                        "python -m tensorrt_model_connect",
                        tensorrt_program,
                    )
                )
                continue
            command = tokens[3]
            if command.startswith("-") or "DOC_PLACEHOLDER" in command:
                continue
            if command not in tensorrt_program.commands:
                findings.append(
                    Finding(
                        block.path,
                        _source_line(block, offset),
                        f"unknown `python -m tensorrt_model_connect` subcommand: {command}",
                    )
                )
                continue
            spec = _merge_command_specs(
                tensorrt_program.root,
                tensorrt_program.commands[command],
            )
            findings.extend(
                _check_command_spec(
                    block,
                    offset,
                    tokens[4:],
                    spec,
                    label=f"Python `{command}` command",
                    unknown_template=(f"unknown option for Python `{command}` command: {{option}}"),
                )
            )
            continue

        module_path = _local_python_module_path(repo_root, module)
        if module_path is None:
            continue
        program = _argparse_program_contract(module_path)
        if program is None:
            continue
        findings.extend(
            _check_argparse_invocation(
                block,
                offset,
                tokens[3:],
                f"python -m {module}",
                program,
            )
        )
    return findings


def _local_python_module_path(repo_root: Path, module: str) -> Path | None:
    """Resolve a dotted module to its repository-local ``-m`` entry point."""
    parts = module.split(".")
    if not parts or any(not part.isidentifier() for part in parts):
        return None
    relative = Path(*parts)
    module_file = (repo_root / relative).with_suffix(".py")
    if module_file.is_file():
        return module_file
    package_main = repo_root / relative / "__main__.py"
    if package_main.is_file():
        return package_main
    return None


def check_positional_inputs(block: ShellBlock, repo_root: Path) -> list[Finding]:
    """Verify explicit repo-local positional inputs for known public CLIs."""
    runtime = _runtime_cli_contract(repo_root)
    python_cli = _python_cli_contract(repo_root)
    findings: list[Finding] = []
    for offset, raw_tokens in _block_commands(block):
        tokens = _strip_shell_wrappers(raw_tokens)
        spec: CommandSpec | None = None
        argument_tokens: list[str] = []

        invocation = _trtmc_tokens(tokens)
        if invocation and len(invocation) >= 2:
            command = invocation[1]
            if command == "build" and python_cli is not None:
                spec = python_cli.commands.get("build")
            elif runtime is not None:
                spec = runtime.commands.get(command)
            argument_tokens = invocation[2:]
        elif (
            len(tokens) >= 4
            and Path(tokens[0]).name in {"python", "python3"}
            and tokens[1:3] == ["-m", "tensorrt_model_connect"]
            and python_cli is not None
        ):
            spec = python_cli.commands.get(tokens[3])
            argument_tokens = tokens[4:]

        if spec is None or not spec.positionals:
            continue
        positionals, _seen, _choices, _errors = _parse_command_arguments(
            argument_tokens,
            spec,
        )
        for value in positionals:
            local = _clean_local_path(value)
            if local and not (repo_root / local).exists():
                findings.append(
                    Finding(
                        block.path,
                        _source_line(block, offset),
                        f"positional command input does not exist: {local}",
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


def _find_subcommand(
    tokens: list[str],
    root: CommandSpec,
    commands: set[str],
) -> tuple[str | None, int | None]:
    positional_values: list[str] = []
    index = 0
    positional_only = False
    while index < len(tokens):
        token = tokens[index]
        if token == "--" and not positional_only:
            if not root.positionals:
                return None, index
            positional_only = True
            index += 1
            continue
        assigned, _extras = _assign_positional_values(
            positional_values,
            root.positionals,
        )
        positionals_complete = all(
            len(values) >= positional.min_values
            for positional, values in zip(root.positionals, assigned)
        )
        if token in commands and positionals_complete:
            return token, index
        expanded_options = _expand_short_option_token(token, root)
        matched_options = [
            _match_option_token(
                expanded,
                root,
                positional_only=positional_only,
            )
            for expanded in expanded_options
        ]
        _option_name, option, inline_form, _inline_value = matched_options[-1]
        if any(candidate is None for _name, candidate, _inline, _value in matched_options):
            option = None
        if option is not None:
            index += 1
            if not inline_form and option.max_values != 0:
                consumed = 0
                while index < len(tokens) and (
                    option.max_values is None or consumed < option.max_values
                ):
                    _candidate_name, candidate_option, _inline, _value = _match_option_token(
                        tokens[index], root
                    )
                    if candidate_option is not None:
                        break
                    index += 1
                    consumed += 1
            continue
        if (
            not positional_only
            and token.startswith("-")
            and _ARGPARSE_NEGATIVE_NUMBER_RE.fullmatch(token) is None
        ):
            index += 1
            continue
        candidate_values = [*positional_values, token]
        _assigned, extras = _assign_positional_values(
            candidate_values,
            root.positionals,
        )
        if not root.positionals or extras:
            return None, index
        positional_values.append(token)
        index += 1
    return None, None


def _check_argparse_invocation(
    block: ShellBlock,
    offset: int,
    tokens: list[str],
    local: str,
    contract: ProgramSpec,
) -> list[Finding]:
    """Validate an argparse script, including a selected subcommand."""
    if contract.uncertain_reason is not None:
        return [
            Finding(
                block.path,
                _source_line(block, offset),
                f"cannot statically validate argparse contract for `{local}`: "
                f"{contract.uncertain_reason}",
            )
        ]

    command: str | None = None
    command_index: int | None = None
    if contract.commands:
        command, command_index = _find_subcommand(
            tokens,
            contract.root,
            set(contract.commands),
        )

    if command_index is None:
        findings = _check_command_spec(
            block,
            offset,
            tokens,
            contract.root,
            label=f"`{local}`",
            unknown_template=f"unknown option for `{local}`: {{option}}",
            allow_extras=contract.allow_extras,
        )
        if (
            contract.command_required
            and block.language != "inline"
            and not any(token in {"-h", "--help"} for token in tokens)
        ):
            findings.append(
                Finding(
                    block.path,
                    _source_line(block, offset),
                    f"missing required subcommand for `{local}`",
                )
            )
        return findings

    if command is None:
        token = tokens[command_index]
        if "DOC_PLACEHOLDER" in token or token == "...":
            return []
        return [
            Finding(
                block.path,
                _source_line(block, offset),
                f"unknown subcommand for `{local}`: {token}",
            )
        ]

    findings = _check_command_spec(
        block,
        offset,
        tokens[:command_index],
        contract.root,
        label=f"`{local}`",
        unknown_template=f"unknown option for `{local}`: {{option}}",
        allow_extras=contract.allow_extras,
    )
    spec = contract.commands[command]
    nested = contract.nested.get(command)
    if nested is not None:
        findings.extend(
            _check_argparse_invocation(
                block,
                offset,
                tokens[command_index + 1 :],
                f"{local} {command}",
                nested,
            )
        )
        return findings

    findings.extend(
        _check_command_spec(
            block,
            offset,
            tokens[command_index + 1 :],
            spec,
            label=f"`{local} {command}`",
            unknown_template=f"unknown option for `{local} {command}`: {{option}}",
            allow_extras=contract.allow_extras,
        )
    )
    return findings


@lru_cache(maxsize=None)
def _shell_script_contract(script_path: Path) -> ProgramSpec | None:
    """Extract options from a conventional Bash ``case "$1"`` parser."""
    if not script_path.is_file():
        return None
    try:
        content = script_path.read_text(encoding="utf-8")
    except OSError:
        return None
    first_line = content.splitlines()[0] if content else ""
    if script_path.suffix != ".sh" and not re.search(r"\b(?:ba)?sh\b", first_line):
        return None

    spec = CommandSpec.empty()
    case_arm = re.compile(
        r"^[ \t]*(?P<labels>--?[A-Za-z0-9][A-Za-z0-9-]*"
        r"(?:\|--?[A-Za-z0-9][A-Za-z0-9-]*)*)\)"
        r"(?P<body>.*?);;",
        re.MULTILINE | re.DOTALL,
    )
    for match in case_arm.finditer(content):
        aliases = tuple(match.group("labels").split("|"))
        body = match.group("body")
        takes_value = "$2" in body or re.search(r"\bshift\s+2\b", body) is not None
        option = OptionSpec(
            aliases=aliases,
            min_values=1 if takes_value else 0,
            max_values=1 if takes_value else 0,
            allow_inline_value=False,
            consume_option_like_value=True,
        )
        for alias in aliases:
            spec.options[alias] = option
    if not spec.options:
        return None
    # Conventional shell wrappers commonly accept one or more positional
    # selectors that cannot be recovered reliably from ``case "$1"`` arms.
    spec.positionals.append(PositionalSpec("arguments", min_values=0, max_values=None))
    return ProgramSpec(root=spec, commands={})


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
        contract = _argparse_program_contract(repo_root / local)
        if contract is None:
            continue
        findings.extend(
            _check_argparse_invocation(
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
        contract = _argparse_program_contract(repo_root / local)
        if contract is None:
            contract = _shell_script_contract(repo_root / local)
        if contract is None:
            continue
        findings.extend(
            _check_argparse_invocation(
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


def check_shell_wrapper_contract(block: ShellBlock) -> list[Finding]:
    """Reject wrapper options whose executable boundary is not statically known."""
    findings: list[Finding] = []
    for offset, raw_tokens in _block_commands(block):
        uncertainty: list[str] = []
        _strip_shell_wrappers(raw_tokens, uncertainty=uncertainty)
        findings.extend(
            Finding(
                block.path,
                _source_line(block, offset),
                f"cannot statically resolve shell wrapper: {reason}",
            )
            for reason in uncertainty
        )
    return findings


def check_command_block(
    block: ShellBlock,
    repo_root: Path,
    *,
    max_nested_depth: int = _MAX_NESTED_SHELL_DEPTH,
) -> list[Finding]:
    """Apply every non-executing command check to a block and nested payloads."""
    findings: list[Finding] = []
    for candidate in shell_validation_blocks(
        block,
        max_nested_depth=max_nested_depth,
    ):
        syntax_finding = check_shell_syntax(candidate)
        if syntax_finding:
            findings.append(syntax_finding)
        findings.extend(check_shell_wrapper_contract(candidate))
        findings.extend(check_local_inputs(candidate, repo_root))
        findings.extend(check_positional_inputs(candidate, repo_root))
        findings.extend(check_trtmc_contract(candidate, repo_root))
        findings.extend(check_python_module_contract(candidate, repo_root))
        findings.extend(check_python_script_contract(candidate, repo_root))
        findings.extend(check_direct_script_contract(candidate, repo_root))
        findings.extend(check_ctest_contract(candidate))
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
            findings.extend(check_command_block(block, repo_root))

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
    print("\nAll documentation shell examples passed syntax, CLI-contract, and local-input checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
