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
import os
import re
import shlex
import signal
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
_TRY_STAR_NODE = getattr(ast, "TryStar", None)
_TRY_NODE_TYPES = (ast.Try,) if _TRY_STAR_NODE is None else (ast.Try, _TRY_STAR_NODE)


@dataclass(frozen=True)
class ShellBlock:
    path: Path
    line: int
    language: str
    body: str
    cwd: Path = Path(".")


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
class _WrapperResolution:
    tokens: tuple[str, ...]
    cwd: Path
    uncertainty: tuple[str, ...] = ()
    cwd_hops: tuple[Path, ...] = ()


@dataclass(frozen=True)
class _NestedShellPayload:
    body: str
    cwd: Path


@dataclass(frozen=True)
class _InputCandidate:
    token: str
    allow_plain: bool = False


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
        marker in token for marker in ("$", "`", "\x00", "DOC_PLACEHOLDER")
    )


def _split_gnu_env_string(value: str) -> list[str] | None:
    """Parse the static GNU ``env -S`` subset used by documentation.

    GNU ``env`` does not use shell word splitting here. In particular, ``\\_``
    is an argument separator outside quotes and a literal space inside double
    quotes. Unsupported escapes and malformed quoting fail closed.
    """
    arguments: list[str] = []
    current: list[str] = []
    quote: str | None = None
    token_started = False
    index = 0
    whitespace = " \t\n\r\v\f"
    escapes = {
        "f": "\f",
        "n": "\n",
        "r": "\r",
        "t": "\t",
        "v": "\v",
        "#": "#",
        "$": "$",
        '"': '"',
        "'": "'",
        "\\": "\\",
    }

    def finish_argument() -> None:
        nonlocal token_started
        if token_started:
            arguments.append("".join(current))
            current.clear()
            token_started = False

    while index < len(value):
        character = value[index]
        if quote is None and character in whitespace:
            finish_argument()
            index += 1
            continue
        if quote is None and character == "#" and not token_started:
            break
        if character in {"'", '"'}:
            if quote is None:
                quote = character
                token_started = True
                index += 1
                continue
            if quote == character:
                quote = None
                index += 1
                continue
            current.append(character)
            token_started = True
            index += 1
            continue
        if character != "\\":
            current.append(character)
            token_started = True
            index += 1
            continue
        if index + 1 >= len(value):
            return None

        escaped = value[index + 1]
        if quote == "'":
            if escaped in {"'", "\\"}:
                current.append(escaped)
            else:
                current.extend(("\\", escaped))
            token_started = True
            index += 2
            continue
        if escaped == "c":
            if quote == '"':
                return None
            finish_argument()
            return arguments
        if escaped == "_":
            if quote == '"':
                current.append(" ")
                token_started = True
            else:
                finish_argument()
            index += 2
            continue
        replacement = escapes.get(escaped)
        if replacement is None:
            return None
        current.append(replacement)
        token_started = True
        index += 2

    if quote is not None:
        return None
    finish_argument()
    return arguments


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


_TIMEOUT_DURATION_RE = re.compile(
    r"\+?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|"
    r"(?i:inf(?:inity)?))(?:s|m|h|d)?\Z",
)
_TIMEOUT_FLAG_OPTIONS = {
    "--foreground",
    "--preserve-status",
    "-v",
    "--verbose",
}
_TIMEOUT_VALUE_OPTIONS = {
    "-k",
    "--kill-after",
    "-s",
    "--signal",
}


def _is_timeout_signal(value: str) -> bool:
    """Return whether GNU ``timeout`` accepts a static signal value."""
    if re.fullmatch(r"[0-9]+", value) is not None:
        number = int(value)
        return number == 0 or number in {int(candidate) for candidate in signal.valid_signals()}

    normalized = value.upper().removeprefix("SIG")
    realtime = re.fullmatch(r"(RTMIN|RTMAX)([+-])([0-9]+)", normalized)
    if realtime is not None:
        base = signal.SIGRTMIN if realtime.group(1) == "RTMIN" else signal.SIGRTMAX
        offset = int(realtime.group(3))
        number = base + offset if realtime.group(2) == "+" else base - offset
        return signal.SIGRTMIN <= number <= signal.SIGRTMAX
    candidate = getattr(signal, f"SIG{normalized}", None)
    return isinstance(candidate, int) and not normalized.startswith("_")


def _strip_timeout_wrapper(tokens: Sequence[str]) -> list[str] | None:
    """Return the command after a statically understood GNU ``timeout``."""
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            index += 1
            break
        if token in {"--help", "--version"}:
            return []
        if token in _TIMEOUT_FLAG_OPTIONS:
            index += 1
            continue
        option, separator, value = token.partition("=")
        if option in {"--kill-after", "--signal"} and separator:
            if not _is_static_env_operand(value):
                return None
            if option == "--kill-after" and _TIMEOUT_DURATION_RE.fullmatch(value) is None:
                return None
            if option == "--signal" and not _is_timeout_signal(value):
                return None
            index += 1
            continue
        if token in _TIMEOUT_VALUE_OPTIONS:
            if index + 1 >= len(tokens):
                return None
            value = tokens[index + 1]
            if not _is_static_env_operand(value):
                return None
            if token in {"-k", "--kill-after"} and _TIMEOUT_DURATION_RE.fullmatch(value) is None:
                return None
            if token in {"-s", "--signal"} and not _is_timeout_signal(value):
                return None
            index += 2
            continue
        if token.startswith("-") and not token.startswith("--"):
            options = token[1:]
            cursor = 0
            consumed_next = False
            while cursor < len(options):
                short_option = options[cursor]
                if short_option == "v":
                    cursor += 1
                    continue
                if short_option not in {"k", "s"}:
                    return None
                value = options[cursor + 1 :]
                if not value:
                    if index + 1 >= len(tokens):
                        return None
                    value = tokens[index + 1]
                    consumed_next = True
                if not _is_static_env_operand(value):
                    return None
                if short_option == "k" and _TIMEOUT_DURATION_RE.fullmatch(value) is None:
                    return None
                if short_option == "s" and not _is_timeout_signal(value):
                    return None
                cursor = len(options)
            index += 2 if consumed_next else 1
            continue
        if token.startswith("-"):
            return None
        break

    if index >= len(tokens):
        return None
    duration = tokens[index]
    if _TIMEOUT_DURATION_RE.fullmatch(duration) is None:
        return None
    index += 1
    if index >= len(tokens):
        return None
    return list(tokens[index:])


def _updated_static_cwd(cwd: Path, operand: str) -> Path:
    """Apply one statically known ``env --chdir`` operand lexically."""
    target = Path(operand)
    if target.is_absolute():
        return target
    return Path(os.path.normpath(str(cwd / target)))


def _wrapper_uncertainty(
    cwd: Path,
    reason: str,
    *,
    cwd_hops: Sequence[Path] = (),
) -> _WrapperResolution:
    return _WrapperResolution(
        (),
        cwd,
        uncertainty=(reason,),
        cwd_hops=tuple(cwd_hops),
    )


def _resolve_shell_wrappers(
    tokens: Sequence[str],
    *,
    cwd: Path = Path("."),
    max_split_depth: int = 8,
) -> _WrapperResolution:
    """Resolve supported shell wrappers without losing their static cwd."""
    remaining = list(tokens)
    split_depth = 0
    cwd_hops: list[Path] = []
    while True:
        while remaining and _ENV_ASSIGNMENT_RE.match(remaining[0]):
            remaining.pop(0)
        if not remaining:
            return _WrapperResolution((), cwd, cwd_hops=tuple(cwd_hops))
        if remaining[0] in {"command", "/usr/bin/command"}:
            stripped = _strip_command_wrapper(remaining)
            if stripped is None:
                return _wrapper_uncertainty(
                    cwd,
                    "unsupported `command` wrapper options",
                    cwd_hops=cwd_hops,
                )
            remaining = stripped
            continue
        if remaining[0] in {"time", "/bin/time", "/usr/bin/time"}:
            stripped = _strip_time_wrapper(remaining)
            if stripped is None:
                return _wrapper_uncertainty(
                    cwd,
                    "unsupported `time` wrapper options",
                    cwd_hops=cwd_hops,
                )
            remaining = stripped
            continue
        if Path(remaining[0]).name == "timeout":
            stripped = _strip_timeout_wrapper(remaining)
            if stripped is None:
                return _wrapper_uncertainty(
                    cwd,
                    "unsupported `timeout` wrapper options or command boundary",
                    cwd_hops=cwd_hops,
                )
            remaining = stripped
            continue
        if remaining[0] not in {"env", "/usr/bin/env"}:
            return _WrapperResolution(
                tuple(remaining),
                cwd,
                cwd_hops=tuple(cwd_hops),
            )

        env_entry_cwd = cwd
        chdir_operand: str | None = None
        index = 1
        while index < len(remaining):
            token = remaining[index]
            if token == "--":
                index += 1
                break
            if token in {"--help", "--version"}:
                return _WrapperResolution((), cwd, cwd_hops=tuple(cwd_hops))
            if token in {"-i", "--ignore-environment", "-"}:
                index += 1
                continue
            if token in {"-u", "--unset"}:
                if (
                    index + 1 >= len(remaining)
                    or _ENV_NAME_RE.fullmatch(remaining[index + 1]) is None
                ):
                    return _wrapper_uncertainty(
                        cwd,
                        "unsupported or dynamic `env` wrapper options",
                        cwd_hops=cwd_hops,
                    )
                index += 2
                continue
            if token.startswith("-u") and token != "-u":
                if _ENV_NAME_RE.fullmatch(token[2:]) is None:
                    return _wrapper_uncertainty(
                        cwd,
                        "unsupported or dynamic `env` wrapper options",
                        cwd_hops=cwd_hops,
                    )
                index += 1
                continue
            if token.startswith("--unset="):
                if _ENV_NAME_RE.fullmatch(token.partition("=")[2]) is None:
                    return _wrapper_uncertainty(
                        cwd,
                        "unsupported or dynamic `env` wrapper options",
                        cwd_hops=cwd_hops,
                    )
                index += 1
                continue
            if token in {"-C", "--chdir"}:
                if index + 1 >= len(remaining) or not _is_static_env_operand(remaining[index + 1]):
                    return _wrapper_uncertainty(
                        cwd,
                        "unsupported or dynamic `env` wrapper options",
                        cwd_hops=cwd_hops,
                    )
                chdir_operand = remaining[index + 1]
                index += 2
                continue
            if token.startswith("-C") and token != "-C":
                operand = token[2:]
                if not _is_static_env_operand(operand):
                    return _wrapper_uncertainty(
                        cwd,
                        "unsupported or dynamic `env` wrapper options",
                        cwd_hops=cwd_hops,
                    )
                chdir_operand = operand
                index += 1
                continue
            if token.startswith("--chdir="):
                operand = token.partition("=")[2]
                if not _is_static_env_operand(operand):
                    return _wrapper_uncertainty(
                        cwd,
                        "unsupported or dynamic `env` wrapper options",
                        cwd_hops=cwd_hops,
                    )
                chdir_operand = operand
                index += 1
                continue

            split_value: str | None = None
            split_end = index + 1
            if token in {"-S", "--split-string"}:
                if index + 1 < len(remaining):
                    split_value = remaining[index + 1]
                    split_end = index + 2
            elif token.startswith("-S") and token != "-S":
                split_value = token[2:]
            elif token.startswith("--split-string="):
                split_value = token.partition("=")[2]
            if split_value is not None:
                if split_depth >= max_split_depth or not _is_static_env_operand(split_value):
                    return _wrapper_uncertainty(
                        cwd,
                        "unsupported or dynamic `env` wrapper options",
                        cwd_hops=cwd_hops,
                    )
                split_tokens = _split_gnu_env_string(split_value)
                if split_tokens is None:
                    return _wrapper_uncertainty(
                        cwd,
                        "unsupported or dynamic `env` wrapper options",
                        cwd_hops=cwd_hops,
                    )
                remaining[index:split_end] = split_tokens
                split_depth += 1
                continue
            if token in {"-S", "--split-string"}:
                return _wrapper_uncertainty(
                    cwd,
                    "unsupported or dynamic `env` wrapper options",
                    cwd_hops=cwd_hops,
                )
            if token.startswith("-"):
                return _wrapper_uncertainty(
                    cwd,
                    "unsupported or dynamic `env` wrapper options",
                    cwd_hops=cwd_hops,
                )
            break

        if chdir_operand is not None:
            cwd = _updated_static_cwd(env_entry_cwd, chdir_operand)
            cwd_hops.append(cwd)
        remaining = remaining[index:]


def _strip_shell_wrappers(
    tokens: list[str],
    *,
    uncertainty: list[str] | None = None,
    cwd: Path = Path("."),
    cwd_out: list[Path] | None = None,
) -> list[str]:
    """Compatibility wrapper returning only the resolved command tokens."""
    resolution = _resolve_shell_wrappers(tokens, cwd=cwd)
    if uncertainty is not None:
        uncertainty.extend(resolution.uncertainty)
    if cwd_out is not None:
        cwd_out.append(resolution.cwd)
    return list(resolution.tokens)


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
    original = list(tokens)
    resolution = _resolve_shell_wrappers(tokens)
    tokens = list(resolution.tokens)
    if resolution.uncertainty:
        while original and _ENV_ASSIGNMENT_RE.match(original[0]):
            original.pop(0)
        return bool(
            original
            and (
                original[0]
                in {
                    "command",
                    "/usr/bin/command",
                    "time",
                    "/bin/time",
                    "/usr/bin/time",
                    "env",
                    "/usr/bin/env",
                }
                or Path(original[0]).name == "timeout"
            )
        )
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


def _nested_shell_payload(
    tokens: Sequence[str],
    *,
    cwd: Path,
) -> _NestedShellPayload | None:
    """Extract a statically knowable shell payload from supported wrappers."""
    resolution = _resolve_shell_wrappers(tokens, cwd=cwd)
    remaining = list(resolution.tokens)
    if not remaining:
        return None

    payload = _shell_c_payload(remaining)
    payload_cwd = resolution.cwd
    if payload is None and Path(remaining[0]).name == "docker":
        docker_command = _docker_exec_command(remaining)
        if docker_command is not None:
            docker_resolution = _resolve_shell_wrappers(
                docker_command,
                cwd=resolution.cwd,
            )
            payload = _shell_c_payload(docker_resolution.tokens)
            payload_cwd = docker_resolution.cwd

    if (
        payload is None
        or not payload.strip()
        or "\x00" in payload
        or "$" in payload
        or "`" in payload
        or "DOC_PLACEHOLDER" in payload
    ):
        return None
    return _NestedShellPayload(payload, payload_cwd)


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
            payload = _nested_shell_payload(
                command.tokens,
                cwd=current.cwd,
            )
            if payload is None:
                continue
            body = payload.body + ("" if payload.body.endswith("\n") else "\n")
            signature = normalize_shell(body).strip()
            if not signature or signature in ancestors:
                continue
            nested = ShellBlock(
                path=current.path,
                line=_source_line(current, offset),
                language=_NESTED_SHELL_LANGUAGE,
                body=body,
                cwd=payload.cwd,
            )
            yield from visit(
                nested,
                depth + 1,
                ancestors | {signature},
            )

    root_signature = normalize_shell(block.body).strip()
    yield from visit(block, 0, frozenset({root_signature}))


def _normalized_path_token(token: str) -> str | None:
    token = token.strip("'\"")
    token = token.split("::", 1)[0]
    token = token.rstrip(".,;:)")
    if (
        not token
        or token == "."
        or token.startswith("/")
        or any(character.isspace() for character in token)
        or "$" in token
        or "*" in token
        or "?" in token
        or "[" in token
        or "{" in token
        or "}" in token
        or "DOC_PLACEHOLDER" in token
    ):
        return None
    return token


def _candidate_repo_path_token(
    token: str,
    *,
    allow_plain: bool = False,
) -> str | None:
    """Return a static relative path token that this checker owns."""
    normalized = _normalized_path_token(token)
    if normalized is None:
        return None
    classification = normalized.removeprefix("./")
    looks_local = classification.startswith(_LOCAL_PREFIXES)
    if allow_plain:
        path = Path(normalized)
        looks_local = looks_local or "/" in normalized or bool(path.suffix)
    if looks_local:
        return normalized
    return None


def _resolved_repo_local_path(
    repo_root: Path,
    cwd: Path,
    token: str,
    *,
    allow_plain: bool = False,
) -> str | None:
    """Resolve a relative input against a static cwd, bounded to the repo."""
    normalized = _candidate_repo_path_token(
        token,
        allow_plain=allow_plain,
    )
    if normalized is None:
        return None

    root = repo_root.resolve()
    base = _resolved_command_cwd(repo_root, cwd)
    resolved = (base / normalized).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return None


def _resolved_command_cwd(repo_root: Path, cwd: Path) -> Path:
    """Return the absolute static cwd used to resolve command operands."""
    root = repo_root.resolve()
    return cwd.resolve() if cwd.is_absolute() else (root / cwd).resolve()


def _external_candidate_path(
    repo_root: Path,
    cwd: Path,
    token: str,
    *,
    allow_plain: bool,
) -> Path | None:
    """Return a static candidate's path when it resolves outside the repo."""
    normalized = _candidate_repo_path_token(
        token,
        allow_plain=allow_plain,
    )
    if normalized is None:
        return None
    root = repo_root.resolve()
    resolved = (_resolved_command_cwd(repo_root, cwd) / normalized).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return resolved
    return None


def _candidate_input_paths(tokens: list[str]) -> list[_InputCandidate]:
    """Return repo-local inputs whose existence is required by the command."""
    tokens = _strip_shell_wrappers(tokens)
    if not tokens:
        return []

    command = tokens[0]
    command_name = Path(command).name
    candidates: list[_InputCandidate] = []

    if command.startswith(("./", "../")):
        candidates.append(_InputCandidate(command, allow_plain=True))

    if command_name in {"bash", "sh"}:
        for token in tokens[1:]:
            if token.startswith("-"):
                continue
            candidates.append(_InputCandidate(token, allow_plain=True))
            break
    if command_name in {"python", "python3"}:
        index = 1
        while index < len(tokens):
            token = tokens[index]
            if token in {"-c", "-m"}:
                break
            if token == "--":
                if index + 1 < len(tokens):
                    candidates.append(_InputCandidate(tokens[index + 1], allow_plain=True))
                break
            if token in {"-W", "-X", "--check-hash-based-pycs"}:
                index += 2
                continue
            if token.startswith("-"):
                index += 1
                continue
            candidates.append(_InputCandidate(token, allow_plain=True))
            break

    if command_name == "find" and len(tokens) > 1:
        candidates.append(_InputCandidate(tokens[1], allow_plain=True))

    if command_name == "cat":
        candidates.extend(_InputCandidate(token, allow_plain=True) for token in tokens[1:])

    if command_name == "rg":
        # Search roots are repo inputs. Quoted patterns and option values do
        # not start with a known repository prefix and are ignored.
        candidates.extend(_InputCandidate(token) for token in tokens[1:])

    is_pytest = command_name in {"pytest", "py.test"}
    if command_name in {"python", "python3"}:
        is_pytest = len(tokens) > 2 and tokens[1:3] == ["-m", "pytest"]
    if is_pytest:
        candidates.extend(_InputCandidate(token, allow_plain=True) for token in tokens[1:])

    results: list[_InputCandidate] = []
    for candidate in candidates:
        if candidate not in results:
            results.append(candidate)
    return results


def check_local_inputs(block: ShellBlock, repo_root: Path) -> list[Finding]:
    """Check commands whose repo-local inputs must already exist."""
    findings: list[Finding] = []
    resolved_root = repo_root.resolve()
    for offset, command in _block_shell_commands(block):
        resolution = _resolve_shell_wrappers(
            command.tokens,
            cwd=block.cwd,
        )
        invalid_cwd = False
        if not resolution.uncertainty:
            for cwd_hop in resolution.cwd_hops:
                command_cwd = _resolved_command_cwd(repo_root, cwd_hop)
                try:
                    displayed_cwd = command_cwd.relative_to(resolved_root).as_posix()
                except ValueError:
                    displayed_cwd = command_cwd.as_posix()
                if command_cwd.exists() and command_cwd.is_dir():
                    continue
                problem = "does not exist" if not command_cwd.exists() else "is not a directory"
                findings.append(
                    Finding(
                        block.path,
                        _source_line(block, offset),
                        f"command working directory {problem}: {displayed_cwd}",
                    )
                )
                invalid_cwd = True
                break
        if invalid_cwd:
            continue
        candidates = [
            *(
                (candidate, resolution.cwd)
                for candidate in _candidate_input_paths(list(resolution.tokens))
            ),
            *(
                (_InputCandidate(token, allow_plain=True), block.cwd)
                for token in command.input_redirections
            ),
        ]
        for candidate, candidate_cwd in candidates:
            allow_plain = candidate.allow_plain and (
                _resolved_command_cwd(repo_root, candidate_cwd) != repo_root.resolve()
            )
            local = _resolved_repo_local_path(
                repo_root,
                candidate_cwd,
                candidate.token,
                allow_plain=allow_plain,
            )
            if local is None:
                external = _external_candidate_path(
                    repo_root,
                    candidate_cwd,
                    candidate.token,
                    allow_plain=allow_plain,
                )
                if external is not None:
                    findings.append(
                        Finding(
                            block.path,
                            _source_line(block, offset),
                            "command input resolves outside repository "
                            f"and cannot be validated: {external}",
                        )
                    )
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
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        operand = _literal_value(node.operand, binding)
        if operand is not _UNRESOLVED:
            return not bool(operand)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        operand = _literal_value(node.operand, binding)
        if isinstance(operand, (int, float, complex)):
            return -operand if isinstance(node.op, ast.USub) else operand
    if isinstance(node, ast.BoolOp):
        values = [_literal_value(value, binding) for value in node.values]
        if any(value is _UNRESOLVED for value in values):
            return _UNRESOLVED
        return (
            all(bool(value) for value in values)
            if isinstance(node.op, ast.And)
            else any(bool(value) for value in values)
        )
    if isinstance(node, ast.Compare):
        left = _literal_value(node.left, binding)
        comparators = [_literal_value(comparator, binding) for comparator in node.comparators]
        if left is _UNRESOLVED or any(value is _UNRESOLVED for value in comparators):
            return _UNRESOLVED
        values = [left, *comparators]
        results: list[bool] = []
        for operator, lhs, rhs in zip(node.ops, values, values[1:]):
            try:
                if isinstance(operator, ast.Eq):
                    result = lhs == rhs
                elif isinstance(operator, ast.NotEq):
                    result = lhs != rhs
                elif isinstance(operator, ast.Is):
                    result = lhs is rhs
                elif isinstance(operator, ast.IsNot):
                    result = lhs is not rhs
                elif isinstance(operator, ast.In):
                    result = lhs in rhs
                elif isinstance(operator, ast.NotIn):
                    result = lhs not in rhs
                elif isinstance(operator, ast.Lt):
                    result = lhs < rhs
                elif isinstance(operator, ast.LtE):
                    result = lhs <= rhs
                elif isinstance(operator, ast.Gt):
                    result = lhs > rhs
                elif isinstance(operator, ast.GtE):
                    result = lhs >= rhs
                else:
                    return _UNRESOLVED
            except (TypeError, ValueError):
                return _UNRESOLVED
            results.append(bool(result))
        return all(results)
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


def _literal_loop_bindings(
    node: ast.For,
    binding: dict[str, object] | None = None,
) -> list[dict[str, object]] | None:
    """Expand a finite loop whose targets and iterable are literal values."""
    if isinstance(node.target, ast.Name):
        target_names = (node.target.id,)
    elif isinstance(node.target, (ast.Tuple, ast.List)) and all(
        isinstance(element, ast.Name) for element in node.target.elts
    ):
        target_names = tuple(element.id for element in node.target.elts)
    else:
        return None
    if not isinstance(node.iter, (ast.Tuple, ast.List)):
        return None

    bindings: list[dict[str, object]] = []
    for item in node.iter.elts:
        if len(target_names) == 1:
            raw_values = (_literal_value(item, binding),)
        elif isinstance(item, (ast.Tuple, ast.List)):
            raw_values = tuple(_literal_value(value, binding) for value in item.elts)
        else:
            return None
        if len(raw_values) != len(target_names) or any(
            value is _UNRESOLVED for value in raw_values
        ):
            return None
        expanded = dict(binding or {})
        expanded.update(zip(target_names, raw_values))
        bindings.append(expanded)
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
    argparse_module_imports: list[tuple[ast.AST, str]] = []
    argparse_constructor_imports: list[tuple[ast.AST, str]] = []
    context_factory_function_scopes: set[tuple[tuple[str, str, int], ...]] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            argparse_module_imports.extend(
                (node, alias.asname or alias.name)
                for alias in node.names
                if alias.name == "argparse"
            )
        elif isinstance(node, ast.ImportFrom) and node.module == "argparse":
            argparse_constructor_imports.extend(
                (node, alias.asname or alias.name)
                for alias in node.names
                if alias.name == "ArgumentParser"
            )

    argparse_evidence = bool(argparse_module_imports or argparse_constructor_imports)

    scope_by_node: dict[int, tuple[tuple[str, str, int], ...]] = {}
    function_scopes: dict[
        tuple[tuple[tuple[str, str, int], ...], str],
        tuple[tuple[str, str, int], ...],
    ] = {}
    function_parameters: dict[
        tuple[tuple[str, str, int], ...],
        tuple[str, ...],
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
    parent_by_node: dict[int, ast.AST] = {
        id(child): parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)
    }

    def is_conditional_binding(node: ast.AST) -> bool:
        """Return whether a binding may not execute before a later use."""
        current = node
        while (parent := parent_by_node.get(id(current))) is not None:
            if isinstance(
                parent,
                (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                return False
            if isinstance(
                parent,
                (
                    ast.If,
                    ast.For,
                    ast.AsyncFor,
                    ast.While,
                    *_TRY_NODE_TYPES,
                    ast.With,
                    ast.AsyncWith,
                    ast.Match,
                    ast.IfExp,
                    ast.BoolOp,
                ),
            ):
                return True
            current = parent
        return False

    def direct_statement_block(
        parent: ast.AST,
        child: ast.AST,
    ) -> tuple[list[ast.stmt], int] | None:
        if not isinstance(child, ast.stmt):
            return None
        for _field, value in ast.iter_fields(parent):
            if isinstance(value, list) and child in value:
                statements = [statement for statement in value if isinstance(statement, ast.stmt)]
                if len(statements) == len(value):
                    return statements, statements.index(child)
        return None

    def statement_abrupt_state(
        statement: ast.stmt,
        binding: dict[str, object],
    ) -> bool | None:
        if isinstance(
            statement,
            (ast.Raise, ast.Return, ast.Break, ast.Continue),
        ):
            return True
        if isinstance(statement, ast.If):
            condition = _literal_value(statement.test, binding)
            if condition is _UNRESOLVED:
                body = block_abrupt_state(statement.body, binding)
                orelse = block_abrupt_state(statement.orelse, binding)
                if body is True and orelse is True:
                    return True
                if body is False and orelse is False:
                    return False
                return None
            selected = statement.body if bool(condition) else statement.orelse
            return block_abrupt_state(selected, binding)
        if isinstance(statement, _TRY_NODE_TYPES):
            final_state = block_abrupt_state(
                statement.finalbody,
                binding,
            )
            if final_state is not False:
                return final_state
        return False

    def block_abrupt_state(
        statements: list[ast.stmt],
        binding: dict[str, object],
    ) -> bool | None:
        uncertain = False
        for statement in statements:
            state = statement_abrupt_state(statement, binding)
            if state is True:
                return True
            if state is None:
                uncertain = True
        return None if uncertain else False

    def statement_may_raise(statement: ast.stmt) -> bool:
        if isinstance(statement, ast.Pass):
            return False
        if isinstance(statement, ast.Raise):
            return True
        if isinstance(statement, (ast.Break, ast.Continue, ast.Return)):
            return False
        return any(
            isinstance(descendant, (ast.Call, ast.Await, ast.Yield, ast.YieldFrom))
            for descendant in ast.walk(statement)
        )

    def pattern_match_state(
        pattern: ast.pattern,
        value: object,
        binding: dict[str, object],
    ) -> bool | None:
        if isinstance(pattern, ast.MatchValue):
            expected = _literal_value(pattern.value, binding)
            return None if expected is _UNRESOLVED else value == expected
        if isinstance(pattern, ast.MatchSingleton):
            return value is pattern.value
        if isinstance(pattern, ast.MatchAs):
            if pattern.pattern is None:
                return True
            return pattern_match_state(pattern.pattern, value, binding)
        if isinstance(pattern, ast.MatchOr):
            states = [
                pattern_match_state(candidate, value, binding) for candidate in pattern.patterns
            ]
            if any(state is True for state in states):
                return True
            return None if any(state is None for state in states) else False
        if isinstance(pattern, ast.MatchSequence) and isinstance(
            value,
            (list, tuple),
        ):
            if len(pattern.patterns) != len(value):
                return False
            states = [
                pattern_match_state(candidate, item, binding)
                for candidate, item in zip(pattern.patterns, value)
            ]
            if any(state is False for state in states):
                return False
            return None if any(state is None for state in states) else True
        return None

    def selected_match_case(
        node: ast.Match,
        binding: dict[str, object],
    ) -> int | object | None:
        subject = _literal_value(node.subject, binding)
        if subject is _UNRESOLVED:
            return _UNRESOLVED
        for index, case in enumerate(node.cases):
            matched = pattern_match_state(case.pattern, subject, binding)
            if matched is None:
                return _UNRESOLVED
            if not matched:
                continue
            if case.guard is not None:
                guard = _literal_value(case.guard, binding)
                if guard is _UNRESOLVED:
                    return _UNRESOLVED
                if not bool(guard):
                    continue
            return index
        return None

    def declaration_contexts(
        node: ast.AST,
    ) -> tuple[list[dict[str, object]], bool]:
        """Return literal execution contexts and whether reachability is unknown."""
        ancestry: list[tuple[ast.AST, ast.AST]] = []
        current = node
        while (parent := parent_by_node.get(id(current))) is not None:
            ancestry.append((parent, current))
            if isinstance(
                parent,
                (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                break
            current = parent

        contexts: list[dict[str, object]] = [{}]
        uncertain = False
        for parent, child in reversed(ancestry):
            direct_block = direct_statement_block(parent, child)
            if direct_block is not None:
                statements, child_index = direct_block
                reachable: list[dict[str, object]] = []
                for binding in contexts:
                    state = block_abrupt_state(
                        statements[:child_index],
                        binding,
                    )
                    if state is True:
                        continue
                    if state is None:
                        uncertain = True
                    reachable.append(binding)
                contexts = reachable
                if not contexts:
                    continue

            if isinstance(parent, ast.For):
                if child in parent.body:
                    expanded: list[dict[str, object]] = []
                    for binding in contexts:
                        loop_bindings = _literal_loop_bindings(parent, binding)
                        if loop_bindings is None:
                            uncertain = True
                            expanded.append(binding)
                        else:
                            expanded.extend(loop_bindings)
                    contexts = expanded
                elif child in parent.orelse:
                    break_states = [
                        statement_abrupt_state(statement, binding)
                        for binding in contexts
                        for statement in parent.body
                        if isinstance(statement, ast.Break)
                    ]
                    if any(state is True for state in break_states):
                        contexts = []
                    elif any(state is None for state in break_states):
                        uncertain = True
                continue

            if isinstance(parent, (ast.If, ast.IfExp)):
                body = parent.body
                orelse = parent.orelse
                in_body = child is body or (isinstance(body, list) and child in body)
                in_orelse = child is orelse or (isinstance(orelse, list) and child in orelse)
                if not (in_body or in_orelse):
                    continue
                reachable = []
                for binding in contexts:
                    condition = _literal_value(parent.test, binding)
                    if condition is _UNRESOLVED:
                        uncertain = True
                        reachable.append(binding)
                    elif bool(condition) == in_body:
                        reachable.append(binding)
                contexts = reachable
                continue

            if isinstance(parent, ast.While):
                in_body = child in parent.body
                in_orelse = child in parent.orelse
                if not (in_body or in_orelse):
                    continue
                reachable = []
                for binding in contexts:
                    condition = _literal_value(parent.test, binding)
                    if condition is _UNRESOLVED:
                        uncertain = True
                        reachable.append(binding)
                    elif in_body and bool(condition):
                        reachable.append(binding)
                    elif in_orelse and not bool(condition):
                        reachable.append(binding)
                contexts = reachable
                continue

            if isinstance(parent, _TRY_NODE_TYPES):
                if child in parent.body:
                    prior = parent.body[: parent.body.index(child)]
                    if any(statement_may_raise(statement) for statement in prior):
                        uncertain = True
                elif child in parent.handlers:
                    if not any(statement_may_raise(statement) for statement in parent.body):
                        contexts = []
                    else:
                        uncertain = True
                elif child in parent.orelse:
                    if any(statement_may_raise(statement) for statement in parent.body):
                        uncertain = True
                continue

            if isinstance(parent, ast.Match) and child in parent.cases:
                case_index = parent.cases.index(child)
                reachable = []
                for binding in contexts:
                    selected = selected_match_case(parent, binding)
                    if selected is _UNRESOLVED:
                        uncertain = True
                        reachable.append(binding)
                    elif selected == case_index:
                        reachable.append(binding)
                contexts = reachable

        return contexts, uncertain

    def has_enclosing_loop(node: ast.AST) -> bool:
        current = node
        while (parent := parent_by_node.get(id(current))) is not None:
            if isinstance(
                parent,
                (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                return False
            if isinstance(parent, (ast.For, ast.AsyncFor)):
                return True
            current = parent
        return False

    def nearest_enclosing_for(node: ast.AST) -> ast.For | None:
        current = node
        while (parent := parent_by_node.get(id(current))) is not None:
            if isinstance(
                parent,
                (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                return None
            if isinstance(parent, ast.For):
                return parent
            current = parent
        return None

    uncertain_parser_declaration_scopes: set[tuple[tuple[str, str, int], ...]] = set()

    def reachable_declaration_contexts(
        node: ast.AST,
    ) -> list[dict[str, object]]:
        contexts, uncertain = declaration_contexts(node)
        if uncertain:
            uncertain_parser_declaration_scopes.add(scope_by_node[id(node)])
        return contexts

    argparse_module_identity = object()
    argparse_constructor_identity = object()
    other_identity = object()
    binding_events: dict[
        tuple[tuple[tuple[str, str, int], ...], str],
        list[tuple[int, int, object]],
    ] = {}

    def binding_position(node: ast.AST) -> tuple[int, int]:
        return (
            node.end_lineno if node.end_lineno is not None else node.lineno,
            (node.end_col_offset if node.end_col_offset is not None else node.col_offset),
        )

    def record_binding(
        name: str,
        scope: tuple[tuple[str, str, int], ...],
        node: ast.AST,
        value: object,
    ) -> None:
        if not name.isidentifier():
            return
        line, column = binding_position(node)
        binding_events.setdefault((scope, name), []).append((line, column, value))

    for node in ast.walk(tree):
        scope = scope_by_node[id(node)]
        binding_contexts, conditional = declaration_contexts(node)
        if not binding_contexts:
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.partition(".")[0]
                identity = (
                    argparse_module_identity
                    if alias.name == "argparse" and not conditional
                    else other_identity
                )
                record_binding(name, scope, node, identity)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    continue
                name = alias.asname or alias.name
                identity = (
                    argparse_constructor_identity
                    if (
                        node.module == "argparse"
                        and alias.name == "ArgumentParser"
                        and not conditional
                    )
                    else other_identity
                )
                record_binding(name, scope, node, identity)
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            record_binding(node.name, scope, node, other_identity)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function_scope = (
                    *scope,
                    ("function", node.name, node.lineno),
                )
                parameters = [
                    *node.args.posonlyargs,
                    *node.args.args,
                    *node.args.kwonlyargs,
                ]
                if node.args.vararg is not None:
                    parameters.append(node.args.vararg)
                if node.args.kwarg is not None:
                    parameters.append(node.args.kwarg)
                for parameter in parameters:
                    binding_events.setdefault(
                        (function_scope, parameter.arg),
                        [],
                    ).append((-1, -1, other_identity))

    def assignment_targets(node: ast.Assign | ast.AnnAssign) -> list[ast.Name]:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return [target for target in targets if isinstance(target, ast.Name)]

    for node in assignment_nodes:
        binding_contexts, conditional = declaration_contexts(node)
        if not binding_contexts:
            continue
        value: object = other_identity if conditional or node.value is None else node.value
        for target in assignment_targets(node):
            record_binding(
                target.id,
                scope_by_node[id(node)],
                node,
                value,
            )

    for node in (
        candidate
        for candidate in ast.walk(tree)
        if isinstance(candidate, ast.NamedExpr) and isinstance(candidate.target, ast.Name)
    ):
        binding_contexts, conditional = declaration_contexts(node)
        if not binding_contexts:
            continue
        record_binding(
            node.target.id,
            scope_by_node[id(node)],
            node,
            (other_identity if conditional else node.value),
        )

    class_identity = object()
    function_identity = object()
    instance_identity = object()
    parameter_identity = object()
    ambiguous_identity = object()
    identity_events: dict[
        tuple[tuple[tuple[str, str, int], ...], str],
        list[tuple[int, int, object]],
    ] = {}
    opaque_identity_barriers: dict[
        tuple[tuple[str, str, int], ...],
        list[tuple[int, int]],
    ] = {}
    function_opaque_default_parameters: dict[
        tuple[tuple[str, str, int], ...],
        dict[str, ast.AST],
    ] = {}
    definition_identity_scopes: dict[
        tuple[
            tuple[tuple[str, str, int], ...],
            str,
            tuple[int, int],
        ],
        tuple[tuple[str, str, int], ...],
    ] = {}
    opaque_identity_event_keys: set[
        tuple[
            tuple[tuple[str, str, int], ...],
            str,
            tuple[int, int],
        ]
    ] = set()

    def record_identity_binding(
        name: str,
        scope: tuple[tuple[str, str, int], ...],
        node: ast.AST,
        value: object,
    ) -> None:
        if not name.isidentifier():
            return
        line, column = binding_position(node)
        identity_events.setdefault((scope, name), []).append((line, column, value))

    def record_opaque_identity_barrier(
        scope: tuple[tuple[str, str, int], ...],
        node: ast.AST,
    ) -> None:
        opaque_identity_barriers.setdefault(scope, []).append((node.lineno, node.col_offset))

    decorator_binding_events: dict[
        tuple[tuple[tuple[str, str, int], ...], str],
        list[tuple[int, int, str | None]],
    ] = {}
    decorator_attribute_mutations: dict[
        tuple[tuple[tuple[str, str, int], ...], str],
        list[tuple[int, int]],
    ] = {}

    def record_decorator_binding(
        scope: tuple[tuple[str, str, int], ...],
        name: str,
        node: ast.AST,
        provenance: str | None,
    ) -> None:
        decorator_binding_events.setdefault((scope, name), []).append(
            (*binding_position(node), provenance)
        )

    def decorator_visible_scopes(
        scope: tuple[tuple[str, str, int], ...],
    ) -> Iterable[tuple[tuple[str, str, int], ...]]:
        for length in range(len(scope), -1, -1):
            candidate = scope[:length]
            if candidate and candidate[-1][0] == "class" and candidate != scope:
                continue
            yield candidate

    def decorator_namespace_owner(expression: ast.AST) -> str | None:
        if isinstance(expression, ast.Attribute) and expression.attr == "__dict__":
            return _expression_reference(expression.value)
        if (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Name)
            and expression.func.id == "vars"
            and len(expression.args) == 1
            and not expression.keywords
        ):
            return _expression_reference(expression.args[0])
        return None

    def record_decorator_mutation(
        scope: tuple[tuple[str, str, int], ...],
        owner: str,
        attribute: object,
        node: ast.AST,
    ) -> None:
        suffix = attribute if isinstance(attribute, str) else "*"
        decorator_attribute_mutations.setdefault(
            (scope, f"{owner}.{suffix}"),
            [],
        ).append(binding_position(node))

    decorator_function_definitions: dict[
        tuple[tuple[tuple[str, str, int], ...], str],
        list[
            tuple[
                tuple[int, int],
                tuple[tuple[str, str, int], ...],
            ]
        ],
    ] = {}
    decorator_global_writes: dict[
        tuple[tuple[str, str, int], ...],
        set[str],
    ] = {}
    for candidate in ast.walk(tree):
        if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scope = scope_by_node[id(candidate)]
            nested_scope = (
                *scope,
                ("function", candidate.name, candidate.lineno),
            )
            decorator_function_definitions.setdefault(
                (scope, candidate.name),
                [],
            ).append((binding_position(candidate), nested_scope))
        elif isinstance(candidate, ast.Global):
            decorator_global_writes.setdefault(
                scope_by_node[id(candidate)],
                set(),
            ).update(candidate.names)

    for candidate in ast.walk(tree):
        scope = scope_by_node[id(candidate)]
        contexts, uncertain = declaration_contexts(candidate)
        if not contexts:
            continue
        if isinstance(candidate, ast.Import):
            for alias in candidate.names:
                record_decorator_binding(
                    scope,
                    alias.asname or alias.name.partition(".")[0],
                    candidate,
                    None if uncertain else alias.name,
                )
        elif isinstance(candidate, ast.ImportFrom):
            for alias in candidate.names:
                if alias.name != "*":
                    record_decorator_binding(
                        scope,
                        alias.asname or alias.name,
                        candidate,
                        (
                            f"{candidate.module}.{alias.name}"
                            if candidate.module and not uncertain
                            else None
                        ),
                    )
        elif isinstance(
            candidate,
            (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            record_decorator_binding(
                scope,
                candidate.name,
                candidate,
                None,
            )
        elif isinstance(candidate, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = candidate.targets if isinstance(candidate, ast.Assign) else [candidate.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    record_decorator_binding(
                        scope,
                        target.id,
                        candidate,
                        None,
                    )
                elif (
                    isinstance(target, ast.Subscript)
                    and (owner := decorator_namespace_owner(target.value)) is not None
                ):
                    record_decorator_mutation(
                        scope,
                        owner,
                        _literal_value(target.slice),
                        candidate,
                    )
                elif (reference := _expression_reference(target)) is not None:
                    decorator_attribute_mutations.setdefault(
                        (scope, reference),
                        [],
                    ).append(binding_position(candidate))
        elif isinstance(candidate, ast.Delete):
            for target in candidate.targets:
                if isinstance(target, ast.Name):
                    record_decorator_binding(
                        scope,
                        target.id,
                        candidate,
                        None,
                    )
                elif (reference := _expression_reference(target)) is not None:
                    decorator_attribute_mutations.setdefault(
                        (scope, reference),
                        [],
                    ).append(binding_position(candidate))
        elif (
            isinstance(candidate, ast.Call)
            and isinstance(candidate.func, ast.Name)
            and candidate.func.id == "setattr"
            and len(candidate.args) >= 2
            and isinstance((owner := _expression_reference(candidate.args[0])), str)
        ):
            record_decorator_mutation(
                scope,
                owner,
                _literal_value(candidate.args[1]),
                candidate,
            )
        elif (
            isinstance(candidate, ast.Call)
            and isinstance(candidate.func, ast.Attribute)
            and candidate.func.attr == "update"
            and (owner := decorator_namespace_owner(candidate.func.value)) is not None
        ):
            recorded = False
            if candidate.args and isinstance(candidate.args[0], ast.Dict):
                for key in candidate.args[0].keys:
                    record_decorator_mutation(
                        scope,
                        owner,
                        _literal_value(key),
                        candidate,
                    )
                    recorded = True
            for keyword in candidate.keywords:
                record_decorator_mutation(
                    scope,
                    owner,
                    keyword.arg,
                    candidate,
                )
                recorded = True
            if not recorded or len(candidate.args) > 1:
                record_decorator_mutation(
                    scope,
                    owner,
                    _UNRESOLVED,
                    candidate,
                )

    for call in (
        candidate
        for candidate in ast.walk(tree)
        if isinstance(candidate, ast.Call) and isinstance(candidate.func, ast.Name)
    ):
        contexts, _uncertain = declaration_contexts(call)
        if not contexts:
            continue
        call_scope = scope_by_node[id(call)]
        before = (call.lineno, call.col_offset)
        function_scope: tuple[tuple[str, str, int], ...] | None = None
        for candidate_scope in decorator_visible_scopes(call_scope):
            definitions = [
                event
                for event in decorator_function_definitions.get(
                    (candidate_scope, call.func.id),
                    (),
                )
                if event[0] < before
            ]
            if definitions:
                function_scope = definitions[-1][1]
                break
        if function_scope is None:
            continue
        for name in decorator_global_writes.get(function_scope, ()):
            record_decorator_binding(
                (),
                name,
                call,
                None,
            )

    for events in decorator_binding_events.values():
        events.sort(key=lambda event: event[:2])
    for events in decorator_attribute_mutations.values():
        events.sort()

    def decorator_name_provenance(
        name: str,
        scope: tuple[tuple[str, str, int], ...],
        before: tuple[int, int],
    ) -> str | None:
        for candidate_scope in decorator_visible_scopes(scope):
            prior = [
                event
                for event in decorator_binding_events.get(
                    (candidate_scope, name),
                    (),
                )
                if event[:2] < before
            ]
            if prior:
                return prior[-1][2]
        if name in {"classmethod", "property", "staticmethod"}:
            return f"builtins.{name}"
        return None

    invalidated_decorator_provenance = object()

    def decorator_resolved_provenance(
        expression: ast.AST,
        scope: tuple[tuple[str, str, int], ...],
    ) -> object:
        target = expression.func if isinstance(expression, ast.Call) else expression
        reference = _expression_reference(target)
        if reference is None:
            return None
        before = (expression.lineno, expression.col_offset)
        if "." not in reference:
            provenance = decorator_name_provenance(
                reference,
                scope,
                before,
            )
        else:
            base, attribute = reference.split(".", 1)
            base_provenance = decorator_name_provenance(
                base,
                scope,
                before,
            )
            if base_provenance is None:
                return None
            for candidate_scope in decorator_visible_scopes(scope):
                mutations = decorator_attribute_mutations.get(
                    (candidate_scope, reference),
                    (),
                )
                if any(position < before for position in mutations):
                    return invalidated_decorator_provenance
            provenance = f"{base_provenance}.{attribute}"
        if provenance is None or "." not in provenance:
            return None
        provenance_base, provenance_attribute = provenance.split(".", 1)
        for candidate_scope in decorator_visible_scopes(scope):
            references = {provenance}
            for (
                event_scope,
                name,
            ), events in decorator_binding_events.items():
                if event_scope != candidate_scope:
                    continue
                prior = [event for event in events if event[:2] < before]
                if prior and prior[-1][2] == provenance_base:
                    references.add(f"{name}.{provenance_attribute}")
            if any(
                any(
                    position < before
                    for mutation_reference in (
                        candidate_reference,
                        f"{candidate_reference.split('.', 1)[0]}.*",
                    )
                    for position in decorator_attribute_mutations.get(
                        (candidate_scope, mutation_reference),
                        (),
                    )
                )
                for candidate_reference in references
            ):
                return invalidated_decorator_provenance
        return provenance

    def decorator_preserves_identity_graph(
        expression: ast.AST,
        scope: tuple[tuple[str, str, int], ...],
    ) -> bool:
        return decorator_resolved_provenance(expression, scope) in {
            "builtins.classmethod",
            "builtins.property",
            "builtins.staticmethod",
            "contextlib.contextmanager",
            "dataclasses.dataclass",
            "functools.cache",
            "functools.cached_property",
            "functools.lru_cache",
        }

    for node in ast.walk(tree):
        scope = scope_by_node[id(node)]
        binding_contexts, conditional = declaration_contexts(node)
        if not binding_contexts:
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                record_identity_binding(
                    alias.asname or alias.name.partition(".")[0],
                    scope,
                    node,
                    other_identity,
                )
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    record_identity_binding(
                        alias.asname or alias.name,
                        scope,
                        node,
                        other_identity,
                    )
        elif isinstance(node, ast.ClassDef):
            nested_scope = (*scope, ("class", node.name, node.lineno))
            definition_identity_scopes[(scope, node.name, binding_position(node))] = nested_scope
            opaque_decorators = [
                decorator
                for decorator in node.decorator_list
                if not decorator_preserves_identity_graph(decorator, scope)
            ]
            if opaque_decorators:
                record_opaque_identity_barrier(scope, node)
            record_identity_binding(
                node.name,
                scope,
                node,
                (
                    ambiguous_identity
                    if conditional or opaque_decorators
                    else (class_identity, nested_scope)
                ),
            )
            if opaque_decorators:
                opaque_identity_event_keys.add((scope, node.name, binding_position(node)))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nested_scope = (*scope, ("function", node.name, node.lineno))
            definition_identity_scopes[(scope, node.name, binding_position(node))] = nested_scope
            if any(
                decorator_resolved_provenance(decorator, scope) == "contextlib.contextmanager"
                for decorator in node.decorator_list
            ):
                context_factory_function_scopes.add(nested_scope)
            opaque_decorators = [
                decorator
                for decorator in node.decorator_list
                if not decorator_preserves_identity_graph(decorator, scope)
            ]
            defaults = [
                *node.args.defaults,
                *(default for default in node.args.kw_defaults if default is not None),
            ]
            opaque_defaults = [
                default for default in defaults if _literal_value(default) is _UNRESOLVED
            ]
            if opaque_decorators:
                record_opaque_identity_barrier(scope, node)
            if opaque_defaults:
                positional = [
                    *node.args.posonlyargs,
                    *node.args.args,
                ]
                opaque_by_name = {
                    argument.arg: default
                    for argument, default in zip(
                        positional[-len(node.args.defaults) :],
                        node.args.defaults,
                    )
                    if _literal_value(default) is _UNRESOLVED
                }
                opaque_by_name.update(
                    {
                        argument.arg: default
                        for argument, default in zip(
                            node.args.kwonlyargs,
                            node.args.kw_defaults,
                        )
                        if (default is not None and _literal_value(default) is _UNRESOLVED)
                    }
                )
                function_opaque_default_parameters[nested_scope] = opaque_by_name
            record_identity_binding(
                node.name,
                scope,
                node,
                (
                    ambiguous_identity
                    if conditional or opaque_decorators
                    else (function_identity, nested_scope)
                ),
            )
            if opaque_decorators:
                opaque_identity_event_keys.add((scope, node.name, binding_position(node)))
            parameters = [
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            ]
            if node.args.vararg is not None:
                parameters.append(node.args.vararg)
            if node.args.kwarg is not None:
                parameters.append(node.args.kwarg)
            for parameter in parameters:
                identity_events.setdefault(
                    (nested_scope, parameter.arg),
                    [],
                ).append(
                    (
                        -1,
                        -1,
                        (
                            parameter_identity,
                            nested_scope,
                            parameter.arg,
                        ),
                    )
                )

    for node in assignment_nodes:
        binding_contexts, conditional = declaration_contexts(node)
        if not binding_contexts:
            continue
        value = ambiguous_identity if conditional or node.value is None else node.value
        for target in assignment_targets(node):
            record_identity_binding(
                target.id,
                scope_by_node[id(node)],
                node,
                value,
            )

    for node in (
        candidate
        for candidate in ast.walk(tree)
        if isinstance(candidate, ast.NamedExpr) and isinstance(candidate.target, ast.Name)
    ):
        binding_contexts, conditional = declaration_contexts(node)
        if not binding_contexts:
            continue
        record_identity_binding(
            node.target.id,
            scope_by_node[id(node)],
            node,
            (ambiguous_identity if conditional else node.value),
        )

    def deleted_names(target: ast.AST) -> Iterable[str]:
        if isinstance(target, ast.Name):
            yield target.id
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                yield from deleted_names(element)

    for node in (candidate for candidate in ast.walk(tree) if isinstance(candidate, ast.Delete)):
        scope = scope_by_node[id(node)]
        if scope:
            continue
        names = [name for target in node.targets for name in deleted_names(target)]
        if names:
            for name in names:
                record_identity_binding(
                    name,
                    scope,
                    node,
                    ambiguous_identity,
                )
                opaque_identity_event_keys.add((scope, name, binding_position(node)))
                record_binding(
                    name,
                    scope,
                    node,
                    ambiguous_identity,
                )
        else:
            record_opaque_identity_barrier(scope, node)

    for events in binding_events.values():
        events.sort(key=lambda event: event[:2])
    for events in identity_events.values():
        events.sort(key=lambda event: event[:2])
    for barriers in opaque_identity_barriers.values():
        barriers.sort()

    global_names_by_scope: dict[
        tuple[tuple[str, str, int], ...],
        set[str],
    ] = {}
    nonlocal_names_by_scope: dict[
        tuple[tuple[str, str, int], ...],
        set[str],
    ] = {}
    written_names_by_scope: dict[
        tuple[tuple[str, str, int], ...],
        set[str],
    ] = {}
    for node in ast.walk(tree):
        scope = scope_by_node[id(node)]
        if isinstance(node, ast.Global):
            global_names_by_scope.setdefault(scope, set()).update(node.names)
        elif isinstance(node, ast.Nonlocal):
            nonlocal_names_by_scope.setdefault(scope, set()).update(node.names)
        elif isinstance(node, ast.Name) and isinstance(
            node.ctx,
            (ast.Store, ast.Del),
        ):
            written_names_by_scope.setdefault(scope, set()).add(node.id)

    function_external_effects: dict[
        tuple[tuple[str, str, int], ...],
        tuple[frozenset[str], frozenset[str]],
    ] = {}
    for scope in global_names_by_scope.keys() | nonlocal_names_by_scope.keys():
        written = written_names_by_scope.get(scope, set())
        global_writes = frozenset(global_names_by_scope.get(scope, set()) & written)
        nonlocal_writes = frozenset(nonlocal_names_by_scope.get(scope, set()) & written)
        if global_writes or nonlocal_writes:
            function_external_effects[scope] = (
                global_writes,
                nonlocal_writes,
            )

    def visible_scopes(
        scope: tuple[tuple[str, str, int], ...],
    ) -> Iterable[tuple[tuple[str, str, int], ...]]:
        for length in range(len(scope), -1, -1):
            candidate = scope[:length]
            # A method does not close over its class namespace.
            if candidate and candidate[-1][0] == "class" and candidate != scope:
                continue
            yield candidate

    def tagged_identity(value: object, tag: object) -> bool:
        return isinstance(value, tuple) and bool(value) and value[0] is tag

    def direct_named_identity(
        name: str,
        scope: tuple[tuple[str, str, int], ...],
        before: tuple[int, int],
        root_before: tuple[int, int] | None,
    ) -> object:
        for candidate_scope in visible_scopes(scope):
            candidate_before = (
                root_before if candidate_scope == () and root_before is not None else before
            )
            prior = [
                event
                for event in identity_events.get(
                    (candidate_scope, name),
                    (),
                )
                if event[:2] < candidate_before
            ]
            if prior:
                value = prior[-1][2]
                return (
                    value
                    if (
                        tagged_identity(value, class_identity)
                        or tagged_identity(value, function_identity)
                    )
                    else other_identity
                )
            if (
                candidate_scope
                and candidate_scope[-1][0] == "function"
                and (candidate_scope, name) in identity_events
            ):
                return other_identity
        return other_identity

    def resolve_identity_name(
        name: str,
        scope: tuple[tuple[str, str, int], ...],
        before: tuple[int, int],
        root_before: tuple[int, int] | None = None,
        seen: frozenset[
            tuple[
                tuple[tuple[str, str, int], ...],
                str,
                tuple[int, int],
            ]
        ] = frozenset(),
    ) -> object:
        for candidate_scope in visible_scopes(scope):
            events = identity_events.get((candidate_scope, name), ())
            candidate_before = (
                root_before if candidate_scope == () and root_before is not None else before
            )
            prior = [event for event in events if event[:2] < candidate_before]
            prior_barriers = [
                position
                for position in opaque_identity_barriers.get(
                    candidate_scope,
                    (),
                )
                if position < candidate_before
            ]
            if prior_barriers and (not prior or prior_barriers[-1] > prior[-1][:2]):
                return ambiguous_identity
            if prior:
                line, column, value = prior[-1]
                event_key = (candidate_scope, name, (line, column))
                if event_key in seen:
                    return ambiguous_identity
                if value is other_identity or value is ambiguous_identity:
                    return value
                if any(
                    tagged_identity(value, tag)
                    for tag in (
                        class_identity,
                        function_identity,
                        instance_identity,
                        parameter_identity,
                    )
                ):
                    return value
                if isinstance(value, ast.AST):
                    if isinstance(value, ast.Name):
                        direct = direct_named_identity(
                            value.id,
                            candidate_scope,
                            (line, column),
                            ((line, column) if candidate_scope == () else root_before),
                        )
                        if direct is not other_identity:
                            return direct
                    return resolve_identity_expression(
                        value,
                        candidate_scope,
                        (line, column),
                        ((line, column) if candidate_scope == () else root_before),
                        seen | {event_key},
                    )
                return other_identity
            if events and candidate_scope and candidate_scope[-1][0] == "function":
                return other_identity
        return other_identity

    def resolve_identity_expression(
        expression: ast.AST,
        scope: tuple[tuple[str, str, int], ...],
        before: tuple[int, int] | None = None,
        root_before: tuple[int, int] | None = None,
        seen: frozenset[
            tuple[
                tuple[tuple[str, str, int], ...],
                str,
                tuple[int, int],
            ]
        ] = frozenset(),
    ) -> object:
        if before is None:
            before = (expression.lineno, expression.col_offset)
        if isinstance(expression, ast.Name):
            return resolve_identity_name(
                expression.id,
                scope,
                before,
                root_before,
                seen,
            )
        if isinstance(expression, ast.NamedExpr):
            return resolve_identity_expression(
                expression.value,
                scope,
                before,
                root_before,
                seen,
            )
        if isinstance(expression, ast.IfExp):
            condition = _literal_value(expression.test)
            if condition is not _UNRESOLVED:
                selected = expression.body if bool(condition) else expression.orelse
                return resolve_identity_expression(
                    selected,
                    scope,
                    before,
                    root_before,
                    seen,
                )
            body = resolve_identity_expression(
                expression.body,
                scope,
                before,
                root_before,
                seen,
            )
            other = resolve_identity_expression(
                expression.orelse,
                scope,
                before,
                root_before,
                seen,
            )
            return body if body == other else ambiguous_identity
        if isinstance(expression, ast.Attribute):
            owner_scope: tuple[tuple[str, str, int], ...] | None = None
            if isinstance(expression.value, ast.Name) and expression.value.id in {"self", "cls"}:
                lexical_class = class_scope(scope)
                if lexical_class and lexical_class[-1][0] == "class":
                    owner_scope = lexical_class
            if owner_scope is None:
                owner = resolve_identity_expression(
                    expression.value,
                    scope,
                    before,
                    root_before,
                    seen,
                )
                if tagged_identity(owner, class_identity) or tagged_identity(
                    owner,
                    instance_identity,
                ):
                    owner_scope = owner[1]
                elif owner is ambiguous_identity:
                    return ambiguous_identity
            if owner_scope is not None:
                method_scope = function_scopes.get((owner_scope, expression.attr))
                if isinstance(method_scope, tuple):
                    return function_identity, method_scope
            return other_identity
        if isinstance(expression, ast.Call):
            if (
                isinstance(expression.func, ast.Name)
                and expression.func.id == "super"
                and not any(
                    event[:2] < (expression.lineno, expression.col_offset)
                    for candidate_scope in visible_scopes(scope)
                    for event in identity_events.get(
                        (candidate_scope, "super"),
                        (),
                    )
                )
            ):
                # ``super`` is interpreted by the dedicated runtime-MRO
                # traversal below; it is not an opaque user factory result.
                return other_identity
            constructor = resolve_identity_expression(
                expression.func,
                scope,
                (expression.lineno, expression.col_offset),
                root_before,
                seen,
            )
            if tagged_identity(constructor, class_identity):
                return (
                    instance_identity,
                    constructor[1],
                    (
                        scope,
                        expression.lineno,
                        expression.col_offset,
                    ),
                )
            if constructor is ambiguous_identity:
                return ambiguous_identity
            return ambiguous_identity
        return other_identity

    class_base_scopes: dict[
        tuple[tuple[str, str, int], ...],
        tuple[tuple[tuple[str, str, int], ...], ...],
    ] = {}
    uncertain_class_bases: set[tuple[tuple[str, str, int], ...]] = set()
    for node in (candidate for candidate in ast.walk(tree) if isinstance(candidate, ast.ClassDef)):
        scope = scope_by_node[id(node)]
        resolved_class = (*scope, ("class", node.name, node.lineno))
        bases: list[tuple[tuple[str, str, int], ...]] = []
        for base in node.bases:
            if _expression_reference(base) == "object":
                continue
            resolved_base = resolve_identity_expression(
                base,
                scope,
                (node.lineno, node.col_offset),
            )
            if tagged_identity(resolved_base, class_identity):
                bases.append(resolved_base[1])
            elif resolved_base is ambiguous_identity:
                uncertain_class_bases.add(resolved_class)
        class_base_scopes[resolved_class] = tuple(bases)

    def resolve_argparse_name(
        name: str,
        scope: tuple[tuple[str, str, int], ...],
        before: tuple[int, int],
        seen: frozenset[
            tuple[
                tuple[tuple[str, str, int], ...],
                str,
                tuple[int, int],
            ]
        ],
    ) -> object:
        for candidate_scope in visible_scopes(scope):
            key = (candidate_scope, name)
            events = binding_events.get(key, ())
            prior = [event for event in events if event[:2] < before]
            if prior:
                line, column, value = prior[-1]
                event_key = (candidate_scope, name, (line, column))
                if event_key in seen:
                    return other_identity
                if value is ambiguous_identity:
                    return ambiguous_identity
                if value in {
                    argparse_module_identity,
                    argparse_constructor_identity,
                    other_identity,
                }:
                    return value
                if isinstance(value, ast.AST):
                    return resolve_argparse_expression(
                        value,
                        candidate_scope,
                        (line, column),
                        seen | {event_key},
                    )
                return other_identity
            # Function-local bindings shadow outer scopes for the whole body,
            # including uses textually before the binding.
            if events and candidate_scope and candidate_scope[-1][0] == "function":
                return other_identity
        return other_identity

    def resolve_argparse_expression(
        expression: ast.AST,
        scope: tuple[tuple[str, str, int], ...],
        before: tuple[int, int],
        seen: frozenset[
            tuple[
                tuple[tuple[str, str, int], ...],
                str,
                tuple[int, int],
            ]
        ] = frozenset(),
    ) -> object:
        if isinstance(expression, ast.Name):
            return resolve_argparse_name(
                expression.id,
                scope,
                before,
                seen,
            )
        if isinstance(expression, ast.NamedExpr):
            return resolve_argparse_expression(
                expression.value,
                scope,
                before,
                seen,
            )
        if isinstance(expression, ast.IfExp):
            condition = _literal_value(expression.test)
            if condition is not _UNRESOLVED:
                selected = expression.body if bool(condition) else expression.orelse
                return resolve_argparse_expression(
                    selected,
                    scope,
                    before,
                    seen,
                )
            body = resolve_argparse_expression(
                expression.body,
                scope,
                before,
                seen,
            )
            other = resolve_argparse_expression(
                expression.orelse,
                scope,
                before,
                seen,
            )
            return body if body is other else ambiguous_identity
        if (
            isinstance(expression, ast.Attribute)
            and expression.attr == "ArgumentParser"
            and resolve_argparse_expression(
                expression.value,
                scope,
                before,
                seen,
            )
            is argparse_module_identity
        ):
            return argparse_constructor_identity
        return other_identity

    def is_argparse_constructor(call: ast.Call) -> bool:
        return (
            resolve_argparse_expression(
                call.func,
                scope_by_node[id(call)],
                (call.lineno, call.col_offset),
            )
            is argparse_constructor_identity
        )

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
    calls = sorted(
        (node for node in ast.walk(tree) if isinstance(node, ast.Call)),
        key=lambda node: (node.lineno, node.col_offset),
    )
    assigned_variable_by_call = {id(call): variable for variable, call, _node in assignments}
    assigned_node_by_call = {id(call): node for _variable, call, node in assignments}

    # A parser path is ``(root parser identity, command-name tuple)``.
    parser_paths: dict[
        tuple[tuple[tuple[str, str, int], ...], str],
        tuple[tuple[object, ...], tuple[str, ...]],
    ] = {}
    parser_event_paths: dict[
        tuple[
            tuple[tuple[tuple[str, str, int], ...], str],
            int,
            int,
        ],
        tuple[tuple[object, ...], tuple[str, ...]],
    ] = {}
    parser_path_positions: dict[
        tuple[tuple[tuple[str, str, int], ...], str],
        tuple[int, int],
    ] = {}

    def parser_event_key(
        variable: tuple[tuple[tuple[str, str, int], ...], str],
        node: ast.AST,
    ) -> tuple[
        tuple[tuple[tuple[str, str, int], ...], str],
        int,
        int,
    ]:
        line, column = binding_position(node)
        return variable, line, column

    def record_parser_binding(
        variable: tuple[tuple[tuple[str, str, int], ...], str],
        node: ast.AST,
        path: tuple[tuple[object, ...], tuple[str, ...]],
    ) -> bool:
        position = binding_position(node)
        changed = False
        if position >= parser_path_positions.get(variable, (-1, -1)):
            changed = parser_paths.get(variable) != path
            parser_paths[variable] = path
            parser_path_positions[variable] = position
        event_key = parser_event_key(variable, node)
        changed = changed or parser_event_paths.get(event_key) != path
        parser_event_paths[event_key] = path
        return changed

    constructor_paths: dict[
        int,
        tuple[tuple[object, ...], tuple[str, ...]],
    ] = {}
    root_ids: set[tuple[object, ...]] = set()
    for call in calls:
        if not is_argparse_constructor(call) or not reachable_declaration_contexts(call):
            continue
        variable = assigned_variable_by_call.get(id(call))
        if variable is None:
            variable = (
                scope_by_node[id(call)],
                f"$direct@{call.lineno}:{call.col_offset}",
            )
        root_id = (variable, call.lineno, call.col_offset)
        root_ids.add(root_id)
        path = (root_id, ())
        constructor_paths[id(call)] = path
        if id(call) in assigned_variable_by_call:
            record_parser_binding(
                variable,
                assigned_node_by_call[id(call)],
                path,
            )

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
        root_before: tuple[int, int] | None = None,
    ) -> tuple[tuple[str, str, int], ...] | None:
        resolved = resolve_identity_expression(
            expression,
            scope,
            root_before=root_before,
        )
        return resolved[1] if tagged_identity(resolved, function_identity) else None

    function_return_paths: dict[
        tuple[tuple[str, str, int], ...],
        tuple[tuple[object, ...], tuple[str, ...]],
    ] = {}

    def ordered_name_binding(
        name: str,
        scope: tuple[tuple[str, str, int], ...],
        before: tuple[int, int],
    ) -> (
        tuple[
            tuple[tuple[str, str, int], ...],
            tuple[int, int, object] | None,
        ]
        | None
    ):
        for candidate_scope in visible_scopes(scope):
            events = binding_events.get((candidate_scope, name), ())
            prior = [event for event in events if event[:2] < before]
            if prior:
                return candidate_scope, prior[-1]
            if events and candidate_scope and candidate_scope[-1][0] == "function":
                return candidate_scope, None
        return None

    def ordered_parser_name(
        name: str,
        scope: tuple[tuple[str, str, int], ...],
        before: tuple[int, int],
    ) -> tuple[
        bool,
        tuple[tuple[object, ...], tuple[str, ...]] | None,
    ]:
        binding = ordered_name_binding(name, scope, before)
        if binding is None:
            return False, None
        candidate_scope, event = binding
        if event is None:
            return True, None
        line, column, _value = event
        return (
            True,
            parser_event_paths.get(((candidate_scope, name), line, column)),
        )

    def resolve_parser_expression(
        expression: ast.AST,
        scope: tuple[tuple[str, str, int], ...],
    ) -> tuple[tuple[object, ...], tuple[str, ...]] | None:
        if isinstance(expression, ast.Name):
            _bound, ordered = ordered_parser_name(
                expression.id,
                scope,
                (expression.lineno, expression.col_offset),
            )
            return ordered
        reference = _expression_reference(expression)
        if reference is not None:
            resolved = resolve_reference(parser_paths, reference, scope)
            if isinstance(resolved, tuple):
                return resolved
        if isinstance(expression, ast.NamedExpr):
            return resolve_parser_expression(expression.value, scope)
        if isinstance(expression, ast.IfExp):
            condition = _literal_value(expression.test)
            if condition is not _UNRESOLVED:
                selected = expression.body if bool(condition) else expression.orelse
                return resolve_parser_expression(selected, scope)
            body = resolve_parser_expression(expression.body, scope)
            other = resolve_parser_expression(expression.orelse, scope)
            return body if body is not None and body == other else None
        if isinstance(expression, ast.Call):
            constructor = constructor_paths.get(id(expression))
            if constructor is not None:
                return constructor
            function_scope = resolve_function(expression.func, scope)
            if function_scope is not None:
                return function_return_paths.get(function_scope)
        return None

    def definitely_non_parser_expression(
        expression: ast.AST,
        scope: tuple[tuple[str, str, int], ...],
    ) -> bool:
        if not isinstance(expression, ast.Name):
            return False
        binding = ordered_name_binding(
            expression.id,
            scope,
            (expression.lineno, expression.col_offset),
        )
        if binding is None:
            return False

        def could_be_parser(
            value: ast.AST,
            value_scope: tuple[tuple[str, str, int], ...],
            seen: frozenset[
                tuple[
                    tuple[tuple[str, str, int], ...],
                    str,
                    tuple[int, int],
                ]
            ] = frozenset(),
        ) -> bool:
            if resolve_parser_expression(value, value_scope) is not None:
                return True
            if isinstance(value, ast.NamedExpr):
                return could_be_parser(value.value, value_scope, seen)
            if isinstance(value, ast.IfExp):
                return could_be_parser(
                    value.body,
                    value_scope,
                    seen,
                ) or could_be_parser(
                    value.orelse,
                    value_scope,
                    seen,
                )
            if isinstance(value, ast.Name):
                nested = ordered_name_binding(
                    value.id,
                    value_scope,
                    (value.lineno, value.col_offset),
                )
                if nested is None or nested[1] is None:
                    return False
                nested_scope, (line, column, nested_value) = nested
                event_key = (nested_scope, value.id, (line, column))
                if event_key in seen:
                    return True
                return isinstance(nested_value, ast.AST) and could_be_parser(
                    nested_value,
                    nested_scope,
                    seen | {event_key},
                )
            if isinstance(value, ast.Call):
                resolved_class = resolve_identity_expression(
                    value.func,
                    value_scope,
                )
                return not tagged_identity(
                    resolved_class,
                    class_identity,
                )
            return False

        binding_scope, event = binding
        if event is None:
            return False
        line, column, value = event
        if (line, column) == (-1, -1):
            return False
        if value is ambiguous_identity:
            return False
        if parser_event_paths.get(((binding_scope, expression.id), line, column)) is not None:
            return False
        return not (
            isinstance(value, ast.AST)
            and could_be_parser(
                value,
                binding_scope,
                frozenset(
                    {
                        (
                            binding_scope,
                            expression.id,
                            (line, column),
                        )
                    }
                ),
            )
        )

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
            and reachable_declaration_contexts(call)
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
        for node in assignment_nodes:
            variable_name = _assigned_name(node)
            if variable_name is None:
                continue
            if not declaration_contexts(node)[0]:
                continue
            scope = scope_by_node[id(node)]
            path = resolve_parser_expression(node.value, scope)
            variable = binding_key(variable_name, scope)
            if path is not None:
                changed = record_parser_binding(variable, node, path) or changed
        for expression in (node for node in ast.walk(tree) if isinstance(node, ast.NamedExpr)):
            target = _expression_reference(expression.target)
            if target is None:
                continue
            if not declaration_contexts(expression)[0]:
                continue
            scope = scope_by_node[id(expression)]
            path = resolve_parser_expression(expression.value, scope)
            variable = binding_key(target, scope)
            if path is not None:
                changed = (
                    record_parser_binding(
                        variable,
                        expression,
                        path,
                    )
                    or changed
                )
        for variable, call, node in assignments:
            if not declaration_contexts(node)[0]:
                continue
            scope = scope_by_node[id(node)]
            if not isinstance(call.func, ast.Attribute):
                function_scope = resolve_function(call.func, scope)
                if function_scope is not None and function_scope in function_return_paths:
                    path = function_return_paths[function_scope]
                    changed = record_parser_binding(variable, node, path) or changed
                continue
            owner = _expression_reference(call.func.value)
            if owner is None:
                continue
            if call.func.attr == "add_subparsers":
                path = resolve_parser_expression(call.func.value, scope)
                if isinstance(path, tuple) and subparser_paths.get(variable) != path:
                    subparser_paths[variable] = path
                    changed = True
            elif call.func.attr == "add_parser" and call.args:
                if has_enclosing_loop(call):
                    continue
                parent = resolve_reference(subparser_paths, owner, scope)
                contexts = reachable_declaration_contexts(call)
                names = {
                    name
                    for binding in contexts
                    if isinstance(
                        (name := _literal_value(call.args[0], binding)),
                        str,
                    )
                }
                if isinstance(parent, tuple) and len(names) == 1:
                    name = next(iter(names))
                    path = (parent[0], (*parent[1], name))
                    changed = record_parser_binding(variable, node, path) or changed

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
        if has_enclosing_loop(call):
            continue
        contexts = reachable_declaration_contexts(call)
        if not contexts:
            continue
        parent = resolve_reference(
            subparser_paths,
            owner,
            scope_by_node[id(call)],
        )
        for binding in contexts:
            name = _literal_value(call.args[0], binding)
            if not (isinstance(parent, tuple) and isinstance(name, str)):
                continue
            path = (parent[0], (*parent[1], name))
            command_paths.add(path)
            record_command_aliases(path, call, binding)
            inherited = resolved_parent_paths(call, scope_by_node[id(call)])
            if inherited:
                parent_paths[path] = inherited

    for variable, call, node in assignments:
        if not is_argparse_constructor(call):
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
        generated: dict[
            tuple[tuple[tuple[str, str, int], ...], str],
            list[
                tuple[
                    dict[str, object],
                    tuple[tuple[object, ...], tuple[str, ...]],
                ]
            ],
        ] = {}
        for call in calls:
            if nearest_enclosing_for(call) is not loop:
                continue
            if not (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "add_parser"
                and call.args
                and (owner := _expression_reference(call.func.value)) is not None
            ):
                continue
            contexts = reachable_declaration_contexts(call)
            if not contexts:
                continue
            parent = resolve_reference(
                subparser_paths,
                owner,
                scope_by_node[id(call)],
            )
            if not isinstance(parent, tuple):
                continue
            variable = assigned_variable_by_call.get(id(call))
            for binding in contexts:
                name = _bound_loop_value(call.args[0], binding)
                if not isinstance(name, str):
                    continue
                path = (parent[0], (*parent[1], name))
                if variable is not None:
                    generated.setdefault(variable, []).append((binding, path))
                command_paths.add(path)
                record_command_aliases(path, call, binding)
                inherited = resolved_parent_paths(
                    call,
                    scope_by_node[id(call)],
                )
                if inherited:
                    parent_paths[path] = inherited
            if variable is not None and generated.get(variable):
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
        generated = loop_generated.get(id(loop), {})
        for node in calls:
            if nearest_enclosing_for(node) is not loop:
                continue
            if not (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and (owner := _expression_reference(node.func.value)) is not None
            ):
                continue
            contexts = reachable_declaration_contexts(node)
            if not contexts:
                loop_arguments[id(node)] = []
                continue
            owner_key = binding_key(owner, scope_by_node[id(node)])
            if owner_key in generated:
                loop_arguments[id(node)] = [
                    (path, binding) for binding, path in generated[owner_key] if binding in contexts
                ]
                continue
            path = resolve_reference(
                argument_containers,
                owner,
                scope_by_node[id(node)],
            )
            if isinstance(path, tuple):
                loop_arguments[id(node)] = [(path, binding) for binding in contexts]

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
        contexts = reachable_declaration_contexts(node)
        if not contexts:
            continue
        scope = scope_by_node[id(node)]
        owner = _expression_reference(node.func.value)
        receiver_path = resolve_parser_expression(
            node.func.value,
            scope,
        )
        if owner is None and receiver_path is None:
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
                (path, binding)
                for path in sorted(
                    helper_argument_paths[id(node)],
                    key=repr,
                )
                for binding in contexts
            ]
        else:
            path = (
                resolve_reference(
                    argument_containers,
                    owner,
                    scope,
                )
                if owner is not None
                else receiver_path
            )
            declarations = (
                [(path, binding) for binding in contexts] if isinstance(path, tuple) else []
            )
        for path, binding in declarations:
            argument = _argument_spec(node, binding)
            if argument is None:
                continue
            target = specs.setdefault(path, CommandSpec.empty())
            if isinstance(argument, OptionSpec):
                for alias in argument.aliases:
                    target.options[alias] = argument
                if owner is not None:
                    group = binding_key(owner, scope)
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
        path = resolve_parser_expression(
            call.func.value,
            scope_by_node[id(call)],
        )
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
    non_parser_parse_scopes: list[tuple[tuple[str, str, int], ...]] = []
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
        elif definitely_non_parser_expression(
            call.func.value,
            scope_by_node[id(call)],
        ):
            non_parser_parse_scopes.append(scope_by_node[id(call)])
        else:
            unresolved_parse_scopes.append(scope_by_node[id(call)])

    main_scopes = {
        scope
        for scope in function_scopes.values()
        if (len(scope) == 1 and scope[-1][0] == "function" and scope[-1][1] == "main")
    }
    reachable_scopes = {(), *main_scopes}
    reachable_execution_points: dict[
        tuple[tuple[str, str, int], ...],
        set[tuple[int, int]],
    ] = {scope: {(sys.maxsize, sys.maxsize)} for scope in main_scopes}
    class_mros: dict[
        tuple[tuple[str, str, int], ...],
        tuple[tuple[tuple[str, str, int], ...], ...] | None,
    ] = {}

    def class_mro(
        resolved_class: tuple[tuple[str, str, int], ...],
        active: frozenset[tuple[tuple[str, str, int], ...]] = frozenset(),
    ) -> tuple[tuple[tuple[str, str, int], ...], ...] | None:
        if resolved_class in class_mros:
            return class_mros[resolved_class]
        if resolved_class in active or resolved_class in uncertain_class_bases:
            return None
        bases = class_base_scopes.get(resolved_class, ())
        base_mros = [class_mro(base, active | {resolved_class}) for base in bases]
        if any(mro is None for mro in base_mros):
            class_mros[resolved_class] = None
            return None
        sequences = [list(mro) for mro in base_mros if mro is not None]
        sequences.append(list(bases))
        merged: list[tuple[tuple[str, str, int], ...]] = []
        while any(sequences):
            sequences = [sequence for sequence in sequences if sequence]
            candidate = next(
                (
                    sequence[0]
                    for sequence in sequences
                    if not any(sequence[0] in other[1:] for other in sequences)
                ),
                None,
            )
            if candidate is None:
                class_mros[resolved_class] = None
                return None
            merged.append(candidate)
            for sequence in sequences:
                if sequence and sequence[0] == candidate:
                    sequence.pop(0)
        result = (resolved_class, *merged)
        class_mros[resolved_class] = result
        return result

    constructor_contexts: set[
        tuple[
            tuple[tuple[str, str, int], ...],
            tuple[tuple[tuple[str, str, int], ...], ...],
            int,
            tuple[int, int],
        ]
    ] = set()

    def next_constructor_context(
        mro: tuple[tuple[tuple[str, str, int], ...], ...],
        method_name: str,
        start: int,
        root_before: tuple[int, int],
    ) -> (
        tuple[
            tuple[tuple[str, str, int], ...],
            tuple[tuple[tuple[str, str, int], ...], ...],
            int,
            tuple[int, int],
        ]
        | None
    ):
        for index in range(start, len(mro)):
            method_scope = function_scopes.get((mro[index], method_name))
            if isinstance(method_scope, tuple):
                return method_scope, mro, index, root_before
        return None

    def explicit_super_constructor_contexts(
        call: ast.Call,
        call_scope: tuple[tuple[str, str, int], ...],
    ) -> set[
        tuple[
            tuple[tuple[str, str, int], ...],
            tuple[tuple[tuple[str, str, int], ...], ...],
            int,
            tuple[int, int],
        ]
    ]:
        if (
            not isinstance(call.func, ast.Attribute)
            or call.func.attr not in {"__new__", "__init__"}
            or not isinstance(call.func.value, ast.Call)
        ):
            return set()
        super_call = call.func.value
        if (
            not isinstance(super_call.func, ast.Name)
            or super_call.func.id != "super"
            or super_call.keywords
        ):
            return set()
        contexts: set[
            tuple[
                tuple[tuple[str, str, int], ...],
                tuple[tuple[tuple[str, str, int], ...], ...],
                int,
                tuple[int, int],
            ]
        ] = set()
        for function_scope, mro, _owner_index, root_before in constructor_contexts:
            if function_scope != call_scope:
                continue
            if not super_call.args:
                start_class = class_scope(call_scope)
            elif len(super_call.args) == 2:
                resolved_start = resolve_identity_expression(
                    super_call.args[0],
                    call_scope,
                    root_before=root_before,
                )
                start_class = (
                    resolved_start[1] if tagged_identity(resolved_start, class_identity) else None
                )
                parameters = function_parameters.get(call_scope, ())
                receiver = resolve_identity_expression(
                    super_call.args[1],
                    call_scope,
                    root_before=root_before,
                )
                if (
                    not isinstance(start_class, tuple)
                    or not parameters
                    or receiver
                    != (
                        parameter_identity,
                        call_scope,
                        parameters[0],
                    )
                ):
                    continue
            else:
                continue
            if not isinstance(start_class, tuple) or start_class not in mro:
                continue
            next_context = next_constructor_context(
                mro,
                call.func.attr,
                mro.index(start_class) + 1,
                root_before,
            )
            if next_context is not None:
                contexts.add(next_context)
        return contexts

    applied_external_effects: set[
        tuple[
            tuple[tuple[str, str, int], ...],
            int,
            tuple[int, int],
        ]
    ] = set()

    def outer_nonlocal_scope(
        function_scope: tuple[tuple[str, str, int], ...],
        name: str,
    ) -> tuple[tuple[str, str, int], ...] | None:
        for length in range(len(function_scope) - 1, 0, -1):
            candidate = function_scope[:length]
            if candidate[-1][0] == "function" and (
                (candidate, name) in identity_events or (candidate, name) in binding_events
            ):
                return candidate
        return None

    def effect_position(
        target_scope: tuple[tuple[str, str, int], ...],
        call: ast.Call,
        call_scope: tuple[tuple[str, str, int], ...],
        root_before: tuple[int, int],
    ) -> tuple[int, int]:
        if target_scope == () and call_scope != ():
            line, column = root_before
            return line, column - 1
        return call.lineno, call.col_offset

    def record_external_effect(
        target_scope: tuple[tuple[str, str, int], ...],
        name: str,
        position: tuple[int, int],
    ) -> None:
        line, column = position
        identity_events.setdefault((target_scope, name), []).append(
            (line, column, ambiguous_identity)
        )
        opaque_identity_event_keys.add((target_scope, name, position))
        identity_events[(target_scope, name)].sort(key=lambda event: event[:2])
        binding_events.setdefault((target_scope, name), []).append(
            (line, column, ambiguous_identity)
        )
        binding_events[(target_scope, name)].sort(key=lambda event: event[:2])

    def apply_external_effects(
        function_scope: tuple[tuple[str, str, int], ...],
        call: ast.Call,
        call_scope: tuple[tuple[str, str, int], ...],
        root_before: tuple[int, int],
    ) -> None:
        effect_key = (function_scope, id(call), root_before)
        if effect_key in applied_external_effects:
            return
        applied_external_effects.add(effect_key)
        global_writes, nonlocal_writes = function_external_effects.get(
            function_scope,
            (frozenset(), frozenset()),
        )
        for name in global_writes:
            target_scope: tuple[tuple[str, str, int], ...] = ()
            record_external_effect(
                target_scope,
                name,
                effect_position(
                    target_scope,
                    call,
                    call_scope,
                    root_before,
                ),
            )
        for name in nonlocal_writes:
            target_scope = outer_nonlocal_scope(function_scope, name)
            if target_scope is None:
                continue
            record_external_effect(
                target_scope,
                name,
                effect_position(
                    target_scope,
                    call,
                    call_scope,
                    root_before,
                ),
            )

    parse_scopes = {scope for scope, _root_id, _allow_extras in parse_calls}

    def identity_scope_has_parse_call(
        identity_scope: tuple[tuple[str, str, int], ...],
    ) -> bool:
        if identity_scope and identity_scope[-1][0] == "function":
            return identity_scope in parse_scopes
        return any(scope[: len(identity_scope)] == identity_scope for scope in parse_scopes)

    parse_method_names = {
        scope[-1][1]
        for scope in parse_scopes
        if (scope and scope[-1][0] == "function" and any(part[0] == "class" for part in scope[:-1]))
    }
    parser_identity_method_names = {
        "add_argument",
        "add_argument_group",
        "add_mutually_exclusive_group",
        "add_parser",
        "add_subparsers",
        "parse_args",
        "parse_intermixed_args",
        "parse_known_args",
        "parse_known_intermixed_args",
    }

    def expression_mentions_parse_identity(
        expression: ast.AST,
        scope: tuple[tuple[str, str, int], ...],
        root_before: tuple[int, int],
    ) -> bool:
        for candidate in ast.walk(expression):
            if not isinstance(candidate, ast.Name):
                continue
            identity = resolve_identity_expression(
                candidate,
                scope,
                (candidate.lineno, candidate.col_offset),
                root_before,
            )
            if (
                tagged_identity(identity, class_identity)
                or tagged_identity(identity, function_identity)
                or tagged_identity(identity, instance_identity)
            ) and identity_scope_has_parse_call(identity[1]):
                return True
        return False

    def name_can_dispatch_to_parse_scope(
        name: str,
        scope: tuple[tuple[str, str, int], ...],
        before: tuple[int, int],
        root_before: tuple[int, int],
    ) -> bool:
        for candidate_scope in visible_scopes(scope):
            candidate_before = root_before if candidate_scope == () else before
            events = [
                event
                for event in identity_events.get(
                    (candidate_scope, name),
                    (),
                )
                if event[:2] < candidate_before
            ]
            for line, column, value in reversed(events):
                definition_scope = definition_identity_scopes.get(
                    (
                        candidate_scope,
                        name,
                        (line, column),
                    )
                )
                if definition_scope is not None and identity_scope_has_parse_call(definition_scope):
                    return True
                if (
                    tagged_identity(value, class_identity)
                    or tagged_identity(value, function_identity)
                    or tagged_identity(value, instance_identity)
                ):
                    if identity_scope_has_parse_call(value[1]):
                        return True
                    continue
                if value in {
                    argparse_constructor_identity,
                    argparse_module_identity,
                }:
                    return True
                if isinstance(value, ast.AST):
                    resolved = resolve_identity_expression(
                        value,
                        candidate_scope,
                        (line, column),
                        ((line, column) if candidate_scope == () else root_before),
                    )
                    if (
                        tagged_identity(resolved, class_identity)
                        or tagged_identity(resolved, function_identity)
                        or tagged_identity(resolved, instance_identity)
                    ) and identity_scope_has_parse_call(resolved[1]):
                        return True
                    if resolved is ambiguous_identity and expression_mentions_parse_identity(
                        value,
                        candidate_scope,
                        ((line, column) if candidate_scope == () else root_before),
                    ):
                        return True
            if events and candidate_scope and candidate_scope[-1][0] == "function":
                return False
        return False

    def name_has_opaque_identity_effect(
        name: str,
        scope: tuple[tuple[str, str, int], ...],
        before: tuple[int, int],
        root_before: tuple[int, int],
    ) -> bool:
        for candidate_scope in visible_scopes(scope):
            candidate_before = root_before if candidate_scope == () else before
            barriers = [
                position
                for position in opaque_identity_barriers.get(
                    candidate_scope,
                    (),
                )
                if position < candidate_before
            ]
            latest_barrier = barriers[-1] if barriers else None
            events = [
                event
                for event in identity_events.get(
                    (candidate_scope, name),
                    (),
                )
                if event[:2] < candidate_before
            ]
            for line, column, value in reversed(events):
                key = (
                    candidate_scope,
                    name,
                    (line, column),
                )
                if key in opaque_identity_event_keys:
                    return True
                if value is ambiguous_identity:
                    continue
                if latest_barrier is not None and latest_barrier > (line, column):
                    return True
                return False
            if latest_barrier is not None:
                return True
            if events and candidate_scope and candidate_scope[-1][0] == "function":
                return False
        return False

    def ambiguous_call_affects_identity_graph(
        call: ast.Call,
        call_scope: tuple[tuple[str, str, int], ...],
        root_before: tuple[int, int],
    ) -> bool:
        if isinstance(call.func, ast.Name):
            return name_can_dispatch_to_parse_scope(
                call.func.id,
                call_scope,
                (call.lineno, call.col_offset),
                root_before,
            )
        if not isinstance(call.func, ast.Attribute):
            return False
        if call.func.attr in parse_method_names:
            if expression_mentions_parse_identity(
                call.func.value,
                call_scope,
                root_before,
            ):
                return True
            if isinstance(call.func.value, ast.Name):
                return name_can_dispatch_to_parse_scope(
                    call.func.value.id,
                    call_scope,
                    (call.lineno, call.col_offset),
                    root_before,
                )
            return False
        return (
            call.func.attr in parser_identity_method_names
            and isinstance(call.func.value, ast.Name)
            and name_has_opaque_identity_effect(
                call.func.value.id,
                call_scope,
                (call.lineno, call.col_offset),
                root_before,
            )
        )

    def invocation_uses_identity_default(
        function_scope: tuple[tuple[str, str, int], ...],
        call: ast.Call,
        *,
        implicit_receiver: bool = False,
    ) -> bool:
        defaults = function_opaque_default_parameters.get(function_scope)
        if not defaults:
            return False
        parameters = function_parameters.get(function_scope, ())
        positional_offset = (
            1
            if (
                implicit_receiver
                or (
                    len(function_scope) >= 2
                    and function_scope[-2][0] == "class"
                    and isinstance(call.func, ast.Attribute)
                )
            )
            else 0
        )
        provided = {keyword.arg for keyword in call.keywords if keyword.arg is not None}
        provided.update(parameters[: len(call.args) + positional_offset])
        definition_scope = function_scope[:-1]
        for name, default in defaults.items():
            if name in provided:
                continue
            identity = resolve_identity_expression(
                default,
                definition_scope,
                (default.lineno, default.col_offset),
            )
            if (
                tagged_identity(identity, class_identity)
                or tagged_identity(identity, function_identity)
                or tagged_identity(identity, instance_identity)
            ) and identity_scope_has_parse_call(identity[1]):
                return True
            if identity is ambiguous_identity and expression_mentions_parse_identity(
                default,
                definition_scope,
                (default.lineno, default.col_offset),
            ):
                return True
        return False

    def parameter_dispatch_reaches_parse(
        parameter: object,
        root_before: tuple[int, int],
    ) -> bool:
        if not tagged_identity(parameter, parameter_identity):
            return False
        function_scope = parameter[1]
        parameter_name = parameter[2]
        parameters = function_parameters.get(function_scope, ())
        if parameter_name not in parameters:
            return False
        parameter_index = parameters.index(parameter_name)
        for invocation in calls:
            invocation_scope = scope_by_node[id(invocation)]
            if (
                resolve_function(
                    invocation.func,
                    invocation_scope,
                    root_before,
                )
                != function_scope
            ):
                continue
            argument: ast.AST | None = None
            if parameter_index < len(invocation.args):
                argument = invocation.args[parameter_index]
            else:
                argument = next(
                    (
                        keyword.value
                        for keyword in invocation.keywords
                        if keyword.arg == parameter_name
                    ),
                    None,
                )
            if argument is None:
                argument = function_opaque_default_parameters.get(
                    function_scope,
                    {},
                ).get(parameter_name)
            if argument is None:
                continue
            identity = resolve_identity_expression(
                argument,
                invocation_scope,
                (argument.lineno, argument.col_offset),
                root_before,
            )
            if (
                tagged_identity(identity, class_identity)
                or tagged_identity(identity, function_identity)
                or tagged_identity(identity, instance_identity)
            ) and identity_scope_has_parse_call(identity[1]):
                return True
            if identity is ambiguous_identity and expression_mentions_parse_identity(
                argument,
                invocation_scope,
                root_before,
            ):
                return True
        return False

    applied_unknown_call_effects: set[tuple[int, tuple[int, int]]] = set()

    def expanded_argument_nodes(
        expression: ast.AST,
        scope: tuple[tuple[str, str, int], ...],
        before: tuple[int, int],
        root_before: tuple[int, int],
        seen: frozenset[
            tuple[
                tuple[tuple[str, str, int], ...],
                str,
                tuple[int, int],
            ]
        ] = frozenset(),
    ) -> Iterable[ast.AST]:
        yield expression
        for child in ast.iter_child_nodes(expression):
            yield from expanded_argument_nodes(
                child,
                scope,
                before,
                root_before,
                seen,
            )
        if not isinstance(expression, ast.Name):
            return
        for candidate_scope in visible_scopes(scope):
            candidate_before = root_before if candidate_scope == () else before
            prior = [
                event
                for event in identity_events.get(
                    (candidate_scope, expression.id),
                    (),
                )
                if event[:2] < candidate_before
            ]
            barriers = [
                position
                for position in opaque_identity_barriers.get(
                    candidate_scope,
                    (),
                )
                if position < candidate_before
            ]
            if not prior or (barriers and barriers[-1] > prior[-1][:2]):
                return
            line, column, value = prior[-1]
            event_key = (
                candidate_scope,
                expression.id,
                (line, column),
            )
            if event_key in seen or not isinstance(value, ast.AST):
                return
            yield from expanded_argument_nodes(
                value,
                candidate_scope,
                (line, column),
                ((line, column) if candidate_scope == () else root_before),
                seen | {event_key},
            )
            return

    def binding_scope_for_name(
        name: str,
        scope: tuple[tuple[str, str, int], ...],
        before: tuple[int, int],
        root_before: tuple[int, int],
    ) -> tuple[tuple[str, str, int], ...] | None:
        for candidate_scope in visible_scopes(scope):
            candidate_before = root_before if candidate_scope == () else before
            if any(
                event[:2] < candidate_before
                for event in identity_events.get(
                    (candidate_scope, name),
                    (),
                )
            ):
                return candidate_scope
        return None

    def apply_unknown_call_effects(
        call: ast.Call,
        call_scope: tuple[tuple[str, str, int], ...],
        root_before: tuple[int, int],
    ) -> bool:
        effect_key = (id(call), root_before)
        if effect_key in applied_unknown_call_effects:
            return False
        applied_unknown_call_effects.add(effect_key)
        if (
            isinstance(call.func, ast.Name)
            and call.func.id == "super"
            and not any(
                event[:2] < (call.lineno, call.col_offset)
                for candidate_scope in visible_scopes(call_scope)
                for event in identity_events.get(
                    (candidate_scope, "super"),
                    (),
                )
            )
        ):
            return False
        expressions = [
            *call.args,
            *(keyword.value for keyword in call.keywords),
        ]
        parser_instances: set[object] = set()
        for expression in expressions:
            for argument in expanded_argument_nodes(
                expression,
                call_scope,
                (call.lineno, call.col_offset),
                root_before,
            ):
                if (
                    isinstance(argument, ast.Call)
                    and isinstance(argument.func, ast.Name)
                    and argument.func.id in {"globals", "locals"}
                    and not argument.args
                    and not argument.keywords
                ):
                    target_scope = () if argument.func.id == "globals" else call_scope
                    position = effect_position(
                        target_scope,
                        call,
                        call_scope,
                        root_before,
                    )
                    opaque_identity_barriers.setdefault(
                        target_scope,
                        [],
                    ).append(position)
                    opaque_identity_barriers[target_scope].sort()
                    continue
                if not isinstance(argument, ast.Name):
                    continue
                identity = resolve_identity_expression(
                    argument,
                    call_scope,
                    (argument.lineno, argument.col_offset),
                    root_before,
                )
                if tagged_identity(
                    identity,
                    instance_identity,
                ) and identity_scope_has_parse_call(identity[1]):
                    parser_instances.add(identity)

        affected_symbols: set[tuple[tuple[tuple[str, str, int], ...], str]] = set()
        before = (call.lineno, call.col_offset)
        for candidate_scope in visible_scopes(call_scope):
            for event_scope, name in identity_events:
                if event_scope != candidate_scope:
                    continue
                identity = resolve_identity_name(
                    name,
                    call_scope,
                    before,
                    root_before,
                )
                if identity in parser_instances:
                    affected_symbols.add((candidate_scope, name))
        for target_scope, name in affected_symbols:
            record_external_effect(
                target_scope,
                name,
                effect_position(
                    target_scope,
                    call,
                    call_scope,
                    root_before,
                ),
            )

        return bool(parser_instances and not affected_symbols)

    function_instance_mutations: dict[
        tuple[tuple[str, str, int], ...],
        list[ast.Name],
    ] = {}

    def mutated_object_name(target: ast.AST) -> ast.Name | None:
        current = target
        while isinstance(current, (ast.Attribute, ast.Subscript)):
            current = current.value
        return current if isinstance(current, ast.Name) else None

    for mutation in ast.walk(tree):
        targets: list[ast.AST] = []
        if isinstance(mutation, ast.Assign):
            targets.extend(mutation.targets)
        elif isinstance(mutation, (ast.AnnAssign, ast.AugAssign)):
            targets.append(mutation.target)
        elif isinstance(mutation, ast.Delete):
            targets.extend(mutation.targets)
        if not targets:
            continue
        mutation_scope = scope_by_node[id(mutation)]
        if not (mutation_scope and mutation_scope[-1][0] == "function"):
            continue
        for target in targets:
            name = mutated_object_name(target)
            if name is not None and name is not target:
                function_instance_mutations.setdefault(
                    mutation_scope,
                    [],
                ).append(name)

    applied_context_instance_effects: set[
        tuple[
            tuple[tuple[str, str, int], ...],
            int,
            tuple[int, int],
        ]
    ] = set()

    def apply_context_instance_effects(
        function_scope: tuple[tuple[str, str, int], ...],
        context_call: ast.AST,
        call_scope: tuple[tuple[str, str, int], ...],
        root_before: tuple[int, int],
    ) -> None:
        effect_key = (function_scope, id(context_call), root_before)
        if effect_key in applied_context_instance_effects:
            return
        applied_context_instance_effects.add(effect_key)
        parser_instances: set[object] = set()
        for name in function_instance_mutations.get(
            function_scope,
            (),
        ):
            identity = resolve_identity_expression(
                name,
                function_scope,
                (name.lineno, name.col_offset),
                root_before,
            )
            if tagged_identity(
                identity,
                instance_identity,
            ) and identity_scope_has_parse_call(identity[1]):
                parser_instances.add(identity)
        if not parser_instances:
            return
        before = (context_call.lineno, context_call.col_offset)
        affected_symbols: set[tuple[tuple[tuple[str, str, int], ...], str]] = set()
        for candidate_scope in visible_scopes(function_scope):
            for event_scope, name in identity_events:
                if event_scope != candidate_scope:
                    continue
                identity = resolve_identity_name(
                    name,
                    function_scope,
                    before,
                    root_before,
                )
                if identity in parser_instances:
                    affected_symbols.add((candidate_scope, name))
        for target_scope, name in affected_symbols:
            record_external_effect(
                target_scope,
                name,
                effect_position(
                    target_scope,
                    context_call,
                    call_scope,
                    root_before,
                ),
            )

    def expression_mentions_user_identity(
        expression: ast.AST,
        scope: tuple[tuple[str, str, int], ...],
        root_before: tuple[int, int],
    ) -> bool:
        return any(
            isinstance(candidate, ast.Name)
            and any(
                tagged_identity(
                    resolve_identity_expression(
                        candidate,
                        scope,
                        (candidate.lineno, candidate.col_offset),
                        root_before,
                    ),
                    tag,
                )
                for tag in (
                    class_identity,
                    function_identity,
                    instance_identity,
                )
            )
            for candidate in expanded_argument_nodes(
                expression,
                scope,
                (expression.lineno, expression.col_offset),
                root_before,
            )
        )

    with_nodes = [
        candidate
        for candidate in ast.walk(tree)
        if isinstance(candidate, (ast.With, ast.AsyncWith))
    ]

    identity_graph_complete = True
    reachability_changed = True
    while reachability_changed:
        reachability_changed = False
        for with_node in with_nodes:
            with_scope = scope_by_node[id(with_node)]
            if with_scope not in reachable_scopes or not declaration_contexts(with_node)[0]:
                continue
            execution_points = (
                {(with_node.lineno, with_node.col_offset)}
                if with_scope == ()
                else reachable_execution_points.get(with_scope, set())
            )
            method_names = (
                ("__aenter__", "__aexit__")
                if isinstance(with_node, ast.AsyncWith)
                else ("__enter__", "__exit__")
            )
            for root_before in execution_points:
                for item in with_node.items:
                    context_expression = item.context_expr
                    context_identity = resolve_identity_expression(
                        context_expression,
                        with_scope,
                        root_before=root_before,
                    )
                    if not tagged_identity(
                        context_identity,
                        instance_identity,
                    ):
                        context_factory_scope = None
                        if isinstance(context_expression, ast.Call):
                            callable_identity = resolve_identity_expression(
                                context_expression.func,
                                with_scope,
                                root_before=root_before,
                            )
                            if (
                                tagged_identity(
                                    callable_identity,
                                    function_identity,
                                )
                                and callable_identity[1] in context_factory_function_scopes
                            ):
                                context_factory_scope = callable_identity[1]
                        if context_factory_scope is not None:
                            apply_external_effects(
                                context_factory_scope,
                                context_expression,
                                with_scope,
                                root_before,
                            )
                            apply_context_instance_effects(
                                context_factory_scope,
                                context_expression,
                                with_scope,
                                root_before,
                            )
                            if context_factory_scope not in reachable_scopes:
                                reachable_scopes.add(context_factory_scope)
                                reachability_changed = True
                            points = reachable_execution_points.setdefault(
                                context_factory_scope,
                                set(),
                            )
                            if root_before not in points:
                                points.add(root_before)
                                reachability_changed = True
                            continue
                        if isinstance(context_expression, ast.Call):
                            context_callable_provenance = decorator_resolved_provenance(
                                context_expression.func,
                                with_scope,
                            )
                            if context_callable_provenance is invalidated_decorator_provenance:
                                identity_graph_complete = False
                                continue
                            if isinstance(context_callable_provenance, str):
                                continue
                        if (
                            context_identity is ambiguous_identity
                            and expression_mentions_user_identity(
                                context_expression,
                                with_scope,
                                root_before,
                            )
                        ):
                            identity_graph_complete = False
                        continue
                    mro = class_mro(context_identity[1])
                    if mro is None:
                        identity_graph_complete = False
                        continue
                    for method_name in method_names:
                        method_scope = next(
                            (
                                scope
                                for owner in mro
                                if (scope := function_scopes.get((owner, method_name))) is not None
                            ),
                            None,
                        )
                        if method_scope is None:
                            identity_graph_complete = False
                            continue
                        apply_external_effects(
                            method_scope,
                            context_expression,
                            with_scope,
                            root_before,
                        )
                        apply_context_instance_effects(
                            method_scope,
                            context_expression,
                            with_scope,
                            root_before,
                        )
                        if method_scope not in reachable_scopes:
                            reachable_scopes.add(method_scope)
                            reachability_changed = True
                        points = reachable_execution_points.setdefault(
                            method_scope,
                            set(),
                        )
                        if root_before not in points:
                            points.add(root_before)
                            reachability_changed = True
        for call in calls:
            if scope_by_node[id(call)] not in reachable_scopes:
                continue
            call_scope = scope_by_node[id(call)]
            for constructor_context in explicit_super_constructor_contexts(
                call,
                call_scope,
            ):
                if constructor_context not in constructor_contexts:
                    constructor_contexts.add(constructor_context)
                    reachability_changed = True
                constructor_scope, _mro, _index, root_before = constructor_context
                apply_external_effects(
                    constructor_scope,
                    call,
                    call_scope,
                    root_before,
                )
                if invocation_uses_identity_default(
                    constructor_scope,
                    call,
                    implicit_receiver=True,
                ):
                    identity_graph_complete = False
                if constructor_scope not in reachable_scopes:
                    reachable_scopes.add(constructor_scope)
                    reachability_changed = True
                points = reachable_execution_points.setdefault(
                    constructor_scope,
                    set(),
                )
                if root_before not in points:
                    points.add(root_before)
                    reachability_changed = True

            execution_points = (
                {(call.lineno, call.col_offset)}
                if call_scope == ()
                else reachable_execution_points.get(call_scope, set())
            )
            for root_before in execution_points:
                callable_identity = resolve_identity_expression(
                    call.func,
                    call_scope,
                    root_before=root_before,
                )
                if (
                    callable_identity is ambiguous_identity
                    and ambiguous_call_affects_identity_graph(
                        call,
                        call_scope,
                        root_before,
                    )
                ):
                    identity_graph_complete = False
                if parameter_dispatch_reaches_parse(
                    callable_identity,
                    root_before,
                ):
                    identity_graph_complete = False
                function_scope = resolve_function(
                    call.func,
                    call_scope,
                    root_before,
                )
                if function_scope is not None:
                    apply_external_effects(
                        function_scope,
                        call,
                        call_scope,
                        root_before,
                    )
                    if invocation_uses_identity_default(
                        function_scope,
                        call,
                    ):
                        identity_graph_complete = False
                    if function_scope not in reachable_scopes:
                        reachable_scopes.add(function_scope)
                        reachability_changed = True
                    points = reachable_execution_points.setdefault(
                        function_scope,
                        set(),
                    )
                    if root_before not in points:
                        points.add(root_before)
                        reachability_changed = True
                elif callable_identity is other_identity:
                    if apply_unknown_call_effects(
                        call,
                        call_scope,
                        root_before,
                    ):
                        identity_graph_complete = False
                resolved_class = (
                    callable_identity[1]
                    if tagged_identity(callable_identity, class_identity)
                    else None
                )
                if not isinstance(resolved_class, tuple):
                    continue
                mro = class_mro(resolved_class)
                if mro is None:
                    identity_graph_complete = False
                    continue
                for method_name in ("__new__", "__init__"):
                    constructor_context = next_constructor_context(
                        mro,
                        method_name,
                        0,
                        root_before,
                    )
                    if constructor_context is None:
                        continue
                    if constructor_context not in constructor_contexts:
                        constructor_contexts.add(constructor_context)
                        reachability_changed = True
                    (
                        constructor_scope,
                        _context_mro,
                        _index,
                        context_root,
                    ) = constructor_context
                    apply_external_effects(
                        constructor_scope,
                        call,
                        call_scope,
                        context_root,
                    )
                    if invocation_uses_identity_default(
                        constructor_scope,
                        call,
                        implicit_receiver=True,
                    ):
                        identity_graph_complete = False
                    if constructor_scope not in reachable_scopes:
                        reachable_scopes.add(constructor_scope)
                        reachability_changed = True
                    points = reachable_execution_points.setdefault(
                        constructor_scope,
                        set(),
                    )
                    if context_root not in points:
                        points.add(context_root)
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
        and any(scope in reachable_scopes for scope in uncertain_parser_declaration_scopes)
    ):
        return uncertain_contract(
            "a reachable argparse declaration has dynamic control-flow reachability"
        )

    if argparse_evidence and root_ids and not identity_graph_complete:
        return uncertain_contract(
            "a reachable call has an unresolved class, function, or instance identity"
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
    if not selected_calls and any(scope in reachable_scopes for scope in non_parser_parse_scopes):
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
        resolution = _resolve_shell_wrappers(
            raw_tokens,
            cwd=block.cwd,
        )
        tokens = list(resolution.tokens)
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

        module_root = resolution.cwd if resolution.cwd.is_absolute() else repo_root / resolution.cwd
        module_path = _local_python_module_path(module_root, module)
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
        resolution = _resolve_shell_wrappers(
            raw_tokens,
            cwd=block.cwd,
        )
        tokens = list(resolution.tokens)
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
            allow_plain = _resolved_command_cwd(repo_root, resolution.cwd) != repo_root.resolve()
            local = _resolved_repo_local_path(
                repo_root,
                resolution.cwd,
                value,
                allow_plain=allow_plain,
            )
            if local and not (repo_root / local).exists():
                findings.append(
                    Finding(
                        block.path,
                        _source_line(block, offset),
                        f"positional command input does not exist: {local}",
                    )
                )
            elif local is None:
                external = _external_candidate_path(
                    repo_root,
                    resolution.cwd,
                    value,
                    allow_plain=allow_plain,
                )
                if external is not None:
                    findings.append(
                        Finding(
                            block.path,
                            _source_line(block, offset),
                            "positional command input resolves outside "
                            "repository and cannot be validated: "
                            f"{external}",
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
        resolution = _resolve_shell_wrappers(
            raw_tokens,
            cwd=block.cwd,
        )
        tokens = list(resolution.tokens)
        if len(tokens) < 2 or Path(tokens[0]).name not in {"python", "python3"}:
            continue
        local = _resolved_repo_local_path(
            repo_root,
            resolution.cwd,
            tokens[1],
            allow_plain=True,
        )
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
        resolution = _resolve_shell_wrappers(
            raw_tokens,
            cwd=block.cwd,
        )
        tokens = list(resolution.tokens)
        if not tokens or not tokens[0].startswith(("./", "../")):
            continue
        local = _resolved_repo_local_path(
            repo_root,
            resolution.cwd,
            tokens[0],
            allow_plain=True,
        )
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
        resolution = _resolve_shell_wrappers(
            raw_tokens,
            cwd=block.cwd,
        )
        findings.extend(
            Finding(
                block.path,
                _source_line(block, offset),
                f"cannot statically resolve shell wrapper: {reason}",
            )
            for reason in resolution.uncertainty
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
