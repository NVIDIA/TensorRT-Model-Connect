#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate shell examples in Git-tracked Markdown documentation.

The checker is deliberately non-executing: it parses every ``bash``, ``sh``,
or ``shell`` fenced block with ``bash -n``; verifies repository-local scripts,
search/test inputs, and explicit positional inputs; and checks statically
discoverable CLI subcommands, option scope/arity, choices, and required inputs.
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
from dataclasses import dataclass
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


@dataclass(frozen=True)
class PositionalSpec:
    """Static positional-argument contract."""

    name: str
    min_values: int = 1
    max_values: int | None = 1


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


def _constant(node: ast.AST | None) -> object | None:
    if isinstance(node, ast.Constant):
        return node.value
    return None


def _call_keyword(call: ast.Call, name: str) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _argument_spec(call: ast.Call) -> OptionSpec | PositionalSpec | None:
    names = tuple(
        argument.value
        for argument in call.args
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
    )
    if not names:
        return None

    action = _constant(_call_keyword(call, "action"))
    nargs = _constant(_call_keyword(call, "nargs"))
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

    literal_choices = _literal_choices(_call_keyword(call, "choices") or ast.Constant(None))
    is_option = any(name.startswith("-") for name in names)
    if not is_option:
        return PositionalSpec(names[0], min_values, max_values)

    aliases = tuple(name for name in names if name.startswith("-"))
    return OptionSpec(
        aliases=aliases,
        min_values=min_values,
        max_values=max_values,
        choices=frozenset(literal_choices),
        required=_constant(_call_keyword(call, "required")) is True,
    )


def _assigned_name(node: ast.Assign | ast.AnnAssign) -> str | None:
    if isinstance(node, ast.Assign):
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return None
        return node.targets[0].id
    return node.target.id if isinstance(node.target, ast.Name) else None


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


@lru_cache(maxsize=None)
def _argparse_program_contract(script_path: Path) -> ProgramSpec | None:
    """Extract root/subcommand argparse contracts without importing a script."""
    if not script_path.is_file():
        return None
    try:
        tree = ast.parse(script_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None

    assignments = [
        assigned for node in ast.walk(tree) if (assigned := _assigned_call(node)) is not None
    ]
    parser_variables: dict[str, str | None] = {}
    for variable, call in assignments:
        if _is_call_named(call, "ArgumentParser"):
            parser_variables[variable] = None
    subparser_variables: dict[str, str | None] = {}
    for variable, call in assignments:
        if not (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "add_subparsers"
            and isinstance(call.func.value, ast.Name)
        ):
            continue
        owner = call.func.value.id
        if owner in parser_variables:
            subparser_variables[variable] = parser_variables[owner]
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_parser"
            and isinstance(node.func.value, ast.Name)
        ):
            subparser_variables.setdefault(node.func.value.id, None)

    commands: dict[str, CommandSpec] = {}
    for variable, call in assignments:
        if not (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "add_parser"
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id in subparser_variables
            and call.args
            and isinstance(call.args[0], ast.Constant)
            and isinstance(call.args[0].value, str)
        ):
            continue
        command = call.args[0].value
        parser_variables[variable] = command
        commands.setdefault(command, CommandSpec.empty())

    argument_containers = dict(parser_variables)
    mutually_exclusive_groups: dict[str, tuple[str | None, bool]] = {}
    unresolved_groups = True
    while unresolved_groups:
        unresolved_groups = False
        for variable, call in assignments:
            if (
                variable in argument_containers
                or not isinstance(call.func, ast.Attribute)
                or call.func.attr not in {"add_argument_group", "add_mutually_exclusive_group"}
                or not isinstance(call.func.value, ast.Name)
            ):
                continue
            owner = call.func.value.id
            if owner not in argument_containers:
                continue
            command = argument_containers[owner]
            argument_containers[variable] = command
            if call.func.attr == "add_mutually_exclusive_group":
                mutually_exclusive_groups[variable] = (
                    command,
                    _constant(_call_keyword(call, "required")) is True,
                )
            unresolved_groups = True

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_parser"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in subparser_variables
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            continue
        commands.setdefault(node.args[0].value, CommandSpec.empty())

    if not parser_variables and not commands:
        return None

    root = CommandSpec.empty()
    mutually_exclusive_options: dict[str, list[OptionSpec]] = {
        variable: [] for variable in mutually_exclusive_groups
    }
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and isinstance(node.func.value, ast.Name)
        ):
            continue
        owner = node.func.value.id
        if owner not in argument_containers:
            continue
        argument = _argument_spec(node)
        if argument is None:
            continue
        command = argument_containers[owner]
        target = root if command is None else commands.setdefault(command, CommandSpec.empty())
        if isinstance(argument, OptionSpec):
            for alias in argument.aliases:
                target.options[alias] = argument
            if owner in mutually_exclusive_options:
                mutually_exclusive_options[owner].append(argument)
        else:
            target.positionals.append(argument)

    for variable, (command, required) in mutually_exclusive_groups.items():
        options = mutually_exclusive_options[variable]
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
        target = root if command is None else commands.setdefault(command, CommandSpec.empty())
        target.required_any.append((aliases, description))

    help_option = OptionSpec(aliases=("-h", "--help"), min_values=0, max_values=0)
    root.options.setdefault("-h", help_option)
    root.options.setdefault("--help", help_option)
    command_required = any(
        _constant(_call_keyword(call, "required")) is True
        for _variable, call in assignments
        if isinstance(call.func, ast.Attribute) and call.func.attr == "add_subparsers"
    )
    return ProgramSpec(
        root=root,
        commands=commands,
        command_required=command_required,
    )


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


def _parse_command_arguments(
    tokens: list[str],
    spec: CommandSpec,
) -> tuple[list[str], set[str], list[tuple[str, str]], list[str]]:
    """Parse tokens by static arity.

    Returns positional values, seen option aliases, choice violations, and
    human-readable arity/unknown-option errors. Known option values are
    consumed even when they begin with ``-``; an exact known option spelling
    still starts the next option.
    """
    positionals: list[str] = []
    seen_options: set[str] = set()
    choice_errors: list[tuple[str, str]] = []
    errors: list[str] = []
    index = 0
    positional_only = False
    while index < len(tokens):
        token = tokens[index]
        if token == "--" and not positional_only:
            positional_only = True
            index += 1
            continue
        option_name, separator, inline_value = token.partition("=")
        option = None if positional_only else spec.options.get(option_name)
        if option is None:
            if (
                not positional_only
                and token.startswith("-")
                and not re.match(r"^-\d", token)
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
        if separator:
            if not option.allow_inline_value:
                errors.append(f"does not support inline value:{option_name}")
            elif option.max_values == 0:
                errors.append(f"does not take a value:{option_name}")
            elif inline_value:
                values.append(inline_value)
            else:
                errors.append(f"requires a value:{option_name}")
        elif option.max_values != 0:
            cursor = index + 1
            while cursor < len(tokens) and (
                option.max_values is None or len(values) < option.max_values
            ):
                candidate = tokens[cursor]
                candidate_option = candidate.partition("=")[0]
                if candidate_option in spec.options:
                    break
                if (
                    candidate.startswith("-")
                    and not re.match(r"^-\d", candidate)
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
                    choice_errors.append((option_name, value))
        index += 1

    return positionals, seen_options, choice_errors, errors


def _check_command_spec(
    block: ShellBlock,
    offset: int,
    tokens: list[str],
    spec: CommandSpec,
    *,
    label: str,
    unknown_template: str,
) -> list[Finding]:
    positionals, seen_options, choice_errors, errors = _parse_command_arguments(tokens, spec)
    line = _source_line(block, offset)
    findings: list[Finding] = []
    for error in errors:
        kind, option = error.split(":", 1)
        if kind == "unknown option":
            message = unknown_template.format(option=option)
        elif kind == "does not support inline value":
            message = f"option for {label} does not support `=` form: {option}"
        elif kind == "does not take a value":
            message = f"option for {label} does not take a value: {option}"
        else:
            message = f"option for {label} requires a value: {option}"
        findings.append(Finding(block.path, line, message))

    for option, value in choice_errors:
        choices = spec.options[option].choices
        choice_label = (
            f"`{label.strip('`')} {option}`"
            if label.startswith("`") and label.endswith("`")
            else f"{label} {option}"
        )
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

    minimum_positionals = sum(positional.min_values for positional in spec.positionals)
    if len(positionals) < minimum_positionals:
        missing = next(
            (
                positional.name
                for position, positional in enumerate(spec.positionals)
                if position >= len(positionals) and positional.min_values
            ),
            "positional argument",
        )
        findings.append(
            Finding(
                block.path,
                line,
                f"missing required positional for {label}: {missing}",
            )
        )
    if (
        spec.positionals
        and not any(error.startswith("unknown option:") for error in errors)
        and all(positional.max_values is not None for positional in spec.positionals)
    ):
        maximum_positionals = sum(positional.max_values or 0 for positional in spec.positionals)
        if len(positionals) > maximum_positionals:
            findings.append(
                Finding(
                    block.path,
                    line,
                    f"unexpected positional argument for {label}: "
                    f"{positionals[maximum_positionals]}",
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
    """Validate ``python -m tensorrt_model_connect`` examples."""
    program = _python_cli_contract(repo_root)
    if program is None:
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
        if command not in program.commands:
            findings.append(
                Finding(
                    block.path,
                    _source_line(block, offset),
                    f"unknown `python -m tensorrt_model_connect` subcommand: {command}",
                )
            )
            continue
        spec = _merge_command_specs(program.root, program.commands[command])
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
    return findings


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
        for value in positionals[: len(spec.positionals)]:
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
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in commands:
            return token, index
        option_name, separator, _inline_value = token.partition("=")
        option = root.options.get(option_name)
        if option is not None:
            index += 1
            if not separator and option.max_values != 0:
                consumed = 0
                while index < len(tokens) and (
                    option.max_values is None or consumed < option.max_values
                ):
                    candidate_option = tokens[index].partition("=")[0]
                    if candidate_option in root.options:
                        break
                    index += 1
                    consumed += 1
            continue
        if token.startswith("-"):
            index += 1
            continue
        return None, index
    return None, None


def _check_argparse_invocation(
    block: ShellBlock,
    offset: int,
    tokens: list[str],
    local: str,
    contract: ProgramSpec,
) -> list[Finding]:
    """Validate an argparse script, including a selected subcommand."""
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
    )
    spec = _merge_command_specs(contract.root, contract.commands[command])
    findings.extend(
        _check_command_spec(
            block,
            offset,
            tokens[command_index + 1 :],
            spec,
            label=f"`{local} {command}`",
            unknown_template=f"unknown option for `{local} {command}`: {{option}}",
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
            findings.extend(check_positional_inputs(block, repo_root))
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
    print("\nAll documentation shell examples passed syntax, CLI-contract, and local-input checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
