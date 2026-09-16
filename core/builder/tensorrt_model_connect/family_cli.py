# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discover family command declarations, parse values, and invoke one owner."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence


_ID = re.compile(r"[a-z][a-z0-9_]*\Z")
_COMMAND = re.compile(r"[a-z][a-z0-9_-]*\Z")
_HANDLER = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*:[a-z][a-z0-9_]*\Z")
_TYPES = {"string", "path", "int", "float", "bool"}
_MISSING = object()


def _keys(value: object, allowed: set[str], required: set[str]) -> None:
    if not isinstance(value, dict) or set(value) - allowed or required - set(value):
        raise ValueError(f"invalid declaration fields; expected {sorted(allowed)}")


def _path_spelling(spec: dict, value: str | Path) -> str:
    if isinstance(value, Path):
        for choice in spec.get("choices", []):
            if value == Path(choice):
                return choice
    return str(value)


def _scalar(spec: dict, value: object) -> object:
    kind = spec.get("type", "string")
    valid = {
        "string": isinstance(value, str),
        "path": isinstance(value, (str, Path)),
        "int": type(value) is int and -(1 << 63) <= value < (1 << 63),
        "float": type(value) in (int, float) and abs(value) <= sys.float_info.max and math.isfinite(value),
        "bool": type(value) is bool,
    }[kind]
    comparable = _path_spelling(spec, value) if kind == "path" and valid else value
    if not valid or ("choices" in spec and comparable not in spec["choices"]):
        raise ValueError(f"invalid value for {spec['name']!r}: {value!r}")
    return value


def _validate(document: object) -> dict:
    _keys(document, {"version", "commands"}, {"version", "commands"})
    if type(document["version"]) is not int or document["version"] != 1:
        raise ValueError("unsupported family CLI version")
    if not isinstance(document["commands"], list) or not document["commands"]:
        raise ValueError("commands must be a non-empty list")
    names: set[str] = set()
    for command in document["commands"]:
        _keys(command, {"name", "help", "executor", "handler", "arguments"},
              {"name", "executor", "handler", "arguments"})
        name = command["name"]
        if not isinstance(name, str) or not _COMMAND.fullmatch(name) or name in names:
            raise ValueError(f"invalid or duplicate command: {name!r}")
        names.add(name)
        executor, handler = command["executor"], command["handler"]
        pattern = _HANDLER if executor == "python" else _ID
        if not isinstance(executor, str) or executor not in {"python", "native"} or not isinstance(handler, str) or not pattern.fullmatch(handler):
            raise ValueError(f"invalid executor or family-local handler for {name!r}")
        if "help" in command and not isinstance(command["help"], str):
            raise ValueError("command help must be a string")
        if not isinstance(command["arguments"], list):
            raise ValueError("arguments must be a list")
        destinations: set[str] = set()
        flags = {"-h", "--help"}
        optional_positional = False
        for argument in command["arguments"]:
            _keys(argument, {"name", "flags", "type", "action", "required", "default", "choices", "help"}, {"name", "type"})
            dest = argument["name"]
            if not isinstance(dest, str) or not _ID.fullmatch(dest) or dest in destinations:
                raise ValueError(f"invalid or duplicate argument: {dest!r}")
            destinations.add(dest)
            aliases = argument.get("flags", [])
            if "flags" in argument and (not isinstance(aliases, list) or not aliases):
                raise ValueError("flags must be a non-empty list")
            for flag in aliases:
                if not isinstance(flag, str) or not re.fullmatch(r"--?[a-zA-Z][a-zA-Z0-9_-]*", flag) or flag in flags:
                    raise ValueError(f"invalid or duplicate option: {flag!r}")
                flags.add(flag)
            if not isinstance(argument["type"], str) or argument["type"] not in _TYPES:
                raise ValueError(f"invalid argument type for {dest!r}")
            action = argument.get("action")
            if "action" in argument and (not isinstance(action, str) or action not in {"store_true", "append"} or not aliases):
                raise ValueError(f"invalid argument action for {dest!r}")
            if action == "store_true" and argument.get("type", "bool") != "bool":
                raise ValueError("store_true requires bool type")
            if "required" in argument and type(argument["required"]) is not bool:
                raise ValueError("required must be a bool")
            if not aliases:
                if optional_positional and argument.get("required", True):
                    raise ValueError("required positional arguments must precede optional positionals")
                optional_positional |= not argument.get("required", True)
            if "help" in argument and not isinstance(argument["help"], str):
                raise ValueError("argument help must be a string")
            if "choices" in argument:
                if not isinstance(argument["choices"], list) or not argument["choices"]:
                    raise ValueError("choices must be a non-empty list")
                for choice in argument["choices"]:
                    _scalar({k: v for k, v in argument.items() if k != "choices"}, choice)
                if len(set(argument["choices"])) != len(argument["choices"]):
                    raise ValueError("choices must be unique")
            if "default" in argument:
                default = argument["default"]
                if action == "append":
                    if not isinstance(default, list):
                        raise ValueError("append default must be a list")
                    for item in default:
                        _scalar(argument, item)
                else:
                    _scalar({**argument, "type": "bool"} if action == "store_true" else argument, default)
    return document


def _root() -> Path:
    return Path(next(iter(importlib.import_module("families").__path__)))


def load_family_cli(family: str) -> dict | None:
    """Read one declaration without loading its implementation or dependencies."""
    if not _ID.fullmatch(family):
        raise ValueError(f"invalid family identifier: {family!r}")
    path = _root() / family / "cli.json"
    if not path.exists():
        return None
    try:
        return _validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError(f"invalid family CLI declaration {path}: {error}") from error


def discover() -> dict[str, dict]:
    """Load every installed declaration; malformed owners never fall back."""
    return {path.parent.name: load_family_cli(path.parent.name)
            for path in sorted(_root().glob("*/cli.json"))}


def _boolean(value: str) -> bool:
    if value not in {"true", "false"}:
        raise argparse.ArgumentTypeError("expected true or false")
    return value == "true"


def _integer(value: str) -> int:
    if not re.fullmatch(r"[+-]?[0-9]+", value):
        raise argparse.ArgumentTypeError("expected a decimal integer")
    return int(value)


def _floating(value: str) -> float:
    if not re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", value):
        raise argparse.ArgumentTypeError("expected a decimal number")
    return float(value)


class _Once(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(self, "used", False):
            parser.error(f"option {option_string} may be supplied only once")
        self.used = True
        setattr(namespace, self.dest, True if self.nargs == 0 else values)


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def _get_help_string(self, action):
        if action.default is _MISSING:
            return action.help
        return super()._get_help_string(action)


def _add_arguments(parser: argparse.ArgumentParser, command: dict) -> None:
    types = {"string": str, "path": str, "int": _integer, "float": _floating, "bool": _boolean}
    for argument in command["arguments"]:
        flags = argument.get("flags")
        action = argument.get("action")
        kwargs = {key: argument[key] for key in ("choices",) if key in argument}
        kwargs["help"] = argument.get("help", f"Value of type {argument['type']}")
        kwargs["default"] = argument.get("default", argparse.SUPPRESS)
        if "default" in argument and argument["type"] == "float":
            default = argument["default"]
            kwargs["default"] = [float(value) for value in default] if action == "append" else float(default)
        if flags:
            kwargs.update(dest=argument["name"], required=argument.get("required", False))
            kwargs["action"] = "append" if action == "append" else _Once
        elif not argument.get("required", True):
            kwargs["nargs"] = "?"
            if "default" not in argument:
                kwargs["default"] = _MISSING
        if action == "store_true":
            kwargs["nargs"] = 0
        else:
            kwargs["type"] = types[argument.get("type", "string")]
        parser.add_argument(*(flags or [argument["name"]]), **kwargs)


def serialize_arguments(command: dict, values: Mapping[str, object]) -> list[str]:
    """Encode declared values as argv using exactly the parsing contract."""
    specs = {item["name"]: item for item in command["arguments"]}
    if unknown := set(values) - set(specs):
        raise ValueError(f"unknown command arguments: {', '.join(sorted(unknown))}")
    argv: list[str] = []
    positionals: list[tuple[str, str]] = []
    for name, spec in specs.items():
        flags = spec.get("flags", [])
        value = values.get(name)
        if value is None:
            if spec.get("required", not flags):
                raise ValueError(f"missing required argument: {name}")
            continue
        action = spec.get("action")
        flag = next((flag for flag in flags if flag.startswith("--")), flags[0] if flags else None)
        if action == "store_true":
            _scalar({**spec, "type": "bool"}, value)
            if value:
                argv.append(flag)
            elif spec.get("default") is not False or spec.get("required", False):
                raise ValueError(f"{name} cannot represent false with this declaration")
            continue
        items = value if action == "append" else [value]
        if action == "append" and not isinstance(items, (list, tuple)):
            raise ValueError(f"{name} must be a list")
        if action == "append":
            prefix = spec.get("default", [])
            for item in items:
                _scalar(spec, item)
            normalize = float if spec["type"] == "float" else lambda value: value
            same_prefix = len(items) >= len(prefix) and all(
                value == Path(default) if spec["type"] == "path" and isinstance(value, Path)
                else normalize(value) == normalize(default)
                for value, default in zip(items, prefix)
            )
            if not same_prefix:
                raise ValueError(f"{name} cannot replace the declared append default")
            items = items[len(prefix):]
            if spec.get("required", False) and not items:
                raise ValueError(f"missing required argument: {name}")
        for item in items:
            _scalar(spec, item)
            encoded = _path_spelling(spec, item) if spec["type"] == "path" else str(item).lower() if type(item) is bool else str(item)
            if flag:
                argv.append(f"{flag}={encoded}")
            else:
                positionals.append((name, encoded))
    positional_specs = [spec for spec in specs.values() if "flags" not in spec]
    position = 0
    for index, spec in enumerate(positional_specs):
        required_after = sum(item.get("required", True) for item in positional_specs[index + 1:])
        if position == len(positionals) or (not spec.get("required", True) and len(positionals) - position <= required_after):
            continue
        if positionals[position][0] != spec["name"]:
            raise ValueError("cannot omit a positional argument before another optional positional")
        position += 1
    return argv + (["--", *(value for _, value in positionals)] if positionals else [])


def _ordered_arguments(parser: argparse.ArgumentParser, command: dict, arguments: Sequence[str]) -> list[str]:
    """Keep argparse and the native parser identical for interspersed options."""
    flags = {flag: argument for argument in command["arguments"] for flag in argument.get("flags", [])}
    for token in arguments:
        if token == "--":
            break
        if token in {"-h", "--help"}:
            parser.print_help()
            parser.exit()
    options, positionals = [], []
    tokens = iter(arguments)
    for token in tokens:
        if token == "--":
            positionals.extend(tokens)
            break
        if token.startswith("-") and len(token) > 1 and not re.fullmatch(r"-([0-9]+|[0-9]*\.[0-9]+)", token):
            spec = flags.get(token.split("=", 1)[0])
            if spec is None:
                parser.error(f"unknown argument: {token}")
            options.append(token)
            if "=" not in token and spec.get("action") != "store_true":
                value = next(tokens, None)
                if value is None:
                    parser.error(f"{token} requires a value")
                if value.startswith("-") and len(value) > 1:
                    try:
                        if spec["type"] not in {"int", "float"}:
                            raise ValueError("leading-dash text requires flag=value")
                        parsed = _integer(value) if spec["type"] == "int" else _floating(value)
                        _scalar(spec, parsed)
                    except (ValueError, argparse.ArgumentTypeError) as error:
                        parser.error(f"{token} requires a value: {error}")
                options[-1] = f"{token}={value}"
        else:
            positionals.append(token)
    return options + (["--", *positionals] if positionals else [])


def main(argv: Sequence[str], declarations: dict[str, dict] | None = None) -> int:
    """Parse one family command and load only its declared handler."""
    if declarations is None:
        declaration = load_family_cli(argv[0]) if argv and _ID.fullmatch(argv[0]) else None
        declarations = {argv[0]: declaration} if declaration is not None else {}
    if not argv or argv[0] not in declarations:
        raise ValueError("expected a family with a CLI declaration")
    family = argv[0]
    parser = argparse.ArgumentParser(prog=f"trtmc {family}", allow_abbrev=False)
    commands = parser.add_subparsers(dest="_command", required=True)
    by_name = {command["name"]: command for command in declarations[family]["commands"]}
    parsers = {}
    for name, command in by_name.items():
        child = commands.add_parser(name, help=command.get("help"), allow_abbrev=False,
                                    formatter_class=_HelpFormatter)
        _add_arguments(child, command)
        parsers[name] = child
    if len(argv) == 1:
        parser.print_help()
        return 0
    if argv[1] not in by_name:
        parser.parse_args(argv[1:])  # Emits the ordinary help or unknown-command diagnostic.
        raise AssertionError("unknown command was accepted")
    command = by_name[argv[1]]
    child = parsers[argv[1]]
    arguments = _ordered_arguments(child, command, argv[2:])
    values = {name: value for name, value in vars(child.parse_args(arguments)).items() if value is not _MISSING}
    serialize_arguments(command, values)  # Also rejects non-finite numbers.
    if command["executor"] == "native":
        native = Path(__file__).resolve().parent / "bin" / "trtmc"
        try:
            os.execv(str(native), [str(native), *argv])
        except OSError as error:
            print(f"Error: cannot execute packaged trtmc: {error}", file=sys.stderr)
            return 1
        raise AssertionError("execv returned without replacing the process")
    module, name = command["handler"].split(":")
    handler = getattr(importlib.import_module(f"families.{family}.{module}"), name, None)
    if not callable(handler):
        raise ValueError(f"family {family!r} does not provide handler {command['handler']!r}")
    for argument in command["arguments"]:
        name = argument["name"]
        if argument["type"] == "path" and name in values:
            values[name] = [Path(value) for value in values[name]] if argument.get("action") == "append" else Path(values[name])
    result = handler(**values)
    if type(result) is not int:
        raise TypeError("family CLI handler must return an integer exit status")
    return result
