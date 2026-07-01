#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inline family-fixed strategy switches and remove unsupported branches."""

from __future__ import annotations

import argparse
import ast
import copy
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from tools import family_specialization
except ModuleNotFoundError:  # Direct execution puts tools/ on sys.path.
    import family_specialization


class _ConstantFolder(ast.NodeTransformer):
    def __init__(self, constants: dict[str, Any]):
        self.constants = constants

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id in self.constants:
            return ast.copy_location(ast.Constant(self.constants[node.id]), node)
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        node = self.generic_visit(node)
        if isinstance(node.op, ast.Not) and isinstance(node.operand, ast.Constant):
            return ast.copy_location(ast.Constant(not node.operand.value), node)
        return node

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        node = self.generic_visit(node)
        values = list(node.values)
        if isinstance(node.op, ast.And):
            if any(
                isinstance(value, ast.Constant) and not bool(value.value)
                for value in values
            ):
                return ast.copy_location(ast.Constant(False), node)
            values = [
                value
                for value in values
                if not (isinstance(value, ast.Constant) and bool(value.value))
            ]
        else:
            if any(
                isinstance(value, ast.Constant) and bool(value.value)
                for value in values
            ):
                return ast.copy_location(ast.Constant(True), node)
            values = [
                value
                for value in values
                if not (isinstance(value, ast.Constant) and not bool(value.value))
            ]
        if not values:
            return ast.copy_location(
                ast.Constant(isinstance(node.op, ast.And)), node
            )
        if len(values) == 1:
            return values[0]
        node.values = values
        return node

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        node = self.generic_visit(node)
        operands = [node.left, *node.comparators]
        if not all(isinstance(operand, ast.Constant) for operand in operands):
            return node
        value = eval(  # noqa: S307 - expression is restricted to AST constants.
            compile(ast.Expression(node), "<family-switch>", "eval"),
            {"__builtins__": {}},
            {},
        )
        return ast.copy_location(ast.Constant(bool(value)), node)

    def visit_If(self, node: ast.If) -> ast.AST | list[ast.stmt]:
        node = self.generic_visit(node)
        if isinstance(node.test, ast.Constant):
            return node.body if bool(node.test.value) else node.orelse
        return node

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        node = self.generic_visit(node)
        if isinstance(node.test, ast.Constant):
            return node.body if bool(node.test.value) else node.orelse
        return node


def _prune_terminal_tails(statements: list[ast.stmt]) -> list[ast.stmt]:
    """Remove statements that became unreachable after constant folding."""
    result: list[ast.stmt] = []
    for statement in statements:
        for attribute in ("body", "orelse", "finalbody"):
            child = getattr(statement, attribute, None)
            if isinstance(child, list):
                setattr(statement, attribute, _prune_terminal_tails(child))
        if isinstance(statement, ast.Try):
            for handler in statement.handlers:
                handler.body = _prune_terminal_tails(handler.body)
        if isinstance(statement, ast.Match):
            for case in statement.cases:
                case.body = _prune_terminal_tails(case.body)
        result.append(statement)
        if isinstance(statement, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
            break
    return result


def _function_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ""


def _offsets(text: str) -> list[int]:
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _span(node: ast.AST, offsets: list[int]) -> tuple[int, int]:
    start = offsets[node.lineno - 1] + node.col_offset
    end = offsets[node.end_lineno - 1] + node.end_col_offset
    return start, end


def _remove_call_keywords(
    path: Path,
    calls: dict[tuple[int, str], set[str]],
) -> None:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    offsets = _offsets(text)
    replacements: list[tuple[int, int, str]] = []
    found: set[tuple[int, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = _function_name(node)
        key = (node.lineno, function)
        parameters = calls.get(key)
        if not parameters:
            continue
        updated = copy.deepcopy(node)
        present = {keyword.arg for keyword in updated.keywords}
        if not parameters <= present:
            missing = ", ".join(sorted(parameters - present))
            raise ValueError(f"Missing audited call keywords in {path}:{node.lineno}: {missing}")
        updated.keywords = [
            keyword for keyword in updated.keywords if keyword.arg not in parameters
        ]
        replacements.append((*_span(node, offsets), ast.unparse(updated)))
        found.add(key)
    missing_calls = set(calls) - found
    if missing_calls:
        raise ValueError(f"Audited calls no longer found in {path}: {sorted(missing_calls)}")
    for start, end, replacement in sorted(replacements, reverse=True):
        text = text[:start] + replacement + text[end:]
    path.write_text(text, encoding="utf-8")


def _specialize_definitions(
    path: Path,
    functions: dict[str, dict[str, Any]],
) -> None:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    offsets = _offsets(text)
    replacements: list[tuple[int, int, str]] = []
    found: set[str] = set()
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in functions:
            continue
        constants = functions[node.name]
        kwonly_names = [argument.arg for argument in node.args.kwonlyargs]
        positional_names = [argument.arg for argument in node.args.args]
        defaulted_positional = set(
            positional_names[len(positional_names) - len(node.args.defaults):]
        )
        if not set(constants) <= set(kwonly_names) | defaulted_positional:
            raise ValueError(
                "Strategy parameters must be keyword-only or defaulted "
                f"in {path}:{node.lineno}"
            )
        transformed = copy.deepcopy(node)
        positional_defaults = [
            None
        ] * (len(transformed.args.args) - len(transformed.args.defaults)) + list(
            transformed.args.defaults
        )
        kept_positional = [
            (argument, default)
            for argument, default in zip(
                transformed.args.args, positional_defaults
            )
            if argument.arg not in constants
        ]
        transformed.args.args = [argument for argument, _ in kept_positional]
        transformed.args.defaults = [
            default for _, default in kept_positional if default is not None
        ]
        kept_args = []
        kept_defaults = []
        for argument, default in zip(
            transformed.args.kwonlyargs, transformed.args.kw_defaults
        ):
            if argument.arg not in constants:
                kept_args.append(argument)
                kept_defaults.append(default)
        transformed.args.kwonlyargs = kept_args
        transformed.args.kw_defaults = kept_defaults
        transformed = _ConstantFolder(constants).visit(transformed)
        transformed.body = _prune_terminal_tails(transformed.body)
        ast.fix_missing_locations(transformed)
        replacements.append((*_span(node, offsets), ast.unparse(transformed)))
        found.add(node.name)
    missing = set(functions) - found
    if missing:
        raise ValueError(f"Audited definitions no longer found in {path}: {sorted(missing)}")
    for start, end, replacement in sorted(replacements, reverse=True):
        text = text[:start] + replacement + text[end:]
    path.write_text(text, encoding="utf-8")


def apply_report(repo_root: Path, report: dict[str, Any]) -> int:
    repo_root = repo_root.resolve()
    changed = 0
    for family in report["families"]:
        switches = family["fixed_strategy_switches"]
        if not switches:
            continue
        family_dir = (
            repo_root
            / "python/tensorrt_model_connect/families"
            / family["family"]
        )
        module_paths = {
            row["module"]: family_dir / row["path"]
            for row in family["modules"]
        }
        call_edits: dict[Path, dict[tuple[int, str], set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        definitions: dict[Path, dict[str, dict[str, Any]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        for switch in switches:
            for call in switch["call_sites"]:
                call_edits[family_dir / call["path"]][
                    (call["line"], switch["function"])
                ].add(switch["parameter"])
            for module in switch["definitions"]:
                definitions[module_paths[module]][switch["function"]][
                    switch["parameter"]
                ] = switch["value"]
        for path, calls in call_edits.items():
            _remove_call_keywords(path, calls)
        for path, functions in definitions.items():
            _specialize_definitions(path, functions)
        changed += len(switches)
    return changed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "apply"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--family", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    report = family_specialization.audit_repo(repo_root, tuple(args.family))
    switches = [
        (family["family"], switch)
        for family in report["families"]
        for switch in family["fixed_strategy_switches"]
    ]
    if args.command == "check":
        for family, switch in switches:
            print(
                f"{family}: {switch['function']}.{switch['parameter']}="
                f"{switch['value']!r}"
            )
        return 1 if switches else 0
    changed = 0
    for _ in range(20):
        iteration = family_specialization.audit_repo(
            repo_root, tuple(args.family)
        )
        pending = sum(
            len(family["fixed_strategy_switches"])
            for family in iteration["families"]
        )
        if not pending:
            break
        changed += apply_report(repo_root, iteration)
    else:
        raise RuntimeError("Family switch specialization did not converge")
    print(f"specialized_family_switches={changed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
