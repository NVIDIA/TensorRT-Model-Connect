#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Require public source surfaces to have an explicit documentation home.

The checker deliberately extracts exact source tokens instead of guessing from
prefixes.  Every discovered token must either occur in the configured
documentation page or have a narrow, reasoned allowlist entry.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_MAPPING_PATH = Path("tools/doc_public_surfaces.json")

_CPP_LONG_OPTION_RE = re.compile(
    r'"(--[A-Za-z0-9][A-Za-z0-9_-]*)(?:=)?"'
)
_LONG_OPTION_RE = re.compile(r"--[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_CPP_SCHEMA_RE = re.compile(
    r"\breturn\s+Schema\s*\{\s*\"(?P<namespace>[A-Za-z0-9_]+)\""
)
_CPP_SCHEMA_FIELD_RE = re.compile(
    r"\bConfigField\s*\{\s*\"(?P<field>[A-Za-z0-9_]+)\""
)


@dataclass(frozen=True)
class Finding:
    category: str
    token: str
    message: str

    def __str__(self) -> str:
        label = f"{self.category}:{self.token}" if self.token else self.category
        return f"ERROR [{label}] {self.message}"


@dataclass
class CheckReport:
    findings: list[Finding] = field(default_factory=list)
    surface_count: int = 0
    mapping_count: int = 0
    allowlisted_count: int = 0

    @property
    def ok(self) -> bool:
        return not self.findings


def extract_cpp_long_options(source: str) -> set[str]:
    """Return complete long-option string literals from C++ source."""
    return set(_CPP_LONG_OPTION_RE.findall(source))


def _function_node(
    source: str, function_name: str
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return node
    raise ValueError(f"function {function_name!r} was not found")


def extract_python_function_parameters(source: str, function_name: str) -> set[str]:
    """Extract positional and keyword-only parameters from one top-level function."""
    node = _function_node(source, function_name)
    parameters = [
        *node.args.posonlyargs,
        *node.args.args,
        *node.args.kwonlyargs,
    ]
    return {parameter.arg for parameter in parameters if parameter.arg not in {"self", "cls"}}


def extract_argparse_long_options(source: str, function_name: str) -> set[str]:
    """Extract exact long options passed to ``add_argument`` in one function."""
    node = _function_node(source, function_name)
    options: set[str] = set()
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        if not isinstance(call.func, ast.Attribute) or call.func.attr != "add_argument":
            continue
        for argument in call.args:
            if (
                isinstance(argument, ast.Constant)
                and isinstance(argument.value, str)
                and _LONG_OPTION_RE.fullmatch(argument.value)
            ):
                options.add(argument.value)
    return options


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def extract_python_schemas(source: str) -> dict[str, set[str]]:
    """Extract literal namespace/field contracts from Python ``Schema`` calls."""
    tree = ast.parse(source)
    schemas: dict[str, set[str]] = {}
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call) or _call_name(call.func) != "Schema":
            continue
        keywords = {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg}
        namespace_node = keywords.get("namespace")
        fields_node = keywords.get("fields")
        if not (
            isinstance(namespace_node, ast.Constant)
            and isinstance(namespace_node.value, str)
            and fields_node is not None
        ):
            continue
        fields: set[str] = set()
        for field_call in ast.walk(fields_node):
            if not isinstance(field_call, ast.Call):
                continue
            constructor = _call_name(field_call.func)
            if constructor != "ConfigField" and not constructor.endswith("_field"):
                continue
            field_keywords = {
                keyword.arg: keyword.value for keyword in field_call.keywords if keyword.arg
            }
            name_node = field_keywords.get("name")
            if name_node is None and field_call.args:
                name_node = field_call.args[0]
            if isinstance(name_node, ast.Constant) and isinstance(name_node.value, str):
                fields.add(name_node.value)
        schemas[namespace_node.value] = fields
    return schemas


def extract_cpp_schemas(source: str) -> dict[str, set[str]]:
    """Extract a literal C++ ``Schema`` namespace and its ``ConfigField`` names."""
    match = _CPP_SCHEMA_RE.search(source)
    if match is None:
        return {}
    namespace = match.group("namespace")
    return {
        namespace: {
            field_match.group("field")
            for field_match in _CPP_SCHEMA_FIELD_RE.finditer(source)
        }
    }


def _find_cpp_type_body(source: str, kind: str, type_name: str) -> str:
    declaration = re.search(rf"\b{re.escape(kind)}\s+{re.escape(type_name)}\b", source)
    if declaration is None:
        raise ValueError(f"{kind} {type_name!r} was not found")
    opening = source.find("{", declaration.end())
    if opening < 0:
        raise ValueError(f"{kind} {type_name!r} has no body")

    depth = 0
    state = "code"
    index = opening
    while index < len(source):
        char = source[index]
        next_char = source[index + 1] if index + 1 < len(source) else ""
        if state == "line_comment":
            if char == "\n":
                state = "code"
        elif state == "block_comment":
            if char == "*" and next_char == "/":
                state = "code"
                index += 1
        elif state == "string":
            if char == "\\":
                index += 1
            elif char == '"':
                state = "code"
        elif state == "character":
            if char == "\\":
                index += 1
            elif char == "'":
                state = "code"
        elif char == "/" and next_char == "/":
            state = "line_comment"
            index += 1
        elif char == "/" and next_char == "*":
            state = "block_comment"
            index += 1
        elif char == '"':
            state = "string"
        elif char == "'":
            state = "character"
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[opening + 1 : index]
        index += 1
    raise ValueError(f"{kind} {type_name!r} has an unterminated body")


def extract_cpp_virtual_methods(source: str, class_name: str) -> set[str]:
    """Extract virtual method names from one C++ class body."""
    body = _find_cpp_type_body(source, "class", class_name)
    methods: set[str] = set()
    for match in re.finditer(r"\bvirtual\b", body):
        signature = body[match.end() :]
        opening = signature.find("(")
        if opening < 0 or opening > 500:
            continue
        prefix = signature[:opening]
        if ";" in prefix or "{" in prefix:
            continue
        name_match = re.search(r"(~?[A-Za-z_][A-Za-z0-9_]*)\s*\Z", prefix)
        if name_match is None:
            continue
        name = name_match.group(1)
        if name != f"~{class_name}":
            methods.add(name)
    return methods


def _strip_cpp_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", " ", source, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", " ", source)


def extract_cpp_struct_fields(source: str, struct_name: str) -> set[str]:
    """Extract data-member names from a plain public C++ struct."""
    body = _strip_cpp_comments(_find_cpp_type_body(source, "struct", struct_name))
    fields: set[str] = set()
    for statement in body.split(";"):
        statement = statement.strip()
        if not statement or "(" in statement:
            continue
        match = re.search(
            r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?:\{.*\})?\s*\Z",
            statement,
            flags=re.DOTALL,
        )
        if match is not None:
            fields.add(match.group("name"))
    return fields


def _mask_nested_cpp_blocks(body: str) -> str:
    output: list[str] = []
    depth = 0
    for char in body:
        if char == "{":
            depth += 1
            output.append(" ")
        elif char == "}":
            depth = max(0, depth - 1)
            output.append(" ")
        elif depth:
            output.append("\n" if char == "\n" else " ")
        else:
            output.append(char)
    return "".join(output)


def extract_cpp_top_level_public_methods(source: str, class_name: str) -> set[str]:
    """Extract non-special public methods declared directly on a C++ class."""
    body = _strip_cpp_comments(_find_cpp_type_body(source, "class", class_name))
    top_level = _mask_nested_cpp_blocks(body)
    visibility = "private"
    methods: set[str] = set()
    for raw_statement in top_level.split(";"):
        statement = raw_statement
        labels = list(re.finditer(r"\b(public|private|protected)\s*:", statement))
        if labels:
            visibility = labels[-1].group(1)
            statement = statement[labels[-1].end() :]
        if visibility != "public" or "(" not in statement:
            continue
        prefix = statement[: statement.find("(")]
        match = re.search(r"(~?[A-Za-z_][A-Za-z0-9_]*)\s*\Z", prefix)
        if match is None:
            continue
        name = match.group(1)
        if name not in {class_name, f"~{class_name}", "operator"}:
            methods.add(name)
    return methods


def _merge_schema_tokens(schemas: Mapping[str, Iterable[str]]) -> set[str]:
    tokens: set[str] = set()
    for namespace, fields in schemas.items():
        tokens.add(namespace)
        tokens.update(f"{namespace}.{field}" for field in fields)
    return tokens


def collect_public_surfaces(repo_root: Path) -> dict[str, set[str]]:
    """Collect every public surface covered by this gate from authoritative source."""
    def text(relative: str) -> str:
        return (repo_root / relative).read_text(encoding="utf-8")

    pipeline_header = text("include/trtmc/pipeline.h")
    schemas: dict[str, set[str]] = {}
    python_schema_paths = sorted(
        (repo_root / "python/tensorrt_model_connect/runtime_config/schemas").glob("*.py")
    )
    python_schema_paths.extend(
        sorted(
            (
                repo_root / "python/tensorrt_model_connect/families"
            ).glob("*/runtime_config_schema.py")
        )
    )
    for path in python_schema_paths:
        for namespace, fields in extract_python_schemas(
            path.read_text(encoding="utf-8")
        ).items():
            schemas.setdefault(namespace, set()).update(fields)

    cpp_schema_paths = sorted((repo_root / "src/runtime/config/schemas").glob("*.cpp"))
    cpp_schema_paths.extend(
        sorted((repo_root / "src/runtime/models").glob("*/config_schema.cpp"))
    )
    for path in cpp_schema_paths:
        for namespace, fields in extract_cpp_schemas(
            path.read_text(encoding="utf-8")
        ).items():
            schemas.setdefault(namespace, set()).update(fields)

    workflows = {
        path.relative_to(repo_root).as_posix()
        for pattern in ("*.yml", "*.yaml")
        for path in (repo_root / ".github/workflows").glob(pattern)
    }

    return {
        "native_cli_options": extract_cpp_long_options(text("src/cli/args.cpp")),
        "python_build_parameters": extract_python_function_parameters(
            text("python/tensorrt_model_connect/engine_builder.py"), "build"
        ),
        "benchmark_cli_options": extract_argparse_long_options(
            text("python/tensorrt_model_connect/benchmark/cli.py"), "build_parser"
        ),
        "cpp_pipeline_methods": extract_cpp_virtual_methods(pipeline_header, "IPipeline"),
        "generate_config_fields": extract_cpp_struct_fields(
            pipeline_header, "GenerateConfig"
        ),
        "pipeline_pool_methods": extract_cpp_top_level_public_methods(
            text("include/trtmc/runtime/pipeline_pool.h"), "PipelinePool"
        ),
        "config_schema_entries": _merge_schema_tokens(schemas),
        "github_workflows": workflows,
    }


def load_mapping(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load {path}: {error}") from error
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError(f"{path}: expected an object with version 1")
    if not isinstance(data.get("surfaces"), dict):
        raise ValueError(f"{path}: surfaces must be an object")
    return data


def _token_present(content: str, token: str) -> bool:
    if token.startswith("--"):
        pattern = rf"(?<![A-Za-z0-9_-]){re.escape(token)}(?![A-Za-z0-9_-])"
    elif "/" in token:
        pattern = re.escape(token)
    elif "." in token:
        pattern = rf"(?<![A-Za-z0-9_.]){re.escape(token)}(?![A-Za-z0-9_.])"
    else:
        # A public identifier may correctly appear as ``cfg.field`` or
        # ``object->method()``.  Only identifier characters extend the token;
        # punctuation is a valid qualification boundary.
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])"
    return re.search(pattern, content) is not None


def check_mappings(
    repo_root: Path,
    surfaces: Mapping[str, set[str]],
    mapping: Mapping[str, Any],
) -> CheckReport:
    report = CheckReport(surface_count=sum(len(tokens) for tokens in surfaces.values()))
    configured = mapping.get("surfaces", {})

    for category in sorted(set(surfaces) - set(configured)):
        report.findings.append(
            Finding(category, "", "source surface has no documentation mapping")
        )
    for category in sorted(set(configured) - set(surfaces)):
        report.findings.append(
            Finding(category, "", "mapping has no corresponding source extractor")
        )

    for category in sorted(set(surfaces) & set(configured)):
        config = configured[category]
        if not isinstance(config, dict):
            report.findings.append(Finding(category, "", "mapping must be an object"))
            continue
        documents = config.get("documents")
        allowlist = config.get("allowlist", {})
        if not (
            isinstance(documents, list)
            and documents
            and all(isinstance(document, str) and document for document in documents)
        ):
            report.findings.append(
                Finding(category, "", "documents must be a non-empty string list")
            )
            continue
        if not isinstance(allowlist, dict):
            report.findings.append(Finding(category, "", "allowlist must be an object"))
            continue

        contents: list[tuple[str, str]] = []
        for document in documents:
            path = repo_root / document
            try:
                contents.append((document, path.read_text(encoding="utf-8")))
            except OSError as error:
                report.findings.append(
                    Finding(category, "", f"cannot read mapped document {document}: {error}")
                )

        tokens = surfaces[category]
        for token in sorted(set(allowlist) - tokens):
            report.findings.append(
                Finding(category, token, "stale allowlist entry is not present in source")
            )

        for token in sorted(tokens):
            report.mapping_count += 1
            if token in allowlist:
                entry = allowlist[token]
                reason = entry.get("reason") if isinstance(entry, dict) else None
                if not isinstance(reason, str) or len(reason.strip()) < 12:
                    report.findings.append(
                        Finding(category, token, "allowlist entry needs a specific reason")
                    )
                    continue
                canonical = entry.get("canonical")
                if canonical is not None:
                    if not isinstance(canonical, str) or canonical not in tokens:
                        report.findings.append(
                            Finding(
                                category,
                                token,
                                "allowlist canonical token is not present in the same source surface",
                            )
                        )
                        continue
                    if not any(_token_present(content, canonical) for _, content in contents):
                        report.findings.append(
                            Finding(
                                category,
                                token,
                                f"canonical token {canonical!r} is not documented",
                            )
                        )
                        continue
                report.allowlisted_count += 1
                continue
            if not any(_token_present(content, token) for _, content in contents):
                joined = ", ".join(documents)
                report.findings.append(
                    Finding(category, token, f"not documented in mapped page(s): {joined}")
                )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check public source surfaces against explicit documentation mappings."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="repository root (defaults to the checker parent)",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        help="mapping JSON path (defaults to tools/doc_public_surfaces.json)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    mapping_path = args.mapping or repo_root / DEFAULT_MAPPING_PATH
    if not mapping_path.is_absolute():
        mapping_path = repo_root / mapping_path
    try:
        surfaces = collect_public_surfaces(repo_root)
        mapping = load_mapping(mapping_path)
    except (OSError, SyntaxError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    report = check_mappings(repo_root, surfaces, mapping)
    print(f"Public surface categories checked: {len(surfaces)}")
    print(f"Public source tokens checked: {report.surface_count}")
    print(f"Documentation mappings checked: {report.mapping_count}")
    print(f"Reasoned allowlist entries used: {report.allowlisted_count}")
    for finding in report.findings:
        print(finding)
    if report.ok:
        print("Public surface documentation coverage: PASS")
        return 0
    print(f"Public surface documentation coverage: FAIL ({len(report.findings)} findings)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
