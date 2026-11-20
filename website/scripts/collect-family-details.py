#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read family configuration and check manifest inputs without downloading/building models.

This is a documentation inventory, not a replacement model policy. Source expressions
remain expressions; a CPU parser/request check is never a GPU qualification result.
Only the standard library and the repository's CPU command/catalog modules are used.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import dataclasses
import importlib
import importlib.util
import inspect
import io
import json
from pathlib import Path
import sys
from typing import Any


def source(repo_root: Path, path: Path, node: ast.AST | None = None) -> dict:
    return {"path": path.relative_to(repo_root).as_posix(), "line": getattr(node, "lineno", 1)}


def spelling(node: ast.AST | None) -> str | None:
    return ast.unparse(node) if node is not None else None


def comments(lines: list[str], node: ast.AST) -> str:
    previous = node.lineno - 2
    result = []
    while previous >= 0 and lines[previous].lstrip().startswith("#"):
        comment = lines[previous].lstrip()[1:].strip()
        if "SPDX-" in comment:
            break
        result.insert(0, comment)
        previous -= 1
    return " ".join(result)


def class_details(repo_root: Path, path: Path, node: ast.ClassDef, lines: list[str]) -> dict:
    name = node.name
    kind = "build" if "Build" in name or path.stem in {"build_request", "cli"} else (
        "runtime" if "runtime" in path.parts else "checkpoint"
    )
    fields = []
    for field in node.body:
        if isinstance(field, ast.AnnAssign) and isinstance(field.target, ast.Name):
            annotation = spelling(field.annotation)
            fields.append({
                "name": field.target.id, "type": annotation,
                "required": field.value is None, "default": spelling(field.value),
                "description": comments(lines, field),
                "readOnly": "ClassVar[" in annotation,
                "source": source(repo_root, path, field),
            })
        elif isinstance(field, ast.FunctionDef) and any(
            spelling(decorator) in {"property", "cached_property"} for decorator in field.decorator_list
        ):
            fields.append({
                "name": field.name, "type": spelling(field.returns) or "computed",
                "required": False, "default": None, "readOnly": True,
                "description": ast.get_docstring(field) or "Computed by the family implementation.",
                "source": source(repo_root, path, field),
            })
    return {
        "name": name, "kind": kind, "description": ast.get_docstring(node) or "",
        "source": source(repo_root, path, node), "fields": fields,
        "bases": [spelling(base) for base in node.bases],
        "declaration": ast.get_source_segment("\n".join(lines), node) or ast.unparse(node),
    }


def dictionary_details(repo_root: Path, path: Path, tree: ast.Module) -> list[dict]:
    """Preserve dict-shaped contracts and metadata lookups, without guessing types."""
    result = []
    is_config_file = "config" in path.stem
    source_text = path.read_text(encoding="utf-8") if path.is_file() else ""
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not is_config_file and "config" not in function.name.lower():
            continue
        fields: list[dict] = []
        for node in ast.walk(function):
            candidates = []
            if isinstance(node, ast.Dict):
                candidates = [(key.value, value, "derived") for key, value in zip(node.keys, node.values)
                              if isinstance(key, ast.Constant) and isinstance(key.value, str)]
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and node.func.attr == "get" and node.args
                  and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
                candidates = [(node.args[0].value, node.args[1] if len(node.args) > 1 else None,
                               "checkpoint metadata")]
            for name, value, kind in candidates:
                item = {
                    "name": name, "type": kind, "required": False,
                    "default": spelling(value), "readOnly": True,
                    "description": ("Expression in a family configuration mapping." if kind == "derived"
                                    else "Checkpoint metadata lookup; this is not a CLI option."),
                    "source": source(repo_root, path, value or node),
                }
                # Repeated keys in nested mappings or conditional branches may
                # have different defaults. Retain every source occurrence.
                fields.append(item)
        if fields:
            result.append({
                "name": f"{function.name}() metadata", "kind": "checkpoint",
                "description": "Source mapping and lookup expressions; defaults may depend on the checkpoint.",
                "source": source(repo_root, path, function), "fields": fields,
                "declaration": ast.get_source_segment(source_text, function) if source_text else ast.unparse(function),
            })
    return result


def rejection_guards(repo_root: Path, path: Path, tree: ast.Module) -> list[dict]:
    result = []

    def visit(node: ast.AST, conditions: tuple[str, ...] = (), scope: str = "") -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scope = f"{scope}.{node.name}".strip(".")
        if isinstance(node, ast.If):
            test = spelling(node.test)
            for child in node.body:
                visit(child, (*conditions, test), scope)
            for child in node.orelse:
                visit(child, (*conditions, f"not ({test})"), scope)
            return
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
            exception = spelling(node.exc.func)
            if exception in {"ValueError", "TypeError", "NotImplementedError", "RuntimeError", "AssertionError"}:
                result.append({
                    "condition": " and ".join(f"({condition})" for condition in conditions) or "unconditional in this scope",
                    "message": spelling(node.exc.args[0]) if node.exc.args else exception,
                    "exception": exception, "scope": scope, "source": source(repo_root, path, node),
                })
        for child in ast.iter_child_nodes(node):
            visit(child, conditions, scope)

    visit(tree)
    return result


def python_details(repo_root: Path, family_root: Path) -> tuple[list, list, list, dict, list]:
    classes, constraints, sources, parsed, issues = [], [], [], {}, []
    for path in sorted(family_root.rglob("*.py")):
        if any(part in {"tests", "__pycache__", "fixtures", "reference"} for part in path.relative_to(family_root).parts):
            continue
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as error:
            issues.append(f"{path.relative_to(repo_root)}: {error.msg} (line {error.lineno})")
            continue
        parsed[path] = tree
        before = len(classes)
        config_file = "config" in path.stem or path.stem in {"cli", "build_request"}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and (
                "config" in node.name.lower() or "BuildRequest" in node.name or config_file
            ):
                details = class_details(repo_root, path, node, text.splitlines())
                if details["fields"]:
                    classes.append(details)
        classes.extend(dictionary_details(repo_root, path, tree))
        if len(classes) > before or path.stem in {"model", "support", "cli", "build_request"}:
            sources.append(source(repo_root, path))
        if config_file or path.stem in {"model", "support"}:
            constraints.extend(rejection_guards(repo_root, path, tree))
    return classes, constraints, sources, parsed, issues


def support_details(repo_root: Path, family_root: Path, parsed: dict) -> dict:
    path = family_root / "support.py"
    tree = parsed.get(path)
    declarations = []
    if tree is not None:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            values = {}
            for keyword in node.keywords:
                if keyword.arg in {"model_types", "tasks", "default_task", "default_precision", "checkpoint_names",
                                   "architectures", "pipeline_classes", "required_files"}:
                    try:
                        values[keyword.arg] = ast.literal_eval(keyword.value)
                    except (ValueError, TypeError):
                        values[keyword.arg] = {"expression": spelling(keyword.value)}
            if values:
                declarations.append({"source": source(repo_root, path, node), "values": values})
    return {"source": source(repo_root, path) if path.exists() else None,
            "declarations": declarations,
            "note": "Support predicates can depend on checkpoint metadata; see the owning source."}


def inspect_handler(repo_root: Path, family_root: Path, command: dict) -> dict:
    if command["executor"] == "native":
        return {"handlerStatus": "native", "handlerSource": None, "issues": []}
    module_name, function_name = command["handler"].split(":")
    path = family_root / (module_name.replace(".", "/") + ".py")
    if not path.is_file():
        return {"handlerStatus": "issues", "handlerSource": None, "issues": [f"Missing Python handler source: {path.relative_to(repo_root)}"]}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next((node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and node.name == function_name), None)
    if function is None:
        return {"handlerStatus": "issues", "handlerSource": source(repo_root, path),
                "issues": [f"Handler {function_name} is not defined in {path.relative_to(repo_root)}"]}
    parameters = {arg.arg for arg in [*function.args.args, *function.args.kwonlyargs]}
    declared = {arg["name"] for arg in command["arguments"]}
    required = {arg.arg for arg in function.args.args[:len(function.args.args) - len(function.args.defaults)]}
    required.update(arg.arg for arg, default in zip(function.args.kwonlyargs, function.args.kw_defaults) if default is None)
    issues = []
    if function.args.posonlyargs:
        issues.append("Handler has positional-only parameters, but CLI dispatch passes keyword arguments")
    if function.args.kwarg is None and declared - parameters:
        issues.append(f"CLI arguments absent from handler: {', '.join(sorted(declared - parameters))}")
    if required - declared:
        issues.append(f"Required handler arguments absent from CLI: {', '.join(sorted(required - declared))}")
    optional_required = {
        argument["name"] for argument in command["arguments"]
        if argument["name"] in required and "default" not in argument
        and not argument.get("required", not argument.get("flags"))
    }
    if optional_required:
        issues.append(f"Required handler arguments are optional in CLI: {', '.join(sorted(optional_required))}")
    return {"handlerStatus": "issues" if issues else "matched", "handlerSource": source(repo_root, path, function), "issues": issues}


def cli_details(repo_root: Path, family_root: Path, family_cli: Any) -> dict:
    path = family_root / "cli.json"
    if not path.exists():
        return {"status": "missing", "source": None, "commands": [], "issues": [],
                "fallbackSource": {"path": "core/builder/tensorrt_model_connect/build_cli.py", "line": 1},
                "note": "No family CLI declaration. The shared build command remains available; family restrictions still apply."}
    try:
        declaration = family_cli._validate(json.loads(path.read_text(encoding="utf-8")))
    except (ValueError, OSError) as error:
        return {"status": "invalid", "source": source(repo_root, path), "commands": [], "issues": [str(error)]}
    commands, issues = [], []
    for command in declaration["commands"]:
        parser = argparse.ArgumentParser(allow_abbrev=False)
        family_cli._add_arguments(parser, command)
        checked = inspect_handler(repo_root, family_root, command)
        issues.extend(checked["issues"])
        commands.append({**command, "source": source(repo_root, path), **checked})
    return {"status": "invalid" if issues else "validated", "source": source(repo_root, path),
            "commands": commands, "issues": issues}


def shared_cli_details(build_cli: Any) -> dict:
    parser = build_cli._parser()
    subparsers = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
    child = subparsers.choices["build"]
    arguments = []
    for action in child._actions:
        if action.dest == "help":
            continue
        kind = "path" if action.type is Path else "int" if action.type is int else "string"
        item = {"name": action.dest, "type": kind, "required": action.required, "help": action.help or ""}
        if action.option_strings:
            item["flags"] = action.option_strings
        if action.default is not None and action.default != argparse.SUPPRESS:
            item["default"] = action.default
        if action.choices is not None:
            item["choices"] = list(action.choices)
        if isinstance(action, argparse._StoreTrueAction):
            item.update(type="bool", action="store_true")
        elif isinstance(action, argparse._AppendAction):
            item["action"] = "append"
        arguments.append(item)
    return {"name": "build", "arguments": arguments,
            "source": {"path": "core/builder/tensorrt_model_connect/build_cli.py", "line": 25},
            "help": "Shared legacy build parser. Accepted options remain conditional on the selected family and checkpoint."}


def request_class(repo_root: Path, family_root: Path, classes: list, shared_request: Any) -> tuple[dict, Any, Any]:
    for relative in ("build_request.py", "cli.py"):
        path = family_root / relative
        match = next((item for item in classes if item["name"] == "BuildRequest" and item["source"]["path"] == path.relative_to(repo_root).as_posix()), None)
        if match is not None:
            try:
                module = importlib.import_module(f"families.{family_root.name}.{path.stem}")
                return {"owner": "family", "status": "checked", **match}, module.BuildRequest, getattr(module, "coerce_request", None)
            except (ImportError, AttributeError) as error:
                return {"owner": "family", "status": "not-checked", **match, "issues": [str(error)]}, None, None
    path = repo_root / "core/builder/tensorrt_model_connect/build.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BuildRequest")
    return {"owner": "shared", "status": "checked", **class_details(repo_root, path, node, path.read_text().splitlines())}, shared_request, None


def direct_build_guards(tree: ast.Module | None, request: Any, *, normalized: bool = False) -> tuple[list[str], int, int]:
    """Check request-only rejection branches with conservative control flow.

    Evaluation has no calls, subscripts or attribute chains. Unknown dispatch or
    request rebinding stops the walk; the complete guards remain linked as source.
    """
    if tree is None:
        return [], 0, 0
    build = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build"), None)
    if build is None:
        return [], 0, 0
    issues, checked, deferred = [], 0, 0
    allowed = (ast.Expression, ast.BoolOp, ast.UnaryOp, ast.Compare, ast.Constant, ast.Set, ast.Tuple,
               ast.List, ast.Attribute, ast.Name, ast.Load, ast.And, ast.Or, ast.Not, ast.USub, ast.UAdd,
               ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn)

    def evaluate(test):
        nodes = list(ast.walk(test))
        if not all(isinstance(node, allowed) for node in nodes):
            return None
        if not all(node.id == "request" for node in nodes if isinstance(node, ast.Name)):
            return None
        if not all(isinstance(node.value, ast.Name) and node.value.id == "request"
                   and not node.attr.startswith("_") for node in nodes if isinstance(node, ast.Attribute)):
            return None
        try:
            return bool(eval(compile(ast.Expression(test), "<family request guard>", "eval"),
                             {"__builtins__": {}}, {"request": request}))
        except (AttributeError, TypeError, ValueError):
            return None

    def rebinds(node):
        return any(isinstance(child, ast.Name) and child.id == "request" and isinstance(child.ctx, ast.Store)
                   for child in ast.walk(node))

    def walk(statements):
        nonlocal checked, deferred
        for statement in statements:
            if isinstance(statement, ast.Return):
                return True
            if isinstance(statement, ast.Raise):
                message = (spelling(statement.exc.args[0]) if isinstance(statement.exc, ast.Call)
                           and statement.exc.args else spelling(statement.exc) or "re-raised exception")
                issues.append(f"Family build guard rejects this request: {message}")
                return True
            if isinstance(statement, ast.If):
                selected = evaluate(statement.test)
                if selected is None:
                    if any(isinstance(child, (ast.Return, ast.Raise)) for child in ast.walk(statement)) or rebinds(statement):
                        deferred += 1
                    if rebinds(statement) or any(isinstance(child, ast.Return) for child in ast.walk(statement)):
                        return True
                    continue
                checked += 1
                if walk(statement.body if selected else statement.orelse):
                    return True
            elif rebinds(statement):
                # The profile check already applied this exact family coercion.
                value = getattr(statement, "value", None)
                already_coerced = (normalized and isinstance(value, ast.Call)
                                   and spelling(value.func) == "coerce_request"
                                   and len(value.args) == 1 and spelling(value.args[0]) == "request")
                if not already_coerced:
                    deferred += 1
                    return True
            elif any(isinstance(child, ast.Return) for child in ast.walk(statement)):
                deferred += 1
                return True
        return False

    walk(build.body)
    return issues, checked, deferred


def profile_check(path: Path, family_root: Path, modules: dict, request_type: Any, coerce: Any, parsed: dict) -> dict:
    checks, issues = [], []
    raw = json.loads(path.read_text(encoding="utf-8"))
    family_cli, build_cli = modules["family_cli"], modules["build_cli"]
    output_path = Path("__documentation_output__.bundle")
    model_path = Path("__documentation_checkpoint__")
    values = {}
    model = None
    stage = "manifest build mapping"
    try:
        model = modules["catalog"].ManifestCatalog(family_root.parent)._load(path)
        unknown = set(raw) - modules["catalog"]._MANIFEST_METADATA_FIELDS - set(model.build_settings)
        if unknown:
            raise ValueError(f"Manifest fields ignored by the benchmark build mapping: {', '.join(sorted(unknown))}")
        command = modules["builder"]._build_command(model, model_path, output_path, ())
        checks.append({"name": "manifest build mapping", "status": "checked", "detail": "Accepted by the current benchmark catalog and build command serializer."})
        stage = "CLI parsing"
        argv = list(command[3:])
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            if argv[0] == family_root.name:
                declaration = family_cli.load_family_cli(family_root.name)
                spec = next(command for command in declaration["commands"] if command["name"] == argv[1])
                parser = argparse.ArgumentParser(allow_abbrev=False)
                family_cli._add_arguments(parser, spec)
                values = vars(parser.parse_args(family_cli._ordered_arguments(parser, spec, argv[2:])))
                values = {key: value for key, value in values.items() if value is not family_cli._MISSING}
                family_cli.serialize_arguments(spec, values)
                module_name, handler_name = spec["handler"].split(":")
                handler = getattr(importlib.import_module(f"families.{family_root.name}.{module_name}"), handler_name)
                inspect.signature(handler).bind(**values)
            else:
                values = vars(build_cli._parser().parse_args(argv))
        checks.append({"name": "CLI parsing", "status": "checked", "detail": "Parsed the generated build arguments without invoking a handler or resolving a checkpoint."})
        missing = []
        for name, expected in model.build_settings.items():
            parser_name = "fp32_layer" if name == "fp32_layers" and argv[0] != family_root.name else name
            if parser_name not in values or values[parser_name] != expected:
                missing.append(name)
        if missing:
            raise ValueError(f"Build settings not preserved by the generated CLI: {', '.join(sorted(missing))}")
        # Catalog metadata includes real build controls such as parallel sizes.
        # They can disappear before build_settings exists, so compare them with
        # the owning request's fixed defaults as well as the serialized settings.
        shared_type = modules.get("shared_request", request_type)
        shared_fields = {field.name: field for field in dataclasses.fields(shared_type)} if shared_type else {}
        owner_fields = {field.name: field for field in dataclasses.fields(request_type)} if request_type else {}
        for name in sorted(set(raw) & modules["catalog"]._MANIFEST_METADATA_FIELDS & shared_fields.keys()):
            if name == "family" or name in values:
                continue
            field = owner_fields.get(name, shared_fields[name])
            fixed = getattr(request_type, name, field.default)
            if fixed is dataclasses.MISSING or raw[name] != fixed:
                raise ValueError(f"Manifest {name}={raw[name]!r} is not represented by the generated CLI; the owning fixed/default value is {fixed!r}")
    except SystemExit as error:
        detail = f"Standard CLI rejected manifest arguments (exit {error.code}): {stderr.getvalue().strip()}"
        issues.append(detail)
        checks.append({"name": stage, "status": "issues", "detail": detail})
    except (ValueError, TypeError, KeyError, AttributeError, ImportError, OSError, RuntimeError, NotImplementedError) as error:
        detail = f"{type(error).__name__}: {error}"
        issues.append(detail)
        checks.append({"name": stage, "status": "issues", "detail": detail})
    # The Python API is checked separately: a custom precision can be accepted by
    # a family-owned E2E request even when the standard CLI cannot express it.
    try:
        if request_type is None:
            checks.append({"name": "BuildRequest", "status": "not-checked", "detail": "The owning request module could not be imported with CPU dependencies."})
        else:
            inputs = {**raw, **raw.get("build", {}), **values, "model_dir": model_path,
                      "output_path": output_path, "family": family_root.name}
            if "fp32_layer" in values:
                inputs["fp32_layers"] = tuple(values["fp32_layer"])
            elif "fp32_layers" in inputs:
                inputs["fp32_layers"] = tuple(inputs["fp32_layers"])
            fields = {field.name for field in dataclasses.fields(request_type) if field.init}
            request = request_type(**{name: value for name, value in inputs.items() if name in fields})
            if coerce is not None:
                # Pass the complete legacy surface to the owner's coercion.
                # Coercing an already-filtered owner request would silently drop
                # unsupported controls such as BERT context_parallel_size.
                shared_type = modules["shared_request"]
                shared_fields = {field.name for field in dataclasses.fields(shared_type) if field.init}
                legacy = shared_type(**{name: value for name, value in inputs.items() if name in shared_fields})
                request = coerce(legacy)
            checks.append({"name": "BuildRequest", "status": "checked", "detail": "Constructed a typed Python request from the manifest and parsed defaults; no build was started."})
            rejected, checked, deferred = direct_build_guards(parsed.get(family_root / "model.py"), request, normalized=coerce is not None)
            issues.extend(rejected)
            checks.append({"name": "family request guards", "status": "issues" if rejected else "checked",
                           "detail": f"Checked {checked} direct request-only rejection guards; {deferred} immediate guards need additional checkpoint or execution context."})
    except (ValueError, TypeError, KeyError, AttributeError, ImportError, OSError, RuntimeError, NotImplementedError) as error:
        detail = f"{type(error).__name__}: {error}"
        issues.append(detail)
        checks.append({"name": "BuildRequest", "status": "issues", "detail": detail})
    case_checks = []
    for case in raw.get("testcases", []):
        name = case.get("name", "unnamed testcase")
        try:
            if model is None:
                raise ValueError("The manifest could not be loaded by the current benchmark catalog")
            resolver = getattr(modules["catalog"], "resolve_case", None)
            if resolver is None:
                raise ValueError("No benchmark testcase adapter is available for this audit")
            resolved = resolver(model, output_path, case_name=name)
            case_checks.append({"name": name, "status": "mapped", "operation": resolved.operation,
                                "detail": "The current benchmark adapter maps this testcase request. No runtime was executed."})
        except (ValueError, TypeError, KeyError, AttributeError, ImportError, OSError, RuntimeError, NotImplementedError) as error:
            case_checks.append({"name": name, "status": "not-checked",
                                "detail": f"{error}. Family E2E may prepare fixture inputs separately; runtime support was not tested."})
    return {"status": "issues" if issues else "not-checked" if any(check["status"] == "not-checked" for check in checks) else "checked",
            "checks": checks, "issues": issues,
            "caseChecks": case_checks,
            "boundary": "CPU source, manifest, parser and request checks only. Checkpoint loading, full family validation, TensorRT build, runtime and numerical parity were not run."}


def collect(repo_root: Path) -> dict:
    repo_root = repo_root.resolve()
    sys.path[:0] = [str(repo_root / "core/builder"), str(repo_root / "apps/benchmark"), str(repo_root)]
    from tensorrt_model_connect import family_cli, build_cli
    from tensorrt_model_connect.build import BuildRequest
    from trtmc_benchmark import catalog, builder

    modules = {"family_cli": family_cli, "build_cli": build_cli, "catalog": catalog, "builder": builder,
               "shared_request": BuildRequest}
    families = {}
    native_path = Path(__file__).with_name("collect-native-details.py")
    if not native_path.is_file():
        raise FileNotFoundError(f"Required native configuration extractor is missing: {native_path}")
    spec = importlib.util.spec_from_file_location("model_support_native_details", native_path)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    for family_root in sorted((repo_root / "families").iterdir()):
        if not family_root.is_dir() or family_root.name.startswith(("_", ".")):
            continue
        classes, constraints, sources, parsed, issues = python_details(repo_root, family_root)
        build_request, request_type, coerce = request_class(repo_root, family_root, classes, BuildRequest)
        classes.extend(native.collect_native_details(repo_root, family_root))
        profiles = {path.relative_to(repo_root).as_posix(): profile_check(path, family_root, modules, request_type, coerce, parsed)
                    for path in sorted((family_root / "tests/manifests").glob("*.json"))}
        families[family_root.name] = {
            "configClasses": classes, "configStatus": "present" if classes else "missing",
            "cli": cli_details(repo_root, family_root, family_cli), "buildRequest": build_request,
            "support": support_details(repo_root, family_root, parsed), "constraints": constraints,
            "sources": sources, "profileChecks": profiles, "issues": issues,
        }
    profiles = [check for family in families.values() for check in family["profileChecks"].values()]
    cases = [case for profile in profiles for case in profile["caseChecks"]]
    return {
        "schemaVersion": 1, "families": families, "sharedBuildCommand": shared_cli_details(build_cli),
        "summary": {
            "familyCount": len(families), "familiesWithCli": sum(value["cli"]["status"] != "missing" for value in families.values()),
            "familiesWithConfig": sum(bool(value["configClasses"]) for value in families.values()),
            "configClassCount": sum(len(value["configClasses"]) for value in families.values()),
            "profileCount": len(profiles), "checkedProfiles": sum(value["status"] == "checked" for value in profiles),
            "issueProfiles": sum(value["status"] == "issues" for value in profiles),
            "uncheckedProfiles": sum(value["status"] == "not-checked" for value in profiles),
            "totalCases": len(cases), "mappedCases": sum(case["status"] == "mapped" for case in cases),
            "unmappedCases": sum(case["status"] == "not-checked" for case in cases),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    json.dump(collect(args.repo_root), sys.stdout, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
