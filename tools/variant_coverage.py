#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report model-family variant coverage and TensorRT version consistency.

A family declares the precision and quantization combinations it builds with
``family_support(variants=...)`` in its ``support.py``. This report compares the
declaration with the family source, its E2E manifests and its hardware tests:

- G1: every quantization the family source accepts is declared.
- G2: every supported variant has a manifest testcase.
- G3: every supported variant has a hardware test marked ``@covers_variant``.
- G4: every manifest matches a declared variant.
- G5: every TensorRT version pin in the Dockerfiles and the CI image tag matches
  ``pyproject.toml``.

A family without a declaration is reported against its manifests only: a
quantization the source accepts without a manifest testcase is a G1 gap.

The report reads source files and never imports family code. It exits 0 unless
``--strict`` is given; the quality stage does not run it yet.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence, TypeVar

from . import test_impact


UNQUANTIZED = "none"
_HARDWARE_MARK = re.compile(r"pytest\.mark\.(?:trt|gpu)\b")
_Test = TypeVar("_Test", bound=Callable[..., object])


def covers_variant(*, precision: str, quantization: str | None = None) -> Callable[[_Test], _Test]:
    """Mark a hardware test as coverage for one declared family variant."""

    def mark(test: _Test) -> _Test:
        test.trtmc_variant = (precision, quantization or UNQUANTIZED)  # type: ignore[attr-defined]
        return test

    return mark


@dataclass(frozen=True)
class DeclaredVariant:
    precision: str
    quantization: str
    status: str = "supported"
    manifests: tuple[str, ...] = ()

    @property
    def key(self) -> tuple[str, str]:
        return (self.precision, self.quantization)

    @property
    def label(self) -> str:
        return f"{self.precision}+{self.quantization}"


@dataclass
class FamilyReport:
    family: str
    declared: bool
    undeclared_quantizations: list[str] = field(default_factory=list)
    variants_without_manifest: list[str] = field(default_factory=list)
    variants_without_test: list[str] = field(default_factory=list)
    manifests_without_variant: list[str] = field(default_factory=list)
    unannotated_hardware_tests: list[str] = field(default_factory=list)

    @property
    def gaps(self) -> int:
        return (
            len(self.undeclared_quantizations)
            + len(self.variants_without_manifest)
            + len(self.variants_without_test)
            + len(self.manifests_without_variant)
        )


def _parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError):
        return None


def _strings(node: ast.AST) -> set[str]:
    """String literals of a constant or a literal set, tuple or list; nothing else."""
    items = node.elts if isinstance(node, (ast.Set, ast.Tuple, ast.List)) else [node]
    return {
        item.value
        for item in items
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    }


def _is_requested_quantization(node: ast.AST) -> bool:
    text = ast.unparse(node)
    return text == "quantization" or (
        text.endswith(".quantization") and not text.startswith("manifest")
    )


def accepted_quantizations(family_root: Path) -> set[str]:
    """Quantizations compared against the build request in the family source.

    A string compared with ``request.quantization`` (or a local of that name) is
    taken as accepted. Rejection branches written the same way are reported too;
    the owner then declares the variant as experimental or rewrites the check.
    """
    values: set[str] = set()
    for path in sorted(family_root.rglob("*.py")):
        if "tests" in path.relative_to(family_root).parts:
            continue
        tree = _parse(path)
        for node in ast.walk(tree) if tree else ():
            if isinstance(node, ast.Compare) and _is_requested_quantization(node.left):
                for comparator in node.comparators:
                    values |= _strings(comparator)
    return {value for value in values if value and value != UNQUANTIZED}


def _variant_call(node: ast.AST) -> DeclaredVariant:
    if not isinstance(node, ast.Call) or ast.unparse(node.func) != "Variant":
        raise ValueError(f"variants must contain Variant(...) calls, got {ast.unparse(node)}")
    values = dict(zip(("precision", "quantization", "status", "manifests"), node.args))
    values.update({keyword.arg: keyword.value for keyword in node.keywords if keyword.arg})
    literal = {name: ast.literal_eval(value) for name, value in values.items()}
    return DeclaredVariant(
        precision=literal["precision"],
        quantization=literal.get("quantization") or UNQUANTIZED,
        status=literal.get("status", "supported"),
        manifests=tuple(literal.get("manifests", ())),
    )


def declared_variants(family_root: Path) -> list[DeclaredVariant] | None:
    """Variants declared in ``support.py``; None when the family declares none."""
    tree = _parse(family_root / "support.py")
    for node in ast.walk(tree) if tree else ():
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "family_support":
            for keyword in node.keywords:
                if keyword.arg == "variants":
                    if not isinstance(keyword.value, (ast.Tuple, ast.List)):
                        raise ValueError(f"{family_root.name}: variants must be a literal tuple")
                    return [_variant_call(item) for item in keyword.value.elts]
    return None


def manifest_variants(family_root: Path) -> dict[str, tuple[str, str]]:
    """Manifest name to (precision, quantization) for one family."""
    result: dict[str, tuple[str, str]] = {}
    for path in sorted((family_root / "tests/manifests").glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        name = str(manifest.get("name") or path.stem)
        quantization = manifest.get("quantization") or UNQUANTIZED
        result[name] = (str(manifest.get("precision", "")), str(quantization))
    return result


def _has_hardware_mark(nodes: Sequence[ast.expr]) -> bool:
    return any(_HARDWARE_MARK.search(ast.unparse(node)) for node in nodes)


def _module_hardware_mark(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "pytestmark" for target in node.targets)
        and _HARDWARE_MARK.search(ast.unparse(node.value))
        for node in tree.body
    )


def _variant_key(decorators: Sequence[ast.expr]) -> set[tuple[str, str]]:
    keys = set()
    for node in decorators:
        if isinstance(node, ast.Call) and ast.unparse(node.func).endswith("covers_variant"):
            values = {k.arg: ast.literal_eval(k.value) for k in node.keywords if k.arg}
            keys.add((str(values.get("precision")), values.get("quantization") or UNQUANTIZED))
    return keys


def _covers_variant_keys(tree: ast.Module) -> set[tuple[str, str]]:
    """Variant keys of pytest-collected test functions that carry a gpu or trt mark.

    The mark may sit on the function, its ``Test*`` class, or the module's
    ``pytestmark``. An annotated helper or CPU-only test covers nothing.
    """
    keys: set[tuple[str, str]] = set()
    module_marked = _module_hardware_mark(tree)

    def visit(functions: Sequence[ast.stmt], marked: bool) -> None:
        for node in functions:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
                "test"
            ):
                if marked or _has_hardware_mark(node.decorator_list):
                    keys.update(_variant_key(node.decorator_list))

    visit(tree.body, module_marked)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            visit(node.body, module_marked or _has_hardware_mark(node.decorator_list))
    return keys


def hardware_tests(family_root: Path) -> tuple[set[tuple[str, str]], list[str]]:
    """Annotated variant keys and hardware test files without an annotation."""
    annotated: set[tuple[str, str]] = set()
    unannotated: list[str] = []
    for path in sorted((family_root / "tests").glob("*.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        tree = _parse(path)
        keys = _covers_variant_keys(tree) if tree else set()
        annotated |= keys
        if _HARDWARE_MARK.search(text) and not keys:
            unannotated.append(path.name)
    return annotated, unannotated


def _matches(variant: DeclaredVariant, manifests: dict[str, tuple[str, str]]) -> bool:
    """Named manifests narrow the coverage to a profile; they must still build the variant."""
    if variant.manifests:
        return all(manifests.get(name) == variant.key for name in variant.manifests)
    return variant.key in manifests.values()


def family_report(family_root: Path) -> FamilyReport:
    accepted = accepted_quantizations(family_root)
    manifests = manifest_variants(family_root)
    declared = declared_variants(family_root)
    annotated, unannotated = hardware_tests(family_root)
    report = FamilyReport(family_root.name, declared is not None)
    report.unannotated_hardware_tests = unannotated
    if declared is None:
        covered = {quantization for _, quantization in manifests.values()}
        report.undeclared_quantizations = sorted(accepted - covered)
        return report
    report.undeclared_quantizations = sorted(accepted - {v.quantization for v in declared})
    supported = [variant for variant in declared if variant.status == "supported"]
    report.variants_without_manifest = [
        variant.label for variant in supported if not _matches(variant, manifests)
    ]
    report.variants_without_test = [
        variant.label for variant in supported if variant.key not in annotated
    ]
    keys = {variant.key for variant in declared}
    report.manifests_without_variant = sorted(
        name for name, key in manifests.items() if key not in keys
    )
    return report


_VERSION_PATTERNS = (
    re.compile(r"\btensorrt==(\d+(?:\.\d+)+)"),
    re.compile(r"\blibnv(?:infer|onnxparsers)[\w-]*=(\d+(?:\.\d+)+)-"),
    re.compile(r"tensorrt\.__version__ == '(\d+(?:\.\d+)+)'"),
    re.compile(r"^ARG TENSORRT_VERSION=(\d+(?:\.\d+)+)", re.MULTILINE),
)
_IMAGE_TAG = re.compile(r"trtmc-ci:trt(\d+\.\d+)\b")


def _pinned_tensorrt(repo: Path) -> str:
    match = _VERSION_PATTERNS[0].search((repo / "pyproject.toml").read_text(encoding="utf-8"))
    if match is None:
        raise ValueError("pyproject.toml does not pin tensorrt==<version>")
    return match.group(1)


def toolchain_mismatches(repo: Path) -> list[str]:
    """TensorRT versions in Dockerfiles and the CI image tag that differ from the pin."""
    pinned = _pinned_tensorrt(repo)
    mismatches: list[str] = []
    for path in sorted(repo.glob("Dockerfile*")):
        text = path.read_text(encoding="utf-8")
        for pattern in _VERSION_PATTERNS:
            for match in pattern.finditer(text):
                if match.group(1) != pinned:
                    line = text.count("\n", 0, match.start()) + 1
                    mismatches.append(f"{path.name}:{line}: {match.group(1)} != {pinned}")
    image = repo / "tools/ci/docker_image.py"
    for match in _IMAGE_TAG.finditer(image.read_text(encoding="utf-8")):
        if not pinned.startswith(match.group(1) + "."):
            mismatches.append(f"tools/ci/docker_image.py: trt{match.group(1)} != {pinned}")
    return mismatches


@dataclass
class Report:
    families: list[FamilyReport]
    toolchain: list[str]

    @property
    def gaps(self) -> int:
        return sum(item.gaps for item in self.families) + len(self.toolchain)


def family_root(repo: Path, family: str) -> Path:
    """The directory of one existing family; a mistyped name is an error, not a clean report."""
    if family not in test_impact.inventory(repo):
        raise ValueError(f"unknown family: {family}")
    return repo / "families" / family


def build_report(repo: Path, families: Sequence[str] = ()) -> Report:
    selected = families or test_impact.inventory(repo)
    reports = [family_report(family_root(repo, family)) for family in selected]
    return Report(reports, toolchain_mismatches(repo))


_COLUMNS = (
    ("G1 undeclared quantization", "undeclared_quantizations"),
    ("G2 variant without manifest", "variants_without_manifest"),
    ("G3 variant without hardware test", "variants_without_test"),
    ("G4 manifest without variant", "manifests_without_variant"),
)


def render(report: Report) -> str:
    declared = sum(item.declared for item in report.families)
    lines = [f"families: {len(report.families)}, declaring variants: {declared}"]
    for title, attribute in _COLUMNS:
        rows = [item for item in report.families if getattr(item, attribute)]
        lines.append(f"\n{title}: {len(rows)} families")
        lines.extend(f"  {item.family}: {', '.join(getattr(item, attribute))}" for item in rows)
    lines.append(f"\nG5 TensorRT version mismatches: {len(report.toolchain)}")
    lines.extend(f"  {item}" for item in report.toolchain)
    return "\n".join(lines)


def bootstrap(family_root: Path) -> str:
    """A ``variants=`` draft from the manifests and the accepted quantizations."""
    manifests = manifest_variants(family_root)
    keys = sorted(set(manifests.values()), key=lambda key: (key[1] != UNQUANTIZED, key))
    precisions = sorted({precision for precision, _ in keys}) or ["fp32"]
    covered = {quantization for _, quantization in keys}
    lines = ["variants=("]
    for precision, quantization in keys:
        argument = "" if quantization == UNQUANTIZED else f', quantization="{quantization}"'
        lines.append(f'    Variant(precision="{precision}"{argument}),')
    for quantization in sorted(accepted_quantizations(family_root) - covered):
        lines.append(
            f'    Variant(precision="{precisions[0]}", quantization="{quantization}", '
            'status="experimental"),  # confirm precision; no manifest yet'
        )
    lines.append("),")
    return "\n".join(lines)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    result.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    commands = result.add_subparsers(dest="command", required=True)
    report = commands.add_parser("report", help="print coverage gaps")
    report.add_argument("--family", action="append", default=[], help="only these families")
    report.add_argument("--json", action="store_true", help="print JSON instead of text")
    report.add_argument("--strict", action="store_true", help="exit 1 when any gap exists")
    draft = commands.add_parser("bootstrap", help="print a variants= draft for one family")
    draft.add_argument("family")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    repo = args.repo_root.resolve()
    try:
        if args.command == "bootstrap":
            print(bootstrap(family_root(repo, args.family)))
            return 0
        report = build_report(repo, tuple(args.family))
    except (OSError, ValueError, SyntaxError) as error:
        print(f"variant-coverage: {error}", file=sys.stderr)
        return 2
    if args.json:
        payload = {
            "families": [asdict(item) for item in report.families],
            "toolchain": report.toolchain,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(render(report))
    return 1 if args.strict and report.gaps else 0


if __name__ == "__main__":
    raise SystemExit(main())
