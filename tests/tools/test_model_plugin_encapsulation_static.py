"""Static regression checks for model plugin encapsulation boundaries.

Trace: ARCH-MODPLUG-001
Intent: keep model builder, runtime, and E2E ownership independently testable.
Preconditions: model-owned builder/runtime/E2E folders are present.
Postconditions: model folders do not import/include sibling model
implementations, and each model-owned manifest has local E2E sidecars.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_MODELS = REPO_ROOT / "src" / "runtime" / "models"
FAMILIES = REPO_ROOT / "python" / "tensorrt_model_connect" / "families"
E2E_MODELS = REPO_ROOT / "tests" / "e2e" / "models"

_RUNTIME_INCLUDE_RE = re.compile(
    r'#\s*include\s+[<"](?P<path>[^">]*runtime/models/(?P<model>[^/]+)/[^">]+)[">]'
)
_FORBIDDEN_E2E_IMPORT_RE = re.compile(
    r"tests\.e2e_harness\.(?:runners|comparators|references)"
)
_FORBIDDEN_SHARED_BUILDER_MODULES = {
    "checkpoint_mapper",
    "config",
    "graph_blocks",
    "graph_ops",
    "utils",
}


def _runtime_model_ids() -> set[str]:
    return {path.name for path in RUNTIME_MODELS.iterdir() if path.is_dir()}


def _family_model_ids() -> set[str]:
    return {
        path.name
        for path in FAMILIES.iterdir()
        if path.is_dir() and not path.name.startswith("__")
    }


def _format_violations(violations: list[tuple[Path, int, str]]) -> str:
    return "\n".join(
        f"{path.relative_to(REPO_ROOT)}:{line}: {detail}"
        for path, line, detail in violations
    )


def test_runtime_models_do_not_include_sibling_model_folders() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: prove each runtime model implementation includes only local model code.
    Preconditions: src/runtime/models/<model> folders exist.
    Postconditions: no file under one runtime model includes another model folder.
    """
    violations: list[tuple[Path, int, str]] = []
    for owner in sorted(_runtime_model_ids()):
        for path in (RUNTIME_MODELS / owner).rglob("*"):
            if path.suffix not in {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp"}:
                continue
            for line_no, line in enumerate(
                path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1
            ):
                match = _RUNTIME_INCLUDE_RE.search(line)
                if match and match.group("model") != owner:
                    violations.append((path, line_no, match.group("path")))

    assert not violations, _format_violations(violations)


def test_family_builders_do_not_import_sibling_or_forbidden_shared_helpers() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: prevent builder code from coupling one family to another family.
    Preconditions: python/tensorrt_model_connect/families/<model> folders exist.
    Postconditions: family builder code imports local helpers or generic APIs only.
    """
    family_ids = _family_model_ids()
    violations: list[tuple[Path, int, str]] = []

    for owner in sorted(family_ids):
        for path in (FAMILIES / owner).rglob("*.py"):
            if "tests" in path.relative_to(FAMILIES / owner).parts:
                continue
            tree = ast.parse(
                path.read_text(encoding="utf-8", errors="ignore"),
                filename=str(path),
            )
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        _check_absolute_import(
                            alias.name, owner, family_ids, path, node.lineno, violations
                        )
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if node.level == 0:
                        if module == "tensorrt_model_connect":
                            for alias in node.names:
                                if alias.name in _FORBIDDEN_SHARED_BUILDER_MODULES:
                                    violations.append((
                                        path,
                                        node.lineno,
                                        f"imports shared {module}.{alias.name}",
                                    ))
                        _check_absolute_import(
                            module, owner, family_ids, path, node.lineno, violations
                        )
                    else:
                        if node.level >= 2 and not module:
                            for alias in node.names:
                                if alias.name in _FORBIDDEN_SHARED_BUILDER_MODULES:
                                    violations.append((
                                        path,
                                        node.lineno,
                                        f"imports shared helper {'.' * node.level}{alias.name}",
                                    ))
                        _check_relative_import(
                            module,
                            node.level,
                            owner,
                            family_ids,
                            path,
                            node.lineno,
                            violations,
                        )

    assert not violations, _format_violations(violations)


def _check_absolute_import(
    module: str,
    owner: str,
    family_ids: set[str],
    path: Path,
    line_no: int,
    violations: list[tuple[Path, int, str]],
) -> None:
    parts = module.split(".")
    if parts[:2] != ["tensorrt_model_connect", "families"]:
        if (
            len(parts) >= 2
            and parts[0] == "tensorrt_model_connect"
            and parts[1] in _FORBIDDEN_SHARED_BUILDER_MODULES
        ):
            violations.append((path, line_no, f"imports shared {module}"))
        return

    if len(parts) >= 3 and parts[2] in family_ids and parts[2] != owner:
        violations.append((path, line_no, f"imports sibling family {module}"))


def _check_relative_import(
    module: str,
    level: int,
    owner: str,
    family_ids: set[str],
    path: Path,
    line_no: int,
    violations: list[tuple[Path, int, str]],
) -> None:
    first = module.split(".", 1)[0] if module else ""
    if level == 2 and first in family_ids and first != owner:
        violations.append((path, line_no, f"imports sibling family ..{module}"))
    if level >= 3 and first in _FORBIDDEN_SHARED_BUILDER_MODULES:
        violations.append((path, line_no, f"imports shared helper ...{module}"))


def test_model_owned_e2e_assets_are_local_and_complete() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep each model E2E contract runnable from its own model folder.
    Preconditions: tests/e2e/models/<model>/MODEL.toml declares manifests.
    Postconditions: each model has local entrypoints, plugins, and thresholds.
    """
    violations: list[tuple[Path, int, str]] = []

    for model_dir in sorted(E2E_MODELS.iterdir()):
        if not (model_dir / "MODEL.toml").is_file():
            continue
        expected_entrypoint = model_dir / f"test_{model_dir.name}_e2e.py"
        for required in (
            model_dir / "runner.py",
            expected_entrypoint,
            model_dir / "e2e_plugins",
        ):
            if not required.exists():
                violations.append((model_dir, 0, f"missing {required.name}"))

        for manifest in sorted((model_dir / "manifests").glob("*.json")):
            threshold = model_dir / "thresholds" / manifest.name
            if not threshold.is_file():
                violations.append((
                    manifest,
                    0,
                    f"missing threshold sidecar {threshold.relative_to(REPO_ROOT)}",
                ))

        for path in model_dir.rglob("*.py"):
            for line_no, line in enumerate(
                path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1
            ):
                if _FORBIDDEN_E2E_IMPORT_RE.search(line):
                    violations.append((
                        path,
                        line_no,
                        "imports shared E2E runner/reference/comparator",
                    ))

    assert not violations, _format_violations(violations)
