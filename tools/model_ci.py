#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact model ownership, impact, and positive source projection for CI.

The directory name below a ``MODEL.toml`` ownership root is its physical owner.
E2E ``runtime_strategy`` metadata normalizes the rare logical/runtime name
split into one independently testable module. Impact analysis selects only
modules touched by a diff. A model projection starts empty and materializes
tracked Git blobs for one module plus an explicit platform allowlist; sibling
model files are never copied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence


MODEL_ROOTS = (
    "python/tensorrt_model_connect/families",
    "src/runtime/models",
    "tests/e2e/models",
    "tests/cpp/models",
)

MODEL_ROOT_PLATFORM_FILES = frozenset(
    {
        "python/tensorrt_model_connect/families/__init__.py",
        "python/tensorrt_model_connect/families/_time_series_trt.py",
        "python/tensorrt_model_connect/families/base.py",
    }
)

# These are the only non-model source surfaces made visible to an isolated
# build.  Model-root handling takes precedence, so a sibling below e.g. src/
# is excluded even though src/ is otherwise an approved platform root.
PLATFORM_PROJECTION_EXACT = frozenset(
    {
        ".clang-format",
        ".github/scripts/write-model-proof-fallback-report.py",
        "CMakeLists.txt",
        "Dockerfile",
        "LICENSE",
        "NOTICE",
        "README.md",
        "_pyproject_backend.py",
        "conanfile.py",
        "conftest.py",
        "examples/byok/identity_copy_kernel.cpp",
        "pyproject.toml",
        "ruff.toml",
        "scripts/generate_e2e_report.py",
        "scripts/generate_e2e_report_assets/e2e_report.css",
        "scripts/generate_e2e_report_assets/e2e_report.js",
        "scripts/reporting/__init__.py",
        "scripts/reporting/vlm_assessment.py",
        "scripts/schedule_e2e.py",
        "scripts/hf_cache_download_worker.py",
        "scripts/warm_hf_cache.py",
        "tests/__init__.py",
        "tests/builder/__init__.py",
        "tests/builder/conftest.py",
        "tests/builder/debug_runner_test_support.py",
        "tests/builder/family_plugin_test_mixin.py",
        "tests/builder/family_plugin_test_support.py",
        "tests/builder/family_plugin_tester.py",
        "tests/e2e_partition.py",
        "tests/runtime_strategy_matrix.yaml",
        "tests/task_eval/validation_suites.yaml",
        "tests/test_e2e.py",
        "tests/test_e2e_selection.py",
        "tests/test_tvm_ffi_e2e.py",
        "tools/__init__.py",
        "tools/diff_logits.py",
        "tools/diff_vl.py",
        "tools/diffusion_helpers.py",
        "tools/elf_hf_reference.py",
        "tools/model_plugin_isolation.py",
        "tools/reference/elf_prepared.py",
        "tools/reference/plugin_reference.py",
        "tools/reference/speech.py",
        "tools/reference/transformers_encoder.py",
        "tools/reference/transformers_text.py",
        "tools/reference/transformers_vlm.py",
        "tools/task_eval.py",
        "tools/test_impact.py",
        "tools/tool_helpers.py",
        "tools/trtmc_reference.py",
        *MODEL_ROOT_PLATFORM_FILES,
    }
)
PLATFORM_PROJECTION_PREFIXES = (
    "cmake/",
    "include/",
    "python/tensorrt_model_connect/",
    "src/",
    "tensorrt_model_connect/",
    "tests/cpp/",
    "tests/assets/",
    "tests/e2e/",
    "tests/e2e_harness/",
    "third_party/",
    "tools/ci/",
)

LEGAL_OR_DOC_EXACT = frozenset(
    {
        "AGENTS.md",
        "CODEOWNERS",
        "CONTRIBUTING.md",
        "LICENSE",
        "NOTICE",
        "README.md",
        ".github/workflows/legal.yml",
        "tools/legal_header_exceptions.toml",
        "tools/legal_headers.py",
    }
)
LEGAL_OR_DOC_PREFIXES = ("website/",)

# Non-code inputs consumed directly by the CPU-only builder contract suite.
# Classify these before the blanket documentation rule so a documentation
# rewrite cannot skip the tests that define its normative contract.
BUILDER_UNIT_TEST_INPUT_EXACT = frozenset(
    {
        "website/docs/wiki/Agentic-Quantization-Core-Minimal-Plan.md",
    }
)
# Inputs consumed by the complete source-only tools/CI contract suite.
FULL_UNIT_TEST_INPUT_EXACT = frozenset(
    {
        "tools/ci/README.md",
    }
)

# These shared surfaces are covered by CPU/C++ unit tests and do not change a
# model implementation or its isolated E2E contract. Model-root ownership is
# resolved before this list, so model-owned C++ tests remain model impact.
UNIT_TEST_ONLY_EXACT = frozenset(
    {
        "include/trtmc/config/cli_support.h",
        "python/tensorrt_model_connect/__main__.py",
        "python/tensorrt_model_connect/build_cli.py",
        "python/tensorrt_model_connect/runtime_config/cli_support.py",
        "src/runtime/config/cli_support.cpp",
    }
)
# Task-eval and the selective impact analyzer are certified by the complete
# source-only tools suite. Selective premerge proofs do not execute these
# entrypoints; the all-model nightly separately consumes task-eval where
# applicable, so treating a tool-only change as broad premerge model impact
# would add GPU work without validating the changed behavior.
FULL_UNIT_TEST_ONLY_EXACT = frozenset(
    {
        "tools/elf_hf_reference.py",
        "tools/model_ci.py",
        "tools/prepare_elf_task_eval_datasets.py",
        "tools/prepare_media_task_eval_datasets.py",
        "tools/task_eval.py",
        "tools/test_impact.py",
    }
)
UNIT_TEST_ONLY_PREFIXES = (
    "tests/builder/",
    "tests/cpp/",
    "tests/task_eval/",
    "tests/tools/",
)
# These shared-location tests require real model plugins or GPU execution and
# therefore cannot be certified by the source-only CPU aggregate. Fail closed
# on a direct edit until each test is moved behind explicit model ownership or
# uses a synthetic CPU fixture.
MODEL_COUPLED_TEST_EXACT = frozenset(
    {
        "tests/builder/test_dynamic_batch_profile.py",
        "tests/builder/test_flashinfer_benchmark.py",
        "tests/builder/test_graph_blocks.py",
        "tests/builder/test_tvm_ffi_plugin.py",
        "tests/cpp/test_c_abi_runtime_regression.cpp",
        "tests/cpp/test_cuda_buffer.cpp",
        "tests/cpp/test_cuda_graph.cpp",
        "tests/cpp/test_cuda_stream.cpp",
        "tests/cpp/test_device_tensor.cpp",
        "tests/cpp/test_model_plugin_loader.cpp",
        "tests/cpp/test_trt_module.cpp",
        "tests/cpp/test_trt_runtime_lifetime.cpp",
        "tests/cpp/test_tvm_ffi_module_loader.cpp",
        "tests/cpp/test_tvm_ffi_plugin.cpp",
        "tests/cpp/test_tvm_ffi_plugin_v2.cpp",
    }
)

PLATFORM_EXACT = frozenset(
    {
        ".clang-format",
        ".dockerignore",
        ".gitignore",
        "CMakeLists.txt",
        "Dockerfile",
        "Dockerfile.tensorrt-sdk",
        "_pyproject_backend.py",
        "conanfile.py",
        "conftest.py",
        "docker-compose.yml",
        "hf_links_wave1.txt",
        "pyproject.toml",
        "ruff.toml",
        "verify_encoder.py",
        *MODEL_ROOT_PLATFORM_FILES,
    }
)
PLATFORM_PREFIXES = (
    "cmake/",
    "examples/",
    "include/",
    "python/",
    "src/",
    "tensorrt_model_connect/",
    "tests/",
    "third_party/",
)
CI_OR_TOOLING_PREFIXES = (
    ".agents/",
    ".ci/",
    ".codex/",
    ".github/",
    "agent_bench/",
    "plugins/",
    "reports/",
    "scripts/",
    "tools/",
)

_MODEL_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_MANIFEST_ID_RE = re.compile(r'(?m)^\s*id\s*=\s*"([^"]+)"\s*$')
_TASK_EVAL_OPTIONAL_EXTRA_RE = re.compile(r"task-eval\s*=\s*\[.*\]\s*\Z")


class ModelCIError(RuntimeError):
    """A fail-closed ownership or projection error."""


@dataclass(frozen=True)
class TreeEntry:
    mode: str
    object_type: str
    object_id: str
    path: str


@dataclass(frozen=True)
class OwnershipCatalog:
    revision: str
    entries: tuple[TreeEntry, ...]
    models: tuple[str, ...]
    manifests: dict[str, tuple[str, ...]]
    owners_by_root: dict[str, dict[str, str]]
    runtime_models: dict[str, tuple[str, ...]]
    e2e_families: dict[str, tuple[str, ...]]
    legacy_shared_runtime: tuple[str, ...]


@dataclass(frozen=True)
class DiffEntry:
    status: str
    old_path: str | None
    new_path: str | None


@dataclass(frozen=True)
class ModelOwnedImpactRule:
    owner: str
    name: str
    path: str
    allowed_tokens: tuple[str, ...]
    changed_lines_sha256: str


def _run_git(repo_root: Path, args: Sequence[str], *, text: bool = False):
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=text,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            stderr = exc.stderr if isinstance(exc.stderr, str) else os.fsdecode(exc.stderr)
            detail = f": {stderr.strip()}"
        raise ModelCIError(f"git {' '.join(args)} failed{detail}") from exc


def _resolve_revision(repo_root: Path, revision: str) -> str:
    return str(_run_git(repo_root, ["rev-parse", f"{revision}^{{commit}}"], text=True)).strip()


def _read_tree(repo_root: Path, revision: str) -> tuple[TreeEntry, ...]:
    raw = _run_git(repo_root, ["ls-tree", "-rz", "--full-tree", revision])
    entries: list[TreeEntry] = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode, object_type, object_id = metadata.decode("ascii").split(" ")
        except ValueError as exc:
            raise ModelCIError(f"could not parse git tree entry at {revision}") from exc
        path = os.fsdecode(raw_path)
        _validate_git_path(path)
        entries.append(TreeEntry(mode, object_type, object_id, path))
    return tuple(entries)


def _validate_git_path(path: str) -> None:
    pure = PurePosixPath(path)
    if not path or pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise ModelCIError(f"unsafe path in Git tree: {path!r}")


def _read_blob(repo_root: Path, object_id: str) -> bytes:
    return bytes(_run_git(repo_root, ["cat-file", "blob", object_id]))


def _toml_strings(text: str, key: str) -> tuple[str, ...]:
    list_match = re.search(rf"(?ms)^\s*{re.escape(key)}\s*=\s*\[([^]]*)\]", text)
    if list_match is not None:
        return tuple(re.findall(r'"([^"]+)"', list_match.group(1)))
    scalar_match = re.search(rf'(?m)^\s*{re.escape(key)}\s*=\s*"([^"]+)"\s*$', text)
    return (scalar_match.group(1),) if scalar_match is not None else ()


def _manifest_location(path: str) -> tuple[str, str] | None:
    for root in MODEL_ROOTS[:-1]:
        prefix = f"{root}/"
        if not path.startswith(prefix):
            continue
        relative = path[len(prefix) :].split("/")
        if len(relative) == 2 and relative[1] == "MODEL.toml":
            return root, relative[0]
    return None


def _validate_model_roots() -> None:
    roots = [PurePosixPath(root) for root in MODEL_ROOTS]
    for index, left in enumerate(roots):
        for right in roots[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise ModelCIError(f"overlapping model ownership roots: {left} and {right}")


def _required_gpu_count(payload: dict[str, object], path: str) -> int:
    """Return the declared device count for one effective E2E testcase."""
    required = 1
    build_args = payload.get("build_args", {})
    distributed = payload.get("distributed_runtime", {})
    if not isinstance(build_args, dict) or not isinstance(distributed, dict):
        raise ModelCIError(f"E2E manifest device settings must be objects: {path}")
    parallel = build_args.get("parallel", {})
    if not isinstance(parallel, dict):
        raise ModelCIError(f"E2E manifest build_args.parallel must be an object: {path}")
    for field, value in (
        ("build_args.parallel.tp_size", parallel.get("tp_size")),
        ("distributed_runtime.world_size", distributed.get("world_size")),
    ):
        if value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ModelCIError(f"E2E manifest {field} must be a positive integer: {path}")
        required = max(required, value)
    preflights = payload.get("preflight_requirements", [])
    if not isinstance(preflights, list):
        raise ModelCIError(f"E2E manifest preflight_requirements must be an array: {path}")
    for preflight in preflights:
        if not isinstance(preflight, dict) or preflight.get("kind") != "gpu_count_min":
            continue
        args = preflight.get("args", {})
        value = args.get("count") if isinstance(args, dict) else None
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ModelCIError(
                f"E2E manifest gpu_count_min count must be a positive integer: {path}"
            )
        required = max(required, value)
    return required


def _validate_e2e_manifest_device_tiers(
    payload: dict[str, object],
    path: str,
    *,
    allow_legacy_tier: bool = False,
) -> None:
    defaults = {key: value for key, value in payload.items() if key != "testcases"}
    testcases = payload.get("testcases")
    effective_cases: list[dict[str, object]]
    if isinstance(testcases, list) and testcases:
        effective_cases = []
        for index, testcase in enumerate(testcases):
            if not isinstance(testcase, dict):
                raise ModelCIError(f"E2E manifest testcases[{index}] must be an object: {path}")
            effective_cases.append({**defaults, **testcase})
    else:
        effective_cases = [defaults]
    for testcase in effective_cases:
        required = _required_gpu_count(testcase, path)
        if (
            required > 1
            and testcase.get("ci_tier") != "multi_device"
            and not allow_legacy_tier
        ):
            name = testcase.get("name", payload.get("name", "<unnamed>"))
            raise ModelCIError(
                f"E2E testcase {name!r} requires {required} GPUs but is not "
                f"ci_tier='multi_device': {path}"
            )


def discover_catalog(
    repo_root: Path,
    revision: str,
    *,
    allow_legacy_shared_runtime: bool = False,
    allow_legacy_device_tier: bool = False,
) -> OwnershipCatalog:
    """Discover model IDs from MODEL.toml blobs at one Git revision."""
    _validate_model_roots()
    resolved = _resolve_revision(repo_root, revision)
    entries = _read_tree(repo_root, resolved)
    manifest_roots_by_physical: dict[str, list[str]] = {}
    manifest_text: dict[tuple[str, str], str] = {}
    seen_locations: set[tuple[str, str]] = set()
    for entry in entries:
        location = _manifest_location(entry.path)
        if location is None:
            continue
        if entry.object_type != "blob":
            raise ModelCIError(f"model manifest is not a blob: {entry.path}")
        root, directory_id = location
        try:
            text = _read_blob(repo_root, entry.object_id).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ModelCIError(f"model manifest is not UTF-8: {entry.path}") from exc
        match = _MANIFEST_ID_RE.search(text)
        if match is None:
            raise ModelCIError(f"model manifest has no string id: {entry.path}")
        declared_id = match.group(1)
        if not _MODEL_ID_RE.fullmatch(declared_id):
            raise ModelCIError(f"model manifest has unsafe id {declared_id!r}: {entry.path}")
        if declared_id != directory_id:
            raise ModelCIError(
                f"model manifest id {declared_id!r} does not match directory "
                f"{directory_id!r}: {entry.path}"
            )
        if location in seen_locations:
            raise ModelCIError(f"duplicate model ownership manifest: {entry.path}")
        seen_locations.add(location)
        manifest_roots_by_physical.setdefault(declared_id, []).append(root)
        manifest_text[(root, declared_id)] = text
    if not manifest_roots_by_physical:
        raise ModelCIError(f"no MODEL.toml ownership manifests found at {resolved}")

    runtime_root = "src/runtime/models"
    e2e_root = "tests/e2e/models"
    runtime_ids = {
        model for model, roots in manifest_roots_by_physical.items() if runtime_root in roots
    }
    e2e_ids = {model for model, roots in manifest_roots_by_physical.items() if e2e_root in roots}
    strategy_owner: dict[str, str] = {}
    for runtime_id in sorted(runtime_ids):
        text = manifest_text[(runtime_root, runtime_id)]
        strategies = _toml_strings(text, "runtime_strategies") or _toml_strings(
            text, "runtime_strategy"
        )
        for strategy in strategies:
            previous = strategy_owner.get(strategy)
            if previous is not None and previous != runtime_id:
                raise ModelCIError(
                    f"runtime strategy {strategy!r} is owned by both {previous!r} "
                    f"and {runtime_id!r}"
                )
            strategy_owner[strategy] = runtime_id

    e2e_runtime_candidates: dict[str, set[str]] = {model: set() for model in e2e_ids}
    e2e_prefix = f"{e2e_root}/"
    for entry in entries:
        if not entry.path.startswith(e2e_prefix) or "/manifests/" not in entry.path:
            continue
        relative = entry.path[len(e2e_prefix) :]
        family = relative.split("/", 1)[0]
        if family not in e2e_ids or not entry.path.endswith(".json"):
            continue
        try:
            payload = json.loads(_read_blob(repo_root, entry.object_id))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ModelCIError(f"invalid E2E manifest JSON: {entry.path}") from exc
        if not isinstance(payload, dict):
            raise ModelCIError(f"E2E manifest must contain an object: {entry.path}")
        _validate_e2e_manifest_device_tiers(
            payload,
            entry.path,
            allow_legacy_tier=allow_legacy_device_tier,
        )
        strategy = payload.get("runtime_strategy")
        if strategy is None:
            continue
        runtime_id = strategy_owner.get(str(strategy))
        if runtime_id is None:
            raise ModelCIError(
                f"E2E manifest uses unowned runtime strategy {strategy!r}: {entry.path}"
            )
        e2e_runtime_candidates[family].add(runtime_id)

    e2e_to_runtime: dict[str, str] = {}
    for family in sorted(e2e_ids):
        candidates = e2e_runtime_candidates[family]
        if family in runtime_ids:
            candidates.add(family)
        if len(candidates) > 1:
            raise ModelCIError(
                f"E2E family {family!r} depends on multiple runtime models: {sorted(candidates)}"
            )
        if candidates:
            e2e_to_runtime[family] = next(iter(candidates))

    runtime_to_logical: dict[str, str] = {}
    legacy_shared_runtime: set[str] = set()
    for runtime_id in sorted(runtime_ids):
        logical_owners = sorted(
            family for family, candidate in e2e_to_runtime.items() if candidate == runtime_id
        )
        if len(logical_owners) > 1:
            if not allow_legacy_shared_runtime:
                raise ModelCIError(
                    f"runtime model {runtime_id!r} is shared by multiple model modules: "
                    f"{logical_owners}"
                )
            # Old merge bases can predate the one-runtime-owner invariant. Keep
            # their shared runtime root as a conservative platform-like owner;
            # a change to it fans out below because it is absent from the head
            # catalog's logical modules.
            runtime_to_logical[runtime_id] = runtime_id
            legacy_shared_runtime.add(runtime_id)
        else:
            runtime_to_logical[runtime_id] = logical_owners[0] if logical_owners else runtime_id

    owners_by_root: dict[str, dict[str, str]] = {root: {} for root in MODEL_ROOTS}
    for physical_id, roots in manifest_roots_by_physical.items():
        for root in roots:
            if root == runtime_root:
                logical = runtime_to_logical[physical_id]
            else:
                logical = physical_id
            owners_by_root[root][physical_id] = logical

    cpp_root = "tests/cpp/models"
    cpp_prefix = f"{cpp_root}/"
    cpp_children = {
        entry.path[len(cpp_prefix) :].split("/", 1)[0]
        for entry in entries
        if entry.path.startswith(cpp_prefix) and "/" in entry.path[len(cpp_prefix) :]
    }
    logical_ids = set(e2e_ids) | set(manifest_roots_by_physical)
    for child in cpp_children:
        if child in runtime_to_logical:
            owners_by_root[cpp_root][child] = runtime_to_logical[child]
        elif child in logical_ids:
            owners_by_root[cpp_root][child] = child

    models = sorted(logical for owners in owners_by_root.values() for logical in owners.values())
    models = sorted(set(models) - legacy_shared_runtime)
    manifests: dict[str, list[str]] = {model: [] for model in models}
    for root, owners in owners_by_root.items():
        for physical, logical in owners.items():
            if logical not in manifests:
                continue
            prefix = f"{root}/{physical}/"
            if any(entry.path.startswith(prefix) for entry in entries):
                manifests[logical].append(f"{root}/{physical}")
    runtime_models: dict[str, tuple[str, ...]] = {}
    for runtime_id, logical in runtime_to_logical.items():
        runtime_models.setdefault(logical, ())
        runtime_models[logical] = tuple(sorted({*runtime_models[logical], runtime_id}))
    e2e_families = {model: (model,) for model in sorted(e2e_ids) if model in models}
    return OwnershipCatalog(
        resolved,
        entries,
        tuple(models),
        {model: tuple(sorted(roots)) for model, roots in manifests.items()},
        owners_by_root,
        runtime_models,
        e2e_families,
        tuple(sorted(legacy_shared_runtime)),
    )


def _owner_for_path(path: str, catalog: OwnershipCatalog) -> tuple[str | None, bool]:
    """Return (owner, under_model_root). An unregistered child has no owner."""
    matches: list[str] = []
    under_model_root = False
    for root in MODEL_ROOTS:
        prefix = f"{root}/"
        if not path.startswith(prefix):
            continue
        under_model_root = True
        relative = path[len(prefix) :]
        if not relative or "/" not in relative:
            continue
        candidate = relative.split("/", 1)[0]
        owner = catalog.owners_by_root[root].get(candidate)
        if owner is not None:
            matches.append(owner)
    unique = sorted(set(matches))
    if len(unique) > 1:
        raise ModelCIError(f"path has overlapping model owners: {path}: {unique}")
    return (unique[0] if unique else None), under_model_root


def _normalize_diff_line(line: str) -> str:
    return re.sub(r"[-\s]+", "_", line.lower())


def _significant_diff_lines(diff_text: str) -> tuple[str, ...]:
    lines: list[str] = []
    for raw_line in diff_text.splitlines():
        if raw_line.startswith(("diff --git", "index ", "@@", "---", "+++")):
            continue
        if not raw_line.startswith(("+", "-")):
            continue
        content = raw_line[1:]
        if not content.strip():
            continue
        lines.append(content)
    return tuple(lines)


def _model_owned_impact_rules(
    repo_root: Path,
    catalog: OwnershipCatalog,
) -> tuple[ModelOwnedImpactRule, ...]:
    rules: list[ModelOwnedImpactRule] = []
    catalog_paths = {entry.path for entry in catalog.entries}
    suffix = "/impact_diff_rules.json"
    prefix = "tests/e2e/models/"
    for entry in catalog.entries:
        if not entry.path.startswith(prefix) or not entry.path.endswith(suffix):
            continue
        owner, under_model_root = _owner_for_path(entry.path, catalog)
        if owner is None or not under_model_root:
            raise ModelCIError(
                f"impact rule file has no model owner: {entry.path}"
            )
        try:
            payload = json.loads(_read_blob(repo_root, entry.object_id))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ModelCIError(f"invalid model impact rules: {entry.path}") from exc
        if not isinstance(payload, list):
            raise ModelCIError(
                f"model impact rules must be a JSON list: {entry.path}"
            )
        for index, raw_rule in enumerate(payload, start=1):
            if not isinstance(raw_rule, dict):
                raise ModelCIError(
                    f"model impact rule must be an object: {entry.path}:{index}"
                )
            name = raw_rule.get("name")
            path = raw_rule.get("path")
            allowed_tokens = raw_rule.get("allowed_tokens")
            scope = raw_rule.get("scope")
            if not (
                isinstance(name, str)
                and name
                and isinstance(path, str)
                and path
                and isinstance(allowed_tokens, list)
                and allowed_tokens
                and isinstance(scope, dict)
            ):
                raise ModelCIError(
                    "model impact rule requires non-empty name, path, "
                    f"allowed_tokens, and object scope: {entry.path}:{index}"
                )
            if not all(
                isinstance(token, str) and _normalize_diff_line(token)
                for token in allowed_tokens
            ):
                raise ModelCIError(
                    "model impact rule allowed_tokens must be non-empty strings: "
                    f"{entry.path}:{index}"
                )
            _validate_git_path(path)
            if scope != {"owner_family": True}:
                continue
            changed_lines_sha256 = raw_rule.get(
                "model_ci_changed_lines_sha256"
            )
            if not isinstance(changed_lines_sha256, str) or re.fullmatch(
                r"[0-9a-f]{64}",
                changed_lines_sha256,
            ) is None:
                continue
            target_owner, under_model_root = _owner_for_path(path, catalog)
            if target_owner is not None or under_model_root:
                raise ModelCIError(
                    "model impact rule target must be a shared path: "
                    f"{entry.path}:{index}: {path}"
                )
            if path not in catalog_paths:
                raise ModelCIError(
                    "model impact rule target is absent from the revision: "
                    f"{entry.path}:{index}: {path}"
                )
            rules.append(
                ModelOwnedImpactRule(
                    owner=owner,
                    name=name,
                    path=path,
                    allowed_tokens=tuple(
                        _normalize_diff_line(token) for token in allowed_tokens
                    ),
                    changed_lines_sha256=changed_lines_sha256,
                )
            )
    return tuple(rules)


def _refined_impact_rule(
    repo_root: Path,
    base: str,
    head: str,
    change: DiffEntry,
    rules: Sequence[ModelOwnedImpactRule],
) -> ModelOwnedImpactRule | None:
    if (
        change.status != "M"
        or change.old_path is None
        or change.new_path != change.old_path
    ):
        return None
    path = change.new_path
    matching_path_rules = [rule for rule in rules if rule.path == path]
    if not matching_path_rules:
        return None
    diff_text = str(
        _run_git(
            repo_root,
            [
                "diff",
                "--no-ext-diff",
                "--no-color",
                "--unified=0",
                base,
                head,
                "--",
                path,
            ],
            text=True,
        )
    )
    lines = _significant_diff_lines(diff_text)
    if not lines:
        return None
    normalized_lines = tuple(_normalize_diff_line(line) for line in lines)
    digest = hashlib.sha256(
        "\n".join(lines).encode("utf-8")
    ).hexdigest()
    matches = [
        rule
        for rule in matching_path_rules
        if rule.changed_lines_sha256 == digest
        and all(
            any(token in line for token in rule.allowed_tokens)
            for line in normalized_lines
        )
    ]
    return matches[0] if len(matches) == 1 else None


def _is_legal_or_docs(path: str) -> bool:
    return (
        path in LEGAL_OR_DOC_EXACT
        or path.endswith(".md")
        or any(path.startswith(prefix) for prefix in LEGAL_OR_DOC_PREFIXES)
        or path.startswith("tests/tools/test_legal_")
    )


def _merge_unit_scope(current: str, requested: str) -> str:
    if current == requested or requested == "none":
        return current
    if current == "none":
        return requested
    if current == "all" or requested == "all":
        return "all"
    return "all"


def _classify_path(path: str, catalog: OwnershipCatalog) -> tuple[str, str | None]:
    if path in MODEL_ROOT_PLATFORM_FILES:
        return "platform", None
    owner, under_model_root = _owner_for_path(path, catalog)
    if owner is not None:
        return "model", owner
    if under_model_root:
        raise ModelCIError(f"path is under a model root but has no MODEL.toml owner: {path}")
    if path in BUILDER_UNIT_TEST_INPUT_EXACT:
        return "unit_builder", None
    if path in FULL_UNIT_TEST_INPUT_EXACT:
        return "unit_tests", None
    if _is_legal_or_docs(path):
        return "legal_docs", None
    if path in MODEL_COUPLED_TEST_EXACT:
        raise ModelCIError(
            "model-coupled test has no isolated model owner; move it into a "
            f"MODEL.toml contract or use a synthetic plugin before changing it: {path}"
        )
    if path in UNIT_TEST_ONLY_EXACT:
        return "unit_cli", None
    if path in FULL_UNIT_TEST_ONLY_EXACT or any(
        path.startswith(prefix) for prefix in UNIT_TEST_ONLY_PREFIXES
    ):
        return "unit_tests", None
    if path in PLATFORM_EXACT or any(path.startswith(prefix) for prefix in PLATFORM_PREFIXES):
        return "platform", None
    if any(path.startswith(prefix) for prefix in CI_OR_TOOLING_PREFIXES):
        return "ci_tooling", None
    return "unknown", None


def _diff_entries(repo_root: Path, base: str, head: str) -> tuple[DiffEntry, ...]:
    base_sha = _resolve_revision(repo_root, base)
    head_sha = _resolve_revision(repo_root, head)
    try:
        merge_base = str(_run_git(repo_root, ["merge-base", base_sha, head_sha], text=True)).strip()
    except ModelCIError:
        merge_base = base_sha
    raw = _run_git(
        repo_root,
        ["diff", "--name-status", "-z", "--find-renames", merge_base, head_sha],
    )
    fields = raw.split(b"\0")
    entries: list[DiffEntry] = []
    index = 0
    while index < len(fields) and fields[index]:
        status = fields[index].decode("ascii")
        index += 1
        code = status[0]
        if code in {"R", "C"}:
            if index + 1 >= len(fields):
                raise ModelCIError("truncated rename/copy record in git diff")
            old_path = os.fsdecode(fields[index])
            new_path = os.fsdecode(fields[index + 1])
            index += 2
        else:
            if index >= len(fields):
                raise ModelCIError("truncated path record in git diff")
            path = os.fsdecode(fields[index])
            index += 1
            old_path = None if code == "A" else path
            new_path = None if code == "D" else path
        if code not in {"A", "C", "D", "M", "R", "T"}:
            raise ModelCIError(f"unsupported git diff status: {status}")
        for path in (old_path, new_path):
            if path is not None:
                _validate_git_path(path)
        entries.append(DiffEntry(status, old_path, new_path))
    return tuple(entries)


def _is_task_eval_optional_extra_only_change(
    repo_root: Path,
    base: str,
    head: str,
) -> bool:
    """Return whether pyproject changed only the one-line task-eval extra.

    Keep this deliberately fail-closed: comments, multiline rewrites, build
    metadata, and runtime dependency changes remain platform impact until a
    reviewer adds a more precise contract.
    """
    diff = str(
        _run_git(
            repo_root,
            [
                "diff",
                "--no-ext-diff",
                "--no-color",
                "--unified=0",
                base,
                head,
                "--",
                "pyproject.toml",
            ],
            text=True,
        )
    )
    changed_lines = [
        line[1:].strip()
        for line in diff.splitlines()
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    ]
    return bool(changed_lines) and all(
        _TASK_EVAL_OPTIONAL_EXTRA_RE.fullmatch(line) is not None for line in changed_lines
    )


def _result(
    models: Iterable[str],
    *,
    mode: str,
    changes: list[dict[str, object]],
    matrix_models: Iterable[str] | None = None,
    direct_models: Iterable[str] | None = None,
    fallback_models: Iterable[str] = (),
    run_unit_tests: bool = False,
    unit_scope: str = "none",
) -> dict[str, object]:
    selected = sorted(set(models))
    direct = set(selected if direct_models is None else direct_models)
    fallback = set(fallback_models)
    if direct.intersection(fallback):
        raise ModelCIError("direct and fallback model selections must not overlap")
    if direct.union(fallback) != set(selected):
        raise ModelCIError("direct and fallback selections must explain every affected model")
    scheduled = list(matrix_models) if matrix_models is not None else selected
    if len(scheduled) != len(set(scheduled)) or set(scheduled) != set(selected):
        raise ModelCIError("matrix model order must contain each affected model exactly once")
    return {
        "schema_version": 3,
        "mode": mode,
        "has_models": bool(selected),
        "expected_count": len(selected),
        "affected_models": selected,
        "direct_models": sorted(direct),
        "fallback_models": sorted(fallback),
        "matrix": {
            "include": [
                {
                    "model": model,
                    "selection_kind": "fallback" if model in fallback else "direct",
                }
                for model in scheduled
            ]
        },
        "run_unit_tests": run_unit_tests,
        "unit_scope": unit_scope,
        "changes": changes,
    }


def _scheduled_models(
    repo_root: Path,
    catalog: OwnershipCatalog,
    models: Iterable[str],
    *,
    exclusive_gpu_first: bool = False,
) -> tuple[list[str], dict[str, list[str]]]:
    """Order model matrix entries by resource class and pinned timing data.

    GitHub starts matrix children in include order when runner capacity becomes
    available.  Nightly can put exclusive-GPU models first so they reserve full
    devices before shared work fills every slot.  Its duration is the sum of
    every nightly-selected case, or unknown when any case lacks an estimate.
    Within each resource class, unknown timings run first because they are the
    least bounded, followed by longest-known first.  Premerge continues to use
    the one selected L0/contract/fastest case.  Allocation safety is still
    enforced later from the projected manifest rather than trusting this hint.
    The second return value maps every owner to the exact sorted production
    single-GPU cases that nightly must certify, including an owner's L0
    fallback only when that owner has no production case.
    """
    entries_by_path = {entry.path: entry for entry in catalog.entries}
    timing_estimates: dict[str, float] = {}
    timing_entry = entries_by_path.get("tests/e2e/timing_estimates.json")
    if timing_entry is not None:
        try:
            payload = json.loads(_read_blob(repo_root, timing_entry.object_id))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ModelCIError("invalid tests/e2e/timing_estimates.json") from exc
        raw_estimates = payload.get("estimates_s", {})
        if isinstance(raw_estimates, dict):
            timing_estimates = {
                str(name): float(value)
                for name, value in raw_estimates.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }

    selected_models = set(models)
    estimates: dict[str, float | None] = {}
    exclusive_gpu: dict[str, bool] = {}
    expected_cases_by_model: dict[str, list[str]] = {}
    for model in selected_models:
        cases: list[dict[str, object]] = []
        for family in catalog.e2e_families.get(model, ()):
            prefix = f"tests/e2e/models/{family}/manifests/"
            for entry in catalog.entries:
                if not entry.path.startswith(prefix) or not entry.path.endswith(".json"):
                    continue
                try:
                    manifest = json.loads(_read_blob(repo_root, entry.object_id))
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise ModelCIError(f"invalid E2E manifest JSON: {entry.path}") from exc
                if manifest.get("skip_reason") or manifest.get("skip"):
                    continue
                testcases = manifest.get("testcases", [])
                if not isinstance(testcases, list):
                    continue
                for testcase in testcases:
                    if not isinstance(testcase, dict):
                        continue
                    if testcase.get("skip_reason") or testcase.get("skip"):
                        continue
                    name = str(testcase.get("name") or "")
                    if not name:
                        continue
                    tier = str(testcase.get("ci_tier") or manifest.get("ci_tier") or "")
                    if tier == "multi_device":
                        continue
                    cases.append(
                        {
                            "name": name,
                            "tier": tier,
                            "l0_replacement": str(testcase.get("l0_replacement") or ""),
                            "estimated_seconds": timing_estimates.get(name),
                            "resource_class": str(
                                manifest.get("e2e_parallel_resource") or "shared"
                            ),
                        }
                    )
        production_cases = [case for case in cases if case["tier"] != "l0_only"]
        nightly_cases = production_cases or cases
        if not nightly_cases and exclusive_gpu_first:
            raise ModelCIError(
                f"model {model!r} has no active single-GPU nightly E2E case"
            )
        expected_cases = sorted(str(case["name"]) for case in nightly_cases)
        if len(expected_cases) != len(set(expected_cases)):
            raise ModelCIError(
                f"model {model!r} has duplicate active nightly E2E case names"
            )
        expected_cases_by_model[model] = expected_cases
        exclusive_gpu[model] = any(
            case["resource_class"] == "exclusive_gpu" for case in nightly_cases
        )
        if exclusive_gpu_first:
            nightly_estimates = [case["estimated_seconds"] for case in nightly_cases]
            estimates[model] = (
                sum(float(estimate) for estimate in nightly_estimates)
                if nightly_estimates
                and all(isinstance(estimate, (int, float)) for estimate in nightly_estimates)
                else None
            )
            continue
        eligible = [case for case in cases if case["tier"] != "nightly_only"]
        replacements = {
            str(case["l0_replacement"])
            for case in cases
            if case["tier"] == "nightly_only" and case["l0_replacement"]
        }
        candidates = [case for case in eligible if case["name"] in replacements] or eligible
        priority = {"l0_only": 0, "contract_only": 1, "": 2}
        candidates.sort(
            key=lambda case: (
                priority.get(str(case["tier"]), 2),
                case["estimated_seconds"]
                if isinstance(case["estimated_seconds"], (int, float))
                else float("inf"),
                str(case["name"]),
            )
        )
        selected_estimate = candidates[0]["estimated_seconds"] if candidates else None
        estimates[model] = (
            float(selected_estimate)
            if isinstance(selected_estimate, (int, float))
            else None
        )

    return (
        sorted(
            selected_models,
            key=lambda model: (
                0 if not exclusive_gpu_first or exclusive_gpu.get(model, False) else 1,
                0 if estimates.get(model) is None else 1,
                -(estimates.get(model) or 0.0),
                model,
            ),
        ),
        {
            model: expected_cases_by_model[model]
            for model in sorted(expected_cases_by_model)
        },
    )


def calculate_impact(
    repo_root: Path,
    base: str,
    head: str,
    *,
    platform_change_policy: str,
    fallback_models: Sequence[str] = (),
) -> dict[str, object]:
    base_sha = _resolve_revision(repo_root, base)
    head_sha = _resolve_revision(repo_root, head)
    try:
        comparison_base = str(
            _run_git(repo_root, ["merge-base", base_sha, head_sha], text=True)
        ).strip()
    except ModelCIError:
        comparison_base = base_sha
    base_catalog = discover_catalog(
        repo_root,
        comparison_base,
        allow_legacy_shared_runtime=True,
        allow_legacy_device_tier=True,
    )
    head_catalog = discover_catalog(repo_root, head, allow_legacy_shared_runtime=True)
    impact_rules = _model_owned_impact_rules(repo_root, head_catalog)
    affected: set[str] = set()
    fallback_selected: set[str] = set()
    broad_change = False
    unit_scope = "none"
    pyproject_task_eval_only = _is_task_eval_optional_extra_only_change(
        repo_root,
        comparison_base,
        head_sha,
    )
    serialized_changes: list[dict[str, object]] = []
    for change in _diff_entries(repo_root, comparison_base, head_sha):
        refined_rule = _refined_impact_rule(
            repo_root,
            comparison_base,
            head_sha,
            change,
            impact_rules,
        )
        classifications: list[dict[str, str]] = []
        path_catalogs = (
            (change.old_path, base_catalog),
            (change.new_path, head_catalog),
        )
        seen_path_revision: set[tuple[str, str]] = set()
        for path, catalog in path_catalogs:
            if path is None or (path, catalog.revision) in seen_path_revision:
                continue
            seen_path_revision.add((path, catalog.revision))
            kind, owner = _classify_path(path, catalog)
            if path == "pyproject.toml" and pyproject_task_eval_only:
                kind = "unit_tests"
            if (
                refined_rule is not None
                and owner is None
                and kind in {"platform", "ci_tooling", "unknown"}
            ):
                kind = "model"
                owner = refined_rule.owner
            item = {"path": path, "kind": kind}
            if owner is not None:
                item["model"] = owner
                if refined_rule is not None and owner == refined_rule.owner:
                    item["impact_rule"] = refined_rule.name
                affected.add(owner)
                unit_scope = _merge_unit_scope(unit_scope, "builder")
            elif kind == "unit_builder":
                unit_scope = _merge_unit_scope(unit_scope, "builder")
            elif kind == "unit_cli":
                unit_scope = _merge_unit_scope(unit_scope, "cli")
            elif kind == "unit_tests":
                unit_scope = _merge_unit_scope(unit_scope, "all")
            elif kind in {"platform", "ci_tooling", "unknown"}:
                broad_change = True
                unit_scope = _merge_unit_scope(unit_scope, "all")
            classifications.append(item)
        serialized_changes.append(
            {
                "status": change.status,
                "old_path": change.old_path,
                "new_path": change.new_path,
                "classifications": classifications,
            }
        )
    direct_affected = set(affected)
    if (
        affected - set(head_catalog.models)
        or affected.intersection(base_catalog.legacy_shared_runtime)
        or affected.intersection(head_catalog.legacy_shared_runtime)
    ):
        broad_change = True
    if broad_change:
        affected.intersection_update(head_catalog.models)
        direct_affected.intersection_update(head_catalog.models)
        if platform_change_policy == "all":
            affected.update(head_catalog.models)
            direct_affected.update(head_catalog.models)
            mode = "all"
        elif platform_change_policy == "fallback":
            requested_fallback = list(fallback_models)
            if not requested_fallback:
                raise ModelCIError(
                    "platform or CI/tooling fallback requires at least one --fallback-model"
                )
            if len(requested_fallback) != len(set(requested_fallback)):
                raise ModelCIError("fallback model list contains duplicates")
            invalid = [model for model in requested_fallback if not _MODEL_ID_RE.fullmatch(model)]
            if invalid:
                raise ModelCIError(f"fallback model list contains unsafe ids: {invalid}")
            missing = sorted(set(requested_fallback) - set(head_catalog.models))
            if missing:
                raise ModelCIError(f"fallback models are absent from the head catalog: {missing}")
            affected.update(requested_fallback)
            fallback_selected.update(set(requested_fallback) - direct_affected)
            mode = "fallback"
        else:
            raise ModelCIError(
                "platform or CI/tooling change requires --platform-change-policy "
                "fallback or all"
            )
    elif affected:
        mode = "models"
    elif unit_scope != "none":
        mode = "unit"
    else:
        mode = "none"
    matrix_models, _ = _scheduled_models(repo_root, head_catalog, affected)
    result = _result(
        affected,
        mode=mode,
        changes=serialized_changes,
        matrix_models=matrix_models,
        direct_models=direct_affected,
        fallback_models=fallback_selected,
        run_unit_tests=unit_scope != "none" or broad_change,
        unit_scope="all" if broad_change else unit_scope,
    )
    result["base_revision"] = base_catalog.revision
    result["head_revision"] = head_catalog.revision
    return result


def _write_github_output(path: Path, result: dict[str, object]) -> None:
    outputs = {
        "matrix": json.dumps(result["matrix"], separators=(",", ":")),
        "has_models": str(bool(result["has_models"])).lower(),
        "affected_models": json.dumps(result["affected_models"], separators=(",", ":")),
        "direct_models": json.dumps(result["direct_models"], separators=(",", ":")),
        "fallback_models": json.dumps(result["fallback_models"], separators=(",", ":")),
        "expected_count": str(result["expected_count"]),
        "mode": str(result["mode"]),
        "run_unit_tests": str(bool(result["run_unit_tests"])).lower(),
        "unit_scope": str(result["unit_scope"]),
    }
    if "expected_cases_by_model" in result:
        outputs["expected_cases_by_model"] = json.dumps(
            result["expected_cases_by_model"], separators=(",", ":"), sort_keys=True
        )
    if "expected_result_count" in result:
        outputs["expected_result_count"] = str(result["expected_result_count"])
    with path.open("a", encoding="utf-8") as stream:
        for key, value in outputs.items():
            stream.write(f"{key}={value}\n")


def _is_platform_projection_path(path: str) -> bool:
    return path in PLATFORM_PROJECTION_EXACT or any(
        path.startswith(prefix) for prefix in PLATFORM_PROJECTION_PREFIXES
    )


def _prepare_output(repo_root: Path, output_dir: Path, *, clean: bool) -> None:
    resolved_repo = repo_root.resolve()
    resolved_output = output_dir.resolve()
    if resolved_output == resolved_repo or resolved_output in resolved_repo.parents:
        raise ModelCIError("projection output must not contain the repository")
    if output_dir.exists() or output_dir.is_symlink():
        if not clean:
            raise ModelCIError(f"projection output already exists: {output_dir}")
        if output_dir.is_symlink() or output_dir.is_file():
            output_dir.unlink()
        else:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def _validate_included_symlinks(
    repo_root: Path,
    included: Sequence[TreeEntry],
) -> dict[str, str]:
    included_paths = {entry.path for entry in included}
    targets: dict[str, str] = {}
    for entry in included:
        if entry.mode != "120000":
            continue
        raw_target = _read_blob(repo_root, entry.object_id)
        try:
            target = raw_target.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ModelCIError(f"symlink target is not UTF-8: {entry.path}") from exc
        if not target or PurePosixPath(target).is_absolute():
            raise ModelCIError(f"symlink escapes projection: {entry.path} -> {target!r}")
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(entry.path), target))
        if resolved == ".." or resolved.startswith("../") or resolved.startswith("/"):
            raise ModelCIError(f"symlink escapes projection: {entry.path} -> {target!r}")
        target_is_present = resolved in included_paths or any(
            path.startswith(f"{resolved}/") for path in included_paths
        )
        if not target_is_present:
            raise ModelCIError(
                f"symlink points outside the positive allowlist: {entry.path} -> {target!r}"
            )
        targets[entry.path] = target
    return targets


def create_projection(
    repo_root: Path,
    revision: str,
    model: str,
    output_dir: Path,
    *,
    clean: bool,
) -> dict[str, object]:
    catalog = discover_catalog(repo_root, revision)
    if model not in catalog.models:
        raise ModelCIError(f"unknown model at {catalog.revision}: {model}")
    runtime_models = catalog.runtime_models.get(model, ())
    if len(runtime_models) != 1:
        raise ModelCIError(
            f"model must resolve to exactly one runtime model: {model}: {list(runtime_models)}"
        )
    runtime_model = runtime_models[0]
    e2e_families = catalog.e2e_families.get(model, ())
    if len(e2e_families) > 1:
        raise ModelCIError(
            f"model resolves to multiple E2E families: {model}: {list(e2e_families)}"
        )
    included: list[TreeEntry] = []
    model_files = 0
    platform_files = 0
    excluded_model_files = 0
    for entry in catalog.entries:
        if entry.path in MODEL_ROOT_PLATFORM_FILES:
            included.append(entry)
            platform_files += 1
            continue
        owner, under_model_root = _owner_for_path(entry.path, catalog)
        if owner is not None:
            if owner == model:
                included.append(entry)
                model_files += 1
            else:
                excluded_model_files += 1
            continue
        if under_model_root:
            # Legacy/unregistered model-root content is unavailable too.
            excluded_model_files += 1
            continue
        if _is_platform_projection_path(entry.path):
            included.append(entry)
            platform_files += 1
    if model_files == 0:
        raise ModelCIError(f"model has no owned files at {catalog.revision}: {model}")
    unsupported = [
        entry.path for entry in included if entry.mode not in {"100644", "100755", "120000"}
    ]
    if unsupported:
        raise ModelCIError(f"unsupported Git entry type in projection: {unsupported[0]}")
    symlink_targets = _validate_included_symlinks(repo_root, included)
    _prepare_output(repo_root, output_dir, clean=clean)
    manifest_entries: list[dict[str, str]] = []
    for entry in included:
        destination = output_dir / PurePosixPath(entry.path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if entry.mode == "120000":
            destination.symlink_to(symlink_targets[entry.path])
            digest = hashlib.sha256(symlink_targets[entry.path].encode("utf-8")).hexdigest()
        else:
            content = _read_blob(repo_root, entry.object_id)
            destination.write_bytes(content)
            destination.chmod(0o755 if entry.mode == "100755" else 0o644)
            digest = hashlib.sha256(content).hexdigest()
        owner, _ = _owner_for_path(entry.path, catalog)
        manifest_entries.append(
            {
                "path": entry.path,
                "mode": entry.mode,
                "blob": entry.object_id,
                "sha256": digest,
                "kind": "model" if owner == model else "platform",
            }
        )
    manifest = {
        "schema_version": 1,
        "revision": catalog.revision,
        "model": model,
        "runtime_model": runtime_model,
        "build_target": f"trtmc_model_{runtime_model}",
        "e2e_family": e2e_families[0] if e2e_families else None,
        "model_roots": list(catalog.manifests[model]),
        "model_files": model_files,
        "platform_files": platform_files,
        "excluded_model_files": excluded_model_files,
        "files": manifest_entries,
    }
    manifest_path = output_dir / ".trtmc-model-projection.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _repo_root(value: str) -> Path:
    path = Path(value).resolve()
    if not (path / ".git").exists():
        raise argparse.ArgumentTypeError(f"not a Git worktree: {path}")
    return path


def _add_common_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--github-output", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    default_root = Path(__file__).resolve().parents[1]

    validate = subparsers.add_parser("validate", help="validate MODEL.toml ownership")
    validate.add_argument("--repo-root", type=_repo_root, default=default_root)
    validate.add_argument("--revision", default="HEAD")

    impact = subparsers.add_parser("impact", help="calculate exact model impact")
    impact.add_argument("--repo-root", type=_repo_root, default=default_root)
    impact.add_argument("--base", required=True)
    impact.add_argument("--head", required=True)
    impact.add_argument(
        "--platform-change-policy",
        choices=("fallback", "all", "fail"),
        default="fail",
    )
    impact.add_argument(
        "--fallback-model",
        action="append",
        default=[],
        help="Fixed representative model to include for broad premerge impact (repeatable)",
    )
    _add_common_output(impact)

    all_models = subparsers.add_parser("all", help="emit every model as a matrix")
    all_models.add_argument("--repo-root", type=_repo_root, default=default_root)
    all_models.add_argument("--revision", default="HEAD")
    _add_common_output(all_models)

    project = subparsers.add_parser("project", help="materialize one positive source projection")
    project.add_argument("--repo-root", type=_repo_root, default=default_root)
    project.add_argument("--revision", default="HEAD")
    project.add_argument("--model", required=True)
    project.add_argument("--output-dir", type=Path, required=True)
    project.add_argument("--clean", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate":
            catalog = discover_catalog(args.repo_root, args.revision)
            result: dict[str, object] = {
                "schema_version": 1,
                "revision": catalog.revision,
                "model_count": len(catalog.models),
                "models": list(catalog.models),
            }
        elif args.command == "impact":
            result = calculate_impact(
                args.repo_root,
                args.base,
                args.head,
                platform_change_policy=args.platform_change_policy,
                fallback_models=args.fallback_model,
            )
            if args.github_output is not None:
                _write_github_output(args.github_output, result)
        elif args.command == "all":
            catalog = discover_catalog(args.repo_root, args.revision)
            matrix_models, expected_cases_by_model = _scheduled_models(
                args.repo_root,
                catalog,
                catalog.models,
                exclusive_gpu_first=True,
            )
            result = _result(
                catalog.models,
                mode="all",
                changes=[],
                matrix_models=matrix_models,
            )
            result["expected_cases_by_model"] = expected_cases_by_model
            result["expected_result_count"] = sum(
                len(cases) for cases in expected_cases_by_model.values()
            )
            result["revision"] = catalog.revision
            if args.github_output is not None:
                _write_github_output(args.github_output, result)
        elif args.command == "project":
            result = create_projection(
                args.repo_root,
                args.revision,
                args.model,
                args.output_dir,
                clean=args.clean,
            )
        else:  # pragma: no cover - argparse enforces the command set.
            raise ModelCIError(f"unsupported command: {args.command}")
    except ModelCIError as exc:
        print(f"model-ci: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
