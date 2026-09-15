# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CMAKE_ROOT = REPO_ROOT / "CMakeLists.txt"
SERVER_CMAKE = REPO_ROOT / "server" / "CMakeLists.txt"
SERVER_ENTRYPOINT = REPO_ROOT / "server" / "native" / "entrypoint.cpp"
CLI_MAIN = REPO_ROOT / "apps" / "cli" / "main.cpp"
LIBRARY_ROOTS = (
    REPO_ROOT / "core" / "runtime",
    REPO_ROOT / "families",
)
SERVER_ROOT = REPO_ROOT / "server" / "native"
SERVER_PYTHON_ROOT = REPO_ROOT / "server" / "python" / "trtmc_server"
PYTHON_LIBRARY_ROOTS = (
    REPO_ROOT / "core",
    REPO_ROOT / "families",
)
CPP_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".cuh", ".cxx", ".h", ".hpp"}
INCLUDE = re.compile(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]', re.MULTILINE)
LIBRARY_PYTHON_IMPORT = re.compile(
    r"^\s*(?:from\s+tensorrt_model_connect(?:[.\s])|"
    r"import\s+tensorrt_model_connect(?:[.\s]))",
    re.MULTILINE,
)
PRIVATE_LIBRARY_PREFIXES = (
    "../",
    "apps/",
    "core/",
    "families/",
)


def _cpp_files(root: Path) -> list[Path]:
    assert root.is_dir(), f"dependency boundary root is missing: {root.relative_to(REPO_ROOT)}"
    files = sorted(path for path in root.rglob("*") if path.suffix in CPP_SUFFIXES)
    assert files, f"dependency boundary root has no C++ sources: {root.relative_to(REPO_ROOT)}"
    return files


def _includes(text: str) -> list[tuple[str, str]]:
    return INCLUDE.findall(text)


def _server_import_lines(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8", errors="strict"), filename=str(path))
    violations: list[int] = []
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            if node.module:
                modules.append(node.module)
            if node.module == "tensorrt_model_connect" and any(
                alias.name == "serve" for alias in node.names
            ):
                modules.append("tensorrt_model_connect.serve")
        if any(
            module == "trtmc_server"
            or module.startswith("trtmc_server.")
            or module == "tensorrt_model_connect.serve"
            or module.startswith("tensorrt_model_connect.serve.")
            for module in modules
        ):
            violations.append(node.lineno)
    return violations


def test_library_roots_do_not_reference_server_headers_or_target() -> None:
    violations: list[str] = []
    for root in LIBRARY_ROOTS:
        for path in _cpp_files(root):
            relative = path.relative_to(REPO_ROOT)
            contents = path.read_text(encoding="utf-8", errors="strict")
            for _, include in _includes(contents):
                if include.startswith("native/"):
                    violations.append(f'{relative}: includes private Server header "{include}"')
            for reference in (
                "trtmc::serve",
                "trtmc::server",
                "trtmc_server_native",
                "server/native/",
            ):
                if reference in contents:
                    violations.append(f'{relative}: references Server boundary "{reference}"')

    assert violations == [], "\n".join(violations)


def test_python_library_does_not_import_optional_server() -> None:
    files = sorted(path for root in PYTHON_LIBRARY_ROOTS for path in root.rglob("*.py"))
    assert files, "Python Library roots have no source files"
    violations = [
        f"{path.relative_to(REPO_ROOT)}:{line}"
        for path in files
        for line in _server_import_lines(path)
    ]
    assert violations == [], "Python Library imports optional Server: " + ", ".join(violations)


def test_python_server_does_not_import_library_implementation() -> None:
    files = sorted(SERVER_PYTHON_ROOT.rglob("*.py"))
    assert files, "Python Server root has no source files"
    violations = [
        str(path.relative_to(REPO_ROOT))
        for path in files
        if LIBRARY_PYTHON_IMPORT.search(path.read_text(encoding="utf-8", errors="strict"))
    ]
    assert violations == [], "Python Server imports Library implementation: " + ", ".join(
        violations
    )


def test_cmake_keeps_server_downstream_of_core() -> None:
    cmake = CMAKE_ROOT.read_text(encoding="utf-8")
    server_cmake = SERVER_CMAKE.read_text(encoding="utf-8")
    core_start = cmake.index("add_library(trtmc_core SHARED")
    server_start = cmake.index("add_subdirectory(server)")
    core_region = cmake[core_start:server_start]
    assert "target_link_libraries(trtmc_core" in core_region
    assert "server/native/" not in core_region
    assert "trtmc_server_native" not in core_region
    assert "add_library(trtmc_server_native STATIC" in server_cmake
    assert re.search(
        r"target_link_libraries\(trtmc_server_native\s+PRIVATE\s+"
        r"trtmc_runtime\b",
        server_cmake,
    )
    for library_target in ("trtmc_core", "trtmc_runtime"):
        reverse_dependency = (
            r"(?m)^\s*(?:target_sources|target_link_libraries|add_dependencies)"
            rf"\s*\(\s*{library_target}\b"
        )
        assert not re.search(reverse_dependency, server_cmake)
        assert not re.search(reverse_dependency, cmake[server_start:])
    assert "install(" not in server_cmake
    assert not re.search(
        r"install\s*\([^)]*\btrtmc_server_native\b",
        cmake + "\n" + server_cmake,
        re.DOTALL,
    )


def test_server_uses_only_local_or_public_library_headers() -> None:
    violations: list[str] = []
    for path in _cpp_files(SERVER_ROOT):
        relative = path.relative_to(REPO_ROOT)
        contents = path.read_text(encoding="utf-8", errors="strict")
        for delimiter, include in _includes(contents):
            if include.startswith("native/") or include.startswith("trtmc/"):
                continue
            if delimiter == '"' or include.startswith(PRIVATE_LIBRARY_PREFIXES):
                violations.append(
                    f'{relative}: Server must not include private Library header "{include}"'
                )

    assert violations == [], "\n".join(violations)


def test_source_build_python_copy_removes_stale_modules_first() -> None:
    server_cmake = SERVER_CMAKE.read_text(encoding="utf-8")
    copy_target = server_cmake.split("add_custom_target(trtmc_server_python ALL", 1)[1].split(
        "\n)", 1
    )[0]
    generated_module = '"${PROJECT_BINARY_DIR}/server/python/trtmc_server"'
    assert copy_target.count(generated_module) == 2
    assert copy_target.index("-E remove_directory") < copy_target.index("-E copy_directory")


def test_server_frontend_isolates_python_module_search() -> None:
    source = SERVER_ENTRYPOINT.read_text(encoding="utf-8")
    assert 'source_mode ? "-P" : "-I"' in source
    assert 'std::string pythonpath = server_python.string();' in source
    assert 'pythonpath += ":" + std::string(existing);' in source

    cmake = SERVER_CMAKE.read_text(encoding="utf-8")
    shadow_test = cmake.split("add_test(\n    NAME serve_cli_ignores_cwd_shadow", 1)[1]
    assert "TRTMC_CWD_SHADOW_EXECUTED" in shadow_test
    assert "PASS_REGULAR_EXPRESSION" in shadow_test
    assert "FAIL_REGULAR_EXPRESSION" in shadow_test


def test_cli_main_keeps_only_thin_server_dispatch() -> None:
    source = CLI_MAIN.read_text(encoding="utf-8")
    assert '#include "native/entrypoint.h"' in source
    assert source.count("trtmc::server::run_server_frontend") == 1
    assert source.count("trtmc::server::run_native_worker") == 1
    for implementation_detail in (
        "execvp",
        "PYTHONPATH",
        "--runtime-root",
        "--kv-cache-size",
        "load_task",
    ):
        assert implementation_detail not in source
