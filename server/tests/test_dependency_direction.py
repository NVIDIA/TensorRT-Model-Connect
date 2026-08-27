# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CMAKE_ROOT = REPO_ROOT / "CMakeLists.txt"
SERVER_CMAKE = REPO_ROOT / "server" / "CMakeLists.txt"
LIBRARY_ROOTS = (
    REPO_ROOT / "include" / "trtmc",
    REPO_ROOT / "src" / "bundle",
    REPO_ROOT / "src" / "cabi",
    REPO_ROOT / "src" / "plugins",
    REPO_ROOT / "src" / "runtime",
    REPO_ROOT / "src" / "tokenizer",
    REPO_ROOT / "src" / "utils",
)
SERVER_ROOT = REPO_ROOT / "server" / "native"
SERVER_PYTHON_ROOT = REPO_ROOT / "server" / "python" / "trtmc_server"
PYTHON_ROOT = REPO_ROOT / "python" / "tensorrt_model_connect"
CPP_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".cuh", ".cxx", ".h", ".hpp"}
INCLUDE = re.compile(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]', re.MULTILINE)
PYTHON_SERVE_IMPORT = re.compile(
    r"^\s*(?:from\s+tensorrt_model_connect(?:\.serve(?:[.\s])|\s+import\s+serve\b)|"
    r"from\s+\.+serve(?:[.\s])|from\s+\.+\s+import\s+serve\b|"
    r"import\s+tensorrt_model_connect\.serve(?:[.\s])|"
    r"from\s+trtmc_server(?:[.\s])|import\s+trtmc_server(?:[.\s]))",
    re.MULTILINE,
)
LIBRARY_PYTHON_IMPORT = re.compile(
    r"^\s*(?:from\s+tensorrt_model_connect(?:[.\s])|"
    r"import\s+tensorrt_model_connect(?:[.\s]))",
    re.MULTILINE,
)
PRIVATE_LIBRARY_PREFIXES = (
    "../",
    "bundle/",
    "cabi/",
    "cli/",
    "plugins/",
    "runtime/",
    "src/",
    "tokenizer/",
    "utils/",
)


def _cpp_files(root: Path) -> list[Path]:
    assert root.is_dir(), f"dependency boundary root is missing: {root.relative_to(REPO_ROOT)}"
    files = sorted(path for path in root.rglob("*") if path.suffix in CPP_SUFFIXES)
    assert files, f"dependency boundary root has no C++ sources: {root.relative_to(REPO_ROOT)}"
    return files


def _includes(text: str) -> list[tuple[str, str]]:
    return INCLUDE.findall(text)


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
    files = sorted(path for path in PYTHON_ROOT.rglob("*.py") if "serve" not in path.parts)
    assert files, "Python Library root has no source files"
    violations = [
        str(path.relative_to(REPO_ROOT))
        for path in files
        if PYTHON_SERVE_IMPORT.search(path.read_text(encoding="utf-8", errors="strict"))
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
        r"target_link_libraries\(trtmc_server_native\s+PRIVATE\s+trtmc_core\s*\)",
        server_cmake,
    )
    assert not re.search(
        r"(?m)^\s*(?:target_sources|target_link_libraries|add_dependencies)"
        r"\s*\(\s*trtmc_core\b",
        server_cmake,
    )
    assert not re.search(
        r"(?m)^\s*(?:target_sources|target_link_libraries|add_dependencies)"
        r"\s*\(\s*trtmc_core\b",
        cmake[server_start:],
    )
    assert "install(" not in server_cmake
    install_region = cmake[cmake.index("# --- Install ---") :]
    assert "trtmc_server_native" not in install_region


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
