# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail closed if native bundle inference grows a Python interpreter bridge."""

from __future__ import annotations

from pathlib import Path
import re


REPOSITORY = Path(__file__).resolve().parents[2]
NATIVE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hpp"}
CLI_MAIN = REPOSITORY / "src" / "cli" / "main.cpp"
CLI_ARGS = REPOSITORY / "src" / "cli" / "args.cpp"

FORBIDDEN_LITERALS = (
    "hf_python",
    "--hf-python",
    "python interpreter",
    "python component",
    "python talker",
    "python tokenizer bridge",
    "python subprocess",
    "/usr/bin/python",
    "python3",
    "libpython",
    "python.h",
    "pybind11/embed.h",
    "scoped_interpreter",
    "boost::process",
    "<boost/process",
    "qprocess",
)
FORBIDDEN_CALLS = re.compile(
    r"\b(?:fork|vfork|popen|_popen|_wpopen|posix_spawn|posix_spawnp|execv|execve|"
    r"execvp|execvpe|fexecve|execveat|execl|execle|execlp|system|CreateProcessA?|"
    r"CreateProcessW|CreateProcessAsUserA?|CreateProcessAsUserW|CreateProcessWithTokenW|"
    r"ShellExecuteA?|ShellExecuteW|WinExec)\s*\(|"
    r"\bPy[A-Z_][A-Za-z0-9_]*\s*\("
)


def _native_runtime_files() -> list[Path]:
    roots = (
        REPOSITORY / "include",
        REPOSITORY / "src",
        REPOSITORY / "cmake",
        REPOSITORY / "examples",
        REPOSITORY / "python",
    )
    files = [
        path
        for root in roots
        for path in root.rglob("*")
        if path.is_file()
        and (
            path.suffix in NATIVE_SUFFIXES
            or path.name.endswith(".cpp.in")
            or (
                path.name == "MODEL.toml"
                and path.is_relative_to(REPOSITORY / "src" / "runtime" / "models")
            )
        )
        and path not in {CLI_MAIN, CLI_ARGS}
    ]
    return sorted(set(files))


def _bridge_violations(path: Path, text: str) -> list[str]:
    violations = []
    lowered = text.lower()
    for marker in FORBIDDEN_LITERALS:
        if marker in lowered:
            violations.append(f"{path.relative_to(REPOSITORY)}: forbidden {marker!r}")
    for match in FORBIDDEN_CALLS.finditer(text):
        violations.append(
            f"{path.relative_to(REPOSITORY)}: forbidden process/interpreter call {match.group(0)!r}"
        )
    return violations


def test_native_runtime_contains_no_python_interpreter_bridge() -> None:
    """The build CLI may use Python; loaded bundle execution may not."""
    violations: list[str] = []

    for path in _native_runtime_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        violations.extend(_bridge_violations(path, text))

    public_runtime_helpers = (
        REPOSITORY / "python" / "tensorrt_model_connect" / "pipeline.py",
        REPOSITORY / "src" / "cli" / "args.h",
    )
    for path in public_runtime_helpers:
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for marker in ("hf_python", "--hf-python", "python interpreter", "runtime_python"):
            if marker in text:
                violations.append(f"{path.relative_to(REPOSITORY)}: forbidden {marker!r}")

    args_text = CLI_ARGS.read_text(encoding="utf-8", errors="ignore")
    args_lowered = args_text.lower()
    for marker in FORBIDDEN_LITERALS:
        if marker != "python3" and marker in args_lowered:
            violations.append(f"{CLI_ARGS.relative_to(REPOSITORY)}: forbidden {marker!r}")
    for match in FORBIDDEN_CALLS.finditer(args_text):
        violations.append(
            f"{CLI_ARGS.relative_to(REPOSITORY)}: forbidden process/interpreter call "
            f"{match.group(0)!r}"
        )
    for match in re.finditer(
        r"--[a-z0-9-]*(?:python|interpreter)[a-z0-9-]*|\b(?:hf|runtime)_python\b",
        args_text,
        re.IGNORECASE,
    ):
        violations.append(
            f"{CLI_ARGS.relative_to(REPOSITORY)}: forbidden runtime option {match.group(0)!r}"
        )

    assert not violations, "\n".join(violations)


def test_native_source_closure_excludes_build_profiles_but_includes_loaded_artifacts() -> None:
    files = set(_native_runtime_files())

    assert REPOSITORY / "cmake" / "register_model_plugin.cpp.in" in files
    assert (
        REPOSITORY
        / "python"
        / "tensorrt_model_connect"
        / "families"
        / "sana_wm"
        / "native_plugins"
        / "sana_wm_gdn_creator.cpp"
        in files
    )
    assert REPOSITORY / "src" / "cli" / "speech_session_helpers.h" in files
    assert REPOSITORY / "src" / "runtime" / "models" / "personaplex" / "MODEL.toml" in files
    assert REPOSITORY / "python" / "tensorrt_model_connect" / "python_profiles.toml" not in files
    assert CLI_MAIN not in files
    assert CLI_ARGS not in files


def test_bridge_detector_rejects_representative_embedding_and_spawn_apis() -> None:
    synthetic_path = REPOSITORY / "include" / "trtmc" / "pipeline.h"
    forbidden_snippets = (
        'std::system("python worker.py");',
        'posix_spawnp(&pid, "python3", nullptr, nullptr, argv, envp);',
        'CreateProcessAsUserW(token, L"python.exe", command, nullptr);',
        'PyRun_SimpleString("import transformers");',
        "pybind11::scoped_interpreter guard{};",
        "#include <Python.h>",
        "boost::process::child worker(command);",
    )

    for snippet in forbidden_snippets:
        assert _bridge_violations(synthetic_path, snippet), snippet


def test_cli_interpreter_bridge_is_build_only() -> None:
    """The shared executable may invoke Python only for build and graph."""
    text = CLI_MAIN.read_text(encoding="utf-8")
    dispatch = re.compile(
        r'if\s*\(args\.command\s*==\s*"build"\s*\|\|\s*'
        r'args\.command\s*==\s*"graph"\)\s*return\s+cmd_python\(args\);'
    )

    assert len(dispatch.findall(text)) == 1
    expected_references = {
        "build_python_executable": 2,
        "build_pythonpath": 2,
        "run_python_module": 2,
        "cmd_python": 2,
    }
    for name, expected in expected_references.items():
        assert len(re.findall(rf"\b{name}\b", text)) == expected, (
            f"{name} must stay confined to its definition, the build bridge, and build/graph dispatch"
        )

    bridge_start = text.index("std::string build_python_executable()")
    bridge_end = text.index("int cmd_version()", bridge_start)
    bridge_text = text[bridge_start:bridge_end]
    top_level_functions = re.findall(
        r"^(?:[A-Za-z_][A-Za-z0-9_:<>]*[ \t*&]+)+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        bridge_text,
        re.MULTILINE,
    )
    assert top_level_functions == [
        "build_python_executable",
        "build_pythonpath",
        "run_python_module",
        "cmd_python",
    ], "the CLI Python exemption must contain only the four build/graph bridge functions"
    runtime_text = text[:bridge_start] + text[bridge_end:]
    violations = _bridge_violations(CLI_MAIN, runtime_text)
    assert not violations, "\n".join(violations)


def test_native_build_configuration_contains_no_embedded_python_link() -> None:
    """Catch source-level embedded-Python links before ELF dependency auditing."""
    cmake_files = {
        REPOSITORY / "CMakeLists.txt",
        *REPOSITORY.glob("cmake/**/*.cmake"),
        *REPOSITORY.glob("cmake/**/*.cmake.in"),
        *REPOSITORY.glob("src/**/CMakeLists.txt"),
        *REPOSITORY.glob("python/**/CMakeLists.txt"),
        *REPOSITORY.glob("examples/**/CMakeLists.txt"),
    }
    forbidden_link_markers = re.compile(
        r"(?:libpython|(?:Python|Python3)::Python|pybind11::embed|"
        r"(?:Python|Python3)_LIBRARIES|Development\.Embed|-lpython)",
        re.IGNORECASE,
    )
    violations = []
    for path in sorted(cmake_files):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in forbidden_link_markers.finditer(text):
            violations.append(
                f"{path.relative_to(REPOSITORY)}: forbidden embedded-Python link marker "
                f"{match.group(0)!r}"
            )

    assert not violations, "\n".join(violations)
