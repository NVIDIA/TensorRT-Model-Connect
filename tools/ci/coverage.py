# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select unit tests, collect coverage, and enforce the reviewed numeric gates.

Boundary: coverage policy and artifacts live here; model E2E validation does not.
"""

from __future__ import annotations

import re
import shlex
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from .context import CiContext
from .process import CiError, CommandRunner


class CppCoverageEngine:
    """Build C++ coverage targets and emit the gcovr reports and gate log."""

    def __init__(self, context: CiContext, report_root: Path):
        self.context = context
        self.repository = context.repository
        self.report_root = report_root

    def run(
        self,
        ctest_args: list[str],
        *,
        build_target: str,
        limit: str | None,
    ) -> int:
        if build_target not in {"trtmc_cpp_tests", "trtmc_platform_cpp_tests"}:
            raise CiError(
                "CPP_COVERAGE_BUILD_TARGET must be trtmc_cpp_tests or trtmc_platform_cpp_tests"
            )
        for tool in ("cmake", "ctest", "gcovr"):
            self.context.executable(tool)
        if "--fail-under-function" not in self.context.output(["gcovr", "--help"]):
            raise CiError(
                "Installed gcovr does not support --fail-under-function; install a newer gcovr"
            )

        build_dir = self._path("BUILD_DIR", self.repository / "build-cov")
        self.report_root.mkdir(parents=True, exist_ok=True)
        build_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "xml": self._path("COBERTURA_XML", self.report_root / "cpp-cobertura.xml"),
            "html": self._path("HTML_REPORT", self.report_root / "cpp-coverage.html"),
            "summary": self._path("SUMMARY_TXT", self.report_root / "cpp-coverage-summary.txt"),
            "gate": self._path("GATE_LOG", self.report_root / "cpp-gate.log"),
        }
        filters = self._words("GCOVR_FILTERS") or [
            str(self.repository / "src"),
            str(self.repository / "include"),
        ]
        excludes = self._words("GCOVR_EXCLUDES") or [
            str(self.repository / "tests"),
            str(self.repository / "build.*"),
            str(self.repository / "python/tensorrt_model_connect/models"),
            ".*/CMakeFiles/.*/CompilerIdCXX/.*",
        ]
        thresholds = {
            "line": self._value("CPP_COVERAGE_MIN_LINE", "100"),
            "function": self._value("CPP_COVERAGE_MIN_FUNCTION", "100"),
            "branch": self._value("CPP_COVERAGE_MIN_BRANCH", "100"),
        }

        print(f"[cpp-coverage] Repo root: {self.repository}")
        print(f"[cpp-coverage] Build dir: {build_dir}")
        print(f"[cpp-coverage] Report root: {self.report_root}")
        print(f"[cpp-coverage] gcovr filters: {' '.join(filters)}")
        print(f"[cpp-coverage] gcovr excludes: {' '.join(excludes)}")
        print(
            "[cpp-coverage] Gate thresholds: "
            f"line>={thresholds['line']}% function>={thresholds['function']}% "
            f"branch>={thresholds['branch']}%"
        )

        self._run(self._cmake_configure(build_dir), limit=limit)
        parallel = ["--parallel"]
        if value := self.context.env.get("BUILD_PARALLEL", ""):
            parallel.append(value)
        if build_target == "trtmc_cpp_tests":
            self._run(["cmake", "--build", build_dir, *parallel], limit=limit)
        self._run(
            ["cmake", "--build", build_dir, "--target", build_target, *parallel],
            limit=limit,
        )
        for gcda in build_dir.rglob("*.gcda"):
            try:
                gcda.unlink()
            except OSError:
                pass
        self._run(
            ["ctest", "--test-dir", build_dir, "--output-on-failure", *ctest_args],
            limit=limit,
        )

        gcovr_base: list[str | Path] = [
            "--root",
            self.repository,
            "--object-directory",
            build_dir,
            "--gcov-ignore-errors",
            "source_not_found",
            "--gcov-ignore-errors",
            "no_working_dir_found",
            # GCC can emit invalid negative branch counters (GCC PR 68080).
            # Keep the report and numeric gates intact while surfacing one
            # warning for each affected source file.
            "--gcov-ignore-parse-errors",
            "negative_hits.warn_once_per_file",
        ]
        for value in filters:
            gcovr_base.extend(("--filter", value))
        for value in excludes:
            gcovr_base.extend(("--exclude", value))

        self._gcovr(
            build_dir,
            [*gcovr_base, "--xml", paths["xml"], "--xml-pretty", "--html-details", paths["html"]],
            limit=limit,
        )
        summary = self._gcovr(
            build_dir,
            [*gcovr_base, "--txt-summary"],
            limit=limit,
            capture_output=True,
        )
        paths["summary"].write_text(summary.stdout, encoding="utf-8")
        print(summary.stdout, end="")
        gate = self._gcovr(
            build_dir,
            [
                *gcovr_base,
                "--print-summary",
                "--fail-under-line",
                thresholds["line"],
                "--fail-under-function",
                thresholds["function"],
                "--fail-under-branch",
                thresholds["branch"],
            ],
            limit=limit,
            check=False,
            capture_output=True,
        )
        gate_text = gate.stdout + gate.stderr
        paths["gate"].write_text(gate_text, encoding="utf-8")
        print(gate_text, end="", file=sys.stderr if gate.returncode else sys.stdout)
        if not gate.returncode:
            print(
                "PASS: C++ coverage gates satisfied (line/function/branch >= configured thresholds)."
            )
            print(f"[cpp-coverage] Cobertura XML: {paths['xml']}")
            print(f"[cpp-coverage] HTML report : {paths['html']}")
            print(f"[cpp-coverage] Text summary: {paths['summary']}")
        return gate.returncode

    def _cmake_configure(self, build_dir: Path) -> list[str | Path]:
        env = self.context.env
        arguments: list[str | Path] = [
            "-S",
            self.repository,
            "-B",
            build_dir,
            f"-DCMAKE_BUILD_TYPE={self._value('CMAKE_BUILD_TYPE', 'Coverage')}",
            f"-DCMAKE_C_FLAGS={self._value('COVERAGE_COMPILE_FLAGS', '--coverage -O0 -g0')}",
            f"-DCMAKE_CXX_FLAGS={self._value('COVERAGE_COMPILE_FLAGS', '--coverage -O0 -g0')}",
            f"-DCMAKE_EXE_LINKER_FLAGS={self._value('COVERAGE_LINK_FLAGS', '--coverage')}",
            f"-DCMAKE_SHARED_LINKER_FLAGS={self._value('COVERAGE_LINK_FLAGS', '--coverage')}",
            "-DTRTMC_ENABLE_LIBTORCH_MULTINOMIAL="
            f"{self._value('TRTMC_ENABLE_LIBTORCH_MULTINOMIAL', 'OFF')}",
        ]
        if generator := self._value("CMAKE_GENERATOR", "Ninja"):
            arguments = ["-G", generator, *arguments]
        trt_include = env.get("TRT_INC_DIR", "")
        trt_library = env.get("TRT_LIB_DIR", "")
        if trt_include and trt_library:
            arguments.extend(
                (
                    f"-DTRTMC_TRT_INCLUDE_DIR={trt_include}",
                    f"-DTRTMC_TRT_LIBRARY={trt_library}/libnvinfer.so",
                    f"-DTRTMC_CUDA_INCLUDE_DIR={self._value('CUDA_INC_DIR', '/usr/local/cuda/include')}",
                    "-DTRTMC_CUDART_LIBRARY="
                    f"{self._value('CUDART_LIBRARY', '/usr/local/cuda/lib64/libcudart.so')}",
                )
            )
        arguments.extend(self._words("CMAKE_EXTRA_ARGS"))
        return ["cmake", *arguments]

    def _words(self, name: str) -> list[str]:
        return shlex.split(self.context.env.get(name, ""))

    def _value(self, name: str, default: str) -> str:
        return self.context.env.get(name) or default

    def _path(self, name: str, default: Path) -> Path:
        path = Path(self.context.env.get(name) or default)
        return path if path.is_absolute() else self.repository / path

    def _run(self, command: list[str | Path], *, limit: str | None) -> None:
        self.context.run(command, limit=limit, updates={"LC_ALL": "C"})

    def _gcovr(
        self,
        build_dir: Path,
        arguments: list[str | Path],
        *,
        limit: str | None,
        check: bool = True,
        capture_output: bool = False,
    ):
        command = ["gcovr", *(str(item) for item in arguments), str(build_dir)]
        if limit:
            command = ["timeout", "--kill-after=2m", limit, *command]
        environment = {**self.context.env, "LC_ALL": "C"}
        return CommandRunner(cwd=build_dir, env=environment).run(
            command,
            check=check,
            capture_output=capture_output,
        )


class PythonCoverageEngine:
    """Run Python tests under coverage.py and enforce line and branch gates."""

    def __init__(self, context: CiContext, report_root: Path):
        self.context = context
        self.repository = context.repository
        self.report_root = report_root

    def run(self, pytest_args: list[str]) -> None:
        python = self.context.env.get("PYTHON_BIN") or "python3"
        self.context.executable(python)
        updates = self._environment()
        for module in ("coverage", "pytest"):
            result = self.context.run(
                [python, "-m", module, "--version"],
                updates=updates,
                check=False,
                capture_output=True,
            )
            if result.returncode:
                raise CiError(f"{module} is required for Python coverage")

        self.report_root.mkdir(parents=True, exist_ok=True)
        html = self._path("HTML_DIR", self.report_root / "python-html")
        html.mkdir(parents=True, exist_ok=True)
        xml = self._path("COBERTURA_XML", self.report_root / "python-cobertura.xml")
        summary = self._path("SUMMARY_TXT", self.report_root / "python-coverage.txt")
        targets = shlex.split(self.context.env.get("PYTHON_TEST_TARGETS", "")) or [
            "tests/builder",
            "tests/tools",
        ]
        line_min = float(self.context.env.get("PYTHON_COVERAGE_MIN_LINE") or "100")
        branch_min = float(self.context.env.get("PYTHON_COVERAGE_MIN_BRANCH") or "100")

        print(f"[python-coverage] Repo root: {self.repository}")
        print(f"[python-coverage] Report root: {self.report_root}")
        print(f"[python-coverage] Test targets: {' '.join(targets)}")
        print(f"[python-coverage] Gate thresholds: line>={line_min:g}% branch>={branch_min:g}%")
        self.context.run([python, "-m", "coverage", "erase"], updates=updates)
        self.context.run(
            [
                python,
                "-m",
                "coverage",
                "run",
                "--branch",
                "-m",
                "pytest",
                *targets,
                *pytest_args,
            ],
            updates=updates,
        )
        report = self.context.run(
            [python, "-m", "coverage", "report", "--show-missing"],
            updates=updates,
            capture_output=True,
        ).stdout
        summary.write_text(report, encoding="utf-8")
        print(report, end="")
        self.context.run([python, "-m", "coverage", "xml", "-o", xml], updates=updates)
        self.context.run([python, "-m", "coverage", "html", "-d", html], updates=updates)

        root = ET.parse(xml).getroot()
        line = float(root.attrib.get("line-rate", "0")) * 100
        branch = float(root.attrib.get("branch-rate", "0")) * 100
        print(f"PYTHON_COVERAGE_LINE={line:.2f}%")
        print(f"PYTHON_COVERAGE_BRANCH={branch:.2f}%")
        failures = []
        if line + 1e-9 < line_min:
            failures.append(f"line coverage {line:.2f}% < {line_min:.2f}%")
        if branch + 1e-9 < branch_min:
            failures.append(f"branch coverage {branch:.2f}% < {branch_min:.2f}%")
        if failures:
            raise CiError("Python coverage gate failed: " + "; ".join(failures))
        print(
            "PASS: Python coverage gates satisfied "
            f"(line={line:.2f}% >= {line_min:.2f}%, "
            f"branch={branch:.2f}% >= {branch_min:.2f}%)."
        )
        print(f"[python-coverage] Cobertura XML: {xml}")
        print(f"[python-coverage] HTML report : {html / 'index.html'}")
        print(f"[python-coverage] Text summary: {summary}")

    def _environment(self) -> dict[str, str]:
        existing = self.context.env.get("PYTHONPATH", "")
        pythonpath = str(self.repository)
        if existing:
            pythonpath += f":{existing}"
        return {
            "PYTHONHASHSEED": self.context.env.get("PYTHONHASHSEED", "0"),
            "LC_ALL": "C",
            "PYTHONPATH": pythonpath,
            "COVERAGE_FILE": str(self._path("COVERAGE_FILE", self.report_root / ".coverage")),
        }

    def _path(self, name: str, default: Path) -> Path:
        path = Path(self.context.env.get(name) or default)
        return path if path.is_absolute() else self.repository / path


class CoverageRunner:
    """Run coverage commands and enforce the existing numeric gates."""

    def __init__(self, context: CiContext):
        self.context = context
        self.directory = self._report_directory("REPORT_ROOT", context.repository / "coverage")

    def python_builder_tests(self) -> None:
        self.context.run(
            [
                "python",
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--quiet",
                "pytest-cov>=6.0",
            ]
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        selected = self._selected_python_tests()
        coverage_required = not selected
        coverage_args: list[str] = []
        if coverage_required:
            config = self._write_python_config()
            coverage_args = [
                "--cov=tensorrt_model_connect",
                "--cov-branch",
                f"--cov-config={config}",
                "--cov-report=term-missing",
                "--cov-report=xml:coverage/python-cobertura.xml",
            ]
        else:
            print(
                "Skipping Python package coverage gate: selected Python subset does not "
                "produce global package coverage"
            )
        environment = {"TRTMC_TEST_INSTALLED_WHEEL": "1"}
        limit = self.context.env.get("PYTHON_BUILDER_TIMEOUT", "40m")
        if selected:
            selected_file = self.directory / "python-selected-tests.txt"
            selected_file.write_text("\n".join(selected) + "\n", encoding="utf-8")
            print("Selective Python tests:")
            for test in selected:
                print(f"  {test}")
            self.context.run(
                ["python", "-m", "pytest", *selected, "-v", "-n", "auto", *coverage_args],
                limit=limit,
                updates=environment,
            )
        else:
            harness_tests = [
                str(path.relative_to(self.context.repository))
                for path in sorted(
                    (self.context.repository / "tests/e2e_harness").glob("test_*.py")
                )
            ]
            self.context.run(
                [
                    "python",
                    "-m",
                    "pytest",
                    "tests/builder/",
                    "tests/tools/",
                    *harness_tests,
                    "-v",
                    "-n",
                    "auto",
                    "-m",
                    "not model_proof_allocator and not gpu and not trt",
                    *coverage_args,
                ],
                limit=limit,
                updates=environment,
            )
            self.context.run(
                [
                    "python",
                    "-m",
                    "pytest",
                    "tests/tools/test_model_proof_runner.py",
                    "-v",
                    "-p",
                    "no:cacheprovider",
                    "-m",
                    "model_proof_allocator",
                    *coverage_args,
                    "--cov-append",
                ],
                limit=limit,
                updates=environment,
            )
        if not coverage_required:
            self._write_skipped_python_coverage(
                "Skipped: selected Python subset does not produce global package coverage"
            )
            return
        report = self.context.run(
            ["python", "-m", "coverage", "report", "--show-missing"],
            check=False,
            capture_output=True,
        ).stdout
        (self.directory / "python-coverage.txt").write_text(report, encoding="utf-8")
        print(report, end="")
        self._enforce_python_gate()

    def cpp(self) -> None:
        event = self.context.env.get("GITHUB_EVENT_NAME", "")
        if event == "pull_request":
            changed = self.context.output(
                [
                    "git",
                    "diff",
                    "--diff-filter=d",
                    "--name-only",
                    f"{self.context.env['CI_BASE_REF']}...HEAD",
                    "--",
                    "src/**/*.cpp",
                    "src/**/*.h",
                    "include/**/*.h",
                    "tests/cpp/**/*.cpp",
                    "tests/cpp/**/*.h",
                    "CMakeLists.txt",
                ],
                check=False,
            )
            if not changed:
                print("Skipping: no C++ source, C++ tests, or CMake changes in premerge diff")
                return
            print("C++ coverage triggered by changed files:")
            print(changed)
        elif event not in {"schedule", "workflow_dispatch"}:
            print(
                "Skipping: C++ coverage only runs for nightly/manual pipelines and C++-affected premerge PRs"
            )
            return
        scope = self.context.env.get("CPP_COVERAGE_SCOPE", "all")
        if scope == "all":
            build_target, ctest_args = "trtmc_cpp_tests", []
        elif scope == "platform":
            build_target, ctest_args = "trtmc_platform_cpp_tests", ["-L", "platform"]
        else:
            raise CiError("CPP_COVERAGE_SCOPE must be all or platform")
        self.context.run(
            [
                "python",
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--quiet",
                "gcovr==8.2",
            ]
        )
        self.cpp_report(
            ctest_args,
            build_target=build_target,
            limit=self.context.env.get("CPP_COVERAGE_TIMEOUT", "40m"),
        )

    def cpp_report(
        self, ctest_args: list[str], *, build_target: str | None = None, limit: str | None = None
    ) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        target = (
            build_target
            or self.context.env.get("CPP_COVERAGE_BUILD_TARGET", "trtmc_cpp_tests")
            or "trtmc_cpp_tests"
        )
        returncode = CppCoverageEngine(self.context, self.directory).run(
            ctest_args,
            build_target=target,
            limit=limit,
        )
        summary = self.directory / "cpp-coverage-summary.txt"
        if not summary.is_file():
            raise CiError(f"C++ coverage summary is missing at {summary}")
        percentages: dict[str, str] = {}
        for line in summary.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^(lines|functions|branches):\s*([0-9]+(?:\.[0-9]+)?)%", line)
            if match:
                percentages[match.group(1)] = match.group(2)
        if set(percentages) != {"lines", "functions", "branches"}:
            raise CiError(f"Failed to parse gcovr summary percentages from {summary}")
        print(f"CPP_COVERAGE_LINE={percentages['lines']}%")
        print(f"CPP_COVERAGE_FUNCTION={percentages['functions']}%")
        print(f"CPP_COVERAGE_BRANCH={percentages['branches']}%")
        if returncode:
            raise CiError(f"C++ coverage gate failed with code {returncode}")

    def python_report(self, pytest_args: list[str] | None = None) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        PythonCoverageEngine(self.context, self.directory).run(
            pytest_args or ["-v", "--ignore=tests/builder/test_cli.py"]
        )

    def all_reports(self) -> None:
        """Run the two standalone local gates in the same order as the retired wrapper."""
        python_args = shlex.split(self.context.env.get("PYTHON_ARGS", ""))
        cpp_args = shlex.split(self.context.env.get("CPP_CTEST_ARGS", ""))
        combined_root = self.directory
        try:
            print("[coverage-all] Running Python coverage gate...")
            self.directory = self._report_directory("PYTHON_REPORT_ROOT", combined_root / "python")
            self.python_report(python_args or ["-v", "--ignore=tests/builder/test_cli.py"])
            print("[coverage-all] Running C++ coverage gates...")
            self.directory = self._report_directory("CPP_REPORT_ROOT", combined_root / "cpp")
            self.cpp_report(cpp_args)
        finally:
            self.directory = combined_root
        print(f"[coverage-all] Completed. Combined report root: {combined_root}")

    def _report_directory(self, name: str, default: Path) -> Path:
        path = Path(self.context.env.get(name) or default)
        return path if path.is_absolute() else self.context.repository / path

    def map(self) -> None:
        if self.context.env.get("RUN_COVERAGE_MAP", "false") != "true":
            print("Skipping: coverage map generation was not requested")
            return
        self.context.run(
            [
                "python",
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--quiet",
                "coverage[toml]==7.6.10",
                "pytest-cov>=6.0",
                "gcovr==8.2",
            ]
        )
        build = self.context.env.get(
            "CPP_COVERAGE_BUILD_DIR", str(self.context.repository / "build-cov")
        )
        self.context.run(
            [
                "python",
                "-m",
                "tools.coverage_map.generate",
                "--output",
                "coverage_map.json",
                "--python-bin",
                "python",
                "--build-dir",
                build,
            ],
            limit=self.context.env.get("COVERAGE_MAP_TIMEOUT", "90m"),
        )
        self.context.run(
            ["python", "-m", "tools.coverage_map.generate", "--validate", "coverage_map.json"]
        )
        data = self.context.read_json("coverage_map.json")
        metadata = data["meta"]
        print(
            f"Python tests: {metadata['python_tests']}, C++ tests: {metadata['cpp_tests']}, "
            f"Source files: {len(data['source_to_tests'])}"
        )

    def _selected_python_tests(self) -> list[str]:
        if self.context.env.get("FULL_E2E", "false") == "true":
            return []
        impact = self.context.read_json("impact.json")
        fallback = set(impact.get("fallback_tiers", []))
        if {"builder", "tools"}.issubset(fallback):
            return []
        selected: list[str] = []

        def add(values) -> None:
            for value in values:
                text = str(value)
                if text and text not in selected:
                    selected.append(text)

        add(["tests/builder/"] if "builder" in fallback else impact.get("builder_tests", []))
        if "tools" in fallback:
            add(["tests/tools/"])
            add(
                str(path.relative_to(self.context.repository))
                for path in sorted(
                    (self.context.repository / "tests/e2e_harness").glob("test_*.py")
                )
            )
        else:
            add(impact.get("tools_tests", []))
        add(
            [
                "tests/tools/test_github_actions_ci.py",
                "tests/tools/test_model_plugin_encapsulation_static.py",
                "tests/tools/test_schedule_e2e.py",
                "tests/tools/test_test_impact.py",
            ]
        )
        return selected

    def _write_python_config(self) -> Path:
        path = self.directory / "python-package-gate.coveragerc"
        path.write_text(
            """[run]
source_pkgs =
    tensorrt_model_connect
branch = True
omit =
    */tests/*
    */__pycache__/*
    */tensorrt_model_connect/models/*

[report]
show_missing = True
precision = 1
exclude_lines =
    pragma: no cover
    if __name__ == .__main__.
    raise NotImplementedError
""",
            encoding="utf-8",
        )
        return path

    def _write_skipped_python_coverage(self, reason: str) -> None:
        (self.directory / "python-cobertura.xml").write_text(
            '<?xml version="1.0" ?><coverage version="7.6" timestamp="0" lines-valid="0" '
            'lines-covered="0" line-rate="1.0" branches-valid="0" branches-covered="0" '
            'branch-rate="1.0" complexity="0"><packages/></coverage>\n',
            encoding="utf-8",
        )
        (self.directory / "python-coverage.txt").write_text(reason + "\n", encoding="utf-8")
        print("PYTHON_COVERAGE_LINE=100.00%")
        print("PYTHON_COVERAGE_BRANCH=100.00%")

    def _enforce_python_gate(self) -> None:
        root = ET.parse(self.directory / "python-cobertura.xml").getroot()
        line = float(root.attrib.get("line-rate", "0")) * 100
        branch = float(root.attrib.get("branch-rate", "0")) * 100
        line_min = float(self.context.env["PYTHON_COVERAGE_MIN_LINE"])
        branch_min = float(self.context.env["PYTHON_COVERAGE_MIN_BRANCH"])
        print(f"PYTHON_COVERAGE_LINE={line:.2f}%")
        print(f"PYTHON_COVERAGE_BRANCH={branch:.2f}%")
        failures = []
        if line + 1e-9 < line_min:
            failures.append(f"Python line coverage {line:.1f}% < {line_min}% gate")
        if branch + 1e-9 < branch_min:
            failures.append(f"Python branch coverage {branch:.1f}% < {branch_min}% gate")
        if failures:
            raise CiError("; ".join(failures))
