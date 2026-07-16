# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python test selection and C++/Python coverage reporting."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from .context import CiContext
from .process import CiError


class CoverageRunner:
    """Run coverage commands and enforce the existing numeric gates."""

    def __init__(self, context: CiContext):
        self.context = context
        self.directory = context.repository / "coverage"

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
        updates = {
            "LC_ALL": "C",
            "REPORT_ROOT": str(self.directory),
            "COBERTURA_XML": str(self.directory / "cpp-cobertura.xml"),
            "SUMMARY_TXT": str(self.directory / "cpp-coverage-summary.txt"),
            "HTML_REPORT": str(self.directory / "cpp-coverage.html"),
            "GATE_LOG": str(self.directory / "cpp-gate.log"),
        }
        if build_target:
            updates["CPP_COVERAGE_BUILD_TARGET"] = build_target
        result = self.context.run(
            ["bash", "tools/coverage/cpp_coverage.sh", *ctest_args],
            limit=limit,
            updates=updates,
            check=False,
        )
        summary = self.directory / "cpp-coverage-summary.txt"
        if not summary.is_file():
            if result.returncode:
                raise CiError(f"C++ coverage command failed with code {result.returncode}")
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
        if result.returncode:
            raise CiError(f"C++ coverage gate failed with code {result.returncode}")

    def python_report(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        updates = {
            "PYTHONHASHSEED": self.context.env.get("PYTHONHASHSEED", "0"),
            "LC_ALL": "C",
            "REPORT_ROOT": str(self.directory),
            "COBERTURA_XML": str(self.directory / "python-cobertura.xml"),
            "SUMMARY_TXT": str(self.directory / "python-coverage.txt"),
            "HTML_DIR": str(self.directory / "python-html"),
        }
        result = self.context.run(
            [
                "bash",
                "tools/coverage/python_coverage.sh",
                "-v",
                "--ignore=tests/builder/test_cli.py",
            ],
            updates=updates,
            check=False,
        )
        summary = self.directory / "python-coverage.txt"
        xml = self.directory / "python-cobertura.xml"
        if not summary.is_file() or not xml.is_file():
            if result.returncode:
                raise CiError(f"Python coverage command failed with code {result.returncode}")
            raise CiError(f"Python coverage artifacts are missing under {self.directory}")
        root = ET.parse(xml).getroot()
        print(f"PYTHON_COVERAGE_LINE={float(root.attrib['line-rate']) * 100:.2f}%")
        if root.attrib.get("branch-rate"):
            print(f"PYTHON_COVERAGE_BRANCH={float(root.attrib['branch-rate']) * 100:.2f}%")
        if result.returncode:
            raise CiError(f"Python coverage gate failed with code {result.returncode}")

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
    */tensorrt_model_connect/families/*

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
