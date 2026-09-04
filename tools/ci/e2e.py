# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run family-owned E2E tests directly, without a central harness."""

from __future__ import annotations

import json
import re
import tempfile
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from pathlib import Path

from tools import test_impact

from .context import CiContext
from .process import CiError


def _require_passing_junit(path: Path, label: str, *, allow_empty: bool = False) -> int:
    if not path.is_file():
        raise CiError(f"{label} report is missing")
    try:
        testcases = ET.parse(path).getroot().findall(".//testcase")
    except (OSError, ET.ParseError) as error:
        raise CiError(f"{label} report is invalid: {error}") from error
    if not testcases and not allow_empty:
        raise CiError(f"{label} report has no test cases")
    for element, outcome in (
        ("skipped", "skipped"),
        ("error", "errors"),
        ("failure", "failures"),
    ):
        affected = [
            test.get("name", "<unnamed>")
            for test in testcases
            if test.find(element) is not None
        ]
        if affected:
            raise CiError(f"{label} {outcome}: " + ", ".join(affected))
    return len(testcases)


class E2ERunner:
    def __init__(self, context: CiContext):
        self.context = context

    def selective(self) -> None:
        impact_path = self.context.repository / "impact.json"
        if not impact_path.is_file():
            raise CiError("selective E2E requires impact.json")
        payload = json.loads(impact_path.read_text(encoding="utf-8"))
        self._run(
            tuple(str(value) for value in payload["families"]),
            tuple(str(value) for value in payload.get("testcases", ())),
        )

    def _run(self, families: tuple[str, ...], testcases: tuple[str, ...] = ()) -> None:
        selected_families = tuple(sorted(set(families)))
        if not selected_families:
            print("No family E2E tests selected")
            return
        if len(selected_families) != 1:
            raise CiError("one E2E job must select exactly one family")
        binary = self._required_path("TRTMC_BINARY")
        runtime_root = self._required_path("TRTMC_RUNTIME_ROOT")
        native_build = self._required_path("TRTMC_NATIVE_BUILD_DIR")
        if not binary.is_file():
            raise CiError(f"TRTMC_BINARY is not a file: {binary}")
        if not runtime_root.is_dir():
            raise CiError(f"TRTMC_RUNTIME_ROOT is not a directory: {runtime_root}")
        if not native_build.is_dir() or not (native_build / "CTestTestfile.cmake").is_file():
            raise CiError("TRTMC_NATIVE_BUILD_DIR is not a configured CTest build tree")
        if not (runtime_root / "libtrtmc_backend_trt.so").is_file():
            raise CiError("TRTMC_RUNTIME_ROOT has no libtrtmc_backend_trt.so")
        known = set(test_impact.inventory(self.context.repository))
        unknown = sorted(set(selected_families) - known)
        if unknown:
            raise CiError("unknown E2E families: " + ", ".join(unknown))
        self._run_family_ctests(native_build, selected_families)
        for family in selected_families:
            family_tests = self.context.repository / "families" / family / "tests"
            ordinary_paths = [
                path
                for path in sorted(family_tests.glob("test_*.py"))
                if path.name != "test_e2e.py"
            ]
            ordinary_tests = [
                str(path.relative_to(self.context.repository)) for path in ordinary_paths
            ]
            if ordinary_tests:
                unit_junit = native_build / f"trtmc-{family}-unit-junit.xml"
                completed = self.context.run(
                    [
                        "python",
                        "-m",
                        "pytest",
                        *ordinary_tests,
                        "-m",
                        "not gpu and not trt",
                        "-q",
                        "-x",
                        "-p",
                        "no:cacheprovider",
                        "--junitxml",
                        unit_junit,
                    ],
                    updates={"PYTHONDONTWRITEBYTECODE": "1"},
                    unset=("PYTEST_ADDOPTS",),
                    check=False,
                    limit=self.context.env.get("PYTHON_UNIT_TIMEOUT", "20m"),
                )
                test_count = _require_passing_junit(
                    unit_junit,
                    "family Python unit tests",
                    allow_empty=completed.returncode == 5,
                )
                if completed.returncode == 5:
                    if test_count:
                        raise CiError(
                            "family Python unit tests reported no tests collected "
                            f"but wrote {test_count} test cases"
                        )
                    print("No CPU family Python tests selected")
                elif completed.returncode:
                    raise CiError(
                        "family Python unit tests failed with exit code "
                        f"{completed.returncode}"
                    )
            hardware_tests = [
                str(path.relative_to(self.context.repository))
                for path in ordinary_paths
                if any(
                    marker in path.read_text(encoding="utf-8")
                    for marker in ("pytest.mark.gpu", "pytest.mark.trt")
                )
            ]
            if hardware_tests:
                hardware_junit = native_build / f"trtmc-{family}-hardware-junit.xml"
                self.context.run(
                    [
                        "python",
                        "-m",
                        "pytest",
                        *hardware_tests,
                        "-m",
                        "gpu or trt",
                        "-q",
                        "-x",
                        "-p",
                        "no:cacheprovider",
                        "--junitxml",
                        hardware_junit,
                    ],
                    updates={"PYTHONDONTWRITEBYTECODE": "1"},
                    unset=("PYTEST_ADDOPTS",),
                    limit=self.context.env.get("TRTMC_E2E_TIMEOUT", "12h"),
                )
                _require_passing_junit(hardware_junit, "family hardware tests")
            test = f"families/{family}/tests/test_e2e.py"
            with self._isolated_runtime_root(runtime_root, family) as isolated:
                command = [
                    "python",
                    "-m",
                    "pytest",
                    test,
                ]
                if testcases:
                    command.extend(("--e2e-testcase", ",".join(sorted(set(testcases)))))
                else:
                    command.extend(("--e2e-model", family))
                command.extend(("-q", "-x", "-p", "no:cacheprovider"))
                self.context.run(
                    command,
                    updates={
                        "TRTMC_BINARY": str(binary),
                        "TRTMC_RUNTIME_ROOT": str(isolated),
                        "PYTHONDONTWRITEBYTECODE": "1",
                    },
                    limit=self.context.env.get("TRTMC_E2E_TIMEOUT", "12h"),
                )

    @contextmanager
    def _isolated_runtime_root(self, runtime_root: Path, family: str):
        required = (
            "libtrtmc_core.so",
            "libtrtmc_backend_trt.so",
            f"libtrtmc_model_{family}.so",
        )
        for name in required:
            if not (runtime_root / name).is_file():
                raise CiError(f"TRTMC_RUNTIME_ROOT has no {name}")

        with tempfile.TemporaryDirectory(prefix=f"trtmc-{family}-runtime-") as directory:
            root = Path(directory)
            isolated = root / "tensorrt_model_connect/bin"
            isolated.mkdir(parents=True)
            for name in required:
                (isolated / name).symlink_to((runtime_root / name).resolve())

            # Preserve only non-family wheel dependencies expected by RUNPATH.
            site_packages = runtime_root.parent.parent
            for package in ("tensorrt_libs", "torch"):
                source = site_packages / package
                if source.is_dir():
                    (root / package).symlink_to(source.resolve(), target_is_directory=True)
            yield isolated

    def _run_family_ctests(self, build: Path, families: tuple[str, ...]) -> None:
        listing = self.context.run(
            ["ctest", "--test-dir", build, "--show-only=json-v1"],
            capture_output=True,
        )
        tests = json.loads(listing.stdout).get("tests", [])
        selected = []
        targets = []
        owners = {f"/families/{family}" for family in families}
        for test in tests:
            command = test.get("command") or []
            properties = {
                str(item["name"]): str(item["value"])
                for item in test.get("properties", [])
                if "name" in item and "value" in item
            }
            location = properties.get("WORKING_DIRECTORY", "")
            if not any(location.endswith(owner) or f"{owner}/" in location for owner in owners):
                continue
            name = str(test["name"])
            selected.append(name)
            targets.append(
                Path(str(command[0])).name
                if command
                else name
                if name.startswith("test_")
                else f"test_{name}"
            )
        if not selected:
            print("No family-owned C++ tests selected")
            return
        self.context.run(
            [
                "cmake",
                "--build",
                build,
                "--parallel",
                "8",
                "--target",
                *sorted(set(targets)),
            ],
            limit=self.context.env.get("CPP_BUILD_TIMEOUT", "30m"),
        )
        expression = "^(" + "|".join(re.escape(name) for name in sorted(selected)) + ")$"
        junit = build / "trtmc-family-ctest.xml"
        self.context.run(
            [
                "ctest",
                "--test-dir",
                build,
                "--output-on-failure",
                "--output-junit",
                junit,
                "-R",
                expression,
            ],
            limit=self.context.env.get("CPP_UNIT_TIMEOUT", "20m"),
        )
        _require_passing_junit(junit, "family C++ tests")

    def _required_path(self, name: str) -> Path:
        value = self.context.env.get(name, "")
        if not value:
            raise CiError(f"{name} is required")
        path = Path(value)
        if not path.exists():
            raise CiError(f"{name} does not exist: {path}")
        return path
