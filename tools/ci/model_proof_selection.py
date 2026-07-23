# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate a one-model projection and select its exact proof contract.

Boundary: read and validate ownership metadata; do not build or execute the model here.
"""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .model_reference_cache import parse_model_reference_contract
from .process import CiError


@dataclass(frozen=True)
class ModelProofSelection:
    """Resolved owners, tests, resource class, and optional reference cache."""

    payload: dict[str, object]

    @property
    def resource_class(self) -> str:
        return str(self.payload["resource_class"])

    @property
    def min_free_gpu_memory_mib(self) -> int:
        return int(self.payload["min_free_gpu_memory_mib"])

    @property
    def e2e_models(self) -> list[str]:
        return list(dict.fromkeys(str(case["model"]) for case in self.payload["e2e_cases"]))

    @property
    def e2e_cases(self) -> list[str]:
        return [str(case["name"]) for case in self.payload["e2e_cases"]]

    @property
    def reference_cache(self) -> dict[str, str] | None:
        value = self.payload.get("model_reference_cache")
        return dict(value) if isinstance(value, dict) else None


class ModelProofSelector:
    """Turn allowlisted projected metadata into one deterministic proof selection."""

    def __init__(self, model: str, suite: str, revision: str, source: Path):
        self.model = model
        self.suite = suite
        self.revision = revision
        self.source = source.resolve()

    def select(self, output: Path, lease: dict[str, object] | None = None) -> ModelProofSelection:
        self._validate_projection()
        owners = self._owners()
        runtime_manifest = (
            self.source / "src/runtime/models" / str(owners["runtime"]) / "MODEL.toml"
        )
        runtime = tomllib.loads(runtime_manifest.read_text(encoding="utf-8"))
        runtime_library = str(
            runtime.get("runtime_library") or f"libtrtmc_model_{owners['runtime']}.so"
        )
        runtime_tests = self._runtime_tests(runtime_manifest, runtime)
        e2e_dir = self.source / "tests/e2e/models" / str(owners["e2e"])
        owner_data = tomllib.loads((e2e_dir / "MODEL.toml").read_text(encoding="utf-8"))
        reference_cache = self._reference_cache(
            owner_data, str(owners["e2e"]), e2e_dir / "MODEL.toml"
        )
        cases = self._cases(e2e_dir)
        selected_cases = self._select_cases(cases, str(owners["e2e"]))
        resource = (
            "exclusive_gpu"
            if any(case["resource_class"] == "exclusive_gpu" for case in selected_cases)
            else "shared"
        )
        min_free_gpu_memory_mib = max(
            int(case["min_free_gpu_memory_mib"]) for case in selected_cases
        )
        lease_fields = (
            self._validate_lease(lease, resource, min_free_gpu_memory_mib) if lease else {}
        )
        e2e_tests = sorted(e2e_dir.glob("test_*_e2e.py"))
        if len(e2e_tests) != 1:
            raise CiError(
                f"projected model must have exactly one canonical E2E test; found {len(e2e_tests)}"
            )
        python_tests = [
            path for path in e2e_dir.rglob("test_*.py") if not path.name.endswith("_e2e.py")
        ]
        python_family = owners["python"]
        if python_family:
            python_tests.extend(
                (self.source / "python/tensorrt_model_connect/families" / str(python_family)).rglob(
                    "test_*.py"
                )
            )
        payload: dict[str, object] = {
            "schema_version": 1,
            "requested_model": self.model,
            "owners": owners,
            "runtime_library": runtime_library,
            "runtime_tests": runtime_tests,
            "python_family": python_family,
            "python_tests": [
                str(path.relative_to(self.source)) for path in sorted(set(python_tests))
            ],
            "suite": self.suite,
            "resource_class": resource,
            "gpu_resource_class": resource,
            "min_free_gpu_memory_mib": min_free_gpu_memory_mib,
            "e2e_cases": selected_cases,
            "e2e_test": str(e2e_tests[0].relative_to(self.source)),
            **lease_fields,
        }
        if reference_cache:
            payload["model_reference_cache"] = reference_cache
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return ModelProofSelection(payload)

    def _validate_projection(self) -> dict[str, object]:
        path = self.source / ".trtmc-model-projection.json"
        projection = json.loads(path.read_text(encoding="utf-8"))
        if projection.get("revision") != self.revision:
            raise CiError(
                f"projection revision {projection.get('revision')!r}, expected {self.revision!r}"
            )
        declared = projection.get(
            "model", projection.get("selected_model", projection.get("selected_models"))
        )
        if isinstance(declared, list) and declared != [self.model]:
            raise CiError(f"projection selected models {declared!r}, expected {[self.model]!r}")
        if not isinstance(declared, list) and declared not in (None, self.model):
            raise CiError(f"projection selected model {declared!r}, expected {self.model!r}")
        for path in self.source.rglob("*"):
            if path.is_symlink() and not (path.parent / os.readlink(path)).resolve().is_relative_to(
                self.source
            ):
                raise CiError(f"projection contains escaping symlink: {path}")
        return projection

    def _owners(self) -> dict[str, str | None]:
        roots = {
            "python": self.source / "python/tensorrt_model_connect/families",
            "runtime": self.source / "src/runtime/models",
            "e2e": self.source / "tests/e2e/models",
        }
        owners: dict[str, str | None] = {}
        for kind, root in roots.items():
            manifests = sorted(root.glob("*/MODEL.toml"))
            minimum = 0 if kind == "python" else 1
            if len(manifests) < minimum or len(manifests) > 1:
                qualifier = "at most" if kind == "python" else "exactly"
                raise CiError(
                    f"projected {kind} ownership root must contain {qualifier} one MODEL.toml; "
                    f"found {len(manifests)}"
                )
            if not manifests:
                owners[kind] = None
                continue
            data = tomllib.loads(manifests[0].read_text(encoding="utf-8"))
            owner = str(data.get("id") or "")
            if not owner or owner != manifests[0].parent.name:
                raise CiError(f"invalid projected {kind} manifest: {manifests[0]}")
            owners[kind] = owner
        projection = json.loads(
            (self.source / ".trtmc-model-projection.json").read_text(encoding="utf-8")
        )
        if projection.get("runtime_model") != owners["runtime"]:
            raise CiError("projection runtime model does not match projected ownership")
        if projection.get("e2e_family") != owners["e2e"]:
            raise CiError("projection E2E family does not match projected ownership")
        return owners

    @staticmethod
    def _runtime_tests(path: Path, data: dict[str, object]) -> list[str]:
        tests = []
        for entry in data.get("runtime_tests", []):
            fields = str(entry).split("|")
            if len(fields) != 5 or not fields[0]:
                raise CiError(f"invalid runtime_tests entry in {path}: {entry!r}")
            tests.append(fields[0])
        return tests

    def _reference_cache(
        self, owner: dict[str, object], family: str, owner_manifest: Path
    ) -> dict[str, str] | None:
        contract = parse_model_reference_contract(
            owner,
            family,
            owner_manifest,
            self.suite,
        )
        return contract.as_payload() if contract else None

    def _cases(self, e2e_dir: Path) -> list[dict[str, object]]:
        timing = {}
        timing_path = self.source / "tests/e2e/timing_estimates.json"
        if timing_path.is_file():
            timing = json.loads(timing_path.read_text(encoding="utf-8")).get("estimates_s", {})
        cases = []
        for path in sorted((e2e_dir / "manifests").glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("skip_reason") or data.get("skip"):
                continue
            resource = str(data.get("e2e_parallel_resource") or "shared")
            if resource not in {"shared", "exclusive_gpu"}:
                raise CiError(f"E2E manifest has invalid e2e_parallel_resource: {path}")
            min_free_gpu_memory_mib = 0
            if "e2e_min_free_gpu_memory_mib" in data:
                raw_minimum = data["e2e_min_free_gpu_memory_mib"]
                if (
                    isinstance(raw_minimum, bool)
                    or not isinstance(raw_minimum, int)
                    or raw_minimum <= 0
                ):
                    raise CiError(
                        "E2E manifest e2e_min_free_gpu_memory_mib must be a positive "
                        f"integer: {path}"
                    )
                if resource != "exclusive_gpu":
                    raise CiError(
                        "E2E manifest e2e_min_free_gpu_memory_mib requires "
                        f"e2e_parallel_resource='exclusive_gpu': {path}"
                    )
                min_free_gpu_memory_mib = raw_minimum
            model = str(data.get("name") or path.stem)
            testcases = data.get("testcases")
            if not isinstance(testcases, list) or not testcases:
                raise CiError(f"E2E manifest has no testcases: {path}")
            for testcase in testcases:
                if not isinstance(testcase, dict):
                    raise CiError(f"E2E manifest has an invalid testcase: {path}")
                if "e2e_min_free_gpu_memory_mib" in testcase:
                    raise CiError(
                        "E2E manifest e2e_min_free_gpu_memory_mib is model-only: "
                        f"{path}"
                    )
                if testcase.get("skip_reason") or testcase.get("skip"):
                    continue
                name = str(testcase.get("name") or "")
                if not name:
                    raise CiError(f"E2E manifest has an unnamed testcase: {path}")
                tier = str(testcase.get("ci_tier") or data.get("ci_tier") or "")
                if tier == "multi_device":
                    continue
                cases.append(
                    {
                        "name": name,
                        "model": model,
                        "manifest": path.name,
                        "ci_tier": tier,
                        "l0_replacement": str(testcase.get("l0_replacement") or ""),
                        "estimated_seconds": timing.get(name),
                        "resource_class": resource,
                        "gpu_resource_class": resource,
                        "min_free_gpu_memory_mib": min_free_gpu_memory_mib,
                    }
                )
        return sorted(cases, key=lambda case: (case["name"], case["model"], case["manifest"]))

    def _select_cases(self, cases: list[dict[str, object]], family: str) -> list[dict[str, object]]:
        if not cases:
            raise CiError(f"no single-GPU E2E case is available for {family}")
        if self.suite == "nightly":
            production = [case for case in cases if case["ci_tier"] != "l0_only"]
            return production or cases
        eligible = [
            case for case in cases if case["ci_tier"] not in {"nightly_only", "multi_device"}
        ]
        if not eligible:
            raise CiError(f"no premerge E2E case is available for {family}")
        replacements = {
            case["l0_replacement"]
            for case in cases
            if case["ci_tier"] == "nightly_only" and case["l0_replacement"]
        }
        candidates = [case for case in eligible if case["name"] in replacements] or eligible
        priority = {"l0_only": 0, "contract_only": 1, "": 2}
        candidates.sort(
            key=lambda case: (
                priority.get(case["ci_tier"], 2),
                case["estimated_seconds"]
                if isinstance(case["estimated_seconds"], (int, float))
                else float("inf"),
                case["name"],
                case["model"],
            )
        )
        return candidates[:1]

    @staticmethod
    def _validate_lease(
        lease: dict[str, object],
        resource: str,
        min_free_gpu_memory_mib: int,
    ) -> dict[str, object]:
        slots = list(lease["gpu_slot_ids"])
        capacity = int(lease["slots_per_gpu"])
        if lease["resource_class"] != resource:
            raise CiError(
                f"leased GPU resource class {lease['resource_class']!r} does not match "
                f"selected E2E resource class {resource!r}"
            )
        if resource == "shared" and len(slots) != 1:
            raise CiError("shared selection must hold exactly one GPU slot")
        if resource == "exclusive_gpu" and slots != list(range(capacity)):
            raise CiError("exclusive_gpu selection must hold every GPU slot")
        lease_minimum = lease.get("min_free_gpu_memory_mib")
        if (
            not isinstance(lease_minimum, int)
            or isinstance(lease_minimum, bool)
            or lease_minimum != min_free_gpu_memory_mib
        ):
            raise CiError(
                "leased minimum free GPU memory does not match selected E2E requirements"
            )
        lease_fields: dict[str, object] = {
            "gpu_id": str(lease["gpu_id"]),
            "gpu_slot": slots[0] if resource == "shared" else None,
            "gpu_slots": slots,
            "gpu_slot_ids": slots,
            "slots_per_gpu": capacity,
            "gpu_slots_per_device": capacity,
            "gpu_resource_class": resource,
            "min_free_gpu_memory_mib": min_free_gpu_memory_mib,
            "gpu_lease_evidence": "gpu-lease.json",
        }
        admission = lease.get("gpu_memory_admission")
        if min_free_gpu_memory_mib:
            if not isinstance(admission, dict):
                raise CiError("capacity-gated selection requires GPU memory admission evidence")
            lease_fields["gpu_memory_admission"] = dict(admission)
        elif admission is not None:
            raise CiError("GPU memory admission is present without a selected requirement")
        return lease_fields
