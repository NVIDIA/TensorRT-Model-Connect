# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute the linear, evidence-producing proof inside the isolated container.

Boundary: projected-source validation, build, tests, comparison, and per-model report.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from .context import CiContext
from .model_proof import ModelProofRequest
from .model_proof_selection import ModelProofSelection, ModelProofSelector
from .process import CiError
from .selected_wheel import SelectedWheelRuntime
from .validation import ValidationRunner


_SUPPORTED_E2E_PROOF_KINDS = {
    "functional_invariant",
    "reference",
    "snapshot_regression",
}


def _classify_e2e_proof_kinds(
    verification: dict[str, object],
) -> tuple[str, list[str]]:
    results = verification.get("results")
    raw_proof_kinds = (
        [result.get("proof_kind") for result in results if isinstance(result, dict)]
        if isinstance(results, list)
        else []
    )
    if (
        not raw_proof_kinds
        or not all(isinstance(kind, str) for kind in raw_proof_kinds)
        or not set(raw_proof_kinds) <= _SUPPORTED_E2E_PROOF_KINDS
    ):
        raise CiError(
            f"Model proof found invalid E2E proof kinds: {raw_proof_kinds}"
        )
    proof_kinds = set(raw_proof_kinds)
    ordered = sorted(proof_kinds)
    return (ordered[0] if len(ordered) == 1 else "mixed"), ordered


class ProofStatus:
    """Persist the status of each proof stage for strict HTML rendering."""

    STEPS = {
        "hf_cache_isolation": "hf-cache-repos.json",
        "model_reference_isolation": "selection.json",
        "projection_validation": "source-projection.json, selection.json",
        "configure": "configure.log",
        "scratch_build": "build.log",
        "dso_isolation": "model-dsos.txt, model-dso.dynamic.txt",
        "cpp_tests": "cpp-tests.log",
        "python_tests": "python-model-tests.xml",
        "e2e_reference": "e2e/junit.xml, e2e/*/result.json",
        "engine_build_budget": "engine-builds/*.json, engine-build-verification.json",
        "result_verification": "e2e-verification.json",
        "validation": "validation/eval_summary.json",
        "html_report": "model-proof-report.html",
    }

    def __init__(
        self,
        path: Path,
        request: ModelProofRequest,
        lease: dict[str, object],
    ):
        self.path = path
        self.payload: dict[str, object] = {
            "schema_version": 1,
            "model": request.model,
            "source_revision": request.revision,
            "suite": request.suite,
            **lease,
            "gpu_lease_evidence": "gpu-lease.json",
            "outcome": "running",
            "steps": {
                name: {
                    "status": "running" if name == "projection_validation" else "pending",
                    "evidence": evidence,
                }
                for name, evidence in self.STEPS.items()
            },
        }
        self.save()

    def step(self, name: str, status: str, evidence: str | None = None) -> None:
        step = self.payload["steps"][name]
        step["status"] = status
        if evidence:
            step["evidence"] = evidence
        self.save()

    def fact(self, name: str, value: object) -> None:
        self.payload[name] = value
        self.save()

    def finalize_validation(self, returncode: int, artifacts: Path) -> None:
        self.payload["validation_exit_code"] = returncode
        self.payload["outcome"] = "report-validation" if returncode == 0 else "failed"
        for step in self.payload["steps"].values():
            if step["status"] == "running":
                step["status"] = "passed" if returncode == 0 else "failed"
            elif step["status"] == "pending" and returncode:
                step["status"] = "not-run"
        self.payload["steps"]["html_report"]["status"] = "running"
        evidence = []
        for item in sorted(artifacts.rglob("*")):
            if not item.is_file() or item.name == "model-proof-report.html":
                continue
            relative = str(item.relative_to(artifacts))
            if item.parent == artifacts or item.suffix in {".json", ".xml", ".log"}:
                evidence.append(relative)
        self.payload["evidence_files"] = evidence[:200]
        self.save()

    def finalize_report(self, validation_rc: int, report_rc: int) -> None:
        self.payload["steps"]["html_report"]["status"] = "passed" if report_rc == 0 else "failed"
        self.payload["report_exit_code"] = report_rc
        exit_code = validation_rc or report_rc
        self.payload["outcome"] = "passed" if exit_code == 0 else "failed"
        self.payload["exit_code"] = exit_code
        self.save()

    def save(self) -> None:
        self.path.write_text(json.dumps(self.payload, indent=2) + "\n", encoding="utf-8")


class ModelProofInnerPipeline:
    """Execute the projected build and proof in the order shown in the CI report."""

    def __init__(self, context: CiContext, request: ModelProofRequest):
        self.context = context
        self.request = request
        self.source = Path("/src")
        self.work = Path("/work")
        self.artifacts = Path("/artifacts")
        self.status: ProofStatus | None = None
        self.selection: ModelProofSelection | None = None
        self.selected_wheel: SelectedWheelRuntime | None = None

    def run(self) -> None:
        validation_rc = 0
        error: BaseException | None = None
        try:
            self._prepare()
            lease = self._validate_gpu_lease()
            self.status = ProofStatus(
                self.artifacts / "model-proof-status.json", self.request, lease
            )
            count = self._validate_hf_cache()
            self.status.step("hf_cache_isolation", "passed")
            self.status.fact("hf_cache_isolation", "selected-repositories-only")
            self.status.fact("hf_cache_repository_count", count)
            shutil.copy2(
                self.source / ".trtmc-model-projection.json",
                self.artifacts / "source-projection.json",
            )
            self.selection = ModelProofSelector(
                self.request.model, self.request.suite, self.request.revision, self.source
            ).select(self.artifacts / "selection.json", lease)
            self._validate_reference_cache()
            self.status.step("projection_validation", "passed")
            self._build_and_test()
            print(f"PASS: isolated model proof completed for {self.request.model}")
        except BaseException as caught:
            validation_rc = 1
            error = caught
        report_rc = self._finalize_report(validation_rc)
        if error:
            raise error
        if report_rc:
            raise CiError(f"model proof report evidence validation failed (exit {report_rc})")

    def _prepare(self) -> None:
        if self.request.output_dir != Path("/artifacts"):
            raise CiError("inner output directory must be /artifacts")
        if not (self.source / ".trtmc-model-projection.json").is_file():
            raise CiError("projection manifest is missing from /src")
        if (self.source / ".git").exists():
            raise CiError("projected source must not contain Git metadata")
        if not self.work.is_dir():
            raise CiError("isolated writable work directory is missing")
        for path in (
            self.artifacts,
            self.artifacts / "e2e",
            self.work / "build",
            self.work / "engines",
            self.work / "model-plugins",
            self.work / "tmp",
            self.work / "hf-home",
            self.work / "hf-modules",
        ):
            path.mkdir(parents=True, exist_ok=True)
        self.selected_wheel = SelectedWheelRuntime.prepare(
            self.context,
            self.work / "selected-wheel-runtime",
            self.artifacts / "selected-wheel.json",
            base_python=self._base_python(),
        )

    def _validate_gpu_lease(self) -> dict[str, object]:
        gpu_id = self.context.env.get("TRTMC_MODEL_PROOF_GPU_ID", "")
        resource = self.context.env.get("TRTMC_MODEL_PROOF_RESOURCE_CLASS", "")
        capacity_text = self.context.env.get("TRTMC_MODEL_PROOF_SLOTS_PER_GPU", "")
        min_free_text = self.context.env.get(
            "TRTMC_MODEL_PROOF_MIN_FREE_GPU_MEMORY_MIB", ""
        )
        if not gpu_id.isdigit():
            raise CiError("TRTMC_MODEL_PROOF_GPU_ID must be passed as a non-negative integer")
        if resource not in {"shared", "exclusive_gpu"}:
            raise CiError("TRTMC_MODEL_PROOF_RESOURCE_CLASS must be shared or exclusive_gpu")
        if not capacity_text.isdigit() or not 1 <= int(capacity_text) <= 16:
            raise CiError("TRTMC_MODEL_PROOF_SLOTS_PER_GPU must be an integer from 1 to 16")
        if not min_free_text.isdigit():
            raise CiError(
                "TRTMC_MODEL_PROOF_MIN_FREE_GPU_MEMORY_MIB must be a non-negative integer"
            )
        min_free_gpu_memory_mib = int(min_free_text)
        if min_free_gpu_memory_mib and resource != "exclusive_gpu":
            raise CiError("minimum free GPU memory requires an exclusive_gpu model proof")
        try:
            slots = [
                int(value)
                for value in self.context.env.get("TRTMC_MODEL_PROOF_GPU_SLOT_IDS", "").split(",")
            ]
        except ValueError as error:
            raise CiError(
                "TRTMC_MODEL_PROOF_GPU_SLOT_IDS must be comma-separated integers"
            ) from error
        capacity = int(capacity_text)
        if (
            not slots
            or len(slots) != len(set(slots))
            or any(not 0 <= slot < capacity for slot in slots)
        ):
            raise CiError("TRTMC_MODEL_PROOF_GPU_SLOT_IDS contains invalid slots")
        if resource == "shared" and len(slots) != 1:
            raise CiError("shared model proof must hold exactly one GPU slot")
        if resource == "exclusive_gpu" and slots != list(range(capacity)):
            raise CiError("exclusive_gpu model proof must hold every GPU slot")
        expected = {
            "gpu_id": gpu_id,
            "gpu_slot": slots[0] if resource == "shared" else None,
            "gpu_slots": slots,
            "gpu_slot_ids": slots,
            "slots_per_gpu": capacity,
            "gpu_slots_per_device": capacity,
            "resource_class": resource,
            "gpu_resource_class": resource,
            "min_free_gpu_memory_mib": min_free_gpu_memory_mib,
        }
        evidence = json.loads((self.artifacts / "gpu-lease.json").read_text(encoding="utf-8"))
        schema_version = evidence.get("schema_version")
        if (
            not isinstance(schema_version, int)
            or isinstance(schema_version, bool)
            or schema_version != 1
        ):
            raise CiError("GPU lease evidence has an unsupported schema")
        for field in ("gpu_slots", "gpu_slot_ids"):
            evidence_slots = evidence.get(field)
            if (
                not isinstance(evidence_slots, list)
                or not evidence_slots
                or any(
                    not isinstance(slot, int) or isinstance(slot, bool)
                    for slot in evidence_slots
                )
            ):
                raise CiError(f"GPU lease evidence has invalid {field}")
        for field in ("slots_per_gpu", "gpu_slots_per_device"):
            evidence_capacity = evidence.get(field)
            if (
                not isinstance(evidence_capacity, int)
                or isinstance(evidence_capacity, bool)
            ):
                raise CiError(f"GPU lease evidence has invalid {field}")
        evidence_slot = evidence.get("gpu_slot")
        if resource == "shared":
            if not isinstance(evidence_slot, int) or isinstance(evidence_slot, bool):
                raise CiError("GPU lease evidence has an invalid gpu_slot")
        elif evidence_slot is not None:
            raise CiError("GPU lease evidence has an invalid gpu_slot")
        evidence_minimum = evidence.get("min_free_gpu_memory_mib")
        if (
            not isinstance(evidence_minimum, int)
            or isinstance(evidence_minimum, bool)
            or evidence_minimum != min_free_gpu_memory_mib
        ):
            raise CiError("GPU lease evidence has an invalid minimum free GPU memory")
        for key, value in {
            "model": self.request.model,
            "source_revision": self.request.revision,
            **expected,
        }.items():
            if evidence.get(key) != value:
                raise CiError(f"GPU lease evidence mismatch for {key}")
        admission = evidence.get("gpu_memory_admission")
        if min_free_gpu_memory_mib:
            if not isinstance(admission, dict):
                raise CiError("GPU lease evidence is missing GPU memory admission")
            admission_fields = {
                "source",
                "required_free_mib",
                "observed_total_mib",
                "observed_used_mib",
                "observed_free_mib",
            }
            if set(admission) != admission_fields:
                raise CiError("GPU memory admission has unexpected or missing fields")
            if admission.get("source") not in {
                "nvidia-smi",
                "linux-numa-meminfo",
            }:
                raise CiError("GPU memory admission has an invalid source")
            required_free_mib = admission.get("required_free_mib")
            if (
                not isinstance(required_free_mib, int)
                or isinstance(required_free_mib, bool)
                or required_free_mib != min_free_gpu_memory_mib
            ):
                raise CiError("GPU memory admission requirement does not match the model proof")
            for field in (
                "observed_total_mib",
                "observed_used_mib",
                "observed_free_mib",
            ):
                value = admission.get(field)
                if (
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < 0
                ):
                    raise CiError(f"GPU memory admission has an invalid {field}")
            if (
                admission["observed_used_mib"] > admission["observed_total_mib"]
                or admission["observed_free_mib"] > admission["observed_total_mib"]
            ):
                raise CiError("GPU memory admission has inconsistent memory values")
            if (
                admission["source"] == "linux-numa-meminfo"
                and admission["observed_used_mib"] + admission["observed_free_mib"]
                != admission["observed_total_mib"]
            ):
                raise CiError("GPU NUMA memory admission values do not reconcile")
            if admission["observed_free_mib"] < min_free_gpu_memory_mib:
                raise CiError("GPU memory admission did not satisfy the required free memory")
            expected["gpu_memory_admission"] = {
                field: admission[field] for field in sorted(admission_fields)
            }
        elif admission is not None:
            raise CiError("GPU memory admission is present without a selected requirement")
        return expected

    def _validate_hf_cache(self) -> int:
        expected_environment = {
            "HF_HOME": "/work/hf-home",
            "HF_MODULES_CACHE": "/work/hf-modules",
            "HF_HUB_CACHE": "/hf-cache/hub",
            "HUGGINGFACE_HUB_CACHE": "/hf-cache/hub",
            "TRANSFORMERS_CACHE": "/hf-cache/hub",
        }
        for name, value in expected_environment.items():
            if self.context.env.get(name) != value:
                raise CiError(f"{name} must use the proof-private cache view")
        hub = Path("/hf-cache/hub")
        if not hub.is_dir() or not os.access(hub, os.W_OK):
            raise CiError("selected-repository HF Hub view is unavailable or not writable")
        if Path("/hf-cache/modules").exists():
            raise CiError("global Hugging Face modules must not be visible in the proof container")
        probe = hub / ".trtmc-write-probe"
        probe.touch(mode=0o600)
        probe.unlink()
        payload = json.loads((self.artifacts / "hf-cache-repos.json").read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or payload.get("hub_cache") != "/hf-cache/hub":
            raise CiError("selected HF cache evidence has an unsupported schema")
        repositories = payload.get("repositories")
        if not isinstance(repositories, list) or not repositories:
            raise CiError("selected HF cache evidence contains no repositories")
        expected_folders = set()
        for entry in repositories:
            repo_id = entry.get("repo_id") if isinstance(entry, dict) else None
            if (
                not isinstance(repo_id, str)
                or not repo_id
                or any(part in {"", ".", ".."} for part in repo_id.split("/"))
            ):
                raise CiError(f"selected HF cache evidence has an unsafe repo ID: {repo_id!r}")
            folder = "models--" + repo_id.replace("/", "--")
            if entry.get("cache_folder") != folder or entry.get("repo_type") != "model":
                raise CiError(f"selected HF cache evidence is noncanonical for {repo_id!r}")
            if entry.get("cache_path") != f"/hf-cache/hub/{folder}":
                raise CiError(f"selected HF cache evidence has an invalid path for {repo_id!r}")
            path = hub / folder
            if (
                path.is_symlink()
                or not path.is_dir()
                or not path.resolve().is_relative_to(hub.resolve())
            ):
                raise CiError(f"selected HF cache repository is unavailable: {repo_id}")
            expected_folders.add(folder)
        actual = {path.name for path in hub.iterdir()}
        if actual != expected_folders:
            raise CiError(
                f"selected HF cache view mismatch: {sorted(actual)} != {sorted(expected_folders)}"
            )
        return len(repositories)

    def _validate_reference_cache(self) -> None:
        assert self.status and self.selection
        contract = self.selection.reference_cache
        evidence_path = self.artifacts / "model-reference-cache.json"
        if not contract:
            if self.context.env.get("TRTMC_STORAGE_ROOT") or evidence_path.exists():
                raise CiError("model reference cache exposed for a model that declares none")
            self.status.step(
                "model_reference_isolation",
                "passed",
                "selection.json (no external model reference required)",
            )
            self.status.fact("model_reference_isolation", "not-required")
            return
        if self.context.env.get("TRTMC_STORAGE_ROOT") != "/work/reference-private":
            raise CiError("TRTMC_STORAGE_ROOT must use the proof-private model reference cache")
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        for key, value in {
            "model": self.request.model,
            "repository": contract["repository"],
            "reference_revision": contract["revision"],
            "relative_path": contract["relative_path"],
            "entrypoint": contract["entrypoint"],
            "container_storage_root": "/work/reference-private",
            "copy_method": "git-archive",
        }.items():
            if evidence.get(key) != value:
                raise CiError(f"model reference evidence mismatch for {key}")
        root = Path("/work/reference-private")
        reference = root / contract["relative_path"]
        entrypoint = reference / contract["entrypoint"]
        if entrypoint.is_symlink() or not entrypoint.is_file():
            raise CiError("selected model reference entrypoint is not a regular file")
        environment_variable = contract.get("environment_variable", "")
        if environment_variable and self.context.env.get(environment_variable) != str(reference):
            raise CiError(
                f"{environment_variable} must select the proof-private model reference"
            )
        if any(path.name == ".git" for path in root.rglob(".git")):
            raise CiError("proof-private model reference must not contain Git metadata")
        self.status.step("model_reference_isolation", "passed", "model-reference-cache.json")
        self.status.fact("model_reference_isolation", "selected-pinned-private")
        self.status.fact("model_reference_revision", contract["revision"])

    def _build_and_test(self) -> None:
        assert self.status and self.selection
        payload = self.selection.payload
        owners = payload["owners"]
        runtime_model = str(owners["runtime"])
        runtime_library = str(payload["runtime_library"])
        build_jobs = self.context.env.get("TRTMC_MODEL_PROOF_BUILD_JOBS", "2")
        if not build_jobs.isdigit() or int(build_jobs) < 1:
            raise CiError("TRTMC_MODEL_PROOF_BUILD_JOBS must be a positive integer")
        configure = [
            "cmake",
            "-S",
            self.source,
            "-B",
            self.work / "build",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DTRTMC_BUILD_TESTS=ON",
            "-DTRTMC_BUILD_BENCHMARKS=OFF",
            "-DTRTMC_ENABLE_LIBTORCH_MULTINOMIAL=OFF",
            f"-DTRTMC_MODEL_PROOF_MODEL={runtime_model}",
        ]
        if shutil.which("ninja"):
            configure.extend(["-G", "Ninja"])
        self.status.step("configure", "running")
        self._run_logged(configure, self.artifacts / "configure.log")
        self.status.step("configure", "passed")
        targets = [
            "trtmc",
            "trtmc_backend_trt",
            f"trtmc_model_{runtime_model}",
            *map(str, payload["runtime_tests"]),
        ]
        self.status.step("scratch_build", "running")
        self._run_logged(
            [
                "cmake",
                "--build",
                self.work / "build",
                "--parallel",
                build_jobs,
                "--target",
                *targets,
            ],
            self.artifacts / "build.log",
        )
        self.status.step("scratch_build", "passed")
        dso, scratch_digest, runtime_library_source = self._validate_dso(
            runtime_model, runtime_library
        )
        self._run_cpp_tests(payload["runtime_tests"])
        self._run_python_tests(payload)
        verification = self._run_e2e(payload)
        self._run_validation(runtime_model)
        digest = hashlib.sha256(dso.read_bytes()).hexdigest()
        proof = {
            "schema_version": 1,
            "passed": True,
            "model": self.request.model,
            "source_revision": self.request.revision,
            "runtime_model": runtime_model,
            "runtime_library": runtime_library,
            "runtime_library_sha256": digest,
            "staged_runtime_library_sha256": digest,
            "scratch_runtime_library_sha256": scratch_digest,
            "runtime_library_source": runtime_library_source,
            "staged_model_dso_count": 1,
            **{
                key: payload[key]
                for key in (
                    "gpu_id",
                    "gpu_slot",
                    "gpu_slots",
                    "gpu_slot_ids",
                    "slots_per_gpu",
                    "gpu_slots_per_device",
                    "resource_class",
                    "gpu_resource_class",
                    "min_free_gpu_memory_mib",
                    "gpu_lease_evidence",
                    "suite",
                )
            },
            "e2e_cases": self.selection.e2e_cases,
            "e2e_proof_kind": verification["e2e_proof_kind"],
            "e2e_proof_kinds": verification["e2e_proof_kinds"],
            "engine_builds_per_model": verification["builds_per_model"],
            "engine_build_count": len(verification["records"]),
            "engine_build_verification": "engine-build-verification.json",
            "sibling_model_count": 0,
            "model_dso_count": 1,
            "network": "disabled",
            "plugin_search": "strict",
            "hf_cache_isolation": "selected-repositories-only",
            "hf_cache_repository_count": len(
                json.loads((self.artifacts / "hf-cache-repos.json").read_text())["repositories"]
            ),
            "hf_cache_evidence": "hf-cache-repos.json",
            "model_reference_isolation": (
                "selected-pinned-private" if self.selection.reference_cache else "not-required"
            ),
        }
        if self.selection.min_free_gpu_memory_mib:
            proof["gpu_memory_admission"] = payload["gpu_memory_admission"]
        if self.selection.reference_cache:
            proof["model_reference_revision"] = self.selection.reference_cache["revision"]
            proof["model_reference_evidence"] = "model-reference-cache.json"
        proof.update(self._selected_wheel_proof())
        (self.artifacts / "proof.json").write_text(
            json.dumps(proof, indent=2) + "\n", encoding="utf-8"
        )

    def _run_validation(self, runtime_model: str) -> None:
        assert self.status
        self.status.step(
            "validation",
            "running",
            "validation/eval_summary.json, validation/models/*/summary.json",
        )
        if ValidationRunner(
            self.context,
            self.request.suite,
            runtime_model,
            python=self._python(),
            trtmc=self._trtmc(),
            pythonpath=self._pythonpath(),
            installed_wheel=self.selected_wheel is not None,
        ).run():
            self.status.step("validation", "passed")
        else:
            self.status.step(
                "validation",
                "skipped",
                "not an ETTh1 time-series nightly model",
            )

    def _validate_dso(
        self, runtime_model: str, runtime_library: str
    ) -> tuple[Path, str, str]:
        assert self.status
        self.status.step("dso_isolation", "running")
        dsos = sorted((self.work / "build/models").rglob("libtrtmc_model_*.so"))
        (self.artifacts / "model-dsos.txt").write_text(
            "".join(f"{path}\n" for path in dsos), encoding="utf-8"
        )
        if len(dsos) != 1 or dsos[0].name != runtime_library:
            raise CiError(
                f"scratch build produced {[path.name for path in dsos]}, expected only {runtime_library}"
            )
        dynamic = self.context.output(["readelf", "-d", dsos[0]])
        (self.artifacts / "model-dso.dynamic.txt").write_text(dynamic + "\n", encoding="utf-8")
        dependencies = set(re.findall(r"libtrtmc_model_[^\] ]*\.so", dynamic)) - {runtime_library}
        if dependencies:
            raise CiError(f"model DSO links a sibling model DSO: {sorted(dependencies)}")
        scratch = dsos[0]
        runtime_dso = scratch
        runtime_core: Path | None = None
        runtime_library_source = "scratch-build"
        if self.selected_wheel:
            native_dir = (
                self.selected_wheel.site_packages / "tensorrt_model_connect" / "bin"
            ).resolve()
            runtime_dso = native_dir / runtime_library
            if (
                Path(runtime_library).name != runtime_library
                or runtime_dso.is_symlink()
                or not runtime_dso.is_file()
                or not runtime_dso.resolve().is_relative_to(native_dir)
            ):
                raise CiError(
                    f"selected wheel model DSO is missing or unsafe: {runtime_library}"
                )
            runtime_core = native_dir / "libtrtmc_core.so"
            if (
                runtime_core.is_symlink()
                or not runtime_core.is_file()
                or not runtime_core.resolve().is_relative_to(native_dir)
            ):
                raise CiError("selected wheel core DSO is missing or unsafe: libtrtmc_core.so")
            runtime_library_source = "selected-wheel"

        plugin_dir = self.work / "model-plugins" / runtime_model
        plugin_dir.mkdir(parents=True)
        staged = plugin_dir / runtime_library
        shutil.copy2(runtime_dso, staged)
        if staged.read_bytes() != runtime_dso.read_bytes():
            raise CiError("staged plugin DSO does not byte-match its runtime source")
        core_facts: dict[str, str] = {}
        if runtime_core is not None:
            staged_core = plugin_dir / runtime_core.name
            shutil.copy2(runtime_core, staged_core)
            if (
                staged_core.is_symlink()
                or not staged_core.is_file()
                or not staged_core.resolve().is_relative_to(plugin_dir.resolve())
                or staged_core.read_bytes() != runtime_core.read_bytes()
            ):
                raise CiError("staged core DSO does not byte-match its selected wheel source")
            core_digest = hashlib.sha256(staged_core.read_bytes()).hexdigest()
            core_facts = {
                "runtime_core_library": runtime_core.name,
                "runtime_core_library_sha256": core_digest,
                "staged_runtime_core_library_sha256": core_digest,
                "runtime_core_library_source": "selected-wheel",
            }
        scratch_digest = hashlib.sha256(scratch.read_bytes()).hexdigest()
        digest = hashlib.sha256(staged.read_bytes()).hexdigest()
        for key, value in {
            "runtime_model": runtime_model,
            "runtime_library": runtime_library,
            "runtime_library_sha256": digest,
            "staged_runtime_library_sha256": digest,
            "scratch_runtime_library_sha256": scratch_digest,
            "runtime_library_source": runtime_library_source,
            "sibling_model_count": "0",
            "model_dso_count": "1",
            "network": "disabled",
            "plugin_search": "strict",
            **core_facts,
        }.items():
            self.status.fact(key, value)
        self.status.step(
            "dso_isolation", "passed", "exactly one model DSO; no sibling model DT_NEEDED"
        )
        return staged, scratch_digest, runtime_library_source

    def _run_cpp_tests(self, tests: list[str]) -> None:
        assert self.status
        self.status.step("cpp_tests", "running")
        log = self.artifacts / "cpp-tests.log"
        log.write_text("", encoding="utf-8")
        for test in tests:
            self._run_logged(
                [
                    "ctest",
                    "--test-dir",
                    self.work / "build",
                    "--output-on-failure",
                    "-R",
                    f"^{test}$",
                ],
                log,
                append=True,
                updates={
                    "TRTMC_MODEL_PLUGIN_STRICT": "1",
                    "TRTMC_MODEL_PLUGIN_DIR": str(
                        self.work
                        / ("build/models" if self.selected_wheel else "model-plugins")
                    ),
                },
            )
        self.status.step("cpp_tests", "passed")

    def _run_python_tests(self, payload: dict[str, object]) -> None:
        assert self.status
        tests = [str(self.source / path) for path in payload["python_tests"]]
        if not tests:
            self.status.step("python_tests", "skipped", "no model-owned Python unit tests")
            return
        self.status.step(
            "python_tests", "running", "python-model-tests.xml, python-model-tests.log"
        )
        pytest_options = ["--import-mode=importlib"] if self.selected_wheel else []
        environment = {
            "TRTMC_ELF_TIMING_CACHE_PATH": "",
            "TRTMC_ELF_TIMING_CACHE_METADATA_PATH": "",
            "TRTMC_ELF_TIMING_CACHE_GENERATE": "0",
        }
        environment.update(self._python_environment())
        self._run_logged(
            [
                self._python(),
                "-m",
                "pytest",
                *tests,
                "-v",
                "-p",
                "no:cacheprovider",
                "--rootdir",
                self.source,
                "-c",
                self.source / "pyproject.toml",
                f"--junitxml={self.artifacts / 'python-model-tests.xml'}",
                *pytest_options,
            ],
            self.artifacts / "python-model-tests.log",
            updates=environment,
        )
        self.status.step("python_tests", "passed")

    def _run_e2e(self, payload: dict[str, object]) -> dict[str, object]:
        assert self.status and self.selection
        models_file = self.work / "e2e-models.txt"
        models_file.write_text(
            "".join(f"{model}\n" for model in self.selection.e2e_models), encoding="utf-8"
        )
        filters = [item for name in self.selection.e2e_models for item in ("--e2e-model", name)] + [
            item for name in self.selection.e2e_cases for item in ("--e2e-testcase", name)
        ]
        pytest_options = ["--import-mode=importlib"] if self.selected_wheel else []
        self.status.step("e2e_reference", "running")
        self.status.step("engine_build_budget", "running")
        environment = {
            "TRTMC_MODEL_PLUGIN_STRICT": "1",
            "TRTMC_MODEL_PLUGIN_DIR": str(self.work / "model-plugins"),
            "TRTMC_ENGINE_BUILD_GUARD_DIR": str(self.artifacts / "engine-builds"),
            "TRTMC_ENGINE_BUILD_REVISION": self.request.revision,
            "LD_LIBRARY_PATH": (
                self.context.env.get("LD_LIBRARY_PATH", "")
                if self.selected_wheel
                else f"{self.work / 'build'}:{self.context.env.get('LD_LIBRARY_PATH', '')}"
            ),
        }
        environment.update(self._python_environment())
        self._run_logged(
            [
                self._python(),
                "-m",
                "pytest",
                self.source / str(payload["e2e_test"]),
                "-v",
                "-rs",
                "-p",
                "no:cacheprovider",
                *filters,
                "--rootdir",
                self.source,
                "-c",
                self.source / "pyproject.toml",
                "--engine-dir",
                self.work / "engines",
                "--trtmc-binary",
                self._trtmc(),
                "--hf-python",
                self._python(),
                "--e2e-artifacts-dir",
                self.artifacts / "e2e",
                "--model-plugin-dir",
                self.work / "model-plugins",
                "--rebuild-engines",
                f"--junitxml={self.artifacts / 'e2e/junit.xml'}",
                *pytest_options,
            ],
            self.artifacts / "e2e.log",
            updates=environment,
        )
        self.context.run(
            [
                self._python(),
                self.source / "tools/model_plugin_isolation.py",
                "verify-builds",
                "--models-file",
                models_file,
                "--ledger-dir",
                self.artifacts / "engine-builds",
                "--source-revision",
                self.request.revision,
                "--report",
                self.artifacts / "engine-build-verification.json",
            ]
        )
        verification = json.loads(
            (self.artifacts / "engine-build-verification.json").read_text(encoding="utf-8")
        )
        if verification.get("passed") is not True:
            raise CiError("engine build verification did not pass")
        self.status.step("engine_build_budget", "passed")
        self.status.step("result_verification", "running")
        self.context.run(
            [
                self._python(),
                self.source / "tools/model_plugin_isolation.py",
                "verify-results",
                "--repo-root",
                self.source,
                "--models-file",
                models_file,
                "--artifacts-dir",
                self.artifacts / "e2e",
                "--build-verification-report",
                self.artifacts / "engine-build-verification.json",
                "--report",
                self.artifacts / "e2e-verification.json",
            ]
        )
        e2e_verification = json.loads(
            (self.artifacts / "e2e-verification.json").read_text(encoding="utf-8")
        )
        e2e_proof_kind, e2e_proof_kinds = _classify_e2e_proof_kinds(
            e2e_verification
        )
        self.status.fact("e2e_proof_kind", e2e_proof_kind)
        self.status.fact("e2e_proof_kinds", e2e_proof_kinds)
        has_reference_proof = "reference" in e2e_proof_kinds
        self.status.step(
            "e2e_reference",
            "passed" if has_reference_proof else "skipped",
            (
                "e2e-verification.json (L1/L2 reference oracle)"
                if has_reference_proof
                else f"not claimed: {', '.join(e2e_proof_kinds)} oracle"
            ),
        )
        self.status.step("result_verification", "passed")
        verification["e2e_proof_kind"] = e2e_proof_kind
        verification["e2e_proof_kinds"] = e2e_proof_kinds
        return verification

    def _finalize_report(self, validation_rc: int) -> int:
        if self.status is None:
            lease = {
                "gpu_id": self.context.env.get("TRTMC_MODEL_PROOF_GPU_ID", ""),
                "gpu_slot": None,
                "gpu_slots": [],
                "gpu_slot_ids": [],
                "slots_per_gpu": 0,
                "gpu_slots_per_device": 0,
                "resource_class": self.context.env.get("TRTMC_MODEL_PROOF_RESOURCE_CLASS", ""),
                "gpu_resource_class": self.context.env.get("TRTMC_MODEL_PROOF_RESOURCE_CLASS", ""),
            }
            self.status = ProofStatus(
                self.artifacts / "model-proof-status.json", self.request, lease
            )
        self.status.finalize_validation(validation_rc, self.artifacts)
        result = self.context.run(
            [
                self._python(),
                self.source / "scripts/generate_e2e_report.py",
                "--artifacts-dir",
                self.artifacts / "e2e",
                "--output",
                self.artifacts / "model-proof-report.html",
                "--project-dir",
                self.source,
                "--title",
                f"Isolated Model Proof: {self.request.model} @ {self.request.revision[:12]}",
                "--proof-status",
                self.artifacts / "model-proof-status.json",
                "--proof-json",
                self.artifacts / "proof.json",
                "--selection-json",
                self.artifacts / "selection.json",
                "--strict-evidence",
                "--max-embed-bytes",
                "33554432",
            ],
            check=False,
        )
        self.status.finalize_report(validation_rc, result.returncode)
        if (self.artifacts / "model-proof-report.html").is_file():
            print(f"Model proof HTML report: {self.artifacts / 'model-proof-report.html'}")
        return result.returncode

    def _run_logged(
        self,
        command: list[object],
        path: Path,
        *,
        append: bool = False,
        updates: dict[str, str] | None = None,
    ) -> None:
        environment = dict(self.context.env)
        environment.update(updates or {})
        with path.open("a" if append else "w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [str(item) for item in command],
                cwd=self.source,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log.write(line)
            returncode = process.wait()
        if returncode:
            raise CiError(f"command failed ({returncode}): {' '.join(map(str, command))}")

    def _python(self) -> str:
        if self.selected_wheel:
            return str(self.selected_wheel.python)
        return self._base_python()

    def _base_python(self) -> str:
        configured = self.context.env.get("TRTMC_HF_PYTHON", "/opt/venv/bin/python")
        return configured if Path(configured).is_file() else shutil.which("python3") or "python3"

    def _trtmc(self) -> str:
        if self.selected_wheel:
            return str(self.selected_wheel.trtmc)
        return str(self.work / "build/trtmc")

    def _pythonpath(self) -> str:
        if self.selected_wheel:
            return f"{self.selected_wheel.site_packages}:{self.source}"
        return f"{self.source / 'python'}:{self.source}"

    def _python_environment(self) -> dict[str, str]:
        if self.selected_wheel:
            return self.selected_wheel.environment(self.source, self.context.env)
        return {
            "PYTHONPATH": self._pythonpath(),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TRTMC_BINARY": self._trtmc(),
        }

    def _selected_wheel_proof(self) -> dict[str, object]:
        if not self.selected_wheel:
            return {}
        selected = json.loads(self.selected_wheel.provenance.read_text(encoding="utf-8"))
        return {
            "python_runtime": "selected-wheel",
            "selected_wheel_evidence": "selected-wheel.json",
            "selected_wheel": selected["wheel"],
            "selected_wheel_package_version": selected["package_version"],
            "selected_wheel_tensorrt_version": selected["tensorrt_version"],
        }
