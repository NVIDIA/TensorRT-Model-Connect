# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and test affected models from source projections containing no peers.

Boundary: multi-model isolation queues; one hermetic proof is owned by ``model_proof``.
"""

from __future__ import annotations

import concurrent.futures
import json
import shutil
import subprocess
import time
from pathlib import Path

from .context import CiContext
from .package import WheelPackageManager
from .process import CiError


class IsolatedModelRunner:
    """Prove model locality by source masking, single-DSO build, E2E, and audit."""

    def __init__(self, context: CiContext):
        self.context = context
        self.package = WheelPackageManager(context)
        self.build_state = self.package.build_metadata()

    def run(self, models_file: Path, result_dir: Path) -> None:
        isolation_root = self.context.state_dir / "model-isolation"
        audit_root = result_dir / "model_isolation"
        self.context.remove(isolation_root, audit_root)
        audit_root.mkdir(parents=True)
        self.context.run(
            [
                "python3",
                "tools/model_plugin_isolation.py",
                "plan",
                "--models-file",
                models_file,
                "--output-dir",
                isolation_root,
                "--clean",
            ]
        )
        gpu_ids = self._healthy_gpus()
        maximum = self.context.positive_integer(
            self.context.env.get("TRTMC_ISOLATION_MAX_PARALLEL_GROUPS", str(len(gpu_ids))),
            "TRTMC_ISOLATION_MAX_PARALLEL_GROUPS",
        )
        gpu_ids = gpu_ids[:maximum]
        if self.context.env.get("TRTMC_ISOLATION_BUILD_JOBS"):
            build_jobs = self.context.positive_integer(
                self.context.env["TRTMC_ISOLATION_BUILD_JOBS"], "TRTMC_ISOLATION_BUILD_JOBS"
            )
        else:
            total = self.context.positive_integer(
                self.context.env.get("TRTMC_ISOLATION_TOTAL_BUILD_JOBS", "16"),
                "TRTMC_ISOLATION_TOTAL_BUILD_JOBS",
            )
            build_jobs = max(2, total // len(gpu_ids))
            self.context.env["TRTMC_ISOLATION_BUILD_JOBS"] = str(build_jobs)
        print(f"Isolation workers: GPUs={gpu_ids} build_jobs_per_worker={build_jobs}")

        schedule_dir = isolation_root / "schedule"
        command = [
            "python3",
            "tools/model_plugin_isolation.py",
            "schedule",
            "--plan",
            isolation_root / "plan.json",
            "--output-dir",
            schedule_dir,
            "--timing-estimates",
            "tests/e2e/timing_estimates.json",
            "--default-estimate-seconds",
            self.context.env.get("TRTMC_ISOLATION_DEFAULT_ESTIMATE_S", "600"),
            "--build-overhead-seconds",
            self.context.env.get("TRTMC_ISOLATION_BUILD_ESTIMATE_S", "60"),
            "--clean",
        ]
        for gpu in gpu_ids:
            command.extend(["--gpu-id", str(gpu)])
        self.context.run(command)
        shutil.copy2(schedule_dir / "schedule.json", audit_root / "schedule.json")

        failures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
            futures = {
                pool.submit(self._run_queue, gpu, schedule_dir / f"gpu-{gpu}.txt", result_dir): gpu
                for gpu in gpu_ids
            }
            for future, gpu in futures.items():
                try:
                    future.result()
                except Exception as error:
                    failures.append(f"GPU queue {gpu}: {error}")
        if failures:
            raise CiError("; ".join(failures))

    def _healthy_gpus(self) -> list[int]:
        listing = self.context.run(["nvidia-smi", "-L"], check=False, capture_output=True)
        count = len([line for line in listing.stdout.splitlines() if line.strip()])
        python = self.context.env.get("HF_PYTHON", "/opt/venv/bin/python")
        probe = (
            "import tensorrt as trt; logger=trt.Logger(trt.Logger.ERROR); "
            "raise SystemExit(0 if trt.Builder(logger) is not None else 1)"
        )
        healthy = []
        for gpu in range(count):
            result = self.context.run(
                [python, "-c", probe],
                updates={"CUDA_VISIBLE_DEVICES": str(gpu)},
                check=False,
                capture_output=True,
            )
            if result.returncode == 0:
                healthy.append(gpu)
            else:
                print(f"WARN: GPU {gpu} failed TensorRT builder health check")
        if not healthy:
            raise CiError("no GPU passed the TensorRT builder health check")
        exclude = self.context.env.get("TRTMC_E2E_EXCLUDE_GPU0")
        if exclude is None:
            exclude = "1" if self.context.env.get("GITHUB_RUN_ID") else "0"
        if exclude != "0" and len(healthy) > 1 and 0 in healthy:
            healthy.remove(0)
        return healthy

    def _run_queue(self, gpu: int, queue: Path, result_dir: Path) -> None:
        failed = []
        for raw in queue.read_text(encoding="utf-8").splitlines():
            manifest = Path(raw.strip())
            if manifest and not self._run_group(manifest, result_dir, gpu):
                failed.append(manifest.parent.name)
        if failed:
            raise CiError(f"isolated groups failed: {', '.join(failed)}")

    def _run_group(self, manifest: Path, result_dir: Path, gpu: int) -> bool:
        group = json.loads(manifest.read_text(encoding="utf-8"))
        group_id = str(group["id"])
        audit = result_dir / "model_isolation" / group_id
        audit.mkdir(parents=True, exist_ok=True)
        log = audit / "console.log"
        started = time.time()
        try:
            with log.open("w", encoding="utf-8") as output:
                self._execute_group(manifest, group, result_dir, gpu, audit, output)
        except Exception as error:
            elapsed = int(time.time() - started)
            print(f"FAIL isolated group={group_id} gpu={gpu} elapsed={elapsed}s: {error}")
            lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
            print("\n".join(lines[-120:]))
            return False
        elapsed = int(time.time() - started)
        print(f"PASS isolated group={group_id} gpu={gpu} elapsed={elapsed}s")
        if self.context.env.get("TRTMC_ISOLATION_KEEP_WORKTREES", "0") == "0":
            for name in ("source", "build", "engines", "model_plugins"):
                self.context.remove(manifest.parent / name)
        return True

    def _execute_group(
        self,
        manifest: Path,
        group: dict[str, object],
        result_dir: Path,
        gpu: int,
        audit: Path,
        output,
    ) -> None:
        group_dir = manifest.parent
        models = group_dir / "models.txt"
        source = group_dir / "source"
        build = group_dir / "build"
        engines = group_dir / "engines"
        plugins = group_dir / "model_plugins"
        self._logged(
            [
                "python3",
                "tools/model_plugin_isolation.py",
                "stage-source",
                "--models-file",
                models,
                "--output-dir",
                source,
                "--clean",
            ],
            output,
        )
        shutil.copy2(manifest, audit / "group.json")
        shutil.copy2(source / ".trtmc-isolation.json", audit / "source-projection.json")
        self._configure(source, build, output)
        self._logged(
            [
                "timeout",
                "--kill-after=2m",
                self.context.env.get("SELECTIVE_E2E_BUILD_TIMEOUT", "30m"),
                "cmake",
                "--build",
                build,
                "--parallel",
                self.context.env.get("TRTMC_ISOLATION_BUILD_JOBS", "16"),
                "--target",
                "trtmc",
                "trtmc_backend_trt",
                str(group["runtime_plugin"]["target"]),
            ],
            output,
        )
        dsos = list((build / "models").rglob("libtrtmc_model_*.so"))
        if len(dsos) != 1:
            raise CiError(
                f"isolated build {group['id']} produced {len(dsos)} model DSOs; expected exactly 1"
            )
        self._logged(
            [
                "python3",
                source / "tools/model_plugin_isolation.py",
                "prepare",
                "--repo-root",
                source,
                "--models-file",
                models,
                "--build-dir",
                build,
                "--output-dir",
                plugins,
            ],
            output,
        )
        family = str(group["family"])
        test_files = sorted((source / "tests/e2e/models" / family).glob("test_*_e2e.py"))
        if len(test_files) != 1:
            raise CiError(f"{group['id']} has {len(test_files)} canonical E2E files; expected 1")
        selected = [
            line.strip() for line in models.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        engines.mkdir(parents=True, exist_ok=True)
        (result_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        library_path = [str(build)]
        for name in ("TRTMC_TRT_LIBRARY", "TRTMC_CUDART_LIBRARY"):
            if self.context.env.get(name):
                library_path.append(str(Path(self.context.env[name]).parent))
        environment = dict(self.context.env)
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "HF_HUB_OFFLINE": "1",
                "PYTHONNOUSERSITE": "1",
                "PYTHONPATH": f"{source / 'python'}:{source}",
                "LD_LIBRARY_PATH": ":".join(library_path),
            }
        )
        command = [
            "timeout",
            "--kill-after=2m",
            self.context.env.get("SELECTIVE_E2E_GROUP_TIMEOUT", "90m"),
            self.context.env.get("HF_PYTHON", "/opt/venv/bin/python"),
            "-m",
            "pytest",
            *test_files,
            "-v",
            "--rootdir",
            source,
            "-c",
            source / "pyproject.toml",
            "--engine-dir",
            engines,
            "--trtmc-binary",
            build / "trtmc",
            "--hf-python",
            self.context.env.get("HF_PYTHON", "/opt/venv/bin/python"),
            "--e2e-artifacts-dir",
            result_dir / "artifacts",
            "--model-plugin-dir",
            plugins,
            "--e2e-models-file",
            models,
            "--e2e-exclude-ci-tier",
            "nightly_only",
            *[item for model in selected for item in ("--e2e-model", model)],
            "--rebuild-engines",
            f"--junitxml={audit / 'junit.xml'}",
        ]
        e2e = subprocess.run(
            [str(item) for item in command],
            cwd=source,
            env=environment,
            text=True,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
        verify = subprocess.run(
            [
                "python3",
                str(source / "tools/model_plugin_isolation.py"),
                "verify-results",
                "--repo-root",
                str(source),
                "--models-file",
                str(models),
                "--artifacts-dir",
                str(result_dir / "artifacts"),
                "--report",
                str(audit / "verification.json"),
            ],
            cwd=self.context.repository,
            env=self.context.env,
            text=True,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if e2e.returncode:
            raise CiError(f"isolated E2E exited with {e2e.returncode}")
        if verify.returncode:
            raise CiError(f"isolation verification exited with {verify.returncode}")

    def _configure(self, source: Path, build: Path, output) -> None:
        cache = Path(self.build_state["cmake_build_dir"]) / "CMakeCache.txt"
        if not cache.is_file():
            raise CiError(f"reusable CMake cache is missing: {cache}")
        values = {}
        for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" in line and ":" in line.split("=", 1)[0]:
                key, value = line.split("=", 1)
                values[key.split(":", 1)[0]] = value
        command = [
            "timeout",
            "--kill-after=2m",
            self.context.env.get("SELECTIVE_E2E_CONFIGURE_TIMEOUT", "10m"),
            "cmake",
            "-S",
            source,
            "-B",
            build,
            "-DCMAKE_BUILD_TYPE=Release",
            "-DTRTMC_BUILD_TESTS=OFF",
            "-DTRTMC_BUILD_BENCHMARKS=OFF",
            "-DTRTMC_ENABLE_LIBTORCH_MULTINOMIAL=OFF",
        ]
        if shutil.which("ninja"):
            command.extend(["-G", "Ninja"])
        for name in (
            "TRTMC_TRT_BACKEND_ABI",
            "TRTMC_TRT_INCLUDE_DIR",
            "TRTMC_TRT_LIBRARY",
            "TRTMC_CUDA_INCLUDE_DIR",
            "TRTMC_CUDART_LIBRARY",
            "CMAKE_CUDA_ARCHITECTURES",
        ):
            if values.get(name):
                command.append(f"-D{name}={values[name]}")
        nlohmann = Path(self.build_state["cmake_build_dir"]) / "_deps/nlohmann_json-src"
        if nlohmann.is_dir():
            command.append(f"-DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON={nlohmann}")
        roots = (Path(self.build_state["conan_out_dir"]), Path(self.build_state["cmake_build_dir"]))
        toolchain = next(
            (path for root in roots for path in root.rglob("conan_toolchain.cmake")), None
        )
        if toolchain:
            command.append(f"-DCMAKE_TOOLCHAIN_FILE={toolchain}")
        self._logged(command, output)

    def _logged(self, command: list[object], output) -> None:
        result = subprocess.run(
            [str(item) for item in command],
            cwd=self.context.repository,
            env=self.context.env,
            text=True,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if result.returncode:
            raise CiError(f"command exited with {result.returncode}: {' '.join(map(str, command))}")
