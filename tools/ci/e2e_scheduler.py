# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect, balance, execute, and summarize E2E tests across healthy GPUs."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from scripts import schedule_e2e

from .context import CiContext
from .process import CiError


@dataclass
class E2EParallelConfig:
    """Command-line configuration for the parallel E2E scheduler."""

    engine_dir: Path
    result_dir: Path
    trtmc_binary: Path
    hf_python: Path
    num_gpus: int | None
    workers_per_gpu: int
    progress_interval: int
    timeout_seconds: int | None
    manifest_dir: Path
    models_file: Path | None
    tests_file: Path | None
    filter_args: list[str] = field(default_factory=list)
    collect_args: list[str] = field(default_factory=list)
    pytest_args: list[str] = field(default_factory=list)

    @classmethod
    def parse(cls, arguments: list[str], env: dict[str, str]) -> "E2EParallelConfig":
        parser = argparse.ArgumentParser(prog="python3 -m tools.ci e2e")
        parser.add_argument(
            "--engine-dir",
            type=Path,
            default=Path(
                env.get("ENGINE_DIR", "/workspace/users/yifeif/tensorrt-model-connect/engines")
            ),
        )
        parser.add_argument(
            "--result-dir",
            type=Path,
            default=Path(
                env.get("RESULT_DIR", "/workspace/users/yifeif/tensorrt-model-connect/test-result")
            ),
        )
        parser.add_argument(
            "--trtmc-binary", type=Path, default=Path(env.get("TRTMC_BINARY", "./build/trtmc"))
        )
        parser.add_argument(
            "--hf-python", type=Path, default=Path(env.get("HF_PYTHON", "/opt/venv/bin/python"))
        )
        parser.add_argument(
            "--num-gpus", type=int, default=int(env["NUM_GPUS"]) if env.get("NUM_GPUS") else None
        )
        parser.add_argument(
            "--workers-per-gpu", type=int, default=int(env.get("WORKERS_PER_GPU", "4"))
        )
        parser.add_argument(
            "--progress-interval", type=int, default=int(env.get("PROGRESS_INTERVAL", "30"))
        )
        parser.add_argument("--timeout")
        parser.add_argument("--task-strategy", action="append", default=[])
        parser.add_argument("--exclude-ci-tier", action="append", default=[])
        parser.add_argument("--models-file", type=Path)
        parser.add_argument("--tests-file", type=Path)
        parser.add_argument(
            "--manifest-dir",
            type=Path,
            default=Path(env.get("TRTMC_E2E_MANIFEST_DIR", "tests/e2e/models")),
        )
        known, extra = parser.parse_known_args(arguments)
        for name in ("num_gpus", "workers_per_gpu", "progress_interval"):
            value = getattr(known, name)
            if value is not None and value < 1:
                parser.error(f"--{name.replace('_', '-')} must be positive")
        return cls(
            engine_dir=known.engine_dir,
            result_dir=known.result_dir,
            trtmc_binary=known.trtmc_binary,
            hf_python=known.hf_python,
            num_gpus=known.num_gpus,
            workers_per_gpu=known.workers_per_gpu,
            progress_interval=known.progress_interval,
            timeout_seconds=cls._duration_seconds(known.timeout) if known.timeout else None,
            manifest_dir=known.manifest_dir,
            models_file=known.models_file,
            tests_file=known.tests_file,
            filter_args=[
                item for value in known.task_strategy for item in ("--e2e-task-strategy", value)
            ],
            collect_args=[
                item for value in known.exclude_ci_tier for item in ("--e2e-exclude-ci-tier", value)
            ],
            pytest_args=extra,
        )

    @staticmethod
    def _duration_seconds(value: str) -> int:
        match = re.fullmatch(r"([1-9][0-9]*)([hms]?)", value)
        if not match:
            raise CiError(f"invalid E2E timeout: {value}")
        multiplier = {"": 1, "s": 1, "m": 60, "h": 3600}[match.group(2)]
        return int(match.group(1)) * multiplier


@dataclass
class E2EWorker:
    """One pytest process bound to one physical GPU."""

    logical_gpu: int
    physical_gpu: int
    phase: str
    index: int
    tests: list[str]
    process: subprocess.Popen[str]
    log_handle: object

    @property
    def label(self) -> str:
        return f"gpu{self.logical_gpu}-{self.phase}-w{self.index}"


class E2EParallelRunner:
    """Run exclusive model builds first, then fill each GPU with shared workers."""

    RESULT_PATTERN = re.compile(
        r"(?P<node>tests/\S+::test_(?:model_)?e2e\[[^]]+\]).*?"
        r"(?P<status>PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS)"
    )

    def __init__(self, context: CiContext, config: E2EParallelConfig):
        self.context = context
        self.config = config
        self.gpu_ids: list[int] = []
        self.test_ids: list[str] = []
        self.workers: list[E2EWorker] = []
        self.failures = 0
        self.started_at = 0.0
        self.deadline: float | None = None

    def run(self) -> int:
        self.gpu_ids = self._healthy_gpus()
        self._prepare_outputs()
        self.test_ids = self._collect_tests()
        if not self.test_ids:
            raise CiError("No E2E entries collected. Check --task-strategy filter.")
        print(f"Collected {len(self.test_ids)} E2E scheduler entries")
        phases = self._schedule()
        self.started_at = time.time()
        if self.config.timeout_seconds:
            self.deadline = time.monotonic() + self.config.timeout_seconds
        try:
            if [str(phase["name"]) for phase in phases] == ["exclusive_gpu", "shared"]:
                self._run_pipelined(phases)
            else:
                for phase in phases:
                    if self._run_phase(phase):
                        print(f"Stopping after failed E2E phase: {phase['name']}")
                        break
        except BaseException:
            self._terminate_workers()
            raise
        elapsed = int(time.time() - self.started_at)
        print(f"\n=== All E2E phases finished in {elapsed // 60}m {elapsed % 60}s ===")
        self._merge_junit()
        self._write_timing_summary()
        self._print_summary()
        return self.failures

    def _healthy_gpus(self) -> list[int]:
        physical_count = self.config.num_gpus
        if physical_count is None:
            result = self.context.run(["nvidia-smi", "-L"], check=False, capture_output=True)
            physical_count = len([line for line in result.stdout.splitlines() if line.strip()])
        if physical_count == 0:
            raise CiError("No GPUs detected. Set NUM_GPUS=1 to run on CPU (will likely fail).")
        probe = (
            "import tensorrt as trt; logger=trt.Logger(trt.Logger.ERROR); "
            "raise SystemExit(0 if trt.Builder(logger) is not None else 1)"
        )
        healthy = []
        for gpu in range(physical_count):
            result = self.context.run(
                [self.config.hf_python, "-c", probe],
                updates={"CUDA_VISIBLE_DEVICES": str(gpu)},
                check=False,
                capture_output=True,
            )
            if result.returncode == 0:
                healthy.append(gpu)
            else:
                print(
                    f"WARN: GPU {gpu} failed TensorRT builder health check; excluding from E2E schedule."
                )
        if not healthy:
            raise CiError("No GPUs passed TensorRT builder health check.")
        exclude_gpu0 = self.context.env.get("TRTMC_E2E_EXCLUDE_GPU0")
        if exclude_gpu0 is None:
            exclude_gpu0 = "1" if self.context.env.get("GITHUB_RUN_ID") else "0"
        if exclude_gpu0 != "0" and len(healthy) > 1 and 0 in healthy:
            healthy.remove(0)
            print(f"INFO: Excluding physical GPU 0 from E2E worker assignment ({healthy}).")
        elif (
            self.context.env.get("TRTMC_E2E_DEPRIORITIZE_GPU0", "1") != "0"
            and len(healthy) > 1
            and healthy[0] == 0
        ):
            healthy = [*healthy[1:], 0]
            print(f"INFO: Scheduling physical GPU 0 last for E2E worker assignment ({healthy}).")
        return healthy

    def _prepare_outputs(self) -> None:
        self.config.result_dir.mkdir(parents=True, exist_ok=True)
        self.config.engine_dir.mkdir(parents=True, exist_ok=True)
        for pattern in ("console-gpu*-w*.log", "junit-gpu*-w*.xml", "junit.xml"):
            for path in self.config.result_dir.glob(pattern):
                path.unlink(missing_ok=True)
        print("=== E2E Parallel Test Runner ===")
        print(f"  GPUs:            {len(self.gpu_ids)} ({self.gpu_ids})")
        print(f"  Workers/GPU:     {self.config.workers_per_gpu}")
        print(f"  Engines:         {self.config.engine_dir}")
        print(f"  Results:         {self.config.result_dir}")
        print(f"  Binary:          {self.config.trtmc_binary}")
        print(f"  HF Python:       {self.config.hf_python}")
        print(f"  Extra args:      {self.config.pytest_args or 'none'}")
        print(f"  Models file:     {self.config.models_file or 'none (collect all)'}")
        print(f"  Tests file:      {self.config.tests_file or 'none'}")

    def _collect_tests(self) -> list[str]:
        for label, path in (("Models", self.config.models_file), ("Tests", self.config.tests_file)):
            if path is not None and not path.is_file():
                raise CiError(f"{label} file not found: {path}")
        if self.config.tests_file and not self.config.models_file:
            print("  Scheduler source: tests file")
            return sorted(
                line.strip()
                for line in self.config.tests_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        test_files = sorted((self.context.repository / "tests/e2e/models").glob("*/test_*_e2e.py"))
        if not test_files:
            raise CiError("No model E2E collection files found.")
        selection = (
            ["--e2e-models-file", self.config.models_file] if self.config.models_file else []
        )
        collected = self.context.output(
            [
                self.config.hf_python,
                "-m",
                "pytest",
                *test_files,
                "--co",
                "-q",
                *self.config.filter_args,
                *self.config.collect_args,
                *selection,
            ]
        )
        tests = sorted(line for line in collected.splitlines() if "test_model_e2e[" in line)
        if self.config.models_file:
            if self.config.tests_file:
                print("  Scheduler source: models file; tests file kept as impact record")
            else:
                print("  Scheduler source: models file")
            selected = {
                self._model_name(line)
                for line in self.config.models_file.read_text(encoding="utf-8").splitlines()
                if self._model_name(line)
            }
            tests = [test for test in tests if self._model_name(test) in selected]
        else:
            print("  Scheduler source: pytest collection")
        return tests

    @staticmethod
    def _model_name(raw: str) -> str:
        value = raw.split("#", 1)[0].strip()
        match = re.search(r"\[([^]]+)\]", value)
        return match.group(1) if match else value

    def _schedule(self) -> list[dict[str, object]]:
        timing_path = schedule_e2e._default_timing_estimates_path(self.config.manifest_dir)
        timing = schedule_e2e._load_timing_estimates(timing_path)
        phases = schedule_e2e.schedule_phases(
            self.test_ids,
            self.config.manifest_dir,
            len(self.gpu_ids),
            self.config.workers_per_gpu,
            timing_estimates=timing,
        )
        schedule_path = self.config.result_dir / "schedule.json"
        schedule_path.write_text(json.dumps({"phases": phases}, indent=2) + "\n", encoding="utf-8")
        workers = sum(
            1
            for phase in phases
            for queues in dict(phase["schedule"]).values()
            for tests in queues
            if tests
        )
        print(f"Workers planned: {workers}")
        return phases

    def _launch(self, logical_gpu: int, phase: str, index: int, tests: list[str]) -> E2EWorker:
        label = f"gpu{logical_gpu}-{phase}-w{index}"
        log_path = self.config.result_dir / f"console-{label}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        selection = (
            ["--e2e-models-file", self.config.models_file] if self.config.models_file else []
        )
        command = [
            str(self.config.hf_python),
            "-m",
            "pytest",
            *tests,
            "-v",
            "--engine-dir",
            str(self.config.engine_dir),
            "--trtmc-binary",
            str(self.config.trtmc_binary),
            "--hf-python",
            str(self.config.hf_python),
            "--e2e-artifacts-dir",
            str(self.config.result_dir / "artifacts"),
            f"--junitxml={self.config.result_dir / f'junit-{label}.xml'}",
            *selection,
            *self.config.collect_args,
            *self.config.pytest_args,
        ]
        environment = dict(self.context.env)
        environment["CUDA_VISIBLE_DEVICES"] = str(self.gpu_ids[logical_gpu])
        process = subprocess.Popen(
            command,
            cwd=self.context.repository,
            env=environment,
            text=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        worker = E2EWorker(
            logical_gpu, self.gpu_ids[logical_gpu], phase, index, tests, process, log_handle
        )
        self.workers.append(worker)
        print(f"  {worker.label}: {len(tests)} entries")
        return worker

    def _run_phase(self, phase: dict[str, object]) -> int:
        name = str(phase["name"])
        print(f"=== E2E phase: {name} ===")
        active = [
            self._launch(int(gpu), name, index, tests)
            for gpu, queues in dict(phase["schedule"]).items()
            for index, tests in enumerate(queues)
            if tests
        ]
        return self._wait(active)

    def _run_pipelined(self, phases: list[dict[str, object]]) -> None:
        print("=== E2E pipelined phases: exclusive_gpu -> shared ===")
        exclusive = dict(phases[0]["schedule"])
        shared = dict(phases[1]["schedule"])
        active: list[E2EWorker] = []
        shared_started: set[int] = set()

        def launch_shared(gpu: int) -> None:
            if gpu in shared_started:
                return
            shared_started.add(gpu)
            for index, tests in enumerate(shared.get(str(gpu), [])):
                if tests:
                    active.append(self._launch(gpu, "shared", index, tests))

        exclusive_gpus = {int(gpu) for gpu in exclusive}
        for gpu, queues in exclusive.items():
            for index, tests in enumerate(queues):
                if tests:
                    active.append(self._launch(int(gpu), "exclusive_gpu", index, tests))
        for gpu in map(int, shared):
            if gpu not in exclusive_gpus:
                launch_shared(gpu)

        last_progress = 0.0
        while active:
            self._check_timeout()
            for worker in list(active):
                rc = worker.process.poll()
                if rc is None:
                    continue
                worker.log_handle.close()
                active.remove(worker)
                self._record_result(worker, rc)
                if worker.phase == "exclusive_gpu":
                    launch_shared(worker.logical_gpu)
            now = time.time()
            if now - last_progress >= self.config.progress_interval or not active:
                self._print_progress(len(active))
                last_progress = now
            if active:
                time.sleep(5)

    def _wait(self, active: list[E2EWorker]) -> int:
        failures_before = self.failures
        last_progress = 0.0
        while active:
            self._check_timeout()
            for worker in list(active):
                rc = worker.process.poll()
                if rc is None:
                    continue
                worker.log_handle.close()
                active.remove(worker)
                self._record_result(worker, rc)
            now = time.time()
            if now - last_progress >= self.config.progress_interval or not active:
                self._print_progress(len(active))
                last_progress = now
            if active:
                time.sleep(5)
        return self.failures - failures_before

    def _check_timeout(self) -> None:
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise CiError(f"E2E timed out after {self.config.timeout_seconds} seconds")

    def _terminate_workers(self) -> None:
        running = [worker for worker in self.workers if worker.process.poll() is None]
        for worker in running:
            worker.process.terminate()
        deadline = time.monotonic() + 10
        for worker in running:
            try:
                worker.process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                worker.process.kill()
                worker.process.wait()
            worker.log_handle.close()

    def _record_result(self, worker: E2EWorker, returncode: int) -> None:
        if returncode:
            self.failures += 1
            print(f"  {worker.label}: FAILED (exit code {returncode})")
        else:
            print(f"  {worker.label}: OK")
        summaries = [
            line
            for line in (self.config.result_dir / f"console-{worker.label}.log")
            .read_text(encoding="utf-8", errors="replace")
            .splitlines()
            if re.match(r"^=+ .* in .* =+$", line)
        ]
        if summaries:
            print(f"    {summaries[-1]}")

    def _progress(self) -> dict[str, int]:
        statuses: dict[str, str] = {}
        for path in self.config.result_dir.glob("console-gpu*-w*.log"):
            for match in self.RESULT_PATTERN.finditer(
                path.read_text(encoding="utf-8", errors="replace")
            ):
                statuses[match.group("node")] = match.group("status")
        counts = {name: 0 for name in ("pass", "fail", "skip", "xfail", "xpass")}
        for status in statuses.values():
            key = {
                "PASSED": "pass",
                "FAILED": "fail",
                "ERROR": "fail",
                "SKIPPED": "skip",
                "XFAIL": "xfail",
                "XPASS": "xpass",
            }[status]
            counts[key] += 1
        counts["done"] = len(statuses)
        return counts

    def _print_progress(self, running: int) -> None:
        counts = self._progress()
        elapsed = int(time.time() - self.started_at)
        percent = 100 * counts["done"] / len(self.test_ids)
        print(
            f"[progress {time.strftime('%H:%M:%S')}] entries {counts['done']}/{len(self.test_ids)} "
            f"({percent:.1f}%) pass={counts['pass']} fail={counts['fail']} skip={counts['skip']} "
            f"xfail={counts['xfail']} xpass={counts['xpass']} | {running} workers running | "
            f"elapsed {elapsed // 60}m {elapsed % 60:02d}s"
        )

    def _merge_junit(self) -> None:
        try:
            from junitparser import JUnitXml
        except ImportError:
            print("(install junitparser to auto-merge: pip install junitparser)")
            return
        files = sorted(self.config.result_dir.glob("junit-gpu*-w*.xml"))
        if not files:
            print("No JUnit XML files found to merge.")
            return
        merged = JUnitXml()
        for path in files:
            try:
                merged += JUnitXml.from_file(str(path))
            except Exception as error:  # corrupt worker output should not hide original failure
                print(f"Warning: could not parse {path}: {error}")
        destination = self.config.result_dir / "junit.xml"
        merged.write(str(destination))
        print(f"Merged {len(files)} files -> {destination}")

    def _write_timing_summary(self) -> None:
        artifacts = self.config.result_dir / "artifacts"
        if not artifacts.is_dir():
            return
        keys = (
            "bundle_build_s",
            "trt_compile_s",
            "trt_load_deserialization_s",
            "reference_s",
            "inference_s",
            "comparison_s",
        )
        aggregate = {key: 0.0 for key in keys}
        rows = []
        for path in sorted(artifacts.glob("*/result.json")):
            try:
                result = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            timing, detailed = result.get("timing") or {}, result.get("detailed_timing") or {}
            row = {
                "case": result.get("case_name") or path.parent.name,
                "status": result.get("status") or "",
                "bundle_build_s": float(timing.get("bundle_build_s") or 0),
                **{key: float(detailed.get(key) or 0) for key in keys if key != "bundle_build_s"},
            }
            row["accounted_s"] = sum(row[key] for key in keys if key != "trt_compile_s")
            rows.append(row)
            for key in keys:
                aggregate[key] += row[key]
        if rows:
            (self.config.result_dir / "timing-summary.json").write_text(
                json.dumps({"aggregate_s": aggregate, "cases": rows}, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )

    def _print_summary(self) -> None:
        print("\nOutput files:")
        print(f"  Schedule:      {self.config.result_dir / 'schedule.json'}")
        print(f"  Console logs:  {self.config.result_dir}/console-gpu*-w*.log")
        print(f"  JUnit XML:     {self.config.result_dir}/junit-gpu*-w*.xml")
        print(f"  Timing JSON:   {self.config.result_dir / 'timing-summary.json'}")
        print(f"  Artifacts:     {self.config.result_dir / 'artifacts'}/")
        print("\n--- Per-test results ---")
        for worker in self.workers:
            log = self.config.result_dir / f"console-{worker.label}.log"
            print(f"\n  [{worker.label}]")
            for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("tests/") and re.search(r"PASSED|FAILED|SKIPPED|ERROR", line):
                    print(f"    {line}")
