# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Choose and run the selective or nightly E2E phases.

Boundary: high-level E2E policy; worker scheduling and model isolation are delegated.
"""

from __future__ import annotations

import re
from pathlib import Path

from .context import CiContext
from .e2e_scheduler import E2EParallelConfig, E2EParallelRunner
from .isolation import IsolatedModelRunner
from .package import WheelPackageManager
from .process import CiError


class E2ERunner:
    """Own the cache warm, standard model proof, strict isolation, and VLM check."""

    def __init__(self, context: CiContext):
        self.context = context
        self.package = WheelPackageManager(context)

    def selective(self) -> None:
        if self.context.env.get("GITHUB_EVENT_NAME") != "pull_request" or self._full_requested():
            print("Skipping: selective E2E only runs for pull_request events without full_e2e")
            return
        impact = self.context.read_json("impact.json")
        models = {str(model) for model in impact.get("e2e_models", [])}
        test_ids = [str(item) for item in impact.get("e2e_test_ids", [])]
        for test_id in test_ids:
            match = re.search(r"::test_model_e2e\[([^]]+)\]", test_id)
            if match:
                models.add(match.group(1))
        models = sorted(model for model in models if model)
        self._write_lines("e2e_models.txt", models)
        self._write_lines("e2e_test_ids.txt", test_ids)
        print(f"Selective E2E: {len(models)} models")
        for model in models[:10]:
            print(f"  {model}")
        isolation_models = self.context.output(
            [
                "python3",
                "tools/model_plugin_isolation.py",
                "impact-models",
                "--impact-json",
                "impact.json",
                "--exclude-ci-tier",
                "multi_device",
                "--exclude-ci-tier",
                "nightly_only",
            ]
        ).splitlines()
        self._write_lines("e2e_isolation_models.txt", isolation_models)
        print(f"Model-owned isolation E2E: {len(isolation_models)} models")
        if not models:
            print("No E2E models affected by this change -- skipping E2E tests")
            (self.context.repository / "e2e_artifacts/artifacts").mkdir(parents=True, exist_ok=True)
            return

        self.context.env.setdefault("TRTMC_BUILDER_OPTIMIZATION_LEVEL", "1")
        self._configure_timing_cache()
        print("=== Phase 1: warming HF cache (online, sequential) ===")
        self.context.run(
            [
                "python",
                "scripts/warm_hf_cache.py",
                "--models-file",
                "e2e_models.txt",
                "--exclude-ci-tier",
                "multi_device",
                "--exclude-ci-tier",
                "nightly_only",
            ],
            unset=("HF_HUB_OFFLINE",),
        )
        print("=== Phase 2: standard selective E2E for the full conservative impact set ===")
        result_dir = self.context.repository / "e2e_artifacts"
        self.context.remove(result_dir)
        (result_dir / "artifacts").mkdir(parents=True)
        plugins = result_dir / "model_plugins"
        self._prepare_plugins(plugins, ["--models-file", "e2e_models.txt"])
        arguments = [
            "--engine-dir",
            self.context.env["ENGINE_DIR"],
            "--result-dir",
            str(result_dir),
            "--trtmc-binary",
            self.context.executable("trtmc"),
            "--workers-per-gpu",
            "4",
            "--models-file",
            "e2e_models.txt",
            "--exclude-ci-tier",
            "nightly_only",
            "--model-plugin-dir",
            str(plugins),
            "--timeout",
            self.context.env.get(
                "SELECTIVE_E2E_STANDARD_TIMEOUT",
                self.context.env.get("SELECTIVE_E2E_TIMEOUT", "4h"),
            ),
        ]
        if test_ids:
            arguments.extend(["--tests-file", "e2e_test_ids.txt"])
        if self.context.env.get("REBUILD_ENGINES", "true") == "true":
            arguments.append("--rebuild-engines")
        config = E2EParallelConfig.parse(arguments, self.context.env)
        standard_rc = E2EParallelRunner(self.context, config).run()
        if standard_rc:
            raise CiError(f"standard selective E2E failed with code {standard_rc}")
        if isolation_models:
            print("=== Phase 3: strict model-owned isolation E2E ===")
            IsolatedModelRunner(self.context).run(
                self.context.repository / "e2e_isolation_models.txt", result_dir
            )
        else:
            print("No model-owned E2E cases changed -- strict isolation rerun not required")
        self.diffusion_vlm_assessment()

    def full(self) -> None:
        if not self._full_requested():
            print("Skipping: full E2E was not requested")
            return
        print("=== Nightly Full E2E: all models ===")
        self.context.run(["nvidia-smi"])
        self._configure_timing_cache()
        plugins = self.context.repository / "e2e_artifacts/model_plugins"
        self._prepare_plugins(plugins, ["--all"])
        print("=== Phase 1: warming HF cache (online, sequential) ===")
        self.context.run(
            [
                "python",
                "scripts/warm_hf_cache.py",
                "--exclude-ci-tier",
                "l0_only",
                "--exclude-ci-tier",
                "multi_device",
            ]
        )
        arguments = [
            "--engine-dir",
            self.context.env["ENGINE_DIR"],
            "--result-dir",
            "e2e_artifacts",
            "--trtmc-binary",
            self.context.executable("trtmc"),
            "--workers-per-gpu",
            "4",
            "--exclude-ci-tier",
            "l0_only",
            "--exclude-ci-tier",
            "multi_device",
            "--model-plugin-dir",
            str(plugins),
            "--timeout",
            self.context.env.get("FULL_E2E_TIMEOUT", "6h"),
        ]
        if self.context.env.get("REBUILD_ENGINES", "true") == "true":
            arguments.append("--rebuild-engines")
        standard_error: CiError | None = None
        try:
            standard_rc = E2EParallelRunner(
                self.context, E2EParallelConfig.parse(arguments, self.context.env)
            ).run()
        except CiError as error:
            standard_error = error
            standard_rc = 1
        try:
            self.diffusion_vlm_assessment()
        except Exception:
            if not standard_rc:
                raise
        if standard_error is not None:
            raise standard_error
        if standard_rc:
            raise CiError(f"E2E exited with code {standard_rc}")

    def diffusion_vlm_assessment(self) -> None:
        if self.context.env.get("DIFFUSION_VLM_ASSESSMENT", "true") != "true":
            print("Skipping: diffusion VLM assessment disabled")
            return
        artifacts = self.context.repository / "e2e_artifacts/artifacts"
        if not artifacts.is_dir():
            print("Skipping: no E2E artifacts directory for diffusion VLM assessment")
            return
        pair_count = int(
            self.context.output(["python3", "tools/count_diffusion_frame_pairs.py", artifacts])
            or "0"
        )
        if pair_count == 0:
            print("Skipping: no TRT/HF diffusion frame pairs for VLM assessment")
            return
        config_path, config = self._default_config(
            "DIFFUSION_VLM_CONFIG", "diffusion_vlm_assessment.json"
        )
        required = ("model_id", "max_side", "max_new_tokens", "timeout")
        missing = [key for key in required if not config.get(key)]
        if missing:
            raise CiError(f"{config_path} missing required diffusion VLM fields: {missing}")
        print(f"=== Phase 3: diffusion VLM semantic assessment ({pair_count} pairs) ===")
        print(f"Using diffusion VLM assessment config {config_path}")
        self.context.run(
            [
                "python",
                "tools/evaluate_diffusion_vlm_similarity.py",
                "--artifacts-dir",
                artifacts,
                "--output",
                "e2e_artifacts/diffusion_vlm_assessment.json",
                "--config",
                config_path,
                "--model-id",
                self.context.env.get("DIFFUSION_VLM_MODEL_ID", str(config["model_id"])),
                "--max-side",
                self.context.env.get("DIFFUSION_VLM_MAX_SIDE", str(config["max_side"])),
                "--max-new-tokens",
                self.context.env.get("DIFFUSION_VLM_MAX_NEW_TOKENS", str(config["max_new_tokens"])),
            ],
            limit=self.context.env.get("DIFFUSION_VLM_TIMEOUT", str(config["timeout"])),
            unset=("HF_HUB_OFFLINE",),
        )

    def _prepare_plugins(self, output: Path, arguments: list[str]) -> None:
        metadata = self.package.build_metadata()
        self.context.remove(output)
        output.mkdir(parents=True)
        self.context.run(
            [
                "python3",
                "tools/model_plugin_isolation.py",
                "prepare",
                "--build-dir",
                metadata["cmake_build_dir"],
                "--output-dir",
                output,
                *arguments,
            ]
        )

    def _configure_timing_cache(self) -> None:
        root = self.context.env.get("TRTMC_STORAGE_ROOT", self.context.env.get("ENGINE_DIR", "."))
        tensorrt_version = self.context.output(
            ["python", "-c", "import tensorrt; print(tensorrt.__version__)"]
        )
        version = re.sub(r"[^0-9A-Za-z_.-]+", "-", tensorrt_version).strip("-")
        if not version:
            raise CiError("TensorRT reported an empty version for timing-cache isolation")
        suffix = (
            f"trt{version}-"
            f"opt{self.context.env.get('TRTMC_BUILDER_OPTIMIZATION_LEVEL', 'default')}"
        )
        if self.context.env.get("TRTMC_MAX_NUM_TACTICS"):
            suffix += f"-tactics{self.context.env['TRTMC_MAX_NUM_TACTICS']}"
        if self.context.env.get("TRTMC_AVG_TIMING_ITERATIONS"):
            suffix += f"-avg{self.context.env['TRTMC_AVG_TIMING_ITERATIONS']}"
        cache = self.context.env.setdefault(
            "TRTMC_TRT_TIMING_CACHE_PATH",
            f"{root.rstrip('/')}/trt-timing-cache/tensorrt-{suffix}.cache",
        )
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        print(f"TRTMC_TRT_TIMING_CACHE_PATH={cache}")

    def _default_config(self, variable: str, filename: str) -> tuple[Path, dict[str, object]]:
        requested = self.context.env.get(variable)
        if requested:
            path = Path(requested)
            if not path.is_file():
                raise CiError(f"{variable} does not exist: {path}")
        else:
            configs = sorted((self.context.repository / "tests/e2e/models").glob(f"*/{filename}"))
            defaults = [
                path for path in configs if self.context.read_json(path).get("default") is True
            ]
            if len(defaults) != 1:
                raise CiError(f"Expected exactly one default {filename}; found {defaults}")
            path = defaults[0]
        return path, self.context.read_json(path)

    def _write_lines(self, path: str, lines: list[str]) -> None:
        (self.context.repository / path).write_text(
            "".join(f"{line}\n" for line in lines), encoding="utf-8"
        )

    def _full_requested(self) -> bool:
        return self.context.env.get("FULL_E2E", "false") == "true"
