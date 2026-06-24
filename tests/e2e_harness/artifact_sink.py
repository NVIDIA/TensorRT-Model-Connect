"""Artifact sink — persist E2E outputs, commands, and results.

Implements the ArtifactSink protocol from contracts.py.  All per-model
test information is written to a single ``result.json`` plus a single
``e2e_run.log`` for subprocess output.  Modality artifacts (PNG, WAV,
NPY) are written directly into the model directory.

Eliminated files (compared to the previous layout):
    case.json, env_fingerprint.json, commands.json,
    stages/ directory, logs/ directory.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import (
    CompareResult,
    E2ECase,
    E2EResult,
    RunContext,
    StageOutput,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Artifact schema definitions
# ---------------------------------------------------------------------------

# Expected fields per ArtifactType for schema validation.
# Each key is an ArtifactType value; each value is a dict of
# field_name -> (required: bool, type_description: str).
ARTIFACT_SCHEMAS: dict[str, dict[str, tuple[bool, str]]] = {
    "token_ids": {
        "token_ids": (True, "list[int]"),
        "text": (False, "str"),
    },
    "normalized_text": {
        "text": (True, "str"),
        "normalized_text": (True, "str"),
    },
    "embedding": {
        "embedding": (True, "list[float] or numpy path"),
        "pooling_method": (False, "str"),
    },
    "class_map": {
        "class_map": (True, "numpy array or path"),
        "class_ids": (False, "list[int]"),
        "num_classes": (False, "int"),
    },
    "semantic_tokens": {
        "semantic_token_ids": (True, "list[int]"),
    },
    "waveform": {
        "wav_path": (True, "str"),
        "sample_rate": (False, "int"),
        "duration_s": (False, "float"),
        "rms": (False, "float"),
    },
    "image": {
        "image_path": (True, "str"),
        "width": (False, "int"),
        "height": (False, "int"),
    },
    "video_frames": {
        "frames_dir": (True, "str"),
        "num_frames": (False, "int"),
        "fps": (False, "float"),
    },
    "logits": {
        "logits_path": (True, "str"),
        "vocab_size": (False, "int"),
    },
    "score": {
        "score": (True, "float"),
    },
    "ranking": {
        "ranking": (True, "list[int] or list[tuple]"),
        "scores": (False, "list[float]"),
    },
}


def validate_artifact_schema(
    artifact_type: str, data: dict[str, Any]
) -> list[str]:
    """Validate stage output data against the expected schema for an artifact type.

    Returns a list of validation error messages (empty if valid).
    """
    schema = ARTIFACT_SCHEMAS.get(artifact_type)
    if schema is None:
        return []  # Unknown type -- no validation

    errors: list[str] = []
    for field_name, (required, type_desc) in schema.items():
        if required and field_name not in data:
            errors.append(
                f"Missing required field '{field_name}' (expected {type_desc})"
            )
    return errors


def _probe_python_environment(python: str) -> dict[str, Any]:
    """Collect package/version info for a specific Python interpreter."""
    if not python:
        return {}

    script = r"""
import importlib
import json
import sys

result = {
    "python_executable": sys.executable,
    "python_version": sys.version,
}
for module_name in ("torch", "transformers", "diffusers", "tensorrt"):
    try:
        module = importlib.import_module(module_name)
        result[f"{module_name}_version"] = getattr(module, "__version__", "unknown")
    except Exception:
        continue
print(json.dumps(result))
"""
    try:
        result = subprocess.run(
            [python, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return {"requested_python": python, "probe_error": result.stderr.strip()}
        payload = json.loads(result.stdout)
        payload["requested_python"] = python
        return payload
    except Exception as exc:
        return {"requested_python": python, "probe_error": str(exc)}


def _collect_env_fingerprint(ctx: RunContext | None = None) -> dict[str, Any]:
    """Collect environment fingerprint: GPU, CUDA, TRT, Python versions."""
    fp: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "hostname": platform.node(),
    }

    # CUDA version
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "release" in line.lower():
                    fp["cuda_version"] = line.strip()
                    break
    except Exception:
        fp["cuda_version"] = "unknown"

    # GPU info
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            gpu_lines = result.stdout.strip().splitlines()
            if gpu_lines:
                parts = gpu_lines[0].split(", ")
                fp["gpu_name"] = parts[0] if len(parts) > 0 else "unknown"
                fp["driver_version"] = parts[1] if len(parts) > 1 else "unknown"
                fp["compute_capability"] = parts[2] if len(parts) > 2 else "unknown"
                fp["gpu_count"] = len(gpu_lines)
    except Exception:
        fp["gpu_name"] = "unknown"

    # TensorRT version
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "import tensorrt; print(tensorrt.__version__)"],
            capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            fp["tensorrt_version"] = result.stdout.strip()
    except Exception:
        fp["tensorrt_version"] = "unknown"

    # Torch version
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "import torch; print(torch.__version__)"],
            capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            fp["torch_version"] = result.stdout.strip()
    except Exception:
        fp["torch_version"] = "unknown"

    # Transformers version
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "import transformers; print(transformers.__version__)"],
            capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            fp["transformers_version"] = result.stdout.strip()
    except Exception:
        fp["transformers_version"] = "unknown"

    if ctx is not None:
        fp["python_profiles"] = {
            "build": {
                "profile": ctx.build_profile,
                **_probe_python_environment(ctx.build_python_path()),
            },
            "runtime": {
                "profile": ctx.runtime_profile,
                **_probe_python_environment(ctx.runtime_python_path()),
            },
            "reference": {
                "profile": ctx.reference_profile,
                **_probe_python_environment(ctx.reference_python_path()),
            },
        }

    return fp


def _case_to_dict(case: E2ECase) -> dict[str, Any]:
    """Serialize an E2ECase to a plain dict for JSON storage."""
    return {
        "name": case.name,
        "hf_id": case.hf_id,
        "family": case.family,
        "runtime_strategy": case.runtime_strategy,
        "task_strategy": case.task_strategy,
        "reference_backend": case.reference_backend,
        "oracle_level": case.oracle_level,
        "reference_family": case.reference_family,
        "user_contract": case.user_contract,
        "ci_lane": case.ci_lane,
        "bundle": case.bundle,
        "inputs": case.inputs,
        "preflight": [
            {"kind": p.kind, "args": p.args, "gating": p.gating}
            for p in case.preflight
        ],
        "stages": [
            {
                "name": s.name,
                "required": s.required,
                "runner_override": s.runner_override,
                "comparator_override": s.comparator_override,
                "artifact_type": s.artifact_type,
                "comparison_mode": s.comparison_mode,
                "ci_lanes": s.ci_lanes,
            }
            for s in case.stages
        ],
        "comparison_profile": case.comparison_profile,
        "threshold_overrides": case.threshold_overrides,
        "determinism": case.determinism,
        "execution_profiles": case.execution_profiles,
        "metadata": case.metadata,
    }


def _serialize_stage_output(output: StageOutput) -> dict[str, Any]:
    """Serialize a StageOutput to a JSON-safe dict (skip non-serializable)."""
    data: dict[str, Any] = {
        "stage_name": output.stage_name,
        "timing_s": output.timing_s,
        "metadata": output.metadata,
    }
    if output.text is not None:
        data["text"] = output.text

    serializable_data: dict[str, Any] = {}
    for k, v in output.data.items():
        try:
            json.dumps(v)
            serializable_data[k] = v
        except (TypeError, ValueError):
            serializable_data[k] = f"<non-serializable: {type(v).__name__}>"
    data["data"] = serializable_data
    return data


class FileArtifactSink:
    """File-based artifact sink — consolidated single-file output.

    Directory layout:
        <artifacts_dir>/<model_name>/
            result.json         — single consolidated result with ALL info
            e2e_run.log         — merged subprocess output
            frames/frame_*.png  — modality artifacts (unchanged)
            *.wav               — modality artifacts (unchanged)
            *_logits.npy        — modality artifacts (unchanged)
    """

    def __init__(self, artifacts_dir: str | Path, case: E2ECase) -> None:
        self._base = Path(artifacts_dir) / case.name
        self._base.mkdir(parents=True, exist_ok=True)

        self._commands: list[dict[str, Any]] = []
        self._stage_outputs: dict[str, dict[str, Any]] = {}
        self._artifacts: dict[str, Any] = {}
        self._case = case
        self._env_fp: dict[str, Any] | None = None
        self._log_path = self._base / "e2e_run.log"

        # Truncate log file at start of run
        self._log_path.write_text("", encoding="utf-8")

    @property
    def base_dir(self) -> Path:
        """Root directory for this model's artifacts."""
        return self._base

    def ensure_env_fingerprint(self, ctx: RunContext | None = None) -> dict[str, Any]:
        """Collect and cache environment fingerprint (in memory only)."""
        if self._env_fp is None:
            self._env_fp = _collect_env_fingerprint(ctx)
        return self._env_fp

    def log_command(
        self,
        command: list[str],
        rc: int,
        stdout: str,
        stderr: str,
        label: str = "",
    ) -> None:
        """Append subprocess output to e2e_run.log with section headers."""
        if label:
            safe_label = label.replace("/", "_").replace(" ", "_")
        else:
            safe_label = f"cmd_{len(self._commands) + 1}"

        entry = {
            "command": command,
            "returncode": rc,
            "stdout_len": len(stdout),
            "stderr_len": len(stderr),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._commands.append(entry)

        # Append to single merged log file
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'=' * 72}\n")
            f.write(f"[{safe_label}] rc={rc}\n")
            cmd_str = " ".join(command) if command else "(no command)"
            f.write(f"$ {cmd_str}\n")
            f.write(f"{'=' * 72}\n")
            if stdout:
                f.write("--- stdout ---\n")
                f.write(stdout)
                if not stdout.endswith("\n"):
                    f.write("\n")
            if stderr:
                f.write("--- stderr ---\n")
                f.write(stderr)
                if not stderr.endswith("\n"):
                    f.write("\n")

    def write_stage_output(
        self,
        stage_name: str,
        output: StageOutput,
        prefix: str = "trt",
        artifact_type: str = "",
    ) -> None:
        """Accumulate stage output in memory (written in finalize).

        If artifact_type is provided, validates the output data against
        the expected schema and logs warnings for missing fields.
        """
        key = f"{prefix}_{stage_name}"
        serialized = _serialize_stage_output(output)

        # Schema validation
        if artifact_type:
            errors = validate_artifact_schema(artifact_type, output.data)
            if errors:
                serialized["schema_warnings"] = errors
                logger.warning(
                    "Stage %s (%s) artifact schema warnings: %s",
                    stage_name, artifact_type, errors,
                )

        self._stage_outputs[key] = serialized

    def write_compare(
        self,
        stage_name: str,
        result: CompareResult,
    ) -> None:
        """No-op — comparison data is already in E2EResult.stages."""
        pass

    def write_golden(
        self,
        stage_name: str,
        output: StageOutput,
        artifact_type: str = "",
    ) -> str:
        """Write a stage output as a golden reference artifact.

        Golden artifacts are stored in a 'goldens/' subdirectory and contain
        the canonical expected output for acceptance CI lane comparisons.

        Returns the path to the golden file.
        """
        goldens_dir = self._base / "goldens"
        goldens_dir.mkdir(parents=True, exist_ok=True)

        golden_data = _serialize_stage_output(output)
        golden_data["artifact_type"] = artifact_type
        golden_data["golden_timestamp"] = datetime.now(timezone.utc).isoformat()

        # Validate schema if artifact_type is specified
        if artifact_type:
            errors = validate_artifact_schema(artifact_type, output.data)
            if errors:
                golden_data["schema_warnings"] = errors
                logger.warning(
                    "Golden artifact for %s has schema warnings: %s",
                    stage_name, errors,
                )

        path = goldens_dir / f"{stage_name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(golden_data, f, indent=2, default=str)

        self.register_artifact(
            f"golden_{stage_name}", str(path.relative_to(self._base))
        )
        return str(path)

    def load_golden(self, stage_name: str) -> dict[str, Any] | None:
        """Load a golden reference artifact for a stage.

        Returns the golden data dict, or None if no golden exists.
        """
        path = self._base / "goldens" / f"{stage_name}.json"
        if not path.is_file():
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def register_artifact(self, key: str, rel_path: str) -> None:
        """Register a modality artifact by key and relative path."""
        if key in self._artifacts:
            existing = self._artifacts[key]
            if isinstance(existing, list):
                existing.append(rel_path)
            else:
                self._artifacts[key] = [existing, rel_path]
        else:
            self._artifacts[key] = rel_path

    def finalize(self, result: E2EResult) -> str:
        """Inject accumulated data into E2EResult and write result.json."""
        from .result_schema import serialize_result

        # Inject case config
        result.case_config = _case_to_dict(self._case)

        # Inject environment fingerprint
        if self._env_fp:
            result.env_fingerprint = self._env_fp

        # Inject commands log
        result.commands = self._commands

        # Inject stage outputs
        result.stage_outputs = self._stage_outputs

        # Inject artifact references
        result.artifacts = self._artifacts

        # Inject log file reference (relative to model dir)
        result.log_file = "e2e_run.log"

        path = self._base / "result.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serialize_result(result), f, indent=2, default=str)
        return str(path)
