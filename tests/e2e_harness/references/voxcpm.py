"""VoxCPM reference backend for openbmb/VoxCPM2.

Runs the official ``voxcpm`` library in a subprocess and preserves the model
card TTS output as a WAV artifact for exact TRT comparison.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

from .. import _case_artifact_dir, save_full_stderr
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[3]


def _input_bool(case: E2ECase, name: str, default: bool) -> bool:
    value = case.inputs.get(name, default)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _shared_locdit_noise_path(case: E2ECase, ctx: RunContext) -> Path | None:
    if case.family != "voxcpm2" or not ctx.artifacts_dir:
        return None
    path = Path(_case_artifact_dir(ctx.artifacts_dir, case.name)) / "locdit_noise.raw"
    return path if path.is_file() else None


class VoxCPMReference:
    """Reference backend for VoxCPM2 via the official ``voxcpm`` library."""

    @property
    def backend_name(self) -> str:
        return "voxcpm"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        if case.task_strategy != "text_to_audio":
            raise ValueError(
                "VoxCPM reference only supports text_to_audio, "
                f"got {case.task_strategy!r}"
            )
        return self._run_text_to_audio_ref(case, stage, ctx)

    def _run_text_to_audio_ref(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        model_id = case.hf_id
        prompt = case.inputs.get("prompt", "Hello, this is a test.")
        prompt_wav_path = case.inputs.get("prompt_wav_path")
        prompt_text = case.inputs.get("prompt_text")
        cfg_value = float(case.inputs.get("cfg_value", 2.0))
        inference_timesteps = int(case.inputs.get("inference_timesteps", 10))
        normalize = _input_bool(case, "normalize", True)
        denoise = _input_bool(case, "denoise", True)
        retry_badcase = _input_bool(case, "retry_badcase", True)
        retry_badcase_max_times = int(case.inputs.get("retry_badcase_max_times", 3))
        retry_badcase_ratio_threshold = float(
            case.inputs.get("retry_badcase_ratio_threshold", 6.0)
        )
        seed = int(case.inputs.get("seed", -1))

        artifacts_dir = ctx.artifacts_dir or tempfile.mkdtemp(prefix="trtmc_voxcpm_ref_")
        case_dir = Path(_case_artifact_dir(artifacts_dir, case.name))
        wav_path = str(case_dir / "hf_reference.wav")
        json_path = str(case_dir / "hf_reference_result.json")
        python = ctx.reference_python_path() or sys.executable

        script = textwrap.dedent(
            """\
            import json
            import math
            import os
            import sys

            import numpy as np
            import soundfile as sf

            seed = %(seed)d
            if seed >= 0:
                np.random.seed(seed)
                try:
                    import torch
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                except Exception:
                    pass

            from voxcpm import VoxCPM

            model = VoxCPM.from_pretrained(%(model_id)r)
            if os.environ.get("TRTMC_VOXCPM2_HF_TENSOR_DUMP_DIR"):
                sys.path.insert(0, %(repo_root)r)
                from tests.e2e_harness.references.voxcpm_debug import install_voxcpm2_tensor_dump
                install_voxcpm2_tensor_dump(model)
            if seed >= 0:
                np.random.seed(seed)
                try:
                    import torch
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                except Exception:
                    pass
            wav = model.generate(
                text=%(prompt)r,
                prompt_wav_path=%(prompt_wav_path)r,
                prompt_text=%(prompt_text)r,
                cfg_value=%(cfg_value)r,
                inference_timesteps=%(inference_timesteps)d,
                normalize=%(normalize)r,
                denoise=%(denoise)r,
                retry_badcase=%(retry_badcase)r,
                retry_badcase_max_times=%(retry_badcase_max_times)d,
                retry_badcase_ratio_threshold=%(retry_badcase_ratio_threshold)r,
            )
            audio = np.asarray(wav, dtype=np.float32).reshape(-1)
            sample_rate = int(getattr(getattr(model, "tts_model", model), "sample_rate", 48000))
            sf.write(%(wav_path)r, audio, sample_rate, subtype="FLOAT")

            rms = float(math.sqrt(float(np.mean(np.square(audio))))) if audio.size else 0.0
            result = {
                "num_samples": int(audio.size),
                "rms": rms,
                "duration_s": float(audio.size / sample_rate) if sample_rate else 0.0,
                "sample_rate": sample_rate,
                "wav_path": %(wav_path)r,
                "cfg_value": %(cfg_value)r,
                "inference_timesteps": %(inference_timesteps)d,
                "seed": seed,
            }
            with open(%(json_path)r, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, sort_keys=True)
            print(json.dumps(result, sort_keys=True))
            """
            % {
                "seed": seed,
                "model_id": model_id,
                "prompt": prompt,
                "prompt_wav_path": prompt_wav_path,
                "prompt_text": prompt_text,
                "cfg_value": cfg_value,
                "inference_timesteps": inference_timesteps,
                "normalize": normalize,
                "denoise": denoise,
                "retry_badcase": retry_badcase,
                "retry_badcase_max_times": retry_badcase_max_times,
                "retry_badcase_ratio_threshold": retry_badcase_ratio_threshold,
                "wav_path": wav_path,
                "json_path": json_path,
                "repo_root": str(REPO_ROOT),
            }
        )

        env = os.environ.copy()
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path
        shared_noise_path = _shared_locdit_noise_path(case, ctx)
        if shared_noise_path is not None and env.get("TRTMC_VOXCPM2_HF_TENSOR_DUMP_DIR"):
            env["TRTMC_VOXCPM2_HF_NOISE_RAW"] = str(shared_noise_path)

        t0 = time.monotonic()
        try:
            result = subprocess.run(
                [python, "-c", script],
                capture_output=True,
                text=True,
                timeout=1800,
                env=env,
            )
            elapsed = time.monotonic() - t0
        except subprocess.TimeoutExpired:
            return StageOutput(
                stage_name=stage.name,
                data={"error": "VoxCPM reference timed out", "returncode": -1},
                timing_s=1800.0,
                metadata={"backend": "voxcpm", "returncode": -1},
            )

        stderr_truncated, stderr_log = save_full_stderr(
            result.stderr or "", artifacts_dir, "voxcpm_ref", case.name
        )
        data: dict = {
            "returncode": result.returncode,
            "stderr_truncated": stderr_truncated,
            "result_json_path": json_path,
        }
        if shared_noise_path is not None and env.get("TRTMC_VOXCPM2_HF_NOISE_RAW"):
            data["locdit_noise_raw"] = str(shared_noise_path)
        if stderr_log:
            data["stderr_log"] = stderr_log

        if result.returncode == 0:
            try:
                parsed = json.loads(result.stdout.strip().splitlines()[-1])
                data.update(parsed)
            except Exception as exc:
                logger.warning("Failed to parse VoxCPM ref output: %s", exc)
                data["parse_error"] = str(exc)
                data["stdout"] = result.stdout
        else:
            data["stdout"] = result.stdout
            data["stderr"] = result.stderr

        return StageOutput(
            stage_name=stage.name,
            data=data,
            timing_s=elapsed,
            metadata={
                "backend": "voxcpm",
                "returncode": result.returncode,
                "command": [python, "-c", "<voxcpm_ref_script>"],
            },
        )


plugin = VoxCPMReference()
