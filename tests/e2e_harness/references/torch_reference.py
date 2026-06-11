"""Torch reference backend for custom models and golden snapshots.

Provides reference outputs for models that do not have a standard HF pipeline:
- PersonaPlex: loads reference tokens from .npy files
- Custom torch models: executes user-provided reference scripts
- Golden snapshots: loads pre-saved reference outputs

Used primarily for speech_to_speech (PersonaPlex) and models without
HF Transformers/Diffusers integration.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from .. import save_full_stderr
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec

logger = logging.getLogger(__name__)

class TorchReference:
    """Reference backend using custom torch code or golden .npy snapshots."""

    @property
    def backend_name(self) -> str:
        return "torch_reference"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        # Dispatch based on task_strategy + stage
        task = case.task_strategy
        if task == "speech_to_speech":
            return self._run_speech_to_speech_ref(case, stage, ctx)
        if task == "text_to_audio":
            return self._run_text_to_audio_ref(case, stage, ctx)
        if task == "speech_to_text":
            return self._run_speech_to_text_ref(case, stage, ctx)
        # Generic: try to load golden snapshot
        return self._run_golden_snapshot(case, stage, ctx)

    def _run_speech_to_speech_ref(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Load reference tokens from .npy file for PersonaPlex-style models."""
        ref_tokens_path = case.inputs.get(
            "speech_reference_tokens",
            case.metadata.get("speech_reference_tokens", ""),
        )
        if not ref_tokens_path:
            return StageOutput(
                stage_name=stage.name,
                data={"error": "No speech_reference_tokens path in manifest"},
            )

        # Resolve relative path against project/tests/e2e directory
        if not os.path.isabs(ref_tokens_path):
            e2e_dir = Path(__file__).resolve().parents[2] / "e2e"
            ref_tokens_path = str(e2e_dir / ref_tokens_path)

        if not os.path.exists(ref_tokens_path):
            return StageOutput(
                stage_name=stage.name,
                data={"error": f"Reference tokens file not found: {ref_tokens_path}"},
            )

        try:
            import numpy as np
            ref_tokens = np.load(ref_tokens_path)
        except Exception as e:
            return StageOutput(
                stage_name=stage.name,
                data={"error": f"Failed to load reference tokens: {e}"},
            )

        return StageOutput(
            stage_name=stage.name,
            data={
                "reference_tokens": ref_tokens,
                "num_frames": ref_tokens.shape[0] if ref_tokens.ndim >= 1 else 0,
                "token_shape": list(ref_tokens.shape),
                "source_path": ref_tokens_path,
            },
            metadata={"backend": "torch_reference"},
        )

    def _run_text_to_audio_ref(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run HF Bark pipeline as reference for text-to-audio."""
        model_id = case.hf_id
        prompt = case.inputs.get("prompt", "Hello, this is a test.")
        python = ctx.reference_python_path() or sys.executable

        script = f"""
import torch
import numpy as np
import json

from transformers import AutoProcessor, BarkModel

processor = AutoProcessor.from_pretrained({model_id!r})
model = BarkModel.from_pretrained({model_id!r})
model = model.to("cuda")

inputs = processor({prompt!r})
inputs = {{k: v.to("cuda") for k, v in inputs.items()}}

with torch.no_grad():
    output = model.generate(**inputs, do_sample=False)

audio = output.cpu().numpy().flatten()
result = {{
    "num_samples": len(audio),
    "rms": float(np.sqrt(np.mean(audio ** 2))),
    "duration_s": len(audio) / 24000.0,
    "sample_rate": 24000,
}}
print(json.dumps(result))

np.save("/tmp/hf_bark_audio.npy", audio)
"""
        t0 = time.monotonic()
        result = subprocess.run(
            [python, "-c", script],
            capture_output=True, text=True, timeout=600)
        elapsed = time.monotonic() - t0

        stderr_truncated, stderr_log = save_full_stderr(
            result.stderr or "", ctx.artifacts_dir or "",
            "torch_text_to_audio", case.name)
        data: dict = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": stderr_truncated,
        }
        if stderr_log:
            data["stderr_log"] = stderr_log

        # Parse JSON output
        try:
            import json as json_mod
            parsed = json_mod.loads(result.stdout.strip())
            data.update(parsed)
        except Exception:
            pass

        # Load audio array
        npy_path = "/tmp/hf_bark_audio.npy"
        if os.path.exists(npy_path):
            try:
                import numpy as np
                data["audio_samples"] = np.load(npy_path)
            except Exception:
                pass

        return StageOutput(
            stage_name=stage.name,
            data=data,
            timing_s=elapsed,
            metadata={"backend": "torch_reference"},
        )

    def _run_speech_to_text_ref(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run HF Whisper pipeline as reference for speech-to-text."""
        model_id = case.hf_id
        python = ctx.reference_python_path() or sys.executable

        # Resolve audio input
        audio_input = case.inputs.get("audio_path", "")
        if not audio_input:
            audio_input = case.metadata.get("test_input_audio", "")
        if audio_input and not os.path.isabs(audio_input):
            e2e_dir = Path(__file__).resolve().parents[2] / "e2e"
            audio_input = str(e2e_dir / audio_input)

        script = f"""
import torch
import numpy as np
import json

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

audio, sr = librosa.load({audio_input!r}, sr=16000)
processor = WhisperProcessor.from_pretrained({model_id!r})
model = WhisperForConditionalGeneration.from_pretrained({model_id!r})
model = model.to("cuda")

input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
input_features = input_features.to("cuda")

with torch.no_grad():
    predicted_ids = model.generate(input_features)

transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
token_ids = predicted_ids[0].cpu().tolist()

result = {{
    "transcript": transcript,
    "token_ids": token_ids,
    "num_tokens": len(token_ids),
}}
print(json.dumps(result))
"""
        t0 = time.monotonic()
        result = subprocess.run(
            [python, "-c", script],
            capture_output=True, text=True, timeout=600)
        elapsed = time.monotonic() - t0

        stderr_truncated, stderr_log = save_full_stderr(
            result.stderr or "", ctx.artifacts_dir or "",
            "torch_speech_to_text", case.name)
        data: dict = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": stderr_truncated,
        }
        if stderr_log:
            data["stderr_log"] = stderr_log

        try:
            import json as json_mod
            parsed = json_mod.loads(result.stdout.strip())
            data.update(parsed)
        except Exception:
            pass

        return StageOutput(
            stage_name=stage.name,
            data=data,
            text=data.get("transcript", ""),
            timing_s=elapsed,
            metadata={"backend": "torch_reference"},
        )

    def _run_golden_snapshot(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Load pre-saved golden reference outputs from .npy files."""
        golden_dir = case.inputs.get("golden_dir", "")
        if not golden_dir:
            golden_dir = os.path.join(ctx.engine_dir, f"{case.name}_golden")

        if not os.path.isabs(golden_dir):
            e2e_dir = Path(__file__).resolve().parents[2] / "e2e"
            golden_dir = str(e2e_dir / golden_dir)

        golden_file = os.path.join(golden_dir, f"{stage.name}.npy")
        if not os.path.exists(golden_file):
            return StageOutput(
                stage_name=stage.name,
                data={"error": f"Golden snapshot not found: {golden_file}"},
            )

        try:
            import numpy as np
            golden_data = np.load(golden_file, allow_pickle=True)
            return StageOutput(
                stage_name=stage.name,
                data={"golden_data": golden_data, "source_path": golden_file},
                metadata={"backend": "torch_reference"},
            )
        except Exception as e:
            return StageOutput(
                stage_name=stage.name,
                data={"error": f"Failed to load golden snapshot: {e}"},
            )

plugin = TorchReference()
