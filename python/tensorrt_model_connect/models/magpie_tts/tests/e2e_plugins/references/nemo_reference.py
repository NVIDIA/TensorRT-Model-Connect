# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo reference backend for models originating from NVIDIA NeMo.

Provides reference outputs for models that use NeMo as their official
inference framework rather than HF Transformers or Diffusers:
- MagpieTTS: NeMo TTS pipeline with NanoCodec decoding
- Canary: NeMo ASR (if not available via HF Transformers)
- Future NeMo models

All inference runs in a subprocess to prevent GPU OOM and NeMo
import-time side effects from polluting the test process.
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

import numpy as np

from .. import save_full_stderr
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec
from ..runtime_config import runtime_config_get

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[7]
MAGPIE_SPEAKER_ENCODER_REPO = "Edresson/Speaker_Encoder_H_ASP"
MAGPIE_SPEAKER_ENCODER_FILENAME = "pytorch_model.bin"
MAGPIE_SPEAKER_ENCODER_URL = (
    "https://huggingface.co/Edresson/Speaker_Encoder_H_ASP/resolve/main/pytorch_model.bin"
)


def _write_wav(path: str, audio: np.ndarray, sample_rate: int) -> None:
    """Write float32 audio array to WAV file."""
    import wave
    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


class NemoReference:
    """Reference backend for NeMo-native models."""

    @property
    def backend_name(self) -> str:
        return "nemo"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        task = case.task_strategy
        runtime = case.runtime_strategy

        if runtime == "text_to_audio_magpie" or "magpie" in case.family:
            return self._run_magpie_tts_ref(case, stage, ctx)

        raise ValueError(
            f"NeMo reference backend does not support "
            f"task_strategy={task!r} runtime_strategy={runtime!r}"
        )

    def _run_magpie_tts_ref(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run MagpieTTS inference via NeMo as ground-truth reference."""
        model_id = case.hf_id
        model_revision = case.hf_revision or None
        prompt = case.inputs.get("prompt", "Hello, this is a test.")
        python = ctx.reference_python_path() or sys.executable

        seed = runtime_config_get(case, "audio_magpie.seed", 42)

        artifacts_dir = ctx.artifacts_dir or tempfile.mkdtemp()
        model_dir = Path(artifacts_dir) / case.name
        model_dir.mkdir(parents=True, exist_ok=True)
        wav_path = str(model_dir / "nemo_ref_audio.wav")
        json_path = str(model_dir / "nemo_ref_result.json")

        script = textwrap.dedent(f"""\
            import json, os, random, sys, warnings
            os.environ["NEMO_LOG_LEVEL"] = "ERROR"
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            warnings.filterwarnings("ignore")

            import fsspec
            import numpy as np
            import torch
            from huggingface_hub import hf_hub_download

            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

            # Magpie's upstream NeMo config stores the speaker-encoder weight as
            # a Hub HTTPS URL.  CI deliberately disables the network after its
            # cache-warm phase, so map that URL to the family-declared,
            # pre-warmed Hub file instead of allowing fsspec to make a request.
            speaker_checkpoint = hf_hub_download(
                repo_id={MAGPIE_SPEAKER_ENCODER_REPO!r},
                filename={MAGPIE_SPEAKER_ENCODER_FILENAME!r},
                local_files_only=True,
            )
            speaker_checkpoint_url = {MAGPIE_SPEAKER_ENCODER_URL!r}
            original_fsspec_open = fsspec.open

            def offline_fsspec_open(path, *args, **kwargs):
                normalized = str(path).split("?", 1)[0]
                if normalized == speaker_checkpoint_url:
                    path = speaker_checkpoint
                return original_fsspec_open(path, *args, **kwargs)

            fsspec.open = offline_fsspec_open

            random.seed({seed})
            np.random.seed({seed})
            torch.manual_seed({seed})
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all({seed})

            # Load MagpieTTS from HuggingFace
            try:
                from nemo.collections.tts.models import MagpieTTSModel
            except ImportError:
                from nemo.collections.tts.models import (
                    MagpieTTS_Model as MagpieTTSModel,
                )
            model_archive = hf_hub_download(
                repo_id={model_id!r},
                filename="magpie_tts_multilingual_357m.nemo",
                revision={model_revision!r},
                local_files_only=True,
            )
            model = MagpieTTSModel.restore_from(restore_path=model_archive)
            model.eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)

            # Model construction and checkpoint restoration can consume RNG
            # state. Reset every sampling stream immediately before NeMo's
            # multinomial decoder so the declared seed owns generation.
            random.seed({seed})
            np.random.seed({seed})
            torch.manual_seed({seed})
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all({seed})

            # Run inference via NeMo's do_tts() API
            # Signature: do_tts(transcript, language="en", apply_TN=False,
            #                   use_cfg=True, speaker_index=None)
            # Returns: (audio [1, T], audio_len [1])
            with torch.no_grad():
                audio_tensor, audio_len = model.do_tts(
                    transcript={prompt!r},
                    language="en",
                    use_cfg=True,
                )

            audio = audio_tensor.cpu().numpy().flatten()
            # Trim to actual length
            alen = int(audio_len.item()) if audio_len.numel() > 0 else len(audio)
            audio = audio[:alen]

            # Normalize if needed
            peak = np.max(np.abs(audio))
            if peak > 1.0:
                audio = audio / peak

            rms = float(np.sqrt(np.mean(audio ** 2)))
            sample_rate = 22050
            duration_s = len(audio) / sample_rate

            # Write WAV
            import wave
            pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
            with wave.open({wav_path!r}, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(pcm.tobytes())

            result = {{
                "num_samples": len(audio),
                "rms": rms,
                "duration_s": duration_s,
                "sample_rate": sample_rate,
                "wav_path": {wav_path!r},
            }}
            with open({json_path!r}, "w") as f:
                json.dump(result, f)
            print(json.dumps(result))
        """)

        env = os.environ.copy()

        t0 = time.monotonic()
        try:
            result = subprocess.run(
                [python, "-c", script],
                capture_output=True, text=True, timeout=600,
                env=env,
            )
            elapsed = time.monotonic() - t0
        except subprocess.TimeoutExpired:
            return StageOutput(
                stage_name=stage.name,
                data={"error": "NeMo reference timed out"},
                timing_s=600.0,
                metadata={"backend": "nemo", "returncode": -1},
            )

        stderr_truncated, stderr_log = save_full_stderr(
            result.stderr or "", artifacts_dir, "nemo_magpie_ref", case.name)

        data: dict = {
            "returncode": result.returncode,
            "stderr_truncated": stderr_truncated,
        }
        if stderr_log:
            data["stderr_log"] = stderr_log

        # Parse JSON output
        if result.returncode == 0:
            try:
                parsed = json.loads(result.stdout.strip().splitlines()[-1])
                data.update(parsed)
            except Exception as e:
                logger.warning("Failed to parse NeMo ref output: %s", e)
                data["parse_error"] = str(e)
                data["stdout"] = result.stdout

        else:
            data["stdout"] = result.stdout
            data["stderr"] = result.stderr

        return StageOutput(
            stage_name=stage.name,
            data=data,
            timing_s=elapsed,
            metadata={
                "backend": "nemo",
                "returncode": result.returncode,
                "command": [python, "-c", "<nemo_magpie_ref_script>"],
            },
        )


plugin = NemoReference()
