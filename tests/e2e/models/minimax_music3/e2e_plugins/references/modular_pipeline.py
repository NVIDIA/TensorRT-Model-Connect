# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference backend: the upstream modular pipeline.

Pinned to diffusers==0.40.0. The model card still tells readers to install
from the commit on the pull request that added the pipeline; that request
merged on 2026-08-13 and 0.40.0, released a week later, ships it.

One operational note this backend has to live with: the checkpoint's
modular_model_index.json names each component by its Hub repository rather
than by a path relative to the snapshot, so load_components fetches its own
copy into the Hugging Face cache instead of reading the directory it was given.
A run therefore needs room for two copies unless the index is rewritten.

The file is not named for the upstream library it drives. ``hf_diffusers.py``
is the sidecar name the repository's diffusion families own, and a family
registered under ``text_to_audio`` that used it would trip the gate that keeps
diffusion behaviour out of non-diffusion folders -- correctly, since this
family is not one of them.
"""

from __future__ import annotations

from ..contracts import E2ECase, RunContext, StageOutput, StageSpec

DIFFUSERS_VERSION = "0.40.0"
PIPELINE_CLASS = "MiniMaxMusic3ModularPipeline"


class HFDiffusersReference:
    @property
    def backend_name(self) -> str:
        return "minimax_music3_modular"

    def run_stage(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        """Generate the reference waveform and describe it like the runner does.

        Run out of process in the reference interpreter: the pipeline holds the
        whole checkpoint in bf16 on the GPU the bundle has just finished using,
        and the two do not have to be resident at once.
        """

        import json
        import os
        import subprocess
        import tempfile
        import textwrap
        import time

        inputs = getattr(case, "inputs", {}) or {}
        runtime = (getattr(case, "metadata", {}) or {}).get("runtime_config", {})
        namespace = (runtime or {}).get("music_minimax_music3", {})
        lyrics = inputs.get("prompt", "")
        caption = str(namespace.get("caption", ""))
        seed = int(inputs.get("seed", namespace.get("seed", 0)) or 0)
        duration = float(namespace.get("audio_duration", 20.0))

        with tempfile.TemporaryDirectory(prefix="mm3_ref_") as tmpdir:
            wav_path = os.path.join(tmpdir, "reference.wav")
            script = textwrap.dedent(
                """
                import glob, json, struct, sys
                import numpy as np
                import torch
                from diffusers.modular_pipelines.minimax_music3.modular_blocks_minimax_music3 \
                    import MiniMaxMusic3Blocks

                payload = json.loads(sys.argv[1])
                snapshots = glob.glob(payload["snapshot_glob"])
                if not snapshots:
                    raise SystemExit("no MiniMax-Music3 snapshot in the cache")

                pipe = MiniMaxMusic3Blocks().init_pipeline(sorted(snapshots)[0])
                pipe.load_components(torch_dtype=torch.bfloat16)
                pipe.to("cuda")
                generator = torch.Generator("cuda").manual_seed(payload["seed"])

                # The reference names the music description `prompt` and the
                # sung text `lyrics`; the task contract is the other way round.
                state = pipe(
                    prompt=payload["caption"],
                    lyrics=payload["lyrics"],
                    audio_duration=payload["duration"],
                    generator=generator,
                )
                audio = np.asarray(state.values["audios"], dtype=np.float32)[0]
                interleaved = audio.T.reshape(-1)
                count = interleaved.size
                channels = audio.shape[0]
                with open(payload["wav_path"], "wb") as handle:
                    handle.write(b"RIFF")
                    handle.write(struct.pack("<i", 36 + count * 4))
                    handle.write(b"WAVEfmt ")
                    handle.write(struct.pack(
                        "<ihhiihh", 16, 3, channels, 44100,
                        44100 * channels * 4, channels * 4, 32))
                    handle.write(b"data")
                    handle.write(struct.pack("<i", count * 4))
                    handle.write(interleaved.astype("<f4").tobytes())
                print(json.dumps({"channels": channels, "samples": int(audio.shape[1])}))
                """
            )
            payload = json.dumps({
                "snapshot_glob": os.path.expanduser(
                    "~/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-Music3/snapshots/*"
                ),
                "wav_path": wav_path,
                "lyrics": lyrics,
                "caption": caption,
                "seed": seed,
                "duration": duration,
            })

            python = ctx.reference_python or ctx.hf_python or "python3"
            started = time.monotonic()
            result = subprocess.run(
                [str(python), "-c", script, payload],
                capture_output=True, text=True, timeout=3600,
            )
            elapsed = time.monotonic() - started

            data: dict = {
                "returncode": result.returncode,
                "stdout": result.stdout[-4000:],
                "stderr": result.stderr[-4000:],
                "elapsed_s": elapsed,
                "wav_exists": False,
            }
            if result.returncode == 0 and os.path.exists(wav_path):
                # Keep the waveform past the temporary directory: the contract
                # transcribes it after this returns.
                kept = os.path.join(
                    ctx.artifacts_dir or tempfile.gettempdir(),
                    f"{case.name}_reference.wav",
                )
                os.makedirs(os.path.dirname(kept), exist_ok=True)
                os.replace(wav_path, kept)
                data.update(_describe_wav(kept))
            return StageOutput(stage_name=stage.name, data=data)


plugin = HFDiffusersReference()


def _describe_wav(path: str) -> dict:
    """Return what the contract scores, read from the header directly.

    The standard library's wave module rejects IEEE float32, which is what
    both this pipeline and the bundle write.
    """

    import math
    import os
    import struct

    if not os.path.exists(path):
        return {"wav_exists": False}
    with open(path, "rb") as handle:
        raw = handle.read()
    if len(raw) < 44 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        return {"wav_exists": True, "wav_valid": False}

    audio_format, channels, sample_rate = struct.unpack("<HHI", raw[20:28])
    bits = struct.unpack("<H", raw[34:36])[0]
    payload = raw[44:]
    if audio_format == 3 and bits == 32:
        count = len(payload) // 4
        samples = struct.unpack(f"<{count}f", payload[: count * 4])
    elif bits == 16:
        count = len(payload) // 2
        samples = [v / 32768.0 for v in struct.unpack(f"<{count}h", payload[: count * 2])]
    else:
        return {"wav_exists": True, "wav_valid": False}

    frames = len(samples) // max(channels, 1)
    total = math.fsum(value * value for value in samples)
    return {
        "wav_exists": True,
        "wav_valid": True,
        "wav_path": path,
        "channels": channels,
        "sample_rate": sample_rate,
        "num_frames": frames,
        "duration_s": frames / sample_rate if sample_rate else 0.0,
        "rms": math.sqrt(total / len(samples)) if samples else 0.0,
    }
