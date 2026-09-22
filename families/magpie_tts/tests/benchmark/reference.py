#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the family-owned Magpie TTS reference."""

from __future__ import annotations

import os
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

from qualification_tests.benchmark_qualification.performance import reference_harness


SPEAKER_REPOSITORY = "Edresson/Speaker_Encoder_H_ASP"
SPEAKER_FILENAME = "pytorch_model.bin"
SPEAKER_URL = "https://huggingface.co/Edresson/Speaker_Encoder_H_ASP/resolve/main/pytorch_model.bin"


def _seed(torch: Any, value: int) -> None:
    import numpy as np

    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _materialize(summary: dict[str, Any], output: Path) -> None:
    import soundfile as sound

    audio = summary.pop("_audio_f32")
    artifact = output.with_suffix(".audio.wav").resolve()
    artifact.parent.mkdir(parents=True, exist_ok=True)
    sound.write(artifact, audio, int(summary["sample_rate"]), subtype="FLOAT")
    summary["audio_artifact"] = str(artifact)


def _load(
    arguments: Any,
    request: Mapping[str, Any],
    options: Mapping[str, Any],
) -> reference_harness.Session:
    if arguments.mode != "pytorch-eager" or arguments.precision != "fp32":
        raise ValueError("Magpie TTS reference requires pytorch-eager fp32")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import fsspec
    import torch
    from huggingface_hub import hf_hub_download
    from nemo.collections.tts.models import MagpieTTSModel

    torch.use_deterministic_algorithms(True)
    speaker_revision = str(options.get("speaker_encoder_revision", "") or "")
    if not speaker_revision:
        raise ValueError("Magpie TTS requires speaker_encoder_revision")
    speaker_checkpoint = hf_hub_download(
        repo_id=SPEAKER_REPOSITORY,
        filename=SPEAKER_FILENAME,
        revision=speaker_revision,
        local_files_only=arguments.local_files_only,
    )
    original_open = fsspec.open

    def offline_open(path: Any, *args: Any, **kwargs: Any) -> Any:
        if str(path).split("?", 1)[0] == SPEAKER_URL:
            path = speaker_checkpoint
        return original_open(path, *args, **kwargs)

    fsspec.open = offline_open
    archive = hf_hub_download(
        repo_id=arguments.model,
        filename="magpie_tts_multilingual_357m.nemo",
        revision=arguments.revision,
        local_files_only=arguments.local_files_only,
    )
    model = MagpieTTSModel.restore_from(restore_path=archive).eval().to("cuda")
    max_new_tokens = request.get("max_new_tokens", 0)
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise ValueError("Magpie max_new_tokens must be an integer")
    if max_new_tokens > 0:
        model.inference_parameters.max_decoder_steps = max_new_tokens
    seed = request.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Magpie seed must be an integer")
    prompt = str(request.get("prompt", ""))

    def invoke() -> Mapping[str, Any]:
        _seed(torch, seed)
        with torch.inference_mode():
            audio, length = model.do_tts(transcript=prompt, language="en", use_cfg=True)
        count = int(length.item()) if length.numel() else int(audio.numel())
        return {
            "audio_samples": count,
            "sample_rate": 22_050,
            "_audio_f32": audio.detach().float().cpu().reshape(-1)[:count].numpy(),
        }

    return reference_harness.Session(
        invoke,
        "nemo",
        timing_scope="task-pipeline-call-wall",
        input_preparation_included=True,
        materialize=_materialize,
    )


def main(argv: Sequence[str] | None = None) -> int:
    return reference_harness.run(
        list(sys.argv[1:] if argv is None else argv),
        description=__doc__,
        load=_load,
    )


if __name__ == "__main__":
    raise SystemExit(main())
