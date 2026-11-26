# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference transcription by the transformers implementation of VibeVoice-ASR.

The repository pins transformers 5.2.0, which has no ``vibevoice_asr``; the model
arrived in transformers 5.3.0, the source pinned in ``reference-source.json``.
``run_reference`` starts this file in a child interpreter (``python -I``) that
imports transformers from that checkout's ``src/`` ahead of the installed one and
takes every other package from the interpreter; nothing is installed or
downloaded. The checkout is ``TRTMC_REFERENCE_SOURCE_DIR``; the interpreter is the
current one unless ``TRTMC_VIBEVOICE_ASR_REFERENCE_PYTHON`` names another.

The child loads ``microsoft/VibeVoice-ASR-HF``, Microsoft's release of the same
weights in the transformers layout, in BF16 with SDPA attention in the language
model (the class default) and torch's cuDNN SDPA backend disabled, and decodes
greedily until the tokenizer's end of text. The acoustic latent is the sigma-VAE mean: the
implementation adds noise with a random scale (``vae_std``) to it on every call,
and ``vae_std = 0`` removes the noise. The audio is read as the native runtime
reads it (16-bit PCM as ``value / 32768``); other rates are resampled to 24 kHz
with torchaudio, because the processor takes samples at 24 kHz only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import wave
from pathlib import Path

REFERENCE_REPOSITORY = "microsoft/VibeVoice-ASR-HF"
REFERENCE_REVISION = "f22241c2062b3b25272bf117397e03d73381037a"
REFERENCE_FILES = (
    "config.json",
    "generation_config.json",
    "processor_config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
    "*.safetensors",
)
TRANSFORMERS_VERSION = "5.3.0"
SAMPLE_RATE = 24000
TOP_K = 5


def run_reference(model_dir: Path, audio: Path, context_info: str | None, output: Path) -> dict:
    """Transcribe ``audio`` in the isolated child and return its result record."""
    value = os.environ.get("TRTMC_REFERENCE_SOURCE_DIR")
    assert value, (
        "selected vibevoice_asr E2E requires TRTMC_REFERENCE_SOURCE_DIR: a checkout of the "
        "transformers source pinned in tests/reference-source.json"
    )
    source = Path(value) / "src"
    assert (source / "transformers" / "__init__.py").is_file(), (
        f"TRTMC_REFERENCE_SOURCE_DIR is not a transformers source checkout: {value}"
    )
    python = os.environ.get("TRTMC_VIBEVOICE_ASR_REFERENCE_PYTHON", sys.executable)
    command = [
        python,
        "-I",
        str(Path(__file__).resolve()),
        "--transformers-src",
        str(source),
        "--model-dir",
        str(model_dir),
        "--audio",
        str(audio),
        "--output",
        str(output),
    ]
    if context_info:
        command += ["--context-info", context_info]
    subprocess.run(command, check=True, timeout=3600)
    return json.loads(output.read_text(encoding="utf-8"))


def _read_pcm16(path: Path):
    import numpy as np

    with wave.open(str(path)) as handle:
        if handle.getsampwidth() != 2 or handle.getnchannels() != 1:
            raise ValueError(f"{path} is not mono 16-bit PCM")
        rate = handle.getframerate()
        raw = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    return raw.astype(np.float32) / 32768.0, rate


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformers-src", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--context-info", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = Path(args.transformers_src).resolve()
    # Only this child imports the pinned transformers; the parent keeps the installed one.
    sys.path.insert(0, str(source))
    import torch
    import transformers

    # With torch 2.12.0+cu130 on a B300, the cuDNN SDPA backend decodes this model
    # wrongly (repeated words); the other SDPA backends do not.
    torch.backends.cuda.enable_cudnn_sdp(False)
    location = Path(transformers.__file__).resolve()
    if not location.is_relative_to(source) or transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"expected transformers {TRANSFORMERS_VERSION} from {source}, "
            f"imported {transformers.__version__} from {location}"
        )
    from transformers import (
        VibeVoiceAsrConfig,
        VibeVoiceAsrForConditionalGeneration,
        VibeVoiceAsrProcessor,
    )

    samples, rate = _read_pcm16(Path(args.audio))
    if rate != SAMPLE_RATE:
        import torchaudio.functional

        samples = torchaudio.functional.resample(
            torch.from_numpy(samples), rate, SAMPLE_RATE
        ).numpy()

    config = VibeVoiceAsrConfig.from_pretrained(args.model_dir)
    config.acoustic_tokenizer_encoder_config.vae_std = 0.0
    model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
        args.model_dir, config=config, dtype=torch.bfloat16
    )
    attention = model.language_model.config._attn_implementation
    if attention != "sdpa":
        raise RuntimeError(f"the language model runs {attention} attention, not sdpa")
    model = model.to("cuda").eval()
    processor = VibeVoiceAsrProcessor.from_pretrained(args.model_dir)
    inputs = processor.apply_transcription_request(audio=samples, prompt=args.context_info)
    inputs = inputs.to(model.device, dtype=model.dtype)
    eos = processor.tokenizer.eos_token_id
    if eos != model.generation_config.eos_token_id:
        raise RuntimeError(
            f"end of text {eos} is not the generation config's {model.generation_config.eos_token_id}"
        )
    with torch.inference_mode():
        output = model.generate(
            **inputs, do_sample=False, output_logits=True, return_dict_in_generate=True
        )
    prompt = inputs["input_ids"][0].tolist()
    generated = output.sequences[0, len(prompt) :].tolist()
    # The best logits of every step, so the test can tell a near-tie from a real difference.
    top = [torch.topk(step[0].float(), TOP_K) for step in output.logits]
    stopped = eos in generated
    if stopped:
        generated = generated[: generated.index(eos)]
    result = {
        "prompt_ids": prompt,
        "generated_ids": generated,
        "top_logits": [
            {"ids": item.indices.tolist(), "values": item.values.tolist()}
            for item in top[: len(generated) + 1]
        ],
        "stopped_at_eos": stopped,
        "eos_id": eos,
        "text": processor.tokenizer.decode(generated, skip_special_tokens=True),
        "speech_samples": int(inputs["padding_mask"].sum()),
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": torch.cuda.get_device_name(0),
            "attention": attention,
            "cudnn_sdp": torch.backends.cuda.cudnn_sdp_enabled(),
        },
    }
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
