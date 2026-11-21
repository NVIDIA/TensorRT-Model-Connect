# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate FP8 activation scales for the PersonaPlex temporal transformer.

Not used when building an engine. Calibrate each supported model shape and commit the
updated ``fp8_activation_scales.json``::

    python -m families.personaplex.calibrate_fp8_scales calibrate \
        --model-dir <original PersonaPlex checkpoint> --librispeech-dir <dir with the parquet files> \
        --official-repo <checkout of the pinned official PersonaPlex source>

``calibrate`` runs the official greedy PersonaPlex pipeline (bf16 PyTorch) with LibriSpeech
validation speech as the user stream, so the temporal transformer sees real decode-time
activations. Forward pre-hooks record ``max(abs(input)) / 448`` for the input of every
quantized projection, the same statistic ModelOpt's FP8 max calibration stores. ModelOpt
itself cannot be used because the checkpoint's gated MLP runs through a fused functional
kernel, not through ``nn.Linear`` modules. It needs a GPU.

The recorded inputs per layer ``N`` are:

  ``layer.N.w_{q,k,v}``    the output of the attention input norm (one shared scale)
  ``layer.N.w_{gate,up}``  the output of the MLP input norm (one shared scale)

The attention and MLP output projections stay fp16 (see ``quantization.py``), so their
inputs are not recorded.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

from .quantization import ACTIVATION_SCALES_FILENAME, FP8_E4M3_MAX, PROJECTION_STEMS, scale_key

_AUDIO_COMPAT = Path(__file__).parent / "tests" / "personaplex_audio_compat"
_SAMPLE_RATE = 24_000
_SPLITS = ("clean/validation/0000.parquet", "other/validation/0000.parquet")


def load_speech(librispeech_dir: Path, seconds: float, count: int):
    """Return ``count`` mono 24 kHz float32 arrays, each about ``seconds`` long.

    Consecutive validation utterances (clean and other, alternating) are joined with
    half a second of silence, so one array is a continuous stretch of speech.
    """
    import numpy as np
    import pyarrow.parquet as parquet
    import soundfile
    from scipy.signal import resample_poly

    utterances = []
    for split in _SPLITS:
        rows = parquet.read_table(librispeech_dir / split).to_pylist()
        clips = []
        for row in rows:
            samples, rate = soundfile.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            if rate != 16_000:
                raise RuntimeError(f"expected 16 kHz LibriSpeech audio, got {rate}")
            clips.append(resample_poly(samples, 3, 2).astype(np.float32))
        utterances.append(clips)
    gap = np.zeros(_SAMPLE_RATE // 2, dtype=np.float32)
    streams = []
    cursor = [0, 0]
    for index in range(count):
        side = index % 2
        parts: list = []
        total = 0.0
        while total < seconds:
            clip = utterances[side][cursor[side] % len(utterances[side])]
            cursor[side] += 1
            parts.extend([clip, gap])
            total += (len(clip) + len(gap)) / _SAMPLE_RATE
        streams.append(np.concatenate(parts)[None, : int(seconds * _SAMPLE_RATE)])
    return streams


def _import_official(official_repo: Path) -> None:
    """Make the pinned official PersonaPlex source and the audio shim importable."""
    source = official_repo.resolve()
    if not (source / "moshi" / "moshi").is_dir():
        raise RuntimeError(f"{source} is not the official PersonaPlex source")
    sys.path[:0] = [str(_AUDIO_COMPAT), str(source / "moshi"), str(source)]


def calibrate(
    model_dir: Path, librispeech_dir: Path, seconds: float, count: int, official_repo: Path
) -> None:
    """Record input amax of every temporal projection and store the scales under the shape key."""
    _import_official(official_repo)
    from moshi.utils.compile import no_compile, no_cuda_graph

    # The hooks must run on every step, so no CUDA graph and no compiled gating kernel.
    with no_cuda_graph(), no_compile():
        _calibrate(model_dir, librispeech_dir, seconds, count)


def _calibrate(model_dir: Path, librispeech_dir: Path, seconds: float, count: int) -> None:
    """Record input amax of every temporal projection and store the scales under the shape key."""
    import torch
    from moshi.models import LMGen, loaders
    from moshi.models.lm import _iterate_audio, encode_from_sphn
    from moshi.offline import warmup

    from .checkpoint_mapper import _open_safetensors
    from .model import _infer_config_from_weights

    inferred = _infer_config_from_weights(_open_safetensors(model_dir))
    key = scale_key(inferred)

    device = "cuda"
    mimi = loaders.get_mimi(str(model_dir / loaders.MIMI_NAME), device)
    other_mimi = loaders.get_mimi(str(model_dir / loaders.MIMI_NAME), device)
    language_model = loaders.get_moshi_lm(
        str(model_dir / loaders.MOSHI_NAME), device=device, dtype=torch.bfloat16
    ).eval()
    layers = language_model.transformer.layers
    amax: dict[str, float] = {}

    def record(name: str, tensor) -> None:
        value = float(tensor.detach().abs().max())
        amax[name] = max(amax.get(name, 0.0), value)

    def attention_hook(index: int):
        def hook(_module, args):
            normed = args[0]
            for stem in ("w_q", "w_k", "w_v"):
                record(f"layer.{index}.{stem}", normed)

        return hook

    def gating_hook(index: int):
        def hook(_module, args):
            x = args[0]
            for stem in ("w_gate", "w_up"):
                record(f"layer.{index}.{stem}", x)

        return hook

    handles = []
    for index, layer in enumerate(layers):
        handles.append(layer.self_attn.register_forward_pre_hook(attention_hook(index)))
        handles.append(layer.gating.register_forward_pre_hook(gating_hook(index)))

    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    generator = LMGen(
        language_model,
        audio_silence_frame_cnt=0,
        sample_rate=mimi.sample_rate,
        device=device,
        frame_rate=mimi.frame_rate,
        use_sampling=False,
        temp=0.8,
        temp_text=0.7,
        top_k=250,
        top_k_text=25,
    )
    for streaming_model in (mimi, other_mimi, generator):
        streaming_model.streaming_forever(1)
    warmup(mimi, other_mimi, generator, device, frame_size)
    for streaming_model in (mimi, other_mimi, generator):
        streaming_model.reset_streaming()
    for name in list(amax):  # the warmup frames are not speech
        del amax[name]

    frames = 0
    for stream in load_speech(librispeech_dir, seconds, count):
        for streaming_model in (mimi, other_mimi, generator):
            streaming_model.reset_streaming()
        with torch.inference_mode():
            for encoded in encode_from_sphn(
                mimi,
                _iterate_audio(stream, sample_interval_size=frame_size, pad=True),
                max_batch=1,
            ):
                for step in range(encoded.shape[-1]):
                    tokens = generator.step(encoded[:, :, step : step + 1])
                    if tokens is not None:
                        frames += 1
    for handle in handles:
        handle.remove()

    scales = {}
    for index in range(len(layers)):
        for stem in PROJECTION_STEMS:
            name = f"layer.{index}.{stem}"
            if not amax.get(name, 0.0) > 0:
                raise RuntimeError(f"no activation amax recorded for {name}")
            scales[name] = amax[name] / FP8_E4M3_MAX

    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    table[key] = dict(sorted(scales.items()))
    path.write_text(json.dumps(dict(sorted(table.items())), indent=2) + "\n", encoding="utf-8")
    print(f"{frames} frames; wrote {len(scales)} scales for {key!r} to {path}")


def main() -> None:
    """Parse the command line and run the calibration."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("calibrate")
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--librispeech-dir", type=Path, required=True)
    run.add_argument("--seconds", type=float, default=16.0)
    run.add_argument("--count", type=int, default=24)
    run.add_argument("--official-repo", type=Path, required=True)
    args = parser.parse_args()
    calibrate(args.model_dir, args.librispeech_dir, args.seconds, args.count, args.official_repo)


if __name__ == "__main__":
    main()
