# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate FP8 activation scales for Nemotron Speech Streaming.

Not used when building an engine. Calibrate each supported checkpoint and commit the
updated ``fp8_activation_scales.json``::

    python -m families.nemotron_speech_streaming.calibrate_fp8_scales fetch --out clips
    python -m families.nemotron_speech_streaming.calibrate_fp8_scales calibrate \
        --model-dir <directory with the .nemo checkpoint> --clips clips

``fetch`` downloads LibriSpeech validation clips (clean and other) as 16 kHz wav files.
``calibrate`` builds the fp32 streaming encoders (first step and steady state, default
attention context) with one extra output per projection (the maximum absolute value of the
projection's input), streams every clip through them with the same chunking, caches and
masks as the native runtime, and stores ``max(abs(input)) / 448`` per projection under the
checkpoint key (see ``quantization.scale_key``). The model graph itself is the reference:
forward pre-hooks on the ModelOpt-wrapped FastConformer attention do not work (the wrapper
fails on attention that passes ``query=`` keywords), and NeMo is not needed. ``calibrate``
needs a GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .quantization import ACTIVATION_SCALES_FILENAME, FP8_E4M3_MAX, projection_names, scale_key

_DATASET = "openslr/librispeech_asr"
_SPLITS = {"clean": "clean/validation/0000.parquet", "other": "other/validation/0000.parquet"}
_SAMPLE_RATE = 16000
_PREEMPHASIS = 0.97
_N_FFT, _WIN_LENGTH, _HOP = 512, 400, 160
_LOG_GUARD = 2.0**-24
_MASKED = -10000.0
_PRE_ENCODE_CACHE = 9
_SUBSAMPLING = 8


class CalibrationContext:
    """Quantization-context stand-in: plain matmuls plus one ``amax`` output per projection."""

    has_fp8_weights = False  # the engines stay unquantized; only the build options differ

    def __init__(self, graph_ops):
        self.graph_ops = graph_ops
        self.outputs: dict[str, str] = {}  # weight name -> output tensor name

    def maybe_quantized_matmul(
        self, network, lhs, lhs_width, rhs_width, rhs_weights, weight_name, dtype=np.float32
    ):
        """Regular matmul; also mark ``max(abs(lhs))`` as an engine output."""
        import tensorrt as trt

        rank = len(tuple(lhs.shape))
        absolute = network.add_unary(lhs, trt.UnaryOperation.ABS).get_output(0)
        reduced = network.add_reduce(
            absolute, trt.ReduceOperation.MAX, (1 << rank) - 1, True
        ).get_output(0)
        reduced.name = "amax_" + weight_name.replace(".", "_")
        network.mark_output(reduced)
        self.outputs[weight_name] = reduced.name
        return self.graph_ops.add_matmul_rhs_constant(
            network, lhs, lhs_width, rhs_width, rhs_weights, dtype=dtype
        )


def fetch_clips(out: Path, per_split: int) -> None:
    """Download ``per_split`` clips from each split as 16 kHz wav files, each with a .txt transcript."""
    import io

    import pyarrow.parquet as parquet
    import soundfile
    from huggingface_hub import hf_hub_download

    out.mkdir(parents=True, exist_ok=True)
    for split, parquet_path in _SPLITS.items():
        local = hf_hub_download(_DATASET, parquet_path, repo_type="dataset")
        table = parquet.read_table(local).slice(0, per_split).to_pylist()
        for index, row in enumerate(table):
            samples, rate = soundfile.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            if rate != _SAMPLE_RATE:
                raise RuntimeError(f"{split} clip {index} is {rate} Hz, expected {_SAMPLE_RATE}")
            soundfile.write(out / f"{split}-{index:04d}.wav", samples, _SAMPLE_RATE)
            (out / f"{split}-{index:04d}.txt").write_text(row["text"], encoding="utf-8")
    print(f"wrote {len(list(out.glob('*.wav')))} clips to {out}")


def mel_spectrogram(samples: np.ndarray, filters: np.ndarray):
    """Log-mel features ``[mel_bins, frames]`` of 16 kHz audio (NeMo FastConformer front end).

    ``filters`` is the ``[n_fft / 2 + 1, mel_bins]`` filterbank the bundle stores.
    """
    import torch

    x = torch.tensor(samples, dtype=torch.float32)
    x = torch.cat([x[:1], x[1:] - _PREEMPHASIS * x[:-1]])
    window = torch.hann_window(_WIN_LENGTH, periodic=False)
    spectrum = torch.stft(
        x,
        n_fft=_N_FFT,
        hop_length=_HOP,
        win_length=_WIN_LENGTH,
        window=window,
        center=True,
        pad_mode="constant",
        return_complex=True,
    )
    power = spectrum.abs() ** 2
    mel = torch.tensor(filters, dtype=torch.float32).t() @ power
    mel = torch.log(mel + _LOG_GUARD)
    return mel[:, : len(samples) // _HOP].numpy()


class StreamingEncoder:
    """Drives the first-step and steady-state streaming encoder plans like the native runtime."""

    def __init__(self, first_plan: bytes, next_plan: bytes, weights, right_context: int):
        import tensorrt as trt
        import torch

        self.torch = torch
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self.engines = {}
        for name, plan in (("first", first_plan), ("next", next_plan)):
            engine = self.runtime.deserialize_cuda_engine(plan)
            self.engines[name] = (engine, engine.create_execution_context())
        self.layers = int(weights["_enc_layers"])
        self.hidden = int(weights["_hidden"])
        self.cache_left = int(weights["_streaming_cache_left"])
        self.time_cache = 8
        self.right = right_context
        self.mel_bins = int(weights["_mel_bins"])

    def _run(self, key: str, feeds: dict[str, Any]) -> dict[str, Any]:
        torch = self.torch
        engine, context = self.engines[key]
        outputs: dict[str, Any] = {}
        for name, tensor in feeds.items():
            context.set_input_shape(name, tuple(tensor.shape))
            context.set_tensor_address(name, tensor.data_ptr())
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            if engine.get_tensor_mode(name).name != "OUTPUT":
                continue
            dtype = {0: torch.float32, 1: torch.float16}[int(engine.get_tensor_dtype(name))]
            outputs[name] = torch.empty(
                tuple(context.get_tensor_shape(name)), dtype=dtype, device="cuda"
            )
            context.set_tensor_address(name, outputs[name].data_ptr())
        context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return outputs

    def stream(self, mel: np.ndarray, observe=None):
        """Encode a whole utterance chunk by chunk; returns the encoder frames ``[frames, hidden]``."""
        torch = self.torch
        right = self.right
        query = right + 1
        first_shift = 1 + _SUBSAMPLING * right
        next_shift = _SUBSAMPLING * query
        total = mel.shape[1]
        channel = torch.zeros(self.layers, self.cache_left, self.hidden, device="cuda")
        time = torch.zeros(self.layers, self.hidden, self.time_cache, device="cuda")
        cache_len, start, frames = 0, 0, []
        while start < total:
            first = start == 0
            shift = first_shift if first else next_shift
            pre = 0 if first else _PRE_ENCODE_CACHE
            new = min(shift, total - start)
            chunk = np.zeros((self.mel_bins, pre + shift), dtype=np.float32)
            lo = max(start - pre, 0)
            chunk[:, pre - (start - lo) : pre] = mel[:, lo:start]
            chunk[:, pre : pre + new] = mel[:, start : start + new]
            drop = 0 if first else 2
            valid_total = pre + new
            sub = valid_total
            for _ in range(3):
                sub = (sub + 2 - 3) // 2 + 1
            valid_query = max(0, min(sub - drop, query))
            if valid_query <= 0:
                break
            key_len = self.cache_left + query
            mask = np.full((1, query, key_len), _MASKED, dtype=np.float32)
            begin = self.cache_left - cache_len
            for q in range(valid_query):
                end = self.cache_left + min(valid_query - 1, q + right)
                mask[0, q, begin : end + 1] = 0.0
            out = self._run(
                "first" if first else "next",
                {
                    "mel_features": torch.tensor(chunk, device="cuda"),
                    "cache_last_channel": channel,
                    "cache_last_time": time,
                    "encoder_mask": torch.tensor(mask, device="cuda"),
                },
            )
            if observe is not None:
                observe(out)
            frames.append(out["encoder_output"][:valid_query].float().cpu().numpy())
            channel = out["cache_last_channel_next"].float().contiguous()
            time = out["cache_last_time_next"].float().contiguous()
            cache_len = min(self.cache_left, cache_len + query)
            start += new
            if new < shift:
                break
        return np.concatenate(frames, 0) if frames else np.zeros((0, self.hidden), np.float32)


def calibrate(model_dir: Path, clips: Path, right_context: int) -> None:
    """Stream the clips through the instrumented encoders and store the projection input scales."""
    import soundfile

    from . import graph_ops
    from .config import ModelConfig
    from .model import (
        _build_mel_filterbank,
        _build_streaming_encoder,
        _NemotronSpeechStreamingModel,
    )

    config = ModelConfig.from_dir(model_dir)
    config.raw["_model_dir"] = str(model_dir)
    model = _NemotronSpeechStreamingModel()
    weights = model.load_weights(str(model_dir), config, precision="fp32")
    key = scale_key(weights)
    header = np.frombuffer(_build_mel_filterbank(weights), dtype=np.int32, count=2)
    filters = np.frombuffer(_build_mel_filterbank(weights), dtype=np.float32, offset=8).reshape(
        int(header[0]), int(header[1])
    )

    contexts = {"first": CalibrationContext(graph_ops), "next": CalibrationContext(graph_ops)}
    plans = {
        name: _build_streaming_encoder(
            weights,
            right_context,
            first_step=name == "first",
            precision="fp32",
            quant_ctx=contexts[name],
        )
        for name in contexts
    }
    encoder = StreamingEncoder(plans["first"], plans["next"], weights, right_context)

    amax: dict[str, float] = {}
    current = {"name": "first"}

    def observe(outputs):
        context = contexts[current["name"]]
        for name, tensor_name in context.outputs.items():
            amax[name] = max(amax.get(name, 0.0), float(outputs[tensor_name].max()))

    files = sorted(clips.glob("*.wav"))
    if not files:
        raise RuntimeError(f"no wav clips in {clips}")
    original_run = encoder._run

    def tracking_run(name, feeds):
        current["name"] = name
        return original_run(name, feeds)

    encoder._run = tracking_run
    for index, path in enumerate(files):
        samples, rate = soundfile.read(path, dtype="float32")
        if rate != _SAMPLE_RATE:
            raise RuntimeError(f"{path} is {rate} Hz, expected {_SAMPLE_RATE}")
        encoder.stream(mel_spectrogram(samples, filters), observe)
        if (index + 1) % 16 == 0:
            print(f"[{index + 1}/{len(files)}]", flush=True)

    scales = {name: value / FP8_E4M3_MAX for name, value in amax.items() if value > 0}
    missing = [name for name in projection_names(weights) if name not in scales]
    if missing:
        raise RuntimeError(
            f"no calibrated input range for {len(missing)} projections: {missing[:4]}"
        )
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    table[key] = dict(sorted(scales.items()))
    path.write_text(json.dumps(dict(sorted(table.items())), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(scales)} scales for {key!r} to {path}")


def main() -> None:
    """Parse the command line and run the fetch or calibrate command."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--out", type=Path, required=True)
    fetch.add_argument("--per-split", type=int, default=96)
    run = commands.add_parser("calibrate")
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--clips", type=Path, required=True)
    run.add_argument("--right-context", type=int, default=13)
    args = parser.parse_args()
    if args.command == "fetch":
        fetch_clips(args.out, args.per_split)
    else:
        calibrate(args.model_dir, args.clips, args.right_context)


if __name__ == "__main__":
    main()
