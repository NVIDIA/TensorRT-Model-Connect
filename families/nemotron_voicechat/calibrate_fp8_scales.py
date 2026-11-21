# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate FP8 activation scales for the VoiceChat thinker.

Not used when building an engine. Calibrate each supported checkpoint and commit the
updated ``fp8_activation_scales.json``::

    python -m families.nemotron_voicechat.calibrate_fp8_scales fetch --out clips
    python -m families.nemotron_voicechat.calibrate_fp8_scales calibrate \
        --model-dir <VoiceChat checkpoint directory> --clips clips

``fetch`` downloads LibriSpeech clips (clean and other) as 16 kHz wav files.
``calibrate`` builds the fp32 streaming perception encoders and the fp32 thinker with one
extra output per quantized projection (the maximum absolute value of the projection's
input), then runs every clip through the same loop as the native runtime: the system
prompt is prefilled, the clip (plus trailing silence) is streamed through the perception
encoder one 80 ms frame at a time, each frame goes through the thinker with the previously
predicted text and function tokens, and the agent response is started with the BOS token
after the speech ends and decoded greedily until EOS. The scale of each projection is
``max(abs(input)) / 448`` over all steps, stored under the model config key (see
``quantization.scale_key``). The model graph itself is the reference: no NeMo is needed.
``calibrate`` needs a GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .quantization import (
    ACTIVATION_SCALES_FILENAME,
    FP8_E4M3_MAX,
    projection_names,
    scale_key,
)

_DATASET = "openslr/librispeech_asr"
_SPLITS = {
    "validation": {
        "clean": "clean/validation/0000.parquet",
        "other": "other/validation/0000.parquet",
    },
    "test": {"clean": "clean/test/0000.parquet", "other": "other/test/0000.parquet"},
}
_SAMPLE_RATE = 16000
_N_FFT, _HOP = 512, 160
_LOG_GUARD = 2.0**-24
_MASKED = -10000.0
_PRE_ENCODE_CACHE = 9
_SUBSAMPLING = 8
_FRAME_SAMPLES = 1280  # one 80 ms model frame at 16 kHz
_PAD_TOKEN, _BOS_TOKEN, _EOS_TOKEN = 12, 1, 2
_SILENCE_SECONDS = 4.0
_MAX_RESPONSE_FRAMES = 120
_SYSTEM_PROMPTS = (
    "You are an AI voice assistant developed by NVIDIA. Your name is NVIDIA Voice Chat. "
    "Answer in a spoken, conversational style rather than a written one. Do not repeat "
    "the same sentence over and over again. Start the conversation by greeting the user.",
    "You are a friendly voice assistant. Keep every answer short and speak naturally.",
    "You are a helpful assistant that talks with people. Answer the question you hear in "
    "two or three spoken sentences.",
    "You are a careful voice assistant. Listen to the user, then answer clearly and politely.",
)


class CalibrationContext:
    """Quantization-context stand-in: plain matmuls plus one ``amax`` output per projection."""

    def __init__(self):
        self.outputs: dict[str, str] = {}  # weight name -> output tensor name

    def wants(self, weight_name: str) -> bool:
        """The weights stay fp32 while calibrating."""
        return False

    def project(self, network, lhs, lhs_width, rhs_width, rhs_weights, weight_name, dtype):
        """Regular matmul; also mark ``max(abs(lhs))`` as an engine output."""
        import tensorrt as trt

        from . import graph_ops

        rank = len(tuple(lhs.shape))
        absolute = network.add_unary(lhs, trt.UnaryOperation.ABS).get_output(0)
        reduced = network.add_reduce(
            absolute, trt.ReduceOperation.MAX, (1 << rank) - 1, True
        ).get_output(0)
        reduced.name = "amax_" + weight_name.replace(".", "_")
        network.mark_output(reduced)
        self.outputs[weight_name] = reduced.name
        return graph_ops.add_matmul_rhs_constant(
            network, lhs, lhs_width, rhs_width, rhs_weights, dtype=dtype
        )


def fetch_clips(out: Path, per_split: int, split: str = "validation") -> None:
    """Download ``per_split`` clips of each LibriSpeech subset as 16 kHz wav files with a .txt."""
    import io

    import pyarrow.parquet as parquet
    import soundfile
    from huggingface_hub import hf_hub_download

    out.mkdir(parents=True, exist_ok=True)
    for subset, parquet_path in _SPLITS[split].items():
        local = hf_hub_download(_DATASET, parquet_path, repo_type="dataset")
        table = parquet.read_table(local).slice(0, per_split).to_pylist()
        for index, row in enumerate(table):
            samples, rate = soundfile.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            if rate != _SAMPLE_RATE:
                raise RuntimeError(f"{subset} clip {index} is {rate} Hz, expected {_SAMPLE_RATE}")
            name = f"{split}-{subset}-{index:04d}"
            soundfile.write(out / f"{name}.wav", samples, _SAMPLE_RATE)
            (out / f"{name}.txt").write_text(row["text"], encoding="utf-8")
    print(f"wrote {len(list(out.glob('*.wav')))} clips to {out}")


def mel_spectrogram(samples: np.ndarray, filters: np.ndarray, window: np.ndarray):
    """Log-mel features ``[mel_bins, frames]`` of 16 kHz audio (the bundle's mel front end).

    ``filters`` is the ``[n_fft / 2 + 1, mel_bins]`` filterbank and ``window`` the
    checkpoint's analysis window; pre-emphasis 0.97, reflected centering, natural log.
    """
    import torch

    x = torch.tensor(samples, dtype=torch.float32)
    x = torch.cat([x[:1], x[1:] - 0.97 * x[:-1]])
    spectrum = torch.stft(
        x,
        n_fft=_N_FFT,
        hop_length=_HOP,
        win_length=len(window),
        window=torch.tensor(window, dtype=torch.float32),
        center=True,
        pad_mode="reflect",
        return_complex=True,
    )
    power = spectrum.abs() ** 2
    mel = torch.tensor(filters, dtype=torch.float32).t() @ power
    mel = torch.log(mel + _LOG_GUARD)
    return mel[:, : len(samples) // _HOP].numpy()


def _torch_dtype(trt_dtype):
    import tensorrt as trt
    import torch

    return {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
    }[trt_dtype]


class _PlanRunner:
    """A deserialized plan with preallocated input and output tensors on the GPU."""

    def __init__(self, plan: bytes):
        import tensorrt as trt
        import torch

        self.torch = torch
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self.engine = self.runtime.deserialize_cuda_engine(plan)
        self.context = self.engine.create_execution_context()
        self.tensors: dict[str, Any] = {}
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            shape = tuple(self.engine.get_tensor_shape(name))
            tensor = torch.zeros(
                shape, dtype=_torch_dtype(self.engine.get_tensor_dtype(name)), device="cuda"
            )
            self.tensors[name] = tensor
            self.context.set_tensor_address(name, tensor.data_ptr())

    def run(self) -> None:
        """Execute once on the current stream and wait for the result."""
        self.context.execute_async_v3(self.torch.cuda.current_stream().cuda_stream)
        self.torch.cuda.synchronize()


class StreamingEncoder:
    """Drives the first-step and steady-state perception plans like the native runtime."""

    def __init__(self, first_plan: bytes, next_plan: bytes, weights):
        self.runners = {"first": _PlanRunner(first_plan), "next": _PlanRunner(next_plan)}
        self.layers = int(weights["_enc_layers"])
        self.hidden = int(weights["_hidden"])
        self.cache_left = int(weights["_streaming_cache_left"])
        self.mel_bins = int(weights["_mel_bins"])
        self.output_dim = int(weights.get("_output_dim", self.hidden))

    def stream(self, mel: np.ndarray) -> np.ndarray:
        """Encode a whole utterance frame by frame; returns the projected ``[frames, dim]``."""
        import torch

        right, query = 0, 1
        total = mel.shape[1]
        channel = torch.zeros(self.layers, self.cache_left, self.hidden, device="cuda")
        time = torch.zeros(self.layers, self.hidden, 8, device="cuda")
        cache_len, start, frames = 0, 0, []
        while start < total:
            first = start == 0
            shift = 1 if first else _SUBSAMPLING
            pre = 0 if first else _PRE_ENCODE_CACHE
            new = min(shift, total - start)
            chunk = np.zeros((self.mel_bins, pre + shift), dtype=np.float32)
            lo = max(start - pre, 0)
            chunk[:, pre - (start - lo) : pre] = mel[:, lo:start]
            chunk[:, pre : pre + new] = mel[:, start : start + new]
            drop = 0 if first else 2
            sub = pre + new
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
            runner = self.runners["first" if first else "next"]
            runner.tensors["mel_features"].copy_(torch.tensor(chunk))
            runner.tensors["cache_last_channel"].copy_(channel)
            runner.tensors["cache_last_time"].copy_(time)
            runner.tensors["encoder_mask"].copy_(torch.tensor(mask))
            runner.run()
            frames.append(runner.tensors["audio_embeddings"][:valid_query].float().cpu().numpy())
            channel = runner.tensors["cache_last_channel_next"].clone()
            time = runner.tensors["cache_last_time_next"].clone()
            cache_len = min(self.cache_left, cache_len + query)
            start += new
            if new < shift:
                break
        return np.concatenate(frames, 0) if frames else np.zeros((0, self.output_dim), np.float32)


class ThinkerRunner:
    """One thinker plan with its recurrent state, stepped one 80 ms frame at a time."""

    def __init__(self, plan: bytes, num_mamba: int, num_attn: int, max_cache_length: int):
        self.plan = _PlanRunner(plan)
        self.torch = self.plan.torch
        self.t = self.plan.tensors
        self.num_mamba, self.num_attn = num_mamba, num_attn
        self.max_cache_length = max_cache_length
        self.position = 0
        self.reset()

    def reset(self) -> None:
        """Clear the Mamba state and the attention cache."""
        for index in range(self.num_mamba):
            self.t[f"conv_state_{index}"].zero_()
            self.t[f"ssm_state_{index}"].zero_()
        self.position = 0

    def step(self, text: int, timeline: int, function: int, audio, use_audio: bool):
        """Run one frame; returns the text and function logits as device tensors."""
        t = self.t
        t["text_token_id"].fill_(text)
        t["timeline_token_id"].fill_(timeline)
        t["function_token_id"].fill_(function)
        if audio is None:
            t["audio_embed"].zero_()
        else:
            t["audio_embed"].copy_(audio.reshape(1, -1))
        t["use_audio_embed"].fill_(1.0 if use_audio else 0.0)
        mask = t["attention_mask"]
        mask.fill_(_MASKED)
        mask[:, : min(self.position, self.max_cache_length)] = 0.0
        mask[:, -1] = 0.0
        self.plan.run()
        for index in range(self.num_mamba):
            t[f"conv_state_{index}"].copy_(t[f"present_conv_{index}"])
            t[f"ssm_state_{index}"].copy_(t[f"present_ssm_{index}"])
        slot = min(self.position, self.max_cache_length - 1)
        for index in range(self.num_attn):
            for kind in ("k", "v"):
                cache, present = t[f"cache_{kind}_{index}"], t[f"present_{kind}_{index}"]
                if self.position >= self.max_cache_length:
                    cache[:-1].copy_(cache[1:].clone())
                cache[slot].copy_(present[0])
        self.position = min(self.position + 1, self.max_cache_length)
        return t["logits"][0], t["function_logits"][0]


def run_conversation(
    thinker: ThinkerRunner,
    audio_frames: np.ndarray,
    speech_frames: int,
    prompt_ids: list[int],
    observe=None,
    forced: list | None = None,
):
    """Run one conversation turn and return the per-step records.

    ``audio_frames`` are the projected perception frames of the speech followed by silence;
    ``speech_frames`` of them are speech. The prompt is prefilled with the timeline channel,
    then each audio frame goes through the thinker with the previous text and function
    tokens. The agent response starts after the speech with a forced BOS token and runs
    until EOS. ``observe(step_index, text_logits, function_logits, inputs)`` is called after
    every step. ``forced`` (optional) is a list of ``(text, function)`` output tokens to feed
    back instead of the engine's own argmax (teacher forcing). Returns the step inputs and
    the produced tokens.
    """
    import torch

    thinker.reset()
    inputs, produced = [], []
    for prompt_id in prompt_ids:
        record = (_PAD_TOKEN, prompt_id, _PAD_TOKEN, None, False)
        text_logits, function_logits = thinker.step(*record)
        inputs.append(record)
        if observe is not None:
            observe(len(inputs) - 1, text_logits, function_logits, record)
    previous_text, previous_function = _PAD_TOKEN, _PAD_TOKEN
    response_start = None
    frame = 0
    while frame < len(audio_frames):
        if response_start is not None and frame >= response_start + _MAX_RESPONSE_FRAMES:
            break
        record = (
            previous_text,
            _PAD_TOKEN,
            previous_function,
            torch.tensor(audio_frames[frame], device="cuda"),
            True,
        )
        text_logits, function_logits = thinker.step(*record)
        inputs.append(record)
        if observe is not None:
            observe(len(inputs) - 1, text_logits, function_logits, record)
        text, function = int(text_logits.argmax()), int(function_logits.argmax())
        if forced is not None and len(produced) < len(forced):
            text, function = forced[len(produced)]
        if frame + 1 == speech_frames and response_start is None:
            response_start = frame + 1
            text = _BOS_TOKEN
        elif response_start is None and text == _BOS_TOKEN:
            text = _PAD_TOKEN
        produced.append((text, function))
        previous_text, previous_function = text, function
        frame += 1
        if response_start is not None and text == _EOS_TOKEN:
            break
    return inputs, produced


def load_prompt_ids(tokenizer_path: Path, prompt: str) -> list[int]:
    """BOS + tokenized prompt + EOS, the way the native runtime prefills the system prompt."""
    from tokenizers import Tokenizer

    body = Tokenizer.from_file(str(tokenizer_path)).encode(prompt, add_special_tokens=False).ids
    return [_BOS_TOKEN, *body, _EOS_TOKEN]


def clip_frames(encoder: StreamingEncoder, samples: np.ndarray, filters, window):
    """Projected perception frames of a clip plus trailing silence, and the speech frame count."""
    speech_frames = max(1, len(samples) // _FRAME_SAMPLES)
    padded = np.concatenate([samples, np.zeros(int(_SILENCE_SECONDS * _SAMPLE_RATE), np.float32)])
    return encoder.stream(mel_spectrogram(padded, filters, window)), speech_frames


def build_calibration_plans(model_dir: Path, max_cache_length: int):
    """Build the fp32 perception plans and the instrumented thinker plan, in memory."""
    import json as _json

    from . import native_core, streaming_perception
    from .model import _THINKER_CONFIG, _thinker_config, _voicechat_sections

    raw = _json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    stt, _ = _voicechat_sections(raw)
    perception_weights = native_core.load_perception_weights(str(model_dir), stt)
    first = streaming_perception._build_streaming_encoder(perception_weights, 0, first_step=True)
    steady = streaming_perception._build_streaming_encoder(perception_weights, 0, first_step=False)
    thinker = _thinker_config(Path(model_dir), "fp32")
    recorder = CalibrationContext()
    weights = native_core.VoiceChatThinkerBuilder().load_weights(str(model_dir), thinker)
    plan = native_core.build_thinker_engine(thinker, weights, max_cache_length, quant=recorder)
    layer_types = native_core._parse_layer_types(str(_THINKER_CONFIG["hybrid_override_pattern"]))
    return perception_weights, (first, steady), (thinker, plan, layer_types, recorder)


def calibrate(model_dir: Path, clips: Path, max_cache_length: int, tokenizer: Path) -> None:
    """Run the clips through the instrumented plans and store the projection input scales."""
    import soundfile

    perception_weights, perception_plans, thinker_parts = build_calibration_plans(
        model_dir, max_cache_length
    )
    thinker_config, thinker_plan, layer_types, recorder = thinker_parts
    encoder = StreamingEncoder(*perception_plans, perception_weights)
    thinker = ThinkerRunner(
        thinker_plan,
        num_mamba=layer_types.count("mamba2"),
        num_attn=layer_types.count("attention"),
        max_cache_length=max_cache_length,
    )
    filters, window = perception_weights["mel_filterbank"], perception_weights["mel_window"]
    prompts = [load_prompt_ids(tokenizer, prompt) for prompt in _SYSTEM_PROMPTS]

    running: dict[str, Any] = {}

    def observe(_step, _text_logits, _function_logits, _record):
        for name, tensor_name in recorder.outputs.items():
            value = thinker.t[tensor_name].max()
            previous = running.get(name)
            running[name] = (
                value.clone() if previous is None else thinker.torch.maximum(previous, value)
            )

    files = sorted(clips.glob("*.wav"))
    if not files:
        raise RuntimeError(f"no wav clips in {clips}")
    for index, path in enumerate(files):
        samples, rate = soundfile.read(path, dtype="float32")
        if rate != _SAMPLE_RATE:
            raise RuntimeError(f"{path} is {rate} Hz, expected {_SAMPLE_RATE}")
        frames, speech_frames = clip_frames(encoder, samples, filters, window)
        run_conversation(thinker, frames, speech_frames, prompts[index % len(prompts)], observe)
        if (index + 1) % 8 == 0:
            print(f"[{index + 1}/{len(files)}]", flush=True)

    amax = {name: float(value) for name, value in running.items()}
    scales = {name: value / FP8_E4M3_MAX for name, value in amax.items() if value > 0}
    missing = [name for name in projection_names(layer_types) if name not in scales]
    if missing:
        raise RuntimeError(
            f"no calibrated input range for {len(missing)} projections: {missing[:4]}"
        )
    key = scale_key(
        thinker_config.hidden_size,
        thinker_config.num_hidden_layers,
        thinker_config.intermediate_size,
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
    fetch.add_argument("--per-split", type=int, default=48)
    fetch.add_argument("--split", choices=sorted(_SPLITS), default="validation")
    run = commands.add_parser("calibrate")
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--clips", type=Path, required=True)
    run.add_argument("--tokenizer", type=Path, required=True, help="tokenizer.json of the LLM")
    run.add_argument("--max-cache-length", type=int, default=8192)
    args = parser.parse_args()
    if args.command == "fetch":
        fetch_clips(args.out, args.per_split, args.split)
    else:
        calibrate(args.model_dir, args.clips, args.max_cache_length, args.tokenizer)


if __name__ == "__main__":
    main()
