# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate FP8 activation scales for a GLM-ASR model shape.

Not used when building an engine. Calibrate each supported model shape and
commit the updated ``fp8_activation_scales.json``::

    python -m families.glmasr.calibrate_fp8_scales fetch --out clips
    python -m families.glmasr.calibrate_fp8_scales calibrate \
        --model-dir <original GLM-ASR checkpoint> --clips clips

``fetch`` downloads LibriSpeech validation clips (clean and other) as 16 kHz wav files
with their transcripts.
``calibrate`` loads the model in PyTorch, runs ModelOpt ``FP8_DEFAULT_CFG`` while the
model transcribes the clips (so the encoder sees real speech and the decoder sees real
prefill and decode-time activations), and stores ``input_quantizer.amax / 448`` for every
audio encoder, projector and decoder projection under the model-shape key (see
``quantization.scale_key``). ``fetch`` needs network access;
``calibrate`` needs a GPU.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .quantization import ACTIVATION_SCALES_FILENAME, FP8_E4M3_MAX, scale_key

_DATASET = "openslr/librispeech_asr"
_SPLITS = {"clean": "clean/validation/0000.parquet", "other": "other/validation/0000.parquet"}
_SAMPLE_RATE = 16000

_DECODER_STEM = {
    "self_attn.q_proj": "w_q",
    "self_attn.k_proj": "w_k",
    "self_attn.v_proj": "w_v",
    "self_attn.o_proj": "w_o",
    "mlp.gate_proj": "w_gate",
    "mlp.up_proj": "w_up",
    "mlp.down_proj": "w_down",
}
_ENCODER_STEM = {
    "self_attn.q_proj": "w_q",
    "self_attn.k_proj": "w_k",
    "self_attn.v_proj": "w_v",
    "self_attn.o_proj": "w_o",
    "mlp.fc1": "w_fc1",
    "mlp.fc2": "w_fc2",
}
_DECODER_NAME = re.compile(
    r"language_model\.(?:model\.)?layers\.(\d+)\.("
    + "|".join(re.escape(stem) for stem in _DECODER_STEM)
    + r")$"
)
_ENCODER_NAME = re.compile(
    r"audio_tower\.layers\.(\d+)\.(" + "|".join(re.escape(stem) for stem in _ENCODER_STEM) + r")$"
)


def weight_name(module_name: str) -> str | None:
    """Internal weight name of an HF module path, or None when it is not a quantized projection."""
    if (match := _DECODER_NAME.search(module_name)) is not None:
        return f"layer.{match.group(1)}.{_DECODER_STEM[match.group(2)]}"
    if (match := _ENCODER_NAME.search(module_name)) is not None:
        return f"audio.layer.{match.group(1)}.{_ENCODER_STEM[match.group(2)]}"
    return None


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


def calibrate(model_dir: Path, clips: Path, max_new_tokens: int) -> None:
    """Run ModelOpt FP8 calibration over the clips and store the activation scales under the shape key."""
    import modelopt.torch.quantization as mtq
    import soundfile
    import torch
    from transformers import AutoProcessor, GlmAsrForConditionalGeneration

    from .config import ModelConfig, audio_encoder_config

    config = ModelConfig.from_dir(model_dir)
    key = scale_key(config)
    processor = AutoProcessor.from_pretrained(model_dir)
    model = (
        GlmAsrForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float16).cuda().eval()
    )
    files = sorted(clips.glob("*.wav"))
    if not files:
        raise RuntimeError(f"no wav clips in {clips}")

    def forward_loop(calibration_model) -> None:
        """Transcribe every clip once."""
        with torch.no_grad():
            for path in files:
                samples, rate = soundfile.read(path, dtype="float32")
                if rate != _SAMPLE_RATE:
                    raise RuntimeError(f"{path} is {rate} Hz, expected {_SAMPLE_RATE}")
                inputs = processor.apply_transcription_request(samples).to("cuda", torch.float16)
                calibration_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    model = mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop)

    scales: dict[str, float] = {}
    skipped: list[str] = []
    for name, module in model.named_modules():
        internal = weight_name(name)
        if internal is None or not hasattr(module, "input_quantizer"):
            continue
        amax = module.input_quantizer.amax
        if amax is None or not float(amax) > 0:
            # ModelOpt left this projection unquantized; it gets no entry and stays unquantized.
            skipped.append(name)
            continue
        scales[internal] = float(amax) / FP8_E4M3_MAX
    if skipped:
        print(
            f"no activation amax, left unquantized: {len(skipped)} projections, e.g. {skipped[:3]}"
        )
    # Every layer must keep its MLP projections.
    encoder = audio_encoder_config(config.raw)
    for side, count, prefix, stems in (
        ("decoder", int(config.num_hidden_layers), "layer", ("w_gate", "w_up", "w_down")),
        ("encoder", int(encoder.num_hidden_layers), "audio.layer", ("w_fc1", "w_fc2")),
    ):
        for layer in range(count):
            for stem in stems:
                if f"{prefix}.{layer}.{stem}" not in scales:
                    raise RuntimeError(f"{side} layer {layer} has no calibrated {stem}")

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
    run.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()
    if args.command == "fetch":
        fetch_clips(args.out, args.per_split)
    else:
        calibrate(args.model_dir, args.clips, args.max_new_tokens)


if __name__ == "__main__":
    main()
