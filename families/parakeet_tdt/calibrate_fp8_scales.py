# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate FP8 activation scales for a Parakeet TDT model config.

Not used when building an engine. Calibrate each supported model config and commit the
updated ``fp8_activation_scales.json``::

    python -m families.parakeet_tdt.calibrate_fp8_scales fetch --out clips
    python -m families.parakeet_tdt.calibrate_fp8_scales calibrate \
        --model-dir <original Parakeet TDT checkpoint> --clips clips

``fetch`` downloads LibriSpeech validation clips (clean and other) as 16 kHz wav files
with their transcripts.
``calibrate`` loads the model in PyTorch (Hugging Face ``ParakeetForTDT``), runs the encoder
over the clips, records the maximum of ``abs(input)`` of every quantized encoder projection
(the statistic ModelOpt ``FP8_DEFAULT_CFG`` max calibration stores) and writes
``amax / 448`` under the model config key (see ``quantization.scale_key``). The predictor and the joint
network are not quantized and are not calibrated. ``fetch`` needs network access; ``calibrate``
needs a GPU.
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

_FFN_NAME = re.compile(r"layers\.(?P<layer>\d+)\.feed_forward(?P<ff>[12])\.linear(?P<lin>[12])$")
_ATTENTION_NAME = re.compile(r"layers\.(?P<layer>\d+)\.self_attn\.(?P<proj>[qkvo])_proj$")


def weight_name(module_name: str) -> str | None:
    """Internal weight name of an encoder module path, or None when it is not quantized."""
    match = _FFN_NAME.search(module_name)
    if match is not None:
        return f"el.{match.group('layer')}.ff{match.group('ff')}.w{match.group('lin')}"
    match = _ATTENTION_NAME.search(module_name)
    if match is not None:
        return f"el.{match.group('layer')}.w_{match.group('proj')}"
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


def calibrate(model_dir: Path, clips: Path) -> None:
    """Collect the input amax of every quantized encoder projection and store the scales under the model config key.

    ModelOpt's ``FP8_DEFAULT_CFG`` stores the per-tensor maximum of ``abs(input)`` of each
    linear layer (max calibration). ModelOpt's Hugging Face attention plugin cannot wrap the
    Parakeet attention call (it passes ``query``, ``key`` and ``value`` as keywords), so the
    same statistic is collected here with forward pre-hooks on the encoder linear layers.
    """
    import soundfile
    import torch
    from transformers import AutoProcessor, ParakeetForTDT

    from .config import ParakeetTDTConfig

    config = ParakeetTDTConfig.from_dir(model_dir)
    config.validate_supported_checkpoint()
    key = scale_key(config)
    processor = AutoProcessor.from_pretrained(model_dir)
    model = ParakeetForTDT.from_pretrained(model_dir, dtype=torch.float16).cuda().eval()
    files = sorted(clips.glob("*.wav"))
    if not files:
        raise RuntimeError(f"no wav clips in {clips}")

    amax: dict[str, torch.Tensor] = {}

    def collector(internal: str):
        def hook(module, args):
            value = args[0].detach().abs().amax().float()
            amax[internal] = torch.maximum(amax[internal], value) if internal in amax else value

        return hook

    handles = []
    for name, module in model.encoder.named_modules():
        internal = weight_name(name)
        if internal is not None and isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_pre_hook(collector(internal)))
    if not handles:
        raise RuntimeError("no encoder projection matched the quantized projection names")

    with torch.no_grad():
        for path in files:
            samples, rate = soundfile.read(path, dtype="float32")
            if rate != _SAMPLE_RATE:
                raise RuntimeError(f"{path} is {rate} Hz, expected {_SAMPLE_RATE}")
            inputs = processor(samples, sampling_rate=_SAMPLE_RATE, return_tensors="pt")
            kwargs = {"input_features": inputs.input_features.to("cuda", torch.float16)}
            if "attention_mask" in inputs:
                kwargs["attention_mask"] = inputs.attention_mask.to("cuda")
            model.encoder(**kwargs)
    for handle in handles:
        handle.remove()

    scales = {name: float(value) / FP8_E4M3_MAX for name, value in amax.items() if float(value) > 0}
    # Every layer must keep its projections.
    for layer in range(int(config.encoder_layers)):
        for stem in ("ff1.w1", "ff1.w2", "ff2.w1", "ff2.w2", "w_q", "w_k", "w_v", "w_o"):
            if f"el.{layer}.{stem}" not in scales:
                raise RuntimeError(f"encoder layer {layer} has no calibrated {stem}")

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
    args = parser.parse_args()
    if args.command == "fetch":
        fetch_clips(args.out, args.per_split)
    else:
        calibrate(args.model_dir, args.clips)


if __name__ == "__main__":
    main()
