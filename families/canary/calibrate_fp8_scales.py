# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate FP8 activation scales for a Canary model shape.

Not used when building an engine. Calibrate each supported model shape and
commit the updated ``fp8_activation_scales.json``::

    python -m families.canary.calibrate_fp8_scales fetch --out clips
    python -m families.canary.calibrate_fp8_scales calibrate \
        --model-dir <Canary checkpoint with model.safetensors> --clips clips

``fetch`` downloads LibriSpeech validation clips (clean and other) as 16 kHz wav files
with their transcripts.
``calibrate`` loads the model with Hugging Face transformers (the checkpoint repository
ships ``model.safetensors`` next to the ``.nemo`` archive, converted from the same
weights), runs ModelOpt ``FP8_DEFAULT_CFG`` while the model transcribes the clips (so the
decoder sees real decode-time activations and the cross-attention sees the real encoder
output), and stores ``input_quantizer.amax / 448`` for every encoder and decoder
projection under the model-shape key (see ``quantization.scale_key``). ``fetch`` needs
network access; ``calibrate`` needs a GPU.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .quantization import ACTIVATION_SCALES_FILENAME, FP8_E4M3_MAX

_DATASET = "openslr/librispeech_asr"
_SPLITS = {"clean": "clean/validation/0000.parquet", "other": "other/validation/0000.parquet"}
_SAMPLE_RATE = 16000

_ATTENTION_STEM = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o"}
_PROJECTION_NAME = re.compile(
    r"(?P<side>encoder|decoder)\.layers\.(?P<layer>\d+)\."
    r"(?:(?P<attn>self_attn|encoder_attn)\.(?P<proj>q_proj|k_proj|v_proj|o_proj)"
    r"|feed_forward(?P<ff>[12])\.linear(?P<lin>[12])"
    r"|mlp\.fc(?P<fc>[12]))$"
)

# Module paths of the Hugging Face model and the weight names they map to.
EXAMPLE_NAMES = {
    "model.encoder.layers.3.self_attn.o_proj": "el.3.w_o",
    "model.encoder.layers.0.feed_forward2.linear1": "el.0.ff2.w1",
    "model.decoder.layers.2.self_attn.k_proj": "layer.2.w_k",
    "model.decoder.layers.2.encoder_attn.v_proj": "layer.2.xw_v",
    "model.decoder.layers.2.mlp.fc2": "layer.2.w_fc2",
}


def weight_name(module_name: str) -> str | None:
    """Weight name of a Hugging Face module path, or None when it is not a quantized projection."""
    match = _PROJECTION_NAME.search(module_name)
    if match is None:
        return None
    layer = match.group("layer")
    if match.group("side") == "encoder":
        if match.group("ff"):
            return f"el.{layer}.ff{match.group('ff')}.w{match.group('lin')}"
        if match.group("attn") != "self_attn":
            return None
        return f"el.{layer}.w_{_ATTENTION_STEM[match.group('proj')]}"
    if match.group("fc"):
        return f"layer.{layer}.w_fc{match.group('fc')}"
    if match.group("attn") is None:
        return None
    stem = _ATTENTION_STEM[match.group("proj")]
    prefix = "x" if match.group("attn") == "encoder_attn" else ""
    return f"layer.{layer}.{prefix}w_{stem}"


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


def shape_key(config) -> str:
    """Model-shape key of a Hugging Face Canary config (same format as ``quantization.scale_key``)."""
    encoder, decoder = config.encoder_config, config.decoder_config
    return (
        f"canary-h{int(encoder.hidden_size)}"
        f"-e{int(encoder.num_hidden_layers)}"
        f"-d{int(decoder.num_hidden_layers)}"
        f"-m{int(encoder.num_mel_bins)}"
    )


def calibrate(model_dir: Path, clips: Path, max_new_tokens: int) -> None:
    """Run ModelOpt FP8 calibration over the clips and store the activation scales under the shape key."""
    import modelopt.torch.quantization as mtq
    import soundfile
    import torch
    from transformers import AutoProcessor, CanaryForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_dir)
    model = (
        CanaryForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float16).cuda().eval()
    )
    key = shape_key(model.config)
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
                inputs = processor.apply_transcription_request(
                    samples, source_language="en", punctuation=True
                ).to("cuda")
                inputs = {
                    name: value.half() if value.is_floating_point() else value
                    for name, value in inputs.items()
                }
                calibration_model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Only the linear layers are quantized. ModelOpt's KV-cache attention wrapper calls the
    # attention function with ``query_states=`` keywords, but the encoder attention (a Parakeet
    # module) passes ``query=``, ``key=`` and ``value=``, so the wrapper fails; skip it.
    from modelopt.torch.quantization.plugins import huggingface as modelopt_huggingface

    modelopt_huggingface.CUSTOM_MODEL_PLUGINS.discard(
        modelopt_huggingface.register_hf_attentions_on_the_fly
    )
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
    # Every layer must keep its feed-forward projections.
    encoder_layers = int(model.config.encoder_config.num_hidden_layers)
    decoder_layers = int(model.config.decoder_config.num_hidden_layers)
    for layer in range(encoder_layers):
        for stem in ("ff1.w1", "ff1.w2", "ff2.w1", "ff2.w2"):
            if f"el.{layer}.{stem}" not in scales:
                raise RuntimeError(f"encoder layer {layer} has no calibrated {stem}")
    for layer in range(decoder_layers):
        for stem in ("w_fc1", "w_fc2"):
            if f"layer.{layer}.{stem}" not in scales:
                raise RuntimeError(f"decoder layer {layer} has no calibrated {stem}")

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
