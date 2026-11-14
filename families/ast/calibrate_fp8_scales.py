# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate FP8 activation scales for an AST checkpoint.

Not used when building an engine. Calibrate each supported checkpoint and commit the
updated ``fp8_activation_scales.json``::

    python -m families.ast.calibrate_fp8_scales fetch --out clips
    python -m families.ast.calibrate_fp8_scales calibrate \
        --model-dir <original AST checkpoint> --clips clips

``fetch`` downloads AudioSet clips (the first rows of one balanced-train shard) as 16 kHz
wav files. ``calibrate`` loads the model with Hugging Face transformers, runs ModelOpt
``FP8_DEFAULT_CFG`` over the clips, and stores ``input_quantizer.amax / 448`` for every
encoder projection under the checkpoint key (see ``quantization.scale_key``). ``fetch``
needs network access; ``calibrate`` needs a GPU.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .quantization import ACTIVATION_SCALES_FILENAME, FP8_E4M3_MAX, scale_key

_DATASET = "agkphysics/AudioSet"
_SHARD = "data/bal_train/00.parquet"
_SAMPLE_RATE = 16000

_ATTENTION_STEM = {"q_proj": "query", "k_proj": "key", "v_proj": "value", "o_proj": "attn_out"}
_PROJECTION_NAME = re.compile(
    r"layers\.(?P<layer>\d+)\."
    r"(?:attention\.(?P<proj>q_proj|k_proj|v_proj|o_proj)|mlp\.(?P<fc>fc[12]))$"
)


def weight_name(module_name: str) -> str | None:
    """Short name of a Hugging Face module path, or None when it is not a quantized projection."""
    match = _PROJECTION_NAME.search(module_name)
    if match is None:
        return None
    layer = match.group("layer")
    if match.group("proj"):
        return f"layer.{layer}.{_ATTENTION_STEM[match.group('proj')]}"
    return f"layer.{layer}.{match.group('fc')}"


def fetch_clips(out: Path, count: int) -> None:
    """Write the first ``count`` clips of the shard as 16 kHz wav files."""
    import io

    import pyarrow.parquet as parquet
    import soundfile
    from huggingface_hub import hf_hub_download
    from scipy.signal import resample_poly

    out.mkdir(parents=True, exist_ok=True)
    local = hf_hub_download(_DATASET, _SHARD, repo_type="dataset")
    table = parquet.ParquetFile(local).read_row_group(0).slice(0, count).to_pylist()
    for index, row in enumerate(table):
        samples, rate = soundfile.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        if rate != _SAMPLE_RATE:
            samples = resample_poly(samples, _SAMPLE_RATE, rate).astype("float32")
        soundfile.write(out / f"clip-{index:04d}.wav", samples, _SAMPLE_RATE)
    print(f"wrote {len(list(out.glob('*.wav')))} clips to {out}")


def calibrate(model_dir: Path, clips: Path) -> None:
    """Run ModelOpt FP8 calibration over the clips and store the scales under the checkpoint key."""
    import modelopt.torch.quantization as mtq
    import soundfile
    import torch
    from transformers import ASTFeatureExtractor, ASTForAudioClassification

    from . import config as config_module
    from .model import _load

    weights, shapes = _load(model_dir)
    key = scale_key(config_module.resolve(model_dir, shapes), weights)
    extractor = ASTFeatureExtractor.from_pretrained(model_dir)
    model = ASTForAudioClassification.from_pretrained(model_dir, dtype=torch.float16).cuda().eval()
    files = sorted(clips.glob("*.wav"))
    if not files:
        raise RuntimeError(f"no wav clips in {clips}")

    def forward_loop(calibration_model) -> None:
        """Classify every clip once."""
        with torch.no_grad():
            for path in files:
                samples, rate = soundfile.read(path, dtype="float32")
                if rate != _SAMPLE_RATE:
                    raise RuntimeError(f"{path} is {rate} Hz, expected {_SAMPLE_RATE}")
                features = extractor(samples, sampling_rate=_SAMPLE_RATE, return_tensors="pt")
                calibration_model(features["input_values"].to("cuda", torch.float16))

    # Only the linear layers are quantized; skip ModelOpt's attention wrapper.
    from modelopt.torch.quantization.plugins import huggingface as modelopt_huggingface

    modelopt_huggingface.CUSTOM_MODEL_PLUGINS.discard(
        modelopt_huggingface.register_hf_attentions_on_the_fly
    )
    model = mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop)

    scales: dict[str, float] = {}
    for name, module in model.named_modules():
        short = weight_name(name)
        if short is None or not hasattr(module, "input_quantizer"):
            continue
        amax = module.input_quantizer.amax
        if amax is None or not float(amax) > 0:
            continue
        scales[short] = float(amax) / FP8_E4M3_MAX
    layers = int(model.config.num_hidden_layers)
    expected = {
        f"layer.{i}.{s}"
        for i in range(layers)
        for s in ("query", "key", "value", "attn_out", "fc1", "fc2")
    }
    if set(scales) != expected:
        raise RuntimeError(
            f"calibrated {len(scales)} of {len(expected)} projections; refusing a partial table"
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
    fetch.add_argument("--count", type=int, default=256)
    run = commands.add_parser("calibrate")
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--clips", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fetch":
        fetch_clips(args.out, args.count)
    else:
        calibrate(args.model_dir, args.clips)


if __name__ == "__main__":
    main()
