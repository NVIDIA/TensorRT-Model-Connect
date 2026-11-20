# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate FP8 activation scales for a DeepSeek-OCR-2 model shape.

Not used when building an engine. Calibrate each supported model shape and
commit the updated ``fp8_activation_scales.json``::

    python -m families.deepseek_ocr.calibrate_fp8_scales calibrate \
        --model-dir <original DeepSeek-OCR-2 checkpoint> --images <directory of document images>

``calibrate`` loads the checkpoint's own remote-code model in PyTorch (bf16, eager attention) and lets
it read the images with the single 768 x 768 view the TensorRT bundle uses, so the decoder sees real
image embeddings at prefill and real decode-time activations. Forward pre-hooks record
``max(abs(input))`` of every quantized decoder projection (attention q/k/v, the dense MLP of layer 0
and the shared experts); the scale is ``amax / 448``. This is the statistic ModelOpt's ``FP8_DEFAULT_CFG``
max calibration stores. ModelOpt itself is not used here because it wraps every one of the 64 routed
experts per layer, which are weight-only in the engine and need no activation scale. The routed
experts are therefore not calibrated.

It needs a GPU, a directory of images and ``transformers==4.46.3`` (the checkpoint's remote code does
not import on transformers 5) with ``addict``, ``easydict``, ``einops`` and ``timm`` installed.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .quantization import (
    ACTIVATION_SCALES_FILENAME,
    ATTENTION_PROJECTIONS,
    DENSE_MLP_PROJECTIONS,
    FP8_E4M3_MAX,
    SHARED_EXPERT_PROJECTIONS,
    UNQUANTIZED,
    scale_key,
)

_STEM = dict((*ATTENTION_PROJECTIONS, *DENSE_MLP_PROJECTIONS, *SHARED_EXPERT_PROJECTIONS))
_NAME = re.compile(
    r"^(?:language\.)?model\.layers\.(\d+)\.(" + "|".join(re.escape(stem) for stem in _STEM) + r")$"
)
_PROMPTS = (
    "<image>\nExtract the text from this image.",
    "<image>\nFree OCR.",
    "<image>\n<|grounding|>Convert the document to markdown.",
    "<image>\nExtract the text from this image.",
)
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
_VIEW_SIZE = 768


def weight_name(module_name: str) -> str | None:
    """Internal weight name of an HF module path, or None when it is not a quantized projection."""
    if (match := _NAME.search(module_name)) is None:
        return None
    return f"layer.{match.group(1)}.{_STEM[match.group(2)]}"


def calibrate(model_dir: Path, images: Path, count: int, max_new_tokens: int) -> None:
    """Record input amax of every quantized projection over the images and store the scales."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    from .config import ModelConfig

    config = ModelConfig.from_dir(model_dir)
    key = scale_key(config)
    files = sorted(p for p in images.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)[:count]
    if not files:
        raise RuntimeError(f"no images in {images}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(
            model_dir,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        .cuda()
        .eval()
    )

    amax: dict[str, float] = {}

    def make_hook(name: str):
        def hook(_module, args):
            value = float(args[0].detach().abs().amax().float())
            amax[name] = max(amax.get(name, 0.0), value)

        return hook

    for module_name, module in model.named_modules():
        internal = weight_name(module_name)
        if internal is not None and not UNQUANTIZED.match(internal):
            module.register_forward_pre_hook(make_hook(internal))

    generate = model.generate

    def capped_generate(*args, **kwargs):
        kwargs["max_new_tokens"] = max_new_tokens
        return generate(*args, **kwargs)

    model.generate = capped_generate
    scratch = Path("/tmp/deepseek_ocr_calibration")
    for index, path in enumerate(files):
        model.infer(
            tokenizer,
            prompt=_PROMPTS[index % len(_PROMPTS)],
            image_file=str(path),
            output_path=str(scratch),
            base_size=_VIEW_SIZE,
            image_size=_VIEW_SIZE,
            crop_mode=False,
            save_results=False,
            eval_mode=True,
        )

    scales = {name: value / FP8_E4M3_MAX for name, value in amax.items() if value > 0}
    # Every layer must keep its attention input projections and its MLP or shared experts.
    first_moe = int(config.raw.get("first_k_dense_replace", 1))
    for layer in range(int(config.num_hidden_layers)):
        mlp = DENSE_MLP_PROJECTIONS if layer < first_moe else SHARED_EXPERT_PROJECTIONS
        for _, stem in (*ATTENTION_PROJECTIONS, *mlp):
            if f"layer.{layer}.{stem}" not in scales:
                raise RuntimeError(f"layer {layer} has no calibrated {stem}")

    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    table[key] = dict(sorted(scales.items()))
    path.write_text(json.dumps(dict(sorted(table.items())), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(scales)} scales for {key!r} to {path} from {len(files)} images")


def main() -> None:
    """Parse the command line and run the calibrate command."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("calibrate")
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--images", type=Path, required=True)
    run.add_argument("--count", type=int, default=128)
    run.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    calibrate(args.model_dir, args.images, args.count, args.max_new_tokens)


if __name__ == "__main__":
    main()
