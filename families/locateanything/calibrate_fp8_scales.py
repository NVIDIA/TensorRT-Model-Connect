# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate FP8 activation scales for a LocateAnything model shape.

Not used when building an engine. Calibrate each supported model shape and
commit the updated ``fp8_activation_scales.json``::

    python -m families.locateanything.calibrate_fp8_scales calibrate \
        --model-dir <original LocateAnything checkpoint> --images <directory of jpg images> \
        --classnames <file with one class name per line, ordered like the .cls files>

``calibrate`` loads the model in PyTorch (the checkpoint's own remote code, with the same
tokenizer, rotary-buffer repair and 448x448 patchified image contract as the family's
reference), runs ModelOpt ``FP8_DEFAULT_CFG`` while the model answers grounding and pointing
prompts about the images (so the language model sees real image embeddings at prefill and real
decode-time activations), and stores ``input_quantizer.amax / 448`` for every language-model
projection under the model-shape key (see ``quantization.scale_key``). Each ``<name>.jpg`` may
have a ``<name>.cls`` file with the index of its class name; images without it use a generic
description. It needs a GPU, ModelOpt and a directory of images.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .quantization import (
    ACTIVATION_SCALES_FILENAME,
    FP8_E4M3_MAX,
    PROJECTIONS,
    UNQUANTIZED,
    scale_key,
)

_STEM = dict(PROJECTIONS)
_NAME = re.compile(
    r"language_model\.(?:model\.)?layers\.(\d+)\.("
    + "|".join(re.escape(stem) for stem in _STEM)
    + r")$"
)
_PROMPTS = (
    "Locate a single instance that matches the following description: {name}.",
    "Point to: {name}.",
)


def weight_name(module_name: str) -> str | None:
    """Internal weight name of an HF module path, or None when it is not a language-model projection."""
    if (match := _NAME.search(module_name)) is None:
        return None
    return f"layer.{match.group(1)}.{_STEM[match.group(2)]}"


def description(path: Path, names: list[str]) -> str:
    """The class name stored next to the image, or a generic description."""
    cls = path.with_suffix(".cls")
    if names and cls.is_file():
        return names[int(cls.read_text().split()[0])].split(",")[0].strip()
    return "object"


def _accept_new_loader_arguments() -> None:
    """Let the checkpoint's remote code load under a newer transformers.

    The published modeling files override ``_check_and_adjust_attn_implementation`` with an
    older signature; newer transformers passes extra keyword arguments to it. Every model
    class created from now on drops the arguments its override does not know.
    """
    import functools

    from transformers import PreTrainedModel

    def wrap(cls) -> None:
        if "all_tied_weights_keys" not in vars(cls):
            # Remote classes that never call post_init() lack the attribute newer code reads.
            cls.all_tied_weights_keys = {}
        tied = cls.__dict__.get("_tied_weights_keys")
        if isinstance(tied, (list, tuple)):
            # Older remote code lists the tied weights; newer transformers wants a mapping.
            cls._tied_weights_keys = {
                key: "model.embed_tokens.weight" for key in tied if key.startswith("lm_head")
            }
        original = cls.__dict__.get("_check_and_adjust_attn_implementation")
        if original is None:
            return

        @functools.wraps(original)
        def adjusted(self, *args, **kwargs):
            try:
                return original(self, *args, **kwargs)
            except TypeError:
                kwargs.pop("allow_all_kernels", None)
                return original(self, *args, **kwargs)

        cls._check_and_adjust_attn_implementation = adjusted

    def hook(cls, **kwargs) -> None:
        super(PreTrainedModel, cls).__init_subclass__(**kwargs)
        wrap(cls)

    PreTrainedModel.__init_subclass__ = classmethod(hook)

    from transformers import DynamicCache

    if not hasattr(DynamicCache, "to_legacy_cache"):
        # The remote generation code still converts caches to and from the tuple format.
        def to_legacy_cache(self):
            return tuple((layer.keys, layer.values) for layer in self.layers)

        def from_legacy_cache(cls, legacy=None):
            cache = cls()
            for index, (key, value) in enumerate(legacy or ()):
                cache.update(key, value, index)
            return cache

        DynamicCache.to_legacy_cache = to_legacy_cache
        DynamicCache.from_legacy_cache = classmethod(from_legacy_cache)


def calibrate(
    model_dir: Path, images: Path, names: list[str], count: int, max_new_tokens: int
) -> None:
    """Run ModelOpt FP8 calibration over the images and store the activation scales under the shape key."""
    import modelopt.torch.quantization as mtq
    import torch
    from transformers import AutoModel

    from .config import ModelConfig
    from .tests.hf_reference import (
        _LocalTokenizer,
        _load_config,
        _repair_rotary_buffers,
        manual_chat_prompt,
    )
    from .tests.vision_oracle import preprocess_image_inputs_for_trt

    config = ModelConfig.from_dir(model_dir)
    key = scale_key(config)
    torch.backends.cudnn.enabled = False
    _accept_new_loader_arguments()
    tokenizer = _LocalTokenizer(model_dir)
    model = AutoModel.from_pretrained(
        model_dir,
        config=_load_config(model_dir),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16,
    ).to("cuda")
    model.eval()
    _repair_rotary_buffers(model)
    files = sorted(images.glob("*.jpg"))[:count]
    if not files:
        raise RuntimeError(f"no jpg images in {images}")

    def forward_loop(calibration_model) -> None:
        """Answer one grounding or pointing prompt per image."""
        with torch.no_grad():
            for index, path in enumerate(files):
                inputs = preprocess_image_inputs_for_trt(
                    path,
                    fixed_image_size=448,
                    patch_size=14,
                    image_mean=(0.5, 0.5, 0.5),
                    image_std=(0.5, 0.5, 0.5),
                    interpolation="bicubic",
                )
                prompt = _PROMPTS[index % len(_PROMPTS)].format(name=description(path, names))
                encoded = tokenizer(manual_chat_prompt(prompt), return_tensors="pt")
                calibration_model.generate(
                    pixel_values=torch.from_numpy(inputs["pixel_values"]).to("cuda", torch.float16),
                    image_grid_hws=torch.from_numpy(inputs["image_grid_hws"]).to(
                        device="cuda", dtype=torch.int32
                    ),
                    input_ids=encoded["input_ids"].to("cuda"),
                    attention_mask=encoded["attention_mask"].to("cuda"),
                    tokenizer=tokenizer,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    generation_mode="slow",
                    do_sample=False,
                )

    model = mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop)

    scales: dict[str, float] = {}
    skipped: list[str] = []
    for name, module in model.named_modules():
        internal = weight_name(name)
        if (
            internal is None
            or UNQUANTIZED.match(internal)
            or not hasattr(module, "input_quantizer")
        ):
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
    # Every layer must keep its MLP and attention input projections.
    for layer in range(int(config.num_hidden_layers)):
        for stem in ("w_q", "w_k", "w_v", "w_gate", "w_up", "w_down"):
            if f"layer.{layer}.{stem}" not in scales:
                raise RuntimeError(f"layer {layer} has no calibrated {stem}")

    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    table[key] = dict(sorted(scales.items()))
    path.write_text(json.dumps(dict(sorted(table.items())), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(scales)} scales for {key!r} to {path}")


def main() -> None:
    """Parse the command line and run the calibrate command."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("calibrate")
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--images", type=Path, required=True)
    run.add_argument("--classnames", type=Path)
    run.add_argument("--count", type=int, default=128)
    run.add_argument("--max-new-tokens", type=int, default=48)
    args = parser.parse_args()
    names = args.classnames.read_text().splitlines() if args.classnames else []
    calibrate(args.model_dir, args.images, names, args.count, args.max_new_tokens)


if __name__ == "__main__":
    main()
