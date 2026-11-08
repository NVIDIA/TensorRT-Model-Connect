# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate FP8 activation scales for an OWLv2 model config.

Not used when building an engine. Calibrate each supported model config and commit the
updated ``fp8_activation_scales.json``::

    python -m families.owlv2.calibrate_fp8_scales fetch --out images
    python -m families.owlv2.calibrate_fp8_scales calibrate \
        --model-dir <original OWLv2 checkpoint> --images images

``fetch`` downloads COCO validation images and the 80 COCO class names (ungated dataset
``detection-datasets/coco``) and writes them as ``img_NNNN.jpg`` plus ``classnames.txt``.
``calibrate`` loads the model in PyTorch and runs ModelOpt ``FP8_DEFAULT_CFG`` (max
calibration) while the image tower and both detection heads see the first
``--calib-images`` images with the 80 class queries, and the text tower sees the class
names under two prompt templates. It stores ``input_quantizer.amax / 448`` for every
projection under the model config key (see ``quantization.scale_key``). A projection whose
scale is above 20 times the median of its kind is stored as ``null`` and stays in fp16.
The images after the first ``--calib-images`` are left for evaluation. ``fetch`` needs
network access; ``calibrate`` needs a GPU.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import re
from pathlib import Path

from .quantization import (
    ACTIVATION_SCALES_FILENAME,
    FP8_E4M3_MAX,
    HEAD_PROJECTIONS,
    LAYER_STEMS,
    scale_key,
    split_outlier_scales,
)

_DATASET = "detection-datasets/coco"
_PROJECTION_NAME = re.compile(
    r"^owlv2\.(?:vision_model|text_model)\.encoder\.layers\.\d+\.(?:"
    + "|".join(re.escape(stem) for stem in LAYER_STEMS)
    + r")$"
)
_HEAD_NAMES = frozenset(HEAD_PROJECTIONS)


def projection_name(module_name: str) -> str | None:
    """Checkpoint name of an HF module path, or None when it is not a quantizable projection."""
    if _PROJECTION_NAME.match(module_name) or module_name in _HEAD_NAMES:
        return module_name
    return None


def fetch_images(out: Path, count: int) -> None:
    """Write the first ``count`` COCO validation images and the class names."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    from PIL import Image

    out.mkdir(parents=True, exist_ok=True)
    files = sorted(
        name
        for name in HfApi().list_repo_files(_DATASET, repo_type="dataset")
        if name.startswith("data/val") and name.endswith(".parquet")
    )
    if not files:
        raise RuntimeError(f"{_DATASET} lists no validation parquet file")
    written = 0
    for name in files:
        path = hf_hub_download(_DATASET, name, repo_type="dataset")
        table = pq.read_table(path)
        for row in table.to_pylist():
            if written < count:
                picture = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
                picture.save(out / f"img_{written:04d}.jpg", quality=95)
                written += 1
        if written >= count:
            break
    (out / "classnames.txt").write_text("\n".join(_COCO_CLASSES) + "\n", encoding="utf-8")
    print(f"wrote {written} images to {out}")


_COCO_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


def _fp8_config(mtq) -> dict:
    """``FP8_DEFAULT_CFG`` with the projections of both towers and the heads switched on.

    ModelOpt's default config leaves modules named ``*vision_model*`` unquantized, which
    would drop the whole image tower. Later entries override earlier ones, so the enable
    entries go last. Newer ModelOpt versions keep the entries in a list, older ones in a
    dict keyed by quantizer pattern.
    """
    config = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    patterns = [
        f"*{tower}.encoder.layers.*.{kind}"
        for tower in ("vision_model", "text_model")
        for kind in ("input_quantizer", "weight_quantizer")
    ] + [
        f"{head}.{kind}"
        for head in ("box_head.dense*", "class_head.dense0")
        for kind in ("input_quantizer", "weight_quantizer")
    ]
    for pattern in patterns:
        if isinstance(config["quant_cfg"], list):
            config["quant_cfg"].append({"quantizer_name": pattern, "enable": True})
        else:
            config["quant_cfg"][pattern] = {"enable": True}
    return config


def load_images(images: Path, start: int, stop: int):
    """RGB PIL images ``start..stop`` of the fetched set."""
    from PIL import Image

    pictures = []
    for index in range(start, stop):
        path = images / f"img_{index:04d}.jpg"
        if not path.is_file():
            break
        pictures.append(Image.open(path).convert("RGB"))
    return pictures


def calibrate(model_dir: Path, images: Path, calib_images: int) -> None:
    """Run ModelOpt FP8 calibration and store the activation scales under the config key."""
    import modelopt.torch.quantization as mtq
    import torch
    from safetensors.numpy import load_file
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    from . import config as config_module

    weights = load_file(str(Path(model_dir) / "model.safetensors"))
    cfg = config_module.resolve(model_dir, {key: value.shape for key, value in weights.items()})
    key = scale_key(cfg)
    window = cfg["text"]["max_position_embeddings"]

    processor = Owlv2Processor.from_pretrained(str(model_dir))
    model = Owlv2ForObjectDetection.from_pretrained(str(model_dir), dtype=torch.float32)
    model = model.cuda().eval()
    pictures = load_images(images, 0, calib_images)
    if not pictures:
        raise RuntimeError(f"no images in {images}; run the fetch command first")
    names = (images / "classnames.txt").read_text().splitlines()
    classes = [f"a photo of a {name}" for name in names]
    templates = classes + list(names)

    def tokens(prompts):
        return processor.tokenizer(
            prompts, padding="max_length", max_length=window, truncation=True, return_tensors="pt"
        )

    class_tokens = tokens(classes)

    def forward_loop(calibration_model) -> None:
        """Detect with every class query on every image, and embed all prompt templates."""
        with torch.no_grad():
            batch = 4
            for start in range(0, len(pictures), batch):
                pixels = processor.image_processor(
                    images=pictures[start : start + batch], return_tensors="pt"
                )["pixel_values"].cuda()
                count = pixels.shape[0]
                calibration_model(
                    input_ids=class_tokens["input_ids"].repeat(count, 1).cuda(),
                    attention_mask=class_tokens["attention_mask"].repeat(count, 1).cuda(),
                    pixel_values=pixels,
                )
            for start in range(0, len(templates), 80):
                encoded = tokens(templates[start : start + 80])
                calibration_model.owlv2.get_text_features(
                    input_ids=encoded["input_ids"].cuda(),
                    attention_mask=encoded["attention_mask"].cuda(),
                )

    model = mtq.quantize(model, _fp8_config(mtq), forward_loop)

    scales: dict[str, float] = {}
    skipped: list[str] = []
    for name, module in model.named_modules():
        internal = projection_name(name)
        if internal is None or not hasattr(module, "input_quantizer"):
            continue
        amax = module.input_quantizer.amax
        if amax is None or not float(amax) > 0:
            skipped.append(name)
            continue
        scales[internal] = float(amax) / FP8_E4M3_MAX
    # Every projection must be calibrated: a partial table would build a half-quantized
    # engine without any error.
    from .quantization import projections

    expected = set(projections(cfg))
    if set(scales) != expected:
        raise RuntimeError(
            f"calibration missed {len(expected - set(scales))} projections "
            f"(no amax: {skipped[:3]}, missing: {sorted(expected - set(scales))[:3]})"
        )
    table_entry = split_outlier_scales(scales)
    outliers = sorted(name for name, value in table_entry.items() if value is None)
    print(f"outlier projections left in fp16: {len(outliers)} {outliers}")

    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    table[key] = dict(sorted(table_entry.items()))
    path.write_text(json.dumps(dict(sorted(table.items())), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(scales)} scales for {key!r} to {path}")


def main() -> None:
    """Parse the command line and run the fetch or calibrate command."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--out", type=Path, required=True)
    fetch.add_argument("--count", type=int, default=400)
    run = commands.add_parser("calibrate")
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--images", type=Path, required=True)
    run.add_argument("--calib-images", type=int, default=200)
    args = parser.parse_args()
    if args.command == "fetch":
        fetch_images(args.out, args.count)
    else:
        calibrate(args.model_dir, args.images, args.calib_images)


if __name__ == "__main__":
    main()
