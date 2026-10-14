# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the unchanged upstream E2E oracle outside the MC build process."""

from pathlib import Path
import json
import sys

import numpy as np


def _reference_yaml(model_dir: Path) -> str:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    model = config.get("model") if isinstance(config, dict) else None
    if not isinstance(model, str) or model not in {"yolov10n.yaml", "yolov10s.yaml", "yolov10x.yaml"}:
        raise ValueError(f"unsupported YOLOv10 reference architecture: {model!r}")
    return model


def _reference_tensors(model_dir: Path) -> dict:
    from families.yolov10.checkpoint import Checkpoint

    checkpoint = Checkpoint.open(model_dir, framework="pt")
    return {name: reader.get_tensor(name) for name, reader in checkpoint.tensor_map.items()}


def _letterbox(image, size: int):
    """Ultralytics preprocessing: keep aspect, pad with 114, scale to [0, 1]."""
    from PIL import Image

    scale = min(size / image.height, size / image.width)
    resized = image.resize(
        (round(image.width * scale), round(image.height * scale)), Image.BILINEAR
    )
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    left = (size - resized.width) // 2
    top = (size - resized.height) // 2
    canvas.paste(resized, (left, top))
    array = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return array[None].copy(), scale, float(left), float(top)


def run(model_dir: Path, image: Path, output: Path) -> None:
    from ultralytics import YOLO

    # The reference must run the checkpoint under test. YOLO(<directory>)
    # quietly downloads its own weights instead, which would compare the engine
    # against a different model.
    reference = YOLO(_reference_yaml(model_dir), task="detect").model
    raw = _reference_tensors(model_dir)
    state = {key[len("model."):]: value for key, value in raw.items() if key.startswith("model.")}
    missing, unexpected = reference.load_state_dict(state, strict=False)
    assert not missing, f"reference is missing checkpoint tensors: {missing[:5]}"
    reference = reference.eval()

    import torch
    from PIL import Image

    source = Image.open(image).convert("RGB")
    letterboxed, scale, pad_x, pad_y = _letterbox(source, 640)
    with torch.no_grad():
        expected = reference(torch.from_numpy(letterboxed))
    expected = (expected[0] if isinstance(expected, (list, tuple)) else expected)[0].numpy()
    np.savez(output, expected=expected, scale=scale, pad_x=pad_x, pad_y=pad_y)


if __name__ == "__main__":
    run(*(Path(value) for value in sys.argv[1:]))
