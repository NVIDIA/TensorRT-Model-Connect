# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic media for the model-card receipt and frame-array examples.

The model card names receipt.jpg but does not distribute that image. Both
implementations consume the same generated receipt here; it is not represented
as an upstream asset.
"""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def create_media_fixtures(root: Path) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=17)
    for y, line in (
        (18, "ACME STORE"),
        (52, "RECEIPT #1042"),
        (92, "Items      $1250.00"),
        (126, "TOTAL: 1250.00 USD"),
        (174, "PAID - THANK YOU"),
    ):
        draw.text((12, y), line, fill="black", font=font)
    draw.line((12, 115, 244, 115), fill="black", width=2)
    image.save(root / "receipt.png")
    request = {
        "model": "clef",
        "state": {"task": "Review the attached receipt."},
        "images": ["receipt.png"],
        "questions": {"legible": {"type": "noul", "instructions": "Is the receipt total legible?"}},
    }
    receipt = root / "receipt.json"
    receipt.write_text(json.dumps(request, indent=2))
    frames = []
    for index in range(5):
        frame = Image.new("RGB", (128, 96), "white")
        canvas = ImageDraw.Draw(frame)
        left = 10 + 15 * index
        canvas.rectangle((left, 26, left + 25, 65), fill="red")
        name = f"frame-{index}.png"
        frame.save(root / name)
        frames.append(name)
    video_request = {
        "model": "clef",
        "state": "Review the video frames.",
        "videos": [frames],
        "questions": {
            "color": {
                "type": "choice",
                "instructions": "What color is the moving square?",
                "criteria": {"red": "Red", "blue": "Blue", "green": "Green"},
            },
            "moving": {"type": "noul", "instructions": "Does the square move to the right?"},
        },
    }
    video = root / "video.json"
    video.write_text(json.dumps(video_request, indent=2))
    return [receipt, video]


def reference_record(path: Path, *, image_paths=(), video_frame_paths=()) -> dict:
    import numpy as np

    value = json.loads(path.read_text())
    if image_paths:
        value["images"] = image_paths
    if video_frame_paths:
        value["videos"] = video_frame_paths
    if "images" in value:
        value["images"] = [
            Image.open(path.parent / name).convert("RGB") for name in value["images"]
        ]
    if "videos" in value:
        value["videos"] = [
            np.stack([np.asarray(Image.open(path.parent / name).convert("RGB")) for name in frames])
            for frames in value["videos"]
        ]
    return value


def fixture_record(path: Path) -> dict:
    """Load repository media from the same manifest used by the native benchmark."""
    root = Path(__file__).resolve().parent
    inputs = {}
    if path.resolve().parent == root / "fixtures":
        cases = json.loads((root / "manifests/clef.json").read_text())["testcases"]
        inputs = next(
            (
                case["inputs"]
                for case in cases
                if Path(case["inputs"]["document_path"]).name == path.name
            ),
            {},
        )
    return reference_record(
        path,
        image_paths=[root / name for name in inputs.get("image_paths", [])],
        video_frame_paths=[
            [root / name for name in frames] for frames in inputs.get("video_frame_paths", [])
        ],
    )
