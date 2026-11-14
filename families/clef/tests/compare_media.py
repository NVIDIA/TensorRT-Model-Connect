# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check native media patches and record tokens against the original processor."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace, MethodType

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.checkpoint))
    from joint_schema_model import encode_record, collate_records
    from transformers import AutoProcessor
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
    import torch

    processor = AutoProcessor.from_pretrained(args.checkpoint)
    position_model = SimpleNamespace(
        config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))
    )
    position_model.get_vision_position_ids = MethodType(
        Qwen3_5Model.get_vision_position_ids, position_model
    )
    receipts = []
    for name, height, width, frames in (
        ("square", 256, 256, 1),
        ("resize", 181, 347, 1),
        ("video", 96, 128, 5),
    ):
        root = args.output / name
        root.mkdir(parents=True, exist_ok=True)
        arrays, paths = [], []
        for i in range(frames):
            values = np.random.default_rng(110 + i).integers(
                0, 256, (height, width, 3), dtype=np.uint8
            )
            path = root / f"frame-{i}.png"
            Image.fromarray(values).save(path)
            arrays.append(values)
            paths.append(path.name)
        base = {
            "model": "clef",
            "state": {"task": "Review the attached receipt."},
            "questions": {
                "legible": {"type": "noul", "instructions": "Is the receipt total legible?"}
            },
        }
        native = dict(base)
        record = dict(base)
        if frames == 1:
            native["images"] = paths
            record["images"] = [Image.fromarray(arrays[0])]
        else:
            native["videos"] = [paths]
            record["videos"] = [np.stack(arrays)]
        input_path = root / "record.json"
        input_path.write_text(json.dumps(native))
        encoded = encode_record(processor.tokenizer, record, processor=processor)
        actual = json.loads(
            subprocess.check_output(
                [
                    str(args.probe),
                    str(args.checkpoint / "tokenizer.json"),
                    str(args.checkpoint / "processor_config.json"),
                    str(input_path),
                    str(root),
                ],
                text=True,
            )
        )
        if actual["input_ids"] != list(encoded.input_ids):
            print(
                name, "actual", actual["input_ids"], "expected", list(encoded.input_ids), flush=True
            )
            raise AssertionError("native media input IDs differ")
        key = "pixel_values" if frames == 1 else "pixel_values_videos"
        expected = encoded.media[key].numpy()
        patches = np.concatenate(
            [
                np.fromfile(root / f"{i}.bin", np.float32).reshape(-1, 1536)
                for i in range(len(actual["grids"]))
            ]
        )
        np.testing.assert_array_equal(patches, expected)
        batch = collate_records([encoded], processor.tokenizer.pad_token_id, torch.device("cpu"))
        expected_positions, _ = Qwen3_5Model.get_rope_index(
            position_model,
            input_ids=batch["input_ids"],
            mm_token_type_ids=batch["media"]["mm_token_type_ids"],
            image_grid_thw=batch["media"].get("image_grid_thw"),
            video_grid_thw=batch["media"].get("video_grid_thw"),
        )
        np.testing.assert_array_equal(actual["positions"], expected_positions[:, 0].T.numpy())
        receipts.append(
            {
                "case": name,
                "exact_input_ids": len(encoded.input_ids),
                "exact_patch_values": patches.size,
            }
        )
        print(receipts[-1], flush=True)
    (args.output / "comparison.json").write_text(json.dumps(receipts, indent=2))


if __name__ == "__main__":
    main()
