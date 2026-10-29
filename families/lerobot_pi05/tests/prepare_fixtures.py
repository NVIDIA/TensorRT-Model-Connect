# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regenerate the recorded observations and LeRobot reference actions under ``data/``.

Frames come from the pinned public ``physical-intelligence/libero`` dataset
(CC-BY-4.0). The reference actions come from ``official_reference.py`` with the
fixed noise, in float32 and bfloat16. Run it only to refresh the fixtures:

    python prepare_fixtures.py --model-dir <pi05_libero_base> --tokenizer-dir <dir with tokenizer.model>

Besides ``lerobot[pi]`` it needs ``pyarrow`` and ``pillow``.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np

from official_reference import fixed_noise, run_reference

HERE = Path(__file__).resolve().parent
SOURCE = json.loads((HERE / "reference-source.json").read_text(encoding="utf-8"))


def _write_frames(out: Path) -> list[Path]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image

    spec = SOURCE["frames"]
    kwargs = {"repo_id": spec["dataset"], "repo_type": "dataset", "revision": spec["revision"]}
    episode = hf_hub_download(filename=f"data/chunk-000/episode_{spec['episode']:06d}.parquet", **kwargs)
    tasks = hf_hub_download(filename="meta/tasks.jsonl", **kwargs)
    task_text = {
        row["task_index"]: row["task"]
        for row in map(json.loads, Path(tasks).read_text(encoding="utf-8").splitlines())
    }
    table = pq.read_table(episode)
    frames = []
    for index in spec["frame_indices"]:
        row = table.slice(index, 1).to_pylist()[0]
        directory = out / "recorded_observation" / f"frame_{index:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        for column, name in (("image", "image.png"), ("wrist_image", "wrist_image.png")):
            # Re-encode so the committed PNG is exactly what the model reads, with no metadata.
            Image.open(io.BytesIO(row[column]["bytes"])).convert("RGB").save(directory / name, optimize=True)
        (directory / "observation.json").write_text(
            json.dumps(
                {
                    "source": f"{spec['dataset']}@{spec['revision']} episode {spec['episode']} frame {index}",
                    "state": [float(v) for v in row["state"]],
                    "task": task_text[row["task_index"]],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        frames.append(directory)
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE / "data")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    frames = _write_frames(args.out)
    fixed_noise().astype("<f4").tofile(args.out / "noise_seed0.f32")
    reference = args.out / "reference"
    reference.mkdir(exist_ok=True)
    for frame in frames:
        for dtype, label in (("float32", "fp32"), ("bfloat16", "bf16")):
            actions = run_reference(
                args.model_dir, args.tokenizer_dir, frame, dtype=dtype, device=args.device
            )
            np.ascontiguousarray(actions, dtype="<f4").tofile(reference / f"{frame.name}.{label}.f32")
            print(frame.name, label, actions.shape, float(np.abs(actions).max()))


if __name__ == "__main__":
    main()
