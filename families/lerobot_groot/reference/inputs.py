# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GR00T conversion-parity inputs: recorded LIBERO-spatial observations of the pinned public dataset the
fixtures come from (tests/fixture-source.json), one seeded frame of each of ``--count`` seeded episodes of the
ten spatial tasks, written as the fixtures are (``image.png``, ``wrist_image.png``, observation.json). Each
record carries the family's fixed flow noise, so both sides draw the same chunk. Run in the reference
environment: ``python inputs.py --count N --output DIR`` prints the records as JSON."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import random

from families.lerobot_groot.tests.official_reference import CAMERAS, fixed_noise

SOURCE = json.loads((Path(__file__).resolve().parents[1] / "tests/fixture-source.json").read_text(encoding="utf-8"))
SEED = 20261113


def spatial(tasks: list[str]) -> bool:
    return bool(tasks) and all(task.startswith("pick up the black bowl") and task.endswith("place it on the plate")
                               for task in tasks)


def main() -> int:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    spec = SOURCE["frames"]
    kwargs = {"repo_id": spec["dataset"], "repo_type": "dataset", "revision": spec["revision"]}
    info = json.loads(Path(hf_hub_download(filename="meta/info.json", **kwargs)).read_text(encoding="utf-8"))
    episodes = [json.loads(line) for line in
                Path(hf_hub_download(filename="meta/episodes.jsonl", **kwargs)).read_text(encoding="utf-8").splitlines()
                if line.strip()]
    generator = random.Random(SEED)
    chosen = generator.sample([episode for episode in episodes if spatial(episode["tasks"])], arguments.count)
    noise = [float(value) for value in fixed_noise().reshape(-1)]
    records = []
    for episode in chosen:
        index, frame = int(episode["episode_index"]), generator.randrange(int(episode["length"]))
        directory = arguments.output / f"ep{index:04d}_f{frame:03d}"
        if not (directory / "observation.json").is_file():
            path = info["data_path"].format(episode_chunk=index // int(info["chunks_size"]), episode_index=index)
            row = pq.read_table(hf_hub_download(filename=path, **kwargs)).slice(frame, 1).to_pylist()[0]
            directory.mkdir(parents=True, exist_ok=True)
            for column in ("image", "wrist_image"):
                Image.open(io.BytesIO(row[column]["bytes"])).convert("RGB").save(directory / f"{column}.png")
            (directory / "observation.json").write_text(json.dumps(
                {"source": f"{spec['dataset']}@{spec['revision']} episode {index} frame {frame}",
                 "state": [float(value) for value in row["state"]], "task": episode["tasks"][0]}), encoding="utf-8")
        observation = json.loads((directory / "observation.json").read_text(encoding="utf-8"))
        records.append({"id": f"libero/episode{index}/frame{frame}", "cluster": f"episode{index}", "request": {
            "images": [{"name": CAMERAS[0], "image_path": str(directory / "image.png")},
                       {"name": CAMERAS[1], "image_path": str(directory / "wrist_image.png")}],
            "state": observation["state"], "prompt": observation["task"], "noise": noise}})
    print(json.dumps(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
