# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ACT conversion-parity inputs: recorded observations of the pinned LeRobot dataset
(lerobot/aloha_sim_transfer_cube_human), episode 0, every eighth frame, as the camera PNG and the 14-value
state file the control request takes. Run in the family's environment (it decodes the dataset video):
``python native_inputs.py --count N --output DIR`` prints the records as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from families.lerobot_act.tests.prepare_recorded_observation import (_DATA_FILE, _VIDEO_FILE, _decode_frame,
                                                                      _download, _recorded_row)

EPISODE = 0
STRIDE = 8


def main() -> int:
    from PIL import Image

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    data, video = _download(_DATA_FILE, local_files_only=False), _download(_VIDEO_FILE, local_files_only=False)
    arguments.output.mkdir(parents=True, exist_ok=True)
    records = []
    for position in range(arguments.count):
        frame = position * STRIDE
        global_index, state = _recorded_row(data, EPISODE, frame)
        image = arguments.output / f"episode{EPISODE}-frame{frame:04d}.png"
        state_path = arguments.output / f"episode{EPISODE}-frame{frame:04d}.state.f32"
        if not image.is_file():
            Image.fromarray(_decode_frame(video, global_index)).save(image)
        np.asarray(state, dtype="<f4").tofile(state_path)
        records.append({"id": f"act/episode{EPISODE}/frame{frame}",
                        "request": {"image_path": str(image), "state_path": str(state_path)}})
    print(json.dumps(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
