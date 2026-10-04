# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The formal run's ledger from calibration.json (DESIGN.md Section 9): a piloted profile takes its measured wall
seconds; any other its smoke-based prediction times its Task's measured ratio (still images and video by their own);
the blocked profiles a fixed allowance; every profile of a settling class adds its settle time (DESIGN.md 4.6: 10 s a
side and timed request). Usage: python ledger.py calibration.json > ledger.json"""

import json
import sys


def settle_seconds(settle: dict, task: str) -> float:
    """Both sides' settle before their timed requests (none in the slow measurement classes)."""
    if task in settle["unsettled_tasks"]:
        return 0.0
    return settle["seconds"] * 2 * settle["timed_requests"].get(task, 1)


def ledger(calibration: dict) -> dict[str, int]:
    seconds = {}
    for profile, row in calibration["smoke_predictions"].items():
        task, predicted = row["task"], float(row["predicted_s"])
        if profile in calibration["pilot_seconds"]:
            value = float(calibration["pilot_seconds"][profile])
        elif task == "image_generation":
            video = any(marker in profile for marker in calibration["video_markers"])
            value = predicted * (1.0 if video else calibration["still_image_ratio"])
        else:
            value = predicted * calibration["task_ratios"].get(task, calibration["small_task_ratio"])
        seconds[profile] = round(value + settle_seconds(calibration["settle"], task))
    for profile in calibration["blocked"]:
        seconds.setdefault(profile, calibration["blocked_seconds"])
    return dict(sorted(seconds.items()))


if __name__ == "__main__":
    json.dump(ledger(json.load(open(sys.argv[1]))), sys.stdout, indent=1)
    print()
