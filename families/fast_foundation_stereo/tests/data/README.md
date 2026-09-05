<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Fast Foundation Stereo visual fixture

`office_left.png` and `office_right.png` are a repo-owned, AI-generated
rectified stereo pair created for human-readable CI review. The scene contains
recognizable foreground, midground, and background objects; it is not derived
from external photography or from the upstream project's restricted demo data.

The right view was generated from the left view as a horizontal camera
translation with depth-dependent parallax. Both images were resized identically
to the runtime's fixed 700x700 RGB input before being committed.

- `office_left.png`
- `office_right.png`

## Middlebury-Q task-accuracy data

The task-accuracy workload uses all 15 official Middlebury-v3 `trainingQ`
scenes. It is prepared outside the repository because the dataset has no SPDX
license. Middlebury permits use and publication of its images and numerical
results with citation.

```bash
python3 families/fast_foundation_stereo/tests/prepare_middlebury_q.py \
  --output /mnt/data/Middlebury-v3-trainingQ-profile-700x700
```

The preparer preserves the Q pixel scale, center-crops only widths above 700,
symmetrically edge-pads smaller images, and marks every padded pixel invalid.
In accordance with the official
MiddEval3 SDK's `evaldisp.cpp`, only `mask0nocc.png` value 255 is scored; values
0 and 128 are excluded. The resulting metrics are profile-specific
Middlebury-Q results, not official full-resolution leaderboard results.
