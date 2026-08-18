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

- `office_left.png`: SHA256 `73cc585a0e38493a5588137fea302b8472f63e76443759bd8ba0a19ce8be76a6`
- `office_right.png`: SHA256 `6c56733d64567e198fa75375ab7042bd26a8aa1fdd8f8fb4908186ca7f2f51c5`
