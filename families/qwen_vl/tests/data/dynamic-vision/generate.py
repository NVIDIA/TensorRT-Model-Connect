# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record the independent HF 5.2 learned-position interpolation contract."""

import json
from pathlib import Path

import torch

cases = []
for height, width in [(2, 2), (4, 6), (6, 4), (14, 28)]:
    side = 48
    row, col = torch.linspace(0, side - 1, height), torch.linspace(0, side - 1, width)
    r0, c0 = row.int(), col.int()
    r1, c1 = (r0 + 1).clip(max=side - 1), (c0 + 1).clip(max=side - 1)
    dr, dc = row - r0, col - c0
    indices = torch.stack([(r0[:, None] * side + c0).flatten(),
                           (r0[:, None] * side + c1).flatten(),
                           (r1[:, None] * side + c0).flatten(),
                           (r1[:, None] * side + c1).flatten()])
    weights = torch.stack([((1 - dr[:, None]) * (1 - dc)).flatten(),
                           ((1 - dr[:, None]) * dc).flatten(),
                           (dr[:, None] * (1 - dc)).flatten(),
                           (dr[:, None] * dc).flatten()])

    def grouped(tensor):
        return tensor.reshape(4, height // 2, 2, width // 2, 2).permute(0, 1, 3, 2, 4).flatten()

    cases.append({"height": height, "width": width, "side": side,
                  "indices": grouped(indices).tolist(), "weights": grouped(weights).tolist()})
Path(__file__).with_name("hf-position-metadata.json").write_text(json.dumps(cases) + "\n")
