# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The upstream PointNet S3DIS checkpoint for ``segment_points``, as tests/official_reference.py runs it: one
point cloud (row-major float32 [num_points, 9]) to the arg-max class of each point."""

from pathlib import Path
import sys
from typing import Any, Mapping

MODEL_DIRECTORY = Path(sys.prefix) / "trtmc-reference" / "pointnet-s3dis"  # reference/prepare.py


class Adapter:
    def __init__(self, spec: Any, host: Any) -> None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from families.pointnet.tests.official_reference import _load

        self.spec, self.host = spec, host
        self.model = _load(MODEL_DIRECTORY, spec.device).to(spec.dtype)

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        import numpy as np
        import torch

        count = int(self.host.required(request, "num_points"))
        dim = int(request.get("input_dim", 9))
        points = np.fromfile(str(self.host.required(request, "points_path")), dtype="<f4").reshape(count, dim)
        tensor = torch.from_numpy(points.T[None].copy()).to(self.spec.device, self.spec.dtype)  # [1, dim, N]
        logits, model_ms = self.host.timed(lambda: self.model(tensor)[0])
        labels = logits.float().argmax(dim=1).cpu().tolist()
        return self.host.invocation({"segmented_points": count, "num_classes": int(logits.shape[1]),
                                     "labels": labels}, model_ms)
