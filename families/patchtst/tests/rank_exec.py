# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assign one visible device to each local MPI rank before native startup."""

import os
import sys

rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
visible = os.environ.get("CUDA_VISIBLE_DEVICES")
devices = visible.split(",") if visible else None
if devices is not None and not 0 <= rank < len(devices):
    raise SystemExit("MPI local rank exceeds the visible GPU count")
os.environ["CUDA_VISIBLE_DEVICES"] = devices[rank] if devices else str(rank)
os.execv(sys.argv[1], sys.argv[1:])
