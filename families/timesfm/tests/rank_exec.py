# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select one visible GPU per local MPI rank before starting a native consumer."""

import os
import sys

rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
visible = os.environ.get("CUDA_VISIBLE_DEVICES")
devices = visible.split(",") if visible else None
if devices is not None and not 0 <= rank < len(devices):
    raise SystemExit("MPI local rank is outside the available CUDA devices")
os.environ["CUDA_VISIBLE_DEVICES"] = devices[rank] if devices else str(rank)
os.execv(sys.argv[1], sys.argv[1:])
