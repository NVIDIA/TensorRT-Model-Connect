# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for hstu."""

from tensorrt_model_connect.model_support import family_support

#: The checkpoint identity ``config.json`` states (``families.hstu.weights``
#: writes it; ``families.hstu.config`` reads it).
MODEL_TYPE = "hstu_ranking"
ARCHITECTURE = "HSTURankingGR"

describe = family_support(
    model_types=(MODEL_TYPE,),
    architectures=(ARCHITECTURE,),
    tasks=("structured_decision",),
    default_task="structured_decision",
    default_precision="bf16",
)
