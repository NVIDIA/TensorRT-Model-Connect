# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for chronos_bolt."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("chronos_bolt", "chronos-bolt", "chronosbolt"),
    architectures=("ChronosBoltModelForForecasting",),
    tasks=("time_series_forecast",),
    default_task="time_series_forecast",
)
