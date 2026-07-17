# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native task adapters migrated from the legacy Task Eval implementation."""

from .time_series import TimeSeriesTaskAdapter

__all__ = ["TimeSeriesTaskAdapter"]
