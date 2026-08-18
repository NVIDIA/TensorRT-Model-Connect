# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SAM2 HOI model-owned E2E comparator entrypoint."""

from .comparators.video_tracking import HoiVideoTrackingComparator

comparator = HoiVideoTrackingComparator()
