# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SAM2 HOI model-owned E2E runner entrypoint."""

from .runners.video_tracking import HoiVideoTrackingRunner

runner = HoiVideoTrackingRunner()
