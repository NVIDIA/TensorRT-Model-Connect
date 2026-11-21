# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in capture of the actual bytes downloaded by published AIPerf's video path."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
import re

from aiperf.transports.aiohttp_transport import AioHttpTransport


class VideoCaptureTransport(AioHttpTransport):
    async def _download_video_content(self, job_id, content_url, headers):
        data = await super()._download_video_content(job_id, content_url, headers)
        if isinstance(data, bytes):
            if re.fullmatch(r"video-[0-9a-f]{32}", job_id) is None:
                raise ValueError("unsafe video job ID")
            root = Path(os.environ["TRTMC_VIDEO_CAPTURE_DIR"])

            def save():
                with (root / f"{job_id}.mp4").open("xb") as file:
                    file.write(data)

            await asyncio.to_thread(save)
        return data
