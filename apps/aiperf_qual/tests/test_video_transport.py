# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio

import pytest

pytest.importorskip("aiperf")
from aiperf.plugin import plugins
from aiperf.transports.aiohttp_transport import AioHttpTransport
from trtmc_aiperf_plugins.video_transport import VideoCaptureTransport


def test_opt_in_transport_preserves_actual_download_and_rejects_unsafe_id(monkeypatch, tmp_path):
    original = b"complete actual download bytes\0\xff"

    async def download(self, job_id, content_url, headers):
        return original

    monkeypatch.setattr(AioHttpTransport, "_download_video_content", download)
    monkeypatch.setenv("TRTMC_VIDEO_CAPTURE_DIR", str(tmp_path))
    assert plugins.get_class("transport", "trtmc_video_capture") is VideoCaptureTransport

    async def scenario():
        transport = object.__new__(VideoCaptureTransport)
        identifier = "video-" + "a" * 32
        actual = await transport._download_video_content(identifier, "http://test/content", {})
        assert actual is original and (tmp_path / f"{identifier}.mp4").read_bytes() == original
        with pytest.raises(ValueError, match="unsafe"):
            await transport._download_video_content("../escape", "http://test/content", {})

    asyncio.run(scenario())
