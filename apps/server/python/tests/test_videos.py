# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import base64
from concurrent.futures import Future
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time

from fastapi.testclient import TestClient
import httpx
from PIL import Image
import pytest

from trtmc_server.app import ServerConfig, create_app
from trtmc_server.errors import WorkerSaturatedError
from trtmc_perf_serving.video_media import decoded_identity, encode, frame_identity
from trtmc_server.videos import VideoJobs


class VideoWorker:
    replica_name = "video-1"

    def __init__(self, *, pending=False, bad_geometry=False):
        self.pending = Future() if pending else None
        self.bad_geometry = bad_geometry
        self.busy = False
        self.calls = []
        self.retired = False

    def start(self):
        pass

    def close(self):
        self.busy = False

    def supports(self, model, capability):
        return model == "video" and capability == "video_generation"

    def video_geometry(self, model):
        return {"width": 8, "height": 4, "num_frames": 3}

    def acquire(self, model):
        if self.busy:
            raise WorkerSaturatedError("busy")
        self.busy = True
        return self

    def submit(self, operation, payload):
        assert operation == "generate_video"
        self.calls.append(payload)
        root = Path(payload["output_path"])
        root.mkdir()
        for index, color in enumerate(("red", "green", "blue")):
            Image.new("RGB", (8, 4), color).save(root / f"frame-{index:06d}.png")
        self.result = {"width": 16 if self.bad_geometry else 8, "height": 4, "num_frames": 3,
                       "channels": 3, "model_call_ms": 12.5, "timing_scope": "public_task_call_wall"}
        if self.pending is not None:
            return self.pending
        result = Future()
        result.set_result(self.result)
        return result

    def retire(self, error):
        self.retired = True
        result = Future()
        result.set_result(None)
        return result


def app_for(worker, records=None, **limits):
    return create_app(worker, ServerConfig(max_body_bytes=16000, max_prompt_bytes=128,
                      max_generation_tokens=128, records=records, **limits))


def submit(client, **changes):
    return client.post("/v1/videos", json={"model": "video", "prompt": "red green blue",
                       "seed": 42, "num_inference_steps": 30, "num_frames": 3, **changes},
                       headers={"X-Request-ID": "clip-42"})


def finished(client, identifier):
    for _ in range(500):
        value = client.get(f"/v1/videos/{identifier}").json()
        if value["status"] in {"completed", "failed"}:
            return value
        time.sleep(0.01)
    raise AssertionError("video job did not finish")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="external FFmpeg codec unavailable")
def test_real_clip_preserves_all_frames_and_completion_identity(tmp_path):
    worker = VideoWorker()
    records = tmp_path / "records.jsonl"
    with TestClient(app_for(worker, records)) as client:
        response = client.post("/v1/videos", files={"model": (None, "video"), "prompt": (None, "red green blue"),
            "seed": (None, "42"), "num_inference_steps": (None, "30"), "num_frames": (None, "3")},
            headers={"X-Request-ID": "clip-42"})
        assert response.status_code == 202
        identifier = response.json()["id"]
        assert finished(client, identifier)["status"] == "completed"
        download = client.get(f"/v1/videos/{identifier}/content")
        assert download.status_code == 200 and download.headers["content-type"] == "video/mp4"
        path = tmp_path / "actual.mp4"
        path.write_bytes(download.content)
        decoded_identity(path, 8, 4, 3, output=tmp_path / "frames")
        assert [Image.open(p).getpixel((0, 0)) for p in sorted((tmp_path / "frames").glob("*.png"))] == [
            (255, 0, 0), (0, 128, 0), (0, 0, 255)]
    entries = [json.loads(line) for line in records.read_text().splitlines()]
    complete, = [r for r in entries if r.get("record_kind") == "video_completion"]
    assert complete["request_id"] == "clip-42" and complete["job_id"] == identifier
    assert complete["model_call_ms"] == 12.5 and complete["output_frames"] == 3
    assert complete["output_sha256"] == hashlib.sha256(download.content).hexdigest()
    assert complete["generation_config"] == {"width": 8, "height": 4, "seed": 42, "num_steps": 30}
    assert not Path(worker.calls[0]["output_path"]).exists()


@pytest.mark.parametrize("changes", [{"num_frames": 4}, {"size": "16x4"}, {"seconds": 4},
                                     {"stream": True}, {"n": 2}, {"fps": 0}, {"mask": "x"}])
def test_rejects_unsupported_or_different_work_before_native_execution(changes):
    worker = VideoWorker()
    with TestClient(app_for(worker)) as client:
        assert submit(client, **changes).status_code == 400
    assert not worker.calls


def test_busy_and_not_ready_are_visible_and_shutdown_keeps_native_files():
    worker = VideoWorker(pending=True)
    with TestClient(app_for(worker)) as client:
        identifier = submit(client).json()["id"]
        while not worker.calls:
            time.sleep(0.01)
        assert submit(client).status_code == 429
        assert client.get(f"/v1/videos/{identifier}/content").status_code == 409
        worker.pending.set_result({**worker.result, "width": 9})
        assert finished(client, identifier)["status"] == "failed"
        assert worker.retired
        assert client.get(f"/v1/videos/{identifier}/content").status_code == 409
    assert not Path(worker.calls[0]["output_path"]).exists()


def test_storage_limit_rejects_before_native_execution():
    worker = VideoWorker()
    with TestClient(app_for(worker, max_video_bytes=100)) as client:
        assert submit(client).status_code == 429
    assert not worker.calls


def test_shutdown_waits_for_encoding_and_records_before_removing_files(monkeypatch, tmp_path):
    async def scenario():
        worker = VideoWorker()
        entered, release = threading.Event(), threading.Event()
        roots = []

        def blocked(directory, *args):
            roots.append(directory)
            entered.set()
            release.wait(10)
            assert directory.exists()
            raise RuntimeError("injected codec failure")

        monkeypatch.setattr("trtmc_server.videos.encode", blocked)
        app = app_for(worker, tmp_path / "records.jsonl")
        manager = app.router.lifespan_context(app)
        await manager.__aenter__()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            response = await client.post("/v1/videos", json={"model": "video", "prompt": "rgb"})
            assert response.status_code == 202
            assert await asyncio.to_thread(entered.wait, 5)
            close = asyncio.create_task(manager.__aexit__(None, None, None))
            await asyncio.sleep(0.03)
            assert not close.done() and roots[0].exists()
            release.set()
            await asyncio.wait_for(close, 5)
        assert not roots[0].parent.exists()
        records = [json.loads(line) for line in (tmp_path / "records.jsonl").read_text().splitlines()]
        completion, = [r for r in records if r.get("record_kind") == "video_completion"]
        assert completion["status"] == 502
    asyncio.run(scenario())


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="external FFmpeg codec unavailable")
def test_codec_rejects_missing_frames_and_incorrect_geometry(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    Image.new("RGB", (8, 4), "red").save(frames / "frame-000000.png")
    with pytest.raises(ValueError, match="incomplete"):
        frame_identity(frames, 8, 4, 2)
    encode(frames, tmp_path / "clip.mp4", 8, 4, 1, 24, 100000)
    with pytest.raises(ValueError, match="dimensions"):
        decoded_identity(tmp_path / "clip.mp4", 4, 8, 1)
    with pytest.raises(ValueError, match="FPS"):
        decoded_identity(tmp_path / "clip.mp4", 8, 4, 1, fps=12)


def test_cancelled_download_releases_expiry_lease_even_before_first_chunk(tmp_path):
    async def scenario():
        root = tmp_path / "job"
        root.mkdir()
        (root / "video.mp4").write_bytes(b"clip")
        config = ServerConfig(max_body_bytes=1000, max_prompt_bytes=100,
                              max_generation_tokens=1, video_ttl_seconds=1)
        jobs = VideoJobs(None, config, None, None, None)
        job = {"root": root, "status": "completed", "downloads": 0,
               "active": False, "finished": time.monotonic()}
        jobs.jobs["video-test"] = job
        response = await jobs.content("video-test")
        assert job["downloads"] == 1
        job["finished"] -= 5
        await jobs.expire()
        assert root.exists()

        async def disconnect(message):
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await response({"type": "http", "asgi": {"spec_version": "2.4"}}, None, disconnect)
        assert job["downloads"] == 0
        await jobs.expire()
        assert not root.exists() and not jobs.jobs

    asyncio.run(scenario())


def test_reserved_worker_and_replay_release_if_capability_disappears(monkeypatch):
    import trtmc_server.videos as videos

    worker = VideoWorker()
    media_paths = []
    original = videos.create_uploaded_input
    checks = 0

    async def capture(*args):
        media = await original(*args)
        media_paths.append(media.path)
        return media

    def draining(model, capability):
        nonlocal checks
        checks += 1
        return checks == 1

    monkeypatch.setattr(videos, "create_uploaded_input", capture)
    worker.supports = draining
    with TestClient(app_for(worker)) as client:
        identifier = submit(client, initial_latents_b64=base64.b64encode(b"\0" * 4).decode()).json()["id"]
        assert finished(client, identifier)["status"] == "failed"
        assert not worker.busy and not worker.calls
        assert media_paths and all(not path.exists() for path in media_paths)


def test_shutdown_waits_for_in_progress_submission_setup(monkeypatch, tmp_path):
    import trtmc_server.videos as videos

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        original = videos.create_uploaded_input
        media_paths = []

        async def blocked(*args):
            entered.set()
            await release.wait()
            media = await original(*args)
            media_paths.append(media.path)
            return media

        def unavailable_codec(*args):
            raise RuntimeError("injected codec failure")

        monkeypatch.setattr(videos, "create_uploaded_input", blocked)
        monkeypatch.setattr(videos, "encode", unavailable_codec)
        app = app_for(VideoWorker(), tmp_path / "records.jsonl")
        manager = app.router.lifespan_context(app)
        await manager.__aenter__()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            posting = asyncio.create_task(client.post("/v1/videos", json={"model": "video", "prompt": "rgb"}))
            await asyncio.wait_for(entered.wait(), 5)
            closing = asyncio.create_task(manager.__aexit__(None, None, None))
            await asyncio.sleep(0.02)
            assert not closing.done()
            release.set()
            assert (await posting).status_code == 202
            await asyncio.wait_for(closing, 5)
        assert media_paths and all(not path.exists() for path in media_paths)

    asyncio.run(scenario())
