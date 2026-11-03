# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from concurrent.futures import Future
import io
from pathlib import Path
import shutil
import struct
import subprocess
import time
import wave

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from trtmc_server.app import ServerConfig, create_app
from trtmc_server.errors import WorkerSaturatedError


def png(color: str) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (6, 4), color).save(buffer, format="PNG")
    return buffer.getvalue()


class ReferenceWorker:
    replica_name = "reference-1"

    def __init__(self):
        self.busy = False
        self.calls = []

    def start(self):
        pass

    def close(self):
        self.busy = False

    def supports(self, model, capability):
        return model == "h3" and capability == "reference_video_generation"

    def video_geometry(self, model):
        raise AssertionError("reference video has no prebuilt geometry")

    def acquire(self, model):
        if self.busy:
            raise WorkerSaturatedError("busy")
        self.busy = True
        return self

    def submit(self, operation, payload):
        assert operation == "generate_reference_video"
        self.calls.append(payload)
        self.references = [Path(path).read_bytes() for path in payload["reference_paths"]]
        frames = Path(payload["output_path"])
        frames.mkdir()
        for index, color in enumerate(("red", "green", "blue")):
            Image.new("RGB", (8, 4), color).save(frames / f"frame-{index:06d}.png")
        with wave.open(payload["audio_path"], "wb") as audio:
            audio.setnchannels(2)
            audio.setsampwidth(2)
            audio.setframerate(32000)
            audio.writeframes(struct.pack("<hh", 0, 0) * 4000)
        result = Future()
        result.set_result({"width": 8, "height": 4, "num_frames": 3, "channels": 3, "fps": 24.0,
                           "audio_sample_rate": 32000, "audio_channels": 2, "audio_samples": 4000,
                           "audio_start_seconds": 0.0, "model_call_ms": 30.0,
                           "timing_scope": "public_task_call_wall"})
        return result

    def retire(self, error):
        result = Future()
        result.set_result(None)
        return result


def app_for(worker):
    return create_app(worker, ServerConfig(max_body_bytes=64000, max_prompt_bytes=128, max_generation_tokens=128))


def finished(client, identifier):
    for _ in range(500):
        value = client.get(f"/v1/videos/{identifier}").json()
        if value["status"] in {"completed", "failed"}:
            return value
        time.sleep(0.01)
    raise AssertionError("video job did not finish")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="external FFmpeg codec unavailable")
def test_reference_images_reach_the_worker_and_the_clip_keeps_its_audio(tmp_path):
    worker = ReferenceWorker()
    with TestClient(app_for(worker)) as client:
        response = client.post("/v1/videos", data={"model": "h3", "prompt": "two people talk", "seed": "7",
                                                   "continuation_prompt": "they keep still"},
                               files=[("input_reference", ("a.png", png("red"), "image/png")),
                                      ("input_reference", ("b.png", png("blue"), "image/png"))])
        assert response.status_code == 202
        view = finished(client, response.json()["id"])
        assert view["status"] == "completed" and view["size"] == "8x4" and view["num_frames"] == 3
        assert view["fps"] == 24
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(client.get(f"/v1/videos/{view['id']}/content").content)
    payload, = worker.calls
    assert payload["prompt"] == "two people talk"
    assert payload["config"] == {"seed": 7, "continuation_prompt": "they keep still"}
    assert worker.references == [png("red"), png("blue")]
    streams = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                              str(clip)], capture_output=True, text=True, check=True).stdout.split()
    assert sorted(streams) == ["audio", "video"]


@pytest.mark.parametrize("data, files", [
    ({"model": "h3"}, [("input_reference", ("a.png", png("red"), "image/png"))]),
    ({"model": "h3", "prompt": "p", "size": "8x4"}, [("input_reference", ("a.png", png("red"), "image/png"))]),
    ({"model": "h3", "prompt": "p", "seed": "-1"}, [("input_reference", ("a.png", png("red"), "image/png"))]),
    ({"model": "h3", "prompt": "p"}, [("image", ("a.png", png("red"), "image/png"))]),
    ({"model": "other", "prompt": "p"}, [("input_reference", ("a.png", png("red"), "image/png"))]),
])
def test_rejects_bad_reference_requests_before_native_execution(data, files):
    worker = ReferenceWorker()
    with TestClient(app_for(worker)) as client:
        assert client.post("/v1/videos", data=data, files=files).status_code == 400
    assert not worker.calls


def test_rejects_references_that_are_not_rgb_images():
    worker = ReferenceWorker()
    with TestClient(app_for(worker)) as client:
        response = client.post("/v1/videos", data={"model": "h3", "prompt": "p"},
                               files=[("input_reference", ("a.png", b"not an image", "image/png"))])
    assert response.status_code == 400
    assert not worker.calls


def test_reference_pixels_count_against_one_limit():
    worker = ReferenceWorker()
    config = ServerConfig(max_body_bytes=64000, max_prompt_bytes=128, max_generation_tokens=128,
                          max_image_pixels=40)
    with TestClient(create_app(worker, config)) as client:
        one = client.post("/v1/videos", data={"model": "h3", "prompt": "p"},
                          files=[("input_reference", ("a.png", png("red"), "image/png"))])
        assert one.status_code == 202
        finished(client, one.json()["id"])
        two = client.post("/v1/videos", data={"model": "h3", "prompt": "p"},
                          files=[("input_reference", ("a.png", png("red"), "image/png")),
                                 ("input_reference", ("b.png", png("blue"), "image/png"))])
    assert two.status_code == 413
    assert len(worker.calls) == 1
