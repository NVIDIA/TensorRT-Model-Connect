# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
from concurrent.futures import Future
import io
import json
from pathlib import Path
import threading

from fastapi.testclient import TestClient
import httpx
from PIL import Image, ImageFile
import pytest

from trtmc_server.app import ServerConfig, create_app
from trtmc_server.errors import ModelNotFoundError, WorkerCrashedError, WorkerSaturatedError
from trtmc_server.registry import ModelRegistry
from trtmc_server import uploads


def image(fmt="PNG", mode="RGB"):
    data = io.BytesIO()
    Image.new(mode, (8, 4)).save(data, format=fmt)
    return data.getvalue()


class Session:
    replica_name = "detector-1"

    def __init__(self, registry):
        self.registry = registry

    def submit(self, operation, payload):
        path = Path(payload["image_path"])
        self.registry.paths.append(path)
        assert path.read_bytes() == self.registry.image
        assert operation == "detect" and "config" not in payload
        self.registry.calls += 1
        if self.registry.submit_error:
            raise WorkerCrashedError("private path")
        result = Future()
        result.set_result(self.registry.result)
        return result

    def close(self):
        self.registry.closed += 1

    def retire(self, error):
        self.registry.retired = True
        result = Future()
        result.set_result(None)
        return result


class Registry:
    def __init__(self):
        self.image = image()
        self.result = {"boxes": [{"class_id": 42, "x_min": -2., "y_min": 1.,
                        "x_max": 10., "y_max": 4., "confidence": 0.75},
                        {"class_id": 42, "x_min": 0., "y_min": 0., "x_max": 1., "y_max": 2.,
                         "confidence": 0.25}], "image_height": 4, "image_width": 8,
                       "model_call_ms": 1.5, "timing_scope": "public_task_call_wall"}
        self.paths = []
        self.calls, self.closed = 0, 0
        self.saturated, self.draining, self.retired, self.submit_error = False, False, False, False

    def start(self):
        pass

    def close(self):
        pass

    def supports(self, model, capability):
        if model == "missing":
            raise ModelNotFoundError("model is not registered")
        return model == "detector" and capability == "object_detection"

    def default_model(self, capability):
        assert capability == "object_detection"
        return "detector"

    def acquire(self, model):
        if self.saturated:
            raise WorkerSaturatedError("busy")
        if self.draining:
            raise RuntimeError("draining")
        return Session(self)


def client_for(registry, records=None, **limits):
    return TestClient(create_app(registry, ServerConfig(max_body_bytes=16000, max_prompt_bytes=128,
                      max_generation_tokens=128, records=records, **limits)))


def payload(registry, **fields):
    mime = "jpeg" if registry.image.startswith(b"\xff\xd8") else "png"
    return {"input": [{"type": "image_url", "url": f"data:image/{mime};base64," +
                      base64.b64encode(registry.image).decode()}], **fields}


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
def test_complete_boxes_coordinates_scores_and_identity(tmp_path, fmt):
    registry = Registry()
    registry.image = image(fmt)
    records = tmp_path / "requests.jsonl"
    with client_for(registry, records) as client:
        response = client.post("/v1/infer", json=payload(registry), headers={"X-Request-ID": "image-42"})
    assert response.status_code == 200 and response.headers["x-request-id"] == "image-42"
    assert response.json() == {"data": [{"index": 0, "bounding_boxes": {"42": [
        {"x_min": -.25, "y_min": .25, "x_max": 1.25, "y_max": 1., "confidence": .75},
        {"x_min": 0., "y_min": 0., "x_max": .125, "y_max": .5, "confidence": .25}]}}]}
    record = json.loads(records.read_text())
    assert record["model"] == "detector" and record["request_id"] == "image-42"
    assert record["image_width"] == 8 and record["image_height"] == 4 and record["image_pixels"] == 32
    assert record["image_bytes"] == len(registry.image) and record["detection_boxes"] == 2
    assert record["model_call_ms"] == 1.5 and registry.closed == 1
    assert all(not path.exists() for path in registry.paths)


def test_empty_detections_are_a_complete_result():
    registry = Registry()
    registry.result["boxes"] = []
    with client_for(registry) as client:
        assert client.post("/v1/infer", json=payload(registry)).json() == {
            "data": [{"index": 0, "bounding_boxes": {}}]}


@pytest.mark.parametrize("url", ["/etc/passwd", "https://example.org/image.png", "data:image/png;base64,???",
                                 "data:image/jpeg;base64," + base64.b64encode(image()).decode(),
                                 "data:image/png;base64," + base64.b64encode(image(mode="L")).decode()])
def test_invalid_media_rejected_before_native_call(url):
    registry = Registry()
    with client_for(registry) as client:
        response = client.post("/v1/infer", json={"input": [{"type": "image_url", "url": url}]})
    assert response.status_code == 400 and registry.calls == 0


@pytest.mark.parametrize("field,value", [("threshold", .5), ("stream", False), ("prompt", "cat")])
def test_unsupported_controls_rejected(field, value):
    registry = Registry()
    with client_for(registry) as client:
        assert client.post("/v1/infer", json=payload(registry, **{field: value})).status_code == 400
    assert registry.calls == 0


def test_multiple_images_rejected():
    registry = Registry()
    body = payload(registry)
    body["input"] *= 2
    with client_for(registry) as client:
        assert client.post("/v1/infer", json=body).status_code == 400
    assert registry.calls == 0


@pytest.mark.parametrize("kind,status", [("saturated", 429), ("draining", 503), ("missing", 404),
                                        ("other", 400), ("submit_error", 503), ("pixels", 413)])
def test_admission_failure_accounts_input_and_cleans_up(tmp_path, kind, status):
    registry = Registry()
    setattr(registry, kind, True)
    records = tmp_path / "records.jsonl"
    with client_for(registry, records, max_image_pixels=16 if kind == "pixels" else 32) as client:
        response = client.post("/v1/infer", json=payload(registry, model=kind if kind in ("missing", "other") else "detector"))
    assert response.status_code == status
    record = json.loads(records.read_text())
    assert record["status"] == status and record["image_bytes"] == len(registry.image)
    assert record["image_width"] == 8 and all(not path.exists() for path in registry.paths)


@pytest.mark.parametrize("patch", [{"image_width": True}, {"image_width": 7}, {"boxes": None},
                                  {"model_call_ms": 10**400}, {"model_call_ms": -1},
                                  {"timing_scope": "kernel"},
                                  {"boxes": [{"class_id": True}]},
                                  {"boxes": [{"class_id": 1, "x_min": 3., "y_min": 0., "x_max": 1.,
                                              "y_max": 1., "confidence": .5}]}])
def test_invalid_worker_output_retires_and_cleans_up(patch):
    registry = Registry()
    registry.result.update(patch)
    with client_for(registry) as client:
        response = client.post("/v1/infer", json=payload(registry))
    assert response.status_code == 502 and "private" not in response.text
    assert registry.retired and registry.closed == 1 and all(not p.exists() for p in registry.paths)


def test_default_model_requires_unique_capability():
    registry = ModelRegistry.__new__(ModelRegistry)
    registry._order = ["detector", "text"]
    registry._metadata = {"detector": {"capabilities": ["object_detection"]},
                          "text": {"capabilities": ["text_generation"]}}
    assert registry.default_model("object_detection") == "detector"
    registry._metadata["text"]["capabilities"] = ["object_detection"]
    with pytest.raises(ValueError, match="specify model"):
        registry.default_model("object_detection")


def test_corrupt_jpeg_pixels_rejected_before_native_call():
    registry = Registry()
    complete = image("JPEG")
    scan = complete.index(b"\xff\xda")
    registry.image = complete[:scan + 2 + int.from_bytes(complete[scan + 2:scan + 4], "big")]
    with Image.open(io.BytesIO(registry.image)) as header:
        assert header.size == (8, 4)
        header.verify()
    with Image.open(io.BytesIO(registry.image)) as pixels, pytest.raises(OSError):
        pixels.load()
    with client_for(registry) as client:
        response = client.post("/v1/infer", json=payload(registry))
    assert response.status_code == 400 and registry.calls == 0


def test_pixel_limit_is_checked_before_decoding(monkeypatch):
    registry = Registry()

    def unexpected_decode(*args, **kwargs):
        raise AssertionError("oversized images must not be decoded")

    monkeypatch.setattr(ImageFile.ImageFile, "load", unexpected_decode)
    with client_for(registry, max_image_pixels=16) as client:
        response = client.post("/v1/infer", json=payload(registry))
    assert response.status_code == 413 and registry.calls == 0


def test_image_preparation_keeps_unrelated_requests_responsive(monkeypatch):
    async def run():
        loop = asyncio.get_running_loop()
        loop_thread = threading.get_ident()
        started, release = asyncio.Event(), threading.Event()
        inspection_threads = []
        inspect = uploads.image_upload

        def blocked_inspection(*args):
            inspection_threads.append(threading.get_ident())
            loop.call_soon_threadsafe(started.set)
            assert release.wait(3), "event loop did not release image inspection"
            return inspect(*args)

        monkeypatch.setattr(uploads, "image_upload", blocked_inspection)
        monkeypatch.setattr("trtmc_server.app.image_upload", blocked_inspection)
        registry = Registry()
        app = create_app(registry, ServerConfig(max_body_bytes=16000, max_prompt_bytes=128,
                                               max_generation_tokens=128))

        @app.get("/test-ping")
        async def ping():
            return {"alive": True}

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            pending = asyncio.create_task(client.post("/v1/infer", json=payload(registry)))
            try:
                await asyncio.wait_for(started.wait(), 2)
                response = await asyncio.wait_for(client.get("/test-ping"), 2)
                assert response.status_code == 200 and response.json() == {"alive": True}
                assert not pending.done()
            finally:
                release.set()
            assert (await asyncio.wait_for(pending, 3)).status_code == 200
        assert inspection_threads and all(value != loop_thread for value in inspection_threads)

    asyncio.run(run())


@pytest.mark.parametrize("phase", ["inspection", "file_creation"])
def test_cancelled_image_preparation_releases_inflight_resources(monkeypatch, phase):
    async def run():
        loop = asyncio.get_running_loop()
        loop_thread = threading.get_ident()
        started = asyncio.Event()
        release, inspected, closed = threading.Event(), threading.Event(), threading.Event()
        paths, preparation_threads, cleanup_threads = [], [], []
        inspect, original_upload = uploads.image_upload, uploads.UploadedInput

        def block():
            loop.call_soon_threadsafe(started.set)
            assert release.wait(3), "event loop did not release image preparation"

        def inspection(*args):
            preparation_threads.append(threading.get_ident())
            try:
                if phase == "inspection":
                    block()
                return inspect(*args)
            finally:
                inspected.set()

        class TrackedUpload(original_upload):
            def __init__(self, *args):
                preparation_threads.append(threading.get_ident())
                super().__init__(*args)
                paths.append(self.path)
                if phase == "file_creation":
                    block()

            def close(self):
                cleanup_threads.append(threading.get_ident())
                super().close()
                closed.set()

        monkeypatch.setattr("trtmc_server.app.image_upload", inspection)
        monkeypatch.setattr(uploads, "UploadedInput", TrackedUpload)
        registry = Registry()
        app = create_app(registry, ServerConfig(max_body_bytes=16000, max_prompt_bytes=128,
                                               max_generation_tokens=128))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            pending = asyncio.create_task(client.post("/v1/infer", json=payload(registry)))
            try:
                await asyncio.wait_for(started.wait(), 2)
                assert not pending.done()
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
                if phase == "file_creation":
                    assert paths and all(path.exists() for path in paths)
            finally:
                release.set()
            completion = inspected if phase == "inspection" else closed
            assert await asyncio.wait_for(asyncio.to_thread(completion.wait, 3), 4)
        assert registry.calls == 0 and all(not path.exists() for path in paths)
        assert preparation_threads and all(value != loop_thread for value in preparation_threads)
        if phase == "file_creation":
            assert len(paths) == 1 and cleanup_threads
            assert all(value != loop_thread for value in cleanup_threads)
        else:
            assert not paths

    asyncio.run(run())
