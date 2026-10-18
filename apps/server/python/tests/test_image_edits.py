# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
from concurrent.futures import Future
import hashlib
import io
import json
from pathlib import Path

from fastapi.testclient import TestClient
import httpx
from PIL import Image
import pytest

from trtmc_server.app import ServerConfig, create_app
from trtmc_server.registry import ModelRegistry


def reference(mode="RGB", color="red"):
    data = io.BytesIO()
    Image.new(mode, (2, 1), color=color).save(data, format="PNG")
    return data.getvalue()


class Editor:
    replica_name = "edit-1"

    def __init__(self, *, pending=False, wrong_dimensions=False):
        self.calls = []
        self.closed = 0
        self.retired = False
        self.pending = Future() if pending else None
        self.wrong_dimensions = wrong_dimensions

    def start(self):
        pass

    def close(self):
        self.closed += 1

    def supports(self, model, capability):
        return model == "edit" and capability == "image_edit"

    def acquire(self, model):
        return self

    def submit(self, operation, payload):
        assert operation == "edit_image"
        self.calls.append(payload)
        self.noise = Path(payload["initial_latents_path"]).read_bytes() if "initial_latents_path" in payload else None
        with Image.open(payload["image_path"]) as source:
            dimensions = source.size
            Image.new("RGB", (8, 4), color=source.getpixel((0, 0))).save(payload["output_path"])
        self.result = {"width": 8, "height": 4, "channels": 3, "output_count": 1,
                       "image_width": 9 if self.wrong_dimensions else dimensions[0],
                       "image_height": dimensions[1], "model_call_ms": 13.0,
                       "timing_scope": "public_task_call_wall"}
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


def app_for(editor, records=None, **limits):
    return create_app(editor, ServerConfig(max_body_bytes=16000, max_prompt_bytes=128,
                      max_generation_tokens=128, records=records, **limits))


def edit(client, *, fields=None, data=None):
    return client.post("/v1/images/edits", data={"model": "edit", "prompt": "Watercolor", **(fields or {})},
                       files={"image": ("../../untrusted.png", data if data is not None else reference(), "image/png")},
                       headers={"X-Request-ID": "edit-42"})


def test_real_multipart_reference_controls_and_complete_output_are_correlated(tmp_path):
    editor = Editor()
    records = tmp_path / "records.jsonl"
    ModelRegistry._validate_ready({"event": "ready", "protocol_version": 1, "capabilities": ["image_edit"]})
    noise = b"\0" * 8
    with TestClient(app_for(editor, records)) as client:
        response = edit(client, fields={"n": "1", "stream": "false", "size": "8x4",
            "seed": "42", "num_inference_steps": "20", "guidance_scale": "4.0", "cfg_scale": "4.0",
            "negative_prompt": "private unwanted scene", "initial_latents_b64": base64.b64encode(noise).decode()})
    assert response.status_code == 200 and response.headers["x-request-id"] == "edit-42"
    call, = editor.calls
    assert call["config"] == {"width": 8, "height": 4, "seed": 42, "num_steps": 20,
                              "guidance_scale": 4.0, "cfg_scale": 4.0, "negative_prompt": "private unwanted scene"}
    decoded = base64.b64decode(response.json()["data"][0]["b64_json"], validate=True)
    with Image.open(io.BytesIO(decoded)) as image:
        assert image.size == (8, 4) and image.getpixel((7, 3)) == (255, 0, 0)
    assert all(not Path(call[key]).exists() for key in ("image_path", "initial_latents_path", "output_path"))
    record = json.loads(records.read_text())
    assert record["request_id"] == "edit-42" and record["status"] == 200
    assert (record["image_width"], record["image_height"]) == (2, 1)
    assert record["image_sha256"] == hashlib.sha256(reference()).hexdigest()
    assert record["output_sha256"] == hashlib.sha256(decoded).hexdigest()
    assert record["initial_latents_sha256"] == hashlib.sha256(noise).hexdigest()
    assert record["model_call_ms"] == 13 and record["timing_scope"] == "public_task_call_wall"
    assert "private unwanted scene" not in records.read_text()


@pytest.mark.parametrize("fields", [{"n": "2"}, {"stream": "true"}, {"response_format": "url"},
    {"mask": "ignored"}, {"url": "https://example.com/image.png"}, {"quality": "hd"},
    {"seed": "1.0"}, {"seed": str(2**32)}, {"n": "1.0"}, {"num_inference_steps": "0"},
    {"guidance_scale": "nan"}, {"initial_latents_b64": "bad"}, {"size": "auto"}])
def test_unsupported_multipart_controls_do_not_run_the_task(fields):
    editor = Editor()
    with TestClient(app_for(editor)) as client:
        response = edit(client, fields=fields)
    assert response.status_code == 400 and not editor.calls


@pytest.mark.parametrize("data", [b"", b"not an image", reference("RGBA"), reference()[:40]])
def test_invalid_reference_is_rejected_before_native_execution(data):
    editor = Editor()
    with TestClient(app_for(editor)) as client:
        response = edit(client, data=data)
    assert response.status_code == 400 and not editor.calls


def test_duplicate_fields_multiple_images_and_generation_route_are_rejected():
    editor = Editor()
    with TestClient(app_for(editor)) as client:
        multiple = client.post("/v1/images/edits", data={"model": "edit", "prompt": "Watercolor"},
            files=[("image", ("a.png", reference())), ("image", ("b.png", reference(color="blue")))])
        duplicate = client.post("/v1/images/edits", files=[("model", (None, "edit")),
            ("model", (None, "edit")), ("prompt", (None, "Watercolor")), ("image", ("a.png", reference()))])
        generation = client.post("/v1/images/generations", json={"model": "edit", "prompt": "Watercolor"})
    assert multiple.status_code == duplicate.status_code == generation.status_code == 400
    assert not editor.calls


def test_reference_pixel_limit_and_invalid_native_input_dimensions():
    editor = Editor()
    with TestClient(app_for(editor, max_image_pixels=1)) as client:
        assert edit(client).status_code == 413
    assert not editor.calls
    editor = Editor(wrong_dimensions=True)
    with TestClient(app_for(editor)) as client:
        assert edit(client).status_code == 502
    assert editor.retired and not Path(editor.calls[0]["image_path"]).exists()


def test_large_replay_form_field_uses_the_configured_body_limit():
    editor = Editor()
    app = create_app(editor, ServerConfig(max_body_bytes=2 * 1024 * 1024,
                     max_prompt_bytes=128, max_generation_tokens=128))
    noise = b"\0" * 800_000
    encoded = base64.b64encode(noise).decode()
    assert len(encoded) > 1024 * 1024
    with TestClient(app) as client:
        response = edit(client, fields={"initial_latents_b64": encoded})
    assert response.status_code == 200 and editor.noise == noise
    assert not Path(editor.calls[0]["initial_latents_path"]).exists()


def test_cancelled_edit_keeps_both_uploads_until_the_native_call_finishes():
    async def run():
        editor = Editor(pending=True)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_for(editor)), base_url="http://test") as client:
            request = asyncio.create_task(client.post("/v1/images/edits",
                data={"model": "edit", "prompt": "Watercolor", "initial_latents_b64": base64.b64encode(b"\0" * 8).decode()},
                files={"image": ("image.png", reference(), "image/png")}))
            for _ in range(100):
                if editor.calls:
                    break
                await asyncio.sleep(0.01)
            call, = editor.calls
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            assert all(Path(call[key]).exists() for key in ("image_path", "initial_latents_path", "output_path"))
            editor.pending.set_result(editor.result)
            for _ in range(20):
                await asyncio.sleep(0)
            assert all(not Path(call[key]).exists() for key in ("image_path", "initial_latents_path", "output_path"))
            assert editor.closed == 1
    asyncio.run(run())
