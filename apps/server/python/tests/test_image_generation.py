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
from PIL import Image
import pytest

from trtmc_server.app import ServerConfig, _worker_request, create_app
from trtmc_server.registry import ModelRegistry
from trtmc_server.uploads import UploadedInput


class Registry:
    replica_name = "image-1"

    def __init__(self):
        data = io.BytesIO()
        Image.new("RGB", (8, 4), color="red").save(data, format="PNG")
        self.png = data.getvalue()
        self.calls = []
        self.closed = 0
        self.retired = False
        self.result = {"width": 8, "height": 4, "output_count": 1, "channels": 3,
                       "model_call_ms": 12.5, "timing_scope": "public_task_call_wall"}

    def start(self):
        pass

    def close(self):
        self.closed += 1

    def acquire(self, model):
        return self
    def supports(self, model, capability):
        return model == "image" and capability == "image_generation"

    def submit(self, operation, payload):
        assert operation == "generate_image"
        self.calls.append(payload)
        Path(payload["output_path"]).write_bytes(self.png)
        if "initial_latents_path" in payload:
            assert Path(payload["initial_latents_path"]).read_bytes() == b"\0" * 8
        result = Future()
        result.set_result(self.result)
        return result

    def retire(self, error):
        self.retired = True
        result = Future()
        result.set_result(None)
        return result


def client_for(registry, records=None, **limits):
    return TestClient(create_app(registry, ServerConfig(max_body_bytes=16000, max_prompt_bytes=128,
                      max_generation_tokens=128, records=records, **limits)))


def test_complete_native_png_remains_decodable_after_request_cleanup(tmp_path):
    registry = Registry()
    records = tmp_path / "requests.jsonl"
    ModelRegistry._validate_ready({"event": "ready", "protocol_version": 1,
                                  "capabilities": ["image_generation"]})
    with client_for(registry, records) as client:
        response = client.post("/v1/images/generations", json={"model": "image", "prompt": "A red rectangle",
            "size": "8x4", "num_inference_steps": 10, "guidance_scale": 7.5, "cfg_scale": 1.0,
            "seed": 42, "negative_prompt": "private unwanted scene",
            "initial_latents_b64": base64.b64encode(b"\0" * 8).decode()},
            headers={"X-Request-ID": "image-42"})
    assert response.status_code == 200 and response.headers["x-request-id"] == "image-42"
    data, = response.json()["data"]
    decoded = base64.b64decode(data["b64_json"], validate=True)
    assert decoded == registry.png
    with Image.open(io.BytesIO(decoded)) as image:
        assert image.size == (8, 4) and image.getpixel((7, 3)) == (255, 0, 0)
    call, = registry.calls
    assert call["config"] == {"width": 8, "height": 4, "num_steps": 10, "guidance_scale": 7.5,
                              "cfg_scale": 1.0, "seed": 42, "negative_prompt": "private unwanted scene"}
    assert not Path(call["output_path"]).exists() and not Path(call["initial_latents_path"]).exists()
    record = json.loads(records.read_text())
    assert record["request_id"] == "image-42" and record["status"] == 200
    assert record["output_sha256"] == hashlib.sha256(decoded).hexdigest()
    assert (record["output_width"], record["output_height"], record["output_count"]) == (8, 4, 1)
    assert record["model_call_ms"] == 12.5 and record["timing_scope"] == "public_task_call_wall"
    assert record["initial_latents_sha256"] == hashlib.sha256(b"\0" * 8).hexdigest()
    assert "private unwanted scene" not in records.read_text()
    assert "negative_prompt" not in record["generation_config"]
    assert record["negative_prompt_sha256"] == hashlib.sha256(b"private unwanted scene").hexdigest()
    assert record["negative_prompt_bytes"] == len(b"private unwanted scene")


@pytest.mark.parametrize("options", [{"n": 2}, {"stream": True}, {"response_format": "url"},
    {"quality": "hd"}, {"size": "auto"}, {"size": "0x4"}, {"num_inference_steps": 0},
    {"seed": 2**32}, {"initial_latents_b64": "??"}, {"initial_latents_b64": "YQ=="},
    {"prompt": " "}, {"model": "encoder"}, {"initial_latents_path": "/private/path"}])
def test_unhonored_options_are_rejected_without_native_execution(options):
    registry = Registry()
    with client_for(registry) as client:
        response = client.post("/v1/images/generations", json={"model": "image", "prompt": "red", **options})
    assert response.status_code == 400 and not registry.calls


@pytest.mark.parametrize("bad", ["missing", "truncated", "geometry", "count", "timing"])
def test_invalid_native_outputs_retire_worker_and_cleanup(bad):
    registry = Registry()
    if bad == "missing":
        registry.png = b""
    elif bad == "truncated":
        registry.png = registry.png[:40]
    else:
        registry.result.update({"geometry": {"width": 9}, "count": {"output_count": 2},
                                "timing": {"model_call_ms": -1}}[bad])
    with client_for(registry) as client:
        response = client.post("/v1/images/generations", json={"model": "image", "prompt": "red"})
    assert response.status_code == 502 and registry.retired
    assert not Path(registry.calls[0]["output_path"]).exists()


def test_cancelled_generation_keeps_output_directory_until_native_finishes():
    async def run():
        media = UploadedInput(b"", ".f32")
        output = media.path.with_name("output.png")
        pending = Future()
        registry = Registry()
        registry.submit = lambda *_: pending
        task = asyncio.create_task(_worker_request(registry, {}, operation="generate_image",
                                                   validate=lambda _: None, release_input=media.close))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert output.parent.exists() and registry.closed == 0
        output.write_bytes(registry.png)
        pending.set_result({})
        for _ in range(10):
            await asyncio.sleep(0)
        assert not output.parent.exists() and registry.closed == 1
    asyncio.run(run())
