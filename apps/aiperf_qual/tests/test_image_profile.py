# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib
import io
import json

from PIL import Image
import pytest

from trtmc_aiperf_qual.config import ConfigError
from trtmc_aiperf_qual.image_profile import bundle_identity, check_image, prepare_inputs


def test_complete_png_identity_and_geometry_are_verified_before_saving(tmp_path):
    image = io.BytesIO()
    Image.new("RGB", (8, 4), color="blue").save(image, format="PNG")
    data = image.getvalue()
    response = {"data": [{"b64_json": base64.b64encode(data).decode()}]}
    record = {"output_sha256": hashlib.sha256(data).hexdigest(), "output_width": 8,
              "output_height": 4, "output_count": 1}
    output = tmp_path / "image.png"
    assert check_image(response, record, output) == 8 * 4 * 3
    assert output.read_bytes() == data
    for change in ({"output_sha256": "0" * 64}, {"output_width": 9}, {"output_count": 2}):
        with pytest.raises(ValueError):
            check_image(response, {**record, **change}, tmp_path / "invalid.png")
    assert not (tmp_path / "invalid.png").exists()


def test_image_workload_keeps_catalog_controls_and_explicit_latent_replay(tmp_path):
    source = tmp_path / "source.jsonl"
    replay = base64.b64encode(b"\0" * 8).decode()
    source.write_text(json.dumps({"text": "A blue rectangle", "extra": {"initial_latents_b64": replay}}) + "\n")
    output = tmp_path / "prepared.jsonl"
    workloads = prepare_inputs(source, output, {"prompt": "catalog prompt", "height": 256, "width": 256,
        "num_steps": 10, "seed": 0, "guidance_scale": 7.5, "cfg_scale": 7.5})
    row = json.loads(output.read_text())
    assert row == {"text": "A blue rectangle", "extra": {"size": "256x256", "num_inference_steps": 10,
        "seed": 0, "guidance_scale": 7.5, "cfg_scale": 7.5, "initial_latents_b64": replay}}
    assert workloads[0]["prompt_sha256"] == hashlib.sha256(b"A blue rectangle").hexdigest()


@pytest.mark.parametrize("changed", ["family", "task", "hf_id", "revision", "precision", "image_width", "quantization"])
def test_wrong_bundle_cannot_be_qualified_as_the_selected_profile(tmp_path, monkeypatch, changed):
    import trtmc_aiperf_qual.image_profile as module

    manifest = {"family": "owner", "task": "image_generation", "hf_id": "owner/model",
                "hf_revision": "a" * 40, "precision": "fp16", "image_width": 256}
    info = {"family": "owner", "task": "image_generation", "provenance": {
        "checkpoint": {"hf_id": "owner/model", "revision": "a" * 40},
        "build": {"precision": "fp16", "image_width": 256}}}
    monkeypatch.setattr(module.subprocess, "check_output", lambda *_, **__: json.dumps(info))
    assert bundle_identity(tmp_path, tmp_path / "model.bundle", manifest) == info
    destination = info if changed in {"family", "task"} else info["provenance"][
        "checkpoint" if changed in {"hf_id", "revision"} else "build"]
    destination[changed] = "wrong"
    with pytest.raises(ConfigError):
        bundle_identity(tmp_path, tmp_path / "wrong.bundle", manifest)
