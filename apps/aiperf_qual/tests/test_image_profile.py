# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib
import io
import json

from PIL import Image
import pytest

from trtmc_aiperf_qual.config import ConfigError
from trtmc_aiperf_qual.image_profile import bundle_identity, check_controls, check_image, check_reference, prepare_inputs


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


def test_edit_dataset_pins_original_jpeg_pixels_without_second_lossy_encoding(tmp_path):
    source = tmp_path / "reference.jpg"
    Image.new("RGB", (8, 4), color=(151, 23, 41)).save(source)
    before = source.read_bytes()
    inputs = tmp_path / "source.jsonl"
    inputs.write_text(json.dumps({"text": "Watercolor", "image": "reference.jpg"}) + "\n")
    output = tmp_path / "prepared.jsonl"
    workloads = prepare_inputs(inputs, output, {"image_path": "catalog.jpg", "height": 1024, "width": 1024,
        "num_steps": 20, "seed": 42, "guidance_scale": 4.0, "cfg_scale": 4.0}, image_edit=True)
    row = json.loads(output.read_text())
    png = base64.b64decode(row["image"].split(",", 1)[1], validate=True)
    with Image.open(io.BytesIO(before)) as original, Image.open(io.BytesIO(png)) as prepared:
        assert prepared.format == "PNG" and prepared.size == original.size
        assert prepared.tobytes() == original.tobytes()
    assert source.read_bytes() == before
    assert workloads[0]["reference"]["upload_sha256"] == hashlib.sha256(png).hexdigest()
    assert workloads[0]["reference"]["source_sha256"] == hashlib.sha256(before).hexdigest()
    record = {"image_sha256": hashlib.sha256(png).hexdigest(), "image_width": 8, "image_height": 4}
    payload = {"image": {"b64_data": base64.b64encode(png).decode()}}
    assert check_reference(payload, record, tmp_path / "checked.png") == record["image_sha256"]
    for changed in ({"image_width": 9}, {"image_sha256": "0" * 64}):
        with pytest.raises(ValueError):
            check_reference(payload, {**record, **changed}, tmp_path / "bad.png")
    assert not (tmp_path / "bad.png").exists()


@pytest.mark.parametrize("images", [{}, {"images": ["a.png", "b.png"]}, {"image": "a.png", "images": ["b.png"]}])
def test_edit_dataset_cannot_silently_drop_reference_images(tmp_path, images):
    inputs = tmp_path / "source.jsonl"
    inputs.write_text(json.dumps({"text": "Watercolor", **images}))
    with pytest.raises(ValueError):
        prepare_inputs(inputs, tmp_path / "prepared.jsonl", {"height": 8, "width": 8,
            "num_steps": 20, "seed": 42, "guidance_scale": 4}, image_edit=True)


def test_edit_bundle_input_geometry_is_recorded_independently_of_requested_output(tmp_path, monkeypatch):
    import trtmc_aiperf_qual.image_profile as module

    manifest = {"family": "owner", "task": "image_edit", "hf_id": "owner/model",
                "hf_revision": "a" * 40, "precision": "bf16", "image_width": 1024, "image_height": 1024}
    info = {"family": "owner", "task": "image_edit", "provenance": {
        "checkpoint": {"hf_id": "owner/model", "revision": "a" * 40},
        "build": {"precision": "bf16", "image_width": 640, "image_height": 382}}}
    monkeypatch.setattr(module.subprocess, "check_output", lambda *_, **__: json.dumps(info))
    assert bundle_identity(tmp_path, tmp_path / "edit.bundle", manifest, check_geometry=False) == info
    info["provenance"]["build"]["precision"] = "fp16"
    with pytest.raises(ConfigError):
        bundle_identity(tmp_path, tmp_path / "wrong.bundle", manifest, check_geometry=False)


def test_qualification_rejects_changed_controls_or_replay_noise():
    payload = {"model": "edit", "prompt": "Watercolor", "size": "1024x1024", "seed": 42,
               "num_inference_steps": 20, "guidance_scale": 4.0, "cfg_scale": 4.0,
               "negative_prompt": " ", "initial_latents_b64": base64.b64encode(b"\0" * 8).decode()}
    record = {"prompt_sha256": hashlib.sha256(b"Watercolor").hexdigest(),
              "negative_prompt_sha256": hashlib.sha256(b" ").hexdigest(),
              "initial_latents_sha256": hashlib.sha256(b"\0" * 8).hexdigest(),
              "generation_config": {"height": 1024, "width": 1024, "seed": 42, "num_steps": 20,
                                    "guidance_scale": 4.0, "cfg_scale": 4.0}}
    check_controls(payload, record)
    workload = {"prompt_sha256": record["prompt_sha256"],
                "controls": {key: value for key, value in payload.items() if key not in {"model", "prompt"}}}
    check_controls(payload, record, [workload])
    with pytest.raises(ValueError):
        check_controls(payload, record, [{**workload, "controls": {**workload["controls"], "seed": 43}}])
    for change in ({"seed": 43}, {"negative_prompt": "other"},
                   {"initial_latents_b64": base64.b64encode(b"\0" * 4).decode()}):
        with pytest.raises(ValueError):
            check_controls({**payload, **change}, record)
