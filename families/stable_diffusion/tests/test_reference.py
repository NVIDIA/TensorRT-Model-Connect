# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from families.stable_diffusion.reference.adapter import Adapter


def test_latent_shape_uses_the_checkpoint_geometry(tmp_path):
    for name, value in (("unet/config.json", {"in_channels": 4}),
                        ("vae/config.json", {"block_out_channels": [128, 256, 512, 512]})):
        path = tmp_path / "checkpoint" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    request = {"height": 512, "width": 512}
    assert Adapter.latent_shape(tmp_path / "checkpoint", request) == (1, 4, 64, 64)
    with pytest.raises(ValueError, match="divisible"):
        Adapter.latent_shape(tmp_path / "checkpoint", {**request, "width": 513})


@pytest.mark.parametrize("pipeline,sampler", [
    ("StableDiffusionPipeline", "DDIM"),
    ("StableDiffusionXLPipeline", "EulerAncestralDiscrete"),
])
@pytest.mark.parametrize("mode", ["eager", "compile"])
def test_reference_selects_the_checkpoint_sampler_and_replays_controls(monkeypatch, tmp_path, pipeline, sampler, mode):
    loads = []
    calls = []
    compiled = []

    class Tensor:
        def __init__(self, values):
            self.values = values

        def to(self, **kwargs):
            assert kwargs == {"device": "cpu", "dtype": "fp32"}
            return self

    class Pipeline:
        unet = SimpleNamespace(config=SimpleNamespace(in_channels=4), forward=object())
        vae_scale_factor = 8

        @classmethod
        def from_pretrained(cls, model, **kwargs):
            loads.append((cls.name, model, kwargs))
            return cls()

        def to(self, device):
            assert device == "cpu"
            return self

        def set_progress_bar_config(self, **kwargs):
            assert kwargs == {"disable": True}

        def __call__(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(images=np.ones((1, 8, 8, 3), dtype=np.float32))

    def scheduler(name):
        def load(model, **kwargs):
            loads.append((name, model, kwargs))
            return name
        return SimpleNamespace(from_pretrained=load)

    def load_config(model, **kwargs):
        loads.append(("config", model, kwargs))
        return {"_class_name": pipeline}

    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(
        DiffusionPipeline=SimpleNamespace(load_config=load_config),
        StableDiffusionPipeline=type("SD", (Pipeline,), {"name": "StableDiffusionPipeline"}),
        StableDiffusionXLPipeline=type("XL", (Pipeline,), {"name": "StableDiffusionXLPipeline"}),
        DDIMScheduler=scheduler("DDIM"), EulerAncestralDiscreteScheduler=scheduler("EulerAncestralDiscrete")))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        from_numpy=Tensor, compile=lambda forward: compiled.append(forward) or "compiled"))
    spec = SimpleNamespace(model="publisher/checkpoint", dtype="fp32", device="cpu", mode=mode,
                           pretrained_kwargs=lambda: {"revision": "pinned"})
    host = SimpleNamespace(required=lambda request, key: request[key], timed=lambda fn: (fn(), 1.0),
                           tensor_observation=lambda values, path: values,
                           invocation=lambda values, ms, **extra: (values, ms, extra))
    adapter = Adapter(spec, host)
    noise = np.arange(4, dtype=np.float32)
    noise.tofile(tmp_path / "latents.f32")
    result, ms, extra = adapter.invoke({"height": 8, "width": 8, "prompt": "cat", "num_steps": 1,
                                      "guidance_scale": 0.0, "initial_latents_path": tmp_path / "latents.f32"}, tmp_path)
    pipeline_options = {"torch_dtype": "fp32", "revision": "pinned"}
    pipeline_options.update({"add_watermarker": False} if pipeline == "StableDiffusionXLPipeline" else
                            {"safety_checker": None, "requires_safety_checker": False})
    assert loads == [("config", "publisher/checkpoint", {"revision": "pinned"}),
                     (pipeline, "publisher/checkpoint", pipeline_options),
                     (sampler, "publisher/checkpoint", {"subfolder": "scheduler", "revision": "pinned"})]
    assert adapter.pipe.scheduler == sampler
    assert len(compiled) == (1 if mode == "compile" else 0)
    assert len(calls) == 1
    assert {key: value for key, value in calls[0].items() if key != "latents"} == {
        "prompt": "cat", "height": 8, "width": 8, "num_inference_steps": 1,
        "guidance_scale": 0.0, "negative_prompt": None, "output_type": "np"}
    np.testing.assert_array_equal(calls[0]["latents"].values, noise.reshape(1, 4, 1, 1))
    assert result.shape == (1, 8, 8, 3) and ms == 1.0
    assert extra == {"sampler": sampler, "safety_checker": False}
