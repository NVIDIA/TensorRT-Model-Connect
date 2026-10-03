# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU recording tests for semantic requests at the Diffusers call boundary."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from qualification_tests.benchmark_qualification.performance.references import generic_reference


@pytest.fixture
def recording_diffusers(monkeypatch, tmp_path):
    recorded = SimpleNamespace(loads=[], constructors=[], calls=[], generators=[], images=[], components=[])

    class Generator:
        def __init__(self, device):
            self.device = device
            self.resets = []
            recorded.generators.append(self)

        def manual_seed(self, seed):
            self.value = seed
            self.resets.append(seed)
            return self

    torch = ModuleType("torch")
    torch.Generator = Generator
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    torch.float16, torch.float32, torch.bfloat16 = "fp16", "fp32", "bf16"
    monkeypatch.setitem(sys.modules, "torch", torch)

    class ConditioningImage:
        def __init__(self, path):
            self.path = Path(path)
            self.data = self.path.read_bytes()

        def convert(self, mode):
            self.mode = mode
            return self

    def open_image(path):
        value = ConditioningImage(path)
        recorded.images.append(value)
        return value

    pil = ModuleType("PIL")
    pil.Image = SimpleNamespace(open=open_image)
    monkeypatch.setitem(sys.modules, "PIL", pil)

    class Transformer:
        def __init__(self):
            self.pre_hooks, self.hooks = [], []

        def register_forward_pre_hook(self, hook, **kwargs):
            self.pre_hooks.append((hook, kwargs))

        def register_forward_hook(self, hook):
            self.hooks.append(hook)

    class Pipeline:
        def __init__(self):
            self.transformer = Transformer()
            self.devices, self.offloads = [], 0
            recorded.constructors.append(self)

        @classmethod
        def from_pretrained(cls, source, **kwargs):
            recorded.loads.append((cls.__name__, source, kwargs))
            return cls()

        def to(self, device):
            self.devices.append(device)
            return self

        def enable_model_cpu_offload(self):
            self.offloads += 1

        def record(self, values):
            values.pop("self", None)
            generators = values["generator"]
            generators = generators if isinstance(generators, list) else [generators]
            values["observed_seeds"] = [item.value for item in generators]
            recorded.calls.append(values)
            for item in generators:
                item.value = -123
            prompt = values["prompt"]
            count = len(prompt) if isinstance(prompt, list) else 1
            return SimpleNamespace(images=np.zeros((count, 8, 10, 3), dtype=np.float32))

        def __call__(self, prompt, negative_prompt=None, height=None, width=None,
                     num_inference_steps=None, guidance_scale=None, true_cfg_scale=4.0,
                     generator=None, output_type="pil"):
            return self.record(locals())

    class EditPipeline(Pipeline):
        def __call__(self, image, prompt, negative_prompt=None, height=None, width=None,
                     num_inference_steps=None, guidance_scale=None, true_cfg_scale=4.0,
                     generator=None, output_type="pil"):
            return self.record(locals())

    diffusers = ModuleType("diffusers")
    for name in ("FluxPipeline", "Flux2Pipeline", "QwenImagePipeline", "PixArtSigmaPipeline", "WanPipeline"):
        setattr(diffusers, name, type(name, (Pipeline,), {}))
    for name in ("QwenImageEditPipeline", "QwenImageEditPlusPipeline"):
        setattr(diffusers, name, type(name, (EditPipeline,), {}))

    class DiffusionPipeline:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            recorded.loads.append((cls.__name__, source, kwargs))
            index = json.loads((Path(source) / "model_index.json").read_text(encoding="utf-8"))
            return getattr(diffusers, index["_class_name"])()

    diffusers.DiffusionPipeline = DiffusionPipeline

    class Component:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            instance = cls()
            recorded.components.append((source, kwargs, instance))
            return instance

    diffusers.AutoencoderKLWan = Component
    transformers = ModuleType("transformers")
    transformers.T5EncoderModel = Component
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    recorded.torch = torch

    def arguments(class_name="FluxPipeline", family="flux", task="text_to_image", **overrides):
        checkpoint = tmp_path / class_name
        checkpoint.mkdir(exist_ok=True)
        (checkpoint / "model_index.json").write_text(json.dumps({"_class_name": class_name}), encoding="utf-8")
        manifest = tmp_path / "manifests" / (class_name + ".json")
        manifest.parent.mkdir(exist_ok=True)
        manifest.write_text(json.dumps({"task": task, "image_height": 32, "image_width": 48}), encoding="utf-8")
        values = dict(model=str(checkpoint), family=family, manifest=manifest, precision="bf16",
                      selected_task=task, local_files_only=True, revision="pinned-revision", trust_remote_code=False)
        values.update(overrides)
        return SimpleNamespace(**values)

    recorded.arguments = arguments
    return recorded


@pytest.mark.parametrize("payload,expected_prompt,expected_seeds", [
    ({"prompt": "legacy"}, "legacy", [42]),
    ({"prompt": "repeat", "batch_size": 2, "seed": 7}, ["repeat", "repeat"], [7, 7]),
    ({"prompts": ["second", "first"], "batch_size": 2, "seeds": [19, 3]}, ["second", "first"], [19, 3]),
    ({"prompts": ["second", "first"], "seeds": [19, 3]}, ["second", "first"], [19, 3]),
    ({"prompt": ["second", "first"], "seeds": [0, 2**32 - 1]}, ["second", "first"], [0, 2**32 - 1]),
    ({"prompt": ["one"], "seeds": [0]}, ["one"], [0]),
    ({"prompt": ["second", "first"], "batch_size": 2, "config": {"seed": 11}}, ["second", "first"], [11, 11]),
])
def test_prompt_and_seed_order_reaches_every_pipeline_invocation(recording_diffusers, payload, expected_prompt, expected_seeds):
    recorded = recording_diffusers
    source = deepcopy(payload)
    session = generic_reference._load_diffusers(recorded.arguments(), generic_reference.flatten_config(payload), {})
    summaries = [session.invoke() for _ in range(3)]
    assert len(recorded.loads) == len(recorded.constructors) == 1
    assert len(recorded.calls) == 3
    assert all(call["prompt"] == expected_prompt for call in recorded.calls)
    assert all(call["observed_seeds"] == expected_seeds for call in recorded.calls)
    assert [generator.resets for generator in recorded.generators] == [[seed] * 4 for seed in expected_seeds]
    assert all(generator.device == "cuda" for generator in recorded.generators)
    for summary in summaries:
        media = summary.pop("_media")
        assert np.asarray(media).shape == (len(expected_seeds), 8, 10, 3)
        assert summary == {"media_type": "image", "media_count": len(expected_seeds),
                           "height": 8, "width": 10, "channels": 3, "finite": True}
    assert payload == source


def test_measurement_resets_each_seed_after_warmups(recording_diffusers):
    recorded = recording_diffusers
    session = generic_reference._load_diffusers(
        recorded.arguments(), {"prompt": ["second", "first"], "seeds": [2**32 - 1, 0]}, {},
    )
    samples, summary = generic_reference._measure(session, warmup=2, iterations=3)
    assert len(samples) == 3
    assert len(recorded.calls) == 5
    assert all(call["observed_seeds"] == [2**32 - 1, 0] for call in recorded.calls)
    assert [generator.resets for generator in recorded.generators] == [[2**32 - 1] * 6, [0] * 6]
    assert len(recorded.loads) == len(recorded.constructors) == 1
    assert summary["media_count"] == 2


@pytest.mark.parametrize("payload,match", [
    ({"prompt": ["a", "b"], "batch_size": 1}, "one string per batch item"),
    ({"prompts": ["a", "b"], "batch_size": 3}, "one string per batch item"),
    ({"prompt": ["a", "b"], "seeds": [1]}, "one integer per batch item"),
    ({"prompt": ["a"], "seeds": [1, 2]}, "one integer per batch item"),
    ({"prompt": [], "seeds": []}, "one string per batch item"),
    ({"prompt": ["a", 1]}, "one string per batch item"),
    ({"prompt": 1}, "prompt must be a string"),
    ({"prompt": "a", "prompts": ["b"]}, "both prompt and prompts"),
    ({"prompt": ["a"], "prompts": ["a"]}, "both prompt and prompts"),
    ({"prompts": "a"}, "one string per batch item"),
    ({"prompt": "a", "batch_size": 0}, "positive integer"),
    ({"prompt": "a", "batch_size": True}, "positive integer"),
    ({"prompt": "a", "batch_size": 1.5}, "positive integer"),
])
def test_invalid_batch_requests_do_not_call_pipeline(recording_diffusers, payload, match):
    recorded = recording_diffusers
    with pytest.raises(ValueError, match=match):
        generic_reference._load_diffusers(recorded.arguments(), payload, {})
    assert recorded.calls == []


@pytest.mark.parametrize("image_key", ["image_path", "image_paths"])
@pytest.mark.parametrize("class_name", ["QwenImageEditPipeline", "QwenImageEditPlusPipeline"])
def test_qwen_edit_checkpoint_selects_edit_class_and_receives_full_request(recording_diffusers, tmp_path, image_key, class_name):
    recorded = recording_diffusers
    arguments = recorded.arguments(class_name, "synthetic", "images_text_to_image_edit")
    image = tmp_path / "condition.ppm"
    image.write_bytes(b"P6\n1 1\n255\n\xff\x00\x80")
    request = {"prompt": "Make it blue", image_key: [image.name] if image_key == "image_paths" else image.name,
               "config": {"seed": 2**32 - 1, "num_steps": 5, "guidance_scale": 3.25, "negative_prompt": " "}}
    session = generic_reference._load_diffusers(
        arguments,
        generic_reference.flatten_config(request),
        {
            "cpu_offload": True,
            "always_negative_prompt": True,
            "guidance_parameter": "true_cfg_scale",
        },
    )
    session.invoke()
    pipeline = recorded.constructors[0]
    assert pipeline.__class__.__name__ == class_name
    assert len(recorded.loads) == len(recorded.constructors) == len(recorded.calls) == 1
    assert recorded.loads[0] == ("DiffusionPipeline", Path(arguments.model), {
        "torch_dtype": "bf16", "trust_remote_code": False, "local_files_only": True})
    assert pipeline.offloads == 1
    assert pipeline.devices == []
    conditioning = recorded.images[0]
    assert conditioning.path == image
    assert conditioning.data == image.read_bytes()
    assert conditioning.mode == "RGB"
    assert recorded.calls[0] == {"image": conditioning, "prompt": "Make it blue", "negative_prompt": " ",
                                 "height": 32, "width": 48, "num_inference_steps": 5,
                                 "guidance_scale": None, "true_cfg_scale": 3.25,
                                 "generator": recorded.generators[0], "output_type": "np",
                                 "observed_seeds": [2**32 - 1]}


@pytest.mark.parametrize("controls,expected", [
    ({"guidance_scale": 2.5}, 2.5),
    ({"guidance_scale": 0.0}, 0.0),
    ({"cfg_scale": 4.0}, 4.0),
    ({"guidance_scale": -1.0, "cfg_scale": 4.0}, 4.0),
    ({"guidance_scale": 2.5, "cfg_scale": 4.0}, 2.5),
    ({}, 4.0),
])
@pytest.mark.parametrize("negative", [{}, {"negative_prompt": ""}])
def test_qwen_native_guidance_controls_true_cfg_and_preserves_empty_negative_prompt(recording_diffusers, controls, expected, negative):
    recorded = recording_diffusers
    arguments = recorded.arguments("QwenImagePipeline", "synthetic")
    session = generic_reference._load_diffusers(
        arguments,
        {"prompt": "a", **negative, **controls},
        {"always_negative_prompt": True, "guidance_parameter": "true_cfg_scale"},
    )
    session.invoke()
    assert recorded.calls[0]["true_cfg_scale"] == expected
    assert recorded.calls[0]["guidance_scale"] is None
    assert recorded.calls[0]["negative_prompt"] == ""


def test_non_qwen_guidance_retains_provider_parameter(recording_diffusers):
    recorded = recording_diffusers
    generic_reference._load_diffusers(recorded.arguments(), {"prompt": "a", "guidance_scale": 1.5}, {}).invoke()
    assert recorded.calls[0]["guidance_scale"] == 1.5
    assert recorded.constructors[0].devices == ["cuda"]


@pytest.mark.parametrize("inputs,match", [
    ({}, "requires one conditioning image"),
    ({"image_paths": []}, "exactly one conditioning image"),
    ({"image_paths": ["a", "b"]}, "exactly one conditioning image"),
    ({"image_paths": "a"}, "exactly one conditioning image"),
    ({"image_path": "a", "image_paths": ["a"]}, "both image_path and image_paths"),
])
def test_qwen_edit_image_contract_fails_before_invocation(recording_diffusers, inputs, match):
    recorded = recording_diffusers
    arguments = recorded.arguments(
        "QwenImageEditPlusPipeline", "synthetic", "images_text_to_image_edit"
    )
    with pytest.raises(ValueError, match=match):
        generic_reference._load_diffusers(arguments, {"prompt": "edit", **inputs}, {})
    assert recorded.calls == recorded.images == []


def test_conditioning_image_is_never_silently_filtered(recording_diffusers):
    recorded = recording_diffusers
    arguments = recorded.arguments("QwenImagePipeline", "synthetic", "images_text_to_image_edit")
    with pytest.raises(ValueError, match="does not accept conditioning image"):
        generic_reference._load_diffusers(arguments, {"prompt": "edit", "image_paths": ["condition.ppm"]}, {})
    assert recorded.calls == []


@pytest.mark.parametrize("class_name,options,component_key,subfolder", [
    (
        "PixArtSigmaPipeline",
        {"component_precision_contract": "pixart_fp16_dit_fp32_t5"},
        "text_encoder",
        "text_encoder",
    ),
    (
        "WanPipeline",
        {"vae_class": "AutoencoderKLWan", "vae_precision": "fp32"},
        "vae",
        "vae",
    ),
])
@pytest.mark.parametrize("local_files_only", [True, False])
def test_checkpoint_owned_loader_preserves_component_precision_and_revision(
    recording_diffusers, class_name, options, component_key, subfolder, local_files_only,
):
    recorded = recording_diffusers
    arguments = recorded.arguments(
        class_name, "synthetic", precision="fp16", local_files_only=local_files_only
    )
    options = {
        **options,
        "model_revision": "override-revision",
        "trust_remote_code": True,
    }
    session = generic_reference._load_diffusers(arguments, {"prompt": "a"}, options)
    session.invoke()
    source, kwargs, component = recorded.components[0]
    expected_source = Path(arguments.model) if local_files_only else arguments.model
    revision = {} if local_files_only else {"revision": "override-revision"}
    assert source == expected_source
    assert kwargs == {"subfolder": subfolder, "torch_dtype": "fp32", "local_files_only": local_files_only, **revision}
    assert recorded.loads == [("DiffusionPipeline", expected_source, {
        "torch_dtype": "fp16", component_key: component, "local_files_only": local_files_only,
        "trust_remote_code": True, **revision})]
    assert recorded.constructors[0].__class__.__name__ == class_name
    if options.get("component_precision_contract"):
        transformer = recorded.constructors[0].transformer
        assert len(transformer.pre_hooks) == len(transformer.hooks) == 1
        assert transformer.pre_hooks[0][1] == {"with_kwargs": True}


def test_model_id_override_and_missing_local_snapshot_keep_load_contract(recording_diffusers, monkeypatch):
    recorded = recording_diffusers
    arguments = recorded.arguments()
    override = recorded.arguments("Flux2Pipeline").model
    pipeline = generic_reference._diffusion_pipeline(arguments, recorded.torch, {"model_id": override})
    assert pipeline.__class__.__name__ == "Flux2Pipeline"
    assert recorded.loads[0][1] == Path(override)
    monkeypatch.setattr(generic_reference, "_cached_snapshot_path", lambda *_: None)
    with pytest.raises(FileNotFoundError, match="cached Diffusers snapshot is missing"):
        generic_reference._diffusion_pipeline(arguments, recorded.torch, {})
    assert len(recorded.loads) == 1
