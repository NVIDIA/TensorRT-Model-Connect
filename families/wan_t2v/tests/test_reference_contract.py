# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from . import test_e2e as e2e


class _Tensor:
    def __init__(self, values: np.ndarray):
        self.values = values

    def to(self, *, device, dtype):
        self.device = device
        self.dtype = dtype
        return self


class _Weight:
    shape = (8, 4)

    def __init__(self, pointer: int):
        self.pointer = pointer

    def data_ptr(self) -> int:
        return self.pointer


def _framework(monkeypatch) -> tuple[dict, object]:
    calls: dict = {}
    fp32 = object()
    torch = ModuleType("torch")
    torch.float16 = object()
    torch.float32 = fp32
    torch.bfloat16 = object()
    torch.from_numpy = _Tensor

    class Generator:
        def __init__(self, device: str):
            self.device = device

        def manual_seed(self, seed: int):
            self.seed = seed
            return self

    torch.Generator = Generator

    class TextEncoder:
        def __init__(self):
            self.shared = SimpleNamespace(weight=_Weight(1))
            self.encoder = SimpleNamespace(embed_tokens=self.shared)
            self.tied = False

        def tie_weights(self):
            self.tied = True

    class Pipeline:
        def __init__(self):
            self.text_encoder = TextEncoder()

        @classmethod
        def from_pretrained(cls, model_dir, **kwargs):
            calls["model_dir"] = model_dir
            calls["load"] = kwargs
            calls["pipeline"] = cls()
            return calls["pipeline"]

        def to(self, device: str):
            calls["device"] = device
            return self

        def __call__(self, **kwargs):
            calls["kwargs"] = kwargs
            pixels = np.array([[[17, 128, 255], [0, 64, 192]]], dtype=np.uint8)
            if kwargs.get("output_type", "np") == "pil":
                return SimpleNamespace(frames=[[Image.fromarray(pixels)]])
            return SimpleNamespace(frames=np.asarray([[pixels]], dtype=np.float32) / 255.0)

    diffusers = ModuleType("diffusers")
    diffusers.WanPipeline = Pipeline
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    return calls, fp32


def test_wan_latents_use_the_family_numpy_contract() -> None:
    _, manifest, case = e2e.CASES["wan21-t2v-1.3b-l0"]
    actual = e2e._initial_latents(manifest, case)
    expected = np.random.default_rng(int(case["seed"])).standard_normal(
        (1, 16, 2, 48, 84), dtype=np.float32
    )
    np.testing.assert_array_equal(actual, expected)


def test_every_wan_reference_is_fp32() -> None:
    assert {case["reference_precision"] for _, _, case in e2e.CASES.values()} == {"fp32"}


def test_native_receives_the_exact_raw_latents(monkeypatch, tmp_path: Path) -> None:
    _, manifest, case = e2e.CASES["wan21-t2v-1.3b-l0"]
    latents = e2e._initial_latents(manifest, case)
    captured = {}

    def run_json(*args):
        captured["arguments"] = args[6:]
        return {"output": "native-frames"}

    monkeypatch.setattr(e2e, "_run_json", run_json)
    e2e._native(
        Path("trtmc"),
        Path("runtime"),
        Path("bundle"),
        Path("model"),
        manifest,
        case,
        tmp_path,
        latents,
    )
    arguments = captured["arguments"]
    path = Path(arguments[arguments.index("--initial-latents-raw") + 1])
    np.testing.assert_array_equal(np.fromfile(path, dtype=np.float32), latents.reshape(-1))


def test_reference_is_fp32_tied_and_consumes_the_same_latents(monkeypatch, tmp_path: Path) -> None:
    calls, fp32 = _framework(monkeypatch)
    _, manifest, case = e2e.CASES["wan21-t2v-1.3b-l0"]
    latents = e2e._initial_latents(manifest, case)
    result = e2e._official_reference(Path("model"), manifest, case, tmp_path, latents)

    assert calls["load"] == {
        "torch_dtype": fp32,
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }
    pipeline = calls["pipeline"]
    assert pipeline.text_encoder.tied is True
    assert pipeline.text_encoder.encoder.embed_tokens is pipeline.text_encoder.shared
    assert calls["kwargs"]["max_sequence_length"] == 226
    assert calls["kwargs"]["guidance_scale"] == 5.0
    tensor = calls["kwargs"]["latents"]
    np.testing.assert_array_equal(tensor.values, latents)
    assert tensor.device == "cuda"
    assert tensor.dtype is fp32
    assert calls["kwargs"]["output_type"] == "pil"
    np.testing.assert_array_equal(
        result["images"],
        np.array([[[[17, 128, 255], [0, 64, 192]]]], dtype=np.float32) / 255.0,
    )


def test_reference_rejects_an_untied_text_encoder() -> None:
    text_encoder = SimpleNamespace(
        shared=SimpleNamespace(weight=_Weight(1)),
        encoder=SimpleNamespace(embed_tokens=SimpleNamespace(weight=_Weight(2))),
        tie_weights=lambda: None,
    )
    with pytest.raises(RuntimeError, match=r"tie_weights\(\) did not bind embeddings"):
        e2e._tie_wan_text_encoder(SimpleNamespace(text_encoder=text_encoder))


def test_reference_restores_shared_inputs_without_changing_output_tying() -> None:
    shared = SimpleNamespace(weight=_Weight(1))
    encoder = SimpleNamespace(embed_tokens=SimpleNamespace(weight=_Weight(2)))
    configuration = SimpleNamespace(tie_word_embeddings=False)
    calls = []

    def set_input_embeddings(value):
        calls.append(value)
        encoder.embed_tokens = value

    text_encoder = SimpleNamespace(
        shared=shared,
        encoder=encoder,
        config=configuration,
        tie_weights=lambda: None,
        set_input_embeddings=set_input_embeddings,
    )
    e2e._tie_wan_text_encoder(SimpleNamespace(text_encoder=text_encoder))
    assert calls == [shared]
    assert encoder.embed_tokens is shared
    assert configuration.tie_word_embeddings is False


def test_reference_rejects_missing_or_mismatched_embeddings() -> None:
    shared = SimpleNamespace(weight=_Weight(1))
    with pytest.raises(RuntimeError, match="no shared embedding binding"):
        e2e._tie_wan_text_encoder(
            SimpleNamespace(text_encoder=SimpleNamespace(shared=shared, tie_weights=lambda: None))
        )

    mismatched = _Weight(2)
    mismatched.shape = (9, 4)
    calls = []
    text_encoder = SimpleNamespace(
        shared=shared,
        encoder=SimpleNamespace(embed_tokens=SimpleNamespace(weight=mismatched)),
        tie_weights=lambda: None,
        set_input_embeddings=calls.append,
    )
    with pytest.raises(RuntimeError, match="embedding shapes do not match"):
        e2e._tie_wan_text_encoder(SimpleNamespace(text_encoder=text_encoder))
    assert calls == []


def test_reference_rejects_an_ineffective_input_setter() -> None:
    text_encoder = SimpleNamespace(
        shared=SimpleNamespace(weight=_Weight(1)),
        encoder=SimpleNamespace(embed_tokens=SimpleNamespace(weight=_Weight(2))),
        tie_weights=lambda: None,
        set_input_embeddings=lambda value: None,
    )
    with pytest.raises(RuntimeError, match=r"tie_weights\(\) did not bind embeddings"):
        e2e._tie_wan_text_encoder(SimpleNamespace(text_encoder=text_encoder))
