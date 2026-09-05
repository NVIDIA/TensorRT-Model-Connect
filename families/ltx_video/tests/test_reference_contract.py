# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np

from . import test_e2e as e2e


class _Tensor:
    def __init__(self, values: np.ndarray):
        self.values = values

    def to(self, *, device, dtype):
        self.device = device
        self.dtype = dtype
        return self


class _Image:
    def convert(self, mode: str) -> np.ndarray:
        assert mode == "RGB"
        return np.zeros((2, 2, 3), dtype=np.uint8)


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

    class Pipeline:
        @classmethod
        def from_pretrained(cls, model_dir, **kwargs):
            calls["model_dir"] = model_dir
            calls["load"] = kwargs
            return cls()

        def to(self, device: str):
            calls["device"] = device
            return self

        def __call__(self, **kwargs):
            calls["kwargs"] = kwargs
            return SimpleNamespace(frames=[[_Image()]])

    diffusers = ModuleType("diffusers")
    diffusers.LTXPipeline = Pipeline
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    return calls, fp32


def test_ltx_latents_use_the_family_packed_numpy_contract() -> None:
    _, manifest, case = e2e.CASES["ltx-video-l0"]
    actual = e2e._initial_latents(manifest, case)
    unpacked = np.random.default_rng(int(case["seed"])).standard_normal(
        (1, 128, 2, 8, 8), dtype=np.float32
    )
    expected = unpacked.transpose(0, 2, 3, 4, 1).reshape(1, 128, 128)
    np.testing.assert_array_equal(actual, expected)


def test_native_receives_the_exact_packed_raw_latents(monkeypatch, tmp_path: Path) -> None:
    _, manifest, case = e2e.CASES["ltx-video-l0"]
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


def test_reference_is_fp32_and_consumes_the_same_packed_latents(
    monkeypatch, tmp_path: Path
) -> None:
    calls, fp32 = _framework(monkeypatch)
    _, manifest, case = e2e.CASES["ltx-video-l0"]
    latents = e2e._initial_latents(manifest, case)
    e2e._official_reference(Path("model"), manifest, case, tmp_path, latents)

    assert calls["load"] == {"torch_dtype": fp32, "local_files_only": True}
    assert calls["kwargs"]["negative_prompt"] == (
        "worst quality, inconsistent motion, blurry, jittery, distorted"
    )
    tensor = calls["kwargs"]["latents"]
    np.testing.assert_array_equal(tensor.values, latents)
    assert tensor.device == "cuda"
    assert tensor.dtype is fp32
