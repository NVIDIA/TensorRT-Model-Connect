# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from . import test_e2e as e2e


class _Tensor:
    def __init__(self, values: np.ndarray):
        self.values = values
        self.device = None
        self.dtype = None

    def to(self, *, device, dtype):
        self.device = device
        self.dtype = dtype
        return self


class _Image:
    def convert(self, mode: str) -> np.ndarray:
        assert mode == "RGB"
        return np.zeros((2, 2, 3), dtype=np.uint8)


class _Pipeline:
    def __init__(self, kind: str, calls: dict):
        self.kind = kind
        self.calls = calls

    def enable_sequential_cpu_offload(self):
        self.calls["sequential_cpu_offload"] = True

    def _pack_latents(self, tensor, *shape):
        self.calls["packed"] = (tensor, shape)
        return ("packed", tensor)

    def __call__(self, **kwargs):
        self.calls["kwargs"] = kwargs
        return SimpleNamespace(images=[_Image()])


def _framework(monkeypatch: pytest.MonkeyPatch) -> tuple[dict, object, object]:
    calls: dict = {}
    fp32 = object()
    bf16 = object()
    torch = ModuleType("torch")
    torch.float16 = object()
    torch.float32 = fp32
    torch.bfloat16 = bf16
    torch.from_numpy = _Tensor

    class Generator:
        def __init__(self, device: str):
            self.device = device

        def manual_seed(self, seed: int):
            self.seed = seed
            return self

    torch.Generator = Generator
    diffusers = ModuleType("diffusers")

    def pipeline_class(kind: str):
        class Pipeline:
            @classmethod
            def from_pretrained(cls, model_dir, **kwargs):
                calls["kind"] = kind
                calls["model_dir"] = model_dir
                calls["load"] = kwargs
                return _Pipeline(kind, calls)

        return Pipeline

    diffusers.FluxPipeline = pipeline_class("flux1")
    diffusers.Flux2Pipeline = pipeline_class("flux2")
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    return calls, fp32, bf16


@pytest.mark.parametrize(
    ("case_name", "shape"),
    (("flux-schnell-l0", (1, 16, 48, 48)), ("flux-2-dev-l0", (1, 128, 24, 24))),
)
def test_single_image_latents_use_the_family_numpy_contract(case_name: str, shape: tuple) -> None:
    _, manifest, case = e2e.CASES[case_name]
    actual = e2e._initial_latents(manifest, case)
    expected = np.random.default_rng(int(case["seed"])).standard_normal(shape, dtype=np.float32)
    np.testing.assert_array_equal(actual, expected)


def test_reference_precision_matches_the_flux_variant_contract() -> None:
    for name, (_, manifest, case) in e2e.CASES.items():
        expected = "bf16" if e2e._is_flux2(manifest) else "fp32"
        if name == "flux-schnell":
            expected = "fp16"
        assert case["reference_precision"] == expected


def test_native_receives_the_exact_raw_latents(monkeypatch, tmp_path: Path) -> None:
    _, manifest, case = e2e.CASES["flux-schnell-l0"]
    latents = e2e._initial_latents(manifest, case)
    captured = {}

    def run_json(*args):
        captured["arguments"] = args[6:]
        return {"output": "native.png"}

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


@pytest.mark.parametrize(
    ("case_name", "kind", "dtype_name"),
    (("flux-schnell-l0", "flux1", "fp32"), ("flux-2-dev-l0", "flux2", "bf16")),
)
def test_reference_uses_concrete_pipeline_dtype_and_same_latents(
    monkeypatch, tmp_path: Path, case_name: str, kind: str, dtype_name: str
) -> None:
    calls, fp32, bf16 = _framework(monkeypatch)
    _, manifest, case = e2e.CASES[case_name]
    latents = e2e._initial_latents(manifest, case)
    e2e._official_reference(Path("model"), manifest, case, tmp_path, latents)

    expected_dtype = {"fp32": fp32, "bf16": bf16}[dtype_name]
    assert calls["kind"] == kind
    assert calls["load"]["torch_dtype"] is expected_dtype
    assert calls["load"]["local_files_only"] is True
    assert calls["sequential_cpu_offload"] is True
    if kind == "flux2":
        assert calls["load"]["low_cpu_mem_usage"] is True
    else:
        assert "low_cpu_mem_usage" not in calls["load"]
    reference_latents = calls["kwargs"]["latents"]
    if kind == "flux1":
        reference_latents = reference_latents[1]
    np.testing.assert_array_equal(reference_latents.values, latents)
    assert reference_latents.device == "cuda"
    assert reference_latents.dtype is expected_dtype
    assert "generator" not in calls["kwargs"]


def test_batch_reference_keeps_per_sample_generators(monkeypatch, tmp_path: Path) -> None:
    calls, fp32, _ = _framework(monkeypatch)
    _, manifest, case = e2e.CASES["flux-schnell-l0-batch2"]
    e2e._official_reference(Path("model"), manifest, case, tmp_path, None)

    assert calls["kind"] == "flux1"
    assert calls["load"]["torch_dtype"] is fp32
    assert calls["sequential_cpu_offload"] is True
    assert "latents" not in calls["kwargs"]
    assert [generator.seed for generator in calls["kwargs"]["generator"]] == case["inputs"][
        "batch_seeds"
    ]
