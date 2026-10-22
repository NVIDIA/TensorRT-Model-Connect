# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the native precision label bound to the official inference option."""

import contextlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from families.moge.reference import adapter
from families.moge.tests import reference_support


@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_adapter_passes_requested_precision(monkeypatch, precision):
    calls = []
    monkeypatch.setattr(
        adapter,
        "OfficialReference",
        lambda source, checkpoint, **options: calls.append((source, checkpoint, options)),
    )
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **kw: "model.pt")
    spec = SimpleNamespace(model="Ruicheng/moge-2-vitl", revision="pinned", precision=precision)

    adapter.Adapter(spec, object())

    assert len(calls) == 1
    assert calls[0][1:] == (Path("model.pt"), {"precision": precision})


class _Tensor:
    def permute(self, *dimensions):
        assert dimensions == (2, 0, 1)
        return self

    def to(self, device):
        assert device == "cuda"
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.ones(1, dtype=np.float32)


@pytest.mark.parametrize("precision,use_fp16", [("fp32", False), ("fp16", True)])
def test_inference_uses_official_precision_switch(monkeypatch, precision, use_fp16):
    reference = reference_support.OfficialReference.__new__(reference_support.OfficialReference)
    reference.precision = precision
    reference.device = "cuda"
    reference.torch = SimpleNamespace(from_numpy=lambda pixels: _Tensor())
    captured = []

    def infer(tensor, **options):
        captured.append(options)
        return {name: _Tensor() for name in ("points", "depth", "intrinsics", "mask")}

    reference.model = SimpleNamespace(infer=infer)
    monkeypatch.setattr(reference_support, "_math_sdpa", lambda torch: contextlib.nullcontext())

    result = reference.infer(np.ones((2, 3, 3), dtype=np.float32), num_tokens=1800, fov_x=60.0)

    assert set(result) == {"points", "depth", "intrinsics", "mask"}
    assert captured == [
        {
            "num_tokens": 1800,
            "use_fp16": use_fp16,
            "force_projection": True,
            "apply_mask": True,
            "fov_x": 60.0,
        }
    ]


def test_reference_rejects_unsupported_precision_before_loading():
    with pytest.raises(ValueError, match="supports fp32 or fp16"):
        reference_support.OfficialReference(
            Path("missing-source"), Path("missing-model"), precision="bf16"
        )
