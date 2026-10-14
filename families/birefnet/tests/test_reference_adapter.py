# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from trtmc_perf_serving.backends.base import BackendError
from trtmc_perf_serving.backends.reference import ReferenceBackend
from trtmc_perf_serving.backends.reference.common import ReferenceSpec


def test_native_loads_once_and_matches_official_preprocessing_and_binary_masks(monkeypatch, tmp_path):
    import transformers

    loads, pixels = [], []

    class Model(torch.nn.Module):
        def forward(self, value):
            pixels.append(value.detach().clone())
            logits = torch.tensor([[0.0, -100.0, 100.0, 100.0]]).repeat(4, 1)[None, None]
            return [[logits]]

    def load(name, **kwargs):
        loads.append((name, kwargs))
        return Model()

    monkeypatch.setattr(transformers, "AutoModelForImageSegmentation", SimpleNamespace(from_pretrained=load))
    adapter = Path(__file__).resolve().parents[1] / "reference/adapter.py"
    spec = ReferenceSpec(operation="segment", model="pinned-checkpoint", revision="immutable-revision",
                         precision="fp32", device="cpu", trust_remote_code=True,
                         adapter=str(adapter), options={"input_size": 4})
    backend = ReferenceBackend(spec)
    try:
        for index, (size, color, expected) in enumerate([
            ((3, 2), (255, 0, 0), [0, 0, 1, 0, 0, 1]),
            ((1, 3), (0, 255, 0), [0, 0, 0]),
        ]):
            path = tmp_path / f"{index}.png"
            Image.new("RGB", size, color).save(path)
            result = backend.invoke({"image_path": str(path)}, tmp_path / f"result-{index}")
            assert result.observation["mask"] == expected
            assert (result.observation["width"], result.observation["height"]) == size
            assert result.observation["num_masks"] == 1
            assert result.observation["mask_pixels"] == size[0] * size[1]
        assert len(loads) == 1
        assert loads[0] == ("pinned-checkpoint", {"trust_remote_code": True,
                                               "revision": "immutable-revision", "dtype": torch.float32})
        assert all(value.shape == (1, 3, 4, 4) and value.dtype == torch.float32 for value in pixels)
        np.testing.assert_allclose(pixels[0][0, :, 0, 0], [2.2489083, -2.0357143, -1.8044444], atol=1e-6)
        np.testing.assert_allclose(pixels[1][0, :, 0, 0], [-2.117904, 2.4285715, -1.8044444], atol=1e-6)
    finally:
        backend.close()


@pytest.mark.parametrize("logits", [torch.full((1, 1, 4, 4), float("nan")), torch.zeros(1, 2, 4, 4)])
def test_native_rejects_invalid_logits(monkeypatch, tmp_path, logits):
    import transformers

    class Model(torch.nn.Module):
        def forward(self, value):
            return [logits]

    monkeypatch.setattr(transformers, "AutoModelForImageSegmentation",
                        SimpleNamespace(from_pretrained=lambda *args, **kwargs: Model()))
    adapter = Path(__file__).resolve().parents[1] / "reference/adapter.py"
    backend = ReferenceBackend(ReferenceSpec(operation="segment", model="checkpoint", precision="fp32",
                                             device="cpu", adapter=str(adapter), options={"input_size": 4}))
    path = tmp_path / "image.png"
    Image.new("RGB", (4, 4)).save(path)
    try:
        with pytest.raises(BackendError, match="invalid or non-finite mask logits"):
            backend.invoke({"image_path": str(path)}, tmp_path / "result")
    finally:
        backend.close()
