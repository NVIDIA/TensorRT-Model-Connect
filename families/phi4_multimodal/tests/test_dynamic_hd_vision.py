# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamic-HD vision engine contracts that do not need a GPU."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def vision_builder(monkeypatch):
    if "tensorrt" not in sys.modules:
        monkeypatch.setitem(sys.modules, "tensorrt", SimpleNamespace())
    return importlib.import_module("families.phi4_multimodal.phi4mm_vision_builder")


def test_navit_buckets_match_siglip_embeddings_for_every_valid_count(vision_builder) -> None:
    torch = pytest.importorskip("torch")
    table = vision_builder.navit_bucket_table()
    boundaries = torch.arange(1 / 32, 1.0, 1 / 32)
    assert not table[0].any()
    for count in range(1, 33):
        # SiglipVisionEmbeddings counts valid patches from the boolean mask.
        mask = torch.zeros(32, 32, dtype=torch.bool)
        mask[:count, :count] = True
        valid = mask[:, 0].sum()
        expected = torch.bucketize(torch.arange(0, 1 - 1e-6, 1 / valid), boundaries, right=True)
        np.testing.assert_array_equal(table[count, :count], expected.numpy())
        assert not table[count, count:].any()


def test_token_budget_covers_every_dynamic_hd_grid(vision_builder) -> None:
    # sub_glb: up to 16h x (16w + 1) crop tokens, glb_GN, and the 16 x 17 global view.
    largest = max(
        16 * rows * (16 * cols + 1)
        for rows in range(1, 37)
        for cols in range(1, 37)
        if rows * cols <= 36
    )
    assert vision_builder.MAX_IMAGE_TOKENS == largest + 1 + 16 * 17
    assert vision_builder.MIN_IMAGE_TOKENS == 2 + 1 + 16 * 17
    assert vision_builder.MAX_ENTRIES == 37


def test_siglip_encoder_gelu_is_tanh_approximation_in_fp32(vision_builder, monkeypatch) -> None:
    # Phi-4's SigLIP tower keeps SiglipVisionConfig's default gelu_pytorch_tanh;
    # only the image projection uses the exact (erf) nn.GELU.
    fake_trt = SimpleNamespace(
        float16="float16",
        float32="float32",
        ActivationType=SimpleNamespace(GELU_TANH="gelu_tanh"),
    )
    calls = []

    class Tensor:
        def __init__(self, dtype):
            self.dtype = dtype

    class Layer:
        def __init__(self, output):
            self.output = output

        def get_output(self, index):
            return self.output

    class Network:
        def add_cast(self, tensor, dtype):
            calls.append(("cast", tensor.dtype, dtype))
            return Layer(Tensor(dtype))

        def add_activation(self, tensor, kind):
            calls.append(("activation", tensor.dtype, kind))
            return Layer(Tensor(tensor.dtype))

    monkeypatch.setattr(vision_builder, "trt", fake_trt)
    output = vision_builder._gelu_tanh(Network(), Tensor("float16"))

    assert output.dtype == "float16"
    assert calls == [
        ("cast", "float16", "float32"),
        ("activation", "float32", "gelu_tanh"),
        ("cast", "float32", "float16"),
    ]


def test_default_context_holds_the_largest_dynamic_hd_image(vision_builder) -> None:
    import json
    from pathlib import Path

    from families.phi4_multimodal import model

    family = Path(model.__file__).resolve().parent
    manifest = json.loads((family / "tests/manifests/phi4-multimodal.json").read_text())
    # The largest image plus a prompt and the benchmark's 128 generated tokens.
    assert model.DEFAULT_MAX_SEQUENCE_LENGTH >= vision_builder.MAX_IMAGE_TOKENS + 256
    assert manifest["max_sequence_length"] == model.DEFAULT_MAX_SEQUENCE_LENGTH
