# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorrt")
pytest.importorskip("torch")

from families.llama import quantization as llama_quantization  # noqa: E402
from families.llama.reference import prepare_int4_awq  # noqa: E402


def test_the_reference_weight_is_the_int4_awq_product_the_engine_computes():
    rng = np.random.default_rng(0)
    weight = rng.normal(size=(8, 256)).astype(np.float32)
    smooth = rng.uniform(0.5, 2.0, size=256).astype(np.float32)

    got = prepare_int4_awq.effective_weight(weight, smooth)

    # The engine: group-128 signed INT4 codes of weight / fp16(smooth), fp16 block scales amax / 7, dequantized to
    # fp16, and the activation multiplied by the smoothing scale.
    scale = smooth.astype(np.float16).astype(np.float32)
    blocks = (weight / scale).reshape(8, 2, 128)
    block_scale = (np.abs(blocks).max(axis=2) / 7).astype(np.float16).astype(np.float32)
    codes = np.clip(np.rint(blocks / block_scale[..., None]), -8, 7)
    dequantized = (
        (codes * block_scale[..., None]).reshape(8, 256).astype(np.float16).astype(np.float32)
    )
    np.testing.assert_allclose(got, dequantized * scale, rtol=1e-6)
    activation = rng.normal(size=(3, 256)).astype(np.float32)
    np.testing.assert_allclose(
        activation @ got.T, (activation * scale) @ dequantized.T, rtol=1e-5, atol=1e-4
    )
    assert (
        0 < np.abs(got - weight).max() < 0.25 * np.abs(weight).max()
    )  # quantized, not the original


def test_only_int4_awq_manifests_are_prepared_and_a_finished_one_is_kept(tmp_path, monkeypatch):
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    for name, quantization in (("m-int4", "int4_awq"), ("m-fp8", "fp8"), ("m", None)):
        (manifests / f"{name}.json").write_text(
            json.dumps(
                {
                    "name": name,
                    "hf_id": f"org/{name}",
                    **({"quantization": quantization} if quantization else {}),
                }
            )
        )
    written = []
    monkeypatch.setattr(prepare_int4_awq, "MANIFESTS", manifests)
    monkeypatch.setattr(
        prepare_int4_awq,
        "write_reference",
        lambda manifest, destination: written.append(destination),
    )

    assert prepare_int4_awq.main(tmp_path / "env") == 0
    assert written == [tmp_path / "env/trtmc-reference/m-int4"]
    done = tmp_path / "env/trtmc-reference/m-int4"
    done.mkdir(parents=True)
    (done / prepare_int4_awq.MARKER).write_text("{}")
    written.clear()
    prepare_int4_awq.main(tmp_path / "env")
    assert written == []  # prepared once per reference environment


def test_the_profile_declares_what_the_preparation_reads_as_its_inputs():
    import yaml

    root = prepare_int4_awq.ROOT
    profile = yaml.safe_load(
        (root / "families/llama/tests/aiperf/nemotron-nano-4b-int4-awq.yaml").read_text()
    )
    inputs = {root / path for path in profile["reference"]["prepare_inputs"]}
    # A change to the quantizer, the committed scales or a manifest must prepare the reference weights again.
    assert inputs == {
        Path(llama_quantization.__file__).resolve(),
        llama_quantization.AWQ_SCALES_DIR.resolve(),
        prepare_int4_awq.MANIFESTS.resolve(),
    }
