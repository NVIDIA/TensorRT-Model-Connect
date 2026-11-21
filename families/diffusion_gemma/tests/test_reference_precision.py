# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference precision and profile identities must survive artifact reuse."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from families.diffusion_gemma.tests.original_reference import (
    TRANSFORMERS_REVISION,
    precision_dtype,
    self_conditioning,
)
from families.diffusion_gemma.tests.test_e2e import CASES, MANIFEST, reference_reusable


@pytest.mark.parametrize("precision", ["bf16", "fp32"])
def test_conditioning_follows_the_actual_embedding_dtype(precision):
    torch = pytest.importorskip("torch")
    dtype = precision_dtype(precision)
    torch.manual_seed(47)
    logits = torch.randn(3, 7, dtype=torch.float32)
    embedding = SimpleNamespace(
        weight=torch.randn(7, 5, dtype=dtype), embed_scale=torch.tensor(5 ** 0.5),
    )
    processed, signal = self_conditioning(logits, 0.79, embedding)
    expected_processed = (logits / 0.79).to(dtype)
    expected = (
        expected_processed.softmax(-1, dtype=torch.float32).to(dtype) @ embedding.weight
    ) * embedding.embed_scale.to(dtype)
    assert processed.dtype == signal.dtype == dtype
    assert torch.equal(processed, expected_processed)
    assert torch.equal(signal, expected)
    if precision == "fp32":
        prematurely_rounded = (
            (logits / 0.79).to(torch.bfloat16).softmax(-1, dtype=torch.float32)
            .to(torch.bfloat16).float() @ embedding.weight
        ) * embedding.embed_scale.to(torch.bfloat16).float()
        assert not torch.equal(signal, prematurely_rounded)


def _saved_reference(tmp_path, precision):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config = {"text_config": {"num_hidden_layers": 30}}
    index = {"weight_map": {"embedding": "weights.safetensors"}}
    (checkpoint / "config.json").write_text(json.dumps(config))
    (checkpoint / "model.safetensors.index.json").write_text(json.dumps(index))
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "tensor.bin").write_bytes(b"\x00\x00\x00\x00")
    request = {"input_ids": [2, 3], "decoder_input_ids": [[4, 5]], "temperatures": [0.8]}
    recorded = {
        "precision": precision,
        "embedding_dtype": "float32" if precision == "fp32" else "bfloat16",
        "max_sequence_length": 512,
        "canvas_length": 256,
        "request": request,
        "checkpoint_config": config,
        "checkpoint_index": index,
        "transformers_revision": TRANSFORMERS_REVISION,
        "tensors": {"tensor": {"file": "tensor.bin"}},
    }
    arguments = {
        "checkpoint": checkpoint, "request": request, "precision": precision,
        "max_sequence_length": 512, "canvas_length": 256,
    }
    return reference, recorded, arguments


@pytest.mark.parametrize("precision", ["bf16", "fp32"])
def test_reference_reuse_requires_matching_compute_precision(tmp_path, precision):
    reference, recorded, arguments = _saved_reference(tmp_path, precision)
    assert reference_reusable(reference, recorded, **arguments)
    other = "fp32" if precision == "bf16" else "bf16"
    assert not reference_reusable(reference, recorded, **{**arguments, "precision": other})


@pytest.mark.parametrize("key", ["precision", "embedding_dtype", "max_sequence_length", "canvas_length"])
def test_unattested_reference_cannot_be_reused_as_fp32(tmp_path, key):
    reference, recorded, arguments = _saved_reference(tmp_path, "fp32")
    del recorded[key]
    assert not reference_reusable(reference, recorded, **arguments)


@pytest.mark.parametrize("mutation", ["embedding_dtype", "capacity", "canvas", "request", "revision"])
def test_reference_reuse_rejects_other_workload_identities(tmp_path, mutation):
    reference, recorded, arguments = _saved_reference(tmp_path, "fp32")
    if mutation == "embedding_dtype":
        recorded["embedding_dtype"] = "bfloat16"
    elif mutation == "capacity":
        recorded["max_sequence_length"] = 1024
    elif mutation == "canvas":
        recorded["canvas_length"] = 128
    elif mutation == "request":
        recorded["request"] = {"input_ids": [2, 4]}
    else:
        recorded["transformers_revision"] = "another-revision"
    assert not reference_reusable(reference, recorded, **arguments)


def test_selected_fp32_case_preserves_the_original_qualification_workload():
    assert set(CASES) == {"diffusiongemma-26b-a4b-fp32-512"}
    assert MANIFEST["precision"] == "fp32" and MANIFEST["max_sequence_length"] == 512
    assert MANIFEST["hf_id"] == "google/diffusiongemma-26B-A4B-it"
    assert MANIFEST["hf_revision"] == "f7f5b7f5fa82ffc52addd066915886d497f5517b"
    assert MANIFEST["family"] == "diffusion_gemma" and MANIFEST["task"] == "text_continuation"
    assert MANIFEST["tensor_parallel_size"] == 1
    case = MANIFEST["testcases"][0]
    assert case["prompt"] == "What is the capital of France? Answer with the city name only."
    assert case["expected_answer"] == "Paris"
    assert case["max_new_tokens"] == 256 and case["seed"] == 19
    assert case["premerge"] is True
