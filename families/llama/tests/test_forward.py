# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end full-forward tests: HF weights -> public bundle -> owned SDK logits."""

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tensorrt")
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from ..build_request import BuildRequest  # noqa: E402
from ..cli import build_bundle  # noqa: E402
from .forward_validation import check_logits, reference, run_case  # noqa: E402


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["fp32", "fp16", "bf16"])
def test_stateless_forward(tmp_path, precision):
    root = os.environ.get("TRTMC_NATIVE_BUILD_DIR")
    if not root:
        pytest.skip("set TRTMC_NATIVE_BUILD_DIR to the selected native build")
    root = Path(root)
    runtime = Path(os.environ.get("TRTMC_RUNTIME_ROOT", str(root)))
    runner = root / "families/llama/llama_forward_runner"
    assert runner.is_file(), "build llama_forward_runner first"
    torch.manual_seed(17)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    config._attn_implementation = "sdpa"
    model = LlamaForCausalLM(config).eval()
    checkpoint = tmp_path / "checkpoint"
    model.save_pretrained(checkpoint)
    bundle = tmp_path / "forward.bundle"
    build_bundle(
        BuildRequest(
            model_dir=checkpoint,
            output_path=bundle,
            family="llama",
            task="text_prediction_positions_to_token_scores",
            precision=precision,
            max_sequence_length=16,
        ),
        bundle,
    )
    model.cuda()
    cases = [
        {"token_ids": [1, 5, 8, 3]},
        {"token_ids": [1, 5, 8, 3], "causal": False},
        {
            "token_ids": [1, 5, 8],
            "causal": False,
            "attention_mask": [1, 1, 0],
            "position_ids": [2, 3, 4],
            "prediction_positions": [2, 0, 2],
        },
        {"token_ids": [8]},
    ]
    for i, request in enumerate(cases):
        values, summary = run_case(runner, bundle, runtime, request, tmp_path, str(i))
        check_logits(values, reference(model, request), precision)
        assert summary["independent_calls_and_owned_results"]


def test_forward_rejects_generation_controls(tmp_path):
    from ..forward import build_forward
    from dataclasses import replace

    request = BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "x.bundle",
        family="llama",
        task="text_prediction_positions_to_token_scores",
        precision="fp32",
    )
    for changes in (
        {"dynamic_kv_cache": True},
        {"quantization": "fp8"},
        {"tensor_parallel_size": 4},
        {"fp32_layers": (0,)},
    ):
        with pytest.raises(NotImplementedError, match="unquantized single-device"):
            build_forward(replace(request, **changes), None)
