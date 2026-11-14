# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise real Pipeline routing retries and cache commits with a mock backend."""

import json
import os
from pathlib import Path
import subprocess

import pytest

from tensorrt_model_connect.bundle_writer import BundleWriter
from families.qwen3_next.tests.tokenizer_contract import tokenizer_document


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_pipeline_routing_retry(mode, tmp_path):
    runtime = os.environ.get("TRTMC_RUNTIME_ROOT")
    if not runtime:
        pytest.skip("requires the family-owned pipeline probe")
    probe = Path(runtime) / "qwen3_next_pipeline_probe"
    assert probe.is_file()
    path = tmp_path / "routing.bundle"
    writer = BundleWriter(path)
    writer.set_header(family="qwen3_next", task="text_generation", backend="mock")
    writer.add_json("runtime.json", {
        "routing_contract": "reference_topk_retry_v1", "precision": "bf16",
        "num_experts": 4, "num_experts_per_tok": 2, "hidden_size": 1,
        "num_key_value_heads": 1, "head_dim": 1, "vocab_size": 2,
        "max_sequence_length": 16, "eos_token_id": 1, "conv_dim": 3,
        "linear_conv_kernel_dim": 2, "linear_num_value_heads": 1,
        "linear_key_head_dim": 1, "linear_value_head_dim": 1,
        "layer_types": ["linear_attention"],
    })
    writer.add_bytes("embedding.bf16", b"\x80\x3f" * 2)
    writer.add_json("tokenizer.json", tokenizer_document({"a": 0, "b": 1}))
    writer.add_bytes("layers/0.plan", b"layer")
    writer.add_bytes("head.plan", b"head")
    writer.finish()
    result = subprocess.run([str(probe), str(path), str(mode)], text=True,
                            capture_output=True, timeout=30)
    if mode == 2:
        assert result.returncode != 0
        assert "forced expert routing was not applied" in result.stderr
        return
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed["generated_ids"] == [0, 0]
    assert observed["selected_experts"] == [0, 1]
    assert observed["routing_retries"] == (2 if mode else 0)
    assert observed["forced"] == (2 if mode else 0)
    assert observed["forwards"] == (4 if mode else 2)
