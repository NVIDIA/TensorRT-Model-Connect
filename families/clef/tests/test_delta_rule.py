# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression for the unstable inverse exposed by the real invoice request."""

import subprocess

import numpy as np
import pytest


@pytest.mark.trt
def test_parallel_keys_do_not_overflow(tmp_path, request):
    torch = pytest.importorskip("torch")
    trt = pytest.importorskip("tensorrt")
    if not torch.cuda.is_available():
        pytest.skip("native delta-rule regression requires CUDA")
    probe = request.getfixturevalue("clef_engine_probe")
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
    from families.clef.backbone import delta_rule
    from families.clef.graph import Graph
    from families.clef.tests.compare_head import dump_inputs

    sequence, heads, width = 193, 4, 16
    torch.manual_seed(4426)
    query = torch.ones(sequence, heads, width, device="cuda", dtype=torch.bfloat16)
    key = query.clone()
    value = torch.randn_like(query)
    beta = torch.full((sequence, heads), 0.992, device="cuda", dtype=torch.bfloat16)
    decay = torch.full((sequence, heads), -1e-5, device="cuda")
    with torch.inference_mode():
        reference = (
            torch_chunk_gated_delta_rule(
                query[None],
                key[None],
                value[None],
                decay[None],
                beta[None],
                use_qk_l2norm_in_kernel=True,
            )[0][0]
            .float()
            .cpu()
            .numpy()
        )
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    graph = Graph(network, {})
    tensors = {
        name: network.add_input(name, dtype, shape)
        for name, dtype, shape in (
            ("query", trt.bfloat16, (sequence, heads, width)),
            ("key", trt.bfloat16, (sequence, heads, width)),
            ("value", trt.bfloat16, (sequence, heads, width)),
            ("decay", trt.float32, (sequence, heads)),
            ("beta", trt.bfloat16, (sequence, heads)),
        )
    }
    result = delta_rule(
        graph,
        tensors["query"],
        tensors["key"],
        tensors["value"],
        tensors["decay"],
        tensors["beta"],
        heads,
        width,
    )
    result = graph.cast(result, trt.float32)
    result.name = "output"
    network.mark_output(result)
    settings = builder.create_builder_config()
    settings.clear_flag(trt.BuilderFlag.TF32)
    plan = builder.build_serialized_network(network, settings)
    assert plan is not None
    path = tmp_path / "parallel-keys.plan"
    path.write_bytes(bytes(plan))
    dump_inputs(
        tmp_path, {"query": query, "key": key, "value": value, "decay": decay, "beta": beta}
    )
    subprocess.run([str(probe), str(path), str(tmp_path)], check=True)
    actual = np.fromfile(tmp_path / "output.out.bin", np.float32).reshape(reference.shape)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, reference, atol=0.001, rtol=0.01)
