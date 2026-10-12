# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frame isolation and empty-block regression for bounded vision attention."""

import subprocess

import numpy as np
import pytest


@pytest.mark.trt
def test_frames_do_not_share_attention(tmp_path, request):
    torch = pytest.importorskip("torch")
    trt = pytest.importorskip("tensorrt")
    if not torch.cuda.is_available():
        pytest.skip("native vision attention requires CUDA")
    from families.clef.graph import Graph
    from families.clef.tests.compare_head import dump_inputs

    probe = request.getfixturevalue("clef_engine_probe")
    lengths, heads, width = (64, 100, 96), 2, 8
    sequence = sum(lengths)
    torch.manual_seed(8127)
    q, k, v = [
        torch.randn(sequence, heads, width, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    ]
    groups = torch.repeat_interleave(torch.arange(3), torch.tensor(lengths)).to(torch.int32)
    v[64:164] += 10
    v[164:] -= 10
    expected = []
    begin = 0
    with torch.inference_mode():
        for length in lengths:
            end = begin + length
            values = [t[begin:end].transpose(0, 1)[None] for t in (q, k, v)]
            output = torch.nn.functional.scaled_dot_product_attention(*values)
            expected.append(output[0].transpose(0, 1).reshape(length, heads * width))
            begin = end
    expected = torch.cat(expected).float().cpu().numpy()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    graph = Graph(network, {})
    inputs = [
        network.add_input(name, trt.bfloat16, (sequence, heads * width)) for name in ("q", "k", "v")
    ]
    graph.attention_groups = network.add_input("frame_ids", trt.int32, (sequence,))
    output = graph.cast(graph.attention(*inputs, heads, fp32_accumulation=True), trt.float32)
    output.name = "output"
    network.mark_output(output)
    settings = builder.create_builder_config()
    settings.builder_optimization_level = 3
    settings.clear_flag(trt.BuilderFlag.TF32)
    plan = builder.build_serialized_network(network, settings)
    assert plan is not None
    path = tmp_path / "frames.plan"
    path.write_bytes(bytes(plan))
    dump_inputs(
        tmp_path,
        {
            "q": q.reshape(sequence, -1),
            "k": k.reshape(sequence, -1),
            "v": v.reshape(sequence, -1),
            "frame_ids": groups,
        },
    )
    subprocess.run([str(probe), str(path), str(tmp_path)], check=True)
    actual = np.fromfile(tmp_path / "output.out.bin", np.float32).reshape(expected.shape)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=0.02, rtol=0.01)
