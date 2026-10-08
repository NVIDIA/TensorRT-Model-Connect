# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exhaustive BF16 activation parity, including the sensitive negative tail."""

import subprocess

import numpy as np
import pytest


@pytest.mark.trt
def test_all_bf16_gelu_inputs(tmp_path, request):
    torch = pytest.importorskip("torch")
    trt = pytest.importorskip("tensorrt")
    if not torch.cuda.is_available():
        pytest.skip("native GELU parity requires CUDA")
    from families.laya.model import Graph
    from families.laya.tests.compare_plan import dump_inputs

    probe = request.getfixturevalue("laya_engine_probe")
    values = torch.arange(65536).to(torch.int16).view(torch.bfloat16).cuda()
    expected = torch.nn.functional.gelu(values).float().cpu().numpy()
    builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    graph = Graph(network, {})
    inputs = network.add_input("x", trt.bfloat16, (65536,))
    output = graph.cast(graph.gelu(inputs), trt.float32)
    output.name = "output"
    network.mark_output(output)
    settings = builder.create_builder_config()
    settings.clear_flag(trt.BuilderFlag.TF32)
    plan = builder.build_serialized_network(network, settings)
    assert plan is not None
    path = tmp_path / "gelu.plan"
    path.write_bytes(bytes(plan))
    dump_inputs(tmp_path, {"x": values})
    subprocess.run([str(probe), str(path), str(tmp_path)], check=True)
    actual = np.fromfile(tmp_path / "output.out.bin", np.float32)
    np.testing.assert_allclose(actual, expected, atol=0, rtol=0, equal_nan=True)
