# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression for early rounding in recurrent and full-attention sigmoid gates."""

import subprocess

import numpy as np
import pytest


@pytest.mark.trt
def test_sigmoid_preserves_reference_opmath(tmp_path, request):
    torch = pytest.importorskip("torch")
    trt = pytest.importorskip("tensorrt")
    if not torch.cuda.is_available():
        pytest.skip("native sigmoid regression requires CUDA")
    from families.clef.graph import Graph
    from families.clef.tests.compare_head import dump_inputs

    probe = request.getfixturevalue("clef_engine_probe")
    values = torch.arange(-2000, 2001).float().div(128).bfloat16().unique().cuda()
    expected = values.sigmoid().float().cpu().numpy()
    builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    graph = Graph(network, {})
    inputs = network.add_input("x", trt.bfloat16, tuple(values.shape))
    output = graph.cast(graph.sigmoid(inputs), trt.float32)
    output.name = "output"
    network.mark_output(output)
    settings = builder.create_builder_config()
    settings.clear_flag(trt.BuilderFlag.TF32)
    plan = builder.build_serialized_network(network, settings)
    assert plan is not None
    path = tmp_path / "sigmoid.plan"
    path.write_bytes(bytes(plan))
    dump_inputs(tmp_path, {"x": values})
    subprocess.run([str(probe), str(path), str(tmp_path)], check=True)
    np.testing.assert_array_equal(np.fromfile(tmp_path / "output.out.bin", np.float32), expected)
