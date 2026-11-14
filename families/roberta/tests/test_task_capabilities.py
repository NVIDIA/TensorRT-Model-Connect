# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check discovery and the actual build boundary without importing TensorRT."""

import ast
from pathlib import Path

import pytest

from tensorrt_model_connect.build import BuildRequest
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


@pytest.mark.parametrize("model_type", ["roberta", "xlm-roberta", "camembert"])
def test_discovery_only_advertises_implemented_encoder_tasks(model_type):
    family, support = resolve_family(ModelMetadata({"model_type": model_type}, {}))
    assert family == "roberta"
    assert support.tasks == ("embedding", "encoding")
    assert support.default_task == "encoding"


@pytest.fixture
def build_boundary():
    source = Path(__file__).parents[1] / "model.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    build = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build"
    )
    namespace = {"Path": Path}

    class CheckpointReached(RuntimeError):
        pass

    class Config:
        @staticmethod
        def from_dir(path):
            raise CheckpointReached("checkpoint loading reached")

    namespace["ModelConfig"] = Config
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            build,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace["build"], CheckpointReached


def test_reranking_is_rejected_before_checkpoint_loading(build_boundary, tmp_path):
    build, _ = build_boundary
    request = BuildRequest(tmp_path, tmp_path / "unused.bundle", "roberta", "reranking", "fp32")
    with pytest.raises(ValueError, match="roberta task must be encoding or embedding"):
        build(request, None)
    assert not request.output_path.exists()


@pytest.mark.parametrize("task", ["encoding", "embedding"])
def test_supported_tasks_still_reach_checkpoint_loading(build_boundary, tmp_path, task):
    build, checkpoint_reached = build_boundary
    request = BuildRequest(tmp_path, tmp_path / "unused.bundle", "roberta", task, "fp32")
    with pytest.raises(checkpoint_reached):
        build(request, None)
